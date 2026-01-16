use std::sync::atomic::{AtomicUsize, Ordering};

use ark_crypto_primitives::{merkle_tree::DigestConverter, Error};

pub mod blake3;
mod digest;
pub mod keccak;
mod parameters;
pub mod proof;

pub use parameters::{default_config, MerkleTreeParams};
#[cfg(feature = "parallel")]
use rayon::{iter::ParallelIterator, slice::ParallelSlice};
use spongefish::Decoding;

use crate::transcript::VerifierMessage;

#[derive(Debug, Default)]
pub struct HashCounter;

static HASH_COUNTER: AtomicUsize = AtomicUsize::new(0);

impl HashCounter {
    pub(crate) fn add() -> usize {
        HASH_COUNTER.fetch_add(1, Ordering::SeqCst)
    }

    pub fn reset() {
        HASH_COUNTER.store(0, Ordering::SeqCst);
    }

    pub fn get() -> usize {
        HASH_COUNTER.load(Ordering::SeqCst)
    }
}

/// A trivial converter where digest of previous layer's hash is the same as next layer's input.
pub struct IdentityDigestConverter<T> {
    _prev_layer_digest: T,
}

impl<T> DigestConverter<T, T> for IdentityDigestConverter<T> {
    type TargetType = T;
    fn convert(item: T) -> Result<T, Error> {
        Ok(item)
    }
}

#[cfg(not(feature = "parallel"))]
pub fn leaves_iter<F>(leaves: &[F], size: usize) -> impl Iterator<Item = &[F]> {
    assert!(leaves.len() % size == 0);
    leaves.chunks_exact(size)
}

#[cfg(feature = "parallel")]
pub fn leaves_iter<F: Sync>(leaves: &[F], size: usize) -> impl ParallelIterator<Item = &[F]> {
    assert!(leaves.len() % size == 0);
    leaves.par_chunks_exact(size)
}

pub fn challenge_indices<T>(transcript: &mut T, num_leaves: usize, count: usize) -> Vec<usize>
where
    T: VerifierMessage,
    u8: Decoding<[T::U]>,
{
    assert!(
        num_leaves.is_power_of_two(),
        "Number of leaves must be a power of two for unbiased results."
    );

    // Calculate the required bytes of entropy
    // TODO: Only round final result to bytes.
    let size_bytes = (num_leaves.ilog2() as usize).div_ceil(8);

    // Get required entropy bits.
    let entropy: Vec<u8> = (0..count * size_bytes)
        .map(|_| transcript.verifier_message())
        .collect();

    // Convert bytes into indices
    entropy
        .chunks_exact(size_bytes)
        .map(|chunk| chunk.iter().fold(0usize, |acc, &b| (acc << 8) | b as usize) % num_leaves)
        .collect::<Vec<usize>>()
}

#[cfg(test)]
mod tests {

    use ark_crypto_primitives::{
        crh::CRHScheme,
        merkle_tree::{MerkleTree, MultiPath},
    };
    use ark_serialize::CanonicalSerialize;
    use spongefish::{domain_separator, session};

    use super::*;
    use crate::{
        crypto::fields::Field64,
        transcript::{ProverState, VerifierState},
    };

    type Field = Field64;
    type LeafHash = blake3::Blake3LeafHash<Field>;
    type NodeHash = blake3::Blake3Compress;
    type Digest = <LeafHash as CRHScheme>::Output;
    type Config = MerkleTreeParams<Field, LeafHash, NodeHash, Digest>;

    #[test]
    fn test_hash_counter() {
        assert_eq!(HashCounter::get(), 0);
        assert_eq!(HashCounter::add(), 0);
        assert_eq!(HashCounter::get(), 1);
        HashCounter::reset();
        assert_eq!(HashCounter::get(), 0);
    }

    #[test]
    fn test_merkle_tree() {
        let num_leaves = 16;
        let leaves = (0..num_leaves).map(Field::from).collect::<Vec<_>>();

        // Create the tree
        let tree: MerkleTree<Config> = MerkleTree::new(&(), &(), leaves_iter(&leaves, 1)).unwrap();

        // Get the root
        let root = tree.root();
        assert_eq!(
            hex::encode(&root),
            "ac8ff7242f84032a4bfcaf2b5734631ee97fd5b6877108395f0580bbc1adc37d"
        );

        // Open two leaves
        let proof = tree.generate_multi_proof([0, 5]).unwrap();
        let mut proof_bytes = Vec::with_capacity(proof.compressed_size());
        proof.serialize_compressed(&mut proof_bytes).unwrap();
        assert_eq!(hex::encode(proof_bytes), "02000000000000001f475c662d5af4c49debea6053494a4be2c44343caab6871f9980fc5fb7b47caaa99def2501febf99f30765c3980ad3ae31e248cd709bf7da3d83c2fc3c7aa1b0200000000000000000000000000000001000000000000000200000000000000030000000000000009af614075b4b10a196d767225c1aeb9a3b537b6a840a43d751f67daa3b813e9c181a48d814f099bc967c7e07f8020ff77381f523c61026fb053c90671e8b5b4a2f4f39441bc04e2b4eca87504a8de6e7627c5ca6b3011b4126415779c29610402000000000000001a9d45abbd4a561996b97defb373f4289ddc3aaa59f8debee3d0ec89388c6da64df47511e712014e6967deab15b6c204108f54396e36af2175a3ebe3c3793b6f020000000000000000000000000000000500000000000000");

        // Verify proof
        let correct = proof
            .verify(&(), &(), &root, [[leaves[0]], [leaves[5]]])
            .unwrap();
        assert!(correct);
    }

    #[test]
    fn test_merkle_tree_transcript() {
        let num_leaves = 16_usize;
        let leaves = (0..num_leaves)
            .map(|i| Field::from(i as u64))
            .collect::<Vec<_>>();

        // Create the transcript
        let instance = num_leaves as u32;
        let ds = domain_separator!("asd")
            .session(session!("Test at {}:{}", file!(), line!()))
            .instance(&instance);
        let mut prover: ProverState = ds.std_prover().into();

        // Create the tree
        let tree: MerkleTree<Config> = MerkleTree::new(&(), &(), leaves_iter(&leaves, 1)).unwrap();

        // Write the root to the transcript
        let root = tree.root();
        prover.prover_message(&root);

        // Prove two random leaf indices from verifier
        let indices = challenge_indices(&mut prover, num_leaves, 2);
        for index in &indices {
            prover.prover_message(&leaves[*index]);
        }
        let proof = tree.generate_multi_proof(indices.iter().copied()).unwrap();
        prover.prover_hint_ark(&proof);

        let proof = prover.proof();
        assert_eq!(hex::encode(&proof.narg_string), "ac8ff7242f84032a4bfcaf2b5734631ee97fd5b6877108395f0580bbc1adc37d0d000000000000000700000000000000");
        assert_eq!(hex::encode(&proof.hints), "02000000000000004dc186f36c8f560cfbac7ccfe9f9e2ff52b569fd463958c57d71ddbefc77471a94ad4a635d5eb054b5ce5e3ba338dd487c623756fd2b12d1c6349d95fe35bc060200000000000000000000000000000000000000000000000200000000000000030000000000000009af614075b4b10a196d767225c1aeb9a3b537b6a840a43d751f67daa3b813e91a9d45abbd4a561996b97defb373f4289ddc3aaa59f8debee3d0ec89388c6da66c64fface42ff6c916281b628a06c4dd2564155bf1e2060132ec0321c317c9fb03000000000000002a374ca79db9453422e7725e57a931cf27898600124aebfe1e0d73158c310c1e475d342873157321cb35a667a5d0a5e47118b70662ced48527ef73f046ddccfb18c3de7682b0bac8831cfa4f379db39880cf8d42a6c288d4d8b37ae54a2be4d0020000000000000007000000000000000d00000000000000");

        //
        // Verifier
        //

        // Create verifier
        let mut verifier = VerifierState::from(ds.std_verifier(&proof.narg_string), &proof.hints);

        // Read root, challenge indices and proof.
        let verifier_root: Digest = verifier.prover_message().unwrap();
        assert_eq!(verifier_root, root);
        let verifier_indices = challenge_indices(&mut verifier, num_leaves, 2);
        assert_eq!(&verifier_indices, &indices);
        let leaves = (0..indices.len())
            .map(|_| [verifier.prover_message().unwrap()])
            .collect::<Vec<_>>();

        let proof: MultiPath<Config> = verifier.prover_hint_ark().unwrap();
        let mut sorted = indices.clone();
        sorted.sort_unstable();
        assert_eq!(proof.leaf_indexes, sorted);

        let sorted_leaves = proof.leaf_indexes.iter().map(|i| {
            let k = indices
                .iter()
                .position(|j| j == i)
                .expect("Missing leaf index");
            leaves[k]
        });
        let ok = proof.verify(&(), &(), &root, sorted_leaves).unwrap();
        assert!(ok);
    }
}
