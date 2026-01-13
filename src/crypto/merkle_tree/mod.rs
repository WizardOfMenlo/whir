use std::sync::atomic::{AtomicUsize, Ordering};

use ark_crypto_primitives::{merkle_tree::DigestConverter, Error};

mod blake3;
mod digest;
mod keccak;
mod parameters;
mod proof;

pub use parameters::{default_config, MerkleTreeParams};
#[cfg(feature = "parallel")]
use rayon::{iter::ParallelIterator, slice::ParallelSlice};
use spongefish::Decoding;

use crate::crypto::transcript::VerifierMessage;

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
    assert!(leaves.len().is_multiple_of(size));
    leaves.chunks_exact(size)
}

#[cfg(feature = "parallel")]
pub fn leaves_iter<F: Sync>(leaves: &[F], size: usize) -> impl ParallelIterator<Item = &[F]> {
    assert!(leaves.len().is_multiple_of(size));
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
        .into()
}

#[cfg(test)]
mod tests {

    use ark_crypto_primitives::{
        crh::CRHScheme,
        merkle_tree::{MerkleTree, MultiPath},
    };
    use ark_serialize::CanonicalSerialize;
    use spongefish::StdHash;

    use super::*;
    use crate::crypto::{
        fields::Field64,
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
        let mut rng = rand::rng();
        let num_leaves = 16;
        let leaves = (0..num_leaves).map(|i| Field::from(i)).collect::<Vec<_>>();

        // Create the tree
        let (leaf_hash, node_hash) = default_config::<Field, LeafHash, NodeHash>(&mut rng);
        let tree: MerkleTree<Config> =
            MerkleTree::new(&leaf_hash, &node_hash, leaves_iter(&leaves, 1)).unwrap();

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
            .verify(&leaf_hash, &node_hash, &root, [[leaves[0]], [leaves[5]]])
            .unwrap();
        assert!(correct);
    }

    #[test]
    fn test_merkle_tree_transcript() {
        let mut rng = rand::rng();
        let num_leaves = 16_usize;
        let leaves = (0..num_leaves)
            .map(|i| Field::from(i as u64))
            .collect::<Vec<_>>();

        // Create the transcript
        let protocol = *b"Merkle Tree with blake3 for testing purposes____________________";
        let sponge = StdHash::default();
        let instance = b"test";
        let mut prover: ProverState = ProverState::new(protocol, sponge, instance);

        // Create the tree
        let (leaf_hash, node_hash) = default_config::<Field, LeafHash, NodeHash>(&mut rng);
        let tree: MerkleTree<Config> =
            MerkleTree::new(&leaf_hash, &node_hash, leaves_iter(&leaves, 1)).unwrap();

        // Write the root to the transcript
        let root = tree.root();
        prover.prover_message(&root);

        // Prove two random leaf indices from verifier
        let indices = challenge_indices(&mut prover, num_leaves, 2);
        for index in &indices {
            prover.prover_message(&leaves[*index]);
        }
        let proof = tree.generate_multi_proof(indices).unwrap();
        prover.prover_hint_ark(&proof);

        let proof = prover.proof();
        let mut buffer = Vec::new();
        ciborium::into_writer(&proof, &mut buffer).unwrap();
        assert_eq!(hex::encode(&buffer), "a26b6e6172675f737472696e67983018ac188f18f71824182f188403182a184b18fc18af182b185718341863181e18e9187f18d518b618871871081839185f05188018bb18c118ad18c3187d010000000000000007000000000000006568696e7473990130020000000000000018741850061897187617184818e718dc030218d31867187818f8189c186a18b3182418ef1894182718731897186b189218a718bb18ef18a1188c18d2184d18c1188618f3186c188f18560c18fb18ac187c18cf18e918f918e218ff185218b5186918fd18461839185818c5187d187118dd18be18fc18771847181a020000000000000000000000000000000100000000000000020000000000000003000000000000000918af18611840187518b418b10a1819186d18761872182518c118ae18b918a318b5183718b618a8184018a4183d1875181f186718da18a318b81318e918c1188118a4188d1881184f09189b18c9186718c718e0187f1880182018ff18771838181f1852183c186102186f18b0185318c906187118e818b518b418a218f418f31894184118bc0418e218b418ec18a818750418a818de186e1876182718c518ca186b18301118b4121864151877189c18291861040200000000000000181a189d184518ab18bd184a18561819189618b9187d18ef18b3187318f41828189d18dc183a18aa185918f818de18be18e318d018ec18891838188c186d18a6186c186418ff18ac18e4182f18f618c9161828181b1862188a0618c418dd1825186415185b18f118e20601183218ec03182118c31718c918fb020000000000000001000000000000000700000000000000");

        //
        // Verifier
        //

        // Create verifier
        let sponge = StdHash::default();
        let mut verifier = VerifierState::new(protocol, sponge, instance, &proof);

        // Read root, challenge indices and proof.
        let root: Digest = verifier.prover_message().unwrap();
        let indices = challenge_indices(&mut verifier, num_leaves, 2);
        let leaves = (0..indices.len())
            .map(|_| [verifier.prover_message().unwrap()])
            .collect::<Vec<_>>();

        let proof: MultiPath<Config> = verifier.prover_hint_ark().unwrap();
        assert_eq!(proof.leaf_indexes, indices);
        let ok = proof.verify(&leaf_hash, &node_hash, &root, leaves).unwrap();
        assert!(ok);
    }
}
