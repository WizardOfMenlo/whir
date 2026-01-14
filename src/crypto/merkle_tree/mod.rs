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
    use spongefish::{domain_separator, session};

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
        let num_leaves = 16;
        let leaves = (0..num_leaves).map(|i| Field::from(i)).collect::<Vec<_>>();

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
        let proof = tree.generate_multi_proof(indices).unwrap();
        prover.prover_hint_ark(&proof);

        let proof = prover.proof();
        let mut buffer = Vec::new();
        ciborium::into_writer(&proof, &mut buffer).unwrap();
        assert_eq!(hex::encode(&buffer), "a26b6e6172675f737472696e67983018ac188f18f71824182f188403182a184b18fc18af182b185718341863181e18e9187f18d518b618871871081839185f05188018bb18c118ad18c3187d05000000000000000f000000000000006568696e7473990150020000000000000018aa189918de18f21850181f18eb18f9189f18301876185c1839188018ad183a18e3181e1824188c18d70918bf187d18a318d8183c182f18c318c718aa181b18af18e218a718c218b3189218400a184b18750618da121819187f185600185e18501877188518e3183b18e91836186e11185c18f9188a1853183d020000000000000000000000000000000000000000000000020000000000000003000000000000000918af18611840187518b418b10a1819186d18761872182518c118ae18b918a318b5183718b618a8184018a4183d1875181f186718da18a318b81318e9181a189d184518ab18bd184a18561819189618b9187d18ef18b3187318f41828189d18dc183a18aa185918f818de18be18e318d018ec18891838188c186d18a6184d18f418751118e71201184e1869186718de18ab1518b618c20410188f18541839186e183618af1821187518a318eb18e318c31879183b186f0300000000000000182a1837184c18a7189d18b918451834182218e71872185e185718a9183118cf1827188918860012184a18eb18fe181e0d187315188c18310c181e1847185d183418281873151873182118cb183518a6186718a518d018a518e41871181818b706186218ce18d41885182718ef187318f0184618dd18cc18fb18f818a318b0181e188e18da189618dd189718ef18f5186418b418d8181f0c1824186d18531618801859185a18ea1845189f1882186d18a318b70418a4020000000000000005000000000000000f00000000000000");

        //
        // Verifier
        //

        // Create verifier
        let mut verifier = VerifierState::from(ds.std_verifier(&proof.narg_string), &proof.hints);

        // Read root, challenge indices and proof.
        let root: Digest = verifier.prover_message().unwrap();
        let indices = challenge_indices(&mut verifier, num_leaves, 2);
        let leaves = (0..indices.len())
            .map(|_| [verifier.prover_message().unwrap()])
            .collect::<Vec<_>>();

        let proof: MultiPath<Config> = verifier.prover_hint_ark().unwrap();
        assert_eq!(proof.leaf_indexes, indices);
        let ok = proof.verify(&(), &(), &root, leaves).unwrap();
        assert!(ok);
    }
}
