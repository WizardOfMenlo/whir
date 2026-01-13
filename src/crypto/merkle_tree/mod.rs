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

#[cfg(test)]
mod tests {

    use ark_crypto_primitives::{crh::CRHScheme, merkle_tree::MerkleTree};
    use ark_serialize::CanonicalSerialize;

    use super::*;
    use crate::crypto::fields::Field64;

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
        pub type Field = Field64;
        pub type LeafHash = blake3::Blake3LeafHash<Field>;
        pub type NodeHash = blake3::Blake3Compress;
        pub type Digest = <LeafHash as CRHScheme>::Output;
        pub type Config = MerkleTreeParams<Field, LeafHash, NodeHash, Digest>;

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
}
