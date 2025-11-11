mod commit;
mod hasher;
mod prove;
mod verify;

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
pub use hasher::{Hash, Hasher, Hashers, HASH_ZERO};
use std::iter;

pub struct MerkleTreeHasher {
    pub root: Hash,
    pub depth: usize,
    pub hasher: Vec<Box<dyn Hasher>>,
    pub layers: Vec<Vec<Hash>>,
}

#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct MerkleProof {
    pub depth: usize,
    pub indices: Vec<usize>,
    pub proof: Vec<Hash>,
}

#[derive(Debug, Clone)]
pub struct MerkleRuntimeConfig {
    pub construct_hasher: fn() -> Box<dyn Hasher>,
}

impl MerkleRuntimeConfig {
    pub fn new(construct_hasher: fn() -> Box<dyn Hasher>) -> Self {
        Self {
            construct_hasher: construct_hasher,
        }
    }
}

#[derive(Debug, Clone)]
pub enum Error {
    IndexOutOfBounds,
    InsufficientSiblings,
    ExcessSiblings,
    LeafMismatch,
    RootMismatch,
}

impl MerkleTreeHasher {
    /// Construct a Merkle tree hasher with a given depth and hasher type.
    pub fn new(depth: usize, hasher: Hashers) -> Self {
        assert!(depth <= 48, "Depth too large"); // Arbitrary limit
        Self {
            root: HASH_ZERO,
            depth,
            hasher: iter::repeat_with(|| hasher.construct())
                .take(depth + 1)
                .collect(),
            layers: Vec::new(),
        }
    }

    /// Construct a Merkle tree hasher from a list of hashers for each level.
    pub fn from_hashers(hashers: &[Hashers]) -> Self {
        Self {
            root: HASH_ZERO,
            depth: hashers.len(),
            hasher: hashers.iter().map(|h| h.construct()).collect(),
            layers: Vec::new(),
        }
    }

    pub fn size_at_depth(&self, depth: usize) -> usize {
        assert!(depth <= self.depth, "Depth too large");
        1 << depth
    }

    pub fn hasher_at_depth(&self, depth: usize) -> &dyn Hasher {
        &*self.hasher[depth]
    }

    pub fn from_hasher_fn(depth: usize, construct: fn() -> Box<dyn Hasher>) -> Self {
        assert!(depth <= 48, "Depth too large");
        Self {
            root: HASH_ZERO,
            depth,
            hasher: iter::repeat_with(|| construct()).take(depth + 1).collect(),
            layers: Vec::new(),
        }
    }
}