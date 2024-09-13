mod commit;
mod hasher;
mod prove;
mod verify;

pub use hasher::{Hash, Hasher, Hashers, HASH_ZERO};
use std::iter;

pub struct MerkleTreeHasher {
    depth: usize,
    hasher: Vec<Box<dyn Hasher>>,
}

pub enum Error {
    IndexOutOfBounds,
    InsufficientSibblings,
    ExcessSibblings,
    LeafMismatch,
    RootMismatch,
}

impl MerkleTreeHasher {
    /// Construct a Merkle tree hasher with a given depth and hasher type.
    pub fn new(depth: usize, hasher: Hashers) -> Self {
        assert!(depth <= 48, "Depth too large"); // Arbitrary limit
        Self {
            depth,
            hasher: iter::repeat_with(|| hasher.construct())
                .take(depth + 1)
                .collect(),
        }
    }

    /// Construct a Merkle tree hasher from a list of hashers for each level.
    pub fn from_hashers(hashers: &[Hashers]) -> Self {
        Self {
            depth: hashers.len(),
            hasher: hashers.iter().map(|h| h.construct()).collect(),
        }
    }

    pub fn size_at_depth(&self, depth: usize) -> usize {
        assert!(depth <= self.depth, "Depth too large");
        1 << depth
    }

    pub fn hasher_at_depth(&self, depth: usize) -> &dyn Hasher {
        &*self.hasher[depth]
    }
}
