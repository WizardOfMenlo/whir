mod blake3;
mod digest;

use {
    self::{blake3::Blake3Hasher, digest::DigestHasher},
    sha3::{Keccak256, Sha3_256},
};

pub const HASH_ZERO: Hash = [0; 32];

pub type Hash = [u8; 32];

pub enum Hashers {
    Blake3,
    Keccak,
    Sha3,
    Poseidon2Goldilocks,
    Poseidon2Bn254,
}

pub trait Hasher: Send + Sync {
    /// Hash pairs of hashes, i.e. construct the next layer of a Merkle tree.
    ///
    /// Note: Implementation should be single-threaded. Parallelization is taken
    /// care of by the caller.
    fn hash_pairs(&self, blocks: &[Hash], out: &mut [Hash]);
}

impl Hashers {
    /// Construct a hasher of the requisted type.
    ///
    /// Using `dyn Trait` here to avoid generics/monomorphization of the caller.
    pub fn construct(self) -> Box<dyn Hasher> {
        match self {
            Hashers::Blake3 => Box::new(Blake3Hasher::new()),
            Hashers::Keccak => Box::new(DigestHasher::<Keccak256>::new()),
            Hashers::Sha3 => Box::new(DigestHasher::<Sha3_256>::new()),
            _ => unimplemented!(),
        }
    }
}

/// Test if two implementations are equivalent.
/// Used to test an optimized implementation against a reference implementation.
#[cfg(test)]
fn test_equivalent(a: &dyn Hasher, b: &dyn Hasher) {
    use std::array;
    for length in 0..32 {
        let mut input = vec![HASH_ZERO; 2 * length];
        for (i, input) in input.iter_mut().enumerate() {
            *input = array::from_fn(|j| (length + i + j) as u8);
        }
        let mut output_a = vec![HASH_ZERO; length];
        let mut output_b = vec![HASH_ZERO; length];
        a.hash_pairs(&input, &mut output_a);
        b.hash_pairs(&input, &mut output_b);
        assert_eq!(output_a, output_b);
    }
}
