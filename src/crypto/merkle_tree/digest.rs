use ark_crypto_primitives::sponge::Absorb;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

/// A generic fixed-size digest wrapper used in cryptographic hashing.
///
/// This struct represents a digest as a `[u8; N]` byte array, where `N` is the
/// compile-time size of the digest.
///
/// `GenericDigest` is intended to be used as the output type for cryptographic
/// hash functions (e.g., Blake3, Keccak).
///
/// # Type Parameters
/// - `N`: The size of the digest in bytes (e.g., 32 for a 256-bit hash).
#[derive(Clone, Debug, Eq, PartialEq, Hash, CanonicalSerialize, CanonicalDeserialize)]
pub struct GenericDigest<const N: usize>(pub(crate) [u8; N]);

impl<const N: usize> Default for GenericDigest<N> {
    fn default() -> Self {
        Self([0; N])
    }
}

impl<const N: usize> AsRef<[u8]> for GenericDigest<N> {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

impl<const N: usize> From<[u8; N]> for GenericDigest<N> {
    fn from(value: [u8; N]) -> Self {
        Self(value)
    }
}

impl<const N: usize> Absorb for GenericDigest<N> {
    fn to_sponge_bytes(&self, dest: &mut Vec<u8>) {
        dest.extend_from_slice(&self.0);
    }

    fn to_sponge_field_elements<F: ark_ff::PrimeField>(&self, dest: &mut Vec<F>) {
        dest.push(F::from_be_bytes_mod_order(&self.0));
    }
}
