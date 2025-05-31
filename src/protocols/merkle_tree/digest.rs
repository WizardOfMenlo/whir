//! Implementation of the proof of work engine for any RustCrypto digest.

use std::{
    any::type_name,
    fmt::{Debug, Display},
    marker::PhantomData,
};

use ark_serialize::CanonicalSerialize;
use digest::{Digest, FixedOutputReset};
use zerocopy::{Immutable, IntoBytes};

use super::{Engine, Hash};

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct DigestEngine<D>
where
    D: Digest + FixedOutputReset + Sync + Send,
{
    _digest: PhantomData<D>,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct ArkDigestEngine<D>
where
    D: Digest + FixedOutputReset + Sync + Send,
{
    _digest: PhantomData<D>,
}

impl<D> DigestEngine<D>
where
    D: Digest + FixedOutputReset + Sync + Send,
{
    pub fn new() -> Self {
        assert!(
            <D as Digest>::output_size() >= 32,
            "Digest must produce at least 32-byte output"
        );
        Self {
            _digest: PhantomData,
        }
    }
}

impl<D> ArkDigestEngine<D>
where
    D: Digest + FixedOutputReset + Sync + Send,
{
    pub fn new() -> Self {
        assert!(
            <D as Digest>::output_size() >= 32,
            "Digest must produce at least 32-byte output"
        );
        Self {
            _digest: PhantomData,
        }
    }
}

impl<T, D> Engine<T> for DigestEngine<D>
where
    T: Immutable + IntoBytes,
    D: Digest + FixedOutputReset + Sync + Send,
{
    fn leaf_hash(&self, input: &[T], leaf_size: usize, out: &mut [Hash]) {
        assert_eq!(out.len() * leaf_size, input.len());
        let mut digest = D::new();
        for (chunk, out) in input.chunks_exact(leaf_size).zip(out.iter_mut()) {
            Digest::update(&mut digest, chunk.as_bytes());
            let hash = digest.finalize_reset();
            out.copy_from_slice(&hash[..32]);
        }
    }

    fn node_hash(&self, input: &[Hash], out: &mut [Hash]) {
        assert_eq!(2 * out.len(), input.len());
        let mut digest = D::new();
        for (chunk, out) in input.chunks_exact(2).zip(out.iter_mut()) {
            Digest::update(&mut digest, chunk.as_bytes());
            let hash = digest.finalize_reset();
            out.copy_from_slice(&hash[..32]);
        }
    }
}

impl<T, D> Engine<T> for ArkDigestEngine<D>
where
    T: CanonicalSerialize,
    D: Digest + FixedOutputReset + Sync + Send,
{
    fn leaf_hash(&self, input: &[T], leaf_size: usize, out: &mut [Hash]) {
        assert_eq!(out.len() * leaf_size, input.len());
        let mut digest = D::new();
        let mut buffer = Vec::new();
        for (chunk, out) in input.chunks_exact(leaf_size).zip(out.iter_mut()) {
            buffer.clear();
            chunk.serialize_compressed(&mut buffer);
            Digest::update(&mut digest, &buffer);
            let hash = digest.finalize_reset();
            out.copy_from_slice(&hash[..32]);
        }
    }

    fn node_hash(&self, input: &[Hash], out: &mut [Hash]) {
        assert_eq!(2 * out.len(), input.len());
        let mut digest = D::new();
        for (chunk, out) in input.chunks_exact(2).zip(out.iter_mut()) {
            Digest::update(&mut digest, chunk.as_bytes());
            let hash = digest.finalize_reset();
            out.copy_from_slice(&hash[..32]);
        }
    }
}

impl<D> Debug for DigestEngine<D>
where
    D: Digest + FixedOutputReset + Sync + Send,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DigestEngine")
            .field("digest_type", &type_name::<D>())
            .finish()
    }
}

impl<D> Debug for ArkDigestEngine<D>
where
    D: Digest + FixedOutputReset + Sync + Send,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ArkDigestEngine")
            .field("digest_type", &type_name::<D>())
            .finish()
    }
}

impl<D> Display for DigestEngine<D>
where
    D: Digest + FixedOutputReset + Sync + Send,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "DigestEngine<{:?}>", type_name::<D>())
    }
}

impl<D> Display for ArkDigestEngine<D>
where
    D: Digest + FixedOutputReset + Sync + Send,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ArkDigestEngine<{:?}>", type_name::<D>())
    }
}

#[cfg(test)]
mod tests {
    use sha3::Keccak256;

    use super::*;

    #[test]
    fn hash_keccak() {
        let engine: Box<dyn Engine<u8>> = Box::new(DigestEngine::<Keccak256>::new());
        let input = [Hash::default(); 2];
        let mut out = [Hash::default(); 1];
        engine.node_hash(&input, &mut out);
        assert_eq!(
            hex::encode(out[0]),
            "ad3228b676f7d3cd4284a5443f17f1962b36e491b30a40b2405849e597ba5fb5"
        );
    }
}
