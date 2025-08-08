//! Implementation of the merkle tree engine for any RustCrypto digest.

use std::{any::type_name, fmt::Debug, marker::PhantomData};

use ark_ff::{Fp, FpConfig};
use ark_serialize::CanonicalSerialize;
use digest::{Digest, FixedOutputReset};
use maybe_rayon::prelude::*;
use zerocopy::{Immutable, IntoBytes};

use super::{Engine, Hash};

pub trait DigestUpdater {
    type Item;

    fn new() -> Self;
    fn update(&mut self, digest: &mut impl Digest, input: &[Self::Item]);
}

/// Implementation of the digest engine for any RustCrypto digest.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct DigestEngine<D, U>
where
    D: Digest + FixedOutputReset + Sync + Send,
    U: DigestUpdater + Sync + Send,
{
    _digest: PhantomData<D>,
    _updater: PhantomData<U>,
}

pub struct ZeroCopyUpdater<T>(PhantomData<T>)
where
    T: Immutable + IntoBytes;

pub struct ArkUpdater<T>(Vec<u8>, PhantomData<T>)
where
    T: CanonicalSerialize;

pub struct ArkFieldUpdater<C, const N: usize>(PhantomData<C>)
where
    C: FpConfig<N>;

impl<T> DigestUpdater for ZeroCopyUpdater<T>
where
    T: Immutable + IntoBytes,
{
    type Item = T;

    fn new() -> Self {
        Self(PhantomData)
    }

    fn update(&mut self, digest: &mut impl Digest, input: &[T]) {
        Digest::update(digest, input.as_bytes());
    }
}

impl<T> DigestUpdater for ArkUpdater<T>
where
    T: CanonicalSerialize,
{
    type Item = T;

    fn new() -> Self {
        Self(Vec::new(), PhantomData)
    }

    fn update(&mut self, digest: &mut impl Digest, input: &[T]) {
        for item in input {
            self.0.clear();
            item.serialize_compressed(&mut self.0)
                .expect("Writing to vec is infallible");
            Digest::update(digest, &self.0);
        }
    }
}

impl<C, const N: usize> DigestUpdater for ArkFieldUpdater<C, N>
where
    C: FpConfig<N>,
{
    type Item = Fp<C, N>;

    fn new() -> Self {
        Self(PhantomData)
    }

    fn update(&mut self, digest: &mut impl Digest, input: &[Fp<C, N>]) {
        for item in input {
            // Note these are in Montgomery form!
            let limbs = item.0 .0;
            Digest::update(digest, limbs.as_bytes());
        }
    }
}

impl<D, U> DigestEngine<D, U>
where
    D: Digest + FixedOutputReset + Sync + Send,
    U: DigestUpdater + Sync + Send,
{
    pub fn new() -> Self {
        assert!(
            <D as Digest>::output_size() >= 32,
            "Digest must produce at least 32-byte output"
        );
        Self {
            _digest: PhantomData,
            _updater: PhantomData,
        }
    }
}

impl<T, D, U> Engine<T> for DigestEngine<D, U>
where
    T: Sync,
    D: Digest + FixedOutputReset + Sync + Send,
    U: DigestUpdater<Item = T> + Sync + Send,
{
    fn leaf_hash(&self, input: &[T], leaf_size: usize, out: &mut [Hash]) {
        assert_eq!(out.len() * leaf_size, input.len());
        input
            .par_chunks_exact(leaf_size)
            .zip(out.par_iter_mut())
            .for_each_init(
                || (D::new(), U::new()),
                |(digest, updater), (chunk, out)| {
                    updater.update(digest, chunk);
                    let hash = digest.finalize_reset();
                    out.copy_from_slice(&hash[..32]);
                },
            );
    }

    fn node_hash(&self, input: &[Hash], out: &mut [Hash]) {
        assert_eq!(2 * out.len(), input.len());
        input
            .par_chunks_exact(2)
            .zip(out.par_iter_mut())
            .for_each_init(
                || D::new(),
                |digest, (chunk, out)| {
                    Digest::update(digest, chunk.as_bytes());
                    let hash = digest.finalize_reset();
                    out.copy_from_slice(&hash[..32]);
                },
            );
    }
}

impl<D, U> Debug for DigestEngine<D, U>
where
    D: Digest + FixedOutputReset + Sync + Send,
    U: DigestUpdater + Sync + Send,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DigestEngine")
            .field("digest_type", &type_name::<D>())
            .field("updater_type", &type_name::<U>())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use sha3::Keccak256;

    use super::*;

    #[test]
    fn hash_keccak() {
        type MyEngine = DigestEngine<Keccak256, ZeroCopyUpdater<u8>>;
        let engine: Box<dyn Engine<u8>> = Box::new(MyEngine::new());
        let input = [Hash::default(); 2];
        let mut out = [Hash::default(); 1];
        engine.node_hash(&input, &mut out);
        assert_eq!(
            hex::encode(out[0]),
            "ad3228b676f7d3cd4284a5443f17f1962b36e491b30a40b2405849e597ba5fb5"
        );
    }
}
