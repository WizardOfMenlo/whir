use ark_crypto_primitives::{crh::CRHScheme, merkle_tree::DigestConverter, Error};
use ark_serialize::CanonicalSerialize;
use rand::RngCore;
use std::sync::atomic::Ordering;
use std::{borrow::Borrow, marker::PhantomData, sync::atomic::AtomicUsize};

pub mod blake3;
pub mod keccak;
pub mod mock;

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

#[derive(Debug, Default)]
pub struct LeafIdentityHasher<F>(PhantomData<F>);

impl<F: CanonicalSerialize + Send> CRHScheme for LeafIdentityHasher<F> {
    type Input = F;
    type Output = Vec<u8>;
    type Parameters = ();

    fn setup<R: RngCore>(_: &mut R) -> Result<Self::Parameters, ark_crypto_primitives::Error> {
        Ok(())
    }

    fn evaluate<T: Borrow<Self::Input>>(
        (): &Self::Parameters,
        input: T,
    ) -> Result<Self::Output, ark_crypto_primitives::Error> {
        let mut buf = vec![];
        CanonicalSerialize::serialize_compressed(input.borrow(), &mut buf)?;
        Ok(buf)
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
