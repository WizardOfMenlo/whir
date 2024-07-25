pub mod blake2;
pub mod blake3;
pub mod mock;
pub mod sha3;

use std::{borrow::Borrow, marker::PhantomData, sync::atomic::AtomicUsize};

use ark_crypto_primitives::crh::CRHScheme;
use ark_serialize::CanonicalSerialize;
use lazy_static::lazy_static;
use rand::RngCore;

#[derive(Debug, Default)]
pub struct HashCounter {
    counter: AtomicUsize,
}

lazy_static! {
    static ref HASH_COUNTER: HashCounter = HashCounter::default();
}

impl HashCounter {
    pub(crate) fn add() -> usize {
        HASH_COUNTER
            .counter
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
    }

    pub fn reset() {
        HASH_COUNTER
            .counter
            .store(0, std::sync::atomic::Ordering::SeqCst)
    }

    pub fn get() -> usize {
        HASH_COUNTER
            .counter
            .load(std::sync::atomic::Ordering::SeqCst)
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
        _: &Self::Parameters,
        input: T,
    ) -> Result<Self::Output, ark_crypto_primitives::Error> {
        let mut buf = vec![];
        CanonicalSerialize::serialize_compressed(input.borrow(), &mut buf)?;
        Ok(buf)
    }
}
