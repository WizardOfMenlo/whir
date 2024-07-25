use std::{borrow::Borrow, marker::PhantomData};

use ark_crypto_primitives::{
    crh::{CRHScheme, TwoToOneCRHScheme},
    merkle_tree::{ByteDigestConverter, Config},
};
use ark_serialize::CanonicalSerialize;
use rand::RngCore;

pub struct Mock;

impl TwoToOneCRHScheme for Mock {
    type Input = [u8];
    type Output = Vec<u8>;
    type Parameters = ();

    fn setup<R: RngCore>(_: &mut R) -> Result<Self::Parameters, ark_crypto_primitives::Error> {
        Ok(())
    }

    fn evaluate<T: Borrow<Self::Input>>(
        _: &Self::Parameters,
        _: T,
        _: T,
    ) -> Result<Self::Output, ark_crypto_primitives::Error> {
        Ok(vec![0u8; 32])
    }

    fn compress<T: Borrow<Self::Output>>(
        _: &Self::Parameters,
        _: T,
        _: T,
    ) -> Result<Self::Output, ark_crypto_primitives::Error> {
        Ok(vec![0u8; 32])
    }
}

pub type LeafH<F> = super::LeafIdentityHasher<F>;
pub type CompressH = Mock;

#[derive(Debug, Default)]
pub struct MerkleTreeParams<F>(PhantomData<F>);

impl<F: CanonicalSerialize + Send> Config for MerkleTreeParams<F> {
    type Leaf = F;

    type LeafDigest = <LeafH<F> as CRHScheme>::Output;
    type LeafInnerDigestConverter = ByteDigestConverter<Self::LeafDigest>;
    type InnerDigest = <CompressH as TwoToOneCRHScheme>::Output;

    type LeafHash = LeafH<F>;
    type TwoToOneHash = CompressH;
}

pub fn default_config<F: CanonicalSerialize + Send>(
    rng: &mut impl RngCore,
    _leaf_arity: usize,
) -> (
    <LeafH<F> as CRHScheme>::Parameters,
    <CompressH as TwoToOneCRHScheme>::Parameters,
) {
    let leaf_hash_params = <LeafH<F> as CRHScheme>::setup(rng).unwrap();
    let two_to_one_params = <CompressH as TwoToOneCRHScheme>::setup(rng)
        .unwrap()
        .clone();

    (leaf_hash_params, two_to_one_params)
}
