use std::{borrow::Borrow, marker::PhantomData};

use ark_crypto_primitives::{
    crh::{CRHScheme, TwoToOneCRHScheme},
    merkle_tree::{Config, IdentityDigestConverter},
    sponge::Absorb,
};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use blake2::Digest;
use rand::RngCore;

use super::HashCounter;

#[derive(
    Debug, Default, Clone, Copy, Eq, PartialEq, Hash, CanonicalSerialize, CanonicalDeserialize,
)]
pub struct Blake2Digest([u8; 32]);

impl Absorb for Blake2Digest {
    fn to_sponge_bytes(&self, dest: &mut Vec<u8>) {
        dest.extend_from_slice(&self.0);
    }

    fn to_sponge_field_elements<F: ark_ff::PrimeField>(&self, dest: &mut Vec<F>) {
        let mut buf = [0; 32];
        buf.copy_from_slice(&self.0);
        dest.push(F::from_be_bytes_mod_order(&buf));
    }
}

pub struct Blake2LeafHash<F>(PhantomData<F>);
pub struct Blake2TwoToOneCRHScheme;

impl<F: CanonicalSerialize + Send> CRHScheme for Blake2LeafHash<F> {
    type Input = Vec<F>;
    type Output = Blake2Digest;
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

        let mut h = blake2::Blake2s256::new();
        h.update(&buf);

        let mut output = [0; 32];
        output.copy_from_slice(&h.finalize()[..]);
        HashCounter::add();
        Ok(Blake2Digest(output))
    }
}

impl TwoToOneCRHScheme for Blake2TwoToOneCRHScheme {
    type Input = Blake2Digest;
    type Output = Blake2Digest;
    type Parameters = ();

    fn setup<R: RngCore>(_: &mut R) -> Result<Self::Parameters, ark_crypto_primitives::Error> {
        Ok(())
    }

    fn evaluate<T: Borrow<Self::Input>>(
        _: &Self::Parameters,
        left_input: T,
        right_input: T,
    ) -> Result<Self::Output, ark_crypto_primitives::Error> {
        let mut h = blake2::Blake2s256::new();
        h.update(&left_input.borrow().0);
        h.update(&right_input.borrow().0);
        let mut output = [0; 32];
        output.copy_from_slice(&h.finalize()[..]);
        HashCounter::add();
        Ok(Blake2Digest(output))
    }

    fn compress<T: Borrow<Self::Output>>(
        parameters: &Self::Parameters,
        left_input: T,
        right_input: T,
    ) -> Result<Self::Output, ark_crypto_primitives::Error> {
        <Self as TwoToOneCRHScheme>::evaluate(parameters, left_input, right_input)
    }
}

pub type LeafH<F> = Blake2LeafHash<F>;
pub type CompressH = Blake2TwoToOneCRHScheme;

#[derive(Debug, Default, Clone)]
pub struct MerkleTreeParams<F>(PhantomData<F>);

impl<F: CanonicalSerialize + Send> Config for MerkleTreeParams<F> {
    type Leaf = Vec<F>;

    type LeafDigest = <LeafH<F> as CRHScheme>::Output;
    type LeafInnerDigestConverter = IdentityDigestConverter<Blake2Digest>;
    type InnerDigest = <CompressH as TwoToOneCRHScheme>::Output;

    type LeafHash = LeafH<F>;
    type TwoToOneHash = CompressH;
}

pub fn default_config<F: CanonicalSerialize + Send>(
    rng: &mut impl RngCore,
) -> (
    <LeafH<F> as CRHScheme>::Parameters,
    <CompressH as TwoToOneCRHScheme>::Parameters,
) {
    let leaf_hash_params = <LeafH<F> as CRHScheme>::setup(rng).unwrap();
    let two_to_one_params = <CompressH as TwoToOneCRHScheme>::setup(rng).unwrap();

    (leaf_hash_params, two_to_one_params)
}
