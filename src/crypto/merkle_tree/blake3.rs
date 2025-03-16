use std::{borrow::Borrow, marker::PhantomData};

use super::{HashCounter, IdentityDigestConverter};
use crate::whir::fs_utils::{DigestReader, DigestWriter};
use crate::whir::iopattern::DigestIOPattern;
use ark_crypto_primitives::{
    crh::{CRHScheme, TwoToOneCRHScheme},
    merkle_tree::Config,
    sponge::Absorb,
};
use ark_ff::Field;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use nimue::{
    Arthur, ByteIOPattern, ByteReader, ByteWriter, IOPattern, Merlin, ProofError, ProofResult,
};
use rand::RngCore;

#[derive(
    Debug, Default, Clone, Copy, Eq, PartialEq, Hash, CanonicalSerialize, CanonicalDeserialize,
)]
pub struct Blake3Digest([u8; 32]);

impl AsRef<[u8]> for Blake3Digest {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

impl From<[u8; 32]> for Blake3Digest {
    fn from(value: [u8; 32]) -> Self {
        Self(value)
    }
}

impl Absorb for Blake3Digest {
    fn to_sponge_bytes(&self, dest: &mut Vec<u8>) {
        dest.extend_from_slice(&self.0);
    }

    fn to_sponge_field_elements<F: ark_ff::PrimeField>(&self, dest: &mut Vec<F>) {
        let mut buf = [0; 32];
        buf.copy_from_slice(&self.0);
        dest.push(F::from_be_bytes_mod_order(&buf));
    }
}

pub struct Blake3LeafHash<F>(PhantomData<F>);
pub struct Blake3TwoToOneCRHScheme;

impl<F: CanonicalSerialize + Send> CRHScheme for Blake3LeafHash<F> {
    type Input = [F];
    type Output = Blake3Digest;
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

        let mut h = blake3::Hasher::new();
        h.update(&buf);

        let mut output = [0; 32];
        output.copy_from_slice(h.finalize().as_bytes());
        HashCounter::add();
        Ok(Blake3Digest(output))
    }
}

impl TwoToOneCRHScheme for Blake3TwoToOneCRHScheme {
    type Input = Blake3Digest;
    type Output = Blake3Digest;
    type Parameters = ();

    fn setup<R: RngCore>(_: &mut R) -> Result<Self::Parameters, ark_crypto_primitives::Error> {
        Ok(())
    }

    fn evaluate<T: Borrow<Self::Input>>(
        (): &Self::Parameters,
        left_input: T,
        right_input: T,
    ) -> Result<Self::Output, ark_crypto_primitives::Error> {
        let mut h = blake3::Hasher::new();
        h.update(&left_input.borrow().0);
        h.update(&right_input.borrow().0);
        let mut output = [0; 32];
        output.copy_from_slice(h.finalize().as_bytes());
        HashCounter::add();
        Ok(Blake3Digest(output))
    }

    fn compress<T: Borrow<Self::Output>>(
        parameters: &Self::Parameters,
        left_input: T,
        right_input: T,
    ) -> Result<Self::Output, ark_crypto_primitives::Error> {
        <Self as TwoToOneCRHScheme>::evaluate(parameters, left_input, right_input)
    }
}

pub type LeafH<F> = Blake3LeafHash<F>;
pub type CompressH = Blake3TwoToOneCRHScheme;

#[derive(Debug, Default, Clone)]
pub struct MerkleTreeParams<F>(PhantomData<F>);

impl<F: CanonicalSerialize + Send> Config for MerkleTreeParams<F> {
    type Leaf = [F];

    type LeafDigest = <LeafH<F> as CRHScheme>::Output;
    type LeafInnerDigestConverter = IdentityDigestConverter<Blake3Digest>;
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
    <LeafH<F> as CRHScheme>::setup(rng).unwrap();
    <CompressH as TwoToOneCRHScheme>::setup(rng).unwrap();

    ((), ())
}

impl<F: Field> DigestIOPattern<MerkleTreeParams<F>> for IOPattern {
    fn add_digest(self, label: &str) -> Self {
        self.add_bytes(32, label)
    }
}

impl<F: Field> DigestWriter<MerkleTreeParams<F>> for Merlin {
    fn add_digest(&mut self, digest: Blake3Digest) -> ProofResult<()> {
        self.add_bytes(&digest.0).map_err(ProofError::InvalidIO)
    }
}

impl<F: Field> DigestReader<MerkleTreeParams<F>> for Arthur<'_> {
    fn read_digest(&mut self) -> ProofResult<Blake3Digest> {
        let mut digest = [0; 32];
        self.fill_next_bytes(&mut digest)?;
        Ok(Blake3Digest(digest))
    }
}
