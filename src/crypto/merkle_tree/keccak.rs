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
use sha3::Digest;

#[derive(
    Debug, Default, Clone, Copy, Eq, PartialEq, Hash, CanonicalSerialize, CanonicalDeserialize,
)]
pub struct KeccakDigest([u8; 32]);

impl Absorb for KeccakDigest {
    fn to_sponge_bytes(&self, dest: &mut Vec<u8>) {
        dest.extend_from_slice(&self.0);
    }

    fn to_sponge_field_elements<F: ark_ff::PrimeField>(&self, dest: &mut Vec<F>) {
        let mut buf = [0; 32];
        buf.copy_from_slice(&self.0);
        dest.push(F::from_be_bytes_mod_order(&buf));
    }
}

impl From<[u8; 32]> for KeccakDigest {
    fn from(value: [u8; 32]) -> Self {
        Self(value)
    }
}

impl AsRef<[u8]> for KeccakDigest {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

pub struct KeccakLeafHash<F>(PhantomData<F>);
pub struct KeccakTwoToOneCRHScheme;

impl<F: CanonicalSerialize + Send> CRHScheme for KeccakLeafHash<F> {
    type Input = [F];
    type Output = KeccakDigest;
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

        let mut h = sha3::Keccak256::new();
        h.update(&buf);

        let mut output = [0; 32];
        output.copy_from_slice(&h.finalize()[..]);
        HashCounter::add();
        Ok(KeccakDigest(output))
    }
}

impl TwoToOneCRHScheme for KeccakTwoToOneCRHScheme {
    type Input = KeccakDigest;
    type Output = KeccakDigest;
    type Parameters = ();

    fn setup<R: RngCore>(_: &mut R) -> Result<Self::Parameters, ark_crypto_primitives::Error> {
        Ok(())
    }

    fn evaluate<T: Borrow<Self::Input>>(
        (): &Self::Parameters,
        left_input: T,
        right_input: T,
    ) -> Result<Self::Output, ark_crypto_primitives::Error> {
        let mut h = sha3::Keccak256::new();
        h.update(left_input.borrow().0);
        h.update(right_input.borrow().0);
        let mut output = [0; 32];
        output.copy_from_slice(&h.finalize()[..]);
        HashCounter::add();
        Ok(KeccakDigest(output))
    }

    fn compress<T: Borrow<Self::Output>>(
        parameters: &Self::Parameters,
        left_input: T,
        right_input: T,
    ) -> Result<Self::Output, ark_crypto_primitives::Error> {
        <Self as TwoToOneCRHScheme>::evaluate(parameters, left_input, right_input)
    }
}

pub type LeafH<F> = KeccakLeafHash<F>;
pub type CompressH = KeccakTwoToOneCRHScheme;

#[derive(Debug, Default, Clone)]
pub struct MerkleTreeParams<F>(PhantomData<F>);

impl<F: CanonicalSerialize + Send> Config for MerkleTreeParams<F> {
    type Leaf = [F];

    type LeafDigest = <LeafH<F> as CRHScheme>::Output;
    type LeafInnerDigestConverter = IdentityDigestConverter<KeccakDigest>;
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
    fn add_digest(&mut self, digest: KeccakDigest) -> ProofResult<()> {
        self.add_bytes(&digest.0).map_err(ProofError::InvalidIO)
    }
}

impl<F: Field> DigestReader<MerkleTreeParams<F>> for Arthur<'_> {
    fn read_digest(&mut self) -> ProofResult<KeccakDigest> {
        let mut digest = [0; 32];
        self.fill_next_bytes(&mut digest)?;
        Ok(KeccakDigest(digest))
    }
}
