use std::{borrow::Borrow, marker::PhantomData};

use ark_crypto_primitives::{
    crh::{CRHScheme, TwoToOneCRHScheme},
    merkle_tree::Config,
};
use ark_ff::Field;
use ark_serialize::CanonicalSerialize;
use rand::RngCore;
use spongefish::{
    ByteDomainSeparator, BytesToUnitDeserialize, BytesToUnitSerialize, DomainSeparator, ProofError,
    ProofResult, ProverState, VerifierState,
};

use super::{digest::GenericDigest, HashCounter, IdentityDigestConverter};
use crate::whir::{
    domainsep::DigestDomainSeparator,
    utils::{DigestToUnitDeserialize, DigestToUnitSerialize},
};

pub type Blake3Digest = GenericDigest<32>;

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
        let mut buf = Vec::new();
        input.borrow().serialize_compressed(&mut buf)?;

        let output: [_; 32] = blake3::hash(&buf).into();
        HashCounter::add();
        Ok(output.into())
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
        let output: [_; 32] =
            blake3::hash(&[left_input.borrow().0, right_input.borrow().0].concat()).into();
        HashCounter::add();
        Ok(output.into())
    }

    fn compress<T: Borrow<Self::Output>>(
        parameters: &Self::Parameters,
        left_input: T,
        right_input: T,
    ) -> Result<Self::Output, ark_crypto_primitives::Error> {
        Self::evaluate(parameters, left_input, right_input)
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
    (
        LeafH::<F>::setup(rng).expect("Failed to setup LeafH"),
        CompressH::setup(rng).expect("Failed to setup CompressH"),
    )
}

impl<F: Field> DigestDomainSeparator<MerkleTreeParams<F>> for DomainSeparator {
    fn add_digest(self, label: &str) -> Self {
        self.add_bytes(32, label)
    }
}

impl<F: Field> DigestToUnitSerialize<MerkleTreeParams<F>> for ProverState {
    fn add_digest(&mut self, digest: Blake3Digest) -> ProofResult<()> {
        self.add_bytes(&digest.0)
            .map_err(ProofError::InvalidDomainSeparator)
    }
}

impl<F: Field> DigestToUnitDeserialize<MerkleTreeParams<F>> for VerifierState<'_> {
    fn read_digest(&mut self) -> ProofResult<Blake3Digest> {
        let mut digest = [0; 32];
        self.fill_next_bytes(&mut digest)?;
        Ok(digest.into())
    }
}
