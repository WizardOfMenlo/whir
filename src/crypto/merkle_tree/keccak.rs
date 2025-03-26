use std::{borrow::Borrow, marker::PhantomData};

use ark_crypto_primitives::{
    crh::{CRHScheme, TwoToOneCRHScheme},
    merkle_tree::Config,
};
use ark_ff::Field;
use ark_serialize::CanonicalSerialize;
use rand::RngCore;
use sha3::Digest;
use spongefish::{
    ByteDomainSeparator, BytesToUnitDeserialize, BytesToUnitSerialize, DomainSeparator, ProofError,
    ProofResult, ProverState, VerifierState,
};

use super::{HashCounter, IdentityDigestConverter};
use crate::{
    crypto::merkle_tree::digest::GenericDigest,
    whir::{
        domainsep::DigestDomainSeparator,
        utils::{DigestToUnitDeserialize, DigestToUnitSerialize},
    },
};

pub type KeccakDigest = GenericDigest<32>;

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
        let mut buf = Vec::new();
        input.borrow().serialize_compressed(&mut buf)?;

        let output = sha3::Keccak256::digest(&buf).into();
        HashCounter::add();
        Ok(GenericDigest::<32>(output))
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
        let output = sha3::Keccak256::new()
            .chain_update(left_input.borrow().0)
            .chain_update(right_input.borrow().0)
            .finalize()
            .into();

        HashCounter::add();
        Ok(GenericDigest::<32>(output))
    }

    fn compress<T: Borrow<Self::Output>>(
        parameters: &Self::Parameters,
        left_input: T,
        right_input: T,
    ) -> Result<Self::Output, ark_crypto_primitives::Error> {
        Self::evaluate(parameters, left_input, right_input)
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
    (
        <LeafH<F> as CRHScheme>::setup(rng).expect("Leaf hash setup failed"),
        <CompressH as TwoToOneCRHScheme>::setup(rng).expect("Compress hash setup failed"),
    )
}

impl<F: Field> DigestDomainSeparator<MerkleTreeParams<F>> for DomainSeparator {
    fn add_digest(self, label: &str) -> Self {
        self.add_bytes(32, label)
    }
}

impl<F: Field> DigestToUnitSerialize<MerkleTreeParams<F>> for ProverState {
    fn add_digest(&mut self, digest: KeccakDigest) -> ProofResult<()> {
        self.add_bytes(&digest.0)
            .map_err(ProofError::InvalidDomainSeparator)
    }
}

impl<F: Field> DigestToUnitDeserialize<MerkleTreeParams<F>> for VerifierState<'_> {
    fn read_digest(&mut self) -> ProofResult<KeccakDigest> {
        let mut digest = [0; 32];
        self.fill_next_bytes(&mut digest)?;
        Ok(digest.into())
    }
}
