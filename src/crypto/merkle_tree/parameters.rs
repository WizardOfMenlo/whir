use std::{hash::Hash, marker::PhantomData};

use ark_crypto_primitives::{
    crh::{CRHScheme, TwoToOneCRHScheme},
    merkle_tree::Config,
    sponge::Absorb,
};
use ark_ff::Field;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use rand::{CryptoRng, RngCore};
use serde::{Deserialize, Serialize};
use spongefish::{
    ByteDomainSeparator, BytesToUnitDeserialize, BytesToUnitSerialize, DomainSeparator, DuplexSpongeInterface, ProofError, ProofResult, ProverState, Unit, VerifierState
};

use super::{digest::GenericDigest, IdentityDigestConverter};
use crate::whir::{
    domainsep::DigestDomainSeparator,
    utils::{DigestToUnitDeserialize, DigestToUnitSerialize, HintDeserialize, HintSerialize},
};

/// A generic Merkle tree config usable across hash types (e.g., Blake3, Keccak).
///
/// # Type Parameters:
/// - `F`: Field element used in the leaves
/// - `LeafH`: Leaf hash function
/// - `CompressH`: Internal node hasher
/// - `Digest`: Digest type
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct MerkleTreeParams<F, LeafH, CompressH, Digest> {
    #[serde(skip)]
    _marker: PhantomData<(F, LeafH, CompressH, Digest)>,
}

impl<F, LeafH, CompressH, Digest> Config for MerkleTreeParams<F, LeafH, CompressH, Digest>
where
    F: CanonicalSerialize + Send,
    LeafH: CRHScheme<Input = [F], Output = Digest>,
    CompressH: TwoToOneCRHScheme<Input = Digest, Output = Digest>,
    Digest: Clone
        + std::fmt::Debug
        + Default
        + CanonicalSerialize
        + CanonicalDeserialize
        + Eq
        + PartialEq
        + Hash
        + Send
        + Absorb,
{
    type Leaf = [F];

    type LeafDigest = Digest;
    type LeafInnerDigestConverter = IdentityDigestConverter<Digest>;
    type InnerDigest = Digest;

    type LeafHash = LeafH;
    type TwoToOneHash = CompressH;
}

impl<F: Field, LeafH, CompressH, const N: usize>
    DigestDomainSeparator<MerkleTreeParams<F, LeafH, CompressH, GenericDigest<N>>>
    for DomainSeparator
where
    LeafH: CRHScheme<Input = [F], Output = GenericDigest<N>>,
    CompressH: TwoToOneCRHScheme<Input = GenericDigest<N>, Output = GenericDigest<N>>,
{
    fn add_digest(self, label: &str) -> Self {
        self.add_bytes(N, label)
    }
}

impl<F: Field, LeafH, CompressH, const N: usize>
    DigestToUnitSerialize<MerkleTreeParams<F, LeafH, CompressH, GenericDigest<N>>> for ProverState
where
    LeafH: CRHScheme<Input = [F], Output = GenericDigest<N>>,
    CompressH: TwoToOneCRHScheme<Input = GenericDigest<N>, Output = GenericDigest<N>>,
{
    fn add_digest(&mut self, digest: GenericDigest<N>) -> ProofResult<()> {
        self.add_bytes(&digest.0)
            .map_err(ProofError::InvalidDomainSeparator)
    }
}

impl<H, U, R> HintSerialize for ProverState<H, U, R>
where
    U: Unit,
    H: DuplexSpongeInterface<U>,
    R: RngCore + CryptoRng,
{
    fn hint<T: CanonicalSerialize>(&mut self, hint: &T) -> ProofResult<()> {
        let mut bytes = Vec::new();
        hint.serialize_compressed(&mut bytes)?;
        self.hint_bytes(&bytes)?;
        Ok(())
    }
}

impl<F: Field, LeafH, CompressH, const N: usize>
    DigestToUnitDeserialize<MerkleTreeParams<F, LeafH, CompressH, GenericDigest<N>>>
    for VerifierState<'_>
where
    LeafH: CRHScheme<Input = [F], Output = GenericDigest<N>>,
    CompressH: TwoToOneCRHScheme<Input = GenericDigest<N>, Output = GenericDigest<N>>,
{
    fn read_digest(&mut self) -> ProofResult<GenericDigest<N>> {
        let mut digest = [0u8; N];
        self.fill_next_bytes(&mut digest)?;
        Ok(digest.into())
    }
}

impl<H, U> HintDeserialize for VerifierState<'_, H, U>
where
    U: Unit,
    H: DuplexSpongeInterface<U>,
{
    fn hint<T: CanonicalDeserialize>(&mut self) -> ProofResult<T> {
        let mut bytes = self.hint_bytes()?;
        Ok(T::deserialize_compressed(&mut bytes)?)
    }
}

/// Returns the `(leaf_hash_params, two_to_one_hash_params)` for any compatible Merkle tree.
///
/// # Type Parameters
/// - `F`: The leaf field element type
/// - `LeafH`: The leaf hash function
/// - `CompressH`: The two-to-one internal hash function
///
/// # Panics
/// Panics if `setup()` fails (which should not happen for deterministic hashers).
pub fn default_config<F, LeafH, CompressH>(
    rng: &mut impl RngCore,
) -> (
    <LeafH as CRHScheme>::Parameters,
    <CompressH as TwoToOneCRHScheme>::Parameters,
)
where
    F: CanonicalSerialize + Send,
    LeafH: CRHScheme<Input = [F]> + Send,
    CompressH: TwoToOneCRHScheme + Send,
{
    (
        LeafH::setup(rng).expect("Failed to setup Leaf hash"),
        CompressH::setup(rng).expect("Failed to setup Compress hash"),
    )
}
