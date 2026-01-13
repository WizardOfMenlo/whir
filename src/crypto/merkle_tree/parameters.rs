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
use spongefish::{DomainSeparator, DuplexSpongeInterface, ProverState, VerifierState};

use super::{digest::GenericDigest, IdentityDigestConverter};
use crate::ark_rand::ArkRand;

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
    let mut rng = ArkRand(rng);
    (
        LeafH::setup(&mut rng).expect("Failed to setup Leaf hash"),
        CompressH::setup(&mut rng).expect("Failed to setup Compress hash"),
    )
}
