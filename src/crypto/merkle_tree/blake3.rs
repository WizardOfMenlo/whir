use std::{borrow::Borrow, marker::PhantomData};

use ark_crypto_primitives::{
    crh::{CRHScheme, TwoToOneCRHScheme},
    Error,
};
use ark_serialize::CanonicalSerialize;
use rand::RngCore;
use serde::{Deserialize, Serialize};

use super::{digest::GenericDigest, parameters::MerkleTreeParams, HashCounter};

/// Digest type used in Blake3-based Merkle trees.
///
/// Alias for a 32-byte generic digest.
pub type Blake3Digest = GenericDigest<32>;

/// Merkle tree configuration using Blake3 as both leaf and node hasher.
pub type Blake3MerkleTreeParams<F> =
    MerkleTreeParams<F, Blake3LeafHash<F>, Blake3Compress, Blake3Digest>;

/// Leaf hash function using Blake3 over compressed `[F]` input.
///
/// This struct implements `CRHScheme` where the input is a slice of
/// canonical-serializable field elements `[F]`, and the output is a
/// 32-byte Blake3 digest.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct Blake3LeafHash<F>(#[serde(skip)] PhantomData<F>);

impl<F: CanonicalSerialize + Send> CRHScheme for Blake3LeafHash<F> {
    type Input = [F];
    type Output = Blake3Digest;
    type Parameters = ();

    fn setup<R: RngCore>(_: &mut R) -> Result<Self::Parameters, Error> {
        Ok(())
    }

    fn evaluate<T: Borrow<Self::Input>>(
        (): &Self::Parameters,
        input: T,
    ) -> Result<Self::Output, Error> {
        let mut buf = Vec::new();
        input.borrow().serialize_compressed(&mut buf)?;

        let output: [_; 32] = blake3::hash(&buf).into();
        HashCounter::add();
        Ok(output.into())
    }
}

/// Node compression function using Blake3 over two 32-byte digests.
///
/// This struct implements `TwoToOneCRHScheme`, combining two digests
/// by concatenation and hashing with Blake3.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Blake3Compress;

impl TwoToOneCRHScheme for Blake3Compress {
    type Input = Blake3Digest;
    type Output = Blake3Digest;
    type Parameters = ();

    fn setup<R: RngCore>(_: &mut R) -> Result<Self::Parameters, Error> {
        Ok(())
    }

    fn evaluate<T: Borrow<Self::Input>>(
        (): &Self::Parameters,
        left_input: T,
        right_input: T,
    ) -> Result<Self::Output, Error> {
        let output: [_; 32] =
            blake3::hash(&[left_input.borrow().0, right_input.borrow().0].concat()).into();
        HashCounter::add();
        Ok(output.into())
    }

    fn compress<T: Borrow<Self::Output>>(
        parameters: &Self::Parameters,
        left_input: T,
        right_input: T,
    ) -> Result<Self::Output, Error> {
        Self::evaluate(parameters, left_input, right_input)
    }
}
