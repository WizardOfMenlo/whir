use std::{borrow::Borrow, marker::PhantomData};

use ark_crypto_primitives::{
    crh::{CRHScheme, TwoToOneCRHScheme},
    Error,
};
use ark_serialize::CanonicalSerialize;
use rand::RngCore;
use sha3::Digest;

use super::{parameters::MerkleTreeParams, HashCounter};
use crate::crypto::merkle_tree::digest::GenericDigest;

pub type KeccakDigest = GenericDigest<32>;
pub type KeccakMerkleTreeParams<F> =
    MerkleTreeParams<F, KeccakLeafHash<F>, KeccakCompress, KeccakDigest>;

#[derive(Clone)]
pub struct KeccakLeafHash<F>(PhantomData<F>);

impl<F: CanonicalSerialize + Send> CRHScheme for KeccakLeafHash<F> {
    type Input = [F];
    type Output = KeccakDigest;
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

        let output: [_; 32] = sha3::Keccak256::digest(&buf).into();
        HashCounter::add();
        Ok(output.into())
    }
}

#[derive(Clone)]
pub struct KeccakCompress;

impl TwoToOneCRHScheme for KeccakCompress {
    type Input = KeccakDigest;
    type Output = KeccakDigest;
    type Parameters = ();

    fn setup<R: RngCore>(_: &mut R) -> Result<Self::Parameters, Error> {
        Ok(())
    }

    fn evaluate<T: Borrow<Self::Input>>(
        (): &Self::Parameters,
        left_input: T,
        right_input: T,
    ) -> Result<Self::Output, Error> {
        let output: [_; 32] = sha3::Keccak256::new()
            .chain_update(left_input.borrow().0)
            .chain_update(right_input.borrow().0)
            .finalize()
            .into();

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
