use ark_crypto_primitives::{
    crh::{CRHScheme, TwoToOneCRHScheme},
    merkle_tree::{Config, IdentityDigestConverter},
};
use ark_serialize::CanonicalSerialize;
use rand::RngCore;
use sha3::Digest;
use std::{borrow::Borrow, marker::PhantomData};

use crate::crypto::merkle_tree::{
    keccak::{KeccakDigest, KeccakTwoToOneCRHScheme, LeafH},
    HashCounter,
};

/// Sorted hasher configuration
/// Replicates the behavior of the OpenZeppelin MerkleTree implementation
/// where the hash inputs are sorted before hashing
/// (see https://github.com/OpenZeppelin/openzeppelin-contracts/blob/01ef448981be9d20ca85f2faf6ebdf591ce409f3/contracts/utils/cryptography/MerkleProof.sol#L217)
pub struct SortedKeccakTwoToOneCRHScheme;

impl TwoToOneCRHScheme for SortedKeccakTwoToOneCRHScheme {
    type Input = KeccakDigest;
    type Output = KeccakDigest;
    type Parameters = ();

    fn setup<R: RngCore>(_: &mut R) -> Result<Self::Parameters, ark_crypto_primitives::Error> {
        KeccakTwoToOneCRHScheme::setup(&mut rand::thread_rng())
    }

    fn evaluate<T: Borrow<Self::Input>>(
        _: &Self::Parameters,
        left_input: T,
        right_input: T,
    ) -> Result<Self::Output, ark_crypto_primitives::Error> {
        let mut inputs = vec![left_input.borrow().as_ref(), right_input.borrow().as_ref()];
        inputs.sort(); // Sort the inputs

        let mut h = sha3::Keccak256::new();
        for input in inputs {
            h.update(input);
        }

        let mut output = [0; 32];
        output.copy_from_slice(&h.finalize()[..]);
        Ok(KeccakDigest::from(output))
    }

    fn compress<T: Borrow<Self::Output>>(
        parameters: &Self::Parameters,
        left_input: T,
        right_input: T,
    ) -> Result<Self::Output, ark_crypto_primitives::Error> {
        <Self as TwoToOneCRHScheme>::evaluate(parameters, left_input, right_input)
    }
}

pub struct EvmKeccakLeafHash<F>(PhantomData<F>);

impl<F: CanonicalSerialize + Send> CRHScheme for EvmKeccakLeafHash<F> {
    type Input = [F];
    type Output = KeccakDigest;
    type Parameters = ();

    fn setup<R: RngCore>(_: &mut R) -> Result<Self::Parameters, ark_crypto_primitives::Error> {
        Ok(())
    }

    fn evaluate<T: Borrow<Self::Input>>(
        _: &Self::Parameters,
        input: T,
    ) -> Result<Self::Output, ark_crypto_primitives::Error> {
        let mut buf = vec![];

        // Replicates the EVM ABI encode behavior:
        // - Encoded array elements are *not* prefixed with their length;
        // - The reverses are necessary to match the endianness of the EVM.
        for element in input.borrow().iter().rev() {
            element.serialize_uncompressed(&mut buf)?;
        }
        buf.reverse();

        let mut h = sha3::Keccak256::new();
        h.update(&buf);

        let mut output = [0; 32];
        output.copy_from_slice(&h.finalize()[..]);
        HashCounter::add();
        Ok(KeccakDigest::from(output))
    }
}

pub fn sorted_hasher_config<F: CanonicalSerialize + Send>(
    rng: &mut impl RngCore,
) -> (
    <LeafH<F> as CRHScheme>::Parameters,
    <SortedKeccakTwoToOneCRHScheme as TwoToOneCRHScheme>::Parameters,
) {
    <LeafH<F> as CRHScheme>::setup(rng).unwrap();
    <SortedKeccakTwoToOneCRHScheme as TwoToOneCRHScheme>::setup(rng).unwrap();

    ((), ())
}

#[derive(Debug, Default, Clone)]
pub struct MerkleTreeEvmParams<F>(PhantomData<F>);

impl<F: CanonicalSerialize + Send> Config for MerkleTreeEvmParams<F> {
    type Leaf = [F];

    type LeafDigest = <EvmKeccakLeafHash<F> as CRHScheme>::Output;
    type LeafInnerDigestConverter = IdentityDigestConverter<KeccakDigest>;
    type InnerDigest = <SortedKeccakTwoToOneCRHScheme as TwoToOneCRHScheme>::Output;

    type LeafHash = EvmKeccakLeafHash<F>;
    type TwoToOneHash = SortedKeccakTwoToOneCRHScheme;
}
