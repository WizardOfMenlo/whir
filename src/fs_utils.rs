use sha3::{Digest, Keccak256};
use std::marker::PhantomData;

use ark_ff::{Field, PrimeField};
use ark_serialize::SerializationError;
use nimue::{
    plugins::{ark::FieldIOPattern, pow::PoWIOPattern},
    IOPattern, ProofError,
};

/// Fiat shamir, for the EVM
pub struct EVMFs<F: Field> {
    _p: PhantomData<F>,
}

impl<F: PrimeField> EVMFs<F> {
    fn scalar_to_bytes(scalar: &F) -> Result<[u8; 32], SerializationError> {
        let mut bytes = [0_u8; 32];
        scalar.serialize_uncompressed(&mut bytes[..])?;
        bytes.reverse(); // follows EVM endianness
        Ok(bytes)
    }

    pub fn evm_encode_u32(v: u32) -> [u8; 4] {
        let mut bytes = [0_u8; 4];
        bytes.copy_from_slice(&v.to_be_bytes());
        bytes
    }

    fn bytes_to_scalar(bytes: &[u8; 32]) -> F {
        F::from_be_bytes_mod_order(bytes)
    }

    fn keccak(left: &[u8], right: &[u8]) -> [u8; 32] {
        let mut keccak = Keccak256::new();
        keccak.update([left, right].concat());
        keccak.finalize().into()
    }

    /// Derive `n` challenges from provided initial `init` value
    /// h(h( ... h(init, 0), n-2), n-1)
    fn derive_challenges(init: &[u8], n: u32) -> Vec<F> {
        let mut challenges =
            Vec::with_capacity(n.try_into().expect("are you on a 16 bits machine?"));

        // push first challenge
        let mut challenge_bytes = Self::keccak(init, &0_u32.to_be_bytes());
        challenges.push(Self::bytes_to_scalar(&challenge_bytes));

        // push remaining challenges
        for i in 1..n {
            challenge_bytes = Self::keccak(&challenge_bytes, &i.to_be_bytes());
            challenges.push(Self::bytes_to_scalar(&challenge_bytes));
        }

        challenges
    }

    /// Derive `n` challenges using `scalars.len()` provided values.
    /// Returns Result, due to possible serialization errors for F to bytes serialization
    pub fn derive_challenges_from_scalars(scalars: &[F], n: u32) -> Result<Vec<F>, ProofError> {
        let value = scalars
            .iter()
            .map(Self::scalar_to_bytes)
            .collect::<Result<Vec<[u8; 32]>, SerializationError>>()?
            .concat();

        Ok(Self::derive_challenges(&value, n))
    }

    // Derive `n` challenges using provided bytes
    pub fn derive_challenges_from_bytes(bytes: &[u8], n: u32) -> Vec<F> {
        Self::derive_challenges(bytes, n)
    }
}

pub trait OODIOPattern<F: Field> {
    fn add_ood(self, num_samples: usize) -> Self;
}

impl<F> OODIOPattern<F> for IOPattern
where
    F: Field,
    IOPattern: FieldIOPattern<F>,
{
    fn add_ood(self, num_samples: usize) -> Self {
        if num_samples > 0 {
            self.challenge_scalars(num_samples, "ood_query")
                .add_scalars(num_samples, "ood_ans")
        } else {
            self
        }
    }
}

pub trait WhirPoWIOPattern {
    fn pow(self, bits: f64) -> Self;
}

impl WhirPoWIOPattern for IOPattern
where
    IOPattern: PoWIOPattern,
{
    fn pow(self, bits: f64) -> Self {
        if bits > 0. {
            self.challenge_pow("pow_queries")
        } else {
            self
        }
    }
}
