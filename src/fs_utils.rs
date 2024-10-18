use sha3::{Digest, Keccak256};
use std::{marker::PhantomData, usize};

use ark_ff::{Field, PrimeField};
use ark_serialize::SerializationError;
use nimue::{
    plugins::{ark::FieldIOPattern, pow::PoWIOPattern},
    IOPattern, ProofError,
};

/// Fiat shamir, for the EVM
/// Prototype implementation
pub struct EVMFs<F: Field> {
    _p: PhantomData<F>,
    transcript: Vec<u8>,
}

impl<F: Field> EVMFs<F> {
    pub fn new() -> Self {
        Self {
            transcript: vec![],
            _p: PhantomData,
        }
    }

    fn scalar_to_bytes(scalar: &F) -> Result<[u8; 32], SerializationError> {
        let mut bytes = [0_u8; 32];
        scalar.serialize_uncompressed(&mut bytes[..])?;
        bytes.reverse(); // follows EVM endianness
        Ok(bytes)
    }

    fn bytes_to_scalar(bytes: &[u8]) -> F {
        F::from_base_prime_field(F::BasePrimeField::from_be_bytes_mod_order(bytes))
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
    /// TODO: be `usize` friendly
    pub fn derive_challenges_from_scalars(scalars: &[F], n: usize) -> Result<Vec<F>, ProofError> {
        let n: u32 = n
            .try_into()
            .expect("EVMFs outputs at most 2^32 - 1 challenges");
        let value = scalars
            .iter()
            .map(Self::scalar_to_bytes)
            .collect::<Result<Vec<[u8; 32]>, SerializationError>>()?
            .concat();

        Ok(Self::derive_challenges(&value, n))
    }

    /// Derive `n` challenges using provided bytes
    /// TODO: be `usize` friendly
    pub fn derive_challenges_from_bytes(bytes: &[u8], n: usize) -> Vec<F> {
        let n: u32 = n
            .try_into()
            .expect("EVMFs outputs at most 2^32 - 1 challenges");
        Self::derive_challenges(bytes, n)
    }

    pub fn transcript(self) -> Vec<u8> {
        self.transcript
    }

    fn push_to_transcript(&mut self, bytes: &[u8]) {
        let mut pushed = bytes.to_vec();
        self.transcript.append(&mut pushed);
    }

    /// Pops `n` scalar elements from the transcript, FIFO
    /// Assumes each scalar is in 32 bytes
    pub fn pop(&mut self, n: usize) -> Vec<F> {
        self.transcript
            .drain(..n * 32)
            .as_slice()
            .chunks(32)
            .map(Self::bytes_to_scalar)
            .collect::<Vec<F>>()
    }

    pub fn push_bytes(&mut self, bytes: &[u8]) {
        assert!(bytes.len() == 32, "EVMFs expects 32 bytes arrays");
        Self::push_to_transcript(self, bytes);
    }

    /// Appends provided scalars to the transcript
    /// Returns Result, due to possible serialization errors for F to bytes serialization
    pub fn push_scalars(&mut self, scalars: &[F]) -> Result<(), ProofError> {
        for scalar in scalars {
            Self::push_to_transcript(self, &Self::scalar_to_bytes(scalar)?);
        }
        Ok(())
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

#[cfg(test)]
pub mod tests {
    use super::EVMFs;
    use crate::crypto::fields::FieldBn256;
    use ark_serialize::CanonicalSerialize;
    pub type F = FieldBn256;

    #[test]
    fn test_evm_fs() {
        let scalars =
            EVMFs::derive_challenges_from_scalars(&[F::from(42), F::from(24), F::from(42)], 1)
                .unwrap();
        assert_eq!(scalars.len(), 1);
        let scalars =
            EVMFs::derive_challenges_from_scalars(&[F::from(42), F::from(24), F::from(42)], 5)
                .unwrap();
        assert_eq!(scalars.len(), 5);
    }

    #[test]
    fn test_transcript() {
        let elements = [F::from(42), F::from(24), F::from(34)];
        let mut bytes = [0_u8; 32];
        let _ = F::from(2).serialize_compressed(&mut bytes[..]);
        let mut evmfs = EVMFs::new();
        evmfs.push_scalars(&elements).unwrap();
        evmfs.push_bytes(&bytes);
    }
}
