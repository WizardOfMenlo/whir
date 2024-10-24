use nimue::plugins::pow::PowStrategy;
use std::{marker::PhantomData, usize};

use ark_ff::{Field, PrimeField};
use ark_serialize::SerializationError;
use ethers_core::{
    abi::{encode_packed, Token},
    utils::keccak256,
};
use nimue::{
    plugins::{ark::FieldIOPattern, pow::PoWIOPattern},
    IOPattern, ProofError,
};
use serde::{ser::SerializeStruct, Serialize};

/// Fiat shamir, for the EVM
/// Prototype implementation
#[derive(Clone)]
pub struct EVMFs<F: Field> {
    _p: PhantomData<F>,
    transcript: Vec<u8>, // holds full script
    state: Vec<u8>,      // holds values to be hashed for next challenge derivation
}

impl<F: Field> EVMFs<F> {
    pub fn new() -> Self {
        Self {
            transcript: vec![],
            state: vec![],
            _p: PhantomData,
        }
    }

    pub fn to_arthur(&self) -> Self {
        Self {
            transcript: self.transcript.clone(),
            state: vec![],
            _p: PhantomData,
        }
    }

    fn scalar_to_bytes(scalar: &F) -> Result<[u8; 32], SerializationError> {
        let mut bytes = [0_u8; 32];
        scalar.serialize_uncompressed(&mut bytes[..])?;
        bytes.reverse(); // follows EVM endianness
        Ok(bytes)
    }

    pub fn bytes_to_scalar(bytes: &[u8]) -> F {
        F::from_base_prime_field(F::BasePrimeField::from_be_bytes_mod_order(&bytes))
    }

    fn keccak(left: &[u8], right: &[u8]) -> [u8; 32] {
        let to_hash = [left, right].concat();
        let encoded = encode_packed(&[Token::Bytes(to_hash)]).unwrap();
        keccak256(encoded)
    }

    /// Arthur/Merlin
    /// Squeezes `n` scalar elements using current `state`
    /// TODO: handle various n: u32, u64; should align with EVM implementation
    pub fn squeeze_scalars(&mut self, n: usize) -> Vec<F> {
        let mut challenges = Vec::with_capacity(n);
        let n: u32 = n.try_into().unwrap();
        let mut challenge_bytes = Self::keccak(self.state.as_slice(), &0_u32.to_be_bytes());
        challenges.push(Self::bytes_to_scalar(&challenge_bytes));

        // push remaining challenges
        for i in 1..n {
            challenge_bytes = Self::keccak(&challenge_bytes, &i.to_be_bytes());
            challenges.push(Self::bytes_to_scalar(&challenge_bytes));
        }

        self.state = vec![];
        challenges
    }

    /// Arthur/Merlin
    /// Squeezes `n` bytes arrays using current `state`
    /// TODO: handle various n: u32, u64; should align with EVM implementation
    pub fn squeeze_bytes(&mut self, n: usize) -> Vec<[u8; 32]> {
        let mut challenges = Vec::with_capacity(n);

        let n: u32 = n.try_into().unwrap();
        let mut challenge_bytes = Self::keccak(self.state.as_slice(), &0_u32.to_be_bytes());
        challenges.push(challenge_bytes);

        // push remaining challenges
        for i in 1..n {
            challenge_bytes = Self::keccak(&challenge_bytes, &i.to_be_bytes());
            challenges.push(challenge_bytes);
        }

        self.state = vec![];
        challenges
    }

    pub fn transcript(&self) -> Vec<u8> {
        self.transcript.to_vec()
    }

    pub fn state(&self) -> Vec<u8> {
        self.state.to_vec()
    }

    fn push_to_transcript(&mut self, bytes: &[u8]) {
        let mut pushed = bytes.to_vec();
        self.transcript.append(&mut pushed.clone());
        self.state.append(&mut pushed);
    }

    /// Arthur
    /// Removes `n` bytes elements from the transcript, appends them to the state and returns them
    pub fn next_bytes(&mut self, n: usize) -> Vec<u8> {
        let mut sliced = self.transcript[..n].to_vec();
        self.state.append(&mut sliced.clone());
        self.transcript = self.transcript[n..].to_vec();
        sliced.reverse(); // return them in same state before absorption
        sliced
    }

    /// Arthur
    /// Removes `n` scalar elements from the transcript, appends them to the state and returns them
    /// Assumes user is using scalars of 32 bytes each
    pub fn next_scalars(&mut self, n: usize) -> Vec<F> {
        let sliced = self.transcript[..n * 32].to_vec();
        let scalar_bytes = sliced.chunks(32);
        let mut scalars = vec![];
        for bytes in scalar_bytes {
            self.state.append(&mut bytes.to_vec().clone()); // was absorbed after reverse, append as given from
                                                            // transcript
            scalars.push(Self::bytes_to_scalar(&bytes));
        }
        self.transcript = self.transcript[n * 32..].to_vec();
        scalars
    }

    /// Merlin
    pub fn absorb_bytes(&mut self, bytes: &[u8]) {
        assert!(bytes.len() == 32, "EVMFs expects 32 bytes arrays");
        // when serialized uncompressed, F is serialized in be
        // so, assume bytes are be, hence need to reverse them
        let mut le_bytes = bytes.to_vec();
        le_bytes.reverse();
        self.push_to_transcript(&le_bytes);
    }

    /// Merlin
    /// "Absorbs" provided scalars: appends them to the state and the transcript
    /// Returns Result, due to possible serialization errors for F to bytes serialization
    pub fn absorb_scalars(&mut self, scalars: &[F]) -> Result<(), ProofError> {
        for scalar in scalars {
            let bytes = Self::scalar_to_bytes(scalar)?;
            self.push_to_transcript(&bytes);
        }
        Ok(())
    }
}

impl<F: Field> Serialize for EVMFs<F> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut state = serializer.serialize_struct("EVMFs", 2)?;
        let transcript_hex = format!("0x{}", hex::encode(&self.transcript));
        state.serialize_field("transcript", &transcript_hex)?;
        assert!(
            self.state.is_empty(),
            "State should be empty when serializing"
        );
        state.skip_field("state")?;
        state.end()
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

// TODO: implement the EVM compatible PoW module
#[derive(Clone, Copy)]
pub struct KeccakEVMPoW {
    challenge: [u8; 32],
    threshold: u64,
}

impl PowStrategy for KeccakEVMPoW {
    fn new(challenge: [u8; 32], bits: f64) -> Self {
        let threshold = (64.0 - bits).exp2().ceil() as u64;
        Self {
            challenge: bytemuck::cast(challenge),
            threshold,
        }
    }

    fn check(&mut self, nonce: u64) -> bool {
        todo!()
    }
}

// TODO: remove print statements below, used for cross-checks with sol
// ideally removed when ffi is setup
#[cfg(test)]
pub mod tests {

    use super::EVMFs;
    use crate::crypto::fields::FieldBn256;
    pub type F = FieldBn256;

    #[test]
    fn test_evm_fs_merlin() {
        let mut evmfs = EVMFs::new();
        let _ = evmfs
            .absorb_scalars(&[F::from(42), F::from(24), F::from(42)])
            .unwrap();
        let scalars = evmfs.squeeze_scalars(1);
        println!("{}", scalars[0]);
        assert_eq!(evmfs.state.len(), 0);

        let _ = evmfs
            .absorb_scalars(&[F::from(42), F::from(24), F::from(42)])
            .unwrap();
        let scalars = evmfs.squeeze_scalars(5);
        for scalar in scalars {
            println!("{}", scalar);
        }

        assert_eq!(evmfs.transcript().len(), 192); // 6 field elements * 32
        assert_eq!(evmfs.state.len(), 0);

        // New transcript
        let mut evmfs = EVMFs::<F>::new();
        let absorbed_bytes = Vec::from_iter(0_u8..32_u8);
        let absorbed_scalars_1 = [F::from(1), F::from(2)];
        let _ = evmfs.absorb_bytes(&absorbed_bytes);
        let squeezed_1 = evmfs.squeeze_scalars(1)[0];
        let _ = evmfs.absorb_scalars(&absorbed_scalars_1);
        let squeezed_2 = evmfs.squeeze_scalars(1)[0];
        assert_eq!(evmfs.transcript().len(), 96);
        assert_eq!(evmfs.state.len(), 0);

        // Resets state, keep transcript only
        let mut arthur_evmfs = evmfs.to_arthur();
        let _ = arthur_evmfs.next_bytes(32);
        assert_eq!(arthur_evmfs.squeeze_scalars(1)[0], squeezed_1);
        let next_scalars = arthur_evmfs.next_scalars(2);
        assert_eq!(next_scalars[0], absorbed_scalars_1[0]);
        assert_eq!(next_scalars[1], absorbed_scalars_1[1]);
        assert_eq!(arthur_evmfs.squeeze_scalars(1)[0], squeezed_2);
    }

    #[test]
    fn test_evm_fs_arthur() {
        let mut evmfs = EVMFs::<F>::new();
        let transcript = (0_u8..160_u8).into_iter().collect::<Vec<u8>>().to_vec();
        evmfs.transcript = transcript;
        let scalars = evmfs.next_scalars(2);
        let squeezed = evmfs.squeeze_scalars(2);
        for scalar in scalars {
            println!("n: {}", scalar);
        }
        for scalar in squeezed {
            println!("sq: {}", scalar);
        }

        let scalars = evmfs.next_scalars(1);
        let squeezed = evmfs.squeeze_scalars(2);
        for scalar in scalars {
            println!("n: {}", scalar);
        }
        for scalar in squeezed {
            println!("sq: {}", scalar);
        }
        let bytes = evmfs.next_bytes(32);
        println!("n: {}", hex::encode(bytes));
        let squeezed = evmfs.squeeze_scalars(1);
        println!("sq: {}", squeezed[0]);

        let bytes = evmfs.next_bytes(32);
        println!("n: {}", hex::encode(bytes));
        let squeezed = evmfs.squeeze_bytes(2);
        println!("sq: {}", hex::encode(squeezed[0]));
        println!("sq: {}", hex::encode(squeezed[1]));
    }
}
