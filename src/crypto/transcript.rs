//! Wrapper around Spongefish to add out-of-band hint messages.
//!
//! We need these for the Merkle tree proofs as doing them in-transcript
//! would roughly double the verifier cost.

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use serde::{Deserialize, Serialize};
use spongefish::{
    Decoding, DomainSeparator, DuplexSpongeInterface, Encoding, NargDeserialize, NargSerialize,
    StdHash, VerificationError, VerificationResult,
};

#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Proof {
    narg_string: Vec<u8>,
    hints: Vec<u8>,
}

pub struct ProverState<H = StdHash>
where
    H: DuplexSpongeInterface,
{
    inner: spongefish::ProverState<H>,
    hints: Vec<u8>,
}

pub struct VerifierState<'a, H = StdHash>
where
    H: DuplexSpongeInterface,
{
    inner: spongefish::VerifierState<'a, H>,
    hints: &'a [u8],
}

pub trait VerifierMessage {
    type U;

    fn verifier_message<T>(&mut self) -> T
    where
        T: Decoding<[Self::U]>;
}

impl<H> ProverState<H>
where
    H: DuplexSpongeInterface,
{
    pub fn new<I>(protocol: [u8; 64], sponge: H, instance: &I) -> Self
    where
        I: Encoding<[H::U]>,
        [u8; 64]: Encoding<[H::U]>,
    {
        let ds: DomainSeparator<_> = DomainSeparator::new(protocol);
        Self {
            inner: ds.instance(instance).to_prover(sponge),
            hints: Vec::new(),
        }
    }

    pub fn prover_message<T>(&mut self, message: &T)
    where
        T: Encoding<[H::U]> + NargSerialize + ?Sized,
    {
        self.inner.prover_message(message);
    }

    pub fn prover_hint_ark<T>(&mut self, value: &T)
    where
        T: CanonicalSerialize + ?Sized,
    {
        value
            .serialize_compressed(&mut self.hints)
            .expect("Failed to serialize hint");
    }

    pub fn proof(self) -> Proof {
        Proof {
            narg_string: self.inner.narg_string().to_owned(),
            hints: self.hints,
        }
    }
}

impl<H> VerifierMessage for ProverState<H>
where
    H: DuplexSpongeInterface,
{
    type U = H::U;

    fn verifier_message<T>(&mut self) -> T
    where
        T: Decoding<[H::U]>,
    {
        self.inner.verifier_message()
    }
}

impl<'a, H> VerifierState<'a, H>
where
    H: DuplexSpongeInterface,
{
    pub fn new<I>(protocol: [u8; 64], sponge: H, instance: &I, proof: &'a Proof) -> Self
    where
        I: Encoding<[H::U]>,
        [u8; 64]: Encoding<[H::U]>,
    {
        let ds: DomainSeparator<_> = DomainSeparator::new(protocol);
        Self {
            inner: ds
                .instance(instance)
                .to_verifier(sponge, &proof.narg_string),
            hints: &proof.hints,
        }
    }

    pub fn prover_message<T>(&mut self) -> VerificationResult<T>
    where
        T: Encoding<[H::U]> + NargDeserialize,
    {
        self.inner.prover_message()
    }

    pub fn prover_messages_vec<T>(&mut self, len: usize) -> VerificationResult<Vec<T>>
    where
        T: Encoding<[H::U]> + NargDeserialize,
    {
        self.inner.prover_messages_vec(len)
    }

    pub fn prover_hint_ark<T>(&mut self) -> VerificationResult<T>
    where
        T: CanonicalDeserialize,
    {
        T::deserialize_compressed(&mut self.hints).map_err(|_| VerificationError)
    }
}

impl<'a, H> VerifierMessage for VerifierState<'a, H>
where
    H: DuplexSpongeInterface,
{
    type U = H::U;

    fn verifier_message<T>(&mut self) -> T
    where
        T: Decoding<[H::U]>,
    {
        self.inner.verifier_message()
    }
}
