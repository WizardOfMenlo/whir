//! Wrapper around Spongefish to add out-of-band hint messages.
//!
//! We need these for the Merkle tree proofs as doing them in-transcript
//! would roughly double the verifier cost.

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::rand::{rngs::StdRng, CryptoRng, RngCore};
use serde::{Deserialize, Serialize};
use spongefish::{
    Decoding, DomainSeparator, DuplexSpongeInterface, Encoding, NargDeserialize, NargSerialize,
    StdHash, VerificationError, VerificationResult,
};

pub trait ProtocolId {
    fn protocol_id(&self) -> [u8; 64];
}

#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Proof {
    pub narg_string: Vec<u8>,
    pub hints: Vec<u8>,
}

pub struct ProverState<H = StdHash, R = StdRng>
where
    H: DuplexSpongeInterface,
    R: RngCore + CryptoRng,
{
    inner: spongefish::ProverState<H, R>,
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

impl<H, R> From<spongefish::ProverState<H, R>> for ProverState<H, R>
where
    H: DuplexSpongeInterface,
    R: RngCore + CryptoRng,
{
    fn from(inner: spongefish::ProverState<H, R>) -> Self {
        Self {
            inner,
            hints: Vec::new(),
        }
    }
}

impl<H, R> ProverState<H, R>
where
    H: DuplexSpongeInterface,
    R: RngCore + CryptoRng,
{
    pub fn into_inner(self) -> (spongefish::ProverState<H, R>, Vec<u8>) {
        (self.inner, self.hints)
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

impl<H> ProverState<H, StdRng>
where
    H: DuplexSpongeInterface,
{
    pub fn new<S, I>(protocol: [u8; 64], session: S, instance: &I, sponge: H) -> Self
    where
        [u8; 64]: Encoding<[H::U]>,
        S: Encoding<[H::U]>,
        I: Encoding<[H::U]>,
    {
        DomainSeparator::new(protocol)
            .session(session)
            .instance(instance)
            .to_prover(sponge)
            .into()
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
    pub fn from(inner: spongefish::VerifierState<'a, H>, hints: &'a [u8]) -> Self {
        Self { inner, hints }
    }

    pub fn into_inner(self) -> (spongefish::VerifierState<'a, H>, &'a [u8]) {
        (self.inner, self.hints)
    }

    pub fn new<I, S>(
        protocol: [u8; 64],
        session: S,
        sponge: H,
        instance: &I,
        proof: &'a Proof,
    ) -> Self
    where
        [u8; 64]: Encoding<[H::U]>,
        S: Encoding<[H::U]>,
        I: Encoding<[H::U]>,
    {
        Self {
            inner: DomainSeparator::new(protocol)
                .session(session)
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
