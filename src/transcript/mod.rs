//! Wrapper around Spongefish to add out-of-band hint messages.
//!
//! We need these for the Merkle tree proofs as doing them in-transcript
//! would roughly double the verifier cost.

pub mod codecs;
mod mock_sponge;

#[cfg(debug_assertions)]
use std::any::type_name;
use std::fmt::Debug;

use ark_ff::{Field, PrimeField};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::rand::{rngs::StdRng, CryptoRng, RngCore};
use serde::{Deserialize, Serialize};
use sha3::{Digest, Sha3_256, Sha3_512};
use spongefish::StdHash;
pub use spongefish::{
    Codec, Decoding, DuplexSpongeInterface, Encoding, NargDeserialize, NargSerialize,
    VerificationError, VerificationResult,
};

/// Zero-allocation wrapper for pre-encoded bytes.
///
/// Used by [`ProverState::prover_messages_bytes`] to send a pre-serialized
/// byte buffer as a single transcript message, avoiding the per-element
/// allocation overhead of [`Encoding::encode`] on individual field elements.
pub struct RawBytes<'a>(pub &'a [u8]);

impl Encoding<[u8]> for RawBytes<'_> {
    fn encode(&self) -> impl AsRef<[u8]> {
        self.0
    }
}
// NargSerialize is provided by the blanket impl:
//   impl<T: Encoding<[u8]>> NargSerialize for T

/// Encode a single field element into `dst` without heap allocations.
///
/// Produces the same byte representation as [`Encoding<[u8]>::encode`] on
/// ark-ff field elements: each base-prime-field coefficient is written in
/// little-endian limb order, truncated/padded to `base_field_size` bytes.
#[inline]
pub fn encode_field_element_into<F: Field>(f: &F, dst: &mut Vec<u8>) {
    let base_field_size = (F::BasePrimeField::MODULUS_BIT_SIZE.div_ceil(8)) as usize;
    for base_element in f.to_base_prime_field_elements() {
        let bigint = base_element.into_bigint();
        let limbs: &[u64] = bigint.as_ref();
        let start = dst.len();
        for limb in limbs {
            dst.extend_from_slice(&limb.to_le_bytes());
        }
        // Match spongefish's encode: resize to exactly base_field_size bytes
        // (truncate high zero bytes if N*8 > base_field_size, pad if less).
        dst.resize(start + base_field_size, 0);
    }
}

/// Decode field elements from a byte buffer produced by [`encode_field_element_into`].
///
/// Returns `count` field elements, advancing `src` past the consumed bytes.
pub fn decode_field_elements_from_bytes<F: Field>(src: &mut &[u8], count: usize) -> Option<Vec<F>> {
    let base_field_size = (F::BasePrimeField::MODULUS_BIT_SIZE.div_ceil(8)) as usize;
    let ext_degree = F::extension_degree() as usize;
    let elem_bytes = base_field_size * ext_degree;
    let total = count * elem_bytes;
    if src.len() < total {
        return None;
    }

    let mut result = Vec::with_capacity(count);
    for _ in 0..count {
        let mut base_elems = Vec::with_capacity(ext_degree);
        for _ in 0..ext_degree {
            let (chunk, rest) = src.split_at(base_field_size);
            *src = rest;
            base_elems.push(F::BasePrimeField::from_le_bytes_mod_order(chunk));
        }
        result.push(F::from_base_prime_field_elems(base_elems)?);
    }
    Some(result)
}

#[cfg(test)]
pub use self::mock_sponge::MockSponge;

#[macro_export]
macro_rules! verify {
    ($cond:expr) => {
        #[allow(clippy::neg_cmp_op_on_partial_ord)]
        if !$cond {
            #[cfg(feature = "verifier_panics")]
            panic!("Verification failed: {}", stringify!($cond));

            #[cfg(not(feature = "verifier_panics"))]
            return Err(spongefish::VerificationError);
        };
    };
}

/// Marker trait for types that can be used as prover messages.
///
/// Like [`spongefish::Codec`], but without the [`Encoding<T>`] requirement.
pub trait ProverMessage<U = [u8]>: NargDeserialize + NargSerialize + Encoding<U>
where
    U: ?Sized,
{
}

#[derive(Clone, Copy, Debug)]
pub struct DomainSeparator<'a, I> {
    protocol_id: [u8; 64],
    session_id: [u8; 32],
    instance: &'a I,
}

#[derive(Clone, PartialEq, Eq, Serialize, Deserialize, Debug)]
pub enum Interaction {
    ProverMessage(String),
    VerifierMessage(String),
    Hint(String),
}

#[derive(Clone, PartialEq, Eq, Serialize, Deserialize, Debug)]
pub struct Proof {
    pub narg_string: Vec<u8>,
    pub hints: Vec<u8>,

    #[cfg(debug_assertions)]
    pub pattern: Vec<Interaction>,
}

pub struct ProverState<H = StdHash, R = StdRng>
where
    H: DuplexSpongeInterface,
    R: RngCore + CryptoRng,
{
    inner: spongefish::ProverState<H, R>,
    hints: Vec<u8>,

    #[cfg(debug_assertions)]
    pattern: Vec<Interaction>,
}

pub struct VerifierState<'a, H = StdHash>
where
    H: DuplexSpongeInterface,
{
    inner: spongefish::VerifierState<'a, H>,
    hints: &'a [u8],

    #[cfg(debug_assertions)]
    pattern: &'a [Interaction],
}

pub trait VerifierMessage {
    type U;

    fn verifier_message<T>(&mut self) -> T
    where
        T: Decoding<[Self::U]>;

    fn verifier_message_vec<T>(&mut self, count: usize) -> Vec<T>
    where
        T: Decoding<[Self::U]>,
    {
        (0..count).map(|_| self.verifier_message()).collect()
    }
}

impl DomainSeparator<'static, ()> {
    pub fn protocol<C: Serialize>(config: &C) -> Self {
        const INSTANCE: &() = &();
        let mut hash = Sha3_512::new();
        ciborium::into_writer(config, &mut hash).expect("Computing protocol hash failed");
        let protocol_id: [u8; 64] = hash.finalize().into();
        Self {
            protocol_id,
            session_id: [0; 32],
            instance: INSTANCE,
        }
    }

    #[must_use]
    pub fn session<S: Serialize>(self, session: &S) -> Self {
        let mut hash = Sha3_256::new();
        ciborium::into_writer(session, &mut hash).expect("Computing session hash failed");
        let session_id: [u8; 32] = hash.finalize().into();
        Self { session_id, ..self }
    }

    pub const fn instance<I>(self, instance: &I) -> DomainSeparator<'_, I> {
        DomainSeparator {
            protocol_id: self.protocol_id,
            session_id: self.session_id,
            instance,
        }
    }
}

impl<T, U> ProverMessage<U> for T
where
    T: NargSerialize + NargDeserialize + Encoding<U>,
    U: ?Sized,
{
}

impl<H> ProverState<H, StdRng>
where
    H: DuplexSpongeInterface,
{
    /// Construct a new prover state with a custom duplex hash function.
    ///
    /// **Note.** The `spongefish` API currently does not allow creating an
    /// instance with a non-standard random number generator.
    pub fn new<I>(ds: &DomainSeparator<'_, I>, duplex: H) -> Self
    where
        u8: Encoding<[H::U]>,
        I: Encoding<[H::U]>,
    {
        Self {
            inner: spongefish::DomainSeparator::new(ds.protocol_id)
                .session(ds.session_id)
                .instance(ds.instance)
                .to_prover(duplex),
            hints: Vec::new(),

            #[cfg(debug_assertions)]
            pattern: Vec::new(),
        }
    }
}

impl ProverState<StdHash, StdRng> {
    /// Construct a new prover state with the standard duplex hash function.
    pub fn new_std<I>(ds: &DomainSeparator<'_, I>) -> Self
    where
        I: Encoding<[u8]>,
    {
        Self::new(ds, StdHash::default())
    }
}

impl<H, R> ProverState<H, R>
where
    H: DuplexSpongeInterface,
    R: RngCore + CryptoRng,
{
    #[cfg_attr(test, track_caller)]
    pub fn prover_message<T>(&mut self, message: &T)
    where
        T: Encoding<[H::U]> + NargSerialize + ?Sized,
    {
        #[cfg(debug_assertions)]
        self.push(Interaction::ProverMessage(type_name::<T>().to_owned()));
        self.inner.prover_message(message);
    }

    #[cfg_attr(test, track_caller)]
    pub fn prover_hint<T>(&mut self, hint: &T)
    where
        T: NargSerialize,
    {
        #[cfg(debug_assertions)]
        self.push(Interaction::Hint(type_name::<T>().to_owned()));
        hint.serialize_into_narg(&mut self.hints);
    }

    #[cfg_attr(test, track_caller)]
    pub fn prover_hint_ark<T>(&mut self, value: &T)
    where
        T: CanonicalSerialize + ?Sized,
    {
        #[cfg(debug_assertions)]
        self.push(Interaction::Hint(type_name::<T>().to_owned()));
        value
            .serialize_compressed(&mut self.hints)
            .expect("Failed to serialize hint");
    }

    /// Send `count` pre-encoded field elements as a **single** transcript message.
    ///
    /// `encoded` must contain the exact same bytes that `count` individual
    /// `prover_message::<T>()` calls would produce (use [`encode_field_element_into`]).
    ///
    /// This reduces allocations from O(count) to O(1) because the sponge
    /// absorbs the whole buffer at once via [`RawBytes`] (zero-alloc Encoding).
    ///
    /// Requires a byte-oriented sponge (`H::U = u8`).
    #[cfg_attr(test, track_caller)]
    pub fn prover_messages_bytes<T>(&mut self, _count: usize, encoded: &[u8])
    where
        H: DuplexSpongeInterface<U = u8>,
    {
        #[cfg(debug_assertions)]
        for _ in 0.._count {
            self.push(Interaction::ProverMessage(type_name::<T>().to_owned()));
        }
        self.inner.prover_message(&RawBytes(encoded));
    }

    /// Access the prover's private RNG (forwarded from the inner spongefish state).
    pub fn rng(&mut self) -> &mut (impl RngCore + CryptoRng) {
        self.inner.rng()
    }

    pub fn proof(self) -> Proof {
        Proof {
            narg_string: self.inner.narg_string().to_owned(),
            hints: self.hints,

            #[cfg(debug_assertions)]
            pattern: self.pattern,
        }
    }

    #[cfg(debug_assertions)]
    fn push(&mut self, interaction: Interaction) {
        self.pattern.push(interaction);
    }
}

impl<H, R> VerifierMessage for ProverState<H, R>
where
    H: DuplexSpongeInterface,
    R: RngCore + CryptoRng,
{
    type U = H::U;

    #[cfg_attr(test, track_caller)]
    fn verifier_message<T>(&mut self) -> T
    where
        T: Decoding<[H::U]>,
    {
        #[cfg(debug_assertions)]
        self.push(Interaction::VerifierMessage(type_name::<T>().to_owned()));
        self.inner.verifier_message()
    }
}

impl<'a, H> VerifierState<'a, H>
where
    H: DuplexSpongeInterface,
{
    pub fn new<'b, I>(ds: &DomainSeparator<'b, I>, proof: &'a Proof, duplex: H) -> Self
    where
        u8: Encoding<[H::U]>,
        I: Encoding<[H::U]>,
    {
        Self {
            inner: spongefish::DomainSeparator::new(ds.protocol_id)
                .session(ds.session_id)
                .instance(ds.instance)
                .to_verifier(duplex, &proof.narg_string),
            hints: &proof.hints,
            #[cfg(debug_assertions)]
            pattern: &proof.pattern,
        }
    }

    pub const fn as_spongefish(&mut self) -> &mut spongefish::VerifierState<'a, H> {
        &mut self.inner
    }

    #[cfg_attr(debug_assertions, track_caller)]
    pub fn check_eof(self) -> VerificationResult<()> {
        #[cfg(debug_assertions)]
        assert!(self.pattern.is_empty());
        verify!(self.inner.check_eof().is_ok());
        verify!(self.hints.is_empty());
        Ok(())
    }

    #[cfg_attr(test, track_caller)]
    pub fn prover_message<T>(&mut self) -> VerificationResult<T>
    where
        T: Encoding<[H::U]> + NargDeserialize,
    {
        #[cfg(debug_assertions)]
        self.pop_pattern(&Interaction::ProverMessage(type_name::<T>().to_owned()));
        self.inner.prover_message()
    }

    #[cfg_attr(test, track_caller)]
    pub fn prover_messages_vec<T>(&mut self, len: usize) -> VerificationResult<Vec<T>>
    where
        T: Encoding<[H::U]> + NargDeserialize,
    {
        (0..len).map(|_| self.prover_message()).collect()
    }

    /// Read `count` field elements that were sent via
    /// [`ProverState::prover_messages_bytes`].
    ///
    /// Internally this still calls `prover_message::<T>()` per element
    /// (spongefish doesn't expose a raw-byte batch read), but it collects
    /// into a single pre-allocated `Vec` rather than creating many small
    /// intermediate vectors.
    ///
    /// The Fiat-Shamir transcript is byte-identical regardless of whether
    /// the prover used individual `prover_message` calls or a single
    /// `prover_messages_bytes` batch — the sponge absorbs the same bytes
    /// in both cases.
    #[cfg_attr(test, track_caller)]
    pub fn read_prover_messages_bytes<T>(&mut self, count: usize) -> VerificationResult<Vec<T>>
    where
        T: Encoding<[H::U]> + NargDeserialize,
    {
        #[cfg(debug_assertions)]
        for _ in 0..count {
            self.pop_pattern(&Interaction::ProverMessage(type_name::<T>().to_owned()));
        }

        let mut result = Vec::with_capacity(count);
        for _ in 0..count {
            result.push(self.inner.prover_message()?);
        }
        Ok(result)
    }

    #[cfg_attr(test, track_caller)]
    pub fn prover_hint<T>(&mut self) -> VerificationResult<T>
    where
        T: NargDeserialize,
    {
        #[cfg(debug_assertions)]
        self.pop_pattern(&Interaction::Hint(type_name::<T>().to_owned()));
        T::deserialize_from_narg(&mut self.hints)
    }

    #[cfg_attr(test, track_caller)]
    pub fn prover_hint_ark<T>(&mut self) -> VerificationResult<T>
    where
        T: CanonicalDeserialize,
    {
        #[cfg(debug_assertions)]
        self.pop_pattern(&Interaction::Hint(type_name::<T>().to_owned()));
        T::deserialize_compressed(&mut self.hints).map_err(|_| VerificationError)
    }

    #[cfg(debug_assertions)]
    #[track_caller]
    fn pop_pattern(&mut self, interaction: &Interaction) {
        assert!(!self.pattern.is_empty());
        let (expected, tail) = self.pattern.split_first().unwrap();
        assert_eq!(
            interaction, expected,
            "Transcript error: Expected interaction {expected:?} got {interaction:?}"
        );
        self.pattern = tail;
    }
}

impl<'a> VerifierState<'a, StdHash> {
    /// Construct a new prover state with the standard duplex hash function.
    pub fn new_std<'b, I>(ds: &DomainSeparator<'b, I>, proof: &'a Proof) -> Self
    where
        I: Encoding<[u8]>,
    {
        Self::new(ds, proof, StdHash::default())
    }
}

impl<H> VerifierMessage for VerifierState<'_, H>
where
    H: DuplexSpongeInterface,
{
    type U = H::U;

    #[cfg_attr(test, track_caller)]
    fn verifier_message<T>(&mut self) -> T
    where
        T: Decoding<[H::U]>,
    {
        #[cfg(debug_assertions)]
        self.pop_pattern(&Interaction::VerifierMessage(type_name::<T>().to_owned()));
        self.inner.verifier_message()
    }
}
