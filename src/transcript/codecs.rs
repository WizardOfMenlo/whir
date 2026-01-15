use spongefish::{Codec, Decoding, Encoding, NargDeserialize};
use static_assertions::assert_impl_all;

/// An empty object.
pub struct None;

impl<T> Encoding<[T]> for None {
    fn encode(&self) -> impl AsRef<[T]> {
        []
    }
}

impl<T> Decoding<[T]> for None {
    type Repr = [T; 0];

    fn decode(_buf: Self::Repr) -> Self {
        Self
    }
}

impl NargDeserialize for None {
    fn deserialize_from_narg(_buf: &mut &[u8]) -> spongefish::VerificationResult<Self> {
        Ok(Self)
    }
}

assert_impl_all!(None: Codec);

/// Wrapper because spongefish is missing NargDeserialize for `u64`.
pub struct U64(pub u64);

impl Encoding<[u8]> for U64 {
    fn encode(&self) -> impl AsRef<[u8]> {
        self.0.to_le_bytes()
    }
}

impl Decoding<[u8]> for U64 {
    type Repr = [u8; 8];

    fn decode(buf: Self::Repr) -> Self {
        U64(u64::from_le_bytes(buf))
    }
}

impl NargDeserialize for U64 {
    fn deserialize_from_narg(buf: &mut &[u8]) -> spongefish::VerificationResult<Self> {
        NargDeserialize::deserialize_from_narg(buf)
            .map(u64::from_le_bytes)
            .map(Self)
    }
}

assert_impl_all!(U64: Codec);
