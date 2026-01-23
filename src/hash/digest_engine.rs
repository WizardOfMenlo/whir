use std::{borrow::Cow, marker::PhantomData};

use const_oid::ObjectIdentifier;
use digest::{const_oid::AssociatedOid, consts::U32, Digest};
use hex_literal::hex;
use zerocopy::IntoBytes;

use super::{Engine, Hash, HASH_COUNTER};
use crate::transcript::ProtocolId;

pub const SHA2: ProtocolId = ProtocolId::new(hex!(
    "49051031d2991bfe411ad7106a2c0091f8a7418946e504319c61dc930318466e"
));
pub const SHA3: ProtocolId = ProtocolId::new(hex!(
    "cfb9d8382a29dea8aa73d88d45c0b6fca9661b420b83fe93c327b07292f3c132"
));
pub const KECCAK: ProtocolId = ProtocolId::new(hex!(
    "5cd80d7df185bbef439921edc0d97c18170939b22f08a958d364e3f9bc01236c"
));

pub type Sha2 = DigestEngine<sha2::Sha256>;
pub type Sha3 = DigestEngine<sha3::Sha3_256>;
pub type Keccak = DigestEngine<sha3::Keccak256>;

/// Wrapper around a `Digest` to implement the `Hasher` trait.
///
/// This allows using any hash in the Rust-Crypto ecosystem,
/// though performance may vary.
#[derive(Clone, Copy, Debug)]
pub struct DigestEngine<D: Digest> {
    name: &'static str,
    oid: Option<ObjectIdentifier>,
    _digest: PhantomData<D>,
}

impl<D> DigestEngine<D>
where
    D: Digest<OutputSize = U32> + Send + Sync,
{
    pub fn from_name(name: &'static str) -> Self {
        assert_eq!(<D as Digest>::output_size(), size_of::<Hash>());
        Self {
            name,
            oid: None,
            _digest: PhantomData,
        }
    }

    pub fn from_name_oid(name: &'static str, oid: ObjectIdentifier) -> Self {
        assert_eq!(<D as Digest>::output_size(), size_of::<Hash>());
        Self {
            name,
            oid: Some(oid),
            _digest: PhantomData,
        }
    }

    pub fn from_name_assoc_oid(name: &'static str) -> Self
    where
        D: AssociatedOid,
    {
        Self::from_name_oid(name, D::OID)
    }
}

impl Default for Sha2 {
    fn default() -> Self {
        Self::new()
    }
}

impl Sha2 {
    pub fn new() -> Self {
        Self::from_name_assoc_oid("sha2")
    }
}

impl Default for Sha3 {
    fn default() -> Self {
        Self::new()
    }
}

impl Sha3 {
    pub fn new() -> Self {
        Self::from_name_assoc_oid("sha3")
    }
}

impl Default for Keccak {
    fn default() -> Self {
        Self::new()
    }
}

impl Keccak {
    pub fn new() -> Self {
        Self::from_name("keccak")
    }
}

impl<D> Engine for DigestEngine<D>
where
    D: Digest<OutputSize = U32> + Send + Sync,
{
    fn name(&self) -> Cow<'_, str> {
        self.name.into()
    }

    fn oid(&self) -> Option<ObjectIdentifier> {
        self.oid
    }

    fn supports_size(&self, _size: usize) -> bool {
        true
    }

    fn preferred_batch_size(&self) -> usize {
        1
    }

    fn hash_many(&self, size: usize, input: &[u8], output: &mut [Hash]) {
        assert_eq!(
            input.len(),
            size * output.len(),
            "Input length ({}) should be size * output.len() = {size} * {}",
            input.len(),
            output.len()
        );

        if size == 0 {
            output.fill(Hash(D::digest([]).into()));
        } else {
            for (input, out) in input.chunks_exact(size).zip(output.iter_mut()) {
                let input = input.as_bytes();
                let hash = D::digest(input);
                out.as_mut_bytes().copy_from_slice(hash.as_ref());
            }
            HASH_COUNTER.add(input.len() / size);
        }
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::transcript::Protocol;

    #[test]
    fn test_protocol_ids() {
        assert_eq!(Sha2::new().protocol_id(), SHA2);
        assert_eq!(Sha3::new().protocol_id(), SHA3);
        assert_eq!(Keccak::new().protocol_id(), KECCAK);
    }
}
