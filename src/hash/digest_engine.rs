use std::{borrow::Cow, marker::PhantomData};

use const_oid::ObjectIdentifier;
use digest::{const_oid::AssociatedOid, Digest};
use hex_literal::hex;
use zerocopy::IntoBytes;

use super::{Engine, Hash};
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
pub struct DigestEngine<D: Digest> {
    name: &'static str,
    oid: Option<ObjectIdentifier>,
    _digest: PhantomData<D>,
}

impl<D> DigestEngine<D>
where
    D: Digest + Send + Sync,
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

impl Sha2 {
    pub fn new() -> Self {
        Self::from_name_assoc_oid("sha2")
    }
}

impl Sha3 {
    pub fn new() -> Self {
        Self::from_name_assoc_oid("sha3")
    }
}

impl Keccak {
    pub fn new() -> Self {
        Self::from_name("keccak")
    }
}

impl<D> Engine for DigestEngine<D>
where
    D: Digest + Send + Sync,
{
    fn name<'a>(&'a self) -> Cow<'a, str> {
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

    fn hash_many(&self, size: usize, input: &[u8], out: &mut [Hash]) {
        assert!(
            input.len().is_multiple_of(size),
            "Input size must be a multiple of the message size."
        );
        if let Some(num_inputs) = input.len().checked_div(size) {
            assert_eq!(num_inputs, out.len(), "Output size mismatch.");
        } else {
            let empty_hash = D::digest(input);
            for out in out.iter_mut() {
                out.as_mut_bytes().copy_from_slice(empty_hash.as_ref());
            }
            return;
        }
        for (input, out) in input.chunks_exact(size).zip(out.iter_mut()) {
            let input = input.as_bytes();
            let hash = D::digest(input);
            out.as_mut_bytes().copy_from_slice(hash.as_ref());
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
