mod blake3_engine;
mod digest_engine;

use core::fmt;
use std::{
    borrow::Cow,
    sync::{Arc, LazyLock},
};

use const_oid::ObjectIdentifier;
use serde::{Deserialize, Serialize};
use static_assertions::assert_obj_safe;
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout, Unaligned};

pub use self::{
    blake3_engine::{Blake3, BLAKE3},
    digest_engine::{DigestEngine, Keccak, Sha2, Sha3, KECCAK, SHA2, SHA3},
};
use crate::transcript::{Engines, Protocol, ProtocolId};

pub static ENGINES: LazyLock<Engines<dyn Engine>> = LazyLock::new(|| {
    let engines = Engines::<dyn Engine>::new();
    engines.register(Arc::new(Sha2::new()));
    engines.register(Arc::new(Sha3::new()));
    engines.register(Arc::new(Blake3::detect()));
    engines
});

#[derive(
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    Default,
    Serialize,
    Deserialize,
    KnownLayout,
    Immutable,
    Unaligned,
    FromBytes,
    IntoBytes,
)]
#[repr(transparent)]
pub struct Hash(pub [u8; 32]);

pub trait Engine: Send + Sync {
    fn name<'a>(&'a self) -> Cow<'a, str>;

    fn oid(&self) -> Option<ObjectIdentifier> {
        None
    }

    /// Check if the engine supports hashing messages of the given size.
    fn supports_size(&self, size: usize) -> bool;

    /// The number of messages that should be hashed together to achieve maximally
    /// utilize single thread parallelism.
    ///
    /// The caller should attermpt to call `hash_many` with multiples of this size.
    ///
    /// Regardless, all batch sizes must be supported.
    fn preferred_batch_size(&self) -> usize {
        1
    }

    /// Hash many messages of size `size`.
    ///
    /// Input contains the messages concatenated together.
    ///
    /// Note: Implementation should be single-threaded. Parallelization is taken
    /// care of by the caller.
    fn hash_many(&self, size: usize, input: &[u8], out: &mut [Hash]);
}

impl<E: Engine + ?Sized> Protocol for E {
    fn protocol_id(&self) -> ProtocolId {
        use digest::Digest;
        use sha3::Sha3_256;

        let mut hasher = Sha3_256::new();
        hasher.update(b"whir::crypto::hash");
        if let Some(oid) = self.oid() {
            hasher.update(oid.as_bytes());
        } else {
            hasher.update(self.name().as_bytes());
        }
        let hash: [u8; 32] = hasher.finalize().into();
        hash.into()
    }
}

assert_obj_safe!(Engine);

impl fmt::Debug for Hash {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for byte in self.0 {
            write!(f, "{:02x}", byte)?;
        }
        Ok(())
    }
}
