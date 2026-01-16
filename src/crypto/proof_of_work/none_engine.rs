use ark_ff::Zero;
use hex_literal::hex;
use sha3::{Digest, Sha3_256};

use super::Engine;
use crate::transcript::{Protocol, ProtocolId};

pub const NONE: ProtocolId = ProtocolId::new(hex!(
    "3a777695dad25e7b8a5e69f2ecf25e6a0f8927fd334377d297644b14a761dd3d"
));

#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub struct NoneEngine;

impl NoneEngine {
    pub const fn new() -> Self {
        Self
    }
}

impl Engine for NoneEngine {
    fn check(&self, _challenge: [u8; 32], difficulty: f64, nonce: u64) -> bool {
        assert!(
            difficulty.is_zero(),
            "Null engine only supports zero difficulty"
        );
        nonce == 0
    }

    fn solve(&self, _challenge: [u8; 32], difficulty: f64) -> Option<u64> {
        assert!(
            difficulty.is_zero(),
            "Null engine only supports zero difficulty"
        );
        Some(0)
    }
}

impl Protocol for NoneEngine {
    fn protocol_id(&self) -> ProtocolId {
        let mut hasher = Sha3_256::new();
        hasher.update(b"whir::protocols::proof_of_work::NullEngine");
        let hash: [u8; 32] = hasher.finalize().into();
        hash.into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn protocol_id() {
        assert_eq!(NoneEngine.protocol_id(), NONE);
    }
}
