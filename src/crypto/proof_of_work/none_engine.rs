use ark_ff::Zero;
use hex_literal::hex;
use sha3::{Digest, Sha3_256};

use super::Engine;
use crate::transcript::{Protocol, ProtocolId};

pub const NONE: ProtocolId = ProtocolId::new(hex!(
    "c2f551bf887c9a5db24d1da57246e5826ce7ea8bd20e072f5ec12f4616c04a50"
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
            "None-engine only supports zero difficulty"
        );
        nonce == 0
    }

    fn solve(&self, _challenge: [u8; 32], difficulty: f64) -> Option<u64> {
        assert!(
            difficulty.is_zero(),
            "None-engine only supports zero difficulty"
        );
        Some(0)
    }
}

impl Protocol for NoneEngine {
    fn protocol_id(&self) -> ProtocolId {
        let mut hasher = Sha3_256::new();
        hasher.update(b"whir::protocols::proof_of_work::NoneEngine");
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
