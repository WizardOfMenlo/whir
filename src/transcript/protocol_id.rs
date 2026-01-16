//! Tools around Protocol Identifiers, used for unique identification
//! of cryptographic protocols and domain separation.

use std::fmt::{Debug, Display};

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ProtocolId([u8; 32]);

pub trait Protocol {
    fn protocol_id(&self) -> ProtocolId;
}

impl From<ProtocolId> for [u8; 32] {
    fn from(id: ProtocolId) -> Self {
        id.0
    }
}

impl From<[u8; 32]> for ProtocolId {
    fn from(bytes: [u8; 32]) -> Self {
        ProtocolId(bytes)
    }
}

impl ProtocolId {
    pub const fn new(bytes: [u8; 32]) -> Self {
        ProtocolId(bytes)
    }

    pub fn as_slice(&self) -> &[u8] {
        &self.0
    }
}

impl Display for ProtocolId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if f.alternate() {
            for bytes in &self.0 {
                write!(f, "{:02x}", bytes)?;
            }
        } else {
            for bytes in &self.0[0..6] {
                write!(f, "{:02x}", bytes)?;
            }
            write!(f, "…")?;
            for bytes in &self.0[26..32] {
                write!(f, "{:02x}", bytes)?;
            }
        }
        Ok(())
    }
}

impl Debug for ProtocolId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:#}")
    }
}
