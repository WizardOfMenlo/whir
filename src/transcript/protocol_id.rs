//! Tools around Protocol Identifiers, used for unique identification
//! of cryptographic protocols and domain separation.

use std::fmt::{Debug, Display};

use serde::{Deserialize, Serialize};

pub const NONE: ProtocolId = ProtocolId([0u8; 32]);

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize)]
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
        Self(bytes)
    }
}

impl ProtocolId {
    pub const fn new(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    pub const fn as_slice(&self) -> &[u8] {
        &self.0
    }
}

impl Display for ProtocolId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if f.alternate() {
            for byte in &self.0 {
                write!(f, "{byte:02x}")?;
            }
        } else {
            for byte in &self.0[0..6] {
                write!(f, "{byte:02x}")?;
            }
            write!(f, "…")?;
            for byte in &self.0[26..32] {
                write!(f, "{byte:02x}")?;
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
