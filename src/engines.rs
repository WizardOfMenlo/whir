//! Registery for Engines based on protocol id.
//!
//! This allows implementations to be agnostic to the underlying engine implementation.

//! Tools around Protocol Identifiers, used for unique identification
//! of cryptographic protocols and domain separation.

use std::{
    collections::HashMap,
    fmt::{Debug, Display},
    sync::{Arc, Mutex},
};

use serde::{Deserialize, Serialize};

pub const NONE: EngineId = EngineId([0u8; 32]);

pub trait Engine {
    fn engine_id(&self) -> EngineId;
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize)]
pub struct EngineId([u8; 32]);

#[derive(Debug, Default)]
pub struct Engines<T: ?Sized>(Mutex<HashMap<EngineId, Arc<T>>>);

impl<T: Engine + ?Sized> Engines<T> {
    pub fn new() -> Self {
        Self(Mutex::new(HashMap::new()))
    }

    pub fn register(&self, engine: Arc<T>) {
        self.0.lock().unwrap().insert(engine.engine_id(), engine);
    }

    pub fn contains(&self, protocol_id: EngineId) -> bool {
        self.0.lock().unwrap().contains_key(&protocol_id)
    }

    pub fn retrieve(&self, protocol_id: EngineId) -> Option<Arc<T>> {
        self.0.lock().unwrap().get(&protocol_id).cloned()
    }
}

impl From<EngineId> for [u8; 32] {
    fn from(id: EngineId) -> Self {
        id.0
    }
}

impl From<[u8; 32]> for EngineId {
    fn from(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }
}

impl EngineId {
    pub const fn new(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    pub const fn as_slice(&self) -> &[u8] {
        &self.0
    }
}

impl Display for EngineId {
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

impl Debug for EngineId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:#}")
    }
}
