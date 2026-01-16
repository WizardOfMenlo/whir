//! Registery for Engines based on protocol id.
//!
//! This allows implementations to be agnostic to the underlying engine implementation.

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use super::{Protocol, ProtocolId};

#[derive(Debug, Default)]
pub struct Engines<T: ?Sized>(Mutex<HashMap<ProtocolId, Arc<T>>>);

impl<T: Protocol + ?Sized> Engines<T> {
    pub fn new() -> Self {
        Self(Mutex::new(HashMap::new()))
    }

    pub fn register(&self, engine: Arc<T>) {
        self.0.lock().unwrap().insert(engine.protocol_id(), engine);
    }

    pub fn contains(&self, protocol_id: ProtocolId) -> bool {
        self.0.lock().unwrap().contains_key(&protocol_id)
    }

    pub fn retrieve(&self, protocol_id: ProtocolId) -> Option<Arc<T>> {
        self.0.lock().unwrap().get(&protocol_id).cloned()
    }
}
