use std::hash::Hash;

use serde::Serialize;

/// Wrapper for `bits` value types.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Bits(f64);

impl Hash for Bits {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.to_bits().hash(state);
    }
}

impl Eq for Bits {}

impl Bits {
    pub fn new(bits: f64) -> Self {
        assert!(bits.is_finite());
        Bits(bits)
    }

    pub fn is_zero(&self) -> bool {
        self.0 == 0.0
    }
}

impl From<f64> for Bits {
    fn from(bits: f64) -> Self {
        Bits::new(bits)
    }
}

impl From<Bits> for f64 {
    fn from(bits: Bits) -> Self {
        bits.0
    }
}
