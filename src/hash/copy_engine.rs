use std::borrow::Cow;

use const_oid::ObjectIdentifier;
use hex_literal::hex;

use super::{Engine, Hash};
use crate::transcript::ProtocolId;

pub const COPY: ProtocolId = ProtocolId::new(hex!(
    "f6e915700df8e5d547ec84b28183c1f74a8cc351adb6e9c4dae727782554ed7c"
));

/// No-op hash engine that copies the input data without hashing it.
///
/// Requires the input data to be at most 32 bytes long.
#[derive(Clone, Copy, Debug, Default)]
pub struct Copy;

impl Copy {
    pub const fn new() -> Self {
        Self
    }
}

impl Engine for Copy {
    fn name(&self) -> Cow<'_, str> {
        "copy".into()
    }

    fn oid(&self) -> Option<ObjectIdentifier> {
        None
    }

    fn supports_size(&self, size: usize) -> bool {
        size <= 32
    }

    fn preferred_batch_size(&self) -> usize {
        1
    }

    fn hash_many(&self, size: usize, input: &[u8], output: &mut [Hash]) {
        assert!(size <= 32);
        assert_eq!(
            input.len(),
            size * output.len(),
            "Input length should be size * output.len() = {size} * {}",
            output.len()
        );

        for (input, out) in input.chunks_exact(size).zip(output.iter_mut()) {
            let mut bytes = [0; 32];
            bytes[..size].copy_from_slice(input);
            *out = Hash(bytes);
        }
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::transcript::Protocol;

    #[test]
    fn test_protocol_ids() {
        assert_eq!(Copy::new().protocol_id(), COPY);
    }
}
