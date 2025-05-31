//! Implementation of the proof of work engine for any RustCrypto digest.

use std::{
    any::type_name,
    fmt::{Debug, Display},
    marker::PhantomData,
};

use digest::Digest;
use zerocopy::transmute;

use super::{
    utils::{f64_to_u256, less_than},
    Engine,
};

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct DigestEngine<D>
where
    D: Digest + Sync + Send,
{
    _digest: PhantomData<D>,
}

impl<D: Digest + Sync + Send> DigestEngine<D> {
    pub fn new() -> Self {
        assert!(
            <D as Digest>::output_size() >= 32,
            "Digest must produce at least 32-byte output"
        );
        Self {
            _digest: PhantomData,
        }
    }
}

impl<D> Engine for DigestEngine<D>
where
    D: Digest + Sync + Send,
{
    // OPT: Efficient implementation of solve.

    fn verify(&self, challenge: [u8; 32], difficulty: f64, nonce: u64) -> bool {
        let threshold = 2_f64.powf(256.0 - difficulty);
        let threshold = f64_to_u256(threshold);

        let mut input = [0_u8; 64];
        input[..32].copy_from_slice(&challenge);
        input[32..40].copy_from_slice(&nonce.to_le_bytes());

        let hash = D::digest(&input);
        let mut output = [0_u8; 32];
        output.copy_from_slice(&hash[..32]);
        less_than(transmute!(output), transmute!(threshold))
    }
}

impl<D> Debug for DigestEngine<D>
where
    D: Digest + Sync + Send,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DigestEngine")
            .field("digest_type", &type_name::<D>())
            .finish()
    }
}

impl<D> Display for DigestEngine<D>
where
    D: Digest + Sync + Send,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "DigestEngine<{:?}>", type_name::<D>())
    }
}

#[cfg(test)]
mod tests {
    use sha3::Keccak256;

    use super::*;

    #[test]
    fn solves_keccak() {
        let engine = DigestEngine::<Keccak256>::new();
        let challenge = [0u8; 32];
        let difficulty = 5.0;
        let nonce = engine.solve(challenge, difficulty);
        assert_eq!(nonce, 24);
        assert!(engine.verify(challenge, difficulty, nonce));
    }
}
