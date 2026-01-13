//! Compatibility module for ark_std::rand

use rand::{CryptoRng, RngCore};

/// Wrapper around rand v0.9 to add v0.8 compatibility.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ArkRand<R: RngCore>(pub R);

impl<R: RngCore> rand::RngCore for ArkRand<R> {
    fn next_u32(&mut self) -> u32 {
        self.0.next_u32()
    }

    fn next_u64(&mut self) -> u64 {
        self.0.next_u64()
    }

    fn fill_bytes(&mut self, dst: &mut [u8]) {
        self.0.fill_bytes(dst);
    }
}

impl<R: RngCore> ark_std::rand::RngCore for ArkRand<R> {
    fn next_u32(&mut self) -> u32 {
        self.0.next_u32()
    }

    fn next_u64(&mut self) -> u64 {
        self.0.next_u64()
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        self.0.fill_bytes(dest);
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), ark_std::rand::Error> {
        Ok(self.0.fill_bytes(dest))
    }
}

impl<R: RngCore + CryptoRng> rand::CryptoRng for ArkRand<R> {}

impl<R: RngCore + CryptoRng> ark_std::rand::CryptoRng for ArkRand<R> {}
