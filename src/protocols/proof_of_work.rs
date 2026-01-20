use core::slice;

use ark_std::rand::{CryptoRng, RngCore};
use serde::{Deserialize, Serialize};
use spongefish::{
    Codec, Decoding, DuplexSpongeInterface, ProverState, VerificationError, VerificationResult,
    VerifierState,
};
use zerocopy::IntoBytes;

use crate::{
    bits::Bits,
    hash::{Hash, BLAKE3, ENGINES},
    transcript::{codecs::U64, ProtocolId},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Config {
    pub hash_id: ProtocolId,
    pub threshold: u64,
}

impl Config {
    /// Creates a new configuration from a difficulty.
    ///
    /// Defaults to Blake3 as the hash function.
    pub fn from_difficulty(difficulty: Bits) -> Self {
        assert!((0.0..60.0).contains(&difficulty.into()));

        let threshold = (64.0 - f64::from(difficulty)).exp2().ceil();
        let threshold = if threshold >= u64::MAX as f64 {
            u64::MAX
        } else {
            threshold as u64
        };

        Self {
            hash_id: BLAKE3,
            threshold,
        }
    }

    pub fn difficulty(&self) -> Bits {
        Bits::from(64.0 - (self.threshold as f64).log2())
    }

    #[cfg_attr(feature = "tracing", instrument(skip(prover_state), fields(engine)))]
    pub fn prove<H, R>(&self, prover_state: &mut ProverState<H, R>)
    where
        H: DuplexSpongeInterface,
        R: RngCore + CryptoRng,
        [u8; 32]: Decoding<[H::U]>,
        U64: Codec<[H::U]>,
    {
        if self.threshold == u64::MAX {
            // If the difficulti is zero, do nothing (also produce no transcript)
            return;
        }

        // Retrieve the engine
        let engine = ENGINES
            .retrieve(self.hash_id)
            .expect("Hash Engine not found");
        #[cfg(feature = "tracing")]
        tracing::Span::current().record("engine", &engine.name());
        let batch_size = engine.preferred_batch_size();

        let challenge: [u8; 32] = prover_state.verifier_message();

        #[cfg(not(feature = "parallel"))]
        let nonce = (0_u64..)
            .step_by(batch_size)
            .find_map({
                let mut inputs = vec![[0u8; 64]; batch_size];
                for input in &mut inputs {
                    input[..32].copy_from_slice(&challenge);
                }
                let mut outputs = vec![Hash::default(); batch_size];
                move |nonce| {
                    for (input, nonce) in inputs.iter_mut().zip(nonce..) {
                        input[32..40].copy_from_slice(&nonce.to_le_bytes());
                    }
                    engine.hash_many(64, inputs.as_bytes(), &mut outputs);
                    for (output, nonce) in outputs.iter().zip(nonce..) {
                        let value = u64::from_le_bytes(output.0[..8].try_into().unwrap());
                        if value <= self.threshold {
                            return Some(nonce);
                        }
                    }
                    None
                }
            })
            .expect("Proof of Work failed to solve.");

        #[cfg(feature = "parallel")]
        let nonce = {
            use std::sync::atomic::{AtomicU64, Ordering};

            // Split the work across all available threads.
            // Use atomics to find the unique deterministic lowest satisfying nonce.
            let global_min = AtomicU64::new(u64::MAX);
            rayon::broadcast(|ctx| {
                let thread_nonces =
                    ((batch_size * ctx.index()) as u64..).step_by(batch_size * ctx.num_threads());
                let mut inputs = vec![[0u8; 64]; batch_size];
                for input in &mut inputs {
                    input[..32].copy_from_slice(&challenge);
                }
                let mut outputs = vec![Hash::default(); batch_size];
                for batch_start in thread_nonces {
                    // Stop work if another thread already found a lower valid nonce.
                    if batch_start >= global_min.load(Ordering::Relaxed) {
                        break;
                    }
                    for (input, nonce) in inputs.iter_mut().zip(batch_start..) {
                        input[32..40].copy_from_slice(&nonce.to_le_bytes());
                    }
                    engine.hash_many(64, inputs.as_bytes(), &mut outputs);
                    for (output, nonce) in outputs.iter().zip(batch_start..) {
                        let value = u64::from_le_bytes(output.0[..8].try_into().unwrap());
                        if value <= self.threshold {
                            // We found a solution, store it in the global_min.
                            // Use fetch_min to solve race condition with simultaneous solutions.
                            global_min.fetch_min(nonce, Ordering::SeqCst);
                            break;
                        }
                    }
                }
            });

            // Return the best found nonce, or fallback check on `u64::MAX`.
            let nonce = global_min.load(Ordering::SeqCst);
            if nonce == u64::MAX {
                panic!("Proof of Work failed to solve.");
            }
            nonce
        };

        prover_state.prover_message(&U64(nonce));
    }

    pub fn verify<H>(&self, verifier_state: &mut VerifierState<H>) -> VerificationResult<()>
    where
        H: DuplexSpongeInterface,
        [u8; 32]: Decoding<[H::U]>,
        U64: Codec<[H::U]>,
    {
        if self.threshold == u64::MAX {
            return Ok(());
        }
        let engine = ENGINES.retrieve(self.hash_id).ok_or(VerificationError)?;
        let challenge: [u8; 32] = verifier_state.verifier_message();
        let nonce: U64 = verifier_state.prover_message()?;

        let mut input = [0u8; 64];
        input[..32].copy_from_slice(&challenge);
        input[32..40].copy_from_slice(&nonce.0.to_le_bytes());
        let mut output = Hash::default();
        engine.hash_many(64, &input, slice::from_mut(&mut output));
        let value = u64::from_le_bytes(output.0[..8].try_into().unwrap());
        if value >= self.threshold {
            return Err(VerificationError);
        }
        Ok(())
    }
}
