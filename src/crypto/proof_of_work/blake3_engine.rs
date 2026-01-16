use blake3::{
    guts::BLOCK_LEN,
    platform::{Platform, MAX_SIMD_DEGREE},
    IncrementCounter, OUT_LEN,
};
use hex_literal::hex;
#[cfg(feature = "parallel")]
use rayon::broadcast;
use sha3::{Digest as _, Sha3_256};

use super::{threshold, Engine};
use crate::transcript::{Protocol, ProtocolId};

pub const BLAKE3: ProtocolId = ProtocolId::new(hex!(
    "c819a8eab7fada325ebb05d542244f6ca56b22f20cc671b03d290829e24eeb8c"
));

pub struct Blake3 {
    platform: Platform,
}

/// Default Blake3 initialization vector. Copied here because it is not publicly exported.
const BLAKE3_IV: [u32; 8] = [
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
];
const BLAKE3_FLAGS: u8 = 0x0B; // CHUNK_START | CHUNK_END | ROOT

/// SIMD batch of hash inputs, each 64 bytes (challenge + nonce).
type InputBuffer = [[u8; BLOCK_LEN]; MAX_SIMD_DEGREE];

/// SIMD batch of hash outputs (32 bytes each).
type OutputBuffer = [u8; OUT_LEN * MAX_SIMD_DEGREE];

impl Blake3 {
    pub fn new() -> Self {
        Self {
            platform: Platform::detect(),
        }
    }
}

impl Engine for Blake3 {
    fn check(&self, challenge: [u8; 32], difficulty: f64, nonce: u64) -> bool {
        // Create a new BLAKE3 hasher instance.
        let mut hasher = blake3::Hasher::new();

        // Feed the challenge prefix.
        hasher.update(&challenge);
        // Feed the nonce as little-endian bytes.
        hasher.update(&nonce.to_le_bytes());
        // Zero-extend the nonce to 32 bytes (challenge + nonce = full block).
        hasher.update(&[0; 24]);

        // Hash the input and extract the first 8 bytes.
        let mut hash = [0u8; 8];
        hasher.finalize_xof().fill(&mut hash);

        // Check whether the result is below the threshold.
        u64::from_le_bytes(hash) < threshold(difficulty)
    }

    #[cfg(not(feature = "parallel"))]
    fn solve(&self, challenge: [u8; 32], difficulty: f64) -> Option<u64> {
        let threshold = super::threshold(difficulty);
        let mut inputs = [[0u8; BLOCK_LEN]; MAX_SIMD_DEGREE];
        for input in &mut inputs {
            input[..32].copy_from_slice(&challenge);
        }
        let mut outputs = [0; OUT_LEN * MAX_SIMD_DEGREE];

        (0..).step_by(MAX_SIMD_DEGREE).find_map(|nonce| {
            check_many(self.platform, nonce, threshold, &mut inputs, &mut outputs)
        })
    }

    #[cfg(feature = "parallel")]
    fn solve(&self, challenge: [u8; 32], difficulty: f64) -> Option<u64> {
        use std::sync::atomic::{AtomicU64, Ordering};
        let threshold = super::threshold(difficulty);

        // Split the work across all available threads.
        // Use atomics to find the unique deterministic lowest satisfying nonce.
        let global_min = AtomicU64::new(u64::MAX);

        // Spawn parallel workers using Rayon's broadcast.
        let _ = broadcast(|ctx| {
            let mut inputs = [[0u8; BLOCK_LEN]; MAX_SIMD_DEGREE];
            for input in &mut inputs {
                input[..32].copy_from_slice(&challenge);
            }
            let mut outputs = [0; OUT_LEN * MAX_SIMD_DEGREE];

            // Each thread searches a distinct subset of nonces.
            let nonces = ((MAX_SIMD_DEGREE * ctx.index()) as u64..)
                .step_by(MAX_SIMD_DEGREE * ctx.num_threads());

            for starting_nonce in nonces {
                // Skip work if another thread already found a lower valid nonce.
                //
                // Use relaxed ordering to eventually get notified of another thread's solution.
                // (Propagation delay should be in the order of tens of nanoseconds.)
                if starting_nonce >= global_min.load(Ordering::Relaxed) {
                    break;
                }
                // Check a batch of nonces starting from `nonce`.
                if let Some(nonce) = check_many(
                    self.platform,
                    starting_nonce,
                    threshold,
                    &mut inputs,
                    &mut outputs,
                ) {
                    // We found a solution, store it in the global_min.
                    // Use fetch_min to solve race condition with simultaneous solutions.
                    global_min.fetch_min(nonce, Ordering::SeqCst);
                    break;
                }
            }
        });

        // Return the best found nonce, or fallback check on `u64::MAX`.
        let nonce = global_min.load(Ordering::SeqCst);
        if nonce == u64::MAX && !self.check(challenge, difficulty, nonce) {
            return None;
        }
        Some(nonce)
    }
}

impl Protocol for Blake3 {
    fn protocol_id(&self) -> crate::transcript::ProtocolId {
        let mut hasher = Sha3_256::new();
        hasher.update(b"whir::protocols::proof_of_work::Blake3");
        let hash: [u8; 32] = hasher.finalize().into();
        hash.into()
    }
}

/// Check a SIMD-width batch of nonces starting at `starting_nonce`.
///
/// Returns the first nonce in the batch that satisfies the challenge threshold,
/// or `None` if none do.
fn check_many(
    platform: Platform,
    starting_nonce: u64,
    threshold: u64,
    inputs: &mut InputBuffer,
    outputs: &mut OutputBuffer,
) -> Option<u64> {
    // Fill each SIMD input block with the challenge + nonce suffix.
    for (i, input) in inputs.iter_mut().enumerate() {
        // Write the nonce as little-endian into bytes 32..40.
        let n = (starting_nonce + i as u64).to_le_bytes();
        input[32..40].copy_from_slice(&n);
    }

    // Create references required by `hash_many`.
    let input_refs: [&[u8; BLOCK_LEN]; MAX_SIMD_DEGREE] = std::array::from_fn(|i| &inputs[i]);

    // Perform parallel hashing over the input blocks.
    platform.hash_many::<BLOCK_LEN>(
        &input_refs,
        &BLAKE3_IV,           // Initialization vector
        0,                    // Counter
        IncrementCounter::No, // Do not increment counter
        BLAKE3_FLAGS,         // Default flags
        0,
        0, // No start/end flags
        outputs,
    );

    // Scan results and return the first nonce under the threshold.
    for (i, chunk) in outputs.chunks_exact(OUT_LEN).enumerate() {
        let hash = u64::from_le_bytes(chunk[..8].try_into().unwrap());
        if hash < threshold {
            return Some(starting_nonce + i as u64);
        }
    }

    // None of the batch satisfied the condition.
    None
}

#[cfg(test)]
mod tests {
    use spongefish::{domain_separator, session};

    use super::*;
    use crate::{bits::Bits, crypto::proof_of_work::Config};

    #[test]
    fn protocol_id() {
        assert_eq!(Blake3::new().protocol_id(), BLAKE3);
    }

    #[test]
    fn test_pow() {
        let config = Config {
            engine_id: BLAKE3,
            difficulty: Bits::new(4.0),
        };
        let ds = domain_separator!("whir::protocols::proof_of_work")
            .session(session!("Test at {}:{}", file!(), line!()))
            .instance(&0_u32);

        let mut prover_state = ds.std_prover();
        config.prove(&mut prover_state);
        let proof = prover_state.narg_string();

        let mut verifier_state = ds.std_verifier(&proof);
        config.verify(&mut verifier_state).unwrap();
    }
}
