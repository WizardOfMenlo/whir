mod blake3_engine;
mod digest_engine;
mod none_engine;

use std::sync::{Arc, LazyLock};

use ark_std::rand::{CryptoRng, RngCore};
use serde::Serialize;
use spongefish::{
    Codec, Decoding, DuplexSpongeInterface, ProverState, VerificationError, VerificationResult,
    VerifierState,
};
use static_assertions::assert_obj_safe;

pub use self::{
    blake3_engine::{Blake3, BLAKE3},
    digest_engine::{DigestEngine, Sha2, Sha3, SHA2, SHA3},
    none_engine::NoneEngine,
};
use crate::{
    bits::Bits,
    ensure,
    transcript::{codecs::U64, Engines, Protocol, ProtocolId},
};

pub const ENGINES: LazyLock<Engines<dyn Engine>> = LazyLock::new(|| {
    let engines = Engines::<dyn Engine>::new();
    engines.register(Arc::new(NoneEngine));
    engines.register(Arc::new(Sha2::new()));
    engines.register(Arc::new(Sha3::new()));
    engines.register(Arc::new(Blake3::new()));
    engines
});

pub trait Engine: Protocol {
    /// Checks if the given nonce satisfies the challenge.
    ///
    /// It does **not** check if it is the minimal solution.
    fn check(&self, challenge: [u8; 32], difficulty: f64, nonce: u64) -> bool;

    /// Finds a `nonce` that satisfies the challenge.
    ///
    /// It is recommended to find the *minimal* solution so that the
    /// proof of work is deterministic.
    fn solve(&self, challenge: [u8; 32], difficulty: f64) -> Option<u64>;
}

assert_obj_safe!(Engine);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub struct Config {
    pub engine_id: ProtocolId,
    pub difficulty: Bits,
}

impl Config {
    pub fn validate(&self) -> Result<(), &'static str> {
        ensure!(
            ENGINES.contains(self.engine_id),
            "Proof of Work Engine not found"
        );
        ensure!(
            f64::from(self.difficulty) >= 0.0,
            "Proof of Work difficulty must be non-negative"
        );
        ensure!(
            f64::from(self.difficulty) < 50.0,
            "Proof of Work difficulty must be less than 50 bits"
        );
        Ok(())
    }

    pub fn none() -> Self {
        Self {
            engine_id: NoneEngine.protocol_id(),
            difficulty: Bits::new(0.0),
        }
    }

    pub fn sha2(difficulty: Bits) -> Self {
        Self {
            engine_id: Sha2::new().protocol_id(),
            difficulty,
        }
    }

    #[cfg_attr(feature = "tracing", instrument(skip(prover_state)))]
    pub fn prove<H, R>(&self, prover_state: &mut ProverState<H, R>)
    where
        H: DuplexSpongeInterface,
        R: RngCore + CryptoRng,
        [u8; 32]: Decoding<[H::U]>,
        U64: Codec<[H::U]>,
    {
        self.validate().expect("Invalid configuration");
        if self.difficulty.is_zero() {
            return;
        }
        let engine = ENGINES
            .retrieve(self.engine_id)
            .expect("Proof of Work Engine not found");
        let challenge: [u8; 32] = prover_state.verifier_message();
        let nonce = engine
            .solve(challenge, self.difficulty.into())
            .expect("Proof of Work failed to solve.");
        prover_state.prover_message(&U64(nonce));
    }

    pub fn verify<H>(&self, verifier_state: &mut VerifierState<H>) -> VerificationResult<()>
    where
        H: DuplexSpongeInterface,
        [u8; 32]: Decoding<[H::U]>,
        U64: Codec<[H::U]>,
    {
        self.validate().map_err(|_| VerificationError)?;
        if self.difficulty.is_zero() {
            return Ok(());
        }
        let engine = ENGINES.retrieve(self.engine_id).ok_or(VerificationError)?;
        let challenge: [u8; 32] = verifier_state.verifier_message();
        let nonce: U64 = verifier_state.prover_message()?;
        if !engine.check(challenge, self.difficulty.into(), nonce.0) {
            return Err(VerificationError);
        }
        Ok(())
    }
}

fn threshold(difficulty: f64) -> u64 {
    assert!(difficulty >= 0.0 && difficulty < 60.0);
    (64.0 - f64::from(difficulty)).exp2().ceil() as u64
}

#[cfg(not(feature = "parallel"))]
pub fn find_min(check: impl Fn(u64) -> bool) -> Option<u64> {
    (0..=u64::MAX).find(|&nonce| check(nonce))
}

#[cfg(feature = "parallel")]
pub fn find_min<F>(check: F) -> Option<u64>
where
    F: Fn(u64) -> bool + Send + Sync,
{
    // Split the work across all available threads.
    // Use atomics to find the unique deterministic lowest satisfying nonce.
    use std::sync::atomic::{AtomicU64, Ordering};

    let global_min = AtomicU64::new(u64::MAX);
    let _ = rayon::broadcast(|ctx| {
        let nonces = (ctx.index() as u64..).step_by(ctx.num_threads());
        for nonce in nonces {
            // Use relaxed ordering to eventually get notified of another thread's solution.
            // (Propagation delay should be in the order of tens of nanoseconds.)
            if nonce >= global_min.load(Ordering::Relaxed) {
                break;
            }
            if check(nonce) {
                // We found a solution, store it in the global_min.
                // Use fetch_min to solve race condition with simultaneous solutions.
                global_min.fetch_min(nonce, Ordering::SeqCst);
                break;
            }
        }
    });
    let nonce = global_min.load(Ordering::SeqCst);
    if nonce == u64::MAX {
        // This may be the initial value or a solution.
        check(nonce).then(|| nonce)
    } else {
        Some(nonce)
    }
}
