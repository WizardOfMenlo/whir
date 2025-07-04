mod digest;
mod utils;

use std::{
    fmt::{Debug, Display},
    hash::Hash,
    sync::Arc,
};

use spongefish::{
    codecs::zerocopy,
    transcript::{self, Label},
};
use thiserror::Error;

pub use self::digest::DigestEngine;

pub trait Engine: Sync + Send + Debug + Display {
    fn solve(&self, challenge: [u8; 32], difficulty: f64) -> u64 {
        for nonce in 0.. {
            if self.verify(challenge, difficulty, nonce) {
                return nonce;
            }
        }
        panic!("No valid nonce found for the given challenge and difficulty.");
    }

    fn verify(&self, challenge: [u8; 32], difficulty: f64, nonce: u64) -> bool;
}

#[derive(Debug, Clone)]
pub struct Config {
    engine: Arc<dyn Engine>,
    difficulty: f64,
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Error)]
pub enum VerifierError {
    #[error(transparent)]
    Verifier(#[from] spongefish::VerifierError),
    #[error("Proof of work does not meet the difficulty requirement.")]
    ProofOfWorkFailed,
}

pub trait Pattern {
    fn proof_of_work(&mut self, label: Label, config: &Config);
}

pub trait Prover {
    fn proof_of_work(&mut self, label: Label, config: &Config);
}

pub trait Verifier {
    fn proof_of_work(&mut self, label: Label, config: &Config) -> Result<(), VerifierError>;
}

impl Config {
    pub fn new(engine: Arc<dyn Engine>, difficulty: f64) -> Self {
        Self { engine, difficulty }
    }
}

impl PartialEq for Config {
    fn eq(&self, other: &Self) -> bool {
        self.engine.to_string() == other.engine.to_string() && self.difficulty == other.difficulty
    }
}

impl Eq for Config {}

impl PartialOrd for Config {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        match self
            .engine
            .to_string()
            .partial_cmp(&other.engine.to_string())
        {
            Some(std::cmp::Ordering::Equal) => self.difficulty.partial_cmp(&other.difficulty),
            other => other,
        }
    }
}

impl Ord for Config {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match self.engine.to_string().cmp(&other.engine.to_string()) {
            std::cmp::Ordering::Equal => self.difficulty.partial_cmp(&other.difficulty).unwrap(),
            other => other,
        }
    }
}

impl Hash for Config {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.engine.to_string().hash(state);
        self.difficulty.to_bits().hash(state);
    }
}

impl Display for Config {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Proof of work {} bits of {}",
            self.difficulty, self.engine
        )
    }
}

impl<P> Pattern for P
where
    P: transcript::Pattern + zerocopy::Pattern,
{
    fn proof_of_work(&mut self, label: Label, _config: &Config) {
        self.begin_protocol::<Config>(label);
        self.challenge_zerocopy::<[u8; 32]>("challenge");
        self.message_zerocopy::<u64>("nonce");
        self.end_protocol::<Config>(label)
    }
}

impl<P> Prover for P
where
    P: transcript::Prover + zerocopy::Prover,
{
    fn proof_of_work(&mut self, label: Label, config: &Config) {
        self.begin_protocol::<Config>(label);
        let challenge = self.challenge_zerocopy::<[u8; 32]>("challenge");
        let nonce = config.engine.solve(challenge, config.difficulty);
        self.message_zerocopy::<u64>("nonce", &nonce);
        self.end_protocol::<Config>(label);
    }
}

impl<'a, P> Verifier for P
where
    P: transcript::Verifier + zerocopy::Verifier<'a>,
{
    fn proof_of_work(&mut self, label: Label, config: &Config) -> Result<(), VerifierError> {
        self.begin_protocol::<Config>(label);
        let challenge = self.challenge_zerocopy::<[u8; 32]>("challenge");
        let nonce = self.message_zerocopy::<u64>("nonce")?;
        if !config.engine.verify(challenge, config.difficulty, nonce) {
            return Err(VerifierError::ProofOfWorkFailed);
        }
        self.end_protocol::<Config>(label);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use sha3::Keccak256;
    use spongefish::{transcript::PatternState, ProverState, VerifierState};

    use super::*;
    use crate::protocols::proof_of_work::digest::DigestEngine;

    #[test]
    fn test_all_ops() -> Result<()> {
        let config = Config {
            engine: Arc::new(DigestEngine::<Keccak256>::new()),
            difficulty: 5.0,
        };

        let mut pattern: PatternState = PatternState::new();
        pattern.proof_of_work("pow", &config);
        let pattern = pattern.finalize();
        eprintln!("{pattern}");

        let mut prover: ProverState = ProverState::from(&pattern);
        prover.proof_of_work("pow", &config);
        let proof = prover.finalize();
        assert_eq!(hex::encode(&proof), "3100000000000000");

        let mut verifier: VerifierState = VerifierState::new(pattern.into(), &proof);
        verifier.proof_of_work("pow", &config)?;
        verifier.finalize();

        Ok(())
    }
}
