mod digest;
mod utils;

use std::{
    fmt::{Debug, Display},
    hash::Hash,
    sync::Arc,
};

use spongefish::{
    codecs::{ZeroCopyPattern, ZeroCopyProver, ZeroCopyVerifier},
    transcript::Label,
    Unit,
};
use thiserror::Error;
use zerocopy::transmute;

pub use self::digest::DigestEngine;
use self::utils::f64_to_u256;

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
pub enum VerifierError<E> {
    #[error(transparent)]
    Inner(#[from] E),
    #[error("Proof of work does not meet the difficulty requirement.")]
    ProofOfWorkFailed,
}

pub trait ProofOfWorkPattern<U>: ZeroCopyPattern<U>
where
    U: Unit,
{
    fn proof_of_work(
        &mut self,
        label: impl Into<Label>,
        config: &Config,
    ) -> Result<(), Self::Error>;
}

pub trait ProofOfWorkProver<U>: ZeroCopyProver<U>
where
    U: Unit,
{
    fn proof_of_work(
        &mut self,
        label: impl Into<Label>,
        config: &Config,
    ) -> Result<(), Self::Error>;
}

pub trait ProofOfWorkVerifier<'a, U>: ZeroCopyVerifier<'a, U>
where
    U: Unit,
{
    fn proof_of_work(
        &mut self,
        label: impl Into<Label>,
        config: &Config,
    ) -> Result<(), VerifierError<Self::Error>>;
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

impl<U, P> ProofOfWorkPattern<U> for P
where
    U: Unit,
    P: ZeroCopyPattern<U>,
{
    fn proof_of_work(
        &mut self,
        label: impl Into<Label>,
        _config: &Config,
    ) -> Result<(), Self::Error> {
        let label = label.into();
        self.begin_protocol::<Config>(label.clone())?;
        self.challenge_zerocopy::<[u8; 32]>("challenge")?;
        self.message_zerocopy::<u64>("nonce")?;
        self.end_protocol::<Config>(label)
    }
}

impl<U, P> ProofOfWorkProver<U> for P
where
    U: Unit,
    P: ZeroCopyProver<U>,
{
    fn proof_of_work(
        &mut self,
        label: impl Into<Label>,
        config: &Config,
    ) -> Result<(), Self::Error> {
        let label = label.into();
        self.begin_protocol::<Config>(label.clone())?;
        let challenge = self.challenge_zerocopy::<[u8; 32]>("challenge")?;
        let nonce = config.engine.solve(challenge, config.difficulty);
        self.message_zerocopy::<u64>("nonce", &nonce)?;
        self.end_protocol::<Config>(label)
    }
}

impl<'a, U, P> ProofOfWorkVerifier<'a, U> for P
where
    U: Unit,
    P: ZeroCopyVerifier<'a, U>,
{
    fn proof_of_work(
        &mut self,
        label: impl Into<Label>,
        config: &Config,
    ) -> Result<(), VerifierError<Self::Error>> {
        let label = label.into();
        self.begin_protocol::<Config>(label.clone())?;
        let challenge = self.challenge_zerocopy::<[u8; 32]>("challenge")?;
        let nonce = self.message_zerocopy::<u64>("nonce")?;
        if !config.engine.verify(challenge, config.difficulty, nonce) {
            return Err(VerifierError::ProofOfWorkFailed);
        }
        self.end_protocol::<Config>(label)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use sha3::Keccak256;
    use spongefish::{transcript::TranscriptRecorder, ProverState, VerifierState};

    use super::*;
    use crate::protocols::proof_of_work::digest::DigestEngine;

    #[test]
    fn test_all_ops() -> Result<()> {
        let config = Config {
            engine: Arc::new(DigestEngine::<Keccak256>::new()),
            difficulty: 5.0,
        };

        let mut pattern: TranscriptRecorder = TranscriptRecorder::new();
        pattern.proof_of_work("pow", &config)?;
        let pattern = pattern.finalize()?;
        eprintln!("{pattern}");

        let mut prover: ProverState = ProverState::from(&pattern);
        prover.proof_of_work("pow", &config)?;
        let proof = prover.finalize()?;
        assert_eq!(hex::encode(&proof), "1b00000000000000");

        let mut verifier: VerifierState = VerifierState::new(pattern.into(), &proof);
        verifier.proof_of_work("pow", &config)?;
        verifier.finalize()?;

        Ok(())
    }
}
