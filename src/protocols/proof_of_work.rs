use spongefish::{
    codecs::{ZeroCopyPattern, ZeroCopyProver, ZeroCopyVerifier},
    transcript::Label,
    Unit,
};
use thiserror::Error;

pub struct Config {
    solver: Box<dyn Fn([u8; 32], f64) -> u64>,
    verifier: Box<dyn Fn([u8; 32], f64, u64) -> bool>,
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
        let nonce = (config.solver)(challenge, config.difficulty);
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
        if !(config.verifier)(challenge, config.difficulty, nonce) {
            return Err(VerifierError::ProofOfWorkFailed);
        }
        self.end_protocol::<Config>(label)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use spongefish::{transcript::TranscriptRecorder, ProverState, VerifierState};

    use super::*;

    #[test]
    fn test_all_ops() -> Result<()> {
        let config = Config {
            solver: Box::new(|challenge, diffifculy| 42_u64),
            verifier: Box::new(|challenge, diffifculy, nonce| nonce == 42_u64),
            difficulty: 5.0,
        };

        let mut pattern: TranscriptRecorder = TranscriptRecorder::new();
        pattern.proof_of_work("pow", &config)?;
        let pattern = pattern.finalize()?;
        eprintln!("{pattern}");

        let mut prover: ProverState = ProverState::from(&pattern);
        prover.proof_of_work("pow", &config)?;
        let proof = prover.finalize()?;
        assert_eq!(hex::encode(&proof), "2a00000000000000");

        let mut verifier: VerifierState = VerifierState::new(pattern.into(), &proof);
        verifier.proof_of_work("pow", &config)?;
        verifier.finalize()?;

        Ok(())
    }
}
