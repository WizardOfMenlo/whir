pub mod transcript;

use spongefish::{DomainSeparator, ProverState, VerifierState};
use transcript::{
    Interaction, Transcript, TranscriptError, TranscriptExt, TranscriptPattern, TranscriptPlayer,
    TranscriptRecorder,
};

/// A prover/verifier state containing a read only configuration object, a transcript and a mutable state.
pub struct State<'a, C: 'static, T: Transcript, S> {
    config: &'a C,
    transcript: &'a mut T,
    inner: &'a mut S,
}

pub type Pattern<'a, C> = State<'a, C, TranscriptRecorder, DomainSeparator>;
pub type Prover<'a, C> = State<'a, C, TranscriptPlayer<'a>, ProverState>;
pub type Verifier<'a, C> = State<'a, C, TranscriptPlayer<'a>, VerifierState<'a>>;

impl<C, T: Transcript, S> Transcript for State<'_, C, T, S> {
    type Error = T::Error;

    fn interact(&mut self, interaction: Interaction) -> Result<(), Self::Error> {
        self.transcript.interact(interaction)
    }
}

impl<'a, C: 'static, T: Transcript, S> State<'a, C, T, S> {
    pub const fn new(config: &'a C, transcript: &'a mut T, inner: &'a mut S) -> Self {
        Self {
            config,
            transcript,
            inner,
        }
    }

    /// Group an interaction between matching BEGIN and END interactions.
    pub fn group<R>(
        &mut self,
        label: &'static str,
        f: impl FnOnce(&mut Self) -> R,
    ) -> Result<R, T::Error> {
        self.transcript.begin::<C>(label)?;
        let result = f(self);
        self.transcript.end::<C>(label)?;
        Ok(result)
    }

    /// Borrow a sub-state with a specified config object.
    pub const fn with<'b, C2: 'static>(&'b mut self, config: &'b C2) -> State<'b, C2, T, S> {
        State {
            config,
            transcript: self.transcript,
            inner: self.inner,
        }
    }
}

/////////////////////////

pub struct ProofOfWork {
    difficulty: f64,
}

impl Pattern<'_, ProofOfWork> {
    pub fn proof_of_work(&mut self) -> Result<(), TranscriptError> {
        self.begin::<ProofOfWork>("proof_of_work")?;
        self.challenge::<[u8; 32]>("challenge")?;
        self.message::<ProofOfWork>("nonce")?;
        self.end::<ProofOfWork>("proof_of_work")?;
        Ok(())
    }
}

/////////////////

pub struct WhirOpening {
    first: ProofOfWork,
    second: ProofOfWork,
}

impl<T: Transcript, S> State<'_, WhirOpening, T, S> {
    const fn first(&mut self) -> State<ProofOfWork, T, S> {
        self.with(&self.config.first)
    }

    const fn second(&mut self) -> State<ProofOfWork, T, S> {
        self.with(&self.config.second)
    }
}

impl Pattern<'_, WhirOpening> {
    pub fn opening(&mut self) -> Result<(), TranscriptError> {
        self.begin::<WhirOpening>("whir_opening")?;

        self.group("first", |this| this.first().proof_of_work())??;

        self.second().proof_of_work()?;

        self.end::<WhirOpening>("whir_opening")?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transcript_builder() {
        let config = WhirOpening {
            first: ProofOfWork { difficulty: 16.0 },
            second: ProofOfWork { difficulty: 20.0 },
        };
        let mut recorder = TranscriptRecorder::new();
        let mut domainsep = DomainSeparator::new("asd");

        let mut pattern = Pattern::new(&config, &mut recorder, &mut domainsep);
        pattern.opening().unwrap();

        let pattern = recorder.finalize().unwrap_or_else(|e| panic!("Error: {e}"));

        eprintln!("{pattern}");
        panic!();
    }
}
