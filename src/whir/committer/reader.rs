use ark_ff::FftField;

use super::ParsedCommitment;
use crate::{
    hash::Hash,
    transcript::{Codec, DuplexSpongeInterface, ProverMessage, VerificationResult, VerifierState},
    whir::parameters::WhirConfig,
};

pub struct CommitmentReader<'a, F: FftField>(&'a WhirConfig<F>);

impl<'a, F: FftField> CommitmentReader<'a, F> {
    pub const fn new(params: &'a WhirConfig<F>) -> Self {
        Self(params)
    }

    pub fn parse_commitment<H>(
        &self,
        verifier_state: &mut VerifierState<H>,
    ) -> VerificationResult<ParsedCommitment<F>>
    where
        H: DuplexSpongeInterface,
        F: Codec<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        let config = &self.0.initial_committer;
        let commitment = config.receive_commitment(verifier_state)?;
        Ok(ParsedCommitment { commitment })
    }
}
