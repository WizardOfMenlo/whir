use ark_ff::FftField;
use spongefish::{Codec, DuplexSpongeInterface, VerificationResult};

use super::ParsedCommitment;
use crate::{
    hash::Hash,
    transcript::{ProverMessage, VerifierMessage, VerifierState},
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
        let commitment = self
            .0
            .initial_committer
            .receive_commitment(verifier_state)?;
        let batching_randomness = if self.0.initial_committer.num_polynomials > 1 {
            verifier_state.verifier_message()
        } else {
            F::zero()
        };
        Ok(ParsedCommitment {
            polynomial_size: self.0.initial_committer.polynomial_size,
            num_polynomials: self.0.initial_committer.num_polynomials,
            commitment,
            batching_randomness,
        })
    }
}
