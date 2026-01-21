use ark_ff::FftField;
use spongefish::{Codec, DuplexSpongeInterface, VerificationResult};

use crate::{
    hash::Hash,
    protocols::matrix_commit,
    transcript::{ProverMessage, VerifierMessage, VerifierState},
    whir::{
        parameters::WhirConfig,
        statement::{Constraint, Weights},
    },
};

///
///  Commitment parsed by the verifier from verifier's FS context.
///
///

#[derive(Clone)]
pub struct ParsedCommitment<F> {
    pub matrix_commitment: matrix_commit::Commitment,
    pub num_variables: usize,
    pub ood_points: Vec<F>,
    pub ood_answers: Vec<Vec<F>>,
    pub batching_randomness: F,
}

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
        ParsedCommitment::receive(
            verifier_state,
            &self.0.initial_matrix_committer,
            self.0.mv_parameters.num_variables,
            self.0.committment_ood_samples,
            self.0.batch_size,
        )
    }
}

impl<F: FftField> ParsedCommitment<F> {
    pub fn receive<H>(
        verifier_state: &mut VerifierState<H>,
        matrix_commit: &matrix_commit::Config<F>,
        num_variables: usize,
        ood_samples: usize,
        batch_size: usize,
    ) -> VerificationResult<ParsedCommitment<F>>
    where
        H: DuplexSpongeInterface,
        F: Codec<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        let matrix_commitment = matrix_commit.receive_commitment(verifier_state)?;

        let mut ood_points = vec![F::ZERO; ood_samples];
        let mut ood_answers = Vec::with_capacity(batch_size);

        if ood_samples > 0 {
            for oods_point in &mut ood_points {
                *oods_point = verifier_state.verifier_message();
            }
            for _ in 0..batch_size {
                let mut virt_answers = vec![F::ZERO; ood_samples];
                for answer in &mut virt_answers {
                    *answer = verifier_state.prover_message()?;
                }
                ood_answers.push(virt_answers);
            }
        }

        let batching_randomness = if batch_size > 1 {
            verifier_state.verifier_message()
        } else {
            F::zero()
        };

        Ok(ParsedCommitment {
            matrix_commitment,
            batching_randomness,
            num_variables,
            ood_points,
            ood_answers,
        })
    }

    /// Return constraints for OODS
    pub fn oods_constraints(&self) -> Vec<Constraint<F>> {
        let mut multiplier = self.batching_randomness;

        let result = self
            .ood_answers
            .clone()
            .into_iter()
            .reduce(|result, this_round| {
                let this_multiplier = multiplier;
                multiplier *= self.batching_randomness;
                result
                    .into_iter()
                    .zip(this_round)
                    .map(|(lhs, v)| lhs + (v * this_multiplier))
                    .collect()
            })
            .unwrap_or_default();

        self.ood_points
            .iter()
            .zip(result.iter())
            .map(|(&point, &eval)| Constraint {
                weights: Weights::univariate(point, self.num_variables),
                sum: eval,
                defer_evaluation: false,
            })
            .collect()
    }
}
