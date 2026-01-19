use ark_crypto_primitives::merkle_tree::Config;
use ark_ff::FftField;
use spongefish::{Codec, DuplexSpongeInterface, VerificationResult, VerifierState};

use crate::{
    transcript::ProverMessage,
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
pub struct ParsedCommitment<F, D> {
    pub root: D,
    pub num_variables: usize,
    pub ood_points: Vec<F>,
    pub ood_answers: Vec<Vec<F>>,
    pub batching_randomness: F,
}

pub struct CommitmentReader<'a, F, MerkleConfig>(&'a WhirConfig<F, MerkleConfig>)
where
    F: FftField,
    MerkleConfig: Config;

impl<'a, F, MerkleConfig> CommitmentReader<'a, F, MerkleConfig>
where
    F: FftField,
    MerkleConfig: Config<Leaf = [F]>,
{
    pub const fn new(params: &'a WhirConfig<F, MerkleConfig>) -> Self {
        Self(params)
    }

    pub fn parse_commitment<H>(
        &self,
        verifier_state: &mut VerifierState<H>,
    ) -> VerificationResult<ParsedCommitment<F, MerkleConfig::InnerDigest>>
    where
        H: DuplexSpongeInterface,
        F: Codec<[H::U]>,
        MerkleConfig::InnerDigest: ProverMessage<[H::U]>,
    {
        let root = verifier_state.prover_message()?;

        let mut ood_points = vec![F::ZERO; self.0.committment_ood_samples];
        let mut ood_answers = Vec::with_capacity(self.0.batch_size);

        if self.0.committment_ood_samples > 0 {
            for oods_point in &mut ood_points {
                *oods_point = verifier_state.verifier_message();
            }
            for _ in 0..self.0.batch_size {
                let mut virt_answers = vec![F::ZERO; self.0.committment_ood_samples];
                for answer in &mut virt_answers {
                    *answer = verifier_state.prover_message()?;
                }
                ood_answers.push(virt_answers);
            }
        }

        let batching_randomness = if self.0.batch_size > 1 {
            verifier_state.verifier_message()
        } else {
            F::zero()
        };

        Ok(ParsedCommitment {
            root,
            batching_randomness,
            num_variables: self.0.mv_parameters.num_variables,
            ood_points,
            ood_answers,
        })
    }
}

impl<F, D> ParsedCommitment<F, D>
where
    F: ark_ff::Field,
{
    pub fn parse<H, MerkleConfig>(
        verifier_state: &mut VerifierState<H>,
        num_variables: usize,
        ood_samples: usize,
    ) -> VerificationResult<ParsedCommitment<F, MerkleConfig::InnerDigest>>
    where
        H: DuplexSpongeInterface,
        F: Codec<[H::U]>,
        MerkleConfig: Config<Leaf = [F]>,
        MerkleConfig::InnerDigest: ProverMessage<[H::U]>,
    {
        let root = verifier_state.prover_message()?;

        let mut ood_points = vec![F::ZERO; ood_samples];
        let mut ood_answers = vec![F::ZERO; ood_samples];
        if ood_samples > 0 {
            for oods_point in &mut ood_points {
                *oods_point = verifier_state.verifier_message();
            }
            for oods_answer in &mut ood_answers {
                *oods_answer = verifier_state.prover_message()?;
            }
        }

        Ok(ParsedCommitment {
            num_variables,
            root,
            ood_points,
            ood_answers: vec![ood_answers],
            batching_randomness: F::ZERO,
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
