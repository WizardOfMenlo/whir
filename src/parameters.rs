use std::{fmt::Display, marker::PhantomData, str::FromStr};

use ark_crypto_primitives::merkle_tree::{Config, LeafParam, TwoToOneParam};
use ark_ff::FftField;
use ark_poly::EvaluationDomain;
use serde::Serialize;
use thiserror::Error;

use crate::whir::{
    parameters::WhirConfig, parsed_proof::ParsedProof, prover::RoundState,
    stir_evaluations::StirEvalContext,
};

/// Computes the default maximum proof-of-work (PoW) bits.
///
/// This function determines the PoW security level based on the number of variables
/// and the logarithmic inverse rate.
pub const fn default_max_pow(num_variables: usize, log_inv_rate: usize) -> usize {
    num_variables + log_inv_rate - 3
}

/// Defines the soundness type for the proof system.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum SoundnessType {
    /// Unique decoding guarantees a single valid witness.
    UniqueDecoding,
    /// Provable list decoding allows multiple valid witnesses but provides proof.
    ProvableList,
    /// Conjecture-based list decoding with no strict guarantees.
    ConjectureList,
}

impl Display for SoundnessType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::ProvableList => "ProvableList",
            Self::ConjectureList => "ConjectureList",
            Self::UniqueDecoding => "UniqueDecoding",
        })
    }
}

impl FromStr for SoundnessType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "ProvableList" => Ok(Self::ProvableList),
            "ConjectureList" => Ok(Self::ConjectureList),
            "UniqueDecoding" => Ok(Self::UniqueDecoding),
            _ => Err(format!("Invalid soundness specification: {s}")),
        }
    }
}

/// Represents the parameters for a multivariate polynomial.
#[derive(Debug, Clone, Copy)]
pub struct MultivariateParameters<F> {
    /// The number of variables in the polynomial.
    pub(crate) num_variables: usize,
    _field: PhantomData<F>,
}

impl<F> MultivariateParameters<F> {
    /// Creates new multivariate parameters.
    pub const fn new(num_variables: usize) -> Self {
        Self {
            num_variables,
            _field: PhantomData,
        }
    }
}

impl<F> Display for MultivariateParameters<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Number of variables: {}", self.num_variables)
    }
}

/// Defines the folding strategy for polynomial commitments.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FoldType {
    /// A naive approach with minimal optimizations.
    Naive,
    /// A strategy where the prover aids in the folding process.
    ProverHelps,
}

impl FoldType {
    /// Computes folded evaluations for a single round of the proof,
    /// based on the current folding strategy.
    ///
    /// Dispatches the STIR evaluation logic according to the chosen folding strategy.
    ///
    /// - If `Naive`, uses coset-based folding via `compute_fold`.
    /// - If `ProverHelps`, assumes coefficients and evaluates directly at $\vec{r}$.
    ///
    /// This method is used by the prover when deriving folded polynomial values at queried points.
    pub(crate) fn stir_evaluations_prover<F, MerkleConfig>(
        self,
        round_state: &RoundState<F, MerkleConfig>,
        stir_challenges_indexes: &[usize],
        answers: &[Vec<F>],
        folding_factor: FoldingFactor,
        stir_evaluations: &mut Vec<F>,
    ) where
        F: FftField,
        MerkleConfig: Config,
    {
        let ctx = match self {
            Self::Naive => StirEvalContext::Naive {
                domain_size: round_state.domain.backing_domain.size(),
                domain_gen_inv: round_state
                    .domain
                    .backing_domain
                    .element(1)
                    .inverse()
                    .unwrap(),
                round: round_state.round,
                stir_challenges_indexes,
                folding_factor: &folding_factor,
                folding_randomness: &round_state.folding_randomness,
            },
            Self::ProverHelps => StirEvalContext::ProverHelps {
                folding_randomness: &round_state.folding_randomness,
            },
        };
        ctx.evaluate(answers, stir_evaluations);
    }

    /// Computes folded evaluations across all rounds of the proof,
    /// based on the configured folding strategy.
    ///
    /// This function is used during proof verification by the verifier.
    ///
    /// # Returns
    /// - A list of folded evaluations for each round, including the final round.
    ///
    /// # Strategy Behavior
    ///
    /// - If `Naive`, performs coset-based folding round by round.
    /// - If `ProverHelps`, reuses the precomputed coefficient evaluations.
    pub(crate) fn stir_evaluations_verifier<F, MerkleConfig, PowStrategy>(
        self,
        parsed: &ParsedProof<F>,
        params: &WhirConfig<F, MerkleConfig, PowStrategy>,
    ) -> Vec<Vec<F>>
    where
        F: FftField,
        MerkleConfig: Config,
    {
        match self {
            Self::Naive => {
                // Start with the domain size and the fold vector
                let mut domain_size = params.starting_domain.backing_domain.size();
                let mut result = Vec::with_capacity(parsed.rounds.len() + 1);

                for (round_index, round) in parsed.rounds.iter().enumerate() {
                    // Compute the folds for this round
                    let mut round_evals = Vec::with_capacity(round.stir_challenges_indexes.len());
                    let stir_evals_context = StirEvalContext::Naive {
                        domain_size,
                        domain_gen_inv: round.domain_gen_inv,
                        round: round_index,
                        stir_challenges_indexes: &round.stir_challenges_indexes,
                        folding_factor: &params.folding_factor,
                        folding_randomness: &round.folding_randomness,
                    };
                    stir_evals_context.evaluate(&round.stir_challenges_answers, &mut round_evals);

                    // Push the folds to the result
                    result.push(round_evals);
                    // Update the domain size
                    domain_size /= 2;
                }

                // Final round
                let final_round_index = parsed.rounds.len();
                let mut final_evals = Vec::with_capacity(parsed.final_randomness_indexes.len());

                let stir_evals_context = StirEvalContext::Naive {
                    domain_size,
                    domain_gen_inv: parsed.final_domain_gen_inv,
                    round: final_round_index,
                    stir_challenges_indexes: &parsed.final_randomness_indexes,
                    folding_factor: &params.folding_factor,
                    folding_randomness: &parsed.final_folding_randomness,
                };

                stir_evals_context.evaluate(&parsed.final_randomness_answers, &mut final_evals);

                result.push(final_evals);
                result
            }
            Self::ProverHelps => parsed.compute_folds_helped(),
        }
    }
}

impl FromStr for FoldType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "Naive" => Ok(Self::Naive),
            "ProverHelps" => Ok(Self::ProverHelps),
            _ => Err(format!("Invalid fold type specification: {s}")),
        }
    }
}

impl Display for FoldType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Naive => "Naive",
            Self::ProverHelps => "ProverHelps",
        })
    }
}

/// Errors that can occur when validating a folding factor.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum FoldingFactorError {
    /// The folding factor is larger than the number of variables.
    #[error(
        "Folding factor {0} is greater than the number of variables {1}. Polynomial too small, just send it directly."
    )]
    TooLarge(usize, usize),

    /// The folding factor cannot be zero.
    #[error("Folding factor shouldn't be zero.")]
    ZeroFactor,
}

/// Defines the folding factor for polynomial commitments.
#[derive(Debug, Clone, Copy)]
pub enum FoldingFactor {
    /// A fixed folding factor used in all rounds.
    Constant(usize),
    /// Uses a different folding factor for the first round and a fixed one for the rest.
    ConstantFromSecondRound(usize, usize),
}

impl FoldingFactor {
    /// Retrieves the folding factor for a given round.
    pub const fn at_round(&self, round: usize) -> usize {
        match self {
            Self::Constant(factor) => *factor,
            Self::ConstantFromSecondRound(first_round_factor, factor) => {
                if round == 0 {
                    *first_round_factor
                } else {
                    *factor
                }
            }
        }
    }

    /// Checks the validity of the folding factor against the number of variables.
    pub const fn check_validity(&self, num_variables: usize) -> Result<(), FoldingFactorError> {
        match self {
            Self::Constant(factor) => {
                if *factor > num_variables {
                    // A folding factor cannot be greater than the number of available variables.
                    Err(FoldingFactorError::TooLarge(*factor, num_variables))
                } else if *factor == 0 {
                    // A folding factor of zero is invalid since folding must reduce variables.
                    Err(FoldingFactorError::ZeroFactor)
                } else {
                    Ok(())
                }
            }
            Self::ConstantFromSecondRound(first_round_factor, factor) => {
                if *first_round_factor > num_variables {
                    // The first round folding factor must not exceed the available variables.
                    Err(FoldingFactorError::TooLarge(
                        *first_round_factor,
                        num_variables,
                    ))
                } else if *factor > num_variables {
                    // Subsequent round folding factors must also not exceed the available
                    // variables.
                    Err(FoldingFactorError::TooLarge(*factor, num_variables))
                } else if *factor == 0 || *first_round_factor == 0 {
                    // Folding should occur at least once; zero is not valid.
                    Err(FoldingFactorError::ZeroFactor)
                } else {
                    Ok(())
                }
            }
        }
    }

    /// Computes the number of WHIR rounds and the number of rounds in the final sumcheck.
    pub fn compute_number_of_rounds(&self, num_variables: usize) -> (usize, usize) {
        match self {
            Self::Constant(factor) => {
                // The number of remaining variables that cannot be fully folded.
                let final_sumcheck_rounds = num_variables % factor;

                // Compute the number of WHIR rounds by subtracting the final sumcheck rounds
                // and dividing by the folding factor. The -1 accounts for the fact that the last
                // round does not require another folding.
                (
                    (num_variables - final_sumcheck_rounds) / factor - 1,
                    final_sumcheck_rounds,
                )
            }
            Self::ConstantFromSecondRound(first_round_factor, factor) => {
                // Compute the number of variables remaining after the first round.
                let nv_except_first_round = num_variables - *first_round_factor;

                // If the remaining variables are too few for a full folding round,
                // treat it as a single final sumcheck step.
                if nv_except_first_round < *factor {
                    return (0, nv_except_first_round);
                }

                // The number of remaining variables that cannot be fully folded.
                let final_sumcheck_rounds = nv_except_first_round % *factor;

                // Compute the number of WHIR rounds by dividing the remaining variables
                // (excluding the first round) by the folding factor.
                (
                    (nv_except_first_round - final_sumcheck_rounds) / factor,
                    final_sumcheck_rounds,
                )
            }
        }
    }

    /// Computes the total number of folding rounds over `n_rounds` iterations.
    pub fn total_number(&self, n_rounds: usize) -> usize {
        match self {
            Self::Constant(factor) => {
                // - Each round folds `factor` variables,
                // - There are `n_rounds + 1` iterations (including the original input size).
                factor * (n_rounds + 1)
            }
            Self::ConstantFromSecondRound(first_round_factor, factor) => {
                // - The first round folds `first_round_factor` variables,
                // - Subsequent rounds fold `factor` variables each.
                first_round_factor + factor * n_rounds
            }
        }
    }
}

/// Configuration parameters for WHIR proofs.
#[derive(Clone)]
pub struct WhirParameters<MerkleConfig, PowStrategy>
where
    MerkleConfig: Config,
{
    /// Whether the initial statement is included in the proof.
    pub initial_statement: bool,
    /// The logarithmic inverse rate for sampling.
    pub starting_log_inv_rate: usize,
    /// The folding factor strategy.
    pub folding_factor: FoldingFactor,
    /// The type of soundness guarantee.
    pub soundness_type: SoundnessType,
    /// The security level in bits.
    pub security_level: usize,
    /// The number of bits required for proof-of-work (PoW).
    pub pow_bits: usize,
    /// The folding optimization strategy.
    pub fold_optimisation: FoldType,
    /// Phantom type for PoW parameters.
    pub _pow_parameters: PhantomData<PowStrategy>,
    /// Parameters for hashing Merkle tree leaves.
    ///
    /// These define how individual leaves in the Merkle tree are hashed.
    pub leaf_hash_params: LeafParam<MerkleConfig>,
    /// Parameters for hashing inner nodes in the Merkle tree.
    ///
    /// These define the hashing function used when combining two child nodes into a parent node.
    pub two_to_one_params: TwoToOneParam<MerkleConfig>,
}

impl<MerkleConfig, PowStrategy> Display for WhirParameters<MerkleConfig, PowStrategy>
where
    MerkleConfig: Config,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "Targeting {}-bits of security with {}-bits of PoW - soundness: {:?}",
            self.security_level, self.pow_bits, self.soundness_type
        )?;
        writeln!(
            f,
            "Starting rate: 2^-{}, folding_factor: {:?}, fold_opt_type: {}",
            self.starting_log_inv_rate, self.folding_factor, self.fold_optimisation,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_max_pow() {
        // Basic cases
        assert_eq!(default_max_pow(10, 3), 10); // 10 + 3 - 3 = 10
        assert_eq!(default_max_pow(5, 2), 4); // 5 + 2 - 3 = 4

        // Edge cases
        assert_eq!(default_max_pow(1, 3), 1); // Smallest valid input
        assert_eq!(default_max_pow(0, 3), 0); // Zero variables (should not happen in practice)
    }

    #[test]
    fn test_soundness_type_display() {
        assert_eq!(SoundnessType::ProvableList.to_string(), "ProvableList");
        assert_eq!(SoundnessType::ConjectureList.to_string(), "ConjectureList");
        assert_eq!(SoundnessType::UniqueDecoding.to_string(), "UniqueDecoding");
    }

    #[test]
    fn test_soundness_type_from_str() {
        assert_eq!(
            SoundnessType::from_str("ProvableList"),
            Ok(SoundnessType::ProvableList)
        );
        assert_eq!(
            SoundnessType::from_str("ConjectureList"),
            Ok(SoundnessType::ConjectureList)
        );
        assert_eq!(
            SoundnessType::from_str("UniqueDecoding"),
            Ok(SoundnessType::UniqueDecoding)
        );

        // Invalid cases
        assert!(SoundnessType::from_str("InvalidType").is_err());
        assert!(SoundnessType::from_str("").is_err()); // Empty string
    }

    #[test]
    fn test_multivariate_parameters() {
        let params = MultivariateParameters::<u32>::new(5);
        assert_eq!(params.num_variables, 5);
        assert_eq!(params.to_string(), "Number of variables: 5");
    }

    #[test]
    fn test_fold_type_from_str() {
        assert_eq!(FoldType::from_str("Naive"), Ok(FoldType::Naive));
        assert_eq!(FoldType::from_str("ProverHelps"), Ok(FoldType::ProverHelps));

        // Invalid cases
        assert!(FoldType::from_str("Invalid").is_err());
        assert!(FoldType::from_str("").is_err()); // Empty string
    }

    #[test]
    fn test_folding_factor_at_round() {
        let factor = FoldingFactor::Constant(4);
        assert_eq!(factor.at_round(0), 4);
        assert_eq!(factor.at_round(5), 4);

        let variable_factor = FoldingFactor::ConstantFromSecondRound(3, 5);
        assert_eq!(variable_factor.at_round(0), 3); // First round uses 3
        assert_eq!(variable_factor.at_round(1), 5); // Subsequent rounds use 5
        assert_eq!(variable_factor.at_round(10), 5);
    }

    #[test]
    fn test_folding_factor_check_validity() {
        // Valid cases
        assert!(FoldingFactor::Constant(2).check_validity(4).is_ok());
        assert!(FoldingFactor::ConstantFromSecondRound(2, 3)
            .check_validity(5)
            .is_ok());

        // Invalid cases
        // Factor too large
        assert_eq!(
            FoldingFactor::Constant(5).check_validity(3),
            Err(FoldingFactorError::TooLarge(5, 3))
        );
        // Zero factor
        assert_eq!(
            FoldingFactor::Constant(0).check_validity(3),
            Err(FoldingFactorError::ZeroFactor)
        );
        // First round factor too large
        assert_eq!(
            FoldingFactor::ConstantFromSecondRound(4, 2).check_validity(3),
            Err(FoldingFactorError::TooLarge(4, 3))
        );
        // Second round factor too large
        assert_eq!(
            FoldingFactor::ConstantFromSecondRound(2, 5).check_validity(4),
            Err(FoldingFactorError::TooLarge(5, 4))
        );
        // First round zero
        assert_eq!(
            FoldingFactor::ConstantFromSecondRound(0, 3).check_validity(4),
            Err(FoldingFactorError::ZeroFactor)
        );
    }

    #[test]
    fn test_compute_number_of_rounds() {
        let factor = FoldingFactor::Constant(2);
        assert_eq!(factor.compute_number_of_rounds(6), (2, 0)); // 6 - 2 rounds, no remainder
        assert_eq!(factor.compute_number_of_rounds(7), (2, 1)); // 7 - 2 rounds, 1 remainder

        let variable_factor = FoldingFactor::ConstantFromSecondRound(3, 2);
        assert_eq!(variable_factor.compute_number_of_rounds(7), (2, 0)); // 7 variables, first round uses 3, then 2 per round
        assert_eq!(variable_factor.compute_number_of_rounds(8), (2, 1)); // 8 variables, remainder 1
    }

    #[test]
    fn test_total_number() {
        let factor = FoldingFactor::Constant(2);
        assert_eq!(factor.total_number(3), 8); // 2 * (3 + 1)

        let variable_factor = FoldingFactor::ConstantFromSecondRound(3, 2);
        assert_eq!(variable_factor.total_number(3), 9); // 3 + 2 * 3
    }
}
