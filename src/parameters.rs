use std::fmt::{Debug, Display};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::engines::EngineId;

/// Computes the default maximum proof-of-work (PoW) bits.
///
/// This function determines the PoW security level based on the number of variables
/// and the logarithmic inverse rate.
pub const fn default_max_pow(num_variables: usize, log_inv_rate: usize) -> usize {
    num_variables + log_inv_rate - 3
}

/// Errors that can occur when validating a folding factor.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum FoldingFactorError {
    /// The folding factor is larger than the number of variables.
    #[error(
        "Folding factor {0} is greater than the number of variables {1}. Vector too small, just send it directly."
    )]
    TooLarge(usize, usize),

    /// The folding factor cannot be zero.
    #[error("Folding factor shouldn't be zero.")]
    ZeroFactor,
}

/// Computes the number of WHIR rounds and the number of rounds in the final sumcheck.
pub fn compute_number_of_rounds(
    initial_folding_factor: usize,
    folding_factor: usize,
    num_variables: usize,
) -> (usize, usize) {
    // Compute the number of variables remaining after the first round.
    let nv_except_first_round = num_variables - initial_folding_factor;

    // If the remaining variables are too few for a full folding round,
    // treat it as a single final sumcheck step.
    if nv_except_first_round < folding_factor {
        return (0, nv_except_first_round);
    }

    // The number of remaining variables that cannot be fully folded.
    let final_sumcheck_rounds = nv_except_first_round % folding_factor;

    // Compute the number of WHIR rounds by dividing the remaining variables
    // (excluding the first round) by the folding factor.
    (
        (nv_except_first_round - final_sumcheck_rounds) / folding_factor,
        final_sumcheck_rounds,
    )
}

/// Computes the total number of folding rounds over `n_rounds` iterations.
pub fn total_number(
    initial_folding_factor: usize,
    folding_factor: usize,
    n_rounds: usize,
) -> usize {
    // - The first round folds `initial_folding_factor` variables,
    // - Subsequent rounds fold `folding_factor` variables each.
    initial_folding_factor + folding_factor * n_rounds
}

/// Configuration parameters for WHIR proofs.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProtocolParameters {
    /// Wheter to require unique decoding.
    pub unique_decoding: bool,
    /// The logarithmic inverse rate for sampling.
    pub starting_log_inv_rate: usize,
    /// Folding factor for the initial round.
    pub initial_folding_factor: usize,
    /// Folding factor for rounds after the initial round.
    pub folding_factor: usize,
    /// The security level in bits.
    pub security_level: usize,
    /// The number of bits required for proof-of-work (PoW).
    pub pow_bits: usize,
    /// Number of vectors committed in the batch.
    pub batch_size: usize,
    /// Hash function identifier.
    pub hash_id: EngineId,
}

impl ProtocolParameters {
    /// Computes the number of WHIR rounds and the number of rounds in the final sumcheck.
    pub fn compute_number_of_rounds(&self, num_variables: usize) -> (usize, usize) {
        compute_number_of_rounds(
            self.initial_folding_factor,
            self.folding_factor,
            num_variables,
        )
    }

    /// Computes the total number of folded variables after `n_rounds`.
    pub fn total_number(&self, n_rounds: usize) -> usize {
        total_number(self.initial_folding_factor, self.folding_factor, n_rounds)
    }
}

impl Display for ProtocolParameters {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "Targeting {}-bits of security with {}-bits of PoW using {} decoding",
            self.security_level,
            self.pow_bits,
            if self.unique_decoding {
                "unique"
            } else {
                "list"
            }
        )?;
        writeln!(
            f,
            "Starting rate: 2^-{}, initial_folding_factor: {}, folding_factor: {}",
            self.starting_log_inv_rate, self.initial_folding_factor, self.folding_factor,
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
    fn test_compute_number_of_rounds() {
        assert_eq!(compute_number_of_rounds(2, 2, 6), (2, 0)); // 6 - 2 rounds, no remainder
        assert_eq!(compute_number_of_rounds(2, 2, 7), (2, 1)); // 7 - 2 rounds, 1 remainder

        assert_eq!(compute_number_of_rounds(3, 2, 7), (2, 0)); // 7 variables, first round uses 3, then 2 per round
        assert_eq!(compute_number_of_rounds(3, 2, 8), (2, 1)); // 8 variables, remainder 1
    }

    #[test]
    fn test_total_number() {
        assert_eq!(total_number(2, 2, 3), 8); // 2 * (3 + 1)
        assert_eq!(total_number(3, 2, 3), 9); // 3 + 2 * 3
    }
}
