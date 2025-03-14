use std::{fmt::Display, marker::PhantomData, str::FromStr};

use ark_crypto_primitives::merkle_tree::{Config, LeafParam, TwoToOneParam};
use serde::Serialize;

pub const fn default_max_pow(num_variables: usize, log_inv_rate: usize) -> usize {
    num_variables + log_inv_rate - 3
}

#[derive(Debug, Clone, Copy, Serialize)]
pub enum SoundnessType {
    UniqueDecoding,
    ProvableList,
    ConjectureList,
}

impl Display for SoundnessType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match &self {
                Self::ProvableList => "ProvableList",
                Self::ConjectureList => "ConjectureList",
                Self::UniqueDecoding => "UniqueDecoding",
            }
        )
    }
}

impl FromStr for SoundnessType {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s == "ProvableList" {
            Ok(Self::ProvableList)
        } else if s == "ConjectureList" {
            Ok(Self::ConjectureList)
        } else if s == "UniqueDecoding" {
            Ok(Self::UniqueDecoding)
        } else {
            Err(format!("Invalid soundness specification: {s}"))
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MultivariateParameters<F> {
    pub num_variables: usize,
    _field: PhantomData<F>,
}

impl<F> MultivariateParameters<F> {
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

#[derive(Debug, Clone, Copy)]
pub enum FoldType {
    Naive,
    ProverHelps,
}

impl FromStr for FoldType {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s == "Naive" {
            Ok(Self::Naive)
        } else if s == "ProverHelps" {
            Ok(Self::ProverHelps)
        } else {
            Err(format!("Invalid fold type specification: {s}"))
        }
    }
}

impl Display for FoldType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Self::Naive => "Naive",
                Self::ProverHelps => "ProverHelps",
            }
        )
    }
}

#[derive(Debug, Clone, Copy)]
pub enum FoldingFactor {
    Constant(usize),                       // Use the same folding factor for all rounds
    ConstantFromSecondRound(usize, usize), // Use the same folding factor for all rounds, but the first round uses a different folding factor
}

impl FoldingFactor {
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

    pub fn check_validity(&self, num_variables: usize) -> Result<(), String> {
        match self {
            Self::Constant(factor) => {
                if *factor > num_variables {
                    Err(format!(
                        "Folding factor {factor} is greater than the number of variables {num_variables}. Polynomial too small, just send it directly."
                    ))
                } else if *factor == 0 {
                    // We should at least fold some time
                    Err("Folding factor shouldn't be zero.".to_string())
                } else {
                    Ok(())
                }
            }
            Self::ConstantFromSecondRound(first_round_factor, factor) => {
                if *first_round_factor > num_variables {
                    Err(format!(
                        "First round folding factor {first_round_factor} is greater than the number of variables {num_variables}. Polynomial too small, just send it directly."
                    ))
                } else if *factor > num_variables {
                    Err(format!(
                        "Folding factor {factor} is greater than the number of variables {num_variables}. Polynomial too small, just send it directly."
                    ))
                } else if *factor == 0 || *first_round_factor == 0 {
                    // We should at least fold some time
                    Err("Folding factor shouldn't be zero.".to_string())
                } else {
                    Ok(())
                }
            }
        }
    }

    /// Compute the number of WHIR rounds and the number of rounds in the final
    /// sumcheck.
    pub fn compute_number_of_rounds(&self, num_variables: usize) -> (usize, usize) {
        match self {
            Self::Constant(factor) => {
                // It's checked that factor > 0 and factor <= num_variables
                let final_sumcheck_rounds = num_variables % factor;
                (
                    (num_variables - final_sumcheck_rounds) / factor - 1,
                    final_sumcheck_rounds,
                )
            }
            Self::ConstantFromSecondRound(first_round_factor, factor) => {
                let nv_except_first_round = num_variables - *first_round_factor;
                if nv_except_first_round < *factor {
                    // This case is equivalent to Constant(first_round_factor)
                    return (0, nv_except_first_round);
                }
                let final_sumcheck_rounds = nv_except_first_round % *factor;
                (
                    // No need to minus 1 because the initial round is already
                    // excepted out
                    (nv_except_first_round - final_sumcheck_rounds) / factor,
                    final_sumcheck_rounds,
                )
            }
        }
    }

    /// Compute folding_factor(0) + ... + folding_factor(n_rounds)
    pub fn total_number(&self, n_rounds: usize) -> usize {
        match self {
            Self::Constant(factor) => {
                // It's checked that factor > 0 and factor <= num_variables
                factor * (n_rounds + 1)
            }
            Self::ConstantFromSecondRound(first_round_factor, factor) => {
                first_round_factor + factor * n_rounds
            }
        }
    }
}

#[derive(Clone)]
pub struct WhirParameters<MerkleConfig, PowStrategy>
where
    MerkleConfig: Config,
{
    pub initial_statement: bool,
    pub starting_log_inv_rate: usize,
    pub folding_factor: FoldingFactor,
    pub soundness_type: SoundnessType,
    pub security_level: usize,
    pub pow_bits: usize,

    pub fold_optimisation: FoldType,

    // PoW parameters
    pub _pow_parameters: PhantomData<PowStrategy>,

    // Merkle tree parameters
    pub leaf_hash_params: LeafParam<MerkleConfig>,
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
