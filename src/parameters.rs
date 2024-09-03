use std::{fmt::Display, marker::PhantomData, str::FromStr};

use ark_crypto_primitives::merkle_tree::{Config, LeafParam, TwoToOneParam};
use serde::Serialize;

// Used to select how much PoW is acceptable.
// To be checked later on after opts are done.
pub const POW_FACTOR: usize = 1;

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
                SoundnessType::ProvableList => "ProvableList",
                SoundnessType::ConjectureList => "ConjectureList",
                SoundnessType::UniqueDecoding => "UniqueDecoding",
            }
        )
    }
}

impl FromStr for SoundnessType {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s == "ProvableList" {
            Ok(SoundnessType::ProvableList)
        } else if s == "ConjectureList" {
            Ok(SoundnessType::ConjectureList)
        } else if s == "UniqueDecoding" {
            Ok(SoundnessType::UniqueDecoding)
        } else {
            Err(format!("Invalid soundness specification: {}", s))
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MultivariateParameters<F> {
    pub(crate) num_variables: usize,
    _field: PhantomData<F>,
}

impl<F> MultivariateParameters<F> {
    pub fn new(num_variables: usize) -> Self {
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
            Ok(FoldType::Naive)
        } else if s == "ProverHelps" {
            Ok(FoldType::ProverHelps)
        } else {
            Err(format!("Invalid fold type specification: {}", s))
        }
    }
}

impl Display for FoldType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                FoldType::Naive => "Naive",
                FoldType::ProverHelps => "ProverHelps",
            }
        )
    }
}

#[derive(Clone)]
pub struct WhirParameters<MerkleConfig>
where
    MerkleConfig: Config,
{
    pub starting_log_inv_rate: usize,
    pub folding_factor: usize,
    pub soundness_type: SoundnessType,
    pub security_level: usize,
    pub pow_bits: usize,

    pub fold_optimisation: FoldType,

    // Merkle tree parameters
    pub leaf_hash_params: LeafParam<MerkleConfig>,
    pub two_to_one_params: TwoToOneParam<MerkleConfig>,
}

impl<MerkleConfig> Display for WhirParameters<MerkleConfig>
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
            "Starting rate: 2^-{}, folding_factor: {}, fold_opt_type: {}",
            self.starting_log_inv_rate, self.folding_factor, self.fold_optimisation,
        )
    }
}
