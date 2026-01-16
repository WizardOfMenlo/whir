use core::panic;
use std::{
    f64::consts::LOG2_10,
    fmt::{Debug, Display},
    sync::Arc,
};

use ark_crypto_primitives::merkle_tree::{Config, LeafParam, TwoToOneParam};
use ark_ff::FftField;
use ark_poly::EvaluationDomain;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use serde::{Deserialize, Serialize};

use crate::{
    crypto::fields::FieldWithSize,
    domain::Domain,
    ntt::{RSDefault, ReedSolomon},
    parameters::{
        DeduplicationStrategy, FoldingFactor, MerkleProofStrategy, MultivariateParameters,
        ProtocolParameters, SoundnessType,
    },
    utils::{ark_eq, f64_eq_abs},
};
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound = r#"
    LeafParam<MerkleConfig>: CanonicalSerialize + CanonicalDeserialize,
    TwoToOneParam<MerkleConfig>: CanonicalSerialize + CanonicalDeserialize
"#)]
pub struct WhirConfig<F, MerkleConfig>
where
    F: FftField,
    MerkleConfig: Config,
{
    pub mv_parameters: MultivariateParameters<F>,
    pub soundness_type: SoundnessType,
    pub security_level: usize,
    pub max_pow_bits: usize,

    pub committment_ood_samples: usize,
    // The WHIR protocol can prove either:
    // 1. The commitment is a valid low degree polynomial. In that case, the
    //    initial statement is set to false.
    // 2. The commitment is a valid folded polynomial, and an additional
    //    polynomial evaluation statement. In that case, the initial statement
    //    is set to true.
    pub initial_statement: bool,
    pub starting_domain: Domain<F>,
    pub starting_log_inv_rate: usize,
    pub starting_folding_pow_bits: f64,

    pub folding_factor: FoldingFactor,
    pub round_parameters: Vec<RoundConfig<F>>,

    pub final_queries: usize,
    pub final_pow_bits: f64,
    pub final_log_inv_rate: usize,
    pub final_sumcheck_rounds: usize,
    pub final_folding_pow_bits: f64,

    // Strategy to decide on the deduplication of challenges
    // (Used for recursive proving where constant length transcript is needed)
    pub deduplication_strategy: DeduplicationStrategy,

    // Merkle tree parameters
    #[serde(with = "crate::ark_serde")]
    pub leaf_hash_params: LeafParam<MerkleConfig>,
    #[serde(with = "crate::ark_serde")]
    pub two_to_one_params: TwoToOneParam<MerkleConfig>,

    pub merkle_proof_strategy: MerkleProofStrategy,

    // Batch size
    pub batch_size: usize,

    // Reed Solomon vtable
    #[serde(skip, default = "default_rs")]
    pub reed_solomon: Arc<dyn ReedSolomon<F>>,
    #[serde(skip, default = "default_rs")]
    pub basefield_reed_solomon: Arc<dyn ReedSolomon<F::BasePrimeField>>,
}

fn default_rs<F: FftField>() -> Arc<dyn ReedSolomon<F>> {
    Arc::new(RSDefault)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound = "F: CanonicalSerialize + CanonicalDeserialize")]
pub struct RoundConfig<F>
where
    F: FftField,
{
    pub pow_bits: f64,
    pub folding_pow_bits: f64,
    pub num_queries: usize,
    pub ood_samples: usize,
    pub log_inv_rate: usize,
    pub num_variables: usize,
    pub folding_factor: usize,
    pub domain_size: usize,
    #[serde(with = "crate::ark_serde")]
    pub domain_gen: F,
    #[serde(with = "crate::ark_serde")]
    pub domain_gen_inv: F,
    #[serde(with = "crate::ark_serde")]
    pub exp_domain_gen: F,
}

impl<F, MerkleConfig> WhirConfig<F, MerkleConfig>
where
    F: FftField + FieldWithSize,
    MerkleConfig: Config,
{
    #[allow(clippy::too_many_lines)]
    pub fn new(
        reed_solomon: Arc<dyn ReedSolomon<F>>,
        basefield_reed_solomon: Arc<dyn ReedSolomon<F::BasePrimeField>>,
        mv_parameters: MultivariateParameters<F>,
        whir_parameters: ProtocolParameters<MerkleConfig>,
    ) -> Self {
        whir_parameters
            .folding_factor
            .check_validity(mv_parameters.num_variables)
            .unwrap();

        let protocol_security_level = whir_parameters
            .security_level
            .saturating_sub(whir_parameters.pow_bits);
        let field_size_bits = F::field_size_in_bits();
        let mut log_inv_rate = whir_parameters.starting_log_inv_rate;
        let mut num_variables = mv_parameters.num_variables;

        let starting_domain = Domain::new(1 << mv_parameters.num_variables, log_inv_rate)
            .expect("Should have found an appropriate domain - check Field 2 adicity?");

        let mut domain_size = starting_domain.size();
        let mut domain_gen: F = starting_domain.backing_domain.group_gen();
        let mut domain_gen_inv = starting_domain.backing_domain.group_gen_inv();
        let mut exp_domain_gen = domain_gen.pow([1 << whir_parameters.folding_factor.at_round(0)]);

        let (num_rounds, final_sumcheck_rounds) = whir_parameters
            .folding_factor
            .compute_number_of_rounds(mv_parameters.num_variables);

        let log_eta_start = Self::log_eta(whir_parameters.soundness_type, log_inv_rate);

        let commitment_ood_samples = if whir_parameters.initial_statement {
            Self::ood_samples(
                whir_parameters.security_level,
                whir_parameters.soundness_type,
                num_variables,
                log_inv_rate,
                log_eta_start,
                field_size_bits,
            )
        } else {
            0
        };

        let starting_folding_pow_bits = if whir_parameters.initial_statement {
            Self::folding_pow_bits(
                whir_parameters.security_level,
                whir_parameters.soundness_type,
                field_size_bits,
                num_variables,
                log_inv_rate,
                log_eta_start,
            )
        } else {
            {
                let prox_gaps_error = Self::rbr_soundness_fold_prox_gaps(
                    whir_parameters.soundness_type,
                    field_size_bits,
                    num_variables,
                    log_inv_rate,
                    log_eta_start,
                ) + (whir_parameters.folding_factor.at_round(0) as f64)
                    .log2();
                (whir_parameters.security_level as f64 - prox_gaps_error).max(0.0)
            }
        };

        let mut round_parameters = Vec::with_capacity(num_rounds);
        num_variables -= whir_parameters.folding_factor.at_round(0);
        for round in 0..num_rounds {
            // Queries are set w.r.t. to old rate, while the rest to the new rate
            let next_rate = log_inv_rate + (whir_parameters.folding_factor.at_round(round) - 1);

            let log_next_eta = Self::log_eta(whir_parameters.soundness_type, next_rate);

            let num_queries = Self::queries(
                whir_parameters.soundness_type,
                protocol_security_level,
                log_inv_rate,
            );

            let ood_samples = Self::ood_samples(
                whir_parameters.security_level,
                whir_parameters.soundness_type,
                num_variables,
                next_rate,
                log_next_eta,
                field_size_bits,
            );

            let query_error =
                Self::rbr_queries(whir_parameters.soundness_type, log_inv_rate, num_queries);

            let combination_error = Self::rbr_soundness_queries_combination(
                whir_parameters.soundness_type,
                field_size_bits,
                num_variables,
                next_rate,
                log_next_eta,
                ood_samples,
                num_queries,
            );

            let pow_bits = 0_f64
                .max(whir_parameters.security_level as f64 - (query_error.min(combination_error)));

            let folding_pow_bits = Self::folding_pow_bits(
                whir_parameters.security_level,
                whir_parameters.soundness_type,
                field_size_bits,
                num_variables,
                next_rate,
                log_next_eta,
            );

            round_parameters.push(RoundConfig {
                pow_bits,
                folding_pow_bits,
                num_queries,
                ood_samples,
                log_inv_rate,
                num_variables,
                folding_factor: whir_parameters.folding_factor.at_round(round),
                domain_size,
                domain_gen,
                domain_gen_inv,
                exp_domain_gen,
            });

            num_variables -= whir_parameters.folding_factor.at_round(round + 1);
            log_inv_rate = next_rate;
            domain_size /= 2;
            domain_gen = domain_gen.square();
            domain_gen_inv = domain_gen_inv.square();
            exp_domain_gen =
                domain_gen.pow([1 << whir_parameters.folding_factor.at_round(round + 1)]);
        }

        let final_queries = Self::queries(
            whir_parameters.soundness_type,
            protocol_security_level,
            log_inv_rate,
        );

        let final_pow_bits = 0_f64.max(
            whir_parameters.security_level as f64
                - Self::rbr_queries(whir_parameters.soundness_type, log_inv_rate, final_queries),
        );

        let final_folding_pow_bits =
            0_f64.max(whir_parameters.security_level as f64 - (field_size_bits - 1) as f64);

        Self {
            security_level: whir_parameters.security_level,
            max_pow_bits: whir_parameters.pow_bits,
            initial_statement: whir_parameters.initial_statement,
            committment_ood_samples: commitment_ood_samples,
            mv_parameters,
            starting_domain,
            soundness_type: whir_parameters.soundness_type,
            starting_log_inv_rate: whir_parameters.starting_log_inv_rate,
            starting_folding_pow_bits,
            folding_factor: whir_parameters.folding_factor,
            round_parameters,
            final_queries,
            final_pow_bits,
            final_sumcheck_rounds,
            final_folding_pow_bits,
            deduplication_strategy: whir_parameters.deduplication_strategy,
            final_log_inv_rate: log_inv_rate,
            leaf_hash_params: whir_parameters.leaf_hash_params,
            two_to_one_params: whir_parameters.two_to_one_params,
            merkle_proof_strategy: whir_parameters.merkle_proof_strategy,
            batch_size: whir_parameters.batch_size,
            reed_solomon,
            basefield_reed_solomon,
        }
    }

    pub fn n_rounds(&self) -> usize {
        self.round_parameters.len()
    }

    pub fn check_pow_bits(&self) -> bool {
        let max_bits = self.max_pow_bits as f64;

        // Check the main pow bits values
        if self.starting_folding_pow_bits > max_bits
            || self.final_pow_bits > max_bits
            || self.final_folding_pow_bits > max_bits
        {
            return false;
        }

        // Check all round parameters
        self.round_parameters
            .iter()
            .all(|r| r.pow_bits <= max_bits && r.folding_pow_bits <= max_bits)
    }

    pub const fn log_eta(soundness_type: SoundnessType, log_inv_rate: usize) -> f64 {
        // Ask me how I did this? At the time, only God and I knew. Now only God knows
        match soundness_type {
            SoundnessType::ProvableList => -(0.5 * log_inv_rate as f64 + LOG2_10 + 1.),
            SoundnessType::UniqueDecoding => 0.,
            SoundnessType::ConjectureList => -(log_inv_rate as f64 + 1.),
        }
    }

    pub const fn list_size_bits(
        soundness_type: SoundnessType,
        num_variables: usize,
        log_inv_rate: usize,
        log_eta: f64,
    ) -> f64 {
        match soundness_type {
            SoundnessType::ConjectureList => (num_variables + log_inv_rate) as f64 - log_eta,
            SoundnessType::ProvableList => {
                let log_inv_sqrt_rate: f64 = log_inv_rate as f64 / 2.;
                log_inv_sqrt_rate - (1. + log_eta)
            }
            SoundnessType::UniqueDecoding => 0.0,
        }
    }

    pub const fn rbr_ood_sample(
        soundness_type: SoundnessType,
        num_variables: usize,
        log_inv_rate: usize,
        log_eta: f64,
        field_size_bits: usize,
        ood_samples: usize,
    ) -> f64 {
        let list_size_bits =
            Self::list_size_bits(soundness_type, num_variables, log_inv_rate, log_eta);

        let error = 2. * list_size_bits + (num_variables * ood_samples) as f64;
        (ood_samples * field_size_bits) as f64 + 1. - error
    }

    pub fn ood_samples(
        security_level: usize, // We don't do PoW for OOD
        soundness_type: SoundnessType,
        num_variables: usize,
        log_inv_rate: usize,
        log_eta: f64,
        field_size_bits: usize,
    ) -> usize {
        match soundness_type {
            SoundnessType::UniqueDecoding => 0,
            _ => (1..64)
                .find(|&ood_samples| {
                    Self::rbr_ood_sample(
                        soundness_type,
                        num_variables,
                        log_inv_rate,
                        log_eta,
                        field_size_bits,
                        ood_samples,
                    ) >= security_level as f64
                })
                .unwrap_or_else(|| panic!("Could not find an appropriate number of OOD samples")),
        }
    }

    // Compute the proximity gaps term of the fold
    pub const fn rbr_soundness_fold_prox_gaps(
        soundness_type: SoundnessType,
        field_size_bits: usize,
        num_variables: usize,
        log_inv_rate: usize,
        log_eta: f64,
    ) -> f64 {
        // Recall, at each round we are only folding by two at a time
        let error = match soundness_type {
            SoundnessType::ConjectureList => (num_variables + log_inv_rate) as f64 - log_eta,
            SoundnessType::ProvableList => {
                LOG2_10 + 3.5 * log_inv_rate as f64 + 2. * num_variables as f64
            }
            SoundnessType::UniqueDecoding => (num_variables + log_inv_rate) as f64,
        };

        field_size_bits as f64 - error
    }

    pub const fn rbr_soundness_fold_sumcheck(
        soundness_type: SoundnessType,
        field_size_bits: usize,
        num_variables: usize,
        log_inv_rate: usize,
        log_eta: f64,
    ) -> f64 {
        let list_size = Self::list_size_bits(soundness_type, num_variables, log_inv_rate, log_eta);

        field_size_bits as f64 - (list_size + 1.)
    }

    pub const fn folding_pow_bits(
        security_level: usize,
        soundness_type: SoundnessType,
        field_size_bits: usize,
        num_variables: usize,
        log_inv_rate: usize,
        log_eta: f64,
    ) -> f64 {
        let prox_gaps_error = Self::rbr_soundness_fold_prox_gaps(
            soundness_type,
            field_size_bits,
            num_variables,
            log_inv_rate,
            log_eta,
        );
        let sumcheck_error = Self::rbr_soundness_fold_sumcheck(
            soundness_type,
            field_size_bits,
            num_variables,
            log_inv_rate,
            log_eta,
        );

        let error = if prox_gaps_error < sumcheck_error {
            prox_gaps_error
        } else {
            sumcheck_error
        };

        let candidate = security_level as f64 - error;
        if candidate > 0_f64 {
            candidate
        } else {
            0_f64
        }
    }

    // Used to select the number of queries
    #[allow(clippy::cast_sign_loss)]
    pub fn queries(
        soundness_type: SoundnessType,
        protocol_security_level: usize,
        log_inv_rate: usize,
    ) -> usize {
        let num_queries_f = match soundness_type {
            SoundnessType::UniqueDecoding => {
                let rate = 1. / f64::from(1 << log_inv_rate);
                let denom = (0.5 * (1. + rate)).log2();

                -(protocol_security_level as f64) / denom
            }
            SoundnessType::ProvableList => {
                (2 * protocol_security_level) as f64 / log_inv_rate as f64
            }
            SoundnessType::ConjectureList => protocol_security_level as f64 / log_inv_rate as f64,
        };
        num_queries_f.ceil() as usize
    }

    // This is the bits of security of the query step
    pub fn rbr_queries(
        soundness_type: SoundnessType,
        log_inv_rate: usize,
        num_queries: usize,
    ) -> f64 {
        let num_queries = num_queries as f64;

        match soundness_type {
            SoundnessType::UniqueDecoding => {
                let rate = 1. / f64::from(1 << log_inv_rate);
                let denom = -(0.5 * (1. + rate)).log2();

                num_queries * denom
            }
            SoundnessType::ProvableList => num_queries * 0.5 * log_inv_rate as f64,
            SoundnessType::ConjectureList => num_queries * log_inv_rate as f64,
        }
    }

    pub fn rbr_soundness_queries_combination(
        soundness_type: SoundnessType,
        field_size_bits: usize,
        num_variables: usize,
        log_inv_rate: usize,
        log_eta: f64,
        ood_samples: usize,
        num_queries: usize,
    ) -> f64 {
        let list_size = Self::list_size_bits(soundness_type, num_variables, log_inv_rate, log_eta);

        let log_combination = ((ood_samples + num_queries) as f64).log2();

        field_size_bits as f64 - (log_combination + list_size + 1.)
    }

    /// Compute the synthetic or derived `RoundConfig` for the final phase.
    ///
    /// - If no folding rounds were configured, constructs a fallback config
    ///   based on the starting domain and folding factor.
    /// - If rounds were configured, derives the final config by adapting
    ///   the last round’s values for the final folding phase.
    ///
    /// This is used by the verifier when verifying the final polynomial,
    /// ensuring consistent challenge selection and STIR constraint handling.
    pub fn final_round_config(&self) -> RoundConfig<F> {
        if self.round_parameters.is_empty() {
            // Fallback: no folding rounds, use initial domain setup
            RoundConfig {
                num_variables: self.mv_parameters.num_variables - self.folding_factor.at_round(0),
                folding_factor: self.folding_factor.at_round(self.n_rounds()),
                num_queries: self.final_queries,
                pow_bits: self.final_pow_bits,
                domain_size: self.starting_domain.size(),
                domain_gen: self.starting_domain.backing_domain.group_gen(),
                domain_gen_inv: self.starting_domain.backing_domain.group_gen_inv(),
                exp_domain_gen: self
                    .starting_domain
                    .backing_domain
                    .group_gen()
                    .pow([1 << self.folding_factor.at_round(0)]),
                ood_samples: 0, // no OOD in synthetic final phase
                folding_pow_bits: self.final_folding_pow_bits,
                log_inv_rate: self.starting_log_inv_rate,
            }
        } else {
            // Derive final round config from last round, adjusted for next fold
            let last = self.round_parameters.last().unwrap();
            RoundConfig {
                num_variables: last.num_variables - self.folding_factor.at_round(self.n_rounds()),
                folding_factor: self.folding_factor.at_round(self.n_rounds()),
                num_queries: self.final_queries,
                pow_bits: self.final_pow_bits,
                domain_size: last.domain_size / 2,
                domain_gen: last.domain_gen.square(),
                domain_gen_inv: last.domain_gen_inv.square(),
                exp_domain_gen: last
                    .domain_gen
                    .square()
                    .pow([1 << self.folding_factor.at_round(self.n_rounds())]),
                ood_samples: last.ood_samples,
                folding_pow_bits: self.final_folding_pow_bits,
                log_inv_rate: last.log_inv_rate,
            }
        }
    }
}

/// Manual implementation to allow error in `f64` and handle ark types missing `PartialEq`.
impl<F, MerkleConfig> PartialEq for WhirConfig<F, MerkleConfig>
where
    F: FftField,
    MerkleConfig: Config,
{
    fn eq(&self, other: &Self) -> bool {
        self.mv_parameters == other.mv_parameters
            && self.soundness_type == other.soundness_type
            && self.security_level == other.security_level
            && self.max_pow_bits == other.max_pow_bits
            && f64_eq_abs(
                self.starting_folding_pow_bits,
                other.starting_folding_pow_bits,
                0.001,
            )
            && self.round_parameters == other.round_parameters
            && self.final_queries == other.final_queries
            && self.final_log_inv_rate == other.final_log_inv_rate
            && f64_eq_abs(self.final_pow_bits, other.final_pow_bits, 0.001)
            && f64_eq_abs(
                self.final_folding_pow_bits,
                other.final_folding_pow_bits,
                0.001,
            )
            && self.committment_ood_samples == other.committment_ood_samples
            && ark_eq(&self.leaf_hash_params, &other.leaf_hash_params)
            && ark_eq(&self.two_to_one_params, &other.two_to_one_params)
            && self.folding_factor == other.folding_factor
            && self.initial_statement == other.initial_statement
    }
}

impl<F> PartialEq for RoundConfig<F>
where
    F: FftField,
{
    fn eq(&self, other: &Self) -> bool {
        f64_eq_abs(self.pow_bits, other.pow_bits, 0.001)
            && f64_eq_abs(self.folding_pow_bits, other.folding_pow_bits, 0.001)
            && self.num_queries == other.num_queries
            && self.ood_samples == other.ood_samples
            && self.log_inv_rate == other.log_inv_rate
    }
}

/// Workaround for `PowStrategy` not implementing `Debug`.
/// TODO: Add Debug in spongefish (and other common traits).
impl<F, MerkleConfig> Debug for WhirConfig<F, MerkleConfig>
where
    F: FftField,
    MerkleConfig: Config,
    LeafParam<MerkleConfig>: Debug,
    TwoToOneParam<MerkleConfig>: Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WhirConfig")
            .field("mv_parameters", &self.mv_parameters)
            .field("soundness_type", &self.soundness_type)
            .field("security_level", &self.security_level)
            .field("max_pow_bits", &self.max_pow_bits)
            .field("committment_ood_samples", &self.committment_ood_samples)
            .field("initial_statement", &self.initial_statement)
            .field("starting_domain", &self.starting_domain)
            .field("starting_log_inv_rate", &self.starting_log_inv_rate)
            .field("starting_folding_pow_bits", &self.starting_folding_pow_bits)
            .field("round_parameters", &self.round_parameters)
            .field("final_queries", &self.final_queries)
            .field("final_pow_bits", &self.final_pow_bits)
            .field("final_log_inv_rate", &self.final_log_inv_rate)
            .field("final_sumcheck_rounds", &self.final_sumcheck_rounds)
            .field("final_folding_pow_bits", &self.final_folding_pow_bits)
            .field("folding_factor", &self.folding_factor)
            .field("leaf_hash_params", &self.leaf_hash_params)
            .field("two_to_one_params", &self.two_to_one_params)
            .field("batch_size", &self.batch_size)
            .field("deduplication_strategy", &self.deduplication_strategy)
            .field("merkle_proof_strategy", &self.merkle_proof_strategy)
            .finish_non_exhaustive()
    }
}

impl<F, MerkleConfig> Display for WhirConfig<F, MerkleConfig>
where
    F: FftField,
    MerkleConfig: Config,
{
    #[allow(clippy::too_many_lines)]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", &self.mv_parameters)?;
        writeln!(f, ", folding factor: {:?}", self.folding_factor)?;
        writeln!(
            f,
            "Security level: {} bits using {} security and {} bits of PoW",
            self.security_level, self.soundness_type, self.max_pow_bits
        )?;

        writeln!(
            f,
            "initial_folding_pow_bits: {}",
            self.starting_folding_pow_bits
        )?;
        for r in &self.round_parameters {
            write!(f, "{r}")?;
        }

        writeln!(
            f,
            "final_queries: {}, final_rate: 2^-{}, final_pow_bits: {}, final_folding_pow_bits: {}",
            self.final_queries,
            self.final_log_inv_rate,
            self.final_pow_bits,
            self.final_folding_pow_bits,
        )?;

        writeln!(f, "------------------------------------")?;
        writeln!(f, "Round by round soundness analysis:")?;
        writeln!(f, "------------------------------------")?;

        let field_size_bits = F::field_size_in_bits();
        let log_eta = Self::log_eta(self.soundness_type, self.starting_log_inv_rate);
        let mut num_variables = self.mv_parameters.num_variables;

        if self.committment_ood_samples > 0 {
            writeln!(
                f,
                "{:.1} bits -- OOD commitment",
                Self::rbr_ood_sample(
                    self.soundness_type,
                    num_variables,
                    self.starting_log_inv_rate,
                    log_eta,
                    field_size_bits,
                    self.committment_ood_samples
                )
            )?;
        }

        let prox_gaps_error = Self::rbr_soundness_fold_prox_gaps(
            self.soundness_type,
            field_size_bits,
            num_variables,
            self.starting_log_inv_rate,
            log_eta,
        );
        let sumcheck_error = Self::rbr_soundness_fold_sumcheck(
            self.soundness_type,
            field_size_bits,
            num_variables,
            self.starting_log_inv_rate,
            log_eta,
        );
        writeln!(
            f,
            "{:.1} bits -- (x{}) prox gaps: {:.1}, sumcheck: {:.1}, pow: {:.1}",
            prox_gaps_error.min(sumcheck_error) + self.starting_folding_pow_bits,
            self.folding_factor.at_round(0),
            prox_gaps_error,
            sumcheck_error,
            self.starting_folding_pow_bits,
        )?;

        num_variables -= self.folding_factor.at_round(0);

        for (round, r) in self.round_parameters.iter().enumerate() {
            let next_rate = r.log_inv_rate + (self.folding_factor.at_round(round) - 1);
            let log_eta = Self::log_eta(self.soundness_type, next_rate);

            if r.ood_samples > 0 {
                writeln!(
                    f,
                    "{:.1} bits -- OOD sample",
                    Self::rbr_ood_sample(
                        self.soundness_type,
                        num_variables,
                        next_rate,
                        log_eta,
                        field_size_bits,
                        r.ood_samples
                    )
                )?;
            }

            let query_error = Self::rbr_queries(self.soundness_type, r.log_inv_rate, r.num_queries);
            let combination_error = Self::rbr_soundness_queries_combination(
                self.soundness_type,
                field_size_bits,
                num_variables,
                next_rate,
                log_eta,
                r.ood_samples,
                r.num_queries,
            );
            writeln!(
                f,
                "{:.1} bits -- query error: {:.1}, combination: {:.1}, pow: {:.1}",
                query_error.min(combination_error) + r.pow_bits,
                query_error,
                combination_error,
                r.pow_bits,
            )?;

            let prox_gaps_error = Self::rbr_soundness_fold_prox_gaps(
                self.soundness_type,
                field_size_bits,
                num_variables,
                next_rate,
                log_eta,
            );
            let sumcheck_error = Self::rbr_soundness_fold_sumcheck(
                self.soundness_type,
                field_size_bits,
                num_variables,
                next_rate,
                log_eta,
            );

            writeln!(
                f,
                "{:.1} bits -- (x{}) prox gaps: {:.1}, sumcheck: {:.1}, pow: {:.1}",
                prox_gaps_error.min(sumcheck_error) + r.folding_pow_bits,
                self.folding_factor.at_round(round + 1),
                prox_gaps_error,
                sumcheck_error,
                r.folding_pow_bits,
            )?;

            num_variables -= self.folding_factor.at_round(round + 1);
        }

        let query_error = Self::rbr_queries(
            self.soundness_type,
            self.final_log_inv_rate,
            self.final_queries,
        );
        writeln!(
            f,
            "{:.1} bits -- query error: {:.1}, pow: {:.1}",
            query_error + self.final_pow_bits,
            query_error,
            self.final_pow_bits,
        )?;

        if self.final_sumcheck_rounds > 0 {
            let combination_error = field_size_bits as f64 - 1.;
            writeln!(
                f,
                "{:.1} bits -- (x{}) combination: {:.1}, pow: {:.1}",
                combination_error + self.final_pow_bits,
                self.final_sumcheck_rounds,
                combination_error,
                self.final_folding_pow_bits,
            )?;
        }

        Ok(())
    }
}

impl<F> Display for RoundConfig<F>
where
    F: FftField,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "Num_queries: {}, rate: 2^-{}, pow_bits: {}, ood_samples: {}, folding_pow: {}",
            self.num_queries,
            self.log_inv_rate,
            self.pow_bits,
            self.ood_samples,
            self.folding_pow_bits,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        crypto::{
            fields::{Field256, Field64},
            merkle_tree::keccak::KeccakMerkleTreeParams,
        },
        parameters::{DeduplicationStrategy, MerkleProofStrategy},
        utils::test_serde,
    };

    /// Generates default WHIR parameters
    fn default_whir_params<F: FftField>() -> ProtocolParameters<KeccakMerkleTreeParams<F>> {
        ProtocolParameters {
            initial_statement: true,
            security_level: 100,
            pow_bits: 20,
            folding_factor: FoldingFactor::ConstantFromSecondRound(4, 4),
            leaf_hash_params: (),
            two_to_one_params: (),
            soundness_type: SoundnessType::ConjectureList,
            starting_log_inv_rate: 1,
            batch_size: 1,
            deduplication_strategy: DeduplicationStrategy::Enabled,
            merkle_proof_strategy: MerkleProofStrategy::Compressed,
        }
    }

    #[test]
    fn test_whir_config_creation() {
        type MerkleConfig = KeccakMerkleTreeParams<Field64>;

        let params = default_whir_params::<Field64>();

        let mv_params = MultivariateParameters::<Field64>::new(10);
        let reed_solomon = Arc::new(RSDefault);
        let basefield_reed_solomon = reed_solomon.clone();
        let config = WhirConfig::<Field64, MerkleConfig>::new(
            reed_solomon,
            basefield_reed_solomon,
            mv_params,
            params,
        );

        assert_eq!(config.security_level, 100);
        assert_eq!(config.max_pow_bits, 20);
        assert_eq!(config.soundness_type, SoundnessType::ConjectureList);
        assert!(config.initial_statement);
    }

    #[test]
    fn test_whir_params_serde() {
        test_serde(&default_whir_params::<Field64>());
        test_serde(&default_whir_params::<Field256>());
    }

    #[test]
    fn test_whir_config_serde() {
        type MerkleConfig = KeccakMerkleTreeParams<Field64>;

        let params = default_whir_params::<Field64>();

        let mv_params = MultivariateParameters::<Field64>::new(10);
        let reed_solomon = Arc::new(RSDefault);
        let basefield_reed_solomon = reed_solomon.clone();
        let config = WhirConfig::<Field64, MerkleConfig>::new(
            reed_solomon,
            basefield_reed_solomon,
            mv_params,
            params,
        );

        test_serde(&config);
    }

    #[test]
    fn test_n_rounds() {
        type MerkleConfig = KeccakMerkleTreeParams<Field64>;

        let params = default_whir_params::<Field64>();
        let mv_params = MultivariateParameters::<Field64>::new(10);
        let reed_solomon = Arc::new(RSDefault);
        let basefield_reed_solomon = reed_solomon.clone();
        let config = WhirConfig::<Field64, MerkleConfig>::new(
            reed_solomon,
            basefield_reed_solomon,
            mv_params,
            params,
        );

        assert_eq!(config.n_rounds(), config.round_parameters.len());
    }

    #[test]
    fn test_folding_pow_bits() {
        let field_size_bits = 64;
        let soundness = SoundnessType::ConjectureList;

        let pow_bits = WhirConfig::<Field64, KeccakMerkleTreeParams<Field64>>::folding_pow_bits(
            100, // Security level
            soundness,
            field_size_bits,
            10,   // Number of variables
            5,    // Log inverse rate
            -3.0, // Log eta
        );

        // PoW bits should never be negative
        assert!(pow_bits >= 0.);
    }

    #[test]
    fn test_queries_unique_decoding() {
        let security_level = 100;
        let log_inv_rate = 5;

        let result = WhirConfig::<Field64, KeccakMerkleTreeParams<Field64>>::queries(
            SoundnessType::UniqueDecoding,
            security_level,
            log_inv_rate,
        );

        assert_eq!(result, 105);
    }

    #[test]
    fn test_queries_provable_list() {
        let security_level = 128;
        let log_inv_rate = 8;

        let result = WhirConfig::<Field64, KeccakMerkleTreeParams<Field64>>::queries(
            SoundnessType::ProvableList,
            security_level,
            log_inv_rate,
        );

        assert_eq!(result, 32);
    }

    #[test]
    fn test_queries_conjecture_list() {
        let security_level = 256;
        let log_inv_rate = 16;

        let result = WhirConfig::<Field64, KeccakMerkleTreeParams<Field64>>::queries(
            SoundnessType::ConjectureList,
            security_level,
            log_inv_rate,
        );

        assert_eq!(result, 16);
    }

    #[test]
    fn test_rbr_queries_unique_decoding() {
        let log_inv_rate = 5; // log_inv_rate = 5
        let num_queries = 10; // Number of queries

        let result = WhirConfig::<Field64, KeccakMerkleTreeParams<Field64>>::rbr_queries(
            SoundnessType::UniqueDecoding,
            log_inv_rate,
            num_queries,
        );

        assert!((result - 9.556_058_806_415_466).abs() < 1e-6);
    }

    #[test]
    fn test_rbr_queries_provable_list() {
        let log_inv_rate = 8; // log_inv_rate = 8
        let num_queries = 16; // Number of queries

        let result = WhirConfig::<Field64, KeccakMerkleTreeParams<Field64>>::rbr_queries(
            SoundnessType::ProvableList,
            log_inv_rate,
            num_queries,
        );

        assert!((result - 64.0) < 1e-6);
    }

    #[test]
    fn test_rbr_queries_conjecture_list() {
        let log_inv_rate = 4; // log_inv_rate = 4
        let num_queries = 20; // Number of queries

        let result = WhirConfig::<Field64, KeccakMerkleTreeParams<Field64>>::rbr_queries(
            SoundnessType::ConjectureList,
            log_inv_rate,
            num_queries,
        );

        assert!((result - 80.) < 1e-6);
    }

    #[test]
    fn test_check_pow_bits_within_limits() {
        type MerkleConfig = KeccakMerkleTreeParams<Field64>;

        let params = default_whir_params::<Field64>();
        let mv_params = MultivariateParameters::<Field64>::new(10);
        let reed_solomon = Arc::new(RSDefault);
        let basefield_reed_solomon = reed_solomon.clone();
        let mut config = WhirConfig::<Field64, MerkleConfig>::new(
            reed_solomon,
            basefield_reed_solomon,
            mv_params,
            params,
        );

        // Set all values within limits
        config.max_pow_bits = 20;
        config.starting_folding_pow_bits = 15.0;
        config.final_pow_bits = 18.0;
        config.final_folding_pow_bits = 19.5;

        // Ensure all rounds are within limits
        config.round_parameters = vec![
            RoundConfig {
                pow_bits: 17.0,
                folding_pow_bits: 19.0,
                num_queries: 5,
                ood_samples: 2,
                log_inv_rate: 3,
                num_variables: 10,
                folding_factor: 2,
                domain_size: 10,
                domain_gen: Field64::from(2),
                domain_gen_inv: Field64::from(2),
                exp_domain_gen: Field64::from(2),
            },
            RoundConfig {
                pow_bits: 18.0,
                folding_pow_bits: 19.5,
                num_queries: 6,
                ood_samples: 2,
                log_inv_rate: 4,
                num_variables: 10,
                folding_factor: 2,
                domain_size: 10,
                domain_gen: Field64::from(2),
                domain_gen_inv: Field64::from(2),
                exp_domain_gen: Field64::from(2),
            },
        ];

        assert!(
            config.check_pow_bits(),
            "All values are within limits, check_pow_bits should return true."
        );
    }

    #[test]
    fn test_check_pow_bits_starting_folding_exceeds() {
        type MerkleConfig = KeccakMerkleTreeParams<Field64>;

        let params = default_whir_params::<Field64>();
        let mv_params = MultivariateParameters::<Field64>::new(10);
        let reed_solomon = Arc::new(RSDefault);
        let basefield_reed_solomon = reed_solomon.clone();
        let mut config = WhirConfig::<Field64, MerkleConfig>::new(
            reed_solomon,
            basefield_reed_solomon,
            mv_params,
            params,
        );

        config.max_pow_bits = 20;
        config.starting_folding_pow_bits = 21.0; // Exceeds max_pow_bits
        config.final_pow_bits = 18.0;
        config.final_folding_pow_bits = 19.5;

        assert!(
            !config.check_pow_bits(),
            "Starting folding pow bits exceeds max_pow_bits, should return false."
        );
    }

    #[test]
    fn test_check_pow_bits_final_pow_exceeds() {
        type MerkleConfig = KeccakMerkleTreeParams<Field64>;

        let params = default_whir_params::<Field64>();
        let mv_params = MultivariateParameters::<Field64>::new(10);
        let reed_solomon = Arc::new(RSDefault);
        let basefield_reed_solomon = reed_solomon.clone();
        let mut config = WhirConfig::<Field64, MerkleConfig>::new(
            reed_solomon,
            basefield_reed_solomon,
            mv_params,
            params,
        );

        config.max_pow_bits = 20;
        config.starting_folding_pow_bits = 15.0;
        config.final_pow_bits = 21.0; // Exceeds max_pow_bits
        config.final_folding_pow_bits = 19.5;

        assert!(
            !config.check_pow_bits(),
            "Final pow bits exceeds max_pow_bits, should return false."
        );
    }

    #[test]
    fn test_check_pow_bits_round_pow_exceeds() {
        type MerkleConfig = KeccakMerkleTreeParams<Field64>;

        let params = default_whir_params::<Field64>();
        let mv_params = MultivariateParameters::<Field64>::new(10);
        let reed_solomon = Arc::new(RSDefault);
        let basefield_reed_solomon = reed_solomon.clone();
        let mut config = WhirConfig::<Field64, MerkleConfig>::new(
            reed_solomon,
            basefield_reed_solomon,
            mv_params,
            params,
        );

        config.max_pow_bits = 20;
        config.starting_folding_pow_bits = 15.0;
        config.final_pow_bits = 18.0;
        config.final_folding_pow_bits = 19.5;

        // One round's pow_bits exceeds limit
        config.round_parameters = vec![RoundConfig {
            pow_bits: 21.0, // Exceeds max_pow_bits
            folding_pow_bits: 19.0,
            num_queries: 5,
            ood_samples: 2,
            log_inv_rate: 3,
            num_variables: 10,
            folding_factor: 2,
            domain_size: 10,
            domain_gen: Field64::from(2),
            domain_gen_inv: Field64::from(2),
            exp_domain_gen: Field64::from(2),
        }];

        assert!(
            !config.check_pow_bits(),
            "A round has pow_bits exceeding max_pow_bits, should return false."
        );
    }

    #[test]
    fn test_check_pow_bits_round_folding_pow_exceeds() {
        type MerkleConfig = KeccakMerkleTreeParams<Field64>;

        let params = default_whir_params::<Field64>();
        let mv_params = MultivariateParameters::<Field64>::new(10);
        let reed_solomon = Arc::new(RSDefault);
        let basefield_reed_solomon = reed_solomon.clone();
        let mut config = WhirConfig::<Field64, MerkleConfig>::new(
            reed_solomon,
            basefield_reed_solomon,
            mv_params,
            params,
        );

        config.max_pow_bits = 20;
        config.starting_folding_pow_bits = 15.0;
        config.final_pow_bits = 18.0;
        config.final_folding_pow_bits = 19.5;

        // One round's folding_pow_bits exceeds limit
        config.round_parameters = vec![RoundConfig {
            pow_bits: 19.0,
            folding_pow_bits: 21.0, // Exceeds max_pow_bits
            num_queries: 5,
            ood_samples: 2,
            log_inv_rate: 3,
            num_variables: 10,
            folding_factor: 2,
            domain_size: 10,
            domain_gen: Field64::from(2),
            domain_gen_inv: Field64::from(2),
            exp_domain_gen: Field64::from(2),
        }];

        assert!(
            !config.check_pow_bits(),
            "A round has folding_pow_bits exceeding max_pow_bits, should return false."
        );
    }

    #[test]
    fn test_check_pow_bits_exactly_at_limit() {
        type MerkleConfig = KeccakMerkleTreeParams<Field64>;

        let params = default_whir_params::<Field64>();
        let mv_params = MultivariateParameters::<Field64>::new(10);
        let reed_solomon = Arc::new(RSDefault);
        let basefield_reed_solomon = reed_solomon.clone();
        let mut config = WhirConfig::<Field64, MerkleConfig>::new(
            reed_solomon,
            basefield_reed_solomon,
            mv_params,
            params,
        );

        config.max_pow_bits = 20;
        config.starting_folding_pow_bits = 20.0;
        config.final_pow_bits = 20.0;
        config.final_folding_pow_bits = 20.0;

        config.round_parameters = vec![RoundConfig {
            pow_bits: 20.0,
            folding_pow_bits: 20.0,
            num_queries: 5,
            ood_samples: 2,
            log_inv_rate: 3,
            num_variables: 10,
            folding_factor: 2,
            domain_size: 10,
            domain_gen: Field64::from(2),
            domain_gen_inv: Field64::from(2),
            exp_domain_gen: Field64::from(2),
        }];

        assert!(
            config.check_pow_bits(),
            "All pow_bits are exactly at max_pow_bits, should return true."
        );
    }

    #[test]
    fn test_check_pow_bits_all_exceed() {
        type MerkleConfig = KeccakMerkleTreeParams<Field64>;

        let params = default_whir_params::<Field64>();
        let mv_params = MultivariateParameters::<Field64>::new(10);
        let reed_solomon = Arc::new(RSDefault);
        let basefield_reed_solomon = reed_solomon.clone();
        let mut config = WhirConfig::<Field64, MerkleConfig>::new(
            reed_solomon,
            basefield_reed_solomon,
            mv_params,
            params,
        );

        config.max_pow_bits = 20;
        config.starting_folding_pow_bits = 22.0;
        config.final_pow_bits = 23.0;
        config.final_folding_pow_bits = 24.0;

        config.round_parameters = vec![RoundConfig {
            pow_bits: 25.0,
            folding_pow_bits: 26.0,
            num_queries: 5,
            ood_samples: 2,
            log_inv_rate: 3,
            num_variables: 10,
            folding_factor: 2,
            domain_size: 10,
            domain_gen: Field64::from(2),
            domain_gen_inv: Field64::from(2),
            exp_domain_gen: Field64::from(2),
        }];

        assert!(
            !config.check_pow_bits(),
            "All values exceed max_pow_bits, should return false."
        );
    }

    #[test]
    fn test_list_size_bits_conjecture_list() {
        // ConjectureList: list_size_bits = num_variables + log_inv_rate - log_eta

        let cases = vec![
            (10, 5, 2.0, 13.0), // Basic case
            (0, 5, 2.0, 3.0),   // Edge case: num_variables = 0
            (10, 0, 2.0, 8.0),  // Edge case: log_inv_rate = 0
            (10, 5, 0.0, 15.0), // Edge case: log_eta = 0
            (10, 5, 10.0, 5.0), // High log_eta
        ];

        for (num_variables, log_inv_rate, log_eta, expected) in cases {
            let result = WhirConfig::<Field64, KeccakMerkleTreeParams<Field64>>::list_size_bits(
                SoundnessType::ConjectureList,
                num_variables,
                log_inv_rate,
                log_eta,
            );
            assert!(
                (result - expected).abs() < 1e-6,
                "Failed for {:?}",
                (num_variables, log_inv_rate, log_eta)
            );
        }
    }

    #[test]
    fn test_list_size_bits_provable_list() {
        // ProvableList: list_size_bits = (log_inv_rate / 2) - (1 + log_eta)

        let cases = vec![
            (10, 8, 2.0, 1.0),   // Basic case
            (10, 0, 2.0, -3.0),  // Edge case: log_inv_rate = 0
            (10, 8, 0.0, 3.0),   // Edge case: log_eta = 0
            (10, 8, 10.0, -7.0), // High log_eta
        ];

        for (num_variables, log_inv_rate, log_eta, expected) in cases {
            let result = WhirConfig::<Field64, KeccakMerkleTreeParams<Field64>>::list_size_bits(
                SoundnessType::ProvableList,
                num_variables,
                log_inv_rate,
                log_eta,
            );
            assert!(
                (result - expected).abs() < 1e-6,
                "Failed for {:?}",
                (num_variables, log_inv_rate, log_eta)
            );
        }
    }

    #[test]
    fn test_list_size_bits_unique_decoding() {
        // UniqueDecoding: always returns 0.0

        let cases = vec![
            (10, 5, 2.0),
            (0, 5, 2.0),
            (10, 0, 2.0),
            (10, 5, 0.0),
            (10, 5, 10.0),
        ];

        for (num_variables, log_inv_rate, log_eta) in cases {
            let result = WhirConfig::<Field64, KeccakMerkleTreeParams<Field64>>::list_size_bits(
                SoundnessType::UniqueDecoding,
                num_variables,
                log_inv_rate,
                log_eta,
            );
            assert!(
                (result - 0.0) < 1e-6,
                "Failed for {:?}",
                (num_variables, log_inv_rate, log_eta)
            );
        }
    }

    #[test]
    fn test_rbr_ood_sample_conjecture_list() {
        // ConjectureList: rbr_ood_sample = (ood_samples * field_size_bits) + 1 - (2 * list_size_bits + num_variables * ood_samples)

        let cases = vec![
            (
                10,
                5,
                2.0,
                256,
                3,
                (3.0 * 256.0) + 1.0 - (2.0 * 13.0 + (10.0 * 3.0)),
            ), // Basic case
            (
                0,
                5,
                2.0,
                256,
                3,
                (3.0 * 256.0) + 1.0 - (2.0 * 3.0 + (0.0 * 3.0)),
            ), // Edge case: num_variables = 0
            (
                10,
                0,
                2.0,
                256,
                3,
                (3.0 * 256.0) + 1.0 - (2.0 * 8.0 + (10.0 * 3.0)),
            ), // Edge case: log_inv_rate = 0
            (
                10,
                5,
                0.0,
                256,
                3,
                (3.0 * 256.0) + 1.0 - (2.0 * 15.0 + (10.0 * 3.0)),
            ), // Edge case: log_eta = 0
            (
                10,
                5,
                10.0,
                256,
                3,
                (3.0 * 256.0) + 1.0 - (2.0 * 5.0 + (10.0 * 3.0)),
            ), // High log_eta
        ];

        for (num_variables, log_inv_rate, log_eta, field_size_bits, ood_samples, expected) in cases
        {
            let result = WhirConfig::<Field64, KeccakMerkleTreeParams<Field64>>::rbr_ood_sample(
                SoundnessType::ConjectureList,
                num_variables,
                log_inv_rate,
                log_eta,
                field_size_bits,
                ood_samples,
            );
            assert!(
                (result - expected).abs() < 1e-6,
                "Failed for {:?}",
                (
                    num_variables,
                    log_inv_rate,
                    log_eta,
                    field_size_bits,
                    ood_samples
                )
            );
        }
    }

    #[test]
    fn test_rbr_ood_sample_provable_list() {
        // ProvableList: Uses a different list_size_bits formula

        let cases = vec![
            (
                10,
                8,
                2.0,
                256,
                3,
                (3.0 * 256.0) + 1.0 - (2.0 * 1.0 + (10.0 * 3.0)),
            ), // Basic case
            (
                10,
                0,
                2.0,
                256,
                3,
                (3.0 * 256.0) + 1.0 - (2.0 * -3.0 + (10.0 * 3.0)),
            ), // log_inv_rate = 0
            (
                10,
                8,
                0.0,
                256,
                3,
                (3.0 * 256.0) + 1.0 - (2.0 * 3.0 + (10.0 * 3.0)),
            ), // log_eta = 0
            (
                10,
                8,
                10.0,
                256,
                3,
                (3.0 * 256.0) + 1.0 - (2.0 * -7.0 + (10.0 * 3.0)),
            ), // High log_eta
        ];

        for (num_variables, log_inv_rate, log_eta, field_size_bits, ood_samples, expected) in cases
        {
            let result = WhirConfig::<Field64, KeccakMerkleTreeParams<Field64>>::rbr_ood_sample(
                SoundnessType::ProvableList,
                num_variables,
                log_inv_rate,
                log_eta,
                field_size_bits,
                ood_samples,
            );
            assert!(
                (result - expected).abs() < 1e-6,
                "Failed for {:?}",
                (
                    num_variables,
                    log_inv_rate,
                    log_eta,
                    field_size_bits,
                    ood_samples
                )
            );
        }
    }

    #[test]
    fn test_ood_samples_unique_decoding() {
        // UniqueDecoding should always return 0 regardless of parameters
        assert_eq!(
            WhirConfig::<Field64, KeccakMerkleTreeParams<Field64>>::ood_samples(
                100,
                SoundnessType::UniqueDecoding,
                10,
                3,
                1.5,
                256
            ),
            0
        );
    }

    #[test]
    fn test_ood_samples_valid_case() {
        // Testing a valid case where the function finds an appropriate `ood_samples`
        assert_eq!(
            WhirConfig::<Field64, KeccakMerkleTreeParams<Field64>>::ood_samples(
                50, // security level
                SoundnessType::ProvableList,
                15,  // num_variables
                4,   // log_inv_rate
                2.0, // log_eta
                256, // field_size_bits
            ),
            1
        );
    }

    #[test]
    fn test_ood_samples_low_security_level() {
        // Lower security level should require fewer OOD samples
        assert_eq!(
            WhirConfig::<Field64, KeccakMerkleTreeParams<Field64>>::ood_samples(
                30, // Lower security level
                SoundnessType::ConjectureList,
                20,  // num_variables
                5,   // log_inv_rate
                2.5, // log_eta
                512, // field_size_bits
            ),
            1
        );
    }

    #[test]
    fn test_ood_samples_high_security_level() {
        // Higher security level should require more OOD samples
        assert_eq!(
            WhirConfig::<Field64, KeccakMerkleTreeParams<Field64>>::ood_samples(
                100, // High security level
                SoundnessType::ProvableList,
                25,   // num_variables
                6,    // log_inv_rate
                3.0,  // log_eta
                1024  // field_size_bits
            ),
            1
        );
    }

    #[test]
    fn test_ood_extremely_high_security_level() {
        assert_eq!(
            WhirConfig::<Field64, KeccakMerkleTreeParams<Field64>>::ood_samples(
                1000, // Extremely high security level
                SoundnessType::ConjectureList,
                10,  // num_variables
                5,   // log_inv_rate
                2.0, // log_eta
                256, // field_size_bits
            ),
            5
        );
    }
}
