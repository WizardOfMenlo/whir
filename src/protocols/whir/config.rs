use core::panic;
use std::{
    f64::consts::LOG2_10,
    fmt::{Debug, Display},
    ops::Neg,
};

use ark_ff::FftField;
use serde::{Deserialize, Serialize};

use crate::{
    algebra::{
        embedding::{self, Basefield, Embedding},
        fields::FieldWithSize,
    },
    bits::Bits,
    parameters::{MultivariateParameters, ProtocolParameters, SoundnessType},
    protocols::{irs_commit, matrix_commit, proof_of_work, sumcheck},
    type_info::{Type, Typed},
};

#[derive(Clone, PartialEq, Eq, Serialize, Deserialize, Debug)]
#[serde(bound = "F: FftField, M: Embedding<Target = F>, M::Source: FftField")]
pub struct Config<F, M = Basefield<F>>
where
    F: FftField,
    M: Embedding<Target = F>,
    M::Source: FftField,
{
    pub initial_committer: irs_commit::Config<M::Source, F, M>,
    pub initial_sumcheck: sumcheck::Config<F>,
    pub round_configs: Vec<RoundConfig<F>>,
    pub final_sumcheck: sumcheck::Config<F>,
    pub final_pow: proof_of_work::Config,

    // TODO: These don't belong in the config. Instead there should be
    // fn like `WhirConfig::soundness(&self, assumptions: SoundnessType) -> Bits`.
    pub soundness_type: SoundnessType,
    pub security_level: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "F: FftField")]
pub struct RoundConfig<F>
where
    F: FftField,
{
    pub irs_committer: irs_commit::Config<F>,
    pub sumcheck: sumcheck::Config<F>,
    pub pow: proof_of_work::Config,
}

impl<F> Config<F, Basefield<F>>
where
    F: FftField + FieldWithSize,
{
    #[allow(clippy::too_many_lines)]
    pub fn new(
        mv_parameters: MultivariateParameters<F>,
        whir_parameters: &ProtocolParameters,
    ) -> Self {
        whir_parameters
            .folding_factor
            .check_validity(mv_parameters.num_variables)
            .unwrap();

        // Proof of work constructor with the requested hash function.
        let pow = |difficulty| proof_of_work::Config {
            hash_id: whir_parameters.hash_id,
            threshold: proof_of_work::threshold(Bits::new(difficulty)),
        };

        let protocol_security_level = whir_parameters
            .security_level
            .saturating_sub(whir_parameters.pow_bits);
        let field_size_bits = F::field_size_in_bits();
        let mut log_inv_rate = whir_parameters.starting_log_inv_rate;
        let mut num_variables = mv_parameters.num_variables;

        let mut domain_size = 1 << (mv_parameters.num_variables + log_inv_rate);

        let (num_rounds, final_sumcheck_rounds) = whir_parameters
            .folding_factor
            .compute_number_of_rounds(mv_parameters.num_variables);

        let log_eta_start = Self::log_eta(whir_parameters.soundness_type, log_inv_rate as f64);

        let commitment_ood_samples = if whir_parameters.initial_statement {
            Self::ood_samples(
                whir_parameters.security_level,
                whir_parameters.soundness_type,
                num_variables,
                log_inv_rate as f64,
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
                log_inv_rate as f64,
                log_eta_start,
            )
        } else {
            {
                let prox_gaps_error = Self::rbr_soundness_fold_prox_gaps(
                    whir_parameters.soundness_type,
                    field_size_bits,
                    num_variables,
                    log_inv_rate as f64,
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

            let log_next_eta = Self::log_eta(whir_parameters.soundness_type, next_rate as f64);

            let num_queries = Self::queries(
                whir_parameters.soundness_type,
                protocol_security_level,
                log_inv_rate,
            );

            let ood_samples = Self::ood_samples(
                whir_parameters.security_level,
                whir_parameters.soundness_type,
                num_variables,
                next_rate as f64,
                log_next_eta,
                field_size_bits,
            );

            let query_error = Self::rbr_queries(
                whir_parameters.soundness_type,
                log_inv_rate as f64,
                num_queries,
            );

            let combination_error = Self::rbr_soundness_queries_combination(
                whir_parameters.soundness_type,
                field_size_bits,
                num_variables,
                next_rate as f64,
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
                next_rate as f64,
                log_next_eta,
            );

            let next_folding_factor = whir_parameters.folding_factor.at_round(round + 1);
            let matrix_committer = matrix_commit::Config::<F>::with_hash(
                whir_parameters.hash_id,
                (domain_size / 2) >> next_folding_factor,
                1 << next_folding_factor,
            );

            round_parameters.push(RoundConfig {
                irs_committer: irs_commit::Config {
                    embedding: Typed::new(embedding::Identity::new()),
                    num_vectors: 1,
                    vector_size: 1 << num_variables,
                    expansion: 1 << next_rate,
                    interleaving_depth: 1 << next_folding_factor,
                    matrix_commit: matrix_committer.clone(),
                    in_domain_samples: Self::queries(
                        whir_parameters.soundness_type,
                        protocol_security_level,
                        next_rate,
                    ),
                    out_domain_samples: ood_samples,
                    deduplicate_in_domain: true, // TODO: Configurable
                },
                sumcheck: sumcheck::Config {
                    field: Type::<F>::new(),
                    initial_size: 1 << num_variables,
                    round_pow: pow(folding_pow_bits),
                    num_rounds: next_folding_factor,
                },
                pow: pow(pow_bits),
            });

            num_variables -= next_folding_factor;
            log_inv_rate = next_rate;
            domain_size /= 2;
        }

        let final_queries = Self::queries(
            whir_parameters.soundness_type,
            protocol_security_level,
            log_inv_rate,
        );

        let final_pow_bits = 0_f64.max(
            whir_parameters.security_level as f64
                - Self::rbr_queries(
                    whir_parameters.soundness_type,
                    log_inv_rate as f64,
                    final_queries,
                ),
        );

        let final_folding_pow_bits =
            0_f64.max(whir_parameters.security_level as f64 - (field_size_bits - 1) as f64);

        Self {
            security_level: whir_parameters.security_level,
            soundness_type: whir_parameters.soundness_type,
            initial_committer: irs_commit::Config {
                embedding: Default::default(),
                num_vectors: whir_parameters.batch_size,
                vector_size: 1 << mv_parameters.num_variables,
                expansion: 1 << whir_parameters.starting_log_inv_rate,
                interleaving_depth: 1 << whir_parameters.folding_factor.at_round(0),
                matrix_commit: matrix_commit::Config::with_hash(
                    whir_parameters.hash_id,
                    1 << (mv_parameters.num_variables + whir_parameters.starting_log_inv_rate
                        - whir_parameters.folding_factor.at_round(0)),
                    whir_parameters.batch_size << whir_parameters.folding_factor.at_round(0),
                ),
                in_domain_samples: Self::queries(
                    whir_parameters.soundness_type,
                    protocol_security_level,
                    whir_parameters.starting_log_inv_rate,
                ),
                out_domain_samples: commitment_ood_samples,
                deduplicate_in_domain: true,
            },
            initial_sumcheck: sumcheck::Config {
                field: Type::<F>::new(),
                initial_size: 1 << mv_parameters.num_variables,
                round_pow: pow(starting_folding_pow_bits),
                num_rounds: whir_parameters.folding_factor.at_round(0),
            },
            round_configs: round_parameters,
            final_sumcheck: sumcheck::Config {
                field: Type::<F>::new(),
                initial_size: 1 << num_variables,
                round_pow: pow(final_folding_pow_bits),
                num_rounds: final_sumcheck_rounds,
            },
            final_pow: pow(final_pow_bits),
        }
    }

    pub fn check_max_pow_bits(&self, max_bits: Bits) -> bool {
        if self.initial_sumcheck.round_pow.difficulty() > max_bits {
            return false;
        }
        for round_config in &self.round_configs {
            if round_config.pow.difficulty() > max_bits {
                return false;
            }
            if round_config.sumcheck.round_pow.difficulty() > max_bits {
                return false;
            }
        }
        if self.final_pow.difficulty() > max_bits {
            return false;
        }
        if self.final_sumcheck.round_pow.difficulty() > max_bits {
            return false;
        }
        true
    }

    pub const fn log_eta(soundness_type: SoundnessType, log_inv_rate: f64) -> f64 {
        // Ask me how I did this? At the time, only God and I knew. Now only God knows
        match soundness_type {
            SoundnessType::ProvableList => -(0.5 * log_inv_rate + LOG2_10 + 1.),
            SoundnessType::UniqueDecoding => 0.,
            SoundnessType::ConjectureList => -(log_inv_rate + 1.),
        }
    }

    pub const fn list_size_bits(
        soundness_type: SoundnessType,
        num_variables: usize,
        log_inv_rate: f64,
        log_eta: f64,
    ) -> f64 {
        match soundness_type {
            SoundnessType::ConjectureList => num_variables as f64 + log_inv_rate - log_eta,
            SoundnessType::ProvableList => {
                let log_inv_sqrt_rate: f64 = log_inv_rate / 2.;
                log_inv_sqrt_rate - (1. + log_eta)
            }
            SoundnessType::UniqueDecoding => 0.0,
        }
    }

    pub const fn rbr_ood_sample(
        soundness_type: SoundnessType,
        num_variables: usize,
        log_inv_rate: f64,
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
        log_inv_rate: f64,
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
        log_inv_rate: f64,
        log_eta: f64,
    ) -> f64 {
        // Recall, at each round we are only folding by two at a time
        let error = match soundness_type {
            SoundnessType::ConjectureList => num_variables as f64 + log_inv_rate - log_eta,
            SoundnessType::ProvableList => LOG2_10 + 3.5 * log_inv_rate + 2. * num_variables as f64,
            SoundnessType::UniqueDecoding => num_variables as f64 + log_inv_rate,
        };

        field_size_bits as f64 - error
    }

    pub const fn rbr_soundness_fold_sumcheck(
        soundness_type: SoundnessType,
        field_size_bits: usize,
        num_variables: usize,
        log_inv_rate: f64,
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
        log_inv_rate: f64,
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
        log_inv_rate: f64,
        num_queries: usize,
    ) -> f64 {
        let num_queries = num_queries as f64;

        match soundness_type {
            SoundnessType::UniqueDecoding => {
                let rate = 1. / log_inv_rate.exp2();
                let denom = -(0.5 * (1. + rate)).log2();

                num_queries * denom
            }
            SoundnessType::ProvableList => num_queries * 0.5 * log_inv_rate,
            SoundnessType::ConjectureList => num_queries * log_inv_rate,
        }
    }

    pub fn rbr_soundness_queries_combination(
        soundness_type: SoundnessType,
        field_size_bits: usize,
        num_variables: usize,
        log_inv_rate: f64,
        log_eta: f64,
        ood_samples: usize,
        num_queries: usize,
    ) -> f64 {
        let list_size = Self::list_size_bits(soundness_type, num_variables, log_inv_rate, log_eta);

        let log_combination = ((ood_samples + num_queries) as f64).log2();

        field_size_bits as f64 - (log_combination + list_size + 1.)
    }
}

impl<F, M> Config<F, M>
where
    F: FftField,
    M: Embedding<Target = F>,
    M::Source: FftField,
{
    pub fn embedding(&self) -> &M {
        self.initial_committer.embedding()
    }

    #[deprecated]
    pub const fn allows_statement(&self) -> bool {
        true
    }

    pub const fn initial_size(&self) -> usize {
        self.initial_committer.vector_size
    }

    pub fn initial_num_variables(&self) -> usize {
        assert!(self.initial_size().is_power_of_two());
        self.initial_size().trailing_zeros() as usize
    }

    pub const fn final_size(&self) -> usize {
        self.final_sumcheck.final_size()
    }

    pub const fn n_rounds(&self) -> usize {
        self.round_configs.len()
    }

    pub fn final_rate(&self) -> f64 {
        self.round_configs.last().map_or_else(
            || self.initial_committer.rate(),
            |round_config| round_config.irs_committer.rate(),
        )
    }

    pub fn final_in_domain_samples(&self) -> usize {
        self.round_configs
            .last()
            .map_or(self.initial_committer.in_domain_samples, |round_config| {
                round_config.irs_committer.in_domain_samples
            })
    }
}

impl<F: FftField> Display for Config<F> {
    #[allow(clippy::too_many_lines)]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "Security level: {} bits using {} security",
            self.security_level, self.soundness_type
        )?;
        writeln!(f, "Initial:\n  commit   {}", self.initial_committer)?;
        writeln!(f, "  sumcheck {}", self.initial_sumcheck)?;
        for (i, r) in self.round_configs.iter().enumerate() {
            write!(f, "Round {i}:\n{r}")?;
        }
        writeln!(
            f,
            "Final:\n  pow      {:.2}bits",
            self.final_pow.difficulty()
        )?;
        writeln!(f, "  sumcheck {}", self.final_sumcheck)?;

        writeln!(f, "------------------------------------")?;
        writeln!(f, "Round by round soundness analysis:")?;
        writeln!(f, "------------------------------------")?;

        let field_size_bits = F::field_size_in_bits();
        let log_eta = Self::log_eta(
            self.soundness_type,
            self.initial_committer.rate().log2().neg(),
        );
        let mut num_variables = self.initial_num_variables();

        if self.initial_committer.out_domain_samples > 0 {
            writeln!(
                f,
                "{:.1} bits -- OOD commitment",
                Self::rbr_ood_sample(
                    self.soundness_type,
                    num_variables,
                    self.initial_committer.rate().log2().neg(),
                    log_eta,
                    field_size_bits,
                    self.initial_committer.out_domain_samples
                )
            )?;
        }

        let prox_gaps_error = Self::rbr_soundness_fold_prox_gaps(
            self.soundness_type,
            field_size_bits,
            num_variables,
            self.initial_committer.rate().log2().neg(),
            log_eta,
        );
        let sumcheck_error = Self::rbr_soundness_fold_sumcheck(
            self.soundness_type,
            field_size_bits,
            num_variables,
            self.initial_committer.rate().log2().neg(),
            log_eta,
        );
        writeln!(
            f,
            "{:.1} bits -- (x{}) prox gaps: {:.1}, sumcheck: {:.1}, pow: {:.1}",
            prox_gaps_error.min(sumcheck_error)
                + f64::from(self.initial_sumcheck.round_pow.difficulty()),
            self.initial_sumcheck.num_rounds,
            prox_gaps_error,
            sumcheck_error,
            self.initial_sumcheck.round_pow.difficulty(),
        )?;

        num_variables -= self.initial_sumcheck.num_rounds;

        for r in &self.round_configs {
            let next_rate = (r.log_inv_rate() + (r.sumcheck.num_rounds - 1)) as f64;
            let log_eta = Self::log_eta(self.soundness_type, next_rate);

            if r.irs_committer.out_domain_samples > 0 {
                writeln!(
                    f,
                    "{:.1} bits -- OOD sample",
                    Self::rbr_ood_sample(
                        self.soundness_type,
                        num_variables,
                        next_rate,
                        log_eta,
                        field_size_bits,
                        r.irs_committer.out_domain_samples
                    )
                )?;
            }

            let query_error = Self::rbr_queries(
                self.soundness_type,
                r.log_inv_rate() as f64,
                r.irs_committer.in_domain_samples,
            );
            let combination_error = Self::rbr_soundness_queries_combination(
                self.soundness_type,
                field_size_bits,
                num_variables,
                next_rate,
                log_eta,
                r.irs_committer.out_domain_samples,
                r.irs_committer.in_domain_samples,
            );
            writeln!(
                f,
                "{:.1} bits -- query error: {:.1}, combination: {:.1}, pow: {:.1}",
                query_error.min(combination_error) + f64::from(r.pow.difficulty()),
                query_error,
                combination_error,
                r.pow.difficulty(),
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
                prox_gaps_error.min(sumcheck_error) + f64::from(r.sumcheck.round_pow.difficulty()),
                r.sumcheck.num_rounds,
                prox_gaps_error,
                sumcheck_error,
                r.sumcheck.round_pow.difficulty(),
            )?;

            num_variables -= r.sumcheck.num_rounds;
        }

        let query_error = Self::rbr_queries(
            self.soundness_type,
            self.final_rate().log2().neg(),
            self.final_in_domain_samples(),
        );
        writeln!(
            f,
            "{:.1} bits -- query error: {:.1}, pow: {:.1}",
            query_error + f64::from(self.final_pow.difficulty()),
            query_error,
            self.final_pow.difficulty(),
        )?;

        if self.final_sumcheck.num_rounds > 0 {
            let combination_error = field_size_bits as f64 - 1.;
            writeln!(
                f,
                "{:.1} bits -- (x{}) combination: {:.1}, pow: {:.1}",
                combination_error + f64::from(self.final_pow.difficulty()),
                self.final_sumcheck.num_rounds,
                combination_error,
                self.final_sumcheck.round_pow.difficulty(),
            )?;
        }

        Ok(())
    }
}

impl<F: FftField> RoundConfig<F> {
    pub fn initial_size(&self) -> usize {
        assert_eq!(self.irs_committer.vector_size, self.sumcheck.initial_size);
        self.sumcheck.initial_size
    }

    pub const fn final_size(&self) -> usize {
        self.sumcheck.final_size()
    }

    pub fn log_inv_rate(&self) -> usize {
        assert!(self.irs_committer.expansion.is_power_of_two());
        self.irs_committer.expansion.ilog2() as usize
    }

    pub fn initial_num_variables(&self) -> usize {
        assert!(self.irs_committer.vector_size.is_power_of_two());
        self.irs_committer.vector_size.ilog2() as usize
    }

    pub fn final_num_variables(&self) -> usize {
        self.initial_num_variables() - self.sumcheck.num_rounds
    }
}

impl<F> Display for RoundConfig<F>
where
    F: FftField,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "  commit   {}", self.irs_committer,)?;
        writeln!(f, "  pow      {:.2} bits", self.pow.difficulty())?;
        writeln!(f, "  sumcheck {}", self.sumcheck)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        algebra::fields::Field64, bits::Bits, hash, parameters::FoldingFactor, utils::test_serde,
    };

    /// Generates default WHIR parameters
    fn default_whir_params() -> ProtocolParameters {
        ProtocolParameters {
            initial_statement: true,
            security_level: 100,
            pow_bits: 20,
            folding_factor: FoldingFactor::ConstantFromSecondRound(4, 4),
            soundness_type: SoundnessType::ConjectureList,
            starting_log_inv_rate: 1,
            batch_size: 1,
            hash_id: hash::BLAKE3,
        }
    }

    #[test]
    fn test_whir_params_serde() {
        test_serde(&default_whir_params());
        test_serde(&default_whir_params());
    }

    #[test]
    fn test_whir_config_serde() {
        let params = default_whir_params();

        let mv_params = MultivariateParameters::<Field64>::new(10);
        let config = Config::<Field64>::new(mv_params, &params);

        test_serde(&config);
    }

    #[test]
    fn test_n_rounds() {
        let params = default_whir_params();
        let mv_params = MultivariateParameters::<Field64>::new(10);
        let config = Config::<Field64>::new(mv_params, &params);

        assert_eq!(config.n_rounds(), config.round_configs.len());
    }

    #[test]
    fn test_folding_pow_bits() {
        let field_size_bits = 64;
        let soundness = SoundnessType::ConjectureList;

        let pow_bits = Config::<Field64>::folding_pow_bits(
            100, // Security level
            soundness,
            field_size_bits,
            10,   // Number of variables
            5.0,  // Log inverse rate
            -3.0, // Log eta
        );

        // PoW bits should never be negative
        assert!(pow_bits >= 0.);
    }

    #[test]
    fn test_queries_unique_decoding() {
        let security_level = 100;
        let log_inv_rate = 5;

        let result =
            Config::<Field64>::queries(SoundnessType::UniqueDecoding, security_level, log_inv_rate);

        assert_eq!(result, 105);
    }

    #[test]
    fn test_queries_provable_list() {
        let security_level = 128;
        let log_inv_rate = 8;

        let result =
            Config::<Field64>::queries(SoundnessType::ProvableList, security_level, log_inv_rate);

        assert_eq!(result, 32);
    }

    #[test]
    fn test_queries_conjecture_list() {
        let security_level = 256;
        let log_inv_rate = 16;

        let result =
            Config::<Field64>::queries(SoundnessType::ConjectureList, security_level, log_inv_rate);

        assert_eq!(result, 16);
    }

    #[test]
    fn test_rbr_queries_unique_decoding() {
        let log_inv_rate = 5.0; // log_inv_rate = 5
        let num_queries = 10; // Number of queries

        let result = Config::<Field64>::rbr_queries(
            SoundnessType::UniqueDecoding,
            log_inv_rate,
            num_queries,
        );

        assert!((result - 9.556_058_806_415_466).abs() < 1e-6);
    }

    #[test]
    fn test_rbr_queries_provable_list() {
        let log_inv_rate = 8.0; // log_inv_rate = 8
        let num_queries = 16; // Number of queries

        let result =
            Config::<Field64>::rbr_queries(SoundnessType::ProvableList, log_inv_rate, num_queries);

        assert!((result - 64.0) < 1e-6);
    }

    #[test]
    fn test_rbr_queries_conjecture_list() {
        let log_inv_rate = 4.0; // log_inv_rate = 4
        let num_queries = 20; // Number of queries

        let result = Config::<Field64>::rbr_queries(
            SoundnessType::ConjectureList,
            log_inv_rate,
            num_queries,
        );

        assert!((result - 80.) < 1e-6);
    }

    #[test]
    fn test_check_pow_bits_within_limits() {
        let params = default_whir_params();
        let mv_params = MultivariateParameters::<Field64>::new(10);
        let mut config = Config::<Field64>::new(mv_params, &params);

        // Set all values within limits
        config.initial_sumcheck.round_pow = proof_of_work::Config::from_difficulty(Bits::new(15.0));
        config.final_pow = proof_of_work::Config::from_difficulty(Bits::new(18.0));
        config.final_sumcheck.round_pow = proof_of_work::Config::from_difficulty(Bits::new(19.5));

        // Ensure all rounds are within limits
        config.round_configs = vec![
            RoundConfig {
                irs_committer: irs_commit::Config {
                    embedding: Typed::new(embedding::Identity::new()),
                    num_vectors: 1,
                    vector_size: 1 << 10,
                    expansion: 1 << 3,
                    interleaving_depth: 1 << 2,
                    matrix_commit: matrix_commit::Config::<Field64>::new(0, 0),
                    in_domain_samples: 5,
                    out_domain_samples: 2,
                    deduplicate_in_domain: true,
                },
                sumcheck: sumcheck::Config {
                    field: Type::<Field64>::new(),
                    initial_size: 1 << 10,
                    round_pow: proof_of_work::Config::from_difficulty(Bits::new(19.0)),
                    num_rounds: 2,
                },
                pow: proof_of_work::Config::from_difficulty(Bits::new(17.0)),
            },
            RoundConfig {
                irs_committer: irs_commit::Config {
                    embedding: Typed::new(embedding::Identity::new()),
                    num_vectors: 1,
                    vector_size: 1 << 10,
                    expansion: 1 << 4,
                    interleaving_depth: 1 << 2,
                    matrix_commit: matrix_commit::Config::<Field64>::new(0, 0),
                    in_domain_samples: 6,
                    out_domain_samples: 2,
                    deduplicate_in_domain: true,
                },
                sumcheck: sumcheck::Config {
                    field: Type::<Field64>::new(),
                    initial_size: 1 << 10,
                    round_pow: proof_of_work::Config::from_difficulty(Bits::new(19.5)),
                    num_rounds: 2,
                },
                pow: proof_of_work::Config::from_difficulty(Bits::new(18.0)),
            },
        ];

        assert!(
            config.check_max_pow_bits(Bits::new(20.0)),
            "All values are within limits, check_pow_bits should return true."
        );
    }

    #[test]
    fn test_check_pow_bits_starting_folding_exceeds() {
        let params = default_whir_params();
        let mv_params = MultivariateParameters::<Field64>::new(10);
        let mut config = Config::<Field64>::new(mv_params, &params);

        config.initial_sumcheck.round_pow = proof_of_work::Config::from_difficulty(Bits::new(21.0));
        config.final_pow = proof_of_work::Config::from_difficulty(Bits::new(18.0));
        config.final_sumcheck.round_pow = proof_of_work::Config::from_difficulty(Bits::new(19.5));

        assert!(
            !config.check_max_pow_bits(Bits::new(20.0)),
            "Starting folding pow bits exceeds max_pow_bits, should return false."
        );
    }

    #[test]
    fn test_list_size_bits_conjecture_list() {
        // ConjectureList: list_size_bits = num_variables + log_inv_rate - log_eta

        let cases = vec![
            (10, 5.0, 2.0, 13.0), // Basic case
            (0, 5.0, 2.0, 3.0),   // Edge case: num_variables = 0
            (10, 0.0, 2.0, 8.0),  // Edge case: log_inv_rate = 0
            (10, 5.0, 0.0, 15.0), // Edge case: log_eta = 0
            (10, 5.0, 10.0, 5.0), // High log_eta
        ];

        for (num_variables, log_inv_rate, log_eta, expected) in cases {
            let result = Config::<Field64>::list_size_bits(
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
            (10, 8.0, 2.0, 1.0),   // Basic case
            (10, 0.0, 2.0, -3.0),  // Edge case: log_inv_rate = 0
            (10, 8.0, 0.0, 3.0),   // Edge case: log_eta = 0
            (10, 8.0, 10.0, -7.0), // High log_eta
        ];

        for (num_variables, log_inv_rate, log_eta, expected) in cases {
            let result = Config::<Field64>::list_size_bits(
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
            (10, 5.0, 2.0),
            (0, 5.0, 2.0),
            (10, 0.0, 2.0),
            (10, 5.0, 0.0),
            (10, 5.0, 10.0),
        ];

        for (num_variables, log_inv_rate, log_eta) in cases {
            let result = Config::<Field64>::list_size_bits(
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
                5.0,
                2.0,
                256,
                3,
                (3.0 * 256.0) + 1.0 - (2.0 * 13.0 + (10.0 * 3.0)),
            ), // Basic case
            (
                0,
                5.0,
                2.0,
                256,
                3,
                (3.0 * 256.0) + 1.0 - (2.0 * 3.0 + (0.0 * 3.0)),
            ), // Edge case: num_variables = 0
            (
                10,
                0.0,
                2.0,
                256,
                3,
                (3.0 * 256.0) + 1.0 - (2.0 * 8.0 + (10.0 * 3.0)),
            ), // Edge case: log_inv_rate = 0
            (
                10,
                5.0,
                0.0,
                256,
                3,
                (3.0 * 256.0) + 1.0 - (2.0 * 15.0 + (10.0 * 3.0)),
            ), // Edge case: log_eta = 0
            (
                10,
                5.0,
                10.0,
                256,
                3,
                (3.0 * 256.0) + 1.0 - (2.0 * 5.0 + (10.0 * 3.0)),
            ), // High log_eta
        ];

        for (num_variables, log_inv_rate, log_eta, field_size_bits, ood_samples, expected) in cases
        {
            let result = Config::<Field64>::rbr_ood_sample(
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
                8.0,
                2.0,
                256,
                3,
                (3.0 * 256.0) + 1.0 - (2.0 * 1.0 + (10.0 * 3.0)),
            ), // Basic case
            (
                10,
                0.0,
                2.0,
                256,
                3,
                (3.0 * 256.0) + 1.0 - (2.0 * -3.0 + (10.0 * 3.0)),
            ), // log_inv_rate = 0
            (
                10,
                8.0,
                0.0,
                256,
                3,
                (3.0 * 256.0) + 1.0 - (2.0 * 3.0 + (10.0 * 3.0)),
            ), // log_eta = 0
            (
                10,
                8.0,
                10.0,
                256,
                3,
                (3.0 * 256.0) + 1.0 - (2.0 * -7.0 + (10.0 * 3.0)),
            ), // High log_eta
        ];

        for (num_variables, log_inv_rate, log_eta, field_size_bits, ood_samples, expected) in cases
        {
            let result = Config::<Field64>::rbr_ood_sample(
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
            Config::<Field64>::ood_samples(100, SoundnessType::UniqueDecoding, 10, 3.0, 1.5, 256),
            0
        );
    }

    #[test]
    fn test_ood_samples_valid_case() {
        // Testing a valid case where the function finds an appropriate `ood_samples`
        assert_eq!(
            Config::<Field64>::ood_samples(
                50, // security level
                SoundnessType::ProvableList,
                15,  // num_variables
                4.0, // log_inv_rate
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
            Config::<Field64>::ood_samples(
                30, // Lower security level
                SoundnessType::ConjectureList,
                20,  // num_variables
                5.0, // log_inv_rate
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
            Config::<Field64>::ood_samples(
                100, // High security level
                SoundnessType::ProvableList,
                25,   // num_variables
                6.0,  // log_inv_rate
                3.0,  // log_eta
                1024  // field_size_bits
            ),
            1
        );
    }

    #[test]
    fn test_ood_extremely_high_security_level() {
        assert_eq!(
            Config::<Field64>::ood_samples(
                1000, // Extremely high security level
                SoundnessType::ConjectureList,
                10,  // num_variables
                5.0, // log_inv_rate
                2.0, // log_eta
                256, // field_size_bits
            ),
            5
        );
    }
}
