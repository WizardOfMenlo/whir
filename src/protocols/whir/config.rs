use std::fmt::Display;

use ark_ff::FftField;

use super::{Config, RoundConfig};
use crate::{
    algebra::{embedding::Embedding, fields::FieldWithSize},
    bits::Bits,
    parameters::ProtocolParameters,
    protocols::{irs_commit, proof_of_work, sumcheck},
    type_info::Type,
};

impl<M: Embedding> Config<M>
where
    M::Source: FftField,
    M::Target: FftField,
{
    #[allow(clippy::too_many_lines)]
    pub fn new(size: usize, whir_parameters: &ProtocolParameters) -> Self
    where
        M: Default,
    {
        assert!(
            size.is_power_of_two(),
            "Only powers of two size are supported at the moment."
        );

        // Proof of work constructor with the requested hash function.
        let pow = |difficulty| proof_of_work::Config {
            hash_id: whir_parameters.hash_id,
            threshold: proof_of_work::threshold(Bits::new(difficulty)),
        };

        let security_level = whir_parameters.security_level as f64;
        let protocol_security_level = whir_parameters
            .security_level
            .saturating_sub(whir_parameters.pow_bits) as f64;
        let field_size_bits = M::Target::field_size_bits();
        let mut log_inv_rate = whir_parameters.starting_log_inv_rate;
        let mut num_variables = size.trailing_zeros() as usize;

        #[allow(clippy::cast_possible_wrap)]
        let initial_committer = irs_commit::Config::new(
            protocol_security_level,
            whir_parameters.unique_decoding,
            whir_parameters.hash_id,
            whir_parameters.batch_size,
            size,
            1 << whir_parameters.initial_folding_factor,
            0.5_f64.powi(whir_parameters.starting_log_inv_rate as i32),
        );

        // Initial sumcheck round pow bits.
        let starting_folding_pow_bits = {
            let prox_gaps_error = initial_committer.rbr_soundness_fold_prox_gaps();
            let log_list_size = initial_committer.list_size().log2();
            let sumcheck_error = field_size_bits - log_list_size - 1.;
            let error = prox_gaps_error.min(sumcheck_error);
            (security_level - error).max(0.)
        };
        // If we skip the initial sumcheck, we do this pow instead:
        let initial_skip_pow_bits = {
            let prox_gaps_error = initial_committer.rbr_soundness_fold_prox_gaps()
                + (whir_parameters.initial_folding_factor as f64).log2();
            (security_level - prox_gaps_error).max(0.0)
        };

        let mut round_configs = Vec::new();
        let mut round = 0;
        num_variables -= whir_parameters.initial_folding_factor;
        while num_variables >= whir_parameters.folding_factor {
            // Queries are set w.r.t. to old rate, while the rest to the new rate
            let round_folding_factor = if round == 0 {
                whir_parameters.initial_folding_factor
            } else {
                whir_parameters.folding_factor
            };
            let next_rate = log_inv_rate + (round_folding_factor - 1);

            #[allow(clippy::cast_possible_wrap)]
            let irs_committer = irs_commit::Config::new(
                protocol_security_level,
                whir_parameters.unique_decoding,
                whir_parameters.hash_id,
                1,
                1 << num_variables,
                1 << whir_parameters.folding_factor,
                0.5_f64.powi(next_rate as i32),
            );
            let combination_error = {
                let log_list_size = irs_committer.list_size().log2();
                let count = irs_committer.out_domain_samples + irs_committer.in_domain_samples;
                let log_combination = (count as f64).log2();
                field_size_bits - (log_combination + log_list_size + 1.)
            };
            let pow_bits =
                0_f64.max(security_level - (irs_committer.rbr_queries().min(combination_error)));
            let folding_pow_bits = {
                let prox_gaps_error = irs_committer.rbr_soundness_fold_prox_gaps();
                let log_list_size = irs_committer.list_size().log2();
                let sumcheck_error = field_size_bits - (log_list_size + 1.);
                let error = prox_gaps_error.min(sumcheck_error);
                (security_level - error).max(0.)
            };

            round_configs.push(RoundConfig {
                irs_committer,
                sumcheck: sumcheck::Config {
                    field: Type::new(),
                    initial_size: 1 << num_variables,
                    round_pow: pow(folding_pow_bits),
                    num_rounds: whir_parameters.folding_factor,
                },
                pow: pow(pow_bits),
            });

            round += 1;
            num_variables -= whir_parameters.folding_factor;
            log_inv_rate = next_rate;
        }

        let rbr_error = round_configs.last().map_or_else(
            || initial_committer.rbr_queries(),
            |r| r.irs_committer.rbr_queries(),
        );
        let final_pow_bits = 0_f64.max(security_level - rbr_error);

        let final_folding_pow_bits = 0_f64.max(security_level - field_size_bits - 1.0);

        Self {
            initial_committer,
            initial_sumcheck: sumcheck::Config {
                field: Type::new(),
                initial_size: size,
                round_pow: pow(starting_folding_pow_bits),
                num_rounds: whir_parameters.initial_folding_factor,
            },
            initial_skip_pow: pow(initial_skip_pow_bits),
            round_configs,
            final_sumcheck: sumcheck::Config {
                field: Type::new(),
                initial_size: 1 << num_variables,
                round_pow: pow(final_folding_pow_bits),
                num_rounds: num_variables,
            },
            final_pow: pow(final_pow_bits),
        }
    }

    // True if we only use the unique decoding regime.
    pub fn unique_decoding(&self) -> bool {
        self.initial_committer.unique_decoding()
            && self
                .round_configs
                .iter()
                .all(|r| r.irs_committer.unique_decoding())
    }

    pub fn security_level(&self, num_vectors: usize, num_linear_forms: usize) -> f64 {
        let field_size_bits = M::Target::field_size_bits();
        let mut security_level = f64::INFINITY;
        if num_vectors > 1 {
            security_level =
                security_level.min(field_size_bits - ((num_vectors - 1) as f64).log2());
        }
        if num_linear_forms > 1 {
            security_level =
                security_level.min(field_size_bits - ((num_linear_forms - 1) as f64).log2());
        }
        let has_initial_constraints =
            num_linear_forms > 0 || self.initial_committer.out_domain_samples > 0;

        if !self.initial_committer.unique_decoding() {
            security_level = security_level.min(self.initial_committer.rbr_ood_sample());
        }

        // Initial sumcheck error (or the skipped version for LDT).
        let initial_prox_gaps_error = self.initial_committer.rbr_soundness_fold_prox_gaps();
        if has_initial_constraints {
            let log_list_size = self.initial_committer.list_size().log2();
            let initial_sumcheck_error = field_size_bits - (log_list_size + 1.);
            let initial_fold_error = initial_prox_gaps_error.min(initial_sumcheck_error)
                + f64::from(self.initial_sumcheck.round_pow.difficulty());
            security_level = security_level.min(initial_fold_error);
        } else {
            let skipped_initial_fold_error = initial_prox_gaps_error
                + (self.initial_sumcheck.num_rounds as f64).log2()
                + f64::from(self.initial_skip_pow.difficulty());
            security_level = security_level.min(skipped_initial_fold_error);
        }

        let mut rbr_queries = self.initial_committer.rbr_queries();
        let mut old_in_domain_samples = self.initial_committer.in_domain_samples;
        for round in &self.round_configs {
            // Query soundness is computed at the old rate, while all fold and OOD terms use the new rate.
            let new_unique_decoding = round.irs_committer.unique_decoding();

            if !new_unique_decoding {
                let ood_error = round.irs_committer.rbr_ood_sample();
                security_level = security_level.min(ood_error);
            }

            let log_list_size = round.irs_committer.list_size().log2();
            let combination_error = {
                let count = round.irs_committer.out_domain_samples + old_in_domain_samples;
                let log_combination = (count as f64).log2();
                field_size_bits - (log_combination + log_list_size + 1.)
            };
            let round_query_error =
                rbr_queries.min(combination_error) + f64::from(round.pow.difficulty());
            security_level = security_level.min(round_query_error);

            let prox_gaps_error = round.irs_committer.rbr_soundness_fold_prox_gaps();
            let sumcheck_error = field_size_bits - (log_list_size + 1.);
            let round_fold_error = prox_gaps_error.min(sumcheck_error)
                + f64::from(round.sumcheck.round_pow.difficulty());
            security_level = security_level.min(round_fold_error);

            old_in_domain_samples = round.irs_committer.in_domain_samples;
            rbr_queries = round.irs_committer.rbr_queries();
        }

        let final_query_error = rbr_queries + f64::from(self.final_pow.difficulty());
        security_level = security_level.min(final_query_error);

        if self.final_sumcheck.num_rounds > 0 {
            let final_combination_error =
                field_size_bits - 1. + f64::from(self.final_sumcheck.round_pow.difficulty());
            security_level = security_level.min(final_combination_error);
        }

        if security_level.is_finite() {
            security_level
        } else {
            0.0
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

    pub fn embedding(&self) -> &M {
        self.initial_committer.embedding()
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

impl<M: Embedding> Display for Config<M>
where
    M::Source: FftField,
    M::Target: FftField,
{
    #[allow(clippy::too_many_lines)]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "Security level: {:.2} bits using {} decoding",
            self.security_level(self.initial_committer.num_vectors, 1),
            if self.unique_decoding() {
                "unique"
            } else {
                "list"
            }
        )?;
        writeln!(
            f,
            "Source field: {:.2} bits, target field: {:.2} bits",
            M::Source::field_size_bits(),
            M::Target::field_size_bits()
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

        let field_size_bits = M::Target::field_size_bits();
        let num_vectors = self.initial_committer.num_vectors;
        let num_linear_forms = 10; // TODO
        if num_vectors > 1 {
            let rlc_error = field_size_bits - ((num_vectors - 1) as f64).log2();
            writeln!(
                f,
                "{rlc_error:.1} bits -- initial vector RLC ({num_vectors} vectors)"
            )?;
        } else {
            writeln!(f, "no loss -- initial vector RLC ({num_vectors} vector)")?;
        }
        if num_linear_forms > 1 {
            let rlc_error = field_size_bits - f64::from(num_linear_forms - 1).log2();
            writeln!(
                f,
                "{rlc_error:.1} bits -- initial linear-form RLC ({num_linear_forms} linear form)"
            )?;
        } else {
            writeln!(
                f,
                "no loss -- initial linear-form RLC ({num_linear_forms} linear form)"
            )?;
        }

        if !self.initial_committer.unique_decoding() {
            writeln!(
                f,
                "{:.1} bits -- OOD commitment",
                self.initial_committer.rbr_ood_sample()
            )?;
        }
        let prox_gaps_error = self.initial_committer.rbr_soundness_fold_prox_gaps();
        let log_list_size = self.initial_committer.list_size().log2();
        let sumcheck_error = field_size_bits - (log_list_size + 1.);
        writeln!(
            f,
            "{:.1} bits -- (x{}) prox gaps: {:.1}, sumcheck: {:.1}, pow: {:.1}, (list size 2^{:.1})",
            prox_gaps_error.min(sumcheck_error)
                + f64::from(self.initial_sumcheck.round_pow.difficulty()),
            self.initial_sumcheck.num_rounds,
            prox_gaps_error,
            sumcheck_error,
            self.initial_sumcheck.round_pow.difficulty(),
            log_list_size,
        )?;

        let mut query_error = self.initial_committer.rbr_queries();
        let mut old_in_domain_samples = self.initial_committer.in_domain_samples;
        for r in &self.round_configs {
            if !r.irs_committer.unique_decoding() {
                writeln!(
                    f,
                    "{:.1} bits -- OOD sample",
                    r.irs_committer.rbr_ood_sample()
                )?;
            }

            let log_list_size = r.irs_committer.list_size().log2();
            let combination_error = {
                let count = r.irs_committer.out_domain_samples + old_in_domain_samples;
                let log_combination = (count as f64).log2();
                field_size_bits - (log_combination + log_list_size + 1.)
            };
            writeln!(
                f,
                "{:.1} bits -- query error: {:.1}, combination: {:.1}, pow: {:.1}",
                query_error.min(combination_error) + f64::from(r.pow.difficulty()),
                query_error,
                combination_error,
                r.pow.difficulty(),
            )?;

            let prox_gaps_error = r.irs_committer.rbr_soundness_fold_prox_gaps();
            let sumcheck_error = field_size_bits - (log_list_size + 1.);
            writeln!(
                f,
                "{:.1} bits -- (x{}) prox gaps: {:.1}, sumcheck: {:.1}, pow: {:.1} (list size 2^{:.1})",
                prox_gaps_error.min(sumcheck_error) + f64::from(r.sumcheck.round_pow.difficulty()),
                r.sumcheck.num_rounds,
                prox_gaps_error,
                sumcheck_error,
                r.sumcheck.round_pow.difficulty(),
                log_list_size
            )?;

            old_in_domain_samples = r.irs_committer.in_domain_samples;
            query_error = r.irs_committer.rbr_queries();
        }

        writeln!(
            f,
            "{:.1} bits -- query error: {:.1}, pow: {:.1}",
            query_error + f64::from(self.final_pow.difficulty()),
            query_error,
            self.final_pow.difficulty(),
        )?;

        if self.final_sumcheck.num_rounds > 0 {
            let combination_error = field_size_bits - 1.;
            writeln!(
                f,
                "{:.1} bits -- (x{}) combination: {:.1}, pow: {:.1}",
                combination_error + f64::from(self.final_sumcheck.round_pow.difficulty()),
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
        assert!(self
            .irs_committer
            .codeword_length
            .is_multiple_of(self.irs_committer.message_length()));
        let expansion = self.irs_committer.codeword_length / self.irs_committer.message_length();
        assert!(expansion.is_power_of_two());
        expansion.ilog2() as usize
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
    use ordered_float::OrderedFloat;

    use super::*;
    use crate::{
        algebra::{
            embedding::{self, Basefield},
            fields::Field64_3,
        },
        bits::Bits,
        hash,
        protocols::matrix_commit,
        type_info::Typed,
        utils::test_serde,
    };

    /// Generates default WHIR parameters
    fn default_whir_params() -> ProtocolParameters {
        ProtocolParameters {
            security_level: 80, // We can't hope for much with a 128bit field.
            pow_bits: 20,
            initial_folding_factor: 4,
            folding_factor: 4,
            unique_decoding: false,
            starting_log_inv_rate: 1,
            batch_size: 1,
            hash_id: hash::BLAKE3,
        }
    }

    #[test]
    fn test_whir_params_serde() {
        test_serde(&default_whir_params());
    }

    #[test]
    fn test_whir_config_serde() {
        let params = default_whir_params();

        let config = Config::<Basefield<Field64_3>>::new(1 << 10, &params);

        test_serde(&config);
    }

    #[test]
    fn test_n_rounds() {
        let params = default_whir_params();
        let config = Config::<Basefield<Field64_3>>::new(1 << 10, &params);

        assert_eq!(config.n_rounds(), config.round_configs.len());
    }

    #[test]
    fn test_check_pow_bits_within_limits() {
        let params = default_whir_params();
        let mut config = Config::<Basefield<Field64_3>>::new(1 << 10, &params);

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
                    codeword_length: 1 << (10 + 3 - 2),
                    interleaving_depth: 1 << 2,
                    matrix_commit: matrix_commit::Config::<Field64_3>::new(0, 0),
                    johnson_slack: OrderedFloat::default(),
                    in_domain_samples: 5,
                    out_domain_samples: 2,
                    deduplicate_in_domain: true,
                },
                sumcheck: sumcheck::Config {
                    field: Type::<Field64_3>::new(),
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
                    codeword_length: 1 << (10 + 4 - 2),
                    interleaving_depth: 1 << 2,
                    matrix_commit: matrix_commit::Config::<Field64_3>::new(0, 0),
                    johnson_slack: OrderedFloat::default(),
                    in_domain_samples: 6,
                    out_domain_samples: 2,
                    deduplicate_in_domain: true,
                },
                sumcheck: sumcheck::Config {
                    field: Type::<Field64_3>::new(),
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
        let mut config = Config::<Basefield<Field64_3>>::new(1 << 10, &params);

        config.initial_sumcheck.round_pow = proof_of_work::Config::from_difficulty(Bits::new(21.0));
        config.final_pow = proof_of_work::Config::from_difficulty(Bits::new(18.0));
        config.final_sumcheck.round_pow = proof_of_work::Config::from_difficulty(Bits::new(19.5));

        assert!(
            !config.check_max_pow_bits(Bits::new(20.0)),
            "Starting folding pow bits exceeds max_pow_bits, should return false."
        );
    }
}
