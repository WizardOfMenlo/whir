use core::panic;
use std::{f64::consts::LOG2_10, fmt::Display, ops::Neg};

use ark_ff::FftField;

use super::{Config, RoundConfig};
use crate::{
    algebra::{
        embedding::{self, Embedding},
        fields::FieldWithSize,
    },
    bits::Bits,
    parameters::ProtocolParameters,
    protocols::{irs_commit, matrix_commit, proof_of_work, sumcheck},
    type_info::{Type, Typed},
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
        let num_variables = size.trailing_zeros() as usize;
        let initial_num_variables = num_variables;
        let initial_folding_factor = whir_parameters.initial_folding_factor;
        let folding_factor = whir_parameters.folding_factor;

        // Proof of work constructor with the requested hash function.
        let pow = |difficulty| proof_of_work::Config {
            hash_id: whir_parameters.hash_id,
            threshold: proof_of_work::threshold(Bits::new(difficulty)),
        };

        let protocol_security_level = whir_parameters
            .security_level
            .saturating_sub(whir_parameters.pow_bits);
        let field_size_bits = M::Target::field_size_in_bits();
        let mut log_inv_rate = whir_parameters.starting_log_inv_rate;
        let mut num_variables = initial_num_variables;

        let mut domain_size = 1 << (initial_num_variables + log_inv_rate);

        let log_eta_start = if whir_parameters.unique_decoding {
            0.0
        } else {
            Self::log_eta(log_inv_rate as f64)
        };

        let commitment_ood_samples = if !whir_parameters.unique_decoding {
            Self::ood_samples(
                whir_parameters.security_level,
                num_variables,
                log_inv_rate as f64,
                log_eta_start,
                field_size_bits,
            )
        } else {
            0
        };

        // Initial sumcheck round pow bits.
        let starting_folding_pow_bits = Self::folding_pow_bits(
            whir_parameters.security_level,
            whir_parameters.unique_decoding,
            field_size_bits,
            num_variables,
            log_inv_rate as f64,
            log_eta_start,
        );
        // If we skip the initial sumcheck, we do this pow instead:
        let initial_skip_pow_bits = {
            let prox_gaps_error = Self::rbr_soundness_fold_prox_gaps(
                whir_parameters.unique_decoding,
                field_size_bits,
                num_variables,
                log_inv_rate as f64,
                log_eta_start,
            ) + (initial_folding_factor as f64).log2();
            (whir_parameters.security_level as f64 - prox_gaps_error).max(0.0)
        };

        let mut round_parameters = Vec::new();
        let mut round = 0;
        num_variables -= initial_folding_factor;
        while num_variables >= folding_factor {
            // Queries are set w.r.t. to old rate, while the rest to the new rate
            let round_folding_factor = if round == 0 {
                initial_folding_factor
            } else {
                folding_factor
            };
            let next_rate = log_inv_rate + (round_folding_factor - 1);

            let log_next_eta = if whir_parameters.unique_decoding {
                0.0
            } else {
                Self::log_eta(next_rate as f64)
            };

            let num_queries = Self::queries(
                whir_parameters.unique_decoding,
                protocol_security_level,
                log_inv_rate,
            );

            let ood_samples = if whir_parameters.unique_decoding {
                0
            } else {
                Self::ood_samples(
                    whir_parameters.security_level,
                    num_variables,
                    next_rate as f64,
                    log_next_eta,
                    field_size_bits,
                )
            };

            let query_error = Self::rbr_queries(
                whir_parameters.unique_decoding,
                log_inv_rate as f64,
                num_queries,
            );

            let combination_error = Self::rbr_soundness_queries_combination(
                whir_parameters.unique_decoding,
                field_size_bits,
                next_rate as f64,
                log_next_eta,
                ood_samples,
                num_queries,
            );

            let pow_bits = 0_f64
                .max(whir_parameters.security_level as f64 - (query_error.min(combination_error)));

            let folding_pow_bits = Self::folding_pow_bits(
                whir_parameters.security_level,
                whir_parameters.unique_decoding,
                field_size_bits,
                num_variables,
                next_rate as f64,
                log_next_eta,
            );

            let next_folding_factor = folding_factor;
            let matrix_committer = matrix_commit::Config::<M::Target>::with_hash(
                whir_parameters.hash_id,
                (domain_size / 2) >> next_folding_factor,
                1 << next_folding_factor,
            );

            round_parameters.push(RoundConfig {
                irs_committer: irs_commit::Config {
                    embedding: Typed::new(embedding::Identity::new()),
                    num_vectors: 1,
                    vector_size: 1 << num_variables,
                    codeword_length: 1 << (num_variables + next_rate - next_folding_factor),
                    interleaving_depth: 1 << next_folding_factor,
                    matrix_commit: matrix_committer.clone(),
                    in_domain_samples: Self::queries(
                        whir_parameters.unique_decoding,
                        protocol_security_level,
                        next_rate,
                    ),
                    out_domain_samples: ood_samples,
                    deduplicate_in_domain: true, // TODO: Configurable
                },
                sumcheck: sumcheck::Config {
                    field: Type::new(),
                    initial_size: 1 << num_variables,
                    round_pow: pow(folding_pow_bits),
                    num_rounds: next_folding_factor,
                },
                pow: pow(pow_bits),
            });

            round += 1;
            num_variables -= next_folding_factor;
            log_inv_rate = next_rate;
            domain_size /= 2;
        }

        let final_queries = Self::queries(
            whir_parameters.unique_decoding,
            protocol_security_level,
            log_inv_rate,
        );

        let final_pow_bits = 0_f64.max(
            whir_parameters.security_level as f64
                - Self::rbr_queries(
                    whir_parameters.unique_decoding,
                    log_inv_rate as f64,
                    final_queries,
                ),
        );

        let final_folding_pow_bits =
            0_f64.max(whir_parameters.security_level as f64 - (field_size_bits - 1) as f64);

        Self {
            initial_committer: irs_commit::Config {
                embedding: Default::default(),
                num_vectors: whir_parameters.batch_size,
                vector_size: 1 << initial_num_variables,
                codeword_length: 1
                    << (whir_parameters.starting_log_inv_rate + initial_num_variables
                        - initial_folding_factor),
                interleaving_depth: 1 << initial_folding_factor,
                matrix_commit: matrix_commit::Config::with_hash(
                    whir_parameters.hash_id,
                    1 << (initial_num_variables + whir_parameters.starting_log_inv_rate
                        - initial_folding_factor),
                    whir_parameters.batch_size << initial_folding_factor,
                ),
                in_domain_samples: Self::queries(
                    whir_parameters.unique_decoding,
                    protocol_security_level,
                    whir_parameters.starting_log_inv_rate,
                ),
                out_domain_samples: commitment_ood_samples,
                deduplicate_in_domain: true,
            },
            initial_sumcheck: sumcheck::Config {
                field: Type::new(),
                initial_size: 1 << initial_num_variables,
                round_pow: pow(starting_folding_pow_bits),
                num_rounds: initial_folding_factor,
            },
            initial_skip_pow: pow(initial_skip_pow_bits),
            round_configs: round_parameters,
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
        let field_size_bits = M::Target::field_size_in_bits();
        let mut num_variables = self.initial_num_variables();
        let mut security_level = f64::INFINITY;

        let initial_vector_rlc =
            Self::rbr_soundness_initial_rlc_combination(field_size_bits, num_vectors);
        security_level = security_level.min(initial_vector_rlc);

        let initial_covector_rlc =
            Self::rbr_soundness_initial_rlc_combination(field_size_bits, num_linear_forms);
        security_level = security_level.min(initial_covector_rlc);

        let initial_log_inv_rate = self.initial_committer.rate().log2().neg();
        let initial_unique_decoding = self.initial_committer.unique_decoding();
        let initial_log_eta = if initial_unique_decoding {
            0.0
        } else {
            Self::log_eta(initial_log_inv_rate)
        };

        if !initial_unique_decoding {
            let ood_error = Self::rbr_ood_sample(
                num_variables,
                initial_log_inv_rate,
                initial_log_eta,
                field_size_bits,
                self.initial_committer.out_domain_samples,
            );
            security_level = security_level.min(ood_error);
        }

        let initial_prox_gaps_error = Self::rbr_soundness_fold_prox_gaps(
            initial_unique_decoding,
            field_size_bits,
            num_variables,
            initial_log_inv_rate,
            initial_log_eta,
        );
        let initial_sumcheck_error = Self::rbr_soundness_fold_sumcheck(
            initial_unique_decoding,
            field_size_bits,
            initial_log_inv_rate,
            initial_log_eta,
        );
        let initial_fold_error = initial_prox_gaps_error.min(initial_sumcheck_error)
            + f64::from(self.initial_sumcheck.round_pow.difficulty());
        security_level = security_level.min(initial_fold_error);

        num_variables -= self.initial_sumcheck.num_rounds;

        for round in &self.round_configs {
            let round_unique_decoding = round.irs_committer.unique_decoding();
            // Query soundness is computed at the old rate, while all fold and OOD terms use the new rate.
            let round_log_inv_rate = round.log_inv_rate() as f64;
            let next_log_inv_rate = (round.log_inv_rate() + (round.sumcheck.num_rounds - 1)) as f64;
            let log_eta = if round_unique_decoding {
                0.0
            } else {
                Self::log_eta(next_log_inv_rate)
            };

            if !round_unique_decoding {
                let ood_error = Self::rbr_ood_sample(
                    num_variables,
                    next_log_inv_rate,
                    log_eta,
                    field_size_bits,
                    round.irs_committer.out_domain_samples,
                );
                security_level = security_level.min(ood_error);
            }

            let query_error = Self::rbr_queries(
                round_unique_decoding,
                round_log_inv_rate,
                round.irs_committer.in_domain_samples,
            );
            let combination_error = Self::rbr_soundness_queries_combination(
                round_unique_decoding,
                field_size_bits,
                next_log_inv_rate,
                log_eta,
                round.irs_committer.out_domain_samples,
                round.irs_committer.in_domain_samples,
            );
            let round_query_error =
                query_error.min(combination_error) + f64::from(round.pow.difficulty());
            security_level = security_level.min(round_query_error);

            let prox_gaps_error = Self::rbr_soundness_fold_prox_gaps(
                round_unique_decoding,
                field_size_bits,
                num_variables,
                next_log_inv_rate,
                log_eta,
            );
            let sumcheck_error = Self::rbr_soundness_fold_sumcheck(
                round_unique_decoding,
                field_size_bits,
                next_log_inv_rate,
                log_eta,
            );
            let round_fold_error = prox_gaps_error.min(sumcheck_error)
                + f64::from(round.sumcheck.round_pow.difficulty());
            security_level = security_level.min(round_fold_error);

            num_variables -= round.sumcheck.num_rounds;
        }

        let final_unique_decoding = self
            .round_configs
            .last()
            .map(|round| round.irs_committer.unique_decoding())
            .unwrap_or(initial_unique_decoding);

        let final_query_error = Self::rbr_queries(
            final_unique_decoding,
            self.final_rate().log2().neg(),
            self.final_in_domain_samples(),
        ) + f64::from(self.final_pow.difficulty());
        security_level = security_level.min(final_query_error);

        if self.final_sumcheck.num_rounds > 0 {
            let final_combination_error =
                field_size_bits as f64 - 1. + f64::from(self.final_sumcheck.round_pow.difficulty());
            security_level = security_level.min(final_combination_error);
        }

        if security_level.is_finite() {
            security_level
        } else {
            0.0
        }
    }

    pub fn rbr_soundness_initial_rlc_combination(field_size_bits: usize, num_terms: usize) -> f64 {
        if num_terms <= 1 {
            f64::INFINITY
        } else {
            field_size_bits as f64 - ((num_terms - 1) as f64).log2()
        }
    }

    /// Construct a suitable $log_2 η$, where $η$ is the gap to the Johnson bound.
    ///
    /// It is related proiximity distance by
    ///
    /// $δ = 1 - sqrt(rate) - η$
    pub const fn log_eta(log_inv_rate: f64) -> f64 {
        // Ask me how I did this? At the time, only God and I knew. Now only God knows
        // This computes $η = sqrt(ρ) / 20$, which amounts to a Johnson slack of $m = 10$.
        // TODO: This seems like a high choice of m ?
        -(0.5 * log_inv_rate + LOG2_10 + 1.)
    }

    /// Compute $log_2 L(δ)$, where $L(δ)$ is the Johnson Reed-Solomon list size bound.
    ///
    /// This is the Johnson bound $1 / (2 * η ρ)$.
    pub const fn log_list_size(log_inv_rate: f64, log_eta: f64) -> f64 {
        0.5 * log_inv_rate - (1. + log_eta)
    }

    pub const fn rbr_ood_sample(
        num_variables: usize,
        log_inv_rate: f64,
        log_eta: f64,
        field_size_bits: usize,
        ood_samples: usize,
    ) -> f64 {
        let list_size_bits = Self::log_list_size(log_inv_rate, log_eta);

        let error = 2. * list_size_bits + (num_variables * ood_samples) as f64;
        (ood_samples * field_size_bits) as f64 + 1. - error
    }

    /// Compute the minimal number of out-of-domain samples to reach the security level.
    pub fn ood_samples(
        security_level: usize, // We don't do PoW for OOD
        num_variables: usize,
        log_inv_rate: f64,
        log_eta: f64,
        field_size_bits: usize,
    ) -> usize {
        (1..64)
            .find(|&ood_samples| {
                Self::rbr_ood_sample(
                    num_variables,
                    log_inv_rate,
                    log_eta,
                    field_size_bits,
                    ood_samples,
                ) >= security_level as f64
            })
            .unwrap_or_else(|| panic!("Could not find an appropriate number of OOD samples"))
    }

    // Compute the proximity gaps term of the fold
    pub fn rbr_soundness_fold_prox_gaps(
        unique_decoding: bool,
        field_size_bits: usize,
        num_variables: usize,
        log_inv_rate: f64,
        log_eta: f64,
    ) -> f64 {
        // See WHIR Theorem 4.8
        // Recall, at each round we are only folding by two at a time
        let error = if unique_decoding {
            num_variables as f64 + log_inv_rate
        } else {
            // Make sure η hits the min bound.
            assert!(log_eta >= -(0.5 * log_inv_rate + LOG2_10 + 1.0));
            7. * LOG2_10 + 3.5 * log_inv_rate + 2. * num_variables as f64
        };
        field_size_bits as f64 - error
    }

    pub const fn rbr_soundness_fold_sumcheck(
        unique_decoding: bool,
        field_size_bits: usize,
        log_inv_rate: f64,
        log_eta: f64,
    ) -> f64 {
        let log_list_size = if unique_decoding {
            0.0
        } else {
            Self::log_list_size(log_inv_rate, log_eta)
        };

        field_size_bits as f64 - (log_list_size + 1.)
    }

    pub fn folding_pow_bits(
        security_level: usize,
        unique_decoding: bool,
        field_size_bits: usize,
        num_variables: usize,
        log_inv_rate: f64,
        log_eta: f64,
    ) -> f64 {
        let prox_gaps_error = Self::rbr_soundness_fold_prox_gaps(
            unique_decoding,
            field_size_bits,
            num_variables,
            log_inv_rate,
            log_eta,
        );
        let sumcheck_error = Self::rbr_soundness_fold_sumcheck(
            unique_decoding,
            field_size_bits,
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
        unique_decoding: bool,
        protocol_security_level: usize,
        log_inv_rate: usize,
    ) -> usize {
        let num_queries_f = if unique_decoding {
            let rate = 1. / f64::from(1 << log_inv_rate);
            let denom = (0.5 * (1. + rate)).log2();
            -(protocol_security_level as f64) / denom
        } else {
            (2 * protocol_security_level) as f64 / log_inv_rate as f64
        };
        num_queries_f.ceil() as usize
    }

    // This is the bits of security of the query step
    pub fn rbr_queries(unique_decoding: bool, log_inv_rate: f64, num_queries: usize) -> f64 {
        let num_queries = num_queries as f64;

        if unique_decoding {
            // (1 - δ)^q for δ = (1 - ρ) / 2.
            let rate = 1. / log_inv_rate.exp2();
            let denom = -(0.5 * (1. + rate)).log2();
            num_queries * denom
        } else {
            num_queries * 0.5 * log_inv_rate
        }
    }

    pub fn rbr_soundness_queries_combination(
        unique_decoding: bool,
        field_size_bits: usize,
        log_inv_rate: f64,
        log_eta: f64,
        ood_samples: usize,
        num_queries: usize,
    ) -> f64 {
        let list_size = if unique_decoding {
            0.0
        } else {
            Self::log_list_size(log_inv_rate, log_eta)
        };

        let log_combination = ((ood_samples + num_queries) as f64).log2();

        field_size_bits as f64 - (log_combination + list_size + 1.)
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
            "Security level: {} bits using {} decoding",
            self.security_level(self.initial_committer.num_vectors, 1),
            if self.unique_decoding() {
                "unique"
            } else {
                "list"
            }
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

        let field_size_bits = M::Target::field_size_in_bits();
        let mut num_variables = self.initial_num_variables();

        let vector_rlc_error = Self::rbr_soundness_initial_rlc_combination(
            field_size_bits,
            self.initial_committer.num_vectors,
        );
        if vector_rlc_error.is_finite() {
            writeln!(
                f,
                "{:.1} bits -- initial vector RLC ({} vectors)",
                vector_rlc_error, self.initial_committer.num_vectors
            )?;
        } else {
            writeln!(
                f,
                "no loss -- initial vector RLC ({} vector)",
                self.initial_committer.num_vectors
            )?;
        }

        let num_linear_forms = 10;
        let linear_form_rlc_error =
            Self::rbr_soundness_initial_rlc_combination(field_size_bits, num_linear_forms);
        if linear_form_rlc_error.is_finite() {
            writeln!(
                f,
                "{:.1} bits -- initial linear-form RLC ({} linear form)",
                linear_form_rlc_error, num_linear_forms
            )?;
        } else {
            writeln!(
                f,
                "no loss -- initial linear-form RLC ({} linear form)",
                num_linear_forms
            )?;
        }

        let log_eta = if self.initial_committer.unique_decoding() {
            0.0
        } else {
            let log_eta = Self::log_eta(self.initial_committer.rate().log2().neg());
            writeln!(
                f,
                "{:.1} bits -- OOD commitment",
                Self::rbr_ood_sample(
                    num_variables,
                    self.initial_committer.rate().log2().neg(),
                    log_eta,
                    field_size_bits,
                    self.initial_committer.out_domain_samples
                )
            )?;
            log_eta
        };

        let prox_gaps_error = Self::rbr_soundness_fold_prox_gaps(
            self.initial_committer.unique_decoding(),
            field_size_bits,
            num_variables,
            self.initial_committer.rate().log2().neg(),
            log_eta,
        );
        let sumcheck_error = Self::rbr_soundness_fold_sumcheck(
            self.initial_committer.unique_decoding(),
            field_size_bits,
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
            let log_eta = if r.irs_committer.unique_decoding() {
                0.0
            } else {
                Self::log_eta(next_rate)
            };

            if !r.irs_committer.unique_decoding() {
                writeln!(
                    f,
                    "{:.1} bits -- OOD sample",
                    Self::rbr_ood_sample(
                        num_variables,
                        next_rate,
                        log_eta,
                        field_size_bits,
                        r.irs_committer.out_domain_samples
                    )
                )?;
            }

            let query_error = Self::rbr_queries(
                r.irs_committer.unique_decoding(),
                r.log_inv_rate() as f64,
                r.irs_committer.in_domain_samples,
            );
            let combination_error = Self::rbr_soundness_queries_combination(
                r.irs_committer.unique_decoding(),
                field_size_bits,
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
                r.irs_committer.unique_decoding(),
                field_size_bits,
                num_variables,
                next_rate,
                log_eta,
            );
            let sumcheck_error = Self::rbr_soundness_fold_sumcheck(
                r.irs_committer.unique_decoding(),
                field_size_bits,
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

        let last_unique = self
            .round_configs
            .last()
            .map(|r| r.irs_committer.unique_decoding())
            .unwrap_or_else(|| self.initial_committer.unique_decoding());

        let query_error = Self::rbr_queries(
            last_unique,
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
    use super::*;
    use crate::{
        algebra::{
            embedding::{Basefield, Identity},
            fields::{Field64, Field64_3},
        },
        bits::Bits,
        hash,
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
    fn test_folding_pow_bits() {
        let field_size_bits = 64;
        let unique_decoding = false;

        let pow_bits = Config::<Basefield<Field64_3>>::folding_pow_bits(
            100, // Security level
            unique_decoding,
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

        let result = Config::<Basefield<Field64_3>>::queries(true, security_level, log_inv_rate);

        assert_eq!(result, 105);
    }

    #[test]
    fn test_queries_provable_list() {
        let security_level = 128;
        let log_inv_rate = 8;

        let result = Config::<Basefield<Field64_3>>::queries(false, security_level, log_inv_rate);

        assert_eq!(result, 32);
    }

    #[test]
    fn test_queries_conjecture_list() {
        let security_level = 256;
        let log_inv_rate = 16;

        let result = Config::<Basefield<Field64_3>>::queries(false, security_level, log_inv_rate);

        assert_eq!(result, 32);
    }

    #[test]
    fn test_rbr_queries_unique_decoding() {
        let log_inv_rate = 5.0; // log_inv_rate = 5
        let num_queries = 10; // Number of queries

        let result = Config::<Basefield<Field64_3>>::rbr_queries(true, log_inv_rate, num_queries);

        assert!((result - 9.556_058_806_415_466).abs() < 1e-6);
    }

    #[test]
    fn test_rbr_queries_provable_list() {
        let log_inv_rate = 8.0; // log_inv_rate = 8
        let num_queries = 16; // Number of queries

        let result = Config::<Basefield<Field64_3>>::rbr_queries(false, log_inv_rate, num_queries);

        assert!((result - 64.0) < 1e-6);
    }

    #[test]
    fn test_rbr_queries_conjecture_list() {
        let log_inv_rate = 4.0; // log_inv_rate = 4
        let num_queries = 20; // Number of queries

        let result = Config::<Basefield<Field64_3>>::rbr_queries(false, log_inv_rate, num_queries);

        assert!((result - 40.) < 1e-6);
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
            let result = Config::<Basefield<Field64_3>>::log_list_size(log_inv_rate, log_eta);
            assert!(
                (result - expected).abs() < 1e-6,
                "Failed for {:?}",
                (num_variables, log_inv_rate, log_eta)
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
            let result = Config::<Basefield<Field64_3>>::rbr_ood_sample(
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
    fn test_ood_samples_valid_case() {
        // Testing a valid case where the function finds an appropriate `ood_samples`
        assert_eq!(
            Config::<Basefield<Field64_3>>::ood_samples(
                50,  // security level
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
            Config::<Identity<Field64>>::ood_samples(
                30,  // Lower security level
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
            Config::<Identity<Field64>>::ood_samples(
                100,  // High security level
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
            Config::<Identity<Field64>>::ood_samples(
                1000, // Extremely high security level
                10,   // num_variables
                5.0,  // log_inv_rate
                2.0,  // log_eta
                256,  // field_size_bits
            ),
            5
        );
    }
}
