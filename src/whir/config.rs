use core::panic;
use std::{
    f64::consts::LOG2_10,
    fmt::{Debug, Display},
    sync::Arc,
};

use ark_ff::FftField;
use ark_poly::EvaluationDomain;
use serde::{Deserialize, Serialize};

use crate::{
    algebra::{
        domain::Domain,
        embedding::{self, Basefield},
        fields::FieldWithSize,
        ntt::{RSDefault, ReedSolomon},
    },
    bits::Bits,
    parameters::{FoldingFactor, MultivariateParameters, ProtocolParameters, SoundnessType},
    protocols::{irs_commit, matrix_commit, proof_of_work, sumcheck},
    type_info::{Type, Typed},
};

#[derive(Clone, Serialize, Deserialize, Debug)]
#[serde(bound = "F: FftField")]
pub struct WhirConfig<F>
where
    F: FftField,
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

    pub initial_sumcheck: Option<sumcheck::Config<F>>,
    pub starting_folding_pow: proof_of_work::Config,

    pub folding_factor: FoldingFactor,
    pub round_configs: Vec<RoundConfig<F>>,

    pub final_sumcheck: sumcheck::Config<F>,
    pub final_queries: usize,
    pub final_pow: proof_of_work::Config,
    pub final_log_inv_rate: usize,
    pub final_sumcheck_rounds: usize,
    pub final_folding_pow: proof_of_work::Config,

    // Merkle tree parameters
    // TODO: This has redundant parameters with starting_log_inv_rate and others.
    pub initial_committer: irs_commit::BasefieldConfig<F>,

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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "F: FftField")]
pub struct RoundConfig<F>
where
    F: FftField,
{
    pub irs_committer: irs_commit::Config<F>,

    pub matrix_committer: matrix_commit::Config<F>,
    pub sumcheck: sumcheck::Config<F>,
    pub pow: proof_of_work::Config,
    pub folding_pow: proof_of_work::Config,
    pub num_queries: usize,
    pub ood_samples: usize,
    pub log_inv_rate: usize,
    pub num_variables: usize,
    pub folding_factor: usize,
    pub domain_size: usize,
    #[serde(with = "crate::ark_serde::field")]
    pub domain_gen: F,
    #[serde(with = "crate::ark_serde::field")]
    pub domain_gen_inv: F,
    #[serde(with = "crate::ark_serde::field")]
    pub exp_domain_gen: F,
}

impl<F> WhirConfig<F>
where
    F: FftField + FieldWithSize,
{
    #[allow(clippy::too_many_lines)]
    pub fn new(
        reed_solomon: Arc<dyn ReedSolomon<F>>,
        basefield_reed_solomon: Arc<dyn ReedSolomon<F::BasePrimeField>>,
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

            let folding_factor = whir_parameters.folding_factor.at_round(round);
            let next_folding_factor = whir_parameters.folding_factor.at_round(round + 1);
            let matrix_committer = matrix_commit::Config::<F>::with_hash(
                whir_parameters.hash_id,
                (domain_size / 2) >> next_folding_factor,
                1 << next_folding_factor,
            );

            round_parameters.push(RoundConfig {
                irs_committer: irs_commit::Config {
                    embedding: Typed::new(embedding::Identity::new()),
                    num_polynomials: 1,
                    polynomial_size: 1 << num_variables,
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
                matrix_committer,
                sumcheck: sumcheck::Config {
                    field: Type::<F>::new(),
                    initial_size: 1 << num_variables,
                    rounds: vec![
                        sumcheck::RoundConfig {
                            pow: pow(folding_pow_bits),
                        };
                        next_folding_factor
                    ],
                },
                pow: pow(pow_bits),
                folding_pow: pow(folding_pow_bits),
                num_queries,
                ood_samples,
                log_inv_rate,
                num_variables,
                folding_factor,
                domain_size,
                domain_gen,
                domain_gen_inv,
                exp_domain_gen,
            });

            num_variables -= next_folding_factor;
            log_inv_rate = next_rate;
            domain_size /= 2;
            domain_gen = domain_gen.square();
            domain_gen_inv = domain_gen_inv.square();
            exp_domain_gen = domain_gen.pow([1 << next_folding_factor]);
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
            starting_domain: starting_domain.clone(),
            soundness_type: whir_parameters.soundness_type,
            starting_log_inv_rate: whir_parameters.starting_log_inv_rate,
            initial_committer: irs_commit::Config {
                embedding: Default::default(),
                num_polynomials: whir_parameters.batch_size,
                polynomial_size: 1 << mv_parameters.num_variables,
                expansion: 1 << whir_parameters.starting_log_inv_rate,
                interleaving_depth: 1 << whir_parameters.folding_factor.at_round(0),
                matrix_commit: matrix_commit::Config::with_hash(
                    whir_parameters.hash_id,
                    1 << (mv_parameters.num_variables + whir_parameters.starting_log_inv_rate
                        - whir_parameters.folding_factor.at_round(0)),
                    whir_parameters.batch_size << whir_parameters.folding_factor.at_round(0),
                ),
                in_domain_samples: round_parameters
                    .first()
                    .map_or(final_queries, |r| r.num_queries),
                out_domain_samples: commitment_ood_samples,
                deduplicate_in_domain: true,
            },
            initial_sumcheck: if whir_parameters.initial_statement {
                Some(sumcheck::Config {
                    field: Type::<F>::new(),
                    initial_size: 1 << mv_parameters.num_variables,
                    rounds: vec![
                        sumcheck::RoundConfig {
                            pow: pow(starting_folding_pow_bits),
                        };
                        whir_parameters.folding_factor.at_round(0)
                    ],
                })
            } else {
                None
            },
            starting_folding_pow: pow(starting_folding_pow_bits),
            folding_factor: whir_parameters.folding_factor,
            round_configs: round_parameters,
            final_sumcheck: sumcheck::Config {
                field: Type::<F>::new(),
                initial_size: 1 << num_variables,
                rounds: vec![
                    sumcheck::RoundConfig {
                        pow: pow(final_folding_pow_bits),
                    };
                    final_sumcheck_rounds
                ],
            },
            final_queries,
            final_pow: pow(final_pow_bits),
            final_sumcheck_rounds,
            final_folding_pow: pow(final_folding_pow_bits),
            final_log_inv_rate: log_inv_rate,
            batch_size: whir_parameters.batch_size,
            reed_solomon,
            basefield_reed_solomon,
        }
    }

    pub fn embedding(&self) -> &Basefield<F> {
        self.initial_committer.embedding()
    }

    pub const fn n_rounds(&self) -> usize {
        self.round_configs.len()
    }

    pub fn check_pow_bits(&self) -> bool {
        let max_bits = Bits::new(self.max_pow_bits as f64);

        // Check the main pow bits values
        if self.starting_folding_pow.difficulty() > max_bits
            || self.final_pow.difficulty() > max_bits
            || self.final_folding_pow.difficulty() > max_bits
        {
            return false;
        }

        // Check all round parameters
        self.round_configs
            .iter()
            .all(|r| r.pow.difficulty() <= max_bits && r.folding_pow.difficulty() <= max_bits)
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
        if self.round_configs.is_empty() {
            panic!("Can not treat initial_commit as a roundconfig");
            /*
            // Fallback: no folding rounds, use initial domain setup
            RoundConfig {
                matrix_committer: self.initial_matrix_committer.clone(),
                num_variables: self.mv_parameters.num_variables - self.folding_factor.at_round(0),
                folding_factor: self.folding_factor.at_round(self.n_rounds()),
                num_queries: self.final_queries,
                pow: self.final_pow,
                domain_size: self.starting_domain.size(),
                domain_gen: self.starting_domain.backing_domain.group_gen(),
                domain_gen_inv: self.starting_domain.backing_domain.group_gen_inv(),
                exp_domain_gen: self
                    .starting_domain
                    .backing_domain
                    .group_gen()
                    .pow([1 << self.folding_factor.at_round(0)]),
                ood_samples: 0, // no OOD in synthetic final phase
                folding_pow: self.final_folding_pow,
                log_inv_rate: self.starting_log_inv_rate,
            }
             */
        } else {
            // Derive final round config from last round, adjusted for next fold
            let last = self.round_configs.last().unwrap();
            let folding_factor = self.folding_factor.at_round(self.n_rounds());
            let num_variables = last.num_variables - folding_factor;
            RoundConfig {
                irs_committer: irs_commit::Config {
                    embedding: Typed::new(embedding::Identity::new()),
                    num_polynomials: 1,
                    polynomial_size: 1 << num_variables,
                    expansion: last.irs_committer.expansion * 2,
                    interleaving_depth: 1 << folding_factor,
                    matrix_commit: last.matrix_committer.clone(),
                    in_domain_samples: self.final_queries,
                    out_domain_samples: last.ood_samples,
                    deduplicate_in_domain: true,
                },
                matrix_committer: last.matrix_committer.clone(),
                sumcheck: sumcheck::Config {
                    field: Type::<F>::new(),
                    initial_size: 1 << num_variables,
                    rounds: vec![
                        sumcheck::RoundConfig {
                            pow: self.final_folding_pow,
                        };
                        folding_factor
                    ],
                },
                num_variables,
                folding_factor,
                num_queries: self.final_queries,
                pow: self.final_pow,
                domain_size: last.domain_size / 2,
                domain_gen: last.domain_gen.square(),
                domain_gen_inv: last.domain_gen_inv.square(),
                exp_domain_gen: last.domain_gen.square().pow([1 << folding_factor]),
                ood_samples: last.ood_samples,
                folding_pow: self.final_folding_pow,
                log_inv_rate: last.log_inv_rate,
            }
        }
    }
}

/// Manual implementation to allow error in `f64` and handle ark types missing `PartialEq`.
impl<F: FftField> PartialEq for WhirConfig<F> {
    fn eq(&self, other: &Self) -> bool {
        self.mv_parameters == other.mv_parameters
            && self.soundness_type == other.soundness_type
            && self.security_level == other.security_level
            && self.max_pow_bits == other.max_pow_bits
            && self.starting_folding_pow == other.starting_folding_pow
            && self.round_configs == other.round_configs
            && self.final_queries == other.final_queries
            && self.final_log_inv_rate == other.final_log_inv_rate
            && self.final_pow == other.final_pow
            && self.final_folding_pow == other.final_folding_pow
            && self.committment_ood_samples == other.committment_ood_samples
            && self.folding_factor == other.folding_factor
            && self.initial_statement == other.initial_statement
    }
}

impl<F: FftField> Display for WhirConfig<F> {
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
            self.starting_folding_pow.difficulty()
        )?;
        writeln!(f, "Initial rate: 2^-{}", self.starting_log_inv_rate)?;
        for (i, r) in self.round_configs.iter().enumerate() {
            write!(f, "Round {i}: {r}")?;
        }

        writeln!(
            f,
            "final_queries: {}, final_rate: 2^-{}, final_pow_bits: {}, final_folding_pow_bits: {}",
            self.final_queries,
            self.final_log_inv_rate,
            self.final_pow.difficulty(),
            self.final_folding_pow.difficulty(),
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
            prox_gaps_error.min(sumcheck_error) + f64::from(self.starting_folding_pow.difficulty()),
            self.folding_factor.at_round(0),
            prox_gaps_error,
            sumcheck_error,
            self.starting_folding_pow.difficulty(),
        )?;

        num_variables -= self.folding_factor.at_round(0);

        for (round, r) in self.round_configs.iter().enumerate() {
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
                prox_gaps_error.min(sumcheck_error) + f64::from(r.folding_pow.difficulty()),
                self.folding_factor.at_round(round + 1),
                prox_gaps_error,
                sumcheck_error,
                r.folding_pow.difficulty(),
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
            query_error + f64::from(self.final_pow.difficulty()),
            query_error,
            self.final_pow.difficulty(),
        )?;

        if self.final_sumcheck_rounds > 0 {
            let combination_error = field_size_bits as f64 - 1.;
            writeln!(
                f,
                "{:.1} bits -- (x{}) combination: {:.1}, pow: {:.1}",
                combination_error + f64::from(self.final_pow.difficulty()),
                self.final_sumcheck_rounds,
                combination_error,
                self.final_folding_pow.difficulty(),
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
            "Num_queries: {}, rate: 2^-{}, pow_bits: {}, ood_samples: {}, folding_factor: {} folding_pow: {}, initial_domain_size: {}",
            self.num_queries,
            self.log_inv_rate,
            self.pow.difficulty(),
            self.ood_samples,
            self.folding_factor,
            self.folding_pow.difficulty(),
            self.domain_size,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{algebra::fields::Field64, bits::Bits, hash, utils::test_serde};

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
    fn test_whir_config_creation() {
        let params = default_whir_params();

        let mv_params = MultivariateParameters::<Field64>::new(10);
        let reed_solomon = Arc::new(RSDefault);
        let basefield_reed_solomon = reed_solomon.clone();
        let config =
            WhirConfig::<Field64>::new(reed_solomon, basefield_reed_solomon, mv_params, &params);

        assert_eq!(config.security_level, 100);
        assert_eq!(config.max_pow_bits, 20);
        assert_eq!(config.soundness_type, SoundnessType::ConjectureList);
        assert!(config.initial_statement);
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
        let reed_solomon = Arc::new(RSDefault);
        let basefield_reed_solomon = reed_solomon.clone();
        let config =
            WhirConfig::<Field64>::new(reed_solomon, basefield_reed_solomon, mv_params, &params);

        test_serde(&config);
    }

    #[test]
    fn test_n_rounds() {
        let params = default_whir_params();
        let mv_params = MultivariateParameters::<Field64>::new(10);
        let reed_solomon = Arc::new(RSDefault);
        let basefield_reed_solomon = reed_solomon.clone();
        let config =
            WhirConfig::<Field64>::new(reed_solomon, basefield_reed_solomon, mv_params, &params);

        assert_eq!(config.n_rounds(), config.round_configs.len());
    }

    #[test]
    fn test_folding_pow_bits() {
        let field_size_bits = 64;
        let soundness = SoundnessType::ConjectureList;

        let pow_bits = WhirConfig::<Field64>::folding_pow_bits(
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

        let result = WhirConfig::<Field64>::queries(
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

        let result = WhirConfig::<Field64>::queries(
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

        let result = WhirConfig::<Field64>::queries(
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

        let result = WhirConfig::<Field64>::rbr_queries(
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

        let result = WhirConfig::<Field64>::rbr_queries(
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

        let result = WhirConfig::<Field64>::rbr_queries(
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
        let reed_solomon = Arc::new(RSDefault);
        let basefield_reed_solomon = reed_solomon.clone();
        let mut config =
            WhirConfig::<Field64>::new(reed_solomon, basefield_reed_solomon, mv_params, &params);

        // Set all values within limits
        config.max_pow_bits = 20;
        config.starting_folding_pow = proof_of_work::Config::from_difficulty(Bits::new(15.0));
        config.final_pow = proof_of_work::Config::from_difficulty(Bits::new(18.0));
        config.final_folding_pow = proof_of_work::Config::from_difficulty(Bits::new(19.5));

        // Ensure all rounds are within limits
        config.round_configs = vec![
            RoundConfig {
                irs_committer: irs_commit::Config {
                    embedding: Typed::new(embedding::Identity::new()),
                    num_polynomials: 1,
                    polynomial_size: 1 << 10,
                    expansion: 1 << 3,
                    interleaving_depth: 1 << 2,
                    matrix_commit: matrix_commit::Config::<Field64>::new(0, 0),
                    in_domain_samples: 5,
                    out_domain_samples: 2,
                    deduplicate_in_domain: true,
                },
                matrix_committer: matrix_commit::Config::<Field64>::new(0, 0),
                sumcheck: sumcheck::Config {
                    field: Type::<Field64>::new(),
                    initial_size: 1 << 10,
                    rounds: vec![
                        sumcheck::RoundConfig {
                            pow: proof_of_work::Config::from_difficulty(Bits::new(19.0)),
                        };
                        2
                    ],
                },
                pow: proof_of_work::Config::from_difficulty(Bits::new(17.0)),
                folding_pow: proof_of_work::Config::from_difficulty(Bits::new(19.0)),
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
                irs_committer: irs_commit::Config {
                    embedding: Typed::new(embedding::Identity::new()),
                    num_polynomials: 1,
                    polynomial_size: 1 << 10,
                    expansion: 1 << 4,
                    interleaving_depth: 1 << 2,
                    matrix_commit: matrix_commit::Config::<Field64>::new(0, 0),
                    in_domain_samples: 6,
                    out_domain_samples: 2,
                    deduplicate_in_domain: true,
                },
                matrix_committer: matrix_commit::Config::<Field64>::new(0, 0),
                sumcheck: sumcheck::Config {
                    field: Type::<Field64>::new(),
                    initial_size: 1 << 10,
                    rounds: vec![
                        sumcheck::RoundConfig {
                            pow: proof_of_work::Config::from_difficulty(Bits::new(19.5)),
                        };
                        2
                    ],
                },
                pow: proof_of_work::Config::from_difficulty(Bits::new(18.0)),
                folding_pow: proof_of_work::Config::from_difficulty(Bits::new(19.5)),
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
        let params = default_whir_params();
        let mv_params = MultivariateParameters::<Field64>::new(10);
        let reed_solomon = Arc::new(RSDefault);
        let basefield_reed_solomon = reed_solomon.clone();
        let mut config =
            WhirConfig::<Field64>::new(reed_solomon, basefield_reed_solomon, mv_params, &params);

        config.max_pow_bits = 20;
        config.starting_folding_pow = proof_of_work::Config::from_difficulty(Bits::new(21.0)); // Exceeds max_pow_bits
        config.final_pow = proof_of_work::Config::from_difficulty(Bits::new(18.0));
        config.final_folding_pow = proof_of_work::Config::from_difficulty(Bits::new(19.5));

        assert!(
            !config.check_pow_bits(),
            "Starting folding pow bits exceeds max_pow_bits, should return false."
        );
    }

    #[test]
    fn test_check_pow_bits_final_pow_exceeds() {
        let params = default_whir_params();
        let mv_params = MultivariateParameters::<Field64>::new(10);
        let reed_solomon = Arc::new(RSDefault);
        let basefield_reed_solomon = reed_solomon.clone();
        let mut config =
            WhirConfig::<Field64>::new(reed_solomon, basefield_reed_solomon, mv_params, &params);

        config.max_pow_bits = 20;
        config.starting_folding_pow = proof_of_work::Config::from_difficulty(Bits::new(15.0));
        config.final_pow = proof_of_work::Config::from_difficulty(Bits::new(21.0)); // Exceeds max_pow_bits
        config.final_folding_pow = proof_of_work::Config::from_difficulty(Bits::new(19.5));

        assert!(
            !config.check_pow_bits(),
            "Final pow bits exceeds max_pow_bits, should return false."
        );
    }

    #[test]
    fn test_check_pow_bits_round_pow_exceeds() {
        let params = default_whir_params();
        let mv_params = MultivariateParameters::<Field64>::new(10);
        let reed_solomon = Arc::new(RSDefault);
        let basefield_reed_solomon = reed_solomon.clone();
        let mut config =
            WhirConfig::<Field64>::new(reed_solomon, basefield_reed_solomon, mv_params, &params);

        config.max_pow_bits = 20;
        config.starting_folding_pow = proof_of_work::Config::from_difficulty(Bits::new(15.0));
        config.final_pow = proof_of_work::Config::from_difficulty(Bits::new(18.0));
        config.final_folding_pow = proof_of_work::Config::from_difficulty(Bits::new(19.5));

        // One round's pow_bits exceeds limit
        config.round_configs = vec![RoundConfig {
            irs_committer: irs_commit::Config {
                embedding: Typed::new(embedding::Identity::new()),
                num_polynomials: 1,
                polynomial_size: 1 << 10,
                expansion: 1 << 3,
                interleaving_depth: 1 << 2,
                matrix_commit: matrix_commit::Config::<Field64>::new(0, 0),
                in_domain_samples: 5,
                out_domain_samples: 2,
                deduplicate_in_domain: true,
            },
            matrix_committer: matrix_commit::Config::<Field64>::new(0, 0),
            sumcheck: sumcheck::Config {
                field: Type::<Field64>::new(),
                initial_size: 1 << 10,
                rounds: vec![
                    sumcheck::RoundConfig {
                        pow: proof_of_work::Config::from_difficulty(Bits::new(19.0)),
                    };
                    2
                ],
            },
            pow: proof_of_work::Config::from_difficulty(Bits::new(21.0)), // Exceeds max_pow_bits
            folding_pow: proof_of_work::Config::from_difficulty(Bits::new(19.0)),
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
        let params = default_whir_params();
        let mv_params = MultivariateParameters::<Field64>::new(10);
        let reed_solomon = Arc::new(RSDefault);
        let basefield_reed_solomon = reed_solomon.clone();
        let mut config =
            WhirConfig::<Field64>::new(reed_solomon, basefield_reed_solomon, mv_params, &params);

        config.max_pow_bits = 20;
        config.starting_folding_pow = proof_of_work::Config::from_difficulty(Bits::new(15.0));
        config.final_pow = proof_of_work::Config::from_difficulty(Bits::new(18.0));
        config.final_folding_pow = proof_of_work::Config::from_difficulty(Bits::new(19.5));

        // One round's folding_pow_bits exceeds limit
        config.round_configs = vec![RoundConfig {
            irs_committer: irs_commit::Config {
                embedding: Typed::new(embedding::Identity::new()),
                num_polynomials: 1,
                polynomial_size: 1 << 10,
                expansion: 1 << 3,
                interleaving_depth: 1 << 2,
                matrix_commit: matrix_commit::Config::<Field64>::new(0, 0),
                in_domain_samples: 5,
                out_domain_samples: 2,
                deduplicate_in_domain: true,
            },
            matrix_committer: matrix_commit::Config::<Field64>::new(0, 0),
            sumcheck: sumcheck::Config {
                field: Type::<Field64>::new(),
                initial_size: 1 << 10,
                rounds: vec![
                    sumcheck::RoundConfig {
                        pow: proof_of_work::Config::from_difficulty(Bits::new(21.0)),
                    };
                    2
                ],
            },
            pow: proof_of_work::Config::from_difficulty(Bits::new(19.0)),
            folding_pow: proof_of_work::Config::from_difficulty(Bits::new(21.0)), // Exceeds max_pow_bits
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
        let params = default_whir_params();
        let mv_params = MultivariateParameters::<Field64>::new(10);
        let reed_solomon = Arc::new(RSDefault);
        let basefield_reed_solomon = reed_solomon.clone();
        let mut config =
            WhirConfig::<Field64>::new(reed_solomon, basefield_reed_solomon, mv_params, &params);

        config.max_pow_bits = 20;
        config.starting_folding_pow = proof_of_work::Config::from_difficulty(Bits::new(20.0));
        config.final_pow = proof_of_work::Config::from_difficulty(Bits::new(20.0));
        config.final_folding_pow = proof_of_work::Config::from_difficulty(Bits::new(20.0));

        config.round_configs = vec![RoundConfig {
            irs_committer: irs_commit::Config {
                embedding: Typed::new(embedding::Identity::new()),
                num_polynomials: 1,
                polynomial_size: 1 << 10,
                expansion: 1 << 3,
                interleaving_depth: 1 << 2,
                matrix_commit: matrix_commit::Config::<Field64>::new(0, 0),
                in_domain_samples: 5,
                out_domain_samples: 2,
                deduplicate_in_domain: true,
            },
            matrix_committer: matrix_commit::Config::<Field64>::new(0, 0),
            sumcheck: sumcheck::Config {
                field: Type::<Field64>::new(),
                initial_size: 1 << 10,
                rounds: vec![
                    sumcheck::RoundConfig {
                        pow: proof_of_work::Config::from_difficulty(Bits::new(20.0)),
                    };
                    2
                ],
            },
            pow: proof_of_work::Config::from_difficulty(Bits::new(20.0)),
            folding_pow: proof_of_work::Config::from_difficulty(Bits::new(20.0)),
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
        let params = default_whir_params();
        let mv_params = MultivariateParameters::<Field64>::new(10);
        let reed_solomon = Arc::new(RSDefault);
        let basefield_reed_solomon = reed_solomon.clone();
        let mut config =
            WhirConfig::<Field64>::new(reed_solomon, basefield_reed_solomon, mv_params, &params);

        config.max_pow_bits = 20;
        config.starting_folding_pow = proof_of_work::Config::from_difficulty(Bits::new(22.0));
        config.final_pow = proof_of_work::Config::from_difficulty(Bits::new(23.0));
        config.final_folding_pow = proof_of_work::Config::from_difficulty(Bits::new(24.0));

        config.round_configs = vec![RoundConfig {
            irs_committer: irs_commit::Config {
                embedding: Typed::new(embedding::Identity::new()),
                num_polynomials: 1,
                polynomial_size: 1 << 10,
                expansion: 1 << 3,
                interleaving_depth: 1 << 2,
                matrix_commit: matrix_commit::Config::<Field64>::new(0, 0),
                in_domain_samples: 5,
                out_domain_samples: 2,
                deduplicate_in_domain: true,
            },
            matrix_committer: matrix_commit::Config::<Field64>::new(0, 0),
            sumcheck: sumcheck::Config {
                field: Type::<Field64>::new(),
                initial_size: 1 << 10,
                rounds: vec![
                    sumcheck::RoundConfig {
                        pow: proof_of_work::Config::from_difficulty(Bits::new(26.0)),
                    };
                    2
                ],
            },
            pow: proof_of_work::Config::from_difficulty(Bits::new(25.0)),
            folding_pow: proof_of_work::Config::from_difficulty(Bits::new(26.0)),
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
            let result = WhirConfig::<Field64>::list_size_bits(
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
            let result = WhirConfig::<Field64>::list_size_bits(
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
            let result = WhirConfig::<Field64>::list_size_bits(
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
            let result = WhirConfig::<Field64>::rbr_ood_sample(
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
            let result = WhirConfig::<Field64>::rbr_ood_sample(
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
            WhirConfig::<Field64>::ood_samples(100, SoundnessType::UniqueDecoding, 10, 3, 1.5, 256),
            0
        );
    }

    #[test]
    fn test_ood_samples_valid_case() {
        // Testing a valid case where the function finds an appropriate `ood_samples`
        assert_eq!(
            WhirConfig::<Field64>::ood_samples(
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
            WhirConfig::<Field64>::ood_samples(
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
            WhirConfig::<Field64>::ood_samples(
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
            WhirConfig::<Field64>::ood_samples(
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
