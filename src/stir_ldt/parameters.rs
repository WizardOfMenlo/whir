use core::panic;
use std::{f64::consts::LOG2_10, fmt::Display, marker::PhantomData};

use ark_crypto_primitives::merkle_tree::{Config, LeafParam, TwoToOneParam};
use ark_ff::FftField;

use crate::{
    crypto::fields::FieldWithSize,
    domain::Domain,
    parameters::{ProtocolParameters, SoundnessType, UnivariateParameters},
};

#[derive(Clone)]
pub struct StirConfig<F, MerkleConfig, PowStrategy>
where
    F: FftField,
    MerkleConfig: Config,
{
    pub(crate) uv_parameters: UnivariateParameters<F>,
    pub(crate) soundness_type: SoundnessType,
    pub(crate) security_level: usize,
    pub(crate) max_pow_bits: usize,

    pub(crate) starting_domain: Domain<F>,
    pub(crate) starting_log_inv_rate: usize,
    pub(crate) starting_folding_pow_bits: f64,

    pub(crate) folding_factor: usize,
    pub(crate) round_parameters: Vec<RoundConfig>,
    // pub(crate) fold_optimisation: FoldType, Just doing prover helps
    pub(crate) final_queries: usize,
    pub(crate) final_pow_bits: f64,
    pub(crate) final_log_inv_rate: usize,
    pub(crate) final_log_degree: usize,

    // PoW parameters
    pub(crate) pow_strategy: PhantomData<PowStrategy>,

    // Merkle tree parameters
    pub(crate) leaf_hash_params: LeafParam<MerkleConfig>,
    pub(crate) two_to_one_params: TwoToOneParam<MerkleConfig>,
}

#[derive(Debug, Clone)]
pub(crate) struct RoundConfig {
    pub(crate) pow_bits: f64,
    pub(crate) folding_pow_bits: f64,
    pub(crate) num_queries: usize,
    pub(crate) ood_samples: usize,
    pub(crate) log_inv_rate: usize,
}

impl<F, MerkleConfig, PowStrategy> StirConfig<F, MerkleConfig, PowStrategy>
where
    F: FftField + FieldWithSize,
    MerkleConfig: Config,
{
    pub fn new(
        uv_parameters: UnivariateParameters<F>,
        stir_parameters: ProtocolParameters<MerkleConfig, PowStrategy>,
    ) -> Self {
        // We need to fold at least some time
        assert!(
            stir_parameters.folding_factor > 0,
            "folding factor should be non zero"
        );
        // If less, just send the damn polynomials
        assert!(uv_parameters.log_degree >= stir_parameters.folding_factor);
        let protocol_security_level =
            0.max(stir_parameters.security_level - stir_parameters.pow_bits);

        let starting_domain = Domain::new(
            1 << uv_parameters.log_degree,
            stir_parameters.starting_log_inv_rate,
        )
        .expect("Should have found an appropriate domain");

        // TODO: Maybe here we should stop earlier
        // let final_log_degree = uv_parameters.log_degree % stir_parameters.folding_factor;
        let final_log_degree = 8;
        let num_rounds =
            ((uv_parameters.log_degree - final_log_degree) / stir_parameters.folding_factor) - 1;

        let field_size_bits = F::field_size_in_bits();

        let prox_gaps_error = Self::rbr_soundness_fold_prox_gaps(
            stir_parameters.soundness_type,
            field_size_bits,
            uv_parameters.log_degree,
            stir_parameters.starting_log_inv_rate,
            Self::log_eta(stir_parameters.starting_log_inv_rate),
        ) + (stir_parameters.folding_factor as f64).log2();
        let starting_folding_pow_bits =
            0_f64.max(stir_parameters.security_level as f64 - prox_gaps_error);

        let mut round_parameters = Vec::with_capacity(num_rounds);
        let mut log_degree = uv_parameters.log_degree - stir_parameters.folding_factor;
        let mut log_inv_rate = stir_parameters.starting_log_inv_rate;
        for _ in 0..num_rounds {
            // Queries are set w.r.t. to old rate, while the rest to the new rate
            let next_rate = log_inv_rate + (stir_parameters.folding_factor - 1);

            let log_next_eta = Self::log_eta(next_rate);
            let num_queries = Self::queries(
                stir_parameters.soundness_type,
                protocol_security_level,
                log_inv_rate,
            );

            let ood_samples = Self::ood_samples(
                stir_parameters.security_level,
                stir_parameters.soundness_type,
                log_degree,
                next_rate,
                log_next_eta,
                field_size_bits,
            );

            assert!(num_queries + ood_samples < (1 << log_degree) + 1);

            let query_error =
                Self::rbr_queries(stir_parameters.soundness_type, log_inv_rate, num_queries);

            let pow_bits = 0_f64.max(stir_parameters.security_level as f64 - query_error);

            let folding_pow_bits = Self::folding_pow_bits(
                stir_parameters.security_level,
                stir_parameters.soundness_type,
                field_size_bits,
                log_degree,
                next_rate,
                log_next_eta,
            );

            round_parameters.push(RoundConfig {
                ood_samples,
                num_queries,
                pow_bits,
                folding_pow_bits,
                log_inv_rate,
            });

            log_degree -= stir_parameters.folding_factor;
            log_inv_rate = next_rate;
        }

        let final_queries = Self::queries(
            stir_parameters.soundness_type,
            protocol_security_level,
            log_inv_rate,
        );

        let final_pow_bits = 0_f64.max(
            stir_parameters.security_level as f64
                - Self::rbr_queries(stir_parameters.soundness_type, log_inv_rate, final_queries),
        );

        StirConfig {
            security_level: stir_parameters.security_level,
            max_pow_bits: stir_parameters.pow_bits,
            uv_parameters,
            starting_domain,
            soundness_type: stir_parameters.soundness_type,
            starting_log_inv_rate: stir_parameters.starting_log_inv_rate,
            starting_folding_pow_bits,
            folding_factor: stir_parameters.folding_factor,
            round_parameters,
            final_queries,
            final_pow_bits,
            pow_strategy: PhantomData::default(),
            final_log_inv_rate: log_inv_rate,
            final_log_degree,
            leaf_hash_params: stir_parameters.leaf_hash_params,
            two_to_one_params: stir_parameters.two_to_one_params,
        }
    }

    pub fn n_rounds(&self) -> usize {
        self.round_parameters.len()
    }

    pub fn check_pow_bits(&self) -> bool {
        [self.starting_folding_pow_bits, self.final_pow_bits]
            .into_iter()
            .all(|x| x <= self.max_pow_bits as f64)
            && self.round_parameters.iter().all(|r| {
                r.pow_bits <= self.max_pow_bits as f64
                    && r.folding_pow_bits <= self.max_pow_bits as f64
            })
    }

    pub fn log_eta(log_inv_rate: usize) -> f64 {
        -(log_inv_rate as f64 + LOG2_10)
    }

    pub fn list_size_bits(
        soundness_type: SoundnessType,
        num_variables: usize,
        log_inv_rate: usize,
        log_eta: f64,
    ) -> f64 {
        match soundness_type {
            SoundnessType::ConjectureList => {
                let result = (num_variables + log_inv_rate) as f64 - log_eta;
                result
            }
            SoundnessType::ProvableList => {
                let log_inv_sqrt_rate: f64 = log_inv_rate as f64 / 2.;
                let result = log_inv_sqrt_rate - (1. + log_eta);
                result
            }
            SoundnessType::UniqueDecoding => 0.0,
        }
    }

    pub fn rbr_ood_sample(
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
        if matches!(soundness_type, SoundnessType::UniqueDecoding) {
            0
        } else {
            for ood_samples in 1..64 {
                if Self::rbr_ood_sample(
                    soundness_type,
                    num_variables,
                    log_inv_rate,
                    log_eta,
                    field_size_bits,
                    ood_samples,
                ) >= security_level as f64
                {
                    return ood_samples;
                }
            }

            panic!("Could not find an appropriate number of OOD samples");
        }
    }

    // Compute the proximity gaps term of the fold
    pub fn rbr_soundness_fold_prox_gaps(
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

    pub fn rbr_soundness_fold_sumcheck(
        soundness_type: SoundnessType,
        field_size_bits: usize,
        num_variables: usize,
        log_inv_rate: usize,
        log_eta: f64,
    ) -> f64 {
        let list_size = Self::list_size_bits(soundness_type, num_variables, log_inv_rate, log_eta);

        field_size_bits as f64 - (list_size + 1.)
    }

    pub fn folding_pow_bits(
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

        0_f64.max(security_level as f64 - prox_gaps_error)
    }

    // Used to select the number of queries
    pub fn queries(
        soundness_type: SoundnessType,
        protocol_security_level: usize,
        log_inv_rate: usize,
    ) -> usize {
        let num_queries_f = match soundness_type {
            SoundnessType::UniqueDecoding => {
                let rate = 1. / ((1 << log_inv_rate) as f64);
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
        let bits_of_sec_queries = match soundness_type {
            SoundnessType::UniqueDecoding => {
                let rate = 1. / ((1 << log_inv_rate) as f64);
                let denom = -(0.5 * (1. + rate)).log2();

                num_queries * denom
            }
            SoundnessType::ProvableList => num_queries * 0.5 * log_inv_rate as f64,
            SoundnessType::ConjectureList => num_queries * log_inv_rate as f64,
        };

        bits_of_sec_queries
    }
}

impl<F, MerkleConfig, PowStrategy> Display for StirConfig<F, MerkleConfig, PowStrategy>
where
    F: FftField,
    MerkleConfig: Config,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.uv_parameters.fmt(f)?;
        writeln!(f, ", folding factor: {}", self.folding_factor)?;
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
            r.fmt(f)?;
        }

        writeln!(
            f,
            "final_queries: {}, final_rate: 2^-{}, final_pow_bits: {}",
            self.final_queries, self.final_log_inv_rate, self.final_pow_bits,
        )?;

        writeln!(f, "------------------------------------")?;
        writeln!(f, "Round by round soundness analysis:")?;
        writeln!(f, "------------------------------------")?;

        let field_size_bits = F::field_size_in_bits();
        let log_eta = Self::log_eta(self.starting_log_inv_rate);
        let mut log_degree = self.uv_parameters.log_degree;

        let prox_gaps_error = Self::rbr_soundness_fold_prox_gaps(
            self.soundness_type,
            field_size_bits,
            log_degree,
            self.starting_log_inv_rate,
            log_eta,
        ) - (self.folding_factor as f64).log2();

        writeln!(
            f,
            "{:.1} bits -- prox gaps: {:.1}, pow: {:.1}",
            prox_gaps_error + self.starting_folding_pow_bits as f64,
            prox_gaps_error,
            self.starting_folding_pow_bits,
        )?;

        log_degree -= self.folding_factor;

        for r in &self.round_parameters {
            let next_rate = r.log_inv_rate + (self.folding_factor - 1);
            let log_eta = Self::log_eta(next_rate);

            if r.ood_samples > 0 {
                writeln!(
                    f,
                    "{:.1} bits -- OOD sample",
                    Self::rbr_ood_sample(
                        self.soundness_type,
                        log_degree,
                        next_rate,
                        log_eta,
                        field_size_bits,
                        r.ood_samples
                    )
                )?;
            }

            let query_error = Self::rbr_queries(self.soundness_type, r.log_inv_rate, r.num_queries);
            writeln!(
                f,
                "{:.1} bits -- query error: {:.1}, pow: {:.1}",
                query_error + r.pow_bits as f64,
                query_error,
                r.pow_bits,
            )?;

            let prox_gaps_error = Self::rbr_soundness_fold_prox_gaps(
                self.soundness_type,
                field_size_bits,
                log_degree,
                next_rate,
                log_eta,
            );
            let sumcheck_error = Self::rbr_soundness_fold_sumcheck(
                self.soundness_type,
                field_size_bits,
                log_degree,
                next_rate,
                log_eta,
            );

            writeln!(
                f,
                "{:.1} bits -- (x{}) prox gaps: {:.1}, sumcheck: {:.1}, pow: {:.1}",
                prox_gaps_error.min(sumcheck_error) + r.folding_pow_bits as f64,
                self.folding_factor,
                prox_gaps_error,
                sumcheck_error,
                r.folding_pow_bits,
            )?;

            log_degree -= self.folding_factor;
        }

        let query_error = Self::rbr_queries(
            self.soundness_type,
            self.final_log_inv_rate,
            self.final_queries,
        );
        writeln!(
            f,
            "{:.1} bits -- query error: {:.1}, pow: {:.1}",
            query_error + self.final_pow_bits as f64,
            query_error,
            self.final_pow_bits,
        )?;

        Ok(())
    }
}

impl Display for RoundConfig {
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
