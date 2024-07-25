use std::fmt::Display;

use ark_crypto_primitives::merkle_tree::{Config, LeafParam, TwoToOneParam};
use ark_ff::FftField;

use crate::{
    domain::Domain,
    parameters::{MultivariateParameters, SoundnessType, WhirParameters},
};

#[derive(Clone)]
pub struct WhirConfig<F, MerkleConfig>
where
    F: FftField,
    MerkleConfig: Config,
{
    pub(crate) committment_ood_samples: usize,
    pub(crate) mv_parameters: MultivariateParameters<F>,
    pub(crate) starting_domain: Domain<F>,
    pub(crate) folding_factor: usize,
    pub(crate) round_parameters: Vec<RoundConfig>,
    pub(crate) final_queries: usize,
    pub(crate) final_pow_bits: usize,
    pub(crate) final_log_inv_rate: usize,

    // Merkle tree parameters
    pub(crate) leaf_hash_params: LeafParam<MerkleConfig>,
    pub(crate) two_to_one_params: TwoToOneParam<MerkleConfig>,
}

impl<F, MerkleConfig> WhirConfig<F, MerkleConfig>
where
    F: FftField,
    MerkleConfig: Config,
{
    fn num_queries(
        soundness_type: SoundnessType,
        security_level: usize,
        protocol_security_level: usize,
        log_inv_rate: usize,
    ) -> (usize, usize) {
        let num_queries_f = match soundness_type {
            SoundnessType::UniqueDecoding => {
                let rate = 1. / ((1 << log_inv_rate) as f64);
                let denom = (0.5 * (1. + rate)).log2();

                (-(protocol_security_level as f64) / denom).ceil()
            }
            SoundnessType::ProvableList => {
                ((2 * protocol_security_level) as f64 / log_inv_rate as f64).ceil()
            }
            SoundnessType::ConjectureList => {
                (protocol_security_level as f64 / log_inv_rate as f64).ceil()
            }
        };

        let bits_of_sec = match soundness_type {
            SoundnessType::UniqueDecoding => {
                let rate = 1. / ((1 << log_inv_rate) as f64);
                let denom = -(0.5 * (1. + rate)).log2();

                num_queries_f * denom
            }
            SoundnessType::ProvableList => num_queries_f * 0.5 * log_inv_rate as f64,
            SoundnessType::ConjectureList => num_queries_f * log_inv_rate as f64,
        };

        (
            num_queries_f as usize,
            (security_level as f64 - bits_of_sec).max(0.).ceil() as usize,
        )
    }

    pub fn ood_samples(soundness_type: SoundnessType) -> usize {
        if matches!(soundness_type, SoundnessType::UniqueDecoding) {
            0
        } else {
            1
        }
    }

    pub fn new(
        mv_parameters: MultivariateParameters<F>,
        whir_parameters: WhirParameters<MerkleConfig>,
    ) -> Self {
        assert_eq!(
            mv_parameters.num_variables % whir_parameters.folding_factor,
            0,
            "folding factor should divide num of variables"
        );

        assert!(whir_parameters.security_level >= whir_parameters.protocol_security_level);

        let starting_domain = Domain::new(
            1 << mv_parameters.num_variables,
            whir_parameters.starting_log_inv_rate,
        )
        .expect("Should have found an appropriate domain");

        // -1 is for the final round of folding
        let num_rounds = (mv_parameters.num_variables / whir_parameters.folding_factor) - 1;

        let mut round_parameters = Vec::with_capacity(num_rounds);

        let mut log_inv_rate = whir_parameters.starting_log_inv_rate;
        for _ in 0..num_rounds {
            let (num_queries, pow_bits) = Self::num_queries(
                whir_parameters.soundness_type,
                whir_parameters.security_level,
                whir_parameters.protocol_security_level,
                log_inv_rate,
            );
            round_parameters.push(RoundConfig {
                ood_samples: Self::ood_samples(whir_parameters.soundness_type),
                num_queries,
                pow_bits,
                log_inv_rate,
            });
            log_inv_rate = log_inv_rate + (whir_parameters.folding_factor - 1);
        }

        let (final_queries, final_pow_bits) = Self::num_queries(
            whir_parameters.soundness_type,
            whir_parameters.security_level,
            whir_parameters.protocol_security_level,
            log_inv_rate,
        );

        WhirConfig {
            committment_ood_samples: Self::ood_samples(whir_parameters.soundness_type),
            mv_parameters,
            starting_domain,
            folding_factor: whir_parameters.folding_factor,
            round_parameters,
            final_queries,
            final_pow_bits,
            final_log_inv_rate: log_inv_rate,
            leaf_hash_params: whir_parameters.leaf_hash_params,
            two_to_one_params: whir_parameters.two_to_one_params,
        }
    }

    pub fn n_rounds(&self) -> usize {
        self.round_parameters.len()
    }
}

impl<F, MerkleConfig> Display for WhirConfig<F, MerkleConfig>
where
    F: FftField,
    MerkleConfig: Config,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.mv_parameters.fmt(f)?;

        writeln!(f, ", folding factor: {}", self.folding_factor)?;

        for r in &self.round_parameters {
            r.fmt(f)?;
        }

        writeln!(
            f,
            "final_queries: {}, final_rate: 2^-{}, final_pow_bits: {}",
            self.final_queries, self.final_log_inv_rate, self.final_pow_bits
        )
    }
}

#[derive(Debug, Clone)]
pub(crate) struct RoundConfig {
    pub(crate) pow_bits: usize,
    pub(crate) num_queries: usize,
    pub(crate) ood_samples: usize,
    pub(crate) log_inv_rate: usize,
}

impl Display for RoundConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "Num_queries: {}, rate: 2^-{}, pow_bits: {}, ood_samples: {}",
            self.num_queries, self.log_inv_rate, self.pow_bits, self.ood_samples
        )
    }
}
