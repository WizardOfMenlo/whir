use ark_crypto_primitives::merkle_tree::Config;
use ark_ff::FftField;
use nimue::plugins::ark::*;

use crate::{
    fs_utils::{OODIOPattern, WhirPoWIOPattern},
    sumcheck::prover_not_skipping::SumcheckNotSkippingIOPattern,
};

use super::parameters::StirConfig;

pub trait StirIOPattern<F: FftField> {
    fn commit_statement<MerkleConfig: Config, PowStrategy>(
        self,
        params: &StirConfig<F, MerkleConfig, PowStrategy>,
    ) -> Self;
    fn add_stir_proof<MerkleConfig: Config, PowStrategy>(
        self,
        params: &StirConfig<F, MerkleConfig, PowStrategy>,
    ) -> Self;
}

impl<F> StirIOPattern<F> for IOPattern
where
    F: FftField,
    IOPattern: ByteIOPattern
        + FieldIOPattern<F>
        + SumcheckNotSkippingIOPattern<F>
        + WhirPoWIOPattern
        + OODIOPattern<F>,
{
    fn commit_statement<MerkleConfig: Config, PowStrategy>(
        self,
        _params: &StirConfig<F, MerkleConfig, PowStrategy>,
    ) -> Self {
        self.add_bytes(32, "merkle_digest")
    }

    fn add_stir_proof<MerkleConfig: Config, PowStrategy>(
        mut self,
        params: &StirConfig<F, MerkleConfig, PowStrategy>,
    ) -> Self {
        // TODO: Add statement
        self = self
            .challenge_scalars(1, "folding_randomness")
            .pow(params.starting_folding_pow_bits);

        for r in &params.round_parameters {
            self = self
                .add_bytes(32, "merkle_digest")
                .add_ood(r.ood_samples)
                .challenge_bytes(32, "stir_queries_seed")
                .pow(r.pow_bits)
                .challenge_scalars(1, "combination_randomness")
                .challenge_scalars(1, "folding_randomness")
        }

        self.add_scalars(1 << params.final_log_degree, "final_coeffs")
            .challenge_bytes(32, "final_queries_seed")
            .pow(params.final_pow_bits)
    }
}
