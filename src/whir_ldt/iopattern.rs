use ark_crypto_primitives::merkle_tree::Config;
use ark_ff::FftField;
use nimue::plugins::ark::*;

use crate::{
    fs_utils::{OODIOPattern, WhirPoWIOPattern},
    sumcheck::prover_not_skipping::SumcheckNotSkippingIOPattern,
};

use super::parameters::WhirConfig;

pub trait WhirIOPattern<F: FftField> {
    fn commit_statement<MerkleConfig: Config>(self, params: &WhirConfig<F, MerkleConfig>) -> Self;
    fn add_whir_proof<MerkleConfig: Config>(self, params: &WhirConfig<F, MerkleConfig>) -> Self;
}

impl<F> WhirIOPattern<F> for IOPattern
where
    F: FftField,
    IOPattern: ByteIOPattern
        + FieldIOPattern<F>
        + SumcheckNotSkippingIOPattern<F>
        + WhirPoWIOPattern
        + OODIOPattern<F>,
{
    fn commit_statement<MerkleConfig: Config>(self, _params: &WhirConfig<F, MerkleConfig>) -> Self {
        self.add_bytes(32, "merkle_digest")
    }

    fn add_whir_proof<MerkleConfig: Config>(
        mut self,
        params: &WhirConfig<F, MerkleConfig>,
    ) -> Self {
        // TODO: Add statement
        self = self
            .challenge_scalars(params.folding_factor, "folding_randomness")
            .pow(params.starting_folding_pow_bits);

        for r in &params.round_parameters {
            self = self
                .add_bytes(32, "merkle_digest")
                .add_ood(r.ood_samples)
                .challenge_bytes(32, "stir_queries_seed")
                .pow(r.pow_bits)
                .challenge_scalars(1, "combination_randomness")
                .add_sumcheck(params.folding_factor, r.folding_pow_bits);
        }

        self.add_scalars(1 << params.final_sumcheck_rounds, "final_coeffs")
            .challenge_bytes(32, "final_queries_seed")
            .pow(params.final_pow_bits)
            .add_sumcheck(params.final_sumcheck_rounds, params.final_folding_pow_bits)
    }
}
