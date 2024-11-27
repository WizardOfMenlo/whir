use ark_crypto_primitives::merkle_tree::Config;
use ark_ff::FftField;
use nimue::plugins::ark::*;

use crate::{
    fs_utils::{OODIOPattern, WhirPoWIOPattern},
    sumcheck::prover_not_skipping::SumcheckNotSkippingIOPattern,
};

use super::parameters::WhirConfig;

pub trait WhirIOPattern<F: FftField> {
    fn commit_statement<MerkleConfig: Config, PowStrategy>(
        self,
        params: &WhirConfig<F, MerkleConfig, PowStrategy>,
    ) -> Self;
    fn add_whir_proof<MerkleConfig: Config, PowStrategy>(
        self,
        params: &WhirConfig<F, MerkleConfig, PowStrategy>,
    ) -> Self;
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
    fn commit_statement<MerkleConfig: Config, PowStrategy>(
        self,
        params: &WhirConfig<F, MerkleConfig, PowStrategy>,
    ) -> Self {
        // TODO: Add params
        let mut this = self.add_bytes(32, "merkle_digest");
        if params.committment_ood_samples > 0 {
            assert!(params.initial_statement);
            this = this.add_ood(params.committment_ood_samples);
        }
        this
    }

    fn add_whir_proof<MerkleConfig: Config, PowStrategy>(
        mut self,
        params: &WhirConfig<F, MerkleConfig, PowStrategy>,
    ) -> Self {
        // TODO: Add statement
        if params.initial_statement {
            self = self
                .challenge_scalars(1, "initial_combination_randomness")
                .add_sumcheck(params.folding_factor, params.starting_folding_pow_bits);
        } else {
            self = self
                .challenge_scalars(params.folding_factor, "folding_randomness")
                .pow(params.starting_folding_pow_bits);
        }

        let mut folded_domain_size = params.starting_domain.folded_size(params.folding_factor);

        for r in &params.round_parameters {
            let domain_size_bytes = ((folded_domain_size * 2 - 1).ilog2() as usize + 7) / 8;
            self = self
                .add_bytes(32, "merkle_digest")
                .add_ood(r.ood_samples)
                .challenge_bytes(r.num_queries * domain_size_bytes, "stir_queries")
                .pow(r.pow_bits)
                .challenge_scalars(1, "combination_randomness")
                .add_sumcheck(params.folding_factor, r.folding_pow_bits);
            folded_domain_size /= 2;
        }

        let domain_size_bytes = ((folded_domain_size * 2 - 1).ilog2() as usize + 7) / 8;

        self.add_scalars(1 << params.final_sumcheck_rounds, "final_coeffs")
            .challenge_bytes(domain_size_bytes * params.final_queries, "final_queries")
            .pow(params.final_pow_bits)
            .add_sumcheck(params.final_sumcheck_rounds, params.final_folding_pow_bits)
    }
}
