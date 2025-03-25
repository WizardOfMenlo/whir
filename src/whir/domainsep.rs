use ark_crypto_primitives::merkle_tree::Config;
use ark_ff::FftField;
use spongefish::codecs::arkworks_algebra::{ByteDomainSeparator, FieldDomainSeparator};

use super::parameters::WhirConfig;
use crate::{
    fs_utils::{OODDomainSeparator, WhirPoWDomainSeparator},
    sumcheck::SumcheckSingleDomainSeparator,
};

pub trait DigestDomainSeparator<MerkleConfig: Config> {
    #[must_use]
    fn add_digest(self, label: &str) -> Self;
}

pub trait WhirDomainSeparator<F: FftField, MerkleConfig: Config> {
    #[must_use]
    fn commit_statement<PowStrategy>(
        self,
        params: &WhirConfig<F, MerkleConfig, PowStrategy>,
    ) -> Self;

    #[must_use]
    fn add_whir_proof<PowStrategy>(self, params: &WhirConfig<F, MerkleConfig, PowStrategy>)
        -> Self;
}

impl<F, MerkleConfig, DomainSeparator> WhirDomainSeparator<F, MerkleConfig> for DomainSeparator
where
    F: FftField,
    MerkleConfig: Config,
    DomainSeparator: ByteDomainSeparator
        + FieldDomainSeparator<F>
        + SumcheckSingleDomainSeparator<F>
        + WhirPoWDomainSeparator
        + OODDomainSeparator<F>
        + DigestDomainSeparator<MerkleConfig>,
{
    fn commit_statement<PowStrategy>(
        self,
        params: &WhirConfig<F, MerkleConfig, PowStrategy>,
    ) -> Self {
        // TODO: Add params
        let mut this = self.add_digest("merkle_digest");
        if params.committment_ood_samples > 0 {
            assert!(params.initial_statement);
            this = this.add_ood(params.committment_ood_samples);
        }
        this
    }

    fn add_whir_proof<PowStrategy>(
        mut self,
        params: &WhirConfig<F, MerkleConfig, PowStrategy>,
    ) -> Self {
        // TODO: Add statement
        if params.initial_statement {
            self = self
                .challenge_scalars(1, "initial_combination_randomness")
                .add_sumcheck(
                    params.folding_factor.at_round(0),
                    params.starting_folding_pow_bits,
                );
        } else {
            self = self
                .challenge_scalars(params.folding_factor.at_round(0), "folding_randomness")
                .pow(params.starting_folding_pow_bits);
        }

        let mut domain_size = params.starting_domain.size();
        for (round, r) in params.round_parameters.iter().enumerate() {
            let folded_domain_size = domain_size >> params.folding_factor.at_round(round);
            let domain_size_bytes = ((folded_domain_size * 2 - 1).ilog2() as usize).div_ceil(8);
            self = self
                .add_digest("merkle_digest")
                .add_ood(r.ood_samples)
                .challenge_bytes(r.num_queries * domain_size_bytes, "stir_queries")
                .pow(r.pow_bits)
                .challenge_scalars(1, "combination_randomness")
                .add_sumcheck(
                    params.folding_factor.at_round(round + 1),
                    r.folding_pow_bits,
                );
            domain_size >>= 1;
        }

        let folded_domain_size = domain_size
            >> params
                .folding_factor
                .at_round(params.round_parameters.len());
        let domain_size_bytes = ((folded_domain_size * 2 - 1).ilog2() as usize).div_ceil(8);

        self.add_scalars(1 << params.final_sumcheck_rounds, "final_coeffs")
            .challenge_bytes(domain_size_bytes * params.final_queries, "final_queries")
            .pow(params.final_pow_bits)
            .add_sumcheck(params.final_sumcheck_rounds, params.final_folding_pow_bits)
    }
}
