use ark_crypto_primitives::merkle_tree::Config;
use ark_ff::FftField;
use spongefish::codecs::arkworks_algebra::{ByteDomainSeparator, FieldDomainSeparator};

use super::parameters::StirConfig;
use crate::{
    fs_utils::{OODDomainSeparator, WhirPoWDomainSeparator},
    whir::domainsep::DigestDomainSeparator,
};

// pub trait DigestDomainSeparator<MerkleConfig: Config> {
//     #[must_use]
//     fn add_digest(self, label: &str) -> Self;
// }

pub trait StirDomainSeparator<F: FftField, MerkleConfig: Config> {
    #[must_use]
    fn commit_statement<PowStrategy>(
        self,
        params: &StirConfig<F, MerkleConfig, PowStrategy>,
    ) -> Self;

    #[must_use]
    fn add_stir_proof<PowStrategy>(self, params: &StirConfig<F, MerkleConfig, PowStrategy>)
        -> Self;
}

impl<F, MerkleConfig, DomainSeparator> StirDomainSeparator<F, MerkleConfig> for DomainSeparator
where
    F: FftField,
    MerkleConfig: Config,
    DomainSeparator:
        ByteDomainSeparator + FieldDomainSeparator<F> + DigestDomainSeparator<MerkleConfig>,
{
    fn commit_statement<PowStrategy>(
        self,
        _params: &StirConfig<F, MerkleConfig, PowStrategy>,
    ) -> Self {
        self.add_digest("merkle_digest")
    }

    fn add_stir_proof<PowStrategy>(
        mut self,
        params: &StirConfig<F, MerkleConfig, PowStrategy>,
    ) -> Self {
        // TODO: Add statement
        self = self
            .challenge_scalars(1, "folding_randomness")
            .pow(params.starting_folding_pow_bits);

        for r in &params.round_parameters {
            self = self
                .add_digest("merkle_digest")
                .add_ood(r.ood_samples)
                .challenge_bytes(32, "stir_queries_seed")
                .pow(r.pow_bits)
                .challenge_scalars(1, "combination_randomness")
                .challenge_scalars(1, "folding_randomness");
        }

        self.add_scalars(1 << params.final_log_degree, "final_coeffs")
            .challenge_bytes(32, "final_queries_seed")
            .pow(params.final_pow_bits)
    }
}
