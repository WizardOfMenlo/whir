use ark_crypto_primitives::merkle_tree::Config;
use ark_ff::FftField;
use nimue::plugins::{ark::*, pow::PoWIOPattern};

use crate::sumcheck::prover_not_skipping::SumcheckNotSkippingIOPattern;

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
        // TODO: This is not generic on the digest type
        self.add_bytes(32, "merkle_digest")
    }

    fn add_whir_proof<MerkleConfig: Config>(
        mut self,
        params: &WhirConfig<F, MerkleConfig>,
    ) -> Self {
        // TODO: Add statement
        self = self.challenge_scalars(params.folding_factor, "folding_randomness");

        for r in &params.round_parameters {
            self = self
                .add_bytes(32, "merkle_digest")
                .add_ood(r.ood_samples)
                .challenge_bytes(32, "stir_queries_seed")
                .pow(r.pow_bits)
                .challenge_scalars(1, "combination_randomness")
                .add_sumcheck(params.folding_factor);
        }

        self.add_scalars(1, "final_coeffs")
            .challenge_bytes(32, "final_queries_seed")
            .pow(params.final_pow_bits)
    }
}

pub trait OODIOPattern<F: FftField> {
    fn add_ood(self, num_samples: usize) -> Self;
}

impl<F> OODIOPattern<F> for IOPattern
where
    F: FftField,
    IOPattern: FieldIOPattern<F>,
{
    fn add_ood(self, num_samples: usize) -> Self {
        if num_samples > 0 {
            self.challenge_scalars(num_samples, "ood_query")
                .add_scalars(num_samples, "ood_ans")
        } else {
            self
        }
    }
}

pub trait WhirPoWIOPattern {
    fn pow(self, bits: usize) -> Self;
}

impl WhirPoWIOPattern for IOPattern
where
    IOPattern: PoWIOPattern,
{
    fn pow(self, bits: usize) -> Self {
        if bits > 0 {
            self.challenge_pow("pow_queries")
        } else {
            self
        }
    }
}
