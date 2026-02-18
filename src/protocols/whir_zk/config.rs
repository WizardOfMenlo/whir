use ark_ff::FftField;
use serde::{Deserialize, Serialize};

use crate::{
    algebra::fields::FieldWithSize,
    parameters::{FoldingFactor, MultivariateParameters, ProtocolParameters, SoundnessType},
    protocols::whir,
};

/// ZK WHIR configuration.
///
/// This mirrors the two-commitment view from the protocol notes:
/// - `blinded_commitment` for the witness-side WHIR,
/// - `blinding_commitment` for batched blinding polynomials.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize, Debug)]
#[serde(bound = "F: FftField")]
pub struct Config<F: FftField> {
    pub blinded_commitment: whir::Config<F>,
    pub blinding_commitment: whir::Config<F>,
}

impl<F> Config<F>
where
    F: FftField + FieldWithSize,
{
    /// Build a draft zkWHIR config from upstream WHIR as source of truth.
    ///
    /// The blinding side follows the same security/hash parameters and uses
    /// `ConjectureList` soundness with `pow_bits = 0`, as captured in the
    /// local zk notes.
    pub fn new(
        main_mv_params: MultivariateParameters<F>,
        main_whir_params: &ProtocolParameters,
        blinding_folding_factor: FoldingFactor,
        num_polynomials: usize,
    ) -> Self {
        let blinded_commitment = whir::Config::new(main_mv_params, main_whir_params);
        let num_witness_variables = blinded_commitment.initial_num_variables();
        let num_blinding_variables = Self::compute_num_blinding_variables(&blinded_commitment);

        let blinding_mv_params = MultivariateParameters::new(num_blinding_variables + 1);
        let blinding_whir_params = ProtocolParameters {
            initial_statement: true,
            security_level: main_whir_params.security_level,
            pow_bits: 0,
            folding_factor: blinding_folding_factor,
            soundness_type: SoundnessType::ConjectureList,
            starting_log_inv_rate: main_whir_params.starting_log_inv_rate,
            batch_size: num_polynomials * (num_witness_variables + 1),
            hash_id: main_whir_params.hash_id,
        };
        let blinding_commitment = whir::Config::new(blinding_mv_params, &blinding_whir_params);

        Self {
            blinded_commitment,
            blinding_commitment,
        }
    }

    fn compute_num_blinding_variables(blinded: &whir::Config<F>) -> usize {
        // Keep the same conservative query upper bound used during local zk debugging.
        let num_witness_variables = blinded.initial_num_variables();
        let folding_factor_size = 1 << blinded.initial_sumcheck.num_rounds;
        let initial_query_count = blinded
            .round_configs
            .first()
            .map_or(blinded.initial_committer.in_domain_samples, |r| {
                r.irs_committer.in_domain_samples
            });

        let query_upper_bound =
            2 * folding_factor_size * initial_query_count + 4 * num_witness_variables + 10;
        #[allow(clippy::cast_sign_loss)]
        let num_blinding_variables = (query_upper_bound as f64).log2().ceil() as usize;
        assert!(num_blinding_variables < num_witness_variables);
        num_blinding_variables
    }
}

impl<F: FftField> Config<F> {
    pub fn num_witness_variables(&self) -> usize {
        self.blinded_commitment.initial_num_variables()
    }

    pub fn num_blinding_variables(&self) -> usize {
        self.blinding_commitment.initial_num_variables() - 1
    }
}
