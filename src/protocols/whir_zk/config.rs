use ark_ff::FftField;

use super::utils::IrsDomainParams;
use crate::{
    algebra::fields::FieldWithSize,
    parameters::{FoldingFactor, MultivariateParameters, ProtocolParameters, SoundnessType},
    protocols::whir,
};

#[derive(Clone)]
pub struct ZkParams {
    /// Number of variables for helper polynomials (ℓ in the paper).
    /// Chosen such that 2^ℓ > conservative query upper bound.
    pub num_helper_variables: usize,

    /// Number of variables in the witness polynomial (μ in the paper).
    pub num_witness_variables: usize,
}

impl ZkParams {
    /// Compute num_helper_variables and num_witness_variables from WHIR parameters.
    pub fn from_whir_params<F: FftField>(whir_params: &whir::Config<F>) -> Self {
        // Number of variables = log2 of polynomial size
        let num_witness_variables = whir_params.initial_sumcheck.initial_size.ilog2() as usize;
        // Folding factor size = 2^folding_factor
        let folding_factor_size = 1 << whir_params.initial_sumcheck.num_rounds;
        // Number of in-domain query samples in the first round
        // (or initial commitment queries if there are no rounds)
        let initial_query_count = whir_params
            .round_configs
            .first()
            .map_or(whir_params.initial_committer.in_domain_samples, |r| {
                r.irs_committer.in_domain_samples
            });

        let query_upper_bound =
            2 * folding_factor_size * initial_query_count + 4 * num_witness_variables + 10;
        let num_helper_variables = (query_upper_bound as f64).log2().ceil() as usize;
        assert!(
            num_helper_variables < num_witness_variables,
            "ZK requires ℓ < μ (ℓ={num_helper_variables}, μ={num_witness_variables}). \
             Increase num_variables or lower security_level/queries. \
             (q_ub={query_upper_bound}, k={folding_factor_size}, q1={initial_query_count})"
        );
        Self {
            num_helper_variables,
            num_witness_variables,
        }
    }

    pub fn helper_batch_size(&self, number_of_polynomials: usize) -> usize {
        number_of_polynomials * (self.num_witness_variables + 1)
    }
}

// ── ZK WHIR Config ───────────────────────────────────────────────────

/// Configuration for the ZK WHIR protocol.
///
/// Bundles the main WHIR config (for the witness polynomial), the helper
/// WHIR config (for helper polynomials committed in a separate Merkle tree),
/// and the derived ZK parameters.
pub struct Config<F: FftField> {
    /// Main WHIR config for the witness polynomial.
    pub main: whir::Config<F>,
    /// Helper WHIR config for helper polynomials (ℓ+1 variables, batch_size = N×(μ+1)).
    pub helper: whir::Config<F>,
    /// ZK parameters derived from the main config.
    pub zk_params: ZkParams,
    /// Precomputed IRS domain parameters (generators, coset roots, sub-domain powers).
    irs_domain: IrsDomainParams<F>,
}

impl<F: FftField + FieldWithSize> Config<F> {
    /// Create a new ZK WHIR config from multivariate and protocol parameters
    /// for both the main and helper WHIR configurations.
    ///
    /// Internally constructs `whir::Config` for both main and helper, and
    /// computes `ZkParams` automatically from the main config.
    pub fn new(
        main_mv_params: MultivariateParameters<F>,
        main_whir_params: &ProtocolParameters,
        helper_mv_params: MultivariateParameters<F>,
        helper_whir_params: &ProtocolParameters,
    ) -> Self {
        let main = whir::Config::new(main_mv_params, main_whir_params);
        let helper = whir::Config::new(helper_mv_params, helper_whir_params);
        let zk_params = ZkParams::from_whir_params(&main);
        let irs_domain = IrsDomainParams::from_config(&main);
        Self {
            main,
            helper,
            zk_params,
            irs_domain,
        }
    }

    /// Build a ZK WHIR config with automatically-derived helper parameters.
    ///
    /// This is a convenience constructor for the common case where the helper
    /// config inherits most settings from the main config. Callers only need
    /// to specify the main parameters plus the helper folding factor and the
    /// number of polynomials (which determines the helper batch size).
    ///
    /// The helper config inherits `security_level`, `soundness_type`,
    /// `starting_log_inv_rate`, and `hash_id` from the main parameters, sets
    /// `pow_bits` to 0, and computes `num_variables` and `batch_size` from
    /// the derived `ZkParams`.
    pub fn with_auto_helper(
        main_mv_params: MultivariateParameters<F>,
        main_whir_params: &ProtocolParameters,
        helper_folding_factor: FoldingFactor,
        num_polynomials: usize,
    ) -> Self {
        let main = whir::Config::new(main_mv_params, main_whir_params);
        let zk_params = ZkParams::from_whir_params(&main);
        let helper_mv_params = MultivariateParameters::new(zk_params.num_helper_variables + 1);
        let helper_whir_params = ProtocolParameters {
            initial_statement: true,
            security_level: main_whir_params.security_level,
            pow_bits: 0,
            folding_factor: helper_folding_factor,
            soundness_type: SoundnessType::ConjectureList,
            starting_log_inv_rate: main_whir_params.starting_log_inv_rate,
            batch_size: zk_params.helper_batch_size(num_polynomials),
            hash_id: main_whir_params.hash_id,
        };
        let helper = whir::Config::new(helper_mv_params, &helper_whir_params);
        let irs_domain = IrsDomainParams::from_config(&main);
        Self {
            main,
            helper,
            zk_params,
            irs_domain,
        }
    }
}

impl<F: FftField> Config<F> {
    /// Build a `DomainSeparator` for the Fiat-Shamir transcript.
    ///
    /// Convenience wrapper around `DomainSeparator::protocol(&self.main)` so
    /// callers do not need to reach into the `main` field.
    pub fn domain_separator(&self) -> crate::transcript::DomainSeparator<'static, ()> {
        crate::transcript::DomainSeparator::protocol(&self.main)
    }

    /// Access precomputed IRS domain parameters.
    pub(crate) fn irs_domain_params(&self) -> &IrsDomainParams<F> {
        &self.irs_domain
    }
}
