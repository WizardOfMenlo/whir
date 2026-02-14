mod committer;
mod prover;
pub(crate) mod utils;
mod verifier;

pub use utils::BlindingEvaluations;

use ark_ff::FftField;
use serde::Serialize;

use crate::{
    algebra::{fields::FieldWithSize, polynomials::CoefficientList},
    parameters::{FoldingFactor, MultivariateParameters, ProtocolParameters, SoundnessType},
    protocols::{irs_commit, whir},
};

// ── ZK WHIR Config ───────────────────────────────────────────────────

/// Configuration for the ZK WHIR protocol.
///
/// Bundles the main WHIR config (for the witness polynomial) and the
/// blinding polynomial WHIR config (for blinding polynomials committed
/// in a separate Merkle tree).
///
/// ZK parameters (`num_witness_variables`, `num_blinding_variables`) are
/// derivable from the two configs and exposed as methods.
#[derive(Clone, Serialize)]
#[serde(bound = "F: FftField")]
pub struct Config<F: FftField> {
    /// Main WHIR config for the witness polynomial.
    pub main: whir::Config<F>,
    /// WHIR config for blinding polynomials (ℓ+1 variables, batch_size = N×(μ+1)).
    pub blinding_poly_commitment: whir::Config<F>,
}

impl<F: FftField + FieldWithSize> Config<F> {
    /// Create a new ZK WHIR config from multivariate and protocol parameters
    /// for both the main and blinding polynomial WHIR configurations.
    pub fn new(
        main_mv_params: MultivariateParameters<F>,
        main_whir_params: &ProtocolParameters,
        blinding_mv_params: MultivariateParameters<F>,
        blinding_whir_params: &ProtocolParameters,
    ) -> Self {
        let main = whir::Config::new(main_mv_params, main_whir_params);
        let blinding_poly_commitment = whir::Config::new(blinding_mv_params, blinding_whir_params);
        Self {
            main,
            blinding_poly_commitment,
        }
    }

    /// Build a ZK WHIR config with automatically-derived blinding parameters.
    ///
    /// This is a convenience constructor for the common case where the blinding
    /// config inherits most settings from the main config. Callers only need
    /// to specify the main parameters plus the blinding folding factor and the
    /// number of polynomials (which determines the blinding batch size).
    ///
    /// The blinding config inherits `security_level`, `soundness_type`,
    /// `starting_log_inv_rate`, and `hash_id` from the main parameters, sets
    /// `pow_bits` to 0, and computes `num_variables` and `batch_size` from
    /// the derived ZK parameters.
    pub fn with_auto_helper(
        main_mv_params: MultivariateParameters<F>,
        main_whir_params: &ProtocolParameters,
        blinding_folding_factor: FoldingFactor,
        num_polynomials: usize,
    ) -> Self {
        let main = whir::Config::new(main_mv_params, main_whir_params);
        let num_blinding_variables = Self::compute_num_blinding_variables(&main);
        let num_witness_variables = main.initial_num_variables();
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
        let blinding_poly_commitment = whir::Config::new(blinding_mv_params, &blinding_whir_params);
        Self {
            main,
            blinding_poly_commitment,
        }
    }

    /// Compute the number of blinding variables (ℓ) from the main WHIR config.
    ///
    /// ℓ is chosen such that 2^ℓ > conservative query upper bound.
    fn compute_num_blinding_variables(main: &whir::Config<F>) -> usize {
        let num_witness_variables = main.initial_num_variables();
        let folding_factor_size = 1 << main.initial_sumcheck.num_rounds;
        let initial_query_count = main
            .round_configs
            .first()
            .map_or(main.initial_committer.in_domain_samples, |r| {
                r.irs_committer.in_domain_samples
            });

        let query_upper_bound =
            2 * folding_factor_size * initial_query_count + 4 * num_witness_variables + 10;
        let num_blinding_variables = (query_upper_bound as f64).log2().ceil() as usize;
        assert!(
            num_blinding_variables < num_witness_variables,
            "ZK requires ℓ < μ (ℓ={num_blinding_variables}, μ={num_witness_variables}). \
             Increase num_variables or lower security_level/queries. \
             (q_ub={query_upper_bound}, k={folding_factor_size}, q1={initial_query_count})"
        );
        num_blinding_variables
    }
}

impl<F: FftField> Config<F> {
    /// Number of variables in the witness polynomial (μ in the paper).
    pub fn num_witness_variables(&self) -> usize {
        self.main.initial_num_variables()
    }

    /// Number of variables for blinding polynomials (ℓ in the paper).
    pub fn num_blinding_variables(&self) -> usize {
        self.blinding_poly_commitment.initial_num_variables() - 1
    }
}

/// Witness: contains commitment witnesses for all ZK components.
///
/// Produced by [`Config::commit`] and consumed by [`Config::prove`].
#[derive(Clone)]
pub struct Witness<F: FftField> {
    /// Witnesses for [[f̂₁]] = [[f₁ + msk₁]], ..., [[fₙ]] = [[fₙ + mskₙ]] in main WHIR
    pub f_hat_witnesses: Vec<irs_commit::Witness<F::BasePrimeField, F>>,

    /// Single batch witness for all blinding polynomials [[M, ĝ₁, ..., ĝμ]]
    /// committed via blinding_poly_commitment config with batch_size = μ+1
    pub blinding_witness: irs_commit::Witness<F::BasePrimeField, F>,

    /// Reference to blinding data for each polynomial
    pub blinding_polynomials: Vec<utils::BlindingPolynomials<F>>,

    /// Base-field representations of M polynomials (for blinding WHIR prove)
    pub m_polys_base: Vec<CoefficientList<F::BasePrimeField>>,

    /// Base-field representations of embedded ĝⱼ polynomials (for blinding WHIR prove)
    /// Each ĝⱼ is embedded from ℓ to (ℓ+1) variables for each polynomial
    pub g_hats_embedded_bases: Vec<Vec<CoefficientList<F::BasePrimeField>>>,
}

#[cfg(test)]
mod tests {

    use ark_std::rand::{rngs::StdRng, SeedableRng};

    use super::*;
    use crate::{
        algebra::{
            fields::{Field64, Field64_2},
            polynomials::{CoefficientList, MultilinearPoint},
            Weights,
        },
        hash,
        parameters::{FoldingFactor, MultivariateParameters, ProtocolParameters, SoundnessType},
        transcript::{codecs::Empty, DomainSeparator, ProverState, VerifierState},
    };

    /// Field type used in the tests.
    type F = Field64;

    /// Extension field type used in the tests.
    type EF = Field64_2;

    /// Run a complete zkWHIR proof lifecycle: sample preprocessing, commit, prove, verify.
    ///
    /// This function:
    /// - builds N multilinear polynomials with the specified number of variables,
    /// - constructs a `whir_zk::Config` bundling main + blinding WHIR configs,
    /// - commits (producing f̂ᵢ = fᵢ + mskᵢ and blinding commitments),
    /// - generates a ZK proof, then verifies it.
    fn make_whir_zk_things(
        num_variables: usize,
        folding_factor: FoldingFactor,
        num_points: usize,
        num_polynomials: usize,
        soundness_type: SoundnessType,
        pow_bits: usize,
    ) {
        assert!(num_polynomials >= 1, "need at least 1 polynomial");
        let num_coeffs = 1 << num_variables;
        let mut rng = StdRng::seed_from_u64(12345);

        // ── Main WHIR parameters ──
        let mv_params = MultivariateParameters::new(num_variables);
        let whir_params = ProtocolParameters {
            initial_statement: true,
            security_level: 32,
            pow_bits,
            folding_factor,
            soundness_type,
            starting_log_inv_rate: 1,
            batch_size: 1,
            hash_id: hash::SHA2,
        };

        // ── Build ZK config with auto-derived blinding parameters ──
        let zk_config = Config::with_auto_helper(
            mv_params,
            &whir_params,
            FoldingFactor::Constant(1),
            num_polynomials,
        );
        eprintln!(
            "ZK params: num_blinding_variables={}, num_witness_variables={}, num_polynomials={}",
            zk_config.num_blinding_variables(),
            zk_config.num_witness_variables(),
            num_polynomials
        );
        eprintln!("{}", zk_config.main);

        // ── Sample N random polynomials ──
        let mut polynomials: Vec<CoefficientList<F>> = Vec::with_capacity(num_polynomials);
        for poly_idx in 0..num_polynomials {
            // Each polynomial has distinct coefficients
            let coeffs: Vec<F> = (0..num_coeffs)
                .map(|coeff_idx| F::from((poly_idx * num_coeffs + coeff_idx + 1) as u64))
                .collect();
            polynomials.push(CoefficientList::new(coeffs));
        }

        // ── Create evaluation constraints ──
        // evaluations layout: row-major [w₀_p₀, w₀_p₁, ..., w₁_p₀, ...]
        let mut weights = Vec::new();
        let mut evaluations = Vec::new();
        for _ in 0..num_points {
            let point = MultilinearPoint::rand(&mut rng, num_variables);
            weights.push(Weights::evaluation(point.clone()));
            for poly in &polynomials {
                evaluations.push(poly.mixed_evaluate(zk_config.main.embedding(), &point));
            }
        }
        let weight_refs: Vec<&Weights<EF>> = weights.iter().collect();
        let polynomial_refs: Vec<&CoefficientList<F>> = polynomials.iter().collect();

        // ── Set up Fiat-Shamir transcript ──
        let ds = DomainSeparator::protocol(&zk_config)
            .session(&format!("ZK Test at {}:{}", file!(), line!()))
            .instance(&Empty);
        let mut prover_state = ProverState::new_std(&ds);

        // ── ZK commitment: commit to f̂ᵢ = fᵢ + mskᵢ, plus blinding polynomials ──
        let zk_witness = zk_config.commit(&mut prover_state, &polynomial_refs);

        // ── ZK proof: prove knowledge of {fᵢ} via blinded virtual oracle ──
        let (_point, _evals) = zk_config.prove(
            &mut prover_state,
            &polynomial_refs,
            &zk_witness,
            &weight_refs,
            &evaluations,
        );
        let proof = prover_state.proof();
        let mut verifier_state = VerifierState::new_std(&ds, &proof);

        // ── Receive commitments from transcript (mirrors commit order) ──
        let (f_hat_commitments, blinding_commitment) = zk_config
            .receive_commitments(&mut verifier_state, num_polynomials)
            .unwrap();
        let f_hat_commitment_refs: Vec<_> = f_hat_commitments.iter().collect();

        // ── Verify ZK proof ──
        let verify_result = zk_config.verify(
            &mut verifier_state,
            &f_hat_commitment_refs,
            &blinding_commitment,
            &weight_refs,
            &evaluations,
        );
        assert!(
            verify_result.is_ok(),
            "ZK verification failed (N={num_polynomials}): {:?}",
            verify_result.err()
        );
    }

    #[test]
    fn test_whir_zk_basic() {
        // ZK requires ℓ < μ. With security_level=32, the query count drives ℓ ≈ 8–10,
        // so num_variables (= μ) must be large enough.
        let configs: &[(usize, usize)] = &[
            // (num_variables, folding_factor)
            (10, 2),
            (12, 2),
            (12, 3),
            (12, 4),
        ];
        let num_points = [0, 1, 2];

        for &(num_variable, folding_factor) in configs {
            for num_points in num_points {
                eprintln!();
                dbg!(num_variable, folding_factor, num_points);

                make_whir_zk_things(
                    num_variable,
                    FoldingFactor::Constant(folding_factor),
                    num_points,
                    1, // single polynomial
                    SoundnessType::ConjectureList,
                    0,
                );
            }
        }
    }

    #[test]
    fn test_whir_zk_with_pow() {
        // Test ZK with proof-of-work enabled
        make_whir_zk_things(
            12,
            FoldingFactor::Constant(2),
            2,
            1,
            SoundnessType::ConjectureList,
            5,
        );
    }

    #[test]
    fn test_whir_zk_soundness_types() {
        // Test ZK across different soundness types
        let soundness_types = [
            SoundnessType::ConjectureList,
            SoundnessType::ProvableList,
            SoundnessType::UniqueDecoding,
        ];

        for soundness_type in soundness_types {
            eprintln!();
            dbg!(soundness_type);
            make_whir_zk_things(12, FoldingFactor::Constant(2), 1, 1, soundness_type, 0);
        }
    }

    #[test]
    fn test_whir_zk_mixed_folding() {
        // Test ZK with mixed folding factors
        make_whir_zk_things(
            12,
            FoldingFactor::ConstantFromSecondRound(3, 3),
            1,
            1,
            SoundnessType::ConjectureList,
            0,
        );
    }

    #[test]
    fn test_whir_zk_multi_polynomial() {
        // Test ZK with multiple polynomials (batched proving/verification)
        let configs: &[(usize, usize, usize, usize)] = &[
            // (num_variables, folding_factor, num_points, num_polynomials)
            (10, 2, 1, 2), // 2 polynomials, 1 constraint
            (10, 2, 2, 2), // 2 polynomials, 2 constraints
            (12, 2, 1, 3), // 3 polynomials, 1 constraint
            (12, 3, 2, 2), // 2 polynomials, 2 constraints, larger folding
        ];

        for &(num_variables, folding_factor, num_points, num_polynomials) in configs {
            eprintln!();
            dbg!(num_variables, folding_factor, num_points, num_polynomials);

            make_whir_zk_things(
                num_variables,
                FoldingFactor::Constant(folding_factor),
                num_points,
                num_polynomials,
                SoundnessType::ConjectureList,
                0,
            );
        }
    }
}
