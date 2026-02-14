mod committer;
pub mod config;
mod prover;
pub(crate) mod utils;
mod verifier;

pub use config::{Config, ZkParams};
pub use utils::{HelperEvaluations, ZkWitness};

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
        transcript::{codecs::Empty, ProverState, VerifierState},
    };

    /// Field type used in the tests.
    type F = Field64;

    /// Extension field type used in the tests.
    type EF = Field64_2;

    /// Run a complete zkWHIR proof lifecycle: sample preprocessing, commit, prove, verify.
    ///
    /// This function:
    /// - builds N multilinear polynomials with the specified number of variables,
    /// - samples N independent ZK preprocessing polynomials (msk, g₀, ĝ₁..ĝμ, M),
    /// - constructs a `whir_zk::Config` bundling main + helper WHIR configs,
    /// - commits (producing f̂ᵢ = fᵢ + mskᵢ and helper commitments),
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

        // ── Build ZK config with auto-derived helper parameters ──
        let zk_config = Config::with_auto_helper(
            mv_params,
            &whir_params,
            FoldingFactor::Constant(1),
            num_polynomials,
        );
        let zk_params = &zk_config.zk_params;
        eprintln!(
            "ZK params: num_helper_variables={}, num_witness_variables={}, num_polynomials={}",
            zk_params.num_helper_variables, zk_params.num_witness_variables, num_polynomials
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
        let ds = zk_config
            .domain_separator()
            .session(&format!("ZK Test at {}:{}", file!(), line!()))
            .instance(&Empty);
        let mut prover_state = ProverState::new_std(&ds);

        // ── ZK commitment: commit to f̂ᵢ = fᵢ + mskᵢ, plus helper polynomials ──
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
        let (f_hat_commitments, helper_commitment) = zk_config
            .receive_commitments(&mut verifier_state, num_polynomials)
            .unwrap();
        let f_hat_commitment_refs: Vec<_> = f_hat_commitments.iter().collect();

        // ── Verify ZK proof ──
        let verify_result = zk_config.verify(
            &mut verifier_state,
            &f_hat_commitment_refs,
            &helper_commitment,
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
