mod committer;
mod prover;
pub mod utils;
mod verifier;

pub use utils::{HelperEvaluations, ZkParams, ZkPreprocessingPolynomials, ZkWitness};

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
        protocols::whir::Config,
        transcript::{codecs::Empty, DomainSeparator, ProverState, VerifierState},
    };

    /// Field type used in the tests.
    type F = Field64;

    /// Extension field type used in the tests.
    type EF = Field64_2;

    /// Run a complete zkWHIR proof lifecycle: sample preprocessing, commit_zk, prove_zk.
    ///
    /// This function:
    /// - builds N multilinear polynomials with the specified number of variables,
    /// - samples N independent ZK preprocessing polynomials (msk, g₀, ĝ₁..ĝμ, M),
    /// - constructs a helper Config for the (ℓ+1)-variate helper WHIR,
    /// - commits using commit_zk (producing f̂ᵢ = fᵢ + mskᵢ and helper commitments),
    /// - generates a ZK proof using prove_zk, then verifies with verify_zk.
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

        // ── Main WHIR config ──
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

        let params = Config::new(mv_params, &whir_params);
        eprintln!("{params}");

        // ── Compute ZK parameters (ℓ, μ) ──
        let zk_params = ZkParams::from_whir_params(&params);
        eprintln!(
            "ZK params: ell={}, mu={}, num_polynomials={}",
            zk_params.ell, zk_params.mu, num_polynomials
        );

        // ── Helper WHIR config for (ℓ+1)-variate helper polynomials ──
        //    batch_size = N×(μ+1) so all N polynomial helpers share one Merkle tree
        let helper_mv_params = MultivariateParameters::new(zk_params.ell + 1);
        let helper_whir_params = ProtocolParameters {
            initial_statement: true,
            security_level: 32,
            pow_bits: 0,
            folding_factor: FoldingFactor::Constant(1),
            soundness_type: SoundnessType::ConjectureList,
            starting_log_inv_rate: 1,
            batch_size: zk_params.helper_batch_size(num_polynomials),
            hash_id: hash::SHA2,
        };
        let helper_config = Config::new(helper_mv_params, &helper_whir_params);

        // ── Sample N random polynomials and N independent preprocessings ──
        let mut polynomials: Vec<CoefficientList<F>> = Vec::with_capacity(num_polynomials);
        let mut preprocessings: Vec<ZkPreprocessingPolynomials<EF>> =
            Vec::with_capacity(num_polynomials);
        for i in 0..num_polynomials {
            // Each polynomial has distinct coefficients
            let coeffs: Vec<F> = (0..num_coeffs)
                .map(|j| F::from((i * num_coeffs + j + 1) as u64))
                .collect();
            polynomials.push(CoefficientList::new(coeffs));
            preprocessings.push(ZkPreprocessingPolynomials::<EF>::sample(
                &mut rng,
                zk_params.clone(),
            ));
        }

        // ── Create evaluation constraints ──
        // evaluations layout: row-major [w₀_p₀, w₀_p₁, ..., w₁_p₀, ...]
        let mut weights = Vec::new();
        let mut evaluations = Vec::new();
        for _ in 0..num_points {
            let point = MultilinearPoint::rand(&mut rng, num_variables);
            weights.push(Weights::evaluation(point.clone()));
            for poly in &polynomials {
                evaluations.push(poly.mixed_evaluate(params.embedding(), &point));
            }
        }
        let weight_refs: Vec<&Weights<EF>> = weights.iter().collect();
        let polynomial_refs: Vec<&CoefficientList<F>> = polynomials.iter().collect();

        // ── Set up Fiat-Shamir transcript ──
        let ds = DomainSeparator::protocol(&params)
            .session(&format!("ZK Test at {}:{}", file!(), line!()))
            .instance(&Empty);
        let mut prover_state = ProverState::new_std(&ds);

        // ── ZK commitment: commit to f̂ᵢ = fᵢ + mskᵢ, plus helper polynomials ──
        let zk_witness = params.commit_zk(
            &mut prover_state,
            &polynomial_refs,
            &helper_config,
            &preprocessings.iter().collect::<Vec<_>>(),
        );

        // ── ZK proof: prove knowledge of {fᵢ} via blinded virtual oracle ──
        let (_point, _evals) = params.prove_zk(
            &mut prover_state,
            &polynomial_refs,
            &zk_witness,
            &helper_config,
            &weight_refs,
            &evaluations,
        );
        let proof = prover_state.proof();
        let mut verifier_state = VerifierState::new_std(&ds, &proof);

        // ── Receive commitments from transcript (mirrors commit_zk order) ──
        // One f̂ commitment per polynomial
        let f_hat_commitments: Vec<_> = (0..num_polynomials)
            .map(|_| params.receive_commitment(&mut verifier_state).unwrap())
            .collect();
        let f_hat_commitment_refs: Vec<_> = f_hat_commitments.iter().collect();
        // Single batch commitment for all N×(μ+1) helper polynomials
        let helper_commitment = helper_config
            .receive_commitment(&mut verifier_state)
            .unwrap();

        // ── Verify ZK proof ──
        let verify_result = params.verify_zk(
            &mut verifier_state,
            &f_hat_commitment_refs,
            &helper_commitment,
            &helper_config,
            &zk_params,
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
