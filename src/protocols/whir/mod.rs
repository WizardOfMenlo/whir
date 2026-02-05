mod batching;
mod committer;
mod config;
mod prover;
mod verifier;

pub use self::{
    committer::{Commitment, Witness},
    config::{Config, RoundConfig},
};

#[cfg(test)]
mod tests {
    use ark_ff::Field;

    use super::*;
    use crate::{
        algebra::{
            embedding::Basefield,
            fields::{Field64, Field64_2},
            polynomials::{CoefficientList, EvaluationsList, MultilinearPoint},
            Weights,
        },
        hash,
        parameters::{FoldingFactor, MultivariateParameters, ProtocolParameters, SoundnessType},
        transcript::{codecs::Empty, DomainSeparator, ProverState, VerifierState},
        utils::test_serde,
    };

    /// Field type used in the tests.
    type F = Field64;

    /// Extension field type used in the tests.
    type EF = Field64_2;

    /// Run a complete WHIR STARK proof lifecycle: commit, prove, and verify.
    ///
    /// This function:
    /// - builds a multilinear polynomial with a specified number of variables,
    /// - constructs a STARK statement with constraints based on evaluations and linear relations,
    /// - commits to the polynomial using a Merkle-based commitment scheme,
    /// - generates a proof using the WHIR prover,
    /// - verifies the proof using the WHIR verifier.
    fn make_whir_things(
        num_variables: usize,
        folding_factor: FoldingFactor,
        num_points: usize,
        soundness_type: SoundnessType,
        pow_bits: usize,
    ) {
        // Number of coefficients in the multilinear polynomial (2^num_variables)
        let num_coeffs = 1 << num_variables;

        // Randomness source
        let mut rng = ark_std::test_rng();

        // Configure multivariate polynomial parameters
        let mv_params = MultivariateParameters::new(num_variables);

        // Configure the WHIR protocol parameters
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

        // Build global configuration from multivariate + protocol parameters
        let params = Config::new(mv_params, &whir_params);
        eprintln!("{params}");

        // Test that the config is serializable
        test_serde(&params);

        // Define the multilinear polynomial: constant 1 across all inputs
        let polynomial = CoefficientList::new(vec![F::ONE; num_coeffs]);

        // Generate `num_points` random points in the multilinear domain
        let points: Vec<_> = (0..num_points)
            .map(|_| MultilinearPoint::rand(&mut rng, num_variables))
            .collect();

        // Initialize a statement with no constraints yet
        let mut weights = Vec::new();
        let mut evaluations = Vec::new();

        // For each random point, evaluate the polynomial and create a constraint
        for point in &points {
            weights.push(Weights::evaluation(point.clone()));
            evaluations.push(polynomial.mixed_evaluate(&Basefield::new(), point));
        }

        // Construct a coefficient vector for linear sumcheck constraint
        let input = CoefficientList::new(
            (0..1 << num_variables)
                .map(<EF as Field>::BasePrimeField::from)
                .collect(),
        );

        // Define weights for linear combination
        let linear_claim_weight = Weights::linear(input.into());

        // Convert polynomial to extension field representation
        let poly = EvaluationsList::from(polynomial.clone().lift(&Basefield::new()));

        // Compute the weighted sum of the polynomial (for sumcheck)
        let sum = linear_claim_weight.evaluate(&poly.to_coeffs());

        // Add linear constraint to the statement
        weights.push(linear_claim_weight);
        evaluations.push(sum);

        // Base of the geometric progression [1, 2, 4, 8, 0, ...]
        let geometric_base = F::from(2u64);

        // Number of non-zero terms in the progression.
        let geometric_n = num_coeffs - 1;

        // Create geometric progression weights: [1, a, a^2, a^3, ..., a^(n-1), 0, 0, ..., 0]
        let mut geometric_weights = vec![F::ONE];
        let mut current_power = geometric_base;
        for _i in 1..geometric_n {
            geometric_weights.push(current_power);
            current_power *= geometric_base;
        }

        // Fill remaining corners with zeros
        geometric_weights.resize(num_coeffs, F::from(0));

        // Create EvaluationsList from the geometric progression
        let geometric_weight_list = EvaluationsList::new(geometric_weights);

        // Create geometric weight function
        let geometric_claim_weight =
            Weights::geometric(geometric_base, geometric_n, geometric_weight_list);

        // Convert polynomial to base field evaluation form for weighted_sum
        let poly_base = EvaluationsList::from(polynomial.clone());

        // Compute the weighted sum for geometric constraint
        let geometric_sum = geometric_claim_weight.evaluate(&poly_base.to_coeffs());

        // Add geometric constraint to statement
        weights.push(geometric_claim_weight);
        evaluations.push(geometric_sum);

        // Define the Fiat-Shamir domain separator for committing and proving
        let ds = DomainSeparator::protocol(&params)
            .session(&format!("Test at {}:{}", file!(), line!()))
            .instance(&Empty);

        // Initialize the Merlin transcript from the domain separator
        let mut prover_state = ProverState::new_std(&ds);

        // Create a commitment to the polynomial and generate auxiliary witness data
        let witness = params.commit(&mut prover_state, &[&polynomial]);

        // Generate a STARK proof for the given statement and witness
        let weight_refs = weights.iter().collect::<Vec<_>>();
        params.prove(
            &mut prover_state,
            &[&polynomial],
            &[&witness],
            &weight_refs,
            &evaluations,
        );

        // Reconstruct verifier's view of the transcript using the DomainSeparator and prover's data
        let proof = prover_state.proof();
        let mut verifier_state = VerifierState::new_std(&ds, &proof);

        // Parse the commitment
        let commitment = params.receive_commitment(&mut verifier_state).unwrap();

        // Verify that the generated proof satisfies the statement
        params
            .verify(
                &mut verifier_state,
                &[&commitment],
                &weight_refs,
                &evaluations,
            )
            .unwrap();
    }

    #[test]
    fn test_whir_1() {
        let folding_factors = [1, 2, 3, 4];
        let soundness_type = [
            SoundnessType::ConjectureList,
            SoundnessType::ProvableList,
            SoundnessType::UniqueDecoding,
        ];
        let num_points = [0, 1, 2];
        let pow_bits = [0, 5, 10];

        for folding_factor in folding_factors {
            let num_variables = folding_factor..=3 * folding_factor;
            for num_variable in num_variables {
                for num_points in num_points {
                    for soundness_type in soundness_type {
                        for pow_bits in pow_bits {
                            eprintln!();
                            dbg!(
                                folding_factor,
                                num_variable,
                                num_points,
                                soundness_type,
                                pow_bits
                            );

                            make_whir_things(
                                num_variable,
                                FoldingFactor::Constant(folding_factor),
                                num_points,
                                soundness_type,
                                pow_bits,
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_fail() {
        make_whir_things(
            3,
            FoldingFactor::Constant(2),
            0,
            SoundnessType::ConjectureList,
            0,
        );
    }

    #[test]
    fn test_whir_mixed_folding_factors() {
        let folding_factors = [1, 2, 3, 4];
        let num_points = [0, 1, 2];

        for initial_folding_factor in folding_factors {
            for folding_factor in folding_factors {
                if initial_folding_factor == folding_factor {
                    continue;
                }
                let n = std::cmp::max(initial_folding_factor, folding_factor);
                let num_variables = n..=3 * n;
                for num_variable in num_variables {
                    for num_points in num_points {
                        eprintln!();
                        dbg!(
                            initial_folding_factor,
                            folding_factor,
                            num_variable,
                            num_points,
                        );

                        make_whir_things(
                            num_variable,
                            FoldingFactor::ConstantFromSecondRound(
                                initial_folding_factor,
                                folding_factor,
                            ),
                            num_points,
                            SoundnessType::ProvableList,
                            5,
                        );
                    }
                }
            }
        }
    }

    /// Test batch proving with multiple independent polynomials and statements.
    ///
    /// Creates N separate polynomials, commits to each independently, and uses RLC to batch-prove
    /// them together. This verifies the full lifecycle: commitment, batch proving, and verification.
    fn make_whir_batch_things(
        num_variables: usize,
        folding_factor: FoldingFactor,
        num_points_per_poly: usize,
        num_polynomials: usize,
        soundness_type: SoundnessType,
        pow_bits: usize,
    ) {
        let num_coeffs = 1 << num_variables;
        let mut rng = ark_std::test_rng();

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

        // Create N different polynomials
        let polynomials: Vec<_> = (0..num_polynomials)
            .map(|i| {
                // Different polynomials: first is all 1s, second is all 2s, etc.
                CoefficientList::new(vec![F::from((i + 1) as u64); num_coeffs])
            })
            .collect();
        let poly_refs = polynomials.iter().collect::<Vec<_>>();

        // Create weights to constraint the polynomials with.
        // Add random point constraints
        let mut weights = Vec::new();
        for _ in 0..num_points_per_poly {
            let point = MultilinearPoint::rand(&mut rng, num_variables);
            weights.push(Weights::evaluation(point));
        }
        // Add linear constraint
        let input = CoefficientList::new(
            (0..1 << num_variables)
                .map(<EF as Field>::BasePrimeField::from)
                .collect(),
        );
        weights.push(Weights::linear(input.into()));
        let weights_refs = weights.iter().collect::<Vec<_>>();

        // Evaluate all polys on all weights to get constraints
        let evaluations = weights
            .iter()
            .flat_map(|weights| poly_refs.iter().map(|poly| weights.evaluate(poly)))
            .collect::<Vec<_>>();

        // Set up domain separator for batch proving
        // Each polynomial needs its own commitment phase
        let ds = DomainSeparator::protocol(&params)
            .session(&format!("Test at {}:{}", file!(), line!()))
            .instance(&Empty);
        let mut prover_state = ProverState::new_std(&ds);

        // Commit to each polynomial and generate witnesses
        let mut witnesses = Vec::new();
        for poly in &polynomials {
            let witness = params.commit(&mut prover_state, &[poly]);
            witnesses.push(witness);
        }
        let witness_refs = witnesses.iter().collect::<Vec<_>>();

        // Batch prove all polynomials together
        let (_point, _evals) = params.prove(
            &mut prover_state,
            &poly_refs,
            &witness_refs,
            &weights_refs,
            &evaluations,
        );

        // Reconstruct verifier's transcript view
        let proof = prover_state.proof();
        let mut verifier_state = VerifierState::new_std(&ds, &proof);

        // Parse all N commitments from the transcript
        let mut commitments = Vec::new();
        for _ in 0..num_polynomials {
            let commitment = params.receive_commitment(&mut verifier_state).unwrap();
            commitments.push(commitment);
        }
        let commitment_refs = commitments.iter().collect::<Vec<_>>();

        // Verify the batched proof
        params
            .verify(
                &mut verifier_state,
                &commitment_refs,
                &weights_refs,
                &evaluations,
            )
            .unwrap();
    }

    #[test]
    fn test_whir_batch_1() {
        // Test with different configurations
        let folding_factors = [1, 2, 3, 4];
        let num_polynomials = [2, 3, 4];
        let num_points = [0, 1, 2];

        for initial_folding_factor in folding_factors {
            for folding_factor in folding_factors {
                let n = std::cmp::max(initial_folding_factor, folding_factor);
                // TODO: Batching with small number of variables..
                for num_variables in (initial_folding_factor + folding_factor)..=3 * n {
                    for num_polys in num_polynomials {
                        for num_points_per_poly in num_points {
                            eprintln!();
                            dbg!(
                                initial_folding_factor,
                                folding_factor,
                                num_variables,
                                num_polys,
                                num_points_per_poly,
                            );
                            make_whir_batch_things(
                                num_variables,
                                if initial_folding_factor == folding_factor {
                                    FoldingFactor::Constant(folding_factor)
                                } else {
                                    FoldingFactor::ConstantFromSecondRound(
                                        initial_folding_factor,
                                        folding_factor,
                                    )
                                },
                                num_points_per_poly,
                                num_polys,
                                SoundnessType::ConjectureList,
                                0, // pow_bits
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_whir_batch_single_polynomial() {
        // Edge case: batch proving with just one polynomial should also work
        make_whir_batch_things(
            6, // num_variables
            FoldingFactor::Constant(2),
            2, // num_points_per_poly
            1, // num_polynomials (single!)
            SoundnessType::ConjectureList,
            0,
        );
    }

    /// Test that batch verification rejects proofs with mismatched polynomials.
    ///
    /// This security test verifies that the cross-term commitment prevents the prover from
    /// using a different polynomial than what was committed. The prover commits to poly2 but
    /// attempts to use poly_wrong for evaluation, which should cause verification to fail.
    #[test]
    #[cfg_attr(feature = "verifier_panics", should_panic)]
    fn test_whir_batch_rejects_invalid_constraint() {
        // Setup parameters
        let num_variables = 4;
        let folding_factor = FoldingFactor::Constant(2);
        let num_polynomials = 2;
        let num_coeffs = 1 << num_variables;
        let mut rng = ark_std::test_rng();

        let mv_params = MultivariateParameters::<F>::new(num_variables);
        let whir_params = ProtocolParameters {
            initial_statement: true,
            security_level: 32,
            pow_bits: 0,
            folding_factor,
            soundness_type: SoundnessType::ConjectureList,
            starting_log_inv_rate: 1,
            batch_size: 1,
            hash_id: hash::SHA2,
        };

        let params = Config::new(mv_params, &whir_params);

        // Create test polynomials
        let poly1 = CoefficientList::new(vec![F::ONE; num_coeffs]);
        let poly2 = CoefficientList::new(vec![F::from(2u64); num_coeffs]);
        let poly_wrong = CoefficientList::new(vec![F::from(999u64); num_coeffs]);

        // Create test weights
        let weights = vec![
            Weights::evaluation(MultilinearPoint::rand(&mut rng, num_variables)),
            Weights::evaluation(MultilinearPoint::rand(&mut rng, num_variables)),
        ];
        let weights_ref = weights.iter().collect::<Vec<_>>();

        // Create valid evaluations for (poly1, polywrong)
        let evaluations = weights
            .iter()
            .flat_map(|weights| [&poly1, &poly_wrong].map(|poly| weights.evaluate(poly)))
            .collect::<Vec<_>>();

        // Commit to the correct polynomials
        let ds = DomainSeparator::protocol(&params)
            .session(&format!("Test at {}:{}", file!(), line!()))
            .instance(&Empty);
        let mut prover_state = ProverState::new_std(&ds);

        let witness1 = params.commit(&mut prover_state, &[&poly1]);
        let witness2 = params.commit(&mut prover_state, &[&poly2]);

        // Generate proof with mismatched polynomials
        // The prover will compute cross-terms using poly_wrong, not poly2
        let (_evalpoint, _values) = params.prove(
            &mut prover_state,
            &[&poly1, &poly_wrong],
            &[&witness1, &witness2],
            &weights_ref,
            &evaluations,
        );

        // Verification should fail because the cross-terms don't match the commitment
        let proof = prover_state.proof();
        let mut verifier_state = VerifierState::new_std(&ds, &proof);

        let mut commitments = Vec::new();
        for _ in 0..num_polynomials {
            let parsed_commitment = params.receive_commitment(&mut verifier_state).unwrap();
            commitments.push(parsed_commitment);
        }

        let verify_result = params.verify(
            &mut verifier_state,
            &[&commitments[0], &commitments[1]],
            &weights_ref,
            &evaluations,
        );
        assert!(
            verify_result.is_err(),
            "Verifier should reject mismatched polynomial"
        );
    }

    /// Test batch proving with batch_size > 1 (multiple polynomials per commitment).
    ///
    /// This tests the case where each commitment contains multiple stacked polynomials
    /// (e.g., masked witness + random blinding for ZK), and we batch-prove multiple
    /// such commitments together.
    ///
    /// This was a regression test for a bug where the RLC combination of stacked
    /// leaf answers was incorrect when batch_size > 1.
    fn make_whir_batch_with_batch_size(
        num_variables: usize,
        folding_factor: FoldingFactor,
        num_points_per_poly: usize,
        num_witnesses: usize,
        batch_size: usize,
        soundness_type: SoundnessType,
        pow_bits: usize,
    ) {
        let num_coeffs = 1 << num_variables;
        let mut rng = ark_std::test_rng();

        let mv_params = MultivariateParameters::new(num_variables);
        let whir_params = ProtocolParameters {
            initial_statement: true,
            security_level: 32,
            pow_bits,
            folding_factor,
            soundness_type,
            starting_log_inv_rate: 1,
            batch_size, // KEY: batch_size > 1
            hash_id: hash::SHA2,
        };

        let params = Config::new(mv_params, &whir_params);

        // Create polynomials for each witness
        // Each witness will contain batch_size polynomials committed together
        let mut all_polynomials: Vec<Vec<CoefficientList<F>>> = Vec::new();
        for w in 0..num_witnesses {
            let witness_polys: Vec<_> = (0..batch_size)
                .map(|b| {
                    // Different polynomials: witness 0 batch 0 = all 1s, witness 0 batch 1 = all 2s, etc.
                    CoefficientList::new(vec![
                        F::from(((w * batch_size + b) + 1) as u64);
                        num_coeffs
                    ])
                })
                .collect();
            all_polynomials.push(witness_polys);
        }
        let polynomial_refs = all_polynomials
            .iter()
            .flat_map(|ps| ps.iter())
            .collect::<Vec<_>>();

        // Create weights for constraints
        let mut weights = Vec::new();
        for _ in 0..num_points_per_poly {
            weights.push(Weights::evaluation(MultilinearPoint::rand(
                &mut rng,
                num_variables,
            )));
        }
        // Add a linear constraint
        let input = CoefficientList::new(
            (0..1 << num_variables)
                .map(<EF as Field>::BasePrimeField::from)
                .collect(),
        );
        weights.push(Weights::linear(input.into()));
        let weights_ref = weights.iter().collect::<Vec<_>>();

        // Create evaluations for each constraint and polynomial
        let evaluations = weights
            .iter()
            .flat_map(|weights| polynomial_refs.iter().map(|poly| weights.evaluate(poly)))
            .collect::<Vec<_>>();

        // Set up domain separator
        let ds = DomainSeparator::protocol(&params)
            .session(&format!("Test at {}:{}", file!(), line!()))
            .instance(&Empty);
        let mut prover_state = ProverState::new_std(&ds);

        // Commit using commit_batch (stacks batch_size polynomials per witness)
        let mut witnesses = Vec::new();
        for witness_polys in &all_polynomials {
            let poly_refs: Vec<_> = witness_polys.iter().collect::<Vec<_>>();
            let witness = params.commit(&mut prover_state, &poly_refs);
            witnesses.push(witness);
        }
        let witness_refs = witnesses.iter().collect::<Vec<_>>();

        // Batch prove all witnesses together
        let (_point, _evals) = params.prove(
            &mut prover_state,
            &polynomial_refs,
            &witness_refs,
            &weights_ref,
            &evaluations,
        );

        // Verify
        let proof = prover_state.proof();
        let mut verifier_state = VerifierState::new_std(&ds, &proof);

        let mut commitments = Vec::new();
        for _ in 0..num_witnesses {
            let commitment = params.receive_commitment(&mut verifier_state).unwrap();
            commitments.push(commitment);
        }
        let commitment_refs = commitments.iter().collect::<Vec<_>>();

        let verify_result = params.verify(
            &mut verifier_state,
            &commitment_refs,
            &weights_ref,
            &evaluations,
        );
        assert!(
            verify_result.is_ok(),
            "Batch verification with batch_size={} failed: {:?}",
            batch_size,
            verify_result.err()
        );
    }

    #[test]
    fn test_whir_batch_with_batch_size_2() {
        // This is the key regression test for the batch_size > 1 bug
        let batch_sizes = [2, 3];
        let num_witnesses = [2, 3];
        let folding_factors = [2, 3];

        for batch_size in batch_sizes {
            for num_witness in num_witnesses {
                for folding_factor in folding_factors {
                    make_whir_batch_with_batch_size(
                        folding_factor * 2, // num_variables
                        FoldingFactor::Constant(folding_factor),
                        1, // num_points_per_poly
                        num_witness,
                        batch_size,
                        SoundnessType::ConjectureList,
                        0, // pow_bits
                    );
                }
            }
        }
    }
}
