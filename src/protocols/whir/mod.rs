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
            linear_form::{Covector, Evaluate, LinearForm, MultilinearExtension},
            MultilinearPoint,
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

    /// Run a complete WHIR proof lifecycle: commit, prove, and verify.
    ///
    /// This function:
    /// - builds a multilinear polynomial with a specified number of variables,
    /// - constructs a statement with constraints based on evaluations and linear relations,
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

        // Our test vector is all ones in the basefield.
        let vector = vec![F::ONE; num_coeffs];

        // Generate `num_points` random points in the multilinear domain
        let points: Vec<_> = (0..num_points)
            .map(|_| MultilinearPoint::rand(&mut rng, num_variables))
            .collect();

        let mut linear_forms: Vec<Box<dyn LinearForm<EF>>> = Vec::new();
        let mut evaluations = Vec::new();

        for point in &points {
            let linear_form = MultilinearExtension {
                point: point.0.clone(),
            };
            evaluations.push(linear_form.evaluate(params.embedding(), &vector));
            linear_forms.push(Box::new(linear_form));
        }

        let covector = Covector {
            deferred: false,
            vector: (0..1 << num_variables).map(EF::from).collect(),
        };
        let sum = covector.evaluate(params.embedding(), &vector);
        linear_forms.push(Box::new(covector));
        evaluations.push(sum);

        let ds = DomainSeparator::protocol(&params)
            .session(&format!("Test at {}:{}", file!(), line!()))
            .instance(&Empty);

        let mut prover_state = ProverState::new_std(&ds);
        let witness = params.commit(&mut prover_state, &[&vector]);

        // Build a second set of owned linear forms for prove (which consumes them).
        let mut prove_linear_forms: Vec<Box<dyn LinearForm<EF>>> = Vec::new();
        for point in &points {
            prove_linear_forms.push(Box::new(MultilinearExtension {
                point: point.0.clone(),
            }));
        }
        prove_linear_forms.push(Box::new(Covector {
            deferred: false,
            vector: (0..1 << num_variables).map(EF::from).collect(),
        }));

        params.prove(
            &mut prover_state,
            vec![vector],
            vec![witness],
            prove_linear_forms,
            evaluations.clone(),
        );

        let proof = prover_state.proof();
        let mut verifier_state = VerifierState::new_std(&ds, &proof);
        let commitment = params.receive_commitment(&mut verifier_state).unwrap();

        let linear_form_refs = linear_forms
            .iter()
            .map(|l| l.as_ref() as &dyn LinearForm<EF>)
            .collect::<Vec<_>>();
        params
            .verify(
                &mut verifier_state,
                &[&commitment],
                &linear_form_refs,
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
        num_vectors: usize,
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

        // Create N different vectors
        let vectors: Vec<_> = (0..num_vectors)
            .map(|i| {
                // Different vectors: first is all 1s, second is all 2s, etc.
                vec![F::from((i + 1) as u64); num_coeffs]
            })
            .collect();
        let vec_refs = vectors.iter().map(|v| v.as_slice()).collect::<Vec<_>>();

        let points: Vec<_> = (0..num_points_per_poly)
            .map(|_| MultilinearPoint::rand(&mut rng, num_variables))
            .collect();

        let mut linear_forms: Vec<Box<dyn Evaluate<Basefield<EF>>>> = Vec::new();
        for point in &points {
            linear_forms.push(Box::new(MultilinearExtension {
                point: point.0.clone(),
            }));
        }
        linear_forms.push(Box::new(Covector {
            deferred: false,
            vector: ((0..1 << num_variables).map(EF::from).collect()),
        }));

        let evaluations = linear_forms
            .iter()
            .flat_map(|linear_form| {
                vec_refs
                    .iter()
                    .map(|vec| linear_form.evaluate(params.embedding(), vec))
            })
            .collect::<Vec<_>>();

        let ds = DomainSeparator::protocol(&params)
            .session(&format!("Test at {}:{}", file!(), line!()))
            .instance(&Empty);
        let mut prover_state = ProverState::new_std(&ds);

        let mut witnesses = Vec::new();
        for &vec in &vec_refs {
            let witness = params.commit(&mut prover_state, &[vec]);
            witnesses.push(witness);
        }

        // Build a second set of owned linear forms for prove (which consumes them).
        let mut prove_linear_forms: Vec<Box<dyn LinearForm<EF>>> = Vec::new();
        for point in &points {
            prove_linear_forms.push(Box::new(MultilinearExtension {
                point: point.0.clone(),
            }));
        }
        prove_linear_forms.push(Box::new(Covector {
            deferred: false,
            vector: (0..1 << num_variables).map(EF::from).collect(),
        }));

        let (_point, _evals) = params.prove(
            &mut prover_state,
            vectors.clone(),
            witnesses,
            prove_linear_forms,
            evaluations.clone(),
        );

        let proof = prover_state.proof();
        let mut verifier_state = VerifierState::new_std(&ds, &proof);

        let mut commitments = Vec::new();
        for _ in 0..num_vectors {
            let commitment = params.receive_commitment(&mut verifier_state).unwrap();
            commitments.push(commitment);
        }
        let commitment_refs = commitments.iter().collect::<Vec<_>>();

        let linear_form_refs = linear_forms
            .iter()
            .map(|l| l.as_ref() as &dyn LinearForm<EF>)
            .collect::<Vec<_>>();
        params
            .verify(
                &mut verifier_state,
                &commitment_refs,
                &linear_form_refs,
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
    #[cfg_attr(
        debug_assertions,
        ignore = "debug_assert in prover panics on intentionally invalid input"
    )]
    fn test_whir_batch_rejects_invalid_constraint() {
        // Setup parameters
        let num_variables = 4;
        let folding_factor = FoldingFactor::Constant(2);
        let num_polynomials = 2;
        let num_coeffs = 1 << num_variables;
        let mut rng = ark_std::test_rng();

        let mv_params = MultivariateParameters::<EF>::new(num_variables);
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

        let params = Config::<EF>::new(mv_params, &whir_params);

        let embedding = Basefield::<EF>::new();

        // Create test vectors
        let vec1 = vec![F::ONE; num_coeffs];
        let vec2 = vec![F::from(2u64); num_coeffs];
        let vec_wrong = vec![F::from(999u64); num_coeffs];

        let constraint_points: Vec<_> = (0..2)
            .map(|_| MultilinearPoint::rand(&mut rng, num_variables))
            .collect();

        let linear_forms: [Box<dyn Evaluate<Basefield<EF>>>; 2] = [
            Box::new(MultilinearExtension {
                point: constraint_points[0].0.clone(),
            }),
            Box::new(MultilinearExtension {
                point: constraint_points[1].0.clone(),
            }),
        ];
        let evaluations = linear_forms
            .iter()
            .flat_map(|weights| [&vec1, &vec_wrong].map(|v| weights.evaluate(&embedding, v)))
            .collect::<Vec<_>>();

        let ds = DomainSeparator::protocol(&params)
            .session(&format!("Test at {}:{}", file!(), line!()))
            .instance(&Empty);
        let mut prover_state = ProverState::new_std(&ds);

        let witness1 = params.commit(&mut prover_state, &[&vec1]);
        let witness2 = params.commit(&mut prover_state, &[&vec2]);

        let prove_linear_forms: Vec<Box<dyn LinearForm<EF>>> = vec![
            Box::new(MultilinearExtension {
                point: constraint_points[0].0.clone(),
            }),
            Box::new(MultilinearExtension {
                point: constraint_points[1].0.clone(),
            }),
        ];
        let (_evalpoint, _values) = params.prove(
            &mut prover_state,
            vec![vec1.clone(), vec_wrong],
            vec![witness1, witness2],
            prove_linear_forms,
            evaluations.clone(),
        );

        // Verification should fail because the cross-terms don't match the commitment
        let proof = prover_state.proof();
        let mut verifier_state = VerifierState::new_std(&ds, &proof);

        let mut commitments = Vec::new();
        for _ in 0..num_polynomials {
            let parsed_commitment = params.receive_commitment(&mut verifier_state).unwrap();
            commitments.push(parsed_commitment);
        }

        let linear_form_refs = linear_forms
            .iter()
            .map(|l| l.as_ref() as &dyn LinearForm<EF>)
            .collect::<Vec<_>>();
        let verify_result = params.verify(
            &mut verifier_state,
            &[&commitments[0], &commitments[1]],
            &linear_form_refs,
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

        let params = Config::<EF>::new(mv_params, &whir_params);

        // Create weights for constraints
        let embedding = Basefield::new();

        // Create polynomials for each witness
        // Each witness will contain batch_size polynomials committed together
        let all_vectors: Vec<Vec<F>> = (0..num_witnesses * batch_size)
            .map(|i| vec![F::from((i + 1) as u64); num_coeffs])
            .collect::<Vec<_>>();
        let vec_refs = all_vectors.iter().map(|p| p.as_slice()).collect::<Vec<_>>();

        let points: Vec<_> = (0..num_points_per_poly)
            .map(|_| MultilinearPoint::rand(&mut rng, num_variables))
            .collect();

        let mut linear_forms: Vec<Box<dyn Evaluate<Basefield<EF>>>> = Vec::new();
        for point in &points {
            linear_forms.push(Box::new(MultilinearExtension {
                point: point.0.clone(),
            }));
        }
        linear_forms.push(Box::new(Covector {
            deferred: false,
            vector: (0..1 << num_variables).map(EF::from).collect(),
        }));

        let evaluations = linear_forms
            .iter()
            .flat_map(|linear_form| {
                vec_refs
                    .iter()
                    .map(|vec| linear_form.evaluate(&embedding, vec))
            })
            .collect::<Vec<_>>();

        let ds = DomainSeparator::protocol(&params)
            .session(&format!("Test at {}:{}", file!(), line!()))
            .instance(&Empty);
        let mut prover_state = ProverState::new_std(&ds);

        let mut witnesses = Vec::new();
        for witness_polys in vec_refs.chunks(batch_size) {
            let witness = params.commit(&mut prover_state, witness_polys);
            witnesses.push(witness);
        }

        // Build a second set of owned linear forms for prove (which consumes them).
        let mut prove_linear_forms: Vec<Box<dyn LinearForm<EF>>> = Vec::new();
        for point in &points {
            prove_linear_forms.push(Box::new(MultilinearExtension {
                point: point.0.clone(),
            }));
        }
        prove_linear_forms.push(Box::new(Covector {
            deferred: false,
            vector: (0..1 << num_variables).map(EF::from).collect(),
        }));

        let (_point, _evals) = params.prove(
            &mut prover_state,
            all_vectors.clone(),
            witnesses,
            prove_linear_forms,
            evaluations.clone(),
        );

        let proof = prover_state.proof();
        let mut verifier_state = VerifierState::new_std(&ds, &proof);

        let mut commitments = Vec::new();
        for _ in 0..num_witnesses {
            let commitment = params.receive_commitment(&mut verifier_state).unwrap();
            commitments.push(commitment);
        }
        let commitment_refs = commitments.iter().collect::<Vec<_>>();

        let linear_form_refs = linear_forms
            .iter()
            .map(|l| l.as_ref() as &dyn LinearForm<EF>)
            .collect::<Vec<_>>();
        let verify_result = params.verify(
            &mut verifier_state,
            &commitment_refs,
            &linear_form_refs,
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
