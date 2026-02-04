use ark_ff::FftField;
use ark_std::rand::{CryptoRng, RngCore};
#[cfg(feature = "tracing")]
use tracing::instrument;

use super::{
    committer::Witness,
    config::WhirConfig,
    statement::{Statement, Weights},
};
use crate::{
    algebra::{
        embedding::Identity,
        lift, mixed_scalar_mul_add,
        poly_utils::{coeffs::CoefficientList, multilinear::MultilinearPoint},
        tensor_product,
    },
    hash::Hash,
    protocols::{geometric_challenge::geometric_challenge, irs_commit, sumcheck::SumcheckSingle},
    transcript::{
        codecs::U64, Codec, Decoding, DuplexSpongeInterface, ProverMessage, ProverState,
        VerifierMessage,
    },
    utils::zip_strict,
    whir::config::InitialSumcheck,
};

pub(crate) enum RoundWitness<'a, F: FftField> {
    Initial {
        witnesses: &'a [&'a irs_commit::Witness<F::BasePrimeField, F>],
        batching_weights: Vec<F>,
    },
    Round {
        witness: irs_commit::Witness<F, F>,
    },
}

impl<F: FftField> WhirConfig<F> {
    #[allow(clippy::too_many_lines)] // TODO
    #[cfg_attr(feature = "tracing", instrument(skip_all))]
    pub fn prove<H, R>(
        &self,
        prover_state: &mut ProverState<H, R>,
        polynomials: &[&CoefficientList<F::BasePrimeField>],
        witnesses: &[&Witness<F>],
        statements: &[&Statement<F>],
    ) -> (MultilinearPoint<F>, Vec<F>)
    where
        H: DuplexSpongeInterface,
        R: RngCore + CryptoRng,
        F: Codec<[H::U]>,
        [u8; 32]: Decoding<[H::U]>,
        U64: Codec<[H::U]>,
        u8: Decoding<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        // Input validation
        assert_eq!(polynomials.len(), statements.len());
        assert_eq!(
            polynomials.len(),
            witnesses.len() * self.initial_committer.num_polynomials
        );
        for (polynomial, statement) in zip_strict(polynomials, statements) {
            assert_eq!(polynomial.num_coeffs(), self.initial_size());
            assert_eq!(statement.num_variables(), self.initial_num_variables());
            // In debug mode, verify all statment.
            // TODO: Add a `mixed_verify` function that takes an embedding into account.
            debug_assert!(statement.verify(&polynomial.lift(self.embedding())));
        }
        if polynomials.is_empty() {
            return (MultilinearPoint::default(), Vec::new());
        }

        // Complete evaluations of EVERY polynomial at EVERY constraint point
        // TODO: Transpose and make the order (constraint, polynomial)
        // This creates an N×M matrix where N = #polynomials, M = #constraints
        // We commit this matrix to the script.
        let constraints = witnesses
            .iter()
            .flat_map(|w| {
                w.out_of_domain()
                    .weights(&Identity::new(), self.initial_num_variables())
            })
            .chain(
                statements
                    .iter()
                    .flat_map(|s| s.constraints.iter().map(|c| c.weights.clone())),
            )
            .collect::<Vec<_>>();
        let num_constraints = constraints.len();
        let mut constraint_evals_matrix = vec![None; polynomials.len() * num_constraints];
        // Fill in provided OODs Points
        let mut constraint_offset = 0;
        let mut polynomial_offset = 0;
        for witness in witnesses {
            for row in witness.out_of_domain().rows() {
                let mut index = polynomial_offset + constraint_offset;
                for value in row {
                    assert!(constraint_evals_matrix[index].is_none());
                    constraint_evals_matrix[index] = Some(*value);
                    index += num_constraints;
                }
                constraint_offset += 1;
            }
            polynomial_offset += witness.out_of_domain().num_columns() * num_constraints;
        }
        // Fill in provided Statements
        let mut index = witnesses
            .iter()
            .map(|w| w.out_of_domain().num_points())
            .sum::<usize>();
        for statement in statements {
            for constraint in &statement.constraints {
                assert!(constraint_evals_matrix[index].is_none());
                constraint_evals_matrix[index] = Some(constraint.sum);
                index += 1; // To next column
            }
            index += num_constraints; // To next row, same column
        }
        // Complete the matrix by evaluating non-filled in terms.
        for (polynomial, row) in zip_strict(
            polynomials,
            constraint_evals_matrix.chunks_exact_mut(constraints.len()),
        ) {
            let mut lifted = None;
            for (weights, cell) in zip_strict(&constraints, row) {
                if cell.is_none() {
                    // TODO: Avoid lifting by evaluating directly through embedding.
                    let lifted = lifted.get_or_insert_with(|| polynomial.lift(self.embedding()));
                    let eval = weights.evaluate(lifted);
                    prover_state.prover_message(&eval);
                    *cell = Some(eval);
                }
            }
        }
        let constraint_evals_matrix: Vec<F> = constraint_evals_matrix
            .into_iter()
            .map(|e| e.unwrap())
            .collect();

        // Random linear combination of the polynomials
        let polynomial_weights: Vec<F> = geometric_challenge(prover_state, polynomials.len());
        assert_eq!(polynomial_weights[0], F::ONE);
        let mut coefficients = lift(self.embedding(), polynomials[0].coeffs());
        for (weight, polynomial) in zip_strict(&polynomial_weights, polynomials).skip(1) {
            mixed_scalar_mul_add(
                self.embedding(),
                &mut coefficients,
                *weight,
                polynomial.coeffs(),
            );
        }
        let mut coefficients = CoefficientList::new(coefficients);

        // Step 4: Build combined statement using RLC of the committed evaluation matrix
        // For each constraint j: combined_eval[j] = Σᵢ γⁱ·eval[i][j]
        let mut statement = Statement::new(self.initial_num_variables());
        for (constraint_idx, weights) in constraints.into_iter().enumerate() {
            let mut combined_eval = F::ZERO;
            for (weight, poly_evals) in zip_strict(
                polynomial_weights.iter(),
                constraint_evals_matrix.chunks_exact(num_constraints),
            ) {
                combined_eval += *weight * poly_evals[constraint_idx];
            }
            statement.add_constraint(weights, combined_eval);
        }

        // Run initial sumcheck on batched polynomial with combined statement
        let mut sumcheck_instance = None;
        let mut folding_randomness = match &self.initial_sumcheck {
            InitialSumcheck::Full(config) => {
                let combination_randomness_gen = prover_state.verifier_message();
                let mut instance = SumcheckSingle::new(
                    coefficients.clone(),
                    &statement,
                    combination_randomness_gen,
                );
                let folding_randomness = config.prove(prover_state, &mut instance);
                sumcheck_instance = Some(instance);
                folding_randomness
            }
            InitialSumcheck::Abridged { folding_size, pow } => {
                let mut folding_randomness = vec![F::ZERO; *folding_size];
                for randomness in &mut folding_randomness {
                    *randomness = prover_state.verifier_message();
                }
                pow.prove(prover_state);
                MultilinearPoint(folding_randomness)
            }
        };

        let mut randomness_vec = Vec::with_capacity(self.mv_parameters.num_variables);
        randomness_vec.extend(folding_randomness.0.iter().rev().copied());

        let mut prev_commitment = RoundWitness::Initial {
            witnesses,
            batching_weights: polynomial_weights,
        };

        // Execute standard WHIR rounds on the batched polynomial
        for (round_index, round_config) in self.round_configs.iter().enumerate() {
            coefficients = coefficients.fold(&folding_randomness);

            let witness = round_config
                .irs_committer
                .commit(prover_state, &[coefficients.coeffs()]);
            let ood_points = witness.out_of_domain().points.clone();
            let ood_answers = witness.out_of_domain().matrix.clone();

            // PoW
            round_config.pow.prove(prover_state);

            // Open the previous round's commitment, producing in-domain evaluations.
            // We prepend these with the OOD points and answers to produce a new
            // set of constraints for the next round.
            let (stir_challenges, stir_evaluations) = match &prev_commitment {
                RoundWitness::Initial {
                    witnesses,
                    batching_weights,
                } => {
                    let in_domain = self.initial_committer.open(prover_state, witnesses);

                    // Convert oods_points
                    let mut points = ood_points
                        .iter()
                        .copied()
                        .map(|a| {
                            MultilinearPoint::expand_from_univariate(
                                a,
                                round_config.initial_num_variables(),
                            )
                        })
                        .collect::<Vec<_>>();
                    let mut evals = ood_answers;

                    // Convert RS evaluation point to multivariate over extension
                    let weights =
                        tensor_product(batching_weights, &folding_randomness.coeff_weights(true));
                    points.extend(
                        in_domain.points(self.embedding(), round_config.initial_num_variables()),
                    );
                    evals.extend(in_domain.values(self.embedding(), &weights));

                    (points, evals)
                }
                RoundWitness::Round { witness } => {
                    let prev_round_config = &self.round_configs[round_index - 1];

                    let in_domain = prev_round_config
                        .irs_committer
                        .open(prover_state, &[&witness]);

                    // Convert oods_points
                    let mut points = ood_points
                        .iter()
                        .copied()
                        .map(|a| {
                            MultilinearPoint::expand_from_univariate(
                                a,
                                round_config.initial_num_variables(),
                            )
                        })
                        .collect::<Vec<_>>();
                    let mut evals = ood_answers;

                    // Convert RS evaluation point to multivariate over extension
                    let weights = folding_randomness.coeff_weights(true);
                    points.extend(
                        in_domain.points(&Identity::new(), round_config.initial_num_variables()),
                    );
                    evals.extend(in_domain.values(&Identity::new(), &weights));

                    (points, evals)
                }
            };

            // Randomness for combination
            let combination_randomness = geometric_challenge(prover_state, stir_challenges.len());

            #[allow(clippy::map_unwrap_or)]
            let mut round_sumcheck_instance = sumcheck_instance
                .take()
                .map(|mut sumcheck_instance| {
                    sumcheck_instance.add_new_equality(
                        &stir_challenges,
                        &stir_evaluations,
                        &combination_randomness,
                    );
                    sumcheck_instance
                })
                .unwrap_or_else(|| {
                    let mut statement = Statement::new(coefficients.num_variables());

                    for (point, eval) in zip_strict(stir_challenges.into_iter(), stir_evaluations) {
                        let weights = Weights::evaluation(point);
                        statement.add_constraint(weights, eval);
                    }
                    SumcheckSingle::new(coefficients.clone(), &statement, combination_randomness[1])
                });

            folding_randomness = round_config
                .sumcheck
                .prove(prover_state, &mut round_sumcheck_instance);

            randomness_vec.extend(folding_randomness.0.iter().rev());

            // Update round state
            sumcheck_instance = Some(round_sumcheck_instance);
            prev_commitment = RoundWitness::Round { witness };
        }

        coefficients = coefficients.fold(&folding_randomness);
        assert_eq!(coefficients.num_coeffs(), self.final_sumcheck.initial_size);

        // Directly send coefficients of the polynomial to the verifier.
        for coeff in coefficients.coeffs() {
            prover_state.prover_message(coeff);
        }

        // PoW
        self.final_pow.prove(prover_state);

        match &prev_commitment {
            RoundWitness::Initial { witnesses, .. } => {
                for witness in *witnesses {
                    let in_domain = self.initial_committer.open(prover_state, &[witness]);

                    // The verifier will directly test these on the final polynomial.
                    // It has the final polynomial in full, so it needs no furhter help
                    // from us.
                    drop(in_domain);
                }
            }
            RoundWitness::Round { witness } => {
                let prev_config = self.round_configs.last().unwrap();
                let in_domain = prev_config.irs_committer.open(prover_state, &[&witness]);

                // The verifier will directly test these on the final polynomial.
                // It has the final polynomial in full, so it needs no furhter help
                // from us.
                drop(in_domain);
            }
        }

        // Final sumcheck
        let mut sumcheck_instance = sumcheck_instance
            .clone()
            .unwrap_or_else(|| SumcheckSingle::new(coefficients.clone(), &statement, F::ONE));

        let final_folding_randomness = self
            .final_sumcheck
            .prove(prover_state, &mut sumcheck_instance);
        randomness_vec.extend(final_folding_randomness.0.iter().rev());

        // Hints for deferred constraints
        let constraint_eval = MultilinearPoint(randomness_vec.iter().copied().rev().collect());
        let deferred = statement
            .constraints
            .iter()
            .filter(|constraint| constraint.defer_evaluation)
            .map(|constraint| constraint.weights.compute(&constraint_eval))
            .collect();

        prover_state.prover_hint_ark(&deferred);

        (constraint_eval, deferred)
    }
}
