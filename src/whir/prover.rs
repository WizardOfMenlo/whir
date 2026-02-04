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
        embedding::{self, Embedding, Identity},
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
        assert_eq!(polynomials.len(), statements.len());
        assert_eq!(
            polynomials.len(),
            witnesses.len() * self.initial_committer.num_polynomials
        );
        if polynomials.is_empty() {
            // TODO: Implement something sensible.
            unimplemented!("Empty case not implemented");
        }
        let num_variables = self.initial_committer.polynomial_size.trailing_zeros() as usize;

        // Validate statements
        for (polynomial, statement) in zip_strict(polynomials.iter(), statements) {
            assert_eq!(polynomial.num_variables(), num_variables);
            assert_eq!(statement.num_variables(), num_variables);

            #[cfg(debug_assertions)]
            {
                // In debug mode, verify all statment.
                // TODO: Add a `mixed_verify` function that takes an embedding into account.
                let polynomial = polynomial.lift(self.embedding());
                assert!(statement.verify(&polynomial));
            }
        }

        // Step 1: Commit to the full N×M constraint evaluation matrix BEFORE sampling γ.
        //
        // Security: The prover must commit to ALL cross-term evaluations (P_i(w_j) for all i,j)
        // before learning the batching randomness γ. This prevents the prover from adaptively
        // choosing P_i(w_j) values after seeing γ, which would allow breaking soundness by
        // constructing polynomials that satisfy P_batched = Σ γ^i·P_i at constraint points
        // but differ elsewhere.

        // Collect all constraint weights from OOD samples and statements
        let mut all_constraint_weights = Vec::new();
        all_constraint_weights.extend(witnesses.iter().flat_map(|w| {
            w.out_of_domain()
                .weights(&embedding::Identity::<F>::new(), num_variables)
        }));
        all_constraint_weights.extend(
            statements
                .iter()
                .flat_map(|s| s.constraints.iter().map(|c| c.weights.clone())),
        );
        let num_constraints = all_constraint_weights.len();

        // Complete evaluations of EVERY polynomial at EVERY constraint point
        // This creates an N×M matrix where N = #polynomials, M = #constraints
        // We commit this matrix to the script.
        let mut constraint_evals_matrix: Vec<Option<F>> =
            vec![None; polynomials.len() * num_constraints];
        // OODs Points
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
        // Statement
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
        // Completion
        for (polynomial, row) in zip_strict(
            polynomials,
            constraint_evals_matrix.chunks_exact_mut(all_constraint_weights.len()),
        ) {
            let mut lifted = None;
            for (weights, cell) in zip_strict(&all_constraint_weights, row) {
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

        // Step 2: Sample batching randomness γ (cryptographically bound to committed matrix)
        let batching_weights: Vec<F> = geometric_challenge(prover_state, polynomials.len());

        // Step 3: Materialize the batched polynomial P_batched = P₀ + γ·P₁ + γ²·P₂ + ...
        // This also lifts the polynomial to the extension field.
        assert_eq!(batching_weights[0], F::ONE);
        let mut batched_coeffs = polynomials
            .first()
            .unwrap()
            .coeffs()
            .iter()
            .map(|c| self.embedding().map(*c))
            .collect::<Vec<_>>();
        for (weight, polynomial) in zip_strict(&batching_weights, polynomials).skip(1) {
            for (acc, src) in zip_strict(batched_coeffs.iter_mut(), polynomial.coeffs()) {
                *acc += self.embedding().mixed_mul(*weight, *src);
            }
        }
        let mut coefficients = CoefficientList::new(batched_coeffs);

        // Step 4: Build combined statement using RLC of the committed evaluation matrix
        // For each constraint j: combined_eval[j] = Σᵢ γⁱ·eval[i][j]
        let mut statement = Statement::new(num_variables);

        for (constraint_idx, weights) in all_constraint_weights.into_iter().enumerate() {
            let mut combined_eval = F::ZERO;
            for (weight, poly_evals) in zip_strict(
                batching_weights.iter(),
                constraint_evals_matrix.chunks_exact(num_constraints),
            ) {
                combined_eval += *weight * poly_evals[constraint_idx];
            }
            statement.add_constraint(weights, combined_eval);
        }

        // Run initial sumcheck on batched polynomial with combined statement
        let mut sumcheck_prover = None;
        let mut folding_randomness = match &self.initial_sumcheck {
            InitialSumcheck::Full(config) => {
                let combination_randomness_gen = prover_state.verifier_message();
                let mut sumcheck = SumcheckSingle::new(
                    coefficients.clone(),
                    &statement,
                    combination_randomness_gen,
                );
                let folding_randomness = config.prove(prover_state, &mut sumcheck);
                sumcheck_prover = Some(sumcheck);
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
            batching_weights,
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
            let mut sumcheck_instance = sumcheck_prover
                .take()
                .map(|mut sumcheck_prover| {
                    sumcheck_prover.add_new_equality(
                        &stir_challenges,
                        &stir_evaluations,
                        &combination_randomness,
                    );
                    sumcheck_prover
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
                .prove(prover_state, &mut sumcheck_instance);

            randomness_vec.extend(folding_randomness.0.iter().rev());

            // Update round state
            sumcheck_prover = Some(sumcheck_instance);
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
        let mut final_folding_sumcheck = sumcheck_prover
            .clone()
            .unwrap_or_else(|| SumcheckSingle::new(coefficients.clone(), &statement, F::ONE));

        let final_folding_randomness = self
            .final_sumcheck
            .prove(prover_state, &mut final_folding_sumcheck);
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
