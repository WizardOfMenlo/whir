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
        dot,
        embedding::Identity,
        mixed_scalar_mul_add,
        poly_utils::{
            coeffs::CoefficientList, evals::EvaluationsList, multilinear::MultilinearPoint,
        },
        tensor_product,
    },
    hash::Hash,
    protocols::{geometric_challenge::geometric_challenge, irs_commit},
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
            // TODO: Should we draw a random evaluation point?
            return (MultilinearPoint::default(), Vec::new());
        }

        // Complete evaluations of EVERY polynomial at EVERY OOD/Statment constraint.
        // We take the caller provided values. The rest are computed and provided to
        // the verifier.
        let (constraint_weights, constraint_matrix) = {
            let mut weigths = Vec::new();
            let mut matrix = Vec::new();
            let mut polynomial_offset = 0;

            // Out of domain samples
            for witness in witnesses {
                for (weights, oods_row) in zip_strict(
                    witness
                        .out_of_domain()
                        .weights(&Identity::new(), self.initial_num_variables()),
                    witness.out_of_domain().rows(),
                ) {
                    for (j, polynomial) in polynomials.iter().enumerate() {
                        if j >= polynomial_offset && j < oods_row.len() + polynomial_offset {
                            matrix.push(oods_row[j - polynomial_offset]);
                        } else {
                            let eval = weights.mixed_evaluate(self.embedding(), polynomial);
                            prover_state.prover_message(&eval);
                            matrix.push(eval);
                        }
                    }
                    weigths.push(weights);
                }
                polynomial_offset += witness.num_polynomials();
            }

            // Constraints from statements over the input polynomials.
            for (i, statement) in statements.iter().enumerate() {
                for constraint in &statement.constraints {
                    weigths.push(constraint.weights.clone());
                    for (j, polynomial) in polynomials.iter().enumerate() {
                        if i == j {
                            matrix.push(constraint.sum);
                        } else {
                            let eval = constraint
                                .weights
                                .mixed_evaluate(self.embedding(), polynomial);
                            prover_state.prover_message(&eval);
                            matrix.push(eval);
                        }
                    }
                }
            }
            (weigths, matrix)
        };

        // Random linear combination of the polynomials.
        let polynomial_rlc_coeffs: Vec<F> = geometric_challenge(prover_state, polynomials.len());
        assert_eq!(polynomial_rlc_coeffs[0], F::ONE);
        let mut coefficients = polynomials[0].lift(self.embedding());
        for (rlc_coeff, polynomial) in zip_strict(&polynomial_rlc_coeffs, polynomials).skip(1) {
            mixed_scalar_mul_add(
                self.embedding(),
                coefficients.coeffs_mut(),
                *rlc_coeff,
                polynomial.coeffs(),
            );
        }
        let mut evaluations = EvaluationsList::from(coefficients.clone());
        let mut prev_commitment = RoundWitness::Initial {
            witnesses,
            batching_weights: polynomial_rlc_coeffs.clone(),
        };

        // Random linear combination of the constraints.
        let constraint_rlc_coeffs: Vec<F> =
            geometric_challenge(prover_state, constraint_weights.len());
        let mut constraints = EvaluationsList::new(vec![F::ZERO; self.initial_size()]);
        for (rlc_coeff, constraint) in zip_strict(&constraint_rlc_coeffs, constraint_weights) {
            constraint.accumulate(&mut constraints, *rlc_coeff)
        }

        // Compute "The Sum"
        let mut the_sum = zip_strict(
            &constraint_rlc_coeffs,
            constraint_matrix.chunks_exact(polynomials.len()),
        )
        .map(|(poly_coeff, row)| *poly_coeff * dot(&polynomial_rlc_coeffs, row))
        .sum();

        // These invariants are maintained throught the proof.
        debug_assert_eq!(evaluations, EvaluationsList::from(coefficients.clone()));
        debug_assert_eq!(dot(evaluations.evals(), constraints.evals()), the_sum);

        // Run initial sumcheck on batched polynomial with combined statement
        // TODO: Instead of `self.initial_sumcheck` we can branch on `constraint_weights.is_empty.`.
        let mut folding_randomness = match &self.initial_sumcheck {
            InitialSumcheck::Full(config) => config.prove(
                prover_state,
                &mut evaluations,
                &mut constraints,
                &mut the_sum,
            ),
            InitialSumcheck::Abridged { folding_size, pow } => {
                let folding_randomness = (0..*folding_size)
                    .map(|_| prover_state.verifier_message())
                    .collect();
                pow.prove(prover_state);
                constraints =
                    EvaluationsList::new(vec![F::ZERO; constraints.num_evals() >> folding_size]);
                MultilinearPoint(folding_randomness)
            }
        };
        coefficients = coefficients.fold(&folding_randomness);
        if !self.allows_statement() {
            evaluations = EvaluationsList::from(coefficients.clone());
        }
        let mut randomness_vec = Vec::with_capacity(self.mv_parameters.num_variables);
        randomness_vec.extend(folding_randomness.0.iter().rev().copied());
        debug_assert_eq!(evaluations, EvaluationsList::from(coefficients.clone()));
        debug_assert_eq!(dot(evaluations.evals(), constraints.evals()), the_sum);

        // Execute standard WHIR rounds on the batched polynomial
        for (round_index, round_config) in self.round_configs.iter().enumerate() {
            // Commit to the polynomial.
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
            prev_commitment = RoundWitness::Round { witness };

            // Add random linear combination of in- and out-domain constraints to the instance
            let stir_rlc_coeffs = geometric_challenge(prover_state, stir_challenges.len());
            for (coeff, point) in zip_strict(&stir_rlc_coeffs, stir_challenges) {
                Weights::evaluation(point).accumulate(&mut constraints, *coeff);
            }
            the_sum += dot(&stir_rlc_coeffs, &stir_evaluations);
            debug_assert_eq!(evaluations, EvaluationsList::from(coefficients.clone()));
            debug_assert_eq!(dot(evaluations.evals(), constraints.evals()), the_sum);

            // Run sumcheck for this round
            folding_randomness = round_config.sumcheck.prove(
                prover_state,
                &mut evaluations,
                &mut constraints,
                &mut the_sum,
            );
            coefficients = coefficients.fold(&folding_randomness);
            randomness_vec.extend(folding_randomness.0.iter().rev());
            debug_assert_eq!(evaluations, EvaluationsList::from(coefficients.clone()));
            debug_assert_eq!(dot(evaluations.evals(), constraints.evals()), the_sum);
        }

        // Directly send coefficients of the polynomial to the verifier.
        assert_eq!(coefficients.num_coeffs(), self.final_sumcheck.initial_size);
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
        let final_folding_randomness = self.final_sumcheck.prove(
            prover_state,
            &mut evaluations,
            &mut constraints,
            &mut the_sum,
        );
        randomness_vec.extend(final_folding_randomness.0.iter().rev());

        // Hints for deferred constraints
        let constraint_eval = MultilinearPoint(randomness_vec.iter().copied().rev().collect());
        let deferred = statements
            .iter()
            .flat_map(|s| &s.constraints)
            .filter(|constraint| constraint.defer_evaluation)
            .map(|constraint| constraint.weights.compute(&constraint_eval))
            .collect();
        prover_state.prover_hint_ark(&deferred);

        (constraint_eval, deferred)
    }
}
