use ark_ff::FftField;
use ark_std::rand::{CryptoRng, RngCore};
#[cfg(feature = "tracing")]
use tracing::instrument;

use super::{committer::Witness, config::Config};
use crate::{
    algebra::{
        dot,
        embedding::Basefield,
        mixed_scalar_mul_add,
        polynomials::{CoefficientList, EvaluationsList, MultilinearPoint},
        tensor_product,
        weights::{Evaluate, UnivariateEvaluation},
    },
    hash::Hash,
    protocols::{geometric_challenge::geometric_challenge, irs_commit},
    transcript::{
        codecs::U64, Codec, Decoding, DuplexSpongeInterface, ProverMessage, ProverState,
        VerifierMessage,
    },
    utils::zip_strict,
};

enum RoundWitness<'a, F: FftField> {
    Initial(&'a [&'a irs_commit::Witness<F::BasePrimeField, F>]),
    Round(irs_commit::Witness<F, F>),
}

impl<F: FftField> Config<F> {
    /// Prove a WHIR opening.
    ///
    /// * `prover_state` the mutable transcript to write the proof to.
    /// * `polynomials` are all the polynomials (in coefficient) form, we are opening.
    /// * `witnesses` witnesses corresponding to the `polynomials`, in the same
    ///   order. Multiple polynomials may share the same witness, in which case
    ///   only one witness should be provided.
    /// * `weights` the weight vectors (if any) to evaluate each polynomial at.
    /// * `evaluations` a matrix of each polynomial evaluated at each weight.
    ///
    /// The `evaluations` matrix is in row-major order with the number of rows
    /// equal to the `weights.len()` and the number of columns equal to
    /// `polynomials.len()`.
    ///
    #[cfg_attr(feature = "tracing", instrument(skip_all))]
    #[allow(clippy::too_many_lines)]
    pub fn prove<H, R>(
        &self,
        prover_state: &mut ProverState<H, R>,
        polynomials: &[&CoefficientList<F::BasePrimeField>],
        witnesses: &[&Witness<F>],
        weights: &[&dyn Evaluate<Basefield<F>>],
        weight_evaluations: &[F],
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
        assert_eq!(
            polynomials.len(),
            witnesses.len() * self.initial_committer.num_polynomials
        );
        for polynomial in polynomials {
            assert_eq!(polynomial.num_coeffs(), self.initial_size());
        }
        for (weight, evaluations) in
            zip_strict(weights, weight_evaluations.chunks_exact(polynomials.len()))
        {
            assert_eq!(weight.size(), self.initial_size());
            for (&polynomial, evaluation) in zip_strict(polynomials, evaluations) {
                debug_assert_eq!(
                    weight.evaluate(self.embedding(), polynomial.coeffs()),
                    *evaluation
                );
            }
        }
        if polynomials.is_empty() {
            // TODO: Should we draw a random evaluation point of the right size?
            return (MultilinearPoint::default(), Vec::new());
        }

        // Complete evaluations of EVERY polynomial at EVERY OOD/Statement constraint.
        let (oods_weights, oods_matrix) = {
            let mut oods_weights = Vec::new();
            let mut oods_matrix = Vec::new();

            // Out of domain samples. Compute missing cross-terms and send to verifier.
            let mut polynomial_offset = 0;
            for witness in witnesses {
                for (weights, oods_row) in zip_strict(
                    witness.out_of_domain().weights(self.initial_size()),
                    witness.out_of_domain().rows(),
                ) {
                    for (j, &polynomial) in polynomials.iter().enumerate() {
                        if j >= polynomial_offset && j < oods_row.len() + polynomial_offset {
                            debug_assert_eq!(
                                oods_row[j - polynomial_offset],
                                weights.evaluate(self.embedding(), polynomial.coeffs())
                            );

                            oods_matrix.push(oods_row[j - polynomial_offset]);
                        } else {
                            let eval = weights.evaluate(self.embedding(), polynomial.coeffs());
                            prover_state.prover_message(&eval);
                            oods_matrix.push(eval);
                        }
                    }
                    oods_weights.push(weights);
                }
                polynomial_offset += witness.num_polynomials();
            }
            (oods_weights, oods_matrix)
        };

        // Random linear combination of the polynomials.
        let mut polynomial_rlc_coeffs: Vec<F> =
            geometric_challenge(prover_state, polynomials.len());
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
        let mut prev_witness = RoundWitness::Initial(witnesses);

        // Random linear combination of the constraints.
        let constraint_rlc_coeffs: Vec<F> =
            geometric_challenge(prover_state, weights.len() + oods_weights.len());
        // TODO: Flip order.
        let (oods_rlc_coeffs, intial_weights_rlc_coeffs) =
            constraint_rlc_coeffs.split_at(oods_weights.len());
        let mut constraints = EvaluationsList::new(vec![F::ZERO; self.initial_size()]);
        for (rlc_coeff, weights) in zip_strict(intial_weights_rlc_coeffs, weights) {
            weights.accumulate(constraints.evals_mut(), *rlc_coeff);
        }

        // Compute "The Sum"
        let mut the_sum = zip_strict(
            intial_weights_rlc_coeffs,
            weight_evaluations.chunks_exact(polynomials.len()),
        )
        .map(|(poly_coeff, row)| *poly_coeff * dot(&polynomial_rlc_coeffs, row))
        .sum();

        debug_assert_eq!(evaluations, EvaluationsList::from(coefficients.clone()));
        debug_assert_eq!(dot(evaluations.evals(), constraints.evals()), the_sum);

        // Add OODS constraints
        UnivariateEvaluation::accumulate_many(
            &oods_weights,
            constraints.evals_mut(),
            oods_rlc_coeffs,
        );
        the_sum += zip_strict(oods_rlc_coeffs, oods_matrix.chunks_exact(polynomials.len()))
            .map(|(poly_coeff, row)| *poly_coeff * dot(&polynomial_rlc_coeffs, row))
            .sum::<F>();

        // These invariants are maintained throughout the proof.
        debug_assert_eq!(evaluations, EvaluationsList::from(coefficients.clone()));
        debug_assert_eq!(dot(evaluations.evals(), constraints.evals()), the_sum);

        // Run initial sumcheck on batched polynomial with combined statement
        let mut folding_randomness = if constraint_rlc_coeffs.is_empty() {
            // There are no constraints yet, so we can skip the sumcheck.
            // (If we did run it, all sumcheck polynomials would be constant zero)
            // TODO: Don't compute evaluations and constraints in the first place.
            let folding_randomness = (0..self.initial_sumcheck.num_rounds)
                .map(|_| prover_state.verifier_message())
                .collect();
            self.initial_sumcheck.round_pow.prove(prover_state);
            constraints = EvaluationsList::new(vec![F::ZERO; self.initial_sumcheck.final_size()]);
            MultilinearPoint(folding_randomness)
        } else {
            self.initial_sumcheck.prove(
                prover_state,
                &mut evaluations,
                &mut constraints,
                &mut the_sum,
            )
        };
        coefficients = coefficients.fold(&folding_randomness);
        if constraint_rlc_coeffs.is_empty() {
            // We didn't fold evaluations, so compute it here.
            evaluations = EvaluationsList::from(coefficients.clone());
        }
        let mut randomness_vec = Vec::with_capacity(self.initial_num_variables());
        randomness_vec.extend(folding_randomness.0.iter().rev().copied());
        debug_assert_eq!(evaluations, EvaluationsList::from(coefficients.clone()));
        debug_assert_eq!(dot(evaluations.evals(), constraints.evals()), the_sum);

        // Execute standard WHIR rounds on the batched polynomial
        for (round_index, round_config) in self.round_configs.iter().enumerate() {
            // Commit to the polynomial, this generates out-of-domain evaluations.
            let witness = round_config
                .irs_committer
                .commit(prover_state, &[coefficients.coeffs()]);

            // Proof of work before in-domain challenges
            round_config.pow.prove(prover_state);

            // Open the previous round's commitment, producing in-domain evaluations.
            let in_domain = match &prev_witness {
                RoundWitness::Initial(witnesses) => self
                    .initial_committer
                    .open(prover_state, witnesses)
                    .lift(self.embedding()),
                RoundWitness::Round(witness) => {
                    let prev_round_config = &self.round_configs[round_index - 1];
                    prev_round_config
                        .irs_committer
                        .open(prover_state, &[witness])
                }
            };

            // Collect constraints for this round and RLC them in
            let stir_challenges = witness
                .out_of_domain()
                .weights(round_config.initial_size())
                .chain(in_domain.weights(round_config.initial_size()))
                .collect::<Vec<_>>();
            let stir_evaluations = witness
                .out_of_domain()
                .values(&[F::ONE])
                .chain(in_domain.values(&tensor_product(
                    &polynomial_rlc_coeffs,
                    &folding_randomness.coeff_weights(true),
                )))
                .collect::<Vec<_>>();
            let stir_rlc_coeffs = geometric_challenge(prover_state, stir_challenges.len());
            UnivariateEvaluation::accumulate_many(
                &stir_challenges,
                constraints.evals_mut(),
                &stir_rlc_coeffs,
            );
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

            prev_witness = RoundWitness::Round(witness);
            polynomial_rlc_coeffs = vec![F::ONE];
        }

        // Directly send coefficients of the polynomial to the verifier.
        assert_eq!(coefficients.num_coeffs(), self.final_sumcheck.initial_size);
        for coeff in coefficients.coeffs() {
            prover_state.prover_message(coeff);
        }

        // PoW
        self.final_pow.prove(prover_state);

        // Open previous witness, but ignore the in-domain samples.
        // The verifier will directly test these on the final polynomial without our help.
        match &prev_witness {
            RoundWitness::Initial(witnesses) => {
                let _in_domain = self.initial_committer.open(prover_state, witnesses);
            }
            RoundWitness::Round(witness) => {
                let prev_config = self.round_configs.last().unwrap();
                let _in_domain = prev_config.irs_committer.open(prover_state, &[witness]);
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
        let deferred = weights
            .iter()
            .filter(|w| w.deferred())
            .map(|w| w.mle_evaluate(&constraint_eval.0))
            .collect();
        prover_state.prover_hint_ark(&deferred);

        (constraint_eval, deferred)
    }
}
