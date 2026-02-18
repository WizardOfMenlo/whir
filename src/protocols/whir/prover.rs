use ark_ff::FftField;
use ark_std::rand::{CryptoRng, RngCore};
#[cfg(feature = "tracing")]
use tracing::instrument;

use super::{committer::Witness, config::Config};
use crate::{
    algebra::{
        dot,
        linear_form::{Evaluate, LinearForm, UnivariateEvaluation},
        mixed_scalar_mul_add,
        sumcheck::fold,
        tensor_product, MultilinearPoint,
    },
    hash::Hash,
    protocols::{geometric_challenge::geometric_challenge, irs_commit},
    transcript::{
        codecs::U64, Codec, Decoding, DuplexSpongeInterface, ProverMessage, ProverState,
        VerifierMessage,
    },
    utils::zip_strict,
};

/// Owns the witness for the current round so it can be freed after opening.
enum RoundWitness<F: FftField> {
    Initial(Vec<irs_commit::Witness<F::BasePrimeField, F>>),
    Round(irs_commit::Witness<F, F>),
}

impl<F: FftField> Config<F> {
    /// Prove a WHIR opening (ownership-based for memory efficiency).
    ///
    /// All large inputs are consumed and freed as soon as they are no longer
    /// needed, significantly reducing peak memory during proving.
    ///
    /// * `prover_state` the mutable transcript to write the proof to.
    /// * `vectors` all the vectors we are opening (consumed after RLC).
    /// * `witnesses` witnesses corresponding to the `vectors` (consumed
    ///   after opening). Multiple vectors may share the same witness, in
    ///   which case only one witness should be provided.
    /// * `linear_forms` the covectors (if any) to evaluate each vector at
    ///   (consumed; deferred forms kept until the end).
    /// * `evaluations` a matrix of each vector evaluated at each linear form
    ///   (consumed after computing the initial sum).
    ///
    /// The `evaluations` matrix is in row-major order with the number of rows
    /// equal to the `linear_forms.len()` and the number of columns equal to
    /// `vectors.len()`.
    ///
    #[cfg_attr(feature = "tracing", instrument(skip_all))]
    #[allow(clippy::too_many_lines)]
    pub fn prove<H, R>(
        &self,
        prover_state: &mut ProverState<H, R>,
        vectors: Vec<Vec<F::BasePrimeField>>,
        witnesses: Vec<Witness<F>>,
        linear_forms: Vec<Box<dyn LinearForm<F>>>,
        evaluations: Vec<F>,
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
        // Save count before any moves — used after vectors are dropped.
        let num_vectors = vectors.len();

        // Input validation
        assert_eq!(
            num_vectors,
            witnesses.len() * self.initial_committer.num_vectors
        );
        assert_eq!(evaluations.len(), num_vectors * linear_forms.len());
        for vector in &vectors {
            assert_eq!(vector.len(), self.initial_size());
        }
        for linear_form in &linear_forms {
            assert_eq!(linear_form.size(), self.initial_size());
        }
        #[cfg(debug_assertions)]
        for (linear_form, evaluations) in
            zip_strict(&linear_forms, evaluations.chunks_exact(num_vectors))
        {
            use crate::algebra::linear_form::Covector;
            let covector = Covector::from(&**linear_form);
            for (vector, evaluation) in zip_strict(&vectors, evaluations) {
                debug_assert_eq!(
                    covector.evaluate(self.embedding(), vector.as_slice()),
                    *evaluation
                );
            }
        }
        if vectors.is_empty() {
            // TODO: Should we draw a random evaluation point of the right size?
            return (MultilinearPoint::default(), Vec::new());
        }

        // Complete evaluations of EVERY vector at EVERY linear form.
        let (oods_evals, oods_matrix) = {
            let mut oods_evals = Vec::new();
            let mut oods_matrix = Vec::new();

            // Out of domain samples. Compute missing cross-terms and send to verifier.
            let mut vector_offset = 0;
            for witness in &witnesses {
                for (oods_eval, oods_row) in zip_strict(
                    witness.out_of_domain().evaluators(self.initial_size()),
                    witness.out_of_domain().rows(),
                ) {
                    for (j, vector) in vectors.iter().enumerate() {
                        if j >= vector_offset && j < oods_row.len() + vector_offset {
                            debug_assert_eq!(
                                oods_row[j - vector_offset],
                                oods_eval.evaluate(self.embedding(), vector.as_slice())
                            );

                            oods_matrix.push(oods_row[j - vector_offset]);
                        } else {
                            let eval = oods_eval.evaluate(self.embedding(), vector.as_slice());
                            prover_state.prover_message(&eval);
                            oods_matrix.push(eval);
                        }
                    }
                    oods_evals.push(oods_eval);
                }
                vector_offset += witness.num_vectors();
            }
            (oods_evals, oods_matrix)
        };

        // Random linear combination of the vectors.
        let mut vector_rlc_coeffs: Vec<F> = geometric_challenge(prover_state, num_vectors);
        let mut vector = vec![F::ZERO; self.initial_size()];
        assert_eq!(vector_rlc_coeffs[0], F::ONE);
        for (rlc_coeff, input_vector) in zip_strict(&vector_rlc_coeffs, &vectors) {
            mixed_scalar_mul_add(
                self.embedding(),
                &mut vector,
                *rlc_coeff,
                input_vector.as_slice(),
            );
        }

        // ── Input vectors are fully consumed; free them. ──
        drop(vectors);

        let mut prev_witness = RoundWitness::Initial(witnesses);

        // Random linear combination of the constraints.
        let constraint_rlc_coeffs: Vec<F> =
            geometric_challenge(prover_state, linear_forms.len() + oods_evals.len());
        // TODO: Flip order.
        let (oods_rlc_coeffs, intial_forms_rlc_coeffs) =
            constraint_rlc_coeffs.split_at(oods_evals.len());
        let mut covector = vec![F::ZERO; self.initial_size()];
        for (rlc_coeff, linear_form) in zip_strict(intial_forms_rlc_coeffs, &linear_forms) {
            linear_form.accumulate(&mut covector, *rlc_coeff);
        }

        // Compute "The Sum"
        let mut the_sum: F = zip_strict(
            intial_forms_rlc_coeffs,
            evaluations.chunks_exact(num_vectors),
        )
        .map(|(poly_coeff, row)| *poly_coeff * dot(&vector_rlc_coeffs, row))
        .sum();

        // ── Evaluations are fully consumed; free them. ──
        drop(evaluations);

        debug_assert_eq!(dot(&vector, &covector), the_sum);

        // Add OODS constraints
        UnivariateEvaluation::accumulate_many(&oods_evals, &mut covector, oods_rlc_coeffs);
        the_sum += zip_strict(oods_rlc_coeffs, oods_matrix.chunks_exact(num_vectors))
            .map(|(poly_coeff, row)| *poly_coeff * dot(&vector_rlc_coeffs, row))
            .sum::<F>();

        // ── OODS intermediates are fully consumed; free them. ──
        drop(oods_evals);
        drop(oods_matrix);

        // These invariants are maintained throughout the proof.
        debug_assert_eq!(dot(&vector, &covector), the_sum);

        // Run initial sumcheck on batched vectors with combined statement
        let mut folding_randomness = if constraint_rlc_coeffs.is_empty() {
            // There are no constraints yet, so we can skip the sumcheck.
            // (If we did run it, all sumcheck vectors would be constant zero)
            // TODO: Don't compute evaluations and constraints in the first place.
            let folding_randomness = (0..self.initial_sumcheck.num_rounds)
                .map(|_| prover_state.verifier_message())
                .collect();
            self.initial_sumcheck.round_pow.prove(prover_state);
            // Fold vector
            for &f in &folding_randomness {
                fold(&mut vector, f);
            }
            // Covector must be all zeros.
            covector = vec![F::ZERO; self.initial_sumcheck.final_size()];
            MultilinearPoint(folding_randomness)
        } else {
            self.initial_sumcheck
                .prove(prover_state, &mut vector, &mut covector, &mut the_sum)
        };

        // Reclaim memory from the folded-away upper halves.
        vector.shrink_to_fit();
        covector.shrink_to_fit();

        let mut randomness_vec = Vec::with_capacity(self.initial_num_variables());
        randomness_vec.extend(folding_randomness.0.iter().copied());
        debug_assert_eq!(dot(&vector, &covector), the_sum);

        // Execute standard WHIR rounds on the batched vectors
        for (round_index, round_config) in self.round_configs.iter().enumerate() {
            // Commit to the vector, this generates out-of-domain evaluations.
            let new_witness = round_config.irs_committer.commit(prover_state, &[&vector]);

            // Proof of work before in-domain challenges
            round_config.pow.prove(prover_state);

            // Open and consume the previous round's witness, freeing its memory.
            let in_domain = match prev_witness {
                RoundWitness::Initial(init_witnesses) => {
                    let witness_refs: Vec<&_> = init_witnesses.iter().collect();
                    let result = self
                        .initial_committer
                        .open(prover_state, &witness_refs)
                        .lift(self.embedding());
                    drop(init_witnesses);
                    result
                }
                RoundWitness::Round(old_witness) => {
                    let prev_round_config = &self.round_configs[round_index - 1];
                    prev_round_config
                        .irs_committer
                        .open(prover_state, &[&old_witness])
                }
            };

            // Collect constraints for this round and RLC them in
            let stir_challenges = new_witness
                .out_of_domain()
                .evaluators(round_config.initial_size())
                .chain(in_domain.evaluators(round_config.initial_size()))
                .collect::<Vec<_>>();
            let stir_evaluations = new_witness
                .out_of_domain()
                .values(&[F::ONE])
                .chain(in_domain.values(&tensor_product(
                    &vector_rlc_coeffs,
                    &folding_randomness.eq_weights(),
                )))
                .collect::<Vec<_>>();
            let stir_rlc_coeffs = geometric_challenge(prover_state, stir_challenges.len());
            UnivariateEvaluation::accumulate_many(
                &stir_challenges,
                &mut covector,
                &stir_rlc_coeffs,
            );
            the_sum += dot(&stir_rlc_coeffs, &stir_evaluations);
            debug_assert_eq!(dot(&vector, &covector), the_sum);

            // Run sumcheck for this round
            folding_randomness =
                round_config
                    .sumcheck
                    .prove(prover_state, &mut vector, &mut covector, &mut the_sum);

            // Reclaim memory from the folded-away upper halves.
            vector.shrink_to_fit();
            covector.shrink_to_fit();

            randomness_vec.extend(folding_randomness.0.iter().copied());
            debug_assert_eq!(dot(&vector, &covector), the_sum);

            prev_witness = RoundWitness::Round(new_witness);
            vector_rlc_coeffs = vec![F::ONE];
        }

        // Directly send the vector to the verifier.
        assert_eq!(vector.len(), self.final_sumcheck.initial_size);
        for coeff in &vector {
            prover_state.prover_message(coeff);
        }

        // PoW
        self.final_pow.prove(prover_state);

        // Open and consume the final previous witness.
        match prev_witness {
            RoundWitness::Initial(init_witnesses) => {
                let witness_refs: Vec<&_> = init_witnesses.iter().collect();
                let _in_domain = self.initial_committer.open(prover_state, &witness_refs);
            }
            RoundWitness::Round(old_witness) => {
                let prev_config = self.round_configs.last().unwrap();
                let _in_domain = prev_config
                    .irs_committer
                    .open(prover_state, &[&old_witness]);
            }
        }

        // Final sumcheck
        let final_folding_randomness =
            self.final_sumcheck
                .prove(prover_state, &mut vector, &mut covector, &mut the_sum);
        randomness_vec.extend(final_folding_randomness.0.iter().copied());

        // Hints for deferred constraints
        let constraint_eval = MultilinearPoint(randomness_vec);
        let deferred = linear_forms
            .iter()
            .filter(|w| w.deferred())
            .map(|w| w.mle_evaluate(&constraint_eval.0))
            .collect();
        prover_state.prover_hint_ark(&deferred);

        (constraint_eval, deferred)
    }
}
