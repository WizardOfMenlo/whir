use ark_ff::FftField;
use ark_std::rand::{CryptoRng, RngCore};
#[cfg(feature = "tracing")]
use tracing::instrument;

use ark_ff::Field;

use super::{committer::Witness, config::Config};
use crate::{
    algebra::{
        dot,
        embedding::Embedding,
        mixed_scalar_mul_add, ntt,
        polynomials::{CoefficientList, EvaluationsList, MultilinearPoint},
        tensor_product, Weights,
    },
    hash::Hash,
    protocols::{
        geometric_challenge::geometric_challenge,
        irs_commit,
        whir::zk::{HelperEvaluations, ZkWitness},
    },
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
        weights: &[&Weights<F>],
        evaluations: &[F],
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
            zip_strict(weights, evaluations.chunks_exact(polynomials.len()))
        {
            assert_eq!(weight.num_variables(), self.initial_num_variables());
            for (polynomial, evaluation) in zip_strict(polynomials, evaluations) {
                debug_assert_eq!(
                    weight.mixed_evaluate(self.embedding(), polynomial),
                    *evaluation
                );
            }
        }
        if polynomials.is_empty() {
            // TODO: Should we draw a random evaluation point?
            return (MultilinearPoint::default(), Vec::new());
        }

        // Complete evaluations of EVERY polynomial at EVERY OOD/Statement constraint.
        let (constraint_weights, constraint_matrix) = {
            let mut all_weights = Vec::new();
            let mut matrix = Vec::new();

            // Out of domain samples. Compute missing cross-terms and send to verifier.
            let mut polynomial_offset = 0;
            for witness in witnesses {
                for (weights, oods_row) in zip_strict(
                    witness
                        .out_of_domain()
                        .weights(self.initial_num_variables()),
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
                    all_weights.push(weights);
                }
                polynomial_offset += witness.num_polynomials();
            }

            // Add caller provided weights and evaluations.
            all_weights.extend(weights.iter().map(|&w| w.clone()));
            matrix.extend_from_slice(evaluations);
            (all_weights, matrix)
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
            geometric_challenge(prover_state, constraint_weights.len());
        let mut constraints = EvaluationsList::new(vec![F::ZERO; self.initial_size()]);
        for (rlc_coeff, constraint) in zip_strict(&constraint_rlc_coeffs, constraint_weights) {
            constraint.accumulate(&mut constraints, *rlc_coeff);
        }

        // Compute "The Sum"
        let mut the_sum = zip_strict(
            &constraint_rlc_coeffs,
            constraint_matrix.chunks_exact(polynomials.len()),
        )
        .map(|(poly_coeff, row)| *poly_coeff * dot(&polynomial_rlc_coeffs, row))
        .sum();

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
        coefficients.fold_in_place(&folding_randomness);
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
                .weights(round_config.initial_num_variables())
                .chain(in_domain.weights(round_config.initial_num_variables()))
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
            for (coeff, weights) in zip_strict(&stir_rlc_coeffs, stir_challenges) {
                weights.accumulate(&mut constraints, *coeff);
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
            coefficients.fold_in_place(&folding_randomness);
            randomness_vec.extend(folding_randomness.0.iter().rev());
            debug_assert_eq!(evaluations, EvaluationsList::from(coefficients.clone()));
            debug_assert_eq!(dot(evaluations.evals(), constraints.evals()), the_sum);

            prev_witness = RoundWitness::Round(witness);
            polynomial_rlc_coeffs = vec![F::ONE];
        }

        // Directly send coefficients of the polynomial to the verifier.
        self.send_final_coefficients(prover_state, &coefficients);

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
        self.compute_deferred_hints(prover_state, weights, &randomness_vec)
    }

    /// Prove a ZK WHIR opening.
    ///
    /// This proves knowledge of a polynomial `f` by:
    /// 1. Blinding with g to form P = ρ·f + g
    /// 2. Running WHIR rounds on P with a virtual oracle L = ρ·f̂ + h
    /// 3. Proving helper polynomial evaluations so verifier can reconstruct L
    pub fn prove_zk<H, R>(
        &self,
        prover_state: &mut ProverState<H, R>,
        polynomials: &[&CoefficientList<F::BasePrimeField>],
        witness: &ZkWitness<F>,
        helper_config: &Config<F>,
        weights: &[&Weights<F>],
        evaluations: &[F],
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
        let mu = witness.preprocessings[0].params.mu;
        let num_polys = polynomials.len();

        // Phase 1: ZK blinding setup — build g, evaluate at constraints, form P = ρ·f + g
        let beta: F = prover_state.verifier_message();
        let mut g_polys = Vec::with_capacity(num_polys);
        for (_polynomial, preprocessing) in zip_strict(polynomials, &witness.preprocessings) {
            let g_poly = self.build_blinding_polynomial(preprocessing, mu, beta);
            g_polys.push(g_poly);
        }

        // Evaluate each gⱼ at each weight and send to verifier.
        // Layout: row-major [weight₀_poly₀, weight₀_poly₁, ..., weight₁_poly₀, ...]
        // This matches the evaluations matrix layout.
        let mut g_eval_matrix = vec![F::ZERO; weights.len() * num_polys];
        for (i, weight) in weights.iter().enumerate() {
            for (j, g_poly) in g_polys.iter().enumerate() {
                let eval = weight.evaluate(g_poly);
                prover_state.prover_message(&eval);
                g_eval_matrix[i * num_polys + j] = eval;
            }
        }

        let rho: F = prover_state.verifier_message();

        // Build Pᵢ = ρ·fᵢ + gᵢ for each polynomial
        let mut p_polys = Vec::with_capacity(num_polys);
        for (polynomial, g_poly) in zip_strict(polynomials, g_polys) {
            let p_poly = self.build_blinded_polynomial_p(g_poly, polynomial, rho);
            p_polys.push(p_poly);
        }

        // RLC the polynomials: P₀ = Σ αᵢ · Pᵢ
        let polynomial_rlc_coeffs: Vec<F> = geometric_challenge(prover_state, num_polys);
        let mut p_poly = {
            let mut iter = p_polys.into_iter();
            let mut acc = iter.next().unwrap();
            for (rlc_coeff, src_poly) in zip_strict(&polynomial_rlc_coeffs[1..], iter) {
                for (c, &src) in acc.coeffs_mut().iter_mut().zip(src_poly.coeffs()) {
                    *c += *rlc_coeff * src;
                }
            }
            acc
        };

        // Phase 2: Build modified evaluations and run initial sumcheck
        // modified_evaluations[w * N + p] = ρ · evaluations[w * N + p] + g_eval_matrix[w * N + p]
        let modified_evaluations: Vec<F> = evaluations
            .iter()
            .zip(g_eval_matrix.iter())
            .map(|(&eval, &g_eval)| rho * eval + g_eval)
            .collect();
        let constraint_rlc_coeffs: Vec<F> = geometric_challenge(prover_state, weights.len());
        let mut constraints = EvaluationsList::new(vec![F::ZERO; self.initial_size()]);
        for (rlc_coeff, weight) in zip_strict(&constraint_rlc_coeffs, weights) {
            weight.accumulate(&mut constraints, *rlc_coeff);
        }

        // Compute "The Sum": Σ_w rlc_w * dot(poly_rlc, modified_evaluations[w*N..(w+1)*N])
        let mut the_sum: F = zip_strict(
            &constraint_rlc_coeffs,
            modified_evaluations.chunks_exact(num_polys),
        )
        .map(|(weight_coeff, row)| *weight_coeff * dot(&polynomial_rlc_coeffs, row))
        .sum();

        let mut eval_list = EvaluationsList::from(p_poly.clone());
        let mut folding_randomness = if constraint_rlc_coeffs.is_empty() {
            let fr = (0..self.initial_sumcheck.num_rounds)
                .map(|_| prover_state.verifier_message())
                .collect();
            self.initial_sumcheck.round_pow.prove(prover_state);
            constraints = EvaluationsList::new(vec![F::ZERO; self.initial_sumcheck.final_size()]);
            MultilinearPoint(fr)
        } else {
            self.initial_sumcheck.prove(
                prover_state,
                &mut eval_list,
                &mut constraints,
                &mut the_sum,
            )
        };

        p_poly.fold_in_place(&folding_randomness);
        let mut coefficients = p_poly;
        if constraint_rlc_coeffs.is_empty() {
            eval_list = EvaluationsList::from(coefficients.clone());
        }
        let mut randomness_vec = Vec::with_capacity(mu);
        randomness_vec.extend(folding_randomness.0.iter().rev().copied());
        debug_assert_eq!(eval_list, EvaluationsList::from(coefficients.clone()));
        debug_assert_eq!(dot(eval_list.evals(), constraints.evals()), the_sum);

        // Phase 3: WHIR round loop
        let mut prev_is_initial = true;
        let mut prev_round_witness: Option<irs_commit::Witness<F, F>> = None;

        for (round_index, round_config) in self.round_configs.iter().enumerate() {
            let round_witness = round_config
                .irs_committer
                .commit(prover_state, &[coefficients.coeffs()]);
            round_config.pow.prove(prover_state);

            let num_variables = round_config.initial_num_variables();

            let (in_domain, stir_evaluations) = if prev_is_initial {
                self.open_initial_zk_round(
                    prover_state,
                    witness,
                    helper_config,
                    rho,
                    &coefficients,
                    &round_witness,
                    num_variables,
                )
            } else {
                self.open_subsequent_round(
                    prover_state,
                    round_index,
                    prev_round_witness.as_ref().unwrap(),
                    &round_witness,
                    &folding_randomness,
                )
            };

            let stir_challenges: Vec<_> = round_witness
                .out_of_domain()
                .weights(num_variables)
                .chain(in_domain.weights(num_variables))
                .collect();

            let stir_rlc_coeffs = geometric_challenge(prover_state, stir_challenges.len());
            for (coeff, w) in zip_strict(&stir_rlc_coeffs, &stir_challenges) {
                w.accumulate(&mut constraints, *coeff);
            }
            the_sum += dot(&stir_rlc_coeffs, &stir_evaluations);
            debug_assert_eq!(eval_list, EvaluationsList::from(coefficients.clone()));
            debug_assert_eq!(dot(eval_list.evals(), constraints.evals()), the_sum);

            folding_randomness = round_config.sumcheck.prove(
                prover_state,
                &mut eval_list,
                &mut constraints,
                &mut the_sum,
            );
            coefficients.fold_in_place(&folding_randomness);
            randomness_vec.extend(folding_randomness.0.iter().rev());
            debug_assert_eq!(eval_list, EvaluationsList::from(coefficients.clone()));
            debug_assert_eq!(dot(eval_list.evals(), constraints.evals()), the_sum);

            prev_is_initial = false;
            prev_round_witness = Some(round_witness);
        }

        // Phase 4: Final round — send coefficients, PoW, open last commitment
        self.send_final_coefficients(prover_state, &coefficients);
        self.final_pow.prove(prover_state);

        if prev_is_initial {
            let f_hat_refs: Vec<_> = witness.f_hat_witnesses.iter().collect();
            let in_domain_base = self.initial_committer.open(prover_state, &f_hat_refs);
            self.prove_zk_helper_evaluations(
                prover_state,
                &in_domain_base,
                witness,
                helper_config,
                rho,
            );
        } else {
            let prev_config = self.round_configs.last().unwrap();
            let _in_domain = prev_config
                .irs_committer
                .open(prover_state, &[prev_round_witness.as_ref().unwrap()]);
        }

        // Phase 5: Final sumcheck and deferred constraint hints
        let final_folding_randomness =
            self.final_sumcheck
                .prove(prover_state, &mut eval_list, &mut constraints, &mut the_sum);
        randomness_vec.extend(final_folding_randomness.0.iter().rev());

        self.compute_deferred_hints(prover_state, weights, &randomness_vec)
    }

    /// Build the blinding polynomial g(X) = g₀(X) + Σᵢ₌₁^μ βⁱ · X^(2^(i-1)) · ĝᵢ(X)
    ///
    /// Returns the blinding polynomial g as a `CoefficientList<F>`.
    fn build_blinding_polynomial(
        &self,
        preprocessing: &super::zk::ZkPreprocessingPolynomials<F>,
        mu: usize,
        beta: F,
    ) -> CoefficientList<F> {
        let poly_size = 1 << mu;
        let mut coeffs = vec![F::ZERO; poly_size];
        let g0_coeffs = preprocessing.g0_hat.coeffs();
        coeffs[..g0_coeffs.len()].copy_from_slice(g0_coeffs);

        let mut beta_power = beta;
        for i in 1..=mu {
            let shift = 1 << (i - 1);
            let g_hat_coeffs = preprocessing.g_hats[i - 1].coeffs();
            let bp = beta_power;
            let target = &mut coeffs[shift..shift + g_hat_coeffs.len()];
            #[cfg(feature = "parallel")]
            {
                use rayon::prelude::*;
                if target.len() >= 1024 {
                    target
                        .par_iter_mut()
                        .zip(g_hat_coeffs.par_iter())
                        .for_each(|(c, &g_c)| *c += bp * g_c);
                } else {
                    for (c, &g_c) in target.iter_mut().zip(g_hat_coeffs) {
                        *c += bp * g_c;
                    }
                }
            }
            #[cfg(not(feature = "parallel"))]
            for (c, &g_c) in target.iter_mut().zip(g_hat_coeffs) {
                *c += bp * g_c;
            }
            beta_power *= beta;
        }

        let g_poly = CoefficientList::new(coeffs);

        g_poly
    }

    /// Transform g → P = ρ·f + g in-place: P(X) = ρ·embed(f(X)) + g(X).
    fn build_blinded_polynomial_p(
        &self,
        g_poly: CoefficientList<F>,
        polynomial: &CoefficientList<F::BasePrimeField>,
        rho: F,
    ) -> CoefficientList<F> {
        let mut coeffs = g_poly.into_coeffs();
        let embedding = self.embedding();
        let f_coeffs = polynomial.coeffs();
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            coeffs[..f_coeffs.len()]
                .par_iter_mut()
                .zip(f_coeffs.par_iter())
                .for_each(|(c, &f_coeff)| *c += rho * embedding.map(f_coeff));
        }
        #[cfg(not(feature = "parallel"))]
        for (c, &f_coeff) in coeffs.iter_mut().zip(f_coeffs.iter()) {
            *c += rho * embedding.map(f_coeff);
        }
        CoefficientList::new(coeffs)
    }

    /// Open the initial f̂ commitment in ZK mode: open f̂, prove helper evaluations,
    /// and compute virtual oracle folded values.
    ///
    /// Returns `(in_domain_evaluations, stir_evaluation_values)`.
    fn open_initial_zk_round<H, R>(
        &self,
        prover_state: &mut ProverState<H, R>,
        witness: &ZkWitness<F>,
        helper_config: &Config<F>,
        rho: F,
        coefficients: &CoefficientList<F>,
        round_witness: &irs_commit::Witness<F, F>,
        num_variables: usize,
    ) -> (irs_commit::Evaluations<F>, Vec<F>)
    where
        H: DuplexSpongeInterface,
        R: RngCore + CryptoRng,
        F: Codec<[H::U]>,
        [u8; 32]: Decoding<[H::U]>,
        U64: Codec<[H::U]>,
        u8: Decoding<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        let f_hat_refs: Vec<_> = witness.f_hat_witnesses.iter().collect();
        let in_domain_base = self.initial_committer.open(prover_state, &f_hat_refs);

        self.prove_zk_helper_evaluations(
            prover_state,
            &in_domain_base,
            witness,
            helper_config,
            rho,
        );

        let in_domain = in_domain_base.lift(self.embedding());

        // Virtual oracle values: evaluate folded P at each query point.
        // L and P agree on the evaluation domain, so fold_k(L, r̄)(α) = P_folded(α).
        let virtual_values: Vec<F> = in_domain
            .points
            .iter()
            .map(|&alpha| {
                let point = MultilinearPoint::expand_from_univariate(alpha, num_variables);
                coefficients.evaluate(&point)
            })
            .collect();

        let evals: Vec<F> = round_witness
            .out_of_domain()
            .values(&[F::ONE])
            .chain(virtual_values)
            .collect();

        (in_domain, evals)
    }

    /// Open a subsequent (non-initial) round's commitment.
    ///
    /// Returns `(in_domain_evaluations, stir_evaluation_values)`.
    fn open_subsequent_round<H, R>(
        &self,
        prover_state: &mut ProverState<H, R>,
        round_index: usize,
        prev_witness: &irs_commit::Witness<F, F>,
        round_witness: &irs_commit::Witness<F, F>,
        folding_randomness: &MultilinearPoint<F>,
    ) -> (irs_commit::Evaluations<F>, Vec<F>)
    where
        H: DuplexSpongeInterface,
        R: RngCore + CryptoRng,
        F: Codec<[H::U]>,
        [u8; 32]: Decoding<[H::U]>,
        U64: Codec<[H::U]>,
        u8: Decoding<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        let prev_round_config = &self.round_configs[round_index - 1];
        let in_domain = prev_round_config
            .irs_committer
            .open(prover_state, &[prev_witness]);

        let evals: Vec<F> = round_witness
            .out_of_domain()
            .values(&[F::ONE])
            .chain(in_domain.values(&folding_randomness.coeff_weights(true)))
            .collect();

        (in_domain, evals)
    }

    /// Send final polynomial coefficients to verifier.
    fn send_final_coefficients<H, R>(
        &self,
        prover_state: &mut ProverState<H, R>,
        coefficients: &CoefficientList<F>,
    ) where
        H: DuplexSpongeInterface,
        R: RngCore + CryptoRng,
        F: Codec<[H::U]>,
    {
        assert_eq!(coefficients.num_coeffs(), self.final_sumcheck.initial_size);
        for coeff in coefficients.coeffs() {
            prover_state.prover_message(coeff);
        }
    }

    /// Compute deferred constraint hints and write them to the transcript.
    ///
    /// Returns `(constraint_evaluation_point, deferred_values)`.
    fn compute_deferred_hints<H, R>(
        &self,
        prover_state: &mut ProverState<H, R>,
        weights: &[&Weights<F>],
        randomness_vec: &[F],
    ) -> (MultilinearPoint<F>, Vec<F>)
    where
        H: DuplexSpongeInterface,
        R: RngCore + CryptoRng,
        F: Codec<[H::U]>,
    {
        let constraint_eval = MultilinearPoint(randomness_vec.iter().copied().rev().collect());
        let deferred: Vec<F> = weights
            .iter()
            .filter(|w| w.deferred())
            .map(|w| w.compute(&constraint_eval))
            .collect();
        prover_state.prover_hint_ark(&deferred);
        (constraint_eval, deferred)
    }

    /// Prove helper polynomial evaluations for the ZK virtual oracle.
    ///
    /// Given the IRS opening of f̂, this:
    /// 1. Computes gamma points (coset elements) for each query
    /// 2. For each polynomial, batch-evaluates M, ĝ₁, ..., ĝμ at all gamma points
    /// 3. Sends evaluations to the verifier (gamma-major, polynomial-minor order)
    /// 4. Runs a helper WHIR proof to bind evaluations to committed polynomials
    ///
    /// For N polynomials, the helper WHIR covers N×(μ+1) polynomials in one batch.
    fn prove_zk_helper_evaluations<H, R>(
        &self,
        prover_state: &mut ProverState<H, R>,
        in_domain_base: &irs_commit::Evaluations<F::BasePrimeField>,
        witness: &ZkWitness<F>,
        helper_config: &Config<F>,
        rho: F,
    ) where
        H: DuplexSpongeInterface,
        R: RngCore + CryptoRng,
        F: Codec<[H::U]>,
        [u8; 32]: Decoding<[H::U]>,
        U64: Codec<[H::U]>,
        u8: Decoding<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        let num_polys = witness.preprocessings.len();
        let mu = witness.preprocessings[0].params.mu;
        let ell = witness.preprocessings[0].params.ell;
        let k = self.initial_committer.interleaving_depth;
        let num_rows = self.initial_committer.num_rows();
        let full_domain_size = num_rows * k;

        // Get generators for the full and sub NTT domains
        let omega_full: F::BasePrimeField =
            ntt::generator(full_domain_size).expect("Full IRS domain should have primitive root");
        let omega_sub: F::BasePrimeField = self.initial_committer.generator();
        let zeta: F::BasePrimeField = omega_full.pow([num_rows as u64]); // k-th root of unity

        // Precompute sub-domain powers for index recovery
        let mut omega_powers = Vec::with_capacity(num_rows);
        let mut power = F::BasePrimeField::ONE;
        for _ in 0..num_rows {
            omega_powers.push(power);
            power *= omega_sub;
        }

        // Compute gammas: for each query point, produce k coset elements
        // These are the SAME for all polynomials (derived from IRS domain structure).
        let embedding = self.embedding();
        let gammas: Vec<F> = in_domain_base
            .points
            .iter()
            .flat_map(|&alpha_base| {
                let idx = omega_powers
                    .iter()
                    .position(|&p| p == alpha_base)
                    .expect("Query point must be in IRS domain");
                let coset_offset = omega_full.pow([idx as u64]);
                (0..k).map(move |j| {
                    let gamma_base = coset_offset * zeta.pow([j as u64]);
                    embedding.map(gamma_base)
                })
            })
            .collect();

        // For each polynomial, batch-evaluate all helper polynomials at all gamma points
        let helper_evals_per_poly: Vec<Vec<HelperEvaluations<F>>> = witness
            .preprocessings
            .iter()
            .map(|preprocessing| preprocessing.batch_evaluate_helpers(&gammas, rho))
            .collect();

        // Send helper evaluations: for each gamma, for each polynomial
        // This groups all polynomial data per-gamma for natural virtual oracle reconstruction.
        for gamma_idx in 0..gammas.len() {
            for poly_idx in 0..num_polys {
                let helper_eval = &helper_evals_per_poly[poly_idx][gamma_idx];
                prover_state.prover_message(&helper_eval.m_eval);
                for g_hat_eval in &helper_eval.g_hat_evals {
                    prover_state.prover_message(g_hat_eval);
                }
            }
        }

        // Sample τ₂ for combining query points
        let tau2: F = prover_state.verifier_message();

        // Construct batched eq weights (uses gammas which are same for all polynomials)
        let beq_weights =
            Self::construct_batched_eq_weights(&helper_evals_per_poly[0], rho, tau2, ell);

        // Compute per-polynomial claims and collect evaluations
        // Layout: [m₁_claim, ĝ₁₁_claim, ..., ĝ₁μ_claim, m₂_claim, ĝ₂₁_claim, ..., ĝ₂μ_claim, ...]
        let mut all_evaluations: Vec<F> = Vec::with_capacity(num_polys * (1 + mu));
        for poly_idx in 0..num_polys {
            let (m_claim, g_hat_claims) =
                Self::compute_per_polynomial_claims(&helper_evals_per_poly[poly_idx], tau2);
            all_evaluations.push(m_claim);
            all_evaluations.extend_from_slice(&g_hat_claims);
        }

        // Collect all helper polynomials (base-field):
        // [M₁, ĝ₁₁, ..., ĝ₁μ, M₂, ĝ₂₁, ..., ĝ₂μ, ...]
        let mut all_polynomials: Vec<&CoefficientList<F::BasePrimeField>> =
            Vec::with_capacity(num_polys * (1 + mu));
        for (m_poly, g_hats) in zip_strict(&witness.m_polys_base, &witness.g_hats_embedded_bases) {
            all_polynomials.push(m_poly);
            for g_hat_base in g_hats {
                all_polynomials.push(g_hat_base);
            }
        }

        // Single batch witness (helper_config.batch_size = N×(μ+1))
        let all_witnesses: Vec<&irs_commit::Witness<F::BasePrimeField, F>> =
            vec![&witness.helper_witness];

        let weight_refs: Vec<&Weights<F>> = vec![&beq_weights];

        // Run helper WHIR proof with existing batch commitment
        helper_config.prove(
            prover_state,
            &all_polynomials,
            &all_witnesses,
            &weight_refs,
            &all_evaluations,
        );
    }

    /// Compute per-polynomial claims from helper evaluations.
    ///
    /// m_claim = Σᵢ τ₂ⁱ · m(γᵢ, ρ)
    /// g_hat_j_claim = Σᵢ τ₂ⁱ · ĝⱼ(pow(γᵢ))
    pub(crate) fn compute_per_polynomial_claims(
        helper_evals: &[HelperEvaluations<F>],
        tau2: F,
    ) -> (F, Vec<F>) {
        let num_g_hats = helper_evals.first().map_or(0, |h| h.g_hat_evals.len());

        let mut m_claim = F::ZERO;
        let mut g_hat_claims = vec![F::ZERO; num_g_hats];
        let mut tau2_power = F::ONE;

        for helper in helper_evals {
            m_claim += tau2_power * helper.m_eval;
            for (j, &g_eval) in helper.g_hat_evals.iter().enumerate() {
                g_hat_claims[j] += tau2_power * g_eval;
            }
            tau2_power *= tau2;
        }

        (m_claim, g_hat_claims)
    }

    /// Construct the weight function for the helper WHIR sumcheck:
    ///
    ///   w(z, t) = eq(-ρ, t) · [Σᵢ τ₂ⁱ · eq(pow(γᵢ), z)]
    ///
    /// Returns a `Weights::Linear` on (ℓ+1) variables.
    pub(crate) fn construct_batched_eq_weights(
        helper_evals: &[HelperEvaluations<F>],
        rho: F,
        tau2: F,
        ell: usize,
    ) -> Weights<F> {
        let neg_rho = -rho;
        let z_size = 1 << ell;
        let weight_size = 1 << (ell + 1);

        // Precompute τ₂ powers
        let tau2_powers: Vec<F> = {
            let mut powers = Vec::with_capacity(helper_evals.len());
            let mut p = F::ONE;
            for _ in 0..helper_evals.len() {
                powers.push(p);
                p *= tau2;
            }
            powers
        };

        // Butterfly expansion: compute τ₂ⁱ · eq(pow(γᵢ), z) for all z per γᵢ,
        // then reduce into a single batched_eq vector.
        //
        // For each γᵢ, eq(pow(γᵢ), z) is computed for ALL z ∈ {0,1}^ℓ using
        // the tensor-product structure of eq in O(2^ℓ).
        const MAX_ELL: usize = 64;
        assert!(
            ell <= MAX_ELL,
            "ℓ={ell} exceeds stack buffer size {MAX_ELL}"
        );

        let compute_weighted_eq = |(helper, &tau2_pow): (&HelperEvaluations<F>, &F)| -> Vec<F> {
            let mut powers_buf = [F::ZERO; MAX_ELL];
            let mut cur = helper.gamma;
            for p in powers_buf[..ell].iter_mut() {
                *p = cur;
                cur *= cur;
            }

            let mut eq_vals = Vec::with_capacity(z_size);
            eq_vals.push(F::ONE);
            for &ci in powers_buf[..ell].iter().rev() {
                let len = eq_vals.len();
                let one_minus_ci = F::ONE - ci;
                eq_vals.resize(2 * len, F::ZERO);
                for j in (0..len).rev() {
                    eq_vals[2 * j + 1] = eq_vals[j] * ci;
                    eq_vals[2 * j] = eq_vals[j] * one_minus_ci;
                }
            }
            for v in eq_vals.iter_mut() {
                *v *= tau2_pow;
            }
            eq_vals
        };

        #[cfg(feature = "parallel")]
        let batched_eq: Vec<F> = {
            use rayon::prelude::*;
            helper_evals
                .par_iter()
                .zip(tau2_powers.par_iter())
                .fold(
                    || vec![F::ZERO; z_size],
                    |mut acc, pair| {
                        let eq_vals = compute_weighted_eq(pair);
                        for (a, v) in acc.iter_mut().zip(eq_vals) {
                            *a += v;
                        }
                        acc
                    },
                )
                .reduce(
                    || vec![F::ZERO; z_size],
                    |mut a, b| {
                        for (ai, bi) in a.iter_mut().zip(b) {
                            *ai += bi;
                        }
                        a
                    },
                )
        };
        #[cfg(not(feature = "parallel"))]
        let batched_eq: Vec<F> = {
            let mut batched = vec![F::ZERO; z_size];
            // Reuse a single buffer for the butterfly expansion across all iterations,
            // instead of allocating a new Vec<F> of size z_size per gamma point.
            let mut eq_buf = Vec::with_capacity(z_size);
            let mut powers_buf = [F::ZERO; MAX_ELL];
            for (helper, &tau2_pow) in helper_evals.iter().zip(tau2_powers.iter()) {
                // Compute pow(γ) components
                let mut cur = helper.gamma;
                for p in powers_buf[..ell].iter_mut() {
                    *p = cur;
                    cur *= cur;
                }
                // Butterfly expansion of eq(pow(γ), z)
                eq_buf.clear();
                eq_buf.push(F::ONE);
                for &ci in powers_buf[..ell].iter().rev() {
                    let len = eq_buf.len();
                    let one_minus_ci = F::ONE - ci;
                    eq_buf.resize(2 * len, F::ZERO);
                    for j in (0..len).rev() {
                        eq_buf[2 * j + 1] = eq_buf[j] * ci;
                        eq_buf[2 * j] = eq_buf[j] * one_minus_ci;
                    }
                }
                // Accumulate τ₂^i · eq(pow(γᵢ), z) into batched
                for (a, &v) in batched.iter_mut().zip(eq_buf.iter()) {
                    *a += tau2_pow * v;
                }
            }
            batched
        };

        // Build weight evaluations on {0,1}^(ℓ+1)
        // w(z, t) = eq(-ρ, t) × batched_eq[z]
        // eq(-ρ, 0) = 1 + ρ,  eq(-ρ, 1) = -ρ
        let eq_neg_rho_at_0 = F::ONE - neg_rho; // = 1 + ρ
        let eq_neg_rho_at_1 = neg_rho; // = -ρ

        let mut weight_evals = vec![F::ZERO; weight_size];
        for (z_idx, &beq_z) in batched_eq.iter().enumerate() {
            weight_evals[z_idx * 2] = eq_neg_rho_at_0 * beq_z; // t = 0
            weight_evals[z_idx * 2 + 1] = eq_neg_rho_at_1 * beq_z; // t = 1
        }

        Weights::linear(EvaluationsList::new(weight_evals))
    }
}
