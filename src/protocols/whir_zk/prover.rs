use ark_ff::{FftField, PrimeField};
use ark_std::rand::{CryptoRng, RngCore};
#[cfg(feature = "tracing")]
use tracing::instrument;

use super::utils::{
    compute_per_polynomial_claims, construct_batched_eq_weights, interleave_helper_poly_refs,
    IrsDomainParams, ZkWitness,
};
use crate::{
    algebra::{
        dot, mixed_scalar_mul_add,
        polynomials::{CoefficientList, EvaluationsList, MultilinearPoint},
        Weights,
    },
    hash::Hash,
    protocols::{geometric_challenge::geometric_challenge, irs_commit, whir::Config},
    transcript::{
        self, codecs::U64, Codec, Decoding, DuplexSpongeInterface, ProverMessage, ProverState,
        VerifierMessage,
    },
    utils::zip_strict,
};

impl<F: FftField> Config<F> {
    /// Prove a ZK WHIR opening.
    ///
    /// This proves knowledge of a polynomial `f` by:
    /// 1. Blinding with g to form P = ρ·f + g
    /// 2. Running WHIR rounds on P with a virtual oracle L = ρ·f̂ + h
    /// 3. Proving helper polynomial evaluations so verifier can reconstruct L
    #[cfg_attr(feature = "tracing", instrument(skip_all, fields(num_polynomials = polynomials.len())))]
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
        H: DuplexSpongeInterface<U = u8>,
        R: RngCore + CryptoRng,
        F: Codec<[H::U]>,
        [u8; 32]: Decoding<[H::U]>,
        U64: Codec<[H::U]>,
        u8: Decoding<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        #[cfg(feature = "alloc-track")]
        let mut __snap = crate::alloc_snap!();

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
                crate::algebra::scalar_mul_add(acc.coeffs_mut(), *rlc_coeff, src_poly.coeffs());
            }
            acc
        };

        #[cfg(feature = "alloc-track")]
        crate::alloc_report!("prove_zk::phase1_blinding_setup", __snap);

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

        #[cfg(feature = "alloc-track")]
        crate::alloc_report!("prove_zk::phase2_initial_sumcheck", __snap);

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

        #[cfg(feature = "alloc-track")]
        crate::alloc_report!("prove_zk::phase3_whir_rounds", __snap);

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

        #[cfg(feature = "alloc-track")]
        crate::alloc_report!("prove_zk::phase4_final_opening", __snap);

        // Phase 5: Final sumcheck and deferred constraint hints
        let final_folding_randomness =
            self.final_sumcheck
                .prove(prover_state, &mut eval_list, &mut constraints, &mut the_sum);
        randomness_vec.extend(final_folding_randomness.0.iter().rev());

        #[cfg(feature = "alloc-track")]
        crate::alloc_report!("prove_zk::phase5_final_sumcheck", __snap);

        self.compute_deferred_hints(prover_state, weights, &randomness_vec)
    }

    /// Build the blinding polynomial g(X) = g₀(X) + Σᵢ₌₁^μ βⁱ · X^(2^(i-1)) · ĝᵢ(X)
    ///
    /// Returns the blinding polynomial g as a `CoefficientList<F>`.
    pub(crate) fn build_blinding_polynomial(
        &self,
        preprocessing: &super::utils::ZkPreprocessingPolynomials<F>,
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
            let target = &mut coeffs[shift..shift + g_hat_coeffs.len()];
            crate::algebra::scalar_mul_add(target, beta_power, g_hat_coeffs);
            beta_power *= beta;
        }

        CoefficientList::new(coeffs)
    }

    /// Transform g → P = ρ·f + g in-place: P(X) = ρ·embed(f(X)) + g(X).
    pub(crate) fn build_blinded_polynomial_p(
        &self,
        g_poly: CoefficientList<F>,
        polynomial: &CoefficientList<F::BasePrimeField>,
        rho: F,
    ) -> CoefficientList<F> {
        let mut coeffs = g_poly.into_coeffs();
        let f_coeffs = polynomial.coeffs();
        mixed_scalar_mul_add(
            self.embedding(),
            &mut coeffs[..f_coeffs.len()],
            rho,
            f_coeffs,
        );
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
        H: DuplexSpongeInterface<U = u8>,
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

    /// Prove helper polynomial evaluations for the ZK virtual oracle.
    ///
    /// Thin wrapper around [`prove_helper_evaluations`] that derives the IRS
    /// domain from this config's initial committer.
    fn prove_zk_helper_evaluations<H, R>(
        &self,
        prover_state: &mut ProverState<H, R>,
        in_domain_base: &irs_commit::Evaluations<F::BasePrimeField>,
        witness: &ZkWitness<F>,
        helper_config: &Config<F>,
        rho: F,
    ) where
        H: DuplexSpongeInterface<U = u8>,
        R: RngCore + CryptoRng,
        F: Codec<[H::U]>,
        [u8; 32]: Decoding<[H::U]>,
        U64: Codec<[H::U]>,
        u8: Decoding<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        let domain = IrsDomainParams::<F>::from_config(self);
        prove_helper_evaluations(
            prover_state,
            &domain,
            in_domain_base,
            witness,
            helper_config,
            rho,
            self.embedding(),
        );
    }
}

/// Prove helper polynomial evaluations for the ZK virtual oracle.
///
/// This is the shared implementation used by both the main ZK prover
/// (via `Config::prove_zk_helper_evaluations`) and the prefold prover.
///
/// Given the IRS opening of f̂, this:
/// 1. Computes gamma points (coset elements) for each query
/// 2. For each polynomial, batch-evaluates M, ĝ₁, ..., ĝμ at all gamma points
/// 3. Sends evaluations to the verifier (gamma-major, polynomial-minor order)
/// 4. Runs a helper WHIR proof to bind evaluations to committed polynomials
///
/// For N polynomials, the helper WHIR covers N×(μ+1) polynomials in one batch.
pub(crate) fn prove_helper_evaluations<F, H, R, M>(
    prover_state: &mut ProverState<H, R>,
    domain: &IrsDomainParams<F>,
    in_domain_base: &irs_commit::Evaluations<F::BasePrimeField>,
    witness: &ZkWitness<F>,
    helper_config: &Config<F>,
    rho: F,
    embedding: &M,
) where
    F: FftField,
    H: DuplexSpongeInterface<U = u8>,
    R: RngCore + CryptoRng,
    F: Codec<[H::U]>,
    [u8; 32]: Decoding<[H::U]>,
    U64: Codec<[H::U]>,
    u8: Decoding<[H::U]>,
    Hash: ProverMessage<[H::U]>,
    M: crate::algebra::embedding::Embedding<Source = F::BasePrimeField, Target = F>,
{
    #[cfg(feature = "alloc-track")]
    let mut __snap = crate::alloc_snap!();

    let num_polys = witness.preprocessings.len();
    let mu = witness.preprocessings[0].params.mu;
    let ell = witness.preprocessings[0].params.ell;

    // Compute gammas: for each query point, produce k coset elements
    // These are the SAME for all polynomials (derived from IRS domain structure).
    let gammas = domain.all_gammas(&in_domain_base.points, embedding);

    // For each polynomial, batch-evaluate all helper polynomials at all gamma points
    let helper_evals_per_poly: Vec<Vec<super::utils::HelperEvaluations<F>>> = witness
        .preprocessings
        .iter()
        .map(|preprocessing| preprocessing.batch_evaluate_helpers(&gammas, rho))
        .collect();

    #[cfg(feature = "alloc-track")]
    crate::alloc_report!("  helper_evals::batch_evaluate", __snap);

    // Send helper evaluations as a single batch message.
    // Order: for each gamma, for each polynomial: m_eval, then mu g_hat_evals.
    // This groups all polynomial data per-gamma for natural virtual oracle reconstruction.
    let evals_per_point = 1 + mu; // m_eval + mu g_hat_evals
    let total_evals = gammas.len() * num_polys * evals_per_point;
    let base_field_size = (F::BasePrimeField::MODULUS_BIT_SIZE.div_ceil(8)) as usize;
    let elem_bytes = base_field_size * F::extension_degree() as usize;
    let mut encoded = Vec::with_capacity(total_evals * elem_bytes);
    for gamma_idx in 0..gammas.len() {
        for poly_idx in 0..num_polys {
            let helper_eval = &helper_evals_per_poly[poly_idx][gamma_idx];
            transcript::encode_field_element_into(&helper_eval.m_eval, &mut encoded);
            for g_hat_eval in &helper_eval.g_hat_evals {
                transcript::encode_field_element_into(g_hat_eval, &mut encoded);
            }
        }
    }
    prover_state.prover_messages_bytes::<F>(total_evals, &encoded);

    #[cfg(feature = "alloc-track")]
    crate::alloc_report!("  helper_evals::send_to_verifier", __snap);

    // Sample τ₂ for combining query points
    let tau2: F = prover_state.verifier_message();

    // Construct batched eq weights (uses gammas which are same for all polynomials)
    let beq_weights = construct_batched_eq_weights(&helper_evals_per_poly[0], rho, tau2, ell);

    // Compute per-polynomial claims and collect evaluations
    // Layout: [m₁_claim, ĝ₁₁_claim, ..., ĝ₁μ_claim, m₂_claim, ĝ₂₁_claim, ..., ĝ₂μ_claim, ...]
    let mut all_evaluations: Vec<F> = Vec::with_capacity(num_polys * (1 + mu));
    for poly_idx in 0..num_polys {
        let (m_claim, g_hat_claims) =
            compute_per_polynomial_claims(&helper_evals_per_poly[poly_idx], tau2);
        all_evaluations.push(m_claim);
        all_evaluations.extend_from_slice(&g_hat_claims);
    }

    // Collect all helper polynomials (base-field):
    // [M₁, ĝ₁₁, ..., ĝ₁μ, M₂, ĝ₂₁, ..., ĝ₂μ, ...]
    let all_polynomials =
        interleave_helper_poly_refs::<F>(&witness.m_polys_base, &witness.g_hats_embedded_bases);

    // Single batch witness (helper_config.batch_size = N×(μ+1))
    let all_witnesses: Vec<&irs_commit::Witness<F::BasePrimeField, F>> =
        vec![&witness.helper_witness];

    let weight_refs: Vec<&Weights<F>> = vec![&beq_weights];

    #[cfg(feature = "alloc-track")]
    crate::alloc_report!("  helper_evals::build_weights_claims", __snap);

    // Run helper WHIR proof with existing batch commitment
    helper_config.prove(
        prover_state,
        &all_polynomials,
        &all_witnesses,
        &weight_refs,
        &all_evaluations,
    );

    #[cfg(feature = "alloc-track")]
    crate::alloc_report!("  helper_evals::helper_whir_prove", __snap);
}
