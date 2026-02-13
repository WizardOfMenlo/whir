//! Staged ZK Folding (Prefold) for mixed-arity polynomial batching.
//!
//! This module implements the prefold approach: given N polynomials at varying
//! arities, fold each down to the minimum arity using sumcheck-derived randomness,
//! then batch-prove all polynomials at the common arity.
//!
//! ## Architecture
//!
//! For each polynomial at arity L > n_min:
//! 1. Commit f̂ = f + msk at arity L (base field, ZK)
//! 2. Build P = ρ·f + g (blinded polynomial at arity L)
//! 3. Run prefold sumcheck on P's constraints → fold randomness
//! 4. Fold: P' = fold(P, fold_randomness) → arity n_min (extension field)
//! 5. Send P' coefficients in the clear (ZK-blinded by g)
//! 6. **Binding equation**: verify v' = Σ_i rlc_i · eq(a_i_high, r) · P'(a_i_low)
//! 7. Open f̂ → virtual oracle → fold → STIR consistency against P'
//!
//! The binding equation (step 6) closes the soundness gap by ensuring the
//! sumcheck's reduced claim matches the actual P' polynomial. The STIR
//! consistency check (step 7) ensures P' is the correct fold of the committed f̂.
//!
//! Then the main WHIR at arity n_min handles all native polynomials.

use ark_ff::{FftField, PrimeField};
use ark_std::rand::{CryptoRng, RngCore};

use super::utils::{
    interleave_helper_poly_refs, prepare_helper_polynomials, IrsDomainParams, ZkParams,
    ZkPreprocessingPolynomials, ZkWitness,
};
use crate::{
    algebra::{
        add_base_with_projection, dot,
        polynomials::{CoefficientList, EvaluationsList, MultilinearPoint},
        Weights,
    },
    bits::Bits,
    hash::Hash,
    parameters::ProtocolParameters,
    protocols::{
        geometric_challenge::geometric_challenge, irs_commit, matrix_commit, proof_of_work,
        sumcheck, whir::Config,
    },
    transcript::{
        codecs::U64, Codec, Decoding, DuplexSpongeInterface, ProverMessage, ProverState,
        VerificationResult, VerifierMessage, VerifierState,
    },
    type_info::Type,
    utils::zip_strict,
    verify,
};

/// Configuration for a single prefold arity level (polynomials at arity > n_min).
///
/// Each distinct arity above n_min needs its own config: an IRS committer for the
/// original f̂ commitment, a helper WHIR config for the virtual oracle sub-proof,
/// and a sumcheck config. After folding, P' coefficients are sent in the clear
/// (they are ZK-blinded by construction) so no separate P' committer is needed.
#[derive(Clone)]
pub struct PrefoldLevelConfig<F: FftField> {
    /// IRS committer for f̂ at this arity (base field → extension field).
    pub f_hat_committer: irs_commit::BasefieldConfig<F>,

    /// Helper WHIR config for virtual oracle proof at this arity level.
    pub helper_config: Config<F>,

    /// ZK parameters (ℓ, μ) for this arity level.
    pub zk_params: ZkParams,

    /// Sumcheck config for the prefold (folds `fold_depth` variables).
    pub prefold_sumcheck: sumcheck::Config<F>,

    /// Proof-of-work config for the prefold STIR queries.
    pub prefold_pow: proof_of_work::Config,

    /// The arity at this level (number of variables).
    pub arity: usize,

    /// Number of extra variables to fold away (arity − n_min).
    pub fold_depth: usize,
}

/// Input for a group of polynomials at the same prefold arity level.
pub struct PrefoldGroupInput<'a, F: FftField> {
    /// Base-field polynomials at this arity level.
    pub polynomials: &'a [&'a CoefficientList<F::BasePrimeField>],
    /// ZK witness (f̂ + helper commitments at this arity).
    pub witness: &'a ZkWitness<F>,
    /// Constraint weights at this arity level.
    pub weights: &'a [&'a Weights<F>],
    /// Evaluations: row-major \[w₀\_p₀, w₀\_p₁, ..., w₁\_p₀, ...\].
    pub evaluations: &'a [F],
    /// Level config for this arity.
    pub level_config: &'a PrefoldLevelConfig<F>,
}

impl<F: FftField> PrefoldLevelConfig<F>
where
    F: crate::algebra::fields::FieldWithSize,
{
    /// Build a prefold level config for polynomials at `arity` > `n_min`.
    ///
    /// `main_config` is the WHIR config at n_min. `whir_params` provides the
    /// security/folding parameters.
    pub fn new(main_config: &Config<F>, arity: usize, whir_params: &ProtocolParameters) -> Self {
        let n_min = main_config.initial_num_variables();
        let fold_depth = arity.checked_sub(n_min).expect("arity must be > n_min");
        assert!(fold_depth > 0, "fold_depth must be positive");

        // f̂ IRS committer at this arity (base field)
        // Interleaving depth = 2^fold_depth so that folding gives 1 value per query.
        let interleaving_depth = 1usize << fold_depth;
        let polynomial_size = 1usize << arity;
        let expansion = main_config.initial_committer.expansion;
        let num_rows = polynomial_size * expansion / interleaving_depth;

        let f_hat_committer = irs_commit::Config {
            embedding: Default::default(),
            num_polynomials: whir_params.batch_size,
            polynomial_size,
            expansion,
            interleaving_depth,
            matrix_commit: matrix_commit::Config::with_hash(
                whir_params.hash_id,
                num_rows,
                whir_params.batch_size * interleaving_depth,
            ),
            in_domain_samples: main_config.initial_committer.in_domain_samples,
            out_domain_samples: main_config.initial_committer.out_domain_samples,
            deduplicate_in_domain: true,
        };

        // ZK params for this arity
        let zk_params = ZkParams::from_whir_params_with_arity(main_config, arity);

        // Helper WHIR config (shared with api.rs)
        let helper_config = zk_params.build_helper_config(whir_params.batch_size, whir_params);

        // Prefold sumcheck
        // Folds `fold_depth` variables of the blinded polynomial at arity L.
        let prefold_sumcheck = sumcheck::Config {
            field: Type::<F>::new(),
            initial_size: polynomial_size,
            round_pow: proof_of_work::Config::from_difficulty(Bits::new(0.0)),
            num_rounds: fold_depth,
        };

        // Prefold PoW (minimal for first implementation)
        let prefold_pow = proof_of_work::Config::from_difficulty(Bits::new(0.0));

        Self {
            f_hat_committer,
            helper_config,
            zk_params,
            prefold_sumcheck,
            prefold_pow,
            arity,
            fold_depth,
        }
    }
}

impl ZkParams {
    /// Compute ZK parameters for a given arity, using the main config's query
    /// parameters as reference.
    pub fn from_whir_params_with_arity<F: FftField>(main_config: &Config<F>, arity: usize) -> Self {
        let mu = arity;
        let k = main_config.initial_committer.interleaving_depth;
        let q1 = main_config
            .round_configs
            .first()
            .map_or(main_config.initial_committer.in_domain_samples, |r| {
                r.irs_committer.in_domain_samples
            });

        let q_ub = 2 * k * q1 + 4 * mu + 10;
        let ell = (q_ub as f64).log2().ceil() as usize;
        assert!(
            ell < mu,
            "ZK requires ℓ < μ (ℓ={ell}, μ={mu}). \
             Increase arity or lower security_level/queries."
        );
        Self { ell, mu }
    }
}

/// Commit polynomials at a prefold arity level.
///
/// This is analogous to `Config::commit_zk` but uses the prefold level's IRS
/// committer (at the higher arity) instead of the main config's.
pub fn commit_zk_at_level<F, H, R>(
    level_config: &PrefoldLevelConfig<F>,
    prover_state: &mut ProverState<H, R>,
    polynomials: &[&CoefficientList<F::BasePrimeField>],
    preprocessings: &[&ZkPreprocessingPolynomials<F>],
) -> ZkWitness<F>
where
    F: FftField,
    H: DuplexSpongeInterface,
    R: RngCore + CryptoRng,
    F: Codec<[H::U]>,
    Hash: ProverMessage<[H::U]>,
{
    // 1. Commit f̂ᵢ = fᵢ + mskᵢ at the prefold arity
    let mut f_hat_witnesses = Vec::new();
    for (polynomial, preprocessing) in zip_strict(polynomials, preprocessings) {
        let f_hat_coeffs =
            add_base_with_projection::<F>(polynomial.coeffs(), preprocessing.msk.coeffs());
        let f_hat = CoefficientList::new(f_hat_coeffs);
        let poly_refs: Vec<&[F::BasePrimeField]> = vec![f_hat.coeffs()];
        let f_hat_witness = level_config
            .f_hat_committer
            .commit(prover_state, &poly_refs);
        f_hat_witnesses.push(f_hat_witness);
    }

    // 2. Prepare helper polynomials (shared with commit_zk)
    let (m_polys_base, g_hats_embedded_bases) = prepare_helper_polynomials(preprocessings);

    // 3. Batch-commit helpers via the level's helper config
    let helper_poly_refs = interleave_helper_poly_refs::<F>(&m_polys_base, &g_hats_embedded_bases);
    let helper_witness = level_config
        .helper_config
        .commit(prover_state, &helper_poly_refs);

    ZkWitness {
        f_hat_witnesses,
        helper_witness,
        preprocessings: preprocessings.iter().copied().cloned().collect(),
        m_polys_base,
        g_hats_embedded_bases,
    }
}

/// Receive commitments for a prefold level on the verifier side.
pub fn receive_prefold_commitments<F, H>(
    level_config: &PrefoldLevelConfig<F>,
    verifier_state: &mut VerifierState<'_, H>,
    num_polynomials: usize,
) -> VerificationResult<(Vec<irs_commit::Commitment<F>>, irs_commit::Commitment<F>)>
where
    F: FftField,
    H: DuplexSpongeInterface,
    F: Codec<[H::U]>,
    Hash: ProverMessage<[H::U]>,
{
    // Read f̂ commitments (one per polynomial)
    let f_hat_commitments: Vec<_> = (0..num_polynomials)
        .map(|_| {
            level_config
                .f_hat_committer
                .receive_commitment(verifier_state)
        })
        .collect::<Result<_, _>>()?;

    // Read helper batch commitment
    let helper_commitment = level_config
        .helper_config
        .receive_commitment(verifier_state)?;

    Ok((f_hat_commitments, helper_commitment))
}

impl<F: FftField> Config<F> {
    /// Full ZK prefold + prove pipeline for mixed-arity polynomials.
    ///
    /// # Architecture
    ///
    /// 1. Phase 1: ZK blinding (shared β, per-group g evaluations, shared ρ)
    /// 2. Phase 2: For each prefold group (highest arity first):
    ///    - RLC polynomials at this level
    ///    - Prefold sumcheck (if constraints) → fold randomness
    ///    - Fold P → P' at n_min
    ///    - Commit P' via extension-field IRS
    ///    - Open f̂ → virtual oracle → fold → STIR consistency values
    /// 3. Phase 3: Standard prove_zk on native polynomials
    ///
    /// # Arguments
    ///
    /// * `self` — Main WHIR config at arity n_min.
    /// * `native_polys`, `native_witness`, etc. — Native-arity group.
    /// * `prefold_groups` — Higher-arity groups sorted by decreasing arity.
    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    pub fn prove_zk_prefold<H, R>(
        &self,
        prover_state: &mut ProverState<H, R>,
        // Native group (arity = n_min)
        native_polys: &[&CoefficientList<F::BasePrimeField>],
        native_witness: &ZkWitness<F>,
        native_helper_config: &Config<F>,
        native_weights: &[&Weights<F>],
        native_evals: &[F],
        // Prefold groups (highest arity first)
        prefold_groups: &[PrefoldGroupInput<'_, F>],
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
        let n_min = self.initial_num_variables();
        let num_native = native_polys.len();

        // ================================================================
        // Phase 1: ZK blinding — shared β, per-group g evals, shared ρ
        // ================================================================
        let beta: F = prover_state.verifier_message();

        // Build g and send evaluations for native group
        let mut native_g_polys = Vec::with_capacity(num_native);
        let mu_native = native_witness.preprocessings[0].params.mu;
        for preprocessing in &native_witness.preprocessings {
            native_g_polys.push(self.build_blinding_polynomial(preprocessing, mu_native, beta));
        }
        let mut native_g_eval_matrix = vec![F::ZERO; native_weights.len() * num_native];
        for (i, weight) in native_weights.iter().enumerate() {
            for (j, g_poly) in native_g_polys.iter().enumerate() {
                let eval = weight.evaluate(g_poly);
                prover_state.prover_message(&eval);
                native_g_eval_matrix[i * num_native + j] = eval;
            }
        }

        // Build g and send evaluations for each prefold group
        let mut prefold_g_polys_per_group: Vec<Vec<CoefficientList<F>>> = Vec::new();
        let mut prefold_g_eval_matrices: Vec<Vec<F>> = Vec::new();
        for group in prefold_groups {
            let mu_level = group.level_config.arity;
            let num_polys = group.polynomials.len();
            let mut g_polys = Vec::with_capacity(num_polys);
            for preprocessing in &group.witness.preprocessings {
                g_polys.push(self.build_blinding_polynomial(preprocessing, mu_level, beta));
            }
            let mut g_eval_matrix = vec![F::ZERO; group.weights.len() * num_polys];
            for (i, weight) in group.weights.iter().enumerate() {
                for (j, g_poly) in g_polys.iter().enumerate() {
                    let eval = weight.evaluate(g_poly);
                    prover_state.prover_message(&eval);
                    g_eval_matrix[i * num_polys + j] = eval;
                }
            }
            prefold_g_polys_per_group.push(g_polys);
            prefold_g_eval_matrices.push(g_eval_matrix);
        }

        let rho: F = prover_state.verifier_message();

        // Build P = ρ·f + g for each polynomial
        let mut native_p_polys: Vec<CoefficientList<F>> = Vec::with_capacity(num_native);
        for (polynomial, g_poly) in zip_strict(native_polys.iter(), native_g_polys.into_iter()) {
            native_p_polys.push(self.build_blinded_polynomial_p(g_poly, polynomial, rho));
        }

        let mut prefold_p_polys: Vec<Vec<CoefficientList<F>>> = Vec::new();
        for (group_idx, group) in prefold_groups.iter().enumerate() {
            let g_polys = std::mem::take(&mut prefold_g_polys_per_group[group_idx]);
            let mut p_polys = Vec::new();
            for (polynomial, g_poly) in zip_strict(group.polynomials.iter(), g_polys.into_iter()) {
                p_polys.push(self.build_blinded_polynomial_p(g_poly, polynomial, rho));
            }
            prefold_p_polys.push(p_polys);
        }

        // ================================================================
        // Phase 2: Prefold stages — fold each group to arity n_min
        // ================================================================
        for (group_idx, group) in prefold_groups.iter().enumerate() {
            let level_config = group.level_config;
            let num_polys = group.polynomials.len();
            let fold_depth = level_config.fold_depth;

            // RLC polynomials at this level
            let level_poly_rlc: Vec<F> = geometric_challenge(prover_state, num_polys);
            let mut p_combined = {
                let p_polys = &prefold_p_polys[group_idx];
                let mut acc = CoefficientList::new(vec![F::ZERO; p_polys[0].num_coeffs()]);
                for (rlc, poly) in zip_strict(&level_poly_rlc, p_polys) {
                    crate::algebra::scalar_mul_add(acc.coeffs_mut(), *rlc, poly.coeffs());
                }
                acc
            };

            // Modified evaluations
            let modified_evals: Vec<F> = group
                .evaluations
                .iter()
                .zip(prefold_g_eval_matrices[group_idx].iter())
                .map(|(&eval, &g_eval)| rho * eval + g_eval)
                .collect();

            // Prefold sumcheck or random fold
            let fold_randomness;
            let has_constraints = !group.weights.is_empty();
            if has_constraints {
                let constraint_rlc: Vec<F> = geometric_challenge(prover_state, group.weights.len());
                let mut constraints = EvaluationsList::new(vec![F::ZERO; 1 << level_config.arity]);
                for (rlc, weight) in zip_strict(&constraint_rlc, group.weights) {
                    weight.accumulate(&mut constraints, *rlc);
                }
                let mut the_sum: F =
                    zip_strict(&constraint_rlc, modified_evals.chunks_exact(num_polys))
                        .map(|(w, row)| *w * dot(&level_poly_rlc, row))
                        .sum();

                let mut eval_list = EvaluationsList::from(p_combined.clone());
                fold_randomness = level_config.prefold_sumcheck.prove(
                    prover_state,
                    &mut eval_list,
                    &mut constraints,
                    &mut the_sum,
                );

                // Fold P → P' at arity n_min
                p_combined.fold_in_place(&fold_randomness);
                let p_prime_ref = &p_combined;
                debug_assert_eq!(p_prime_ref.num_variables(), n_min);

                // Prover-side binding equation sanity check
                #[cfg(debug_assertions)]
                {
                    let mut expected = F::ZERO;
                    for (i, weight) in group.weights.iter().enumerate() {
                        if let Weights::Evaluation { ref point } = weight {
                            let a_low = MultilinearPoint(point.0[..n_min].to_vec());
                            let a_high = MultilinearPoint(point.0[n_min..].to_vec());
                            let eq_factor = a_high.eq_poly_outside(&fold_randomness);
                            let p_prime_eval = p_prime_ref.evaluate(&a_low);
                            expected += constraint_rlc[i] * eq_factor * p_prime_eval;
                        }
                    }
                    assert_eq!(the_sum, expected, "[PROVER] Binding equation mismatch");
                }
            } else {
                // No constraints — sample fold randomness directly
                let r: Vec<F> = (0..fold_depth)
                    .map(|_| prover_state.verifier_message())
                    .collect();
                level_config.prefold_pow.prove(prover_state);
                fold_randomness = MultilinearPoint(r);

                // Fold P → P' at arity n_min
                p_combined.fold_in_place(&fold_randomness);
                debug_assert_eq!(p_combined.num_variables(), n_min);
            };
            let p_prime = p_combined;

            // Send P' coefficients in the clear
            // P' = fold(ρ·f + g, r) is ZK-blinded by g, so revealing
            // coefficients does not leak information about f.
            // The coefficients are absorbed into the Fiat-Shamir transcript,
            // binding the prover to this specific P'.
            {
                let p_prime_coeffs = p_prime.coeffs();
                let num_coeffs = p_prime_coeffs.len();
                let base_field_size = (F::BasePrimeField::MODULUS_BIT_SIZE.div_ceil(8)) as usize;
                let elem_bytes = base_field_size * F::extension_degree() as usize;
                let mut encoded = Vec::with_capacity(num_coeffs * elem_bytes);
                for c in p_prime_coeffs {
                    crate::transcript::encode_field_element_into(c, &mut encoded);
                }
                prover_state.prover_messages_bytes::<F>(num_coeffs, &encoded);
            }

            // PoW
            level_config.prefold_pow.prove(prover_state);

            // Open f̂ at native arity → virtual oracle → fold → STIR consistency
            let f_hat_refs: Vec<_> = group.witness.f_hat_witnesses.iter().collect();
            let in_domain_base = level_config.f_hat_committer.open(prover_state, &f_hat_refs);

            // Prove helper evaluations for the virtual oracle at this level
            // (shared implementation with the native ZK prover)
            let domain = IrsDomainParams::from_irs_committer(&level_config.f_hat_committer);
            super::prover::prove_helper_evaluations(
                prover_state,
                &domain,
                &in_domain_base,
                group.witness,
                &level_config.helper_config,
                rho,
                self.embedding(),
            );
        }

        // ================================================================
        // Phase 3: Standard prove_zk on native polynomials
        // ================================================================
        // The prefold groups are fully proven by their sumcheck + STIR consistency.
        // The native group is proven by the standard ZK-WHIR protocol.
        self.prove_zk(
            prover_state,
            native_polys,
            native_witness,
            native_helper_config,
            native_weights,
            native_evals,
        )
    }
}

/// Commitments for a prefold group as seen by the verifier.
pub struct PrefoldGroupCommitments<F: FftField> {
    /// f̂ commitments at this level's arity.
    pub f_hat_commitments: Vec<irs_commit::Commitment<F>>,
    /// Helper commitment for this level.
    pub helper_commitment: irs_commit::Commitment<F>,
}

impl<F: FftField> Config<F> {
    /// Verify a ZK prefold + prove proof for mixed-arity polynomials.
    ///
    /// Mirrors `prove_zk_prefold` on the verifier side.
    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    pub fn verify_zk_prefold<H>(
        &self,
        verifier_state: &mut VerifierState<'_, H>,
        // Native group
        native_f_hat_commitments: &[&irs_commit::Commitment<F>],
        native_helper_commitment: &irs_commit::Commitment<F>,
        native_helper_config: &Config<F>,
        native_zk_params: &ZkParams,
        native_weights: &[&Weights<F>],
        native_evals: &[F],
        // Prefold groups (same order as prover: highest arity first)
        prefold_groups: &[(&PrefoldGroupCommitments<F>, &PrefoldLevelConfig<F>)],
        prefold_group_weights: &[&[&Weights<F>]],
        prefold_group_evals: &[&[F]],
        prefold_num_polys: &[usize],
    ) -> VerificationResult<(MultilinearPoint<F>, Vec<F>)>
    where
        H: DuplexSpongeInterface,
        F: Codec<[H::U]>,
        u8: Decoding<[H::U]>,
        [u8; 32]: Decoding<[H::U]>,
        U64: Codec<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        // ================================================================
        // Phase 1: Read β, g evals, ρ
        // ================================================================
        let beta: F = verifier_state.verifier_message();

        // Read native g evaluations
        let num_native = native_f_hat_commitments.len();
        let _native_g_evals: Vec<F> =
            verifier_state.prover_messages_vec(native_weights.len() * num_native)?;

        // Read prefold g evaluations
        let mut prefold_g_evals_per_group: Vec<Vec<F>> = Vec::new();
        for (group_idx, &num_polys) in prefold_num_polys.iter().enumerate() {
            let num_weights = prefold_group_weights[group_idx].len();
            let g_evals: Vec<F> = verifier_state.prover_messages_vec(num_weights * num_polys)?;
            prefold_g_evals_per_group.push(g_evals);
        }

        let rho: F = verifier_state.verifier_message();

        // ================================================================
        // Phase 2: Verify prefold stages
        // ================================================================
        let n_min = self.initial_num_variables();

        for (group_idx, (commitments, level_config)) in prefold_groups.iter().enumerate() {
            let weights = prefold_group_weights[group_idx];
            let evals = prefold_group_evals[group_idx];
            let num_polys = prefold_num_polys[group_idx];
            let fold_depth = level_config.fold_depth;

            // Modified evaluations for this group
            let modified_evals: Vec<F> =
                zip_strict(evals.iter(), prefold_g_evals_per_group[group_idx].iter())
                    .map(|(&eval, &g_eval)| rho * eval + g_eval)
                    .collect();

            // RLC
            let level_poly_rlc: Vec<F> = geometric_challenge(verifier_state, num_polys);

            // Prefold sumcheck or random fold
            let (fold_randomness, reduced_sum, constraint_rlc) = if !weights.is_empty() {
                let constraint_rlc: Vec<F> = geometric_challenge(verifier_state, weights.len());
                let mut the_sum: F =
                    zip_strict(&constraint_rlc, modified_evals.chunks_exact(num_polys))
                        .map(|(w, row)| *w * dot(&level_poly_rlc, row))
                        .sum();

                let fold_rand = level_config
                    .prefold_sumcheck
                    .verify(verifier_state, &mut the_sum)?;
                (fold_rand, Some(the_sum), Some(constraint_rlc))
            } else {
                let r: Vec<F> = (0..fold_depth)
                    .map(|_| verifier_state.verifier_message())
                    .collect();
                level_config.prefold_pow.verify(verifier_state)?;
                (MultilinearPoint(r), None, None)
            };

            // Read P' coefficients (sent in the clear by the prover)
            let p_prime_coeffs: Vec<F> = verifier_state.read_prover_messages_bytes(1 << n_min)?;
            let p_prime = CoefficientList::new(p_prime_coeffs);

            // Binding equation check
            // After the prefold sumcheck, the reduced sum v' must satisfy:
            //   v' = Σ_i constraint_rlc[i] · eq(a_i_high, r) · P'(a_i_low)
            // where a_i is the evaluation point for weight i, split into
            // a_i_high (first fold_depth components) and a_i_low (rest).
            if let (Some(v_prime), Some(ref c_rlc)) = (reduced_sum, &constraint_rlc) {
                let mut expected = F::ZERO;
                for (i, weight) in weights.iter().enumerate() {
                    let point = match weight {
                        Weights::Evaluation { point } => point,
                        _ => panic!(
                            "prefold binding equation requires Weights::Evaluation; \
                             got a non-evaluation weight at index {i}"
                        ),
                    };
                    // In MultilinearPoint, point[0] is the MSB (x_{L-1}).
                    // The fold eliminates the LSB variables x_0,...,x_{d-1}
                    // which are the LAST fold_depth elements: point[n_min..].
                    // The remaining n_min MSB variables are point[..n_min].
                    let a_low = MultilinearPoint(point.0[..n_min].to_vec());
                    let a_high = MultilinearPoint(point.0[n_min..].to_vec());
                    let eq_factor = a_high.eq_poly_outside(&fold_randomness);
                    let p_prime_eval = p_prime.evaluate(&a_low);
                    expected += c_rlc[i] * eq_factor * p_prime_eval;
                }
                verify!(v_prime == expected);
            }

            // PoW
            level_config.prefold_pow.verify(verifier_state)?;

            // Verify f̂ opening
            let f_hat_refs: Vec<&irs_commit::Commitment<F>> =
                commitments.f_hat_commitments.iter().collect();
            let in_domain_base = level_config
                .f_hat_committer
                .verify(verifier_state, &f_hat_refs)?;

            // Verify helper evaluations and reconstruct virtual oracle
            // (shared implementation with the native ZK verifier)
            let domain = IrsDomainParams::from_irs_committer(&level_config.f_hat_committer);
            let virtual_values = super::verifier::verify_helper_evaluations(
                verifier_state,
                &domain,
                &in_domain_base,
                &commitments.helper_commitment,
                &level_config.helper_config,
                &level_config.zk_params,
                rho,
                beta,
                &fold_randomness,
                num_polys,
                &level_poly_rlc,
                self.embedding(),
            )?;

            // ── STIR consistency check ──
            // The folded virtual oracle at each query point α must match
            // P' evaluated at the multilinear expansion of α.
            // Since we have P' coefficients, we evaluate directly (no hints).
            for (qi, &alpha_base) in in_domain_base.points.iter().enumerate() {
                let alpha_ext: F =
                    crate::algebra::embedding::Embedding::map(self.embedding(), alpha_base);
                let point = MultilinearPoint::expand_from_univariate(alpha_ext, n_min);
                let p_prime_at_point = p_prime.evaluate(&point);
                verify!(virtual_values[qi] == p_prime_at_point);
            }
        }

        // ================================================================
        // Phase 3: Verify main WHIR on native polynomials
        // ================================================================
        self.verify_zk(
            verifier_state,
            native_f_hat_commitments,
            native_helper_commitment,
            native_helper_config,
            native_zk_params,
            native_weights,
            native_evals,
        )
    }
}
