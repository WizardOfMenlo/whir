//! Unified API for ZK-WHIR batch proving with mixed-arity polynomial groups.
//!
//! The caller provides polynomials, weights, and evaluations grouped by arity.
//! The library handles all internal bookkeeping: config creation, preprocessing
//! sampling, commitment, and proof generation.
//!
//! # Example
//!
//! ```ignore
//! // Prover side
//! let groups = vec![
//!     ProverInput::new(vec![&poly_10var], weights_10, evals_10),
//!     ProverInput::new(vec![&poly_12var], weights_12, evals_12),
//! ];
//! let (point, evals) = main_config.batch_prove_zk(
//!     &mut prover_state, &whir_params, &groups, &mut rng,
//! );
//!
//! // Verifier side
//! let claims: Vec<VerifierInput<_>> = groups.iter().map(|g| g.to_verifier_input()).collect();
//! let result = main_config.batch_verify_zk(
//!     &mut verifier_state, &whir_params, &claims,
//! );
//! ```

use ark_ff::FftField;
use ark_std::rand::{CryptoRng, RngCore};

use crate::{
    algebra::{
        fields::FieldWithSize,
        polynomials::{CoefficientList, MultilinearPoint},
        Weights,
    },
    hash::Hash,
    parameters::ProtocolParameters,
    protocols::whir::Config,
    transcript::{
        codecs::U64, Codec, Decoding, DuplexSpongeInterface, ProverMessage, ProverState,
        VerificationResult, VerifierState,
    },
};

use super::{
    prefold::{
        commit_zk_at_level, receive_prefold_commitments, PrefoldGroupCommitments,
        PrefoldGroupInput, PrefoldLevelConfig,
    },
    utils::{ZkParams, ZkPreprocessingPolynomials, ZkWitness},
};

/// Prover input: a group of polynomials at the same arity with shared constraints.
///
/// All polynomials in a group must have the same number of variables (arity).
/// The library automatically determines which groups are "native" (minimum arity)
/// and which need prefolding.
pub struct ProverInput<'a, EF: FftField> {
    /// Base-field polynomials (all at the same arity).
    pub polynomials: Vec<&'a CoefficientList<EF::BasePrimeField>>,
    /// Shared evaluation constraint weights.
    pub weights: Vec<Weights<EF>>,
    /// Evaluations: row-major `[w₀_p₀, w₀_p₁, ..., w₁_p₀, ...]`.
    pub evaluations: Vec<EF>,
}

/// Verifier input: a claim about a group of polynomials at the same arity.
///
/// Same layout as [`ProverInput`] but without the actual polynomial coefficients —
/// the verifier only needs the arity, group size, and the claimed evaluations.
pub struct VerifierInput<EF: FftField> {
    /// Number of variables (arity) for this group.
    pub arity: usize,
    /// Number of polynomials in this group.
    pub num_polynomials: usize,
    /// Shared evaluation constraint weights.
    pub weights: Vec<Weights<EF>>,
    /// Evaluations: row-major `[w₀_p₀, w₀_p₁, ..., w₁_p₀, ...]`.
    pub evaluations: Vec<EF>,
}

impl<'a, EF: FftField> ProverInput<'a, EF> {
    /// Create a new prover input group.
    pub fn new(
        polynomials: Vec<&'a CoefficientList<EF::BasePrimeField>>,
        weights: Vec<Weights<EF>>,
        evaluations: Vec<EF>,
    ) -> Self {
        assert!(
            !polynomials.is_empty(),
            "ProverInput must have at least one polynomial"
        );
        let arity = polynomials[0].num_variables();
        debug_assert!(
            polynomials.iter().all(|p| p.num_variables() == arity),
            "All polynomials in a ProverInput must have the same arity"
        );
        debug_assert_eq!(
            evaluations.len(),
            weights.len() * polynomials.len(),
            "evaluations.len() must equal weights.len() × polynomials.len()"
        );
        Self {
            polynomials,
            weights,
            evaluations,
        }
    }

    /// Infer the arity (number of variables) from the polynomials.
    pub fn arity(&self) -> usize {
        self.polynomials[0].num_variables()
    }

    /// Number of polynomials in this group.
    pub fn num_polynomials(&self) -> usize {
        self.polynomials.len()
    }

    /// Create a corresponding [`VerifierInput`] from this prover input.
    pub fn to_verifier_input(&self) -> VerifierInput<EF> {
        VerifierInput {
            arity: self.arity(),
            num_polynomials: self.polynomials.len(),
            weights: self.weights.clone(),
            evaluations: self.evaluations.clone(),
        }
    }
}

impl<EF: FftField> VerifierInput<EF> {
    /// Create a new verifier input claim.
    pub fn new(
        arity: usize,
        num_polynomials: usize,
        weights: Vec<Weights<EF>>,
        evaluations: Vec<EF>,
    ) -> Self {
        debug_assert_eq!(
            evaluations.len(),
            weights.len() * num_polynomials,
            "evaluations.len() must equal weights.len() × num_polynomials"
        );
        Self {
            arity,
            num_polynomials,
            weights,
            evaluations,
        }
    }
}

/// Separate groups into native (arity == n_min) and prefold (arity > n_min).
///
/// Returns `(native_index, prefold_indices)` where prefold indices are sorted
/// by decreasing arity (highest first), matching the prefold proof order.
fn separate_by_arity(n_min: usize, arities: &[usize]) -> (usize, Vec<usize>) {
    let mut native_idx: Option<usize> = None;
    let mut prefold_indices: Vec<usize> = Vec::new();

    for (i, &arity) in arities.iter().enumerate() {
        if arity == n_min {
            assert!(
                native_idx.is_none(),
                "only one native group (arity == n_min = {n_min}) allowed"
            );
            native_idx = Some(i);
        } else {
            assert!(
                arity > n_min,
                "group arity ({arity}) must be >= n_min ({n_min})"
            );
            prefold_indices.push(i);
        }
    }

    let native_idx = native_idx.expect("must have exactly one group at the native arity (n_min)");
    prefold_indices.sort_by(|&a, &b| arities[b].cmp(&arities[a]));

    (native_idx, prefold_indices)
}

impl<F: FftField + FieldWithSize> Config<F> {
    /// Unified ZK-WHIR batch proof for mixed-arity polynomial groups.
    ///
    /// The caller provides polynomial groups (each at its own arity, with its own
    /// constraints and evaluations). The library automatically:
    ///
    /// 1. Identifies the minimum arity (native) group.
    /// 2. Creates prefold configs for any higher-arity groups.
    /// 3. Samples ZK preprocessing (blinding polynomials) for every polynomial.
    /// 4. Commits all polynomials (native + prefold).
    /// 5. Runs the prefold + ZK-WHIR proof pipeline.
    ///
    /// If all groups are at the same arity (no prefold needed), falls back to
    /// the standard `prove_zk` path.
    ///
    /// # Arguments
    ///
    /// * `self` — Main WHIR config (must be configured at the minimum arity).
    /// * `prover_state` — Fiat-Shamir prover transcript.
    /// * `whir_params` — Protocol parameters (used to build prefold sub-configs).
    /// * `groups` — Polynomial groups at arbitrary arities.
    /// * `rng` — RNG for sampling ZK preprocessing polynomials.
    #[allow(clippy::too_many_lines)]
    pub fn batch_prove_zk<H, R, R2>(
        &self,
        prover_state: &mut ProverState<H, R>,
        whir_params: &ProtocolParameters,
        groups: &[ProverInput<'_, F>],
        rng: &mut R2,
    ) -> (MultilinearPoint<F>, Vec<F>)
    where
        H: DuplexSpongeInterface<U = u8>,
        R: RngCore + CryptoRng,
        R2: RngCore + CryptoRng,
        F: Codec<[H::U]>,
        [u8; 32]: Decoding<[H::U]>,
        U64: Codec<[H::U]>,
        u8: Decoding<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        assert!(!groups.is_empty(), "must provide at least one group");

        let n_min = self.initial_num_variables();

        // ── Separate groups into native and prefold ──
        let arities: Vec<usize> = groups.iter().map(|g| g.arity()).collect();
        let (native_idx, prefold_indices) = separate_by_arity(n_min, &arities);

        let native_group = &groups[native_idx];
        let num_native = native_group.num_polynomials();

        // Native: ZK params, helper config, preprocessings
        let native_zk_params = ZkParams::from_whir_params(self);
        let native_helper_config =
            native_zk_params.build_helper_config::<F>(num_native, whir_params);
        let native_preprocessings: Vec<ZkPreprocessingPolynomials<F>> = (0..num_native)
            .map(|_| ZkPreprocessingPolynomials::sample(rng, native_zk_params.clone()))
            .collect();
        let native_preproc_refs: Vec<&ZkPreprocessingPolynomials<F>> =
            native_preprocessings.iter().collect();

        // Commit native
        let native_witness = self.commit_zk(
            prover_state,
            &native_group.polynomials,
            &native_helper_config,
            &native_preproc_refs,
        );

        // Fast path: no prefold groups → standard prove_zk
        if prefold_indices.is_empty() {
            let native_weight_refs: Vec<&Weights<F>> = native_group.weights.iter().collect();
            return self.prove_zk(
                prover_state,
                &native_group.polynomials,
                &native_witness,
                &native_helper_config,
                &native_weight_refs,
                &native_group.evaluations,
            );
        }

        // Prefold groups: configs, preprocessings, commitments
        let mut prefold_configs: Vec<PrefoldLevelConfig<F>> = Vec::new();
        let mut prefold_witnesses: Vec<ZkWitness<F>> = Vec::new();

        for &gi in &prefold_indices {
            let group = &groups[gi];
            let arity = group.arity();
            let num_polys = group.num_polynomials();

            let level_config = PrefoldLevelConfig::new(self, arity, whir_params);
            let preprocs: Vec<ZkPreprocessingPolynomials<F>> = (0..num_polys)
                .map(|_| ZkPreprocessingPolynomials::sample(rng, level_config.zk_params.clone()))
                .collect();
            let preproc_refs: Vec<&ZkPreprocessingPolynomials<F>> = preprocs.iter().collect();

            let witness = commit_zk_at_level(
                &level_config,
                prover_state,
                &group.polynomials,
                &preproc_refs,
            );

            prefold_configs.push(level_config);
            prefold_witnesses.push(witness);
        }

        // Build PrefoldGroupInputs
        let prefold_weight_refs: Vec<Vec<&Weights<F>>> = prefold_indices
            .iter()
            .map(|&gi| groups[gi].weights.iter().collect())
            .collect();

        let prefold_group_inputs: Vec<PrefoldGroupInput<'_, F>> = prefold_indices
            .iter()
            .enumerate()
            .map(|(ci, &gi)| PrefoldGroupInput {
                polynomials: &groups[gi].polynomials,
                witness: &prefold_witnesses[ci],
                weights: &prefold_weight_refs[ci],
                evaluations: &groups[gi].evaluations,
                level_config: &prefold_configs[ci],
            })
            .collect();

        // Prove
        let native_weight_refs: Vec<&Weights<F>> = native_group.weights.iter().collect();

        self.prove_zk_prefold(
            prover_state,
            &native_group.polynomials,
            &native_witness,
            &native_helper_config,
            &native_weight_refs,
            &native_group.evaluations,
            &prefold_group_inputs,
        )
    }

    /// Unified ZK-WHIR batch verification for mixed-arity polynomial groups.
    ///
    /// Mirrors [`batch_prove_zk`](Self::batch_prove_zk): the verifier provides
    /// the same group structure (arity, number of polynomials, weights, evaluations)
    /// and the library handles all config re-creation, commitment reception, and
    /// verification routing.
    ///
    /// If all groups are at the same arity (no prefold), falls back to `verify_zk`.
    #[allow(clippy::too_many_lines)]
    pub fn batch_verify_zk<H>(
        &self,
        verifier_state: &mut VerifierState<'_, H>,
        whir_params: &ProtocolParameters,
        claims: &[VerifierInput<F>],
    ) -> VerificationResult<(MultilinearPoint<F>, Vec<F>)>
    where
        H: DuplexSpongeInterface,
        F: Codec<[H::U]>,
        u8: Decoding<[H::U]>,
        [u8; 32]: Decoding<[H::U]>,
        U64: Codec<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        assert!(!claims.is_empty(), "must provide at least one claim");

        let n_min = self.initial_num_variables();

        // Separate into native and prefold
        let arities: Vec<usize> = claims.iter().map(|c| c.arity).collect();
        let (native_idx, prefold_indices) = separate_by_arity(n_min, &arities);

        let native_claim = &claims[native_idx];
        let num_native = native_claim.num_polynomials;

        // Native: ZK params, helper config
        let native_zk_params = ZkParams::from_whir_params(self);
        let native_helper_config =
            native_zk_params.build_helper_config::<F>(num_native, whir_params);

        // Receive native commitments
        let native_f_hat_comms: Vec<_> = (0..num_native)
            .map(|_| self.receive_commitment(verifier_state))
            .collect::<Result<_, _>>()?;
        let native_f_hat_comm_refs: Vec<_> = native_f_hat_comms.iter().collect();
        let native_helper_comm = native_helper_config.receive_commitment(verifier_state)?;

        let native_weight_refs: Vec<&Weights<F>> = native_claim.weights.iter().collect();

        // Fast path: no prefold → standard verify_zk
        if prefold_indices.is_empty() {
            return self.verify_zk(
                verifier_state,
                &native_f_hat_comm_refs,
                &native_helper_comm,
                &native_helper_config,
                &native_zk_params,
                &native_weight_refs,
                &native_claim.evaluations,
            );
        }

        // Prefold configs
        let prefold_configs: Vec<PrefoldLevelConfig<F>> = prefold_indices
            .iter()
            .map(|&i| PrefoldLevelConfig::new(self, claims[i].arity, whir_params))
            .collect();

        // Receive prefold commitments
        let mut prefold_group_commitments: Vec<PrefoldGroupCommitments<F>> = Vec::new();
        for (ci, &gi) in prefold_indices.iter().enumerate() {
            let (f_hat_comms, helper_comm) = receive_prefold_commitments(
                &prefold_configs[ci],
                verifier_state,
                claims[gi].num_polynomials,
            )?;
            prefold_group_commitments.push(PrefoldGroupCommitments {
                f_hat_commitments: f_hat_comms,
                helper_commitment: helper_comm,
            });
        }

        // Build verify arguments
        let prefold_groups_verify: Vec<(&PrefoldGroupCommitments<F>, &PrefoldLevelConfig<F>)> =
            prefold_group_commitments
                .iter()
                .zip(prefold_configs.iter())
                .collect();

        let prefold_weight_refs: Vec<Vec<&Weights<F>>> = prefold_indices
            .iter()
            .map(|&gi| claims[gi].weights.iter().collect())
            .collect();
        let prefold_weight_slices: Vec<&[&Weights<F>]> =
            prefold_weight_refs.iter().map(|v| v.as_slice()).collect();

        let prefold_eval_slices: Vec<&[F]> = prefold_indices
            .iter()
            .map(|&gi| claims[gi].evaluations.as_slice())
            .collect();

        let prefold_num_polys: Vec<usize> = prefold_indices
            .iter()
            .map(|&gi| claims[gi].num_polynomials)
            .collect();

        self.verify_zk_prefold(
            verifier_state,
            &native_f_hat_comm_refs,
            &native_helper_comm,
            &native_helper_config,
            &native_zk_params,
            &native_weight_refs,
            &native_claim.evaluations,
            &prefold_groups_verify,
            &prefold_weight_slices,
            &prefold_eval_slices,
            &prefold_num_polys,
        )
    }
}
