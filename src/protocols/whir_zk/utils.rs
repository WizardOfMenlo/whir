use ark_ff::{FftField, Field};
use ark_std::{
    rand::{CryptoRng, RngCore},
    UniformRand,
};

use crate::{
    algebra::{
        embedding::Embedding,
        ntt,
        polynomials::{CoefficientList, EvaluationsList, MultilinearPoint},
        Weights,
    },
    protocols::{irs_commit, whir::Config},
};

// ── IRS Domain Parameters (shared by prover & verifier) ──────────────────

/// Precomputed IRS domain structure for the initial commitment.
///
/// Both prover and verifier need the same domain generators, coset roots, and
/// sub-domain powers to compute gamma query points. This struct deduplicates
/// that computation.
pub(crate) struct IrsDomainParams<F: FftField> {
    /// Interleaving depth k (= 2^folding_factor for first round)
    pub k: usize,
    /// Generator ω of the full NTT domain of size num_rows × k
    pub omega_full: F::BasePrimeField,
    /// k-th root of unity ζ = ω^num_rows
    pub zeta: F::BasePrimeField,
    /// Precomputed sub-domain powers [1, ω_sub, ω_sub², ..., ω_sub^(num_rows-1)]
    pub omega_powers: Vec<F::BasePrimeField>,
}

impl<F: FftField> IrsDomainParams<F> {
    /// Compute domain parameters from a WHIR config's initial committer.
    pub fn from_config(config: &Config<F>) -> Self {
        let k = config.initial_committer.interleaving_depth;
        let num_rows = config.initial_committer.num_rows();
        let full_domain_size = num_rows * k;

        let omega_full: F::BasePrimeField =
            ntt::generator(full_domain_size).expect("Full IRS domain should have primitive root");
        let omega_sub: F::BasePrimeField = config.initial_committer.generator();
        let zeta: F::BasePrimeField = omega_full.pow([num_rows as u64]);

        let omega_powers = crate::algebra::geometric_sequence(omega_sub, num_rows);

        Self {
            k,
            omega_full,
            zeta,
            omega_powers,
        }
    }

    /// Find the index of `alpha_base` in the sub-domain.
    #[inline]
    pub fn query_index(&self, alpha_base: F::BasePrimeField) -> usize {
        self.omega_powers
            .iter()
            .position(|&p| p == alpha_base)
            .expect("Query point must be in IRS domain")
    }

    /// Compute the k coset gamma points for a query at `alpha_base`,
    /// lifted to the extension field via `embedding`.
    pub fn coset_gammas<M: Embedding<Source = F::BasePrimeField, Target = F>>(
        &self,
        alpha_base: F::BasePrimeField,
        embedding: &M,
    ) -> Vec<F> {
        let idx = self.query_index(alpha_base);
        let coset_offset = self.omega_full.pow([idx as u64]);
        (0..self.k)
            .map(|j| {
                let gamma_base = coset_offset * self.zeta.pow([j as u64]);
                embedding.map(gamma_base)
            })
            .collect()
    }

    /// Compute all gamma points for a set of query points (flat list).
    pub fn all_gammas<M: Embedding<Source = F::BasePrimeField, Target = F>>(
        &self,
        query_points: &[F::BasePrimeField],
        embedding: &M,
    ) -> Vec<F> {
        query_points
            .iter()
            .flat_map(|&alpha| self.coset_gammas(alpha, embedding))
            .collect()
    }
}

#[derive(Clone)]
pub struct ZkParams {
    /// ℓ: Number of variables for helper polynomials
    /// Chosen such that 2^ℓ > conservative query upper bound
    pub ell: usize,

    /// μ: Number of variables in the witness polynomial
    pub mu: usize,
}

impl ZkParams {
    /// Compute ell and mu from WHIR parameters.
    pub fn from_whir_params<F: FftField>(whir_params: &Config<F>) -> Self {
        // mu = number of variables (log2 of polynomial size)
        let mu = whir_params.initial_sumcheck.initial_size.ilog2() as usize;
        // k = folding factor size (2^folding_factor)
        let k = 1 << whir_params.initial_sumcheck.num_rounds;
        // q1 = number of in-domain query samples in the first round
        //      (or initial commitment queries if there are no rounds)
        let q1 = whir_params
            .round_configs
            .first()
            .map_or(whir_params.initial_committer.in_domain_samples, |r| {
                r.irs_committer.in_domain_samples
            });

        let q_ub = 2 * k * q1 + 4 * mu + 10;
        let ell = (q_ub as f64).log2().ceil() as usize;
        assert!(
            ell < mu,
            "ZK requires ℓ < μ (ℓ={ell}, μ={mu}). \
             Increase num_variables or lower security_level/queries. \
             (q_ub={q_ub}, k={k}, q1={q1})"
        );
        Self { ell, mu }
    }

    pub fn helper_batch_size(&self, number_of_polynomials: usize) -> usize {
        number_of_polynomials * (self.mu + 1)
    }
}

/// Sampling random polynomials before the witness polynomial
#[derive(Clone)]
pub struct ZkPreprocessingPolynomials<F: FftField> {
    pub msk: CoefficientList<F>,
    pub g0_hat: CoefficientList<F>,
    pub m_poly: CoefficientList<F>,
    pub g_hats: Vec<CoefficientList<F>>,
    pub params: ZkParams,
}

impl<F: FftField> ZkPreprocessingPolynomials<F> {
    pub fn sample<R: RngCore + CryptoRng>(rng: &mut R, params: ZkParams) -> Self {
        let poly_size = 1 << params.ell;
        let m_poly_size = 1 << (params.ell + 1);

        // Sample all preprocessing polynomials from the BASE FIELD, then lift to extension.
        // This is required because these polynomials are committed via base-field IRS commitment,
        // and the conversion back to base field (to_base_prime_field_elements().next()) must be
        // lossless.
        let msk_coeffs: Vec<F> = (0..poly_size)
            .map(|_| F::from_base_prime_field(F::BasePrimeField::rand(rng)))
            .collect();
        let msk = CoefficientList::new(msk_coeffs.clone());

        let g0_coeffs: Vec<F> = (0..poly_size)
            .map(|_| F::from_base_prime_field(F::BasePrimeField::rand(rng)))
            .collect();
        let g0 = CoefficientList::new(g0_coeffs.clone());

        let mut m_coeffs = vec![F::ZERO; m_poly_size];
        for (i, (g0_c, &msk_c)) in g0_coeffs.iter().zip(msk_coeffs.iter()).enumerate() {
            m_coeffs[2 * i] = *g0_c;
            m_coeffs[2 * i + 1] = msk_c;
        }
        let m_poly = CoefficientList::new(m_coeffs);

        let g_hats = (0..params.mu)
            .map(|_| {
                let coeffs = (0..poly_size)
                    .map(|_| F::from_base_prime_field(F::BasePrimeField::rand(rng)))
                    .collect();
                CoefficientList::new(coeffs)
            })
            .collect();

        Self {
            msk,
            g0_hat: g0,
            m_poly,
            g_hats,
            params,
        }
    }

    /// Extend msk to μ variables by padding with zeros
    pub fn extend_msk(&self) -> CoefficientList<F> {
        let target_size = 1 << self.params.mu;
        let mut coeffs = self.msk.coeffs().to_vec();
        coeffs.resize(target_size, F::ZERO);
        CoefficientList::new(coeffs)
    }

    /// Batch-evaluate all helper polynomials at multiple gamma points using
    /// fused univariate Horner evaluation.
    ///
    /// For each gamma, evaluates msk, g₀, and all ĝⱼ in a single pass per gamma
    /// point, avoiding intermediate per-polynomial allocation vectors.
    ///
    /// Returns a Vec of `HelperEvaluations` (one per gamma point), in the same
    /// order as the input gammas.
    pub fn batch_evaluate_helpers(&self, gammas: &[F], rho: F) -> Vec<HelperEvaluations<F>> {
        use crate::algebra::univariate_evaluate;

        // Evaluate all helper polynomials at a single gamma point.
        // This fuses msk, g₀, and ĝⱼ evaluations per-gamma, avoiding
        // μ+2 intermediate Vec<F> allocations of size |gammas|.
        let eval_at = |&gamma: &F| -> HelperEvaluations<F> {
            let msk_val = univariate_evaluate(self.msk.coeffs(), gamma);
            let g0_val = univariate_evaluate(self.g0_hat.coeffs(), gamma);
            let m_eval = g0_val - rho * msk_val;
            let g_hat_evals = self
                .g_hats
                .iter()
                .map(|g_hat| univariate_evaluate(g_hat.coeffs(), gamma))
                .collect();
            HelperEvaluations {
                gamma,
                m_eval,
                g_hat_evals,
            }
        };

        // Parallelize across gamma points (typically q×k, often hundreds).
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            gammas.par_iter().map(eval_at).collect()
        }
        #[cfg(not(feature = "parallel"))]
        {
            gammas.iter().map(eval_at).collect()
        }
    }
}

/// ZK Witness: contains commitment witnesses for all ZK components
#[derive(Clone)]
pub struct ZkWitness<F: FftField> {
    /// Witnesses for [[f̂₁]] = [[f₁ + msk₁]], ..., [[fₙ]] = [[fₙ + mskₙ]] in main WHIR
    pub f_hat_witnesses: Vec<irs_commit::Witness<F::BasePrimeField, F>>,

    /// Single batch witness for all helper polynomials [[M, ĝ₁, ..., ĝμ]]
    /// committed via helper_config with batch_size = μ+1
    pub helper_witness: irs_commit::Witness<F::BasePrimeField, F>,

    /// Reference to preprocessing data for each polynomial
    pub preprocessings: Vec<ZkPreprocessingPolynomials<F>>,

    /// Base-field representations of M polynomials (for helper WHIR prove)
    pub m_polys_base: Vec<CoefficientList<F::BasePrimeField>>,

    /// Base-field representations of embedded ĝⱼ polynomials (for helper WHIR prove)
    /// Each ĝⱼ is embedded from ℓ-variate to (ℓ+1)-variate for each polynomial
    pub g_hats_embedded_bases: Vec<Vec<CoefficientList<F::BasePrimeField>>>,
}

/// Collect interleaved references to all helper polynomials in batch order:
/// `[M₁, ĝ₁₁, …, ĝ₁μ, M₂, ĝ₂₁, …, ĝ₂μ, …]`
///
/// Used by both committer (batch-commit) and prover (batch-prove).
pub(crate) fn interleave_helper_poly_refs<'a, F: FftField>(
    m_polys: &'a [CoefficientList<F::BasePrimeField>],
    g_hats: &'a [Vec<CoefficientList<F::BasePrimeField>>],
) -> Vec<&'a CoefficientList<F::BasePrimeField>> {
    let num_polys = m_polys.len();
    let mu = g_hats.first().map_or(0, |g| g.len());
    let mut refs = Vec::with_capacity(num_polys * (1 + mu));
    for (m_poly, g_hat_list) in m_polys.iter().zip(g_hats) {
        refs.push(m_poly);
        for g_hat in g_hat_list {
            refs.push(g_hat);
        }
    }
    refs
}

/// Compute per-polynomial claims from helper evaluations.
///
/// m_claim = Σᵢ τ₂ⁱ · m(γᵢ, ρ)
/// g_hat_j_claim = Σᵢ τ₂ⁱ · ĝⱼ(pow(γᵢ))
pub(crate) fn compute_per_polynomial_claims<F: FftField>(
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
pub(crate) fn construct_batched_eq_weights<F: FftField>(
    helper_evals: &[HelperEvaluations<F>],
    rho: F,
    tau2: F,
    ell: usize,
) -> Weights<F> {
    let neg_rho = -rho;
    let z_size = 1 << ell;
    let weight_size = 1 << (ell + 1);

    // Precompute τ₂ powers
    let tau2_powers = crate::algebra::geometric_sequence(tau2, helper_evals.len());

    // For each γᵢ, compute eq(pow(γᵢ), z) for all z ∈ {0,1}^ℓ using the
    // O(2^ℓ) butterfly expansion in MultilinearPoint::eq_weights(), then
    // accumulate τ₂ⁱ · eq(pow(γᵢ), ·) into a single batched_eq vector.
    #[cfg(feature = "parallel")]
    let batched_eq: Vec<F> = {
        use rayon::prelude::*;
        helper_evals
            .par_iter()
            .zip(tau2_powers.par_iter())
            .fold(
                || vec![F::ZERO; z_size],
                |mut acc, (helper, &tau2_pow)| {
                    let eq_vals =
                        MultilinearPoint::expand_from_univariate(helper.gamma, ell).eq_weights();
                    for (a, v) in acc.iter_mut().zip(eq_vals) {
                        *a += tau2_pow * v;
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
        for (helper, &tau2_pow) in helper_evals.iter().zip(tau2_powers.iter()) {
            let eq_vals = MultilinearPoint::expand_from_univariate(helper.gamma, ell).eq_weights();
            for (a, &v) in batched.iter_mut().zip(eq_vals.iter()) {
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

/// Helper evaluations at a single query point γ
#[derive(Clone, Debug)]
pub struct HelperEvaluations<F> {
    /// The query point γ
    pub gamma: F,

    /// m(γ,ρ) = M(pow(γ), -ρ)
    pub m_eval: F,

    /// [ĝ₁(pow(γ)), ..., ĝμ(pow(γ))]
    pub g_hat_evals: Vec<F>,
}

impl<F: FftField> HelperEvaluations<F> {
    /// Compute the helper polynomial value h(γ) (without the ρ·f̂ term).
    ///
    /// h(γ) = m(γ,ρ) + Σᵢ βⁱ·γ^(2^(i-1))·ĝᵢ(pow(γ))
    pub fn compute_h_value(&self, beta: F) -> F {
        let mut value = self.m_eval;

        let mut beta_power = beta;
        let mut gamma_power = self.gamma;

        for (i, &g_hat_eval) in self.g_hat_evals.iter().enumerate() {
            value += beta_power * gamma_power * g_hat_eval;

            beta_power *= beta;
            if i < self.g_hat_evals.len() - 1 {
                gamma_power = gamma_power.square();
            }
        }

        value
    }

    /// Compute the full virtual oracle value L(γ) = ρ·f̂(γ) + h(γ)
    ///
    /// L(γ) = ρ·f̂(γ) + m(γ,ρ) + Σᵢ βⁱ·γ^(2^(i-1))·ĝᵢ(pow(γ))
    ///      = ρ·(f + msk)(γ) + (ĝ₀ - ρ·msk)(pow(γ)) + blinding_terms
    ///      = ρ·f(γ) + g(γ)
    pub fn compute_virtual_value(&self, f_hat_val: F, rho: F, beta: F) -> F {
        rho * f_hat_val + self.compute_h_value(beta)
    }
}
