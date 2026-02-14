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
    protocols::whir,
};

// ── IRS Domain Parameters (shared by prover & verifier) ──────────────────

/// Precomputed IRS domain structure for the initial commitment.
///
/// Both prover and verifier need the same domain generators, coset roots, and
/// sub-domain powers to compute gamma query points. This struct deduplicates
/// that computation.
pub(crate) struct IrsDomainParams<F: FftField> {
    /// Interleaving depth (= 2^folding_factor for first round)
    pub interleaving_depth: usize,
    /// Generator ω of the full NTT domain of size num_rows × interleaving_depth
    pub omega_full: F::BasePrimeField,
    /// interleaving_depth-th root of unity ζ = ω^num_rows
    pub zeta: F::BasePrimeField,
    /// Precomputed sub-domain powers [1, ω_sub, ω_sub², ..., ω_sub^(num_rows-1)]
    pub omega_powers: Vec<F::BasePrimeField>,
}

impl<F: FftField> IrsDomainParams<F> {
    /// Compute domain parameters from a WHIR config's initial committer.
    pub fn from_config(config: &whir::Config<F>) -> Self {
        let interleaving_depth = config.initial_committer.interleaving_depth;
        let num_rows = config.initial_committer.num_rows();
        let full_domain_size = num_rows * interleaving_depth;

        let omega_full: F::BasePrimeField =
            ntt::generator(full_domain_size).expect("Full IRS domain should have primitive root");
        let omega_sub: F::BasePrimeField = config.initial_committer.generator();
        let zeta: F::BasePrimeField = omega_full.pow([num_rows as u64]);

        let omega_powers = crate::algebra::geometric_sequence(omega_sub, num_rows);

        Self {
            interleaving_depth,
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
        (0..self.interleaving_depth)
            .map(|coset_elem_idx| {
                let gamma_base = coset_offset * self.zeta.pow([coset_elem_idx as u64]);
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

/// Random blinding polynomials sampled before the witness polynomial.
///
/// For each committed polynomial, these provide the ZK blinding:
/// msk (masking), g₀ (initial blinding), M (combined), and ĝ₁..ĝμ (per-round blinding).
#[derive(Clone)]
pub struct BlindingPolynomials<F: FftField> {
    pub msk: CoefficientList<F>,
    pub g0_hat: CoefficientList<F>,
    pub m_poly: CoefficientList<F>,
    pub g_hats: Vec<CoefficientList<F>>,
}

impl<F: FftField> BlindingPolynomials<F> {
    pub fn sample<R: RngCore + CryptoRng>(
        rng: &mut R,
        num_blinding_variables: usize,
        num_witness_variables: usize,
    ) -> Self {
        let blinding_poly_size = 1 << num_blinding_variables;
        let m_poly_size = 1 << (num_blinding_variables + 1);

        // Sample all preprocessing polynomials from the BASE FIELD, then lift to extension.
        // This is required because these polynomials are committed via base-field IRS commitment,
        // and the conversion back to base field (to_base_prime_field_elements().next()) must be
        // lossless.
        let msk_coeffs: Vec<F> = (0..blinding_poly_size)
            .map(|_| F::from_base_prime_field(F::BasePrimeField::rand(rng)))
            .collect();
        let msk = CoefficientList::new(msk_coeffs.clone());

        let g0_coeffs: Vec<F> = (0..blinding_poly_size)
            .map(|_| F::from_base_prime_field(F::BasePrimeField::rand(rng)))
            .collect();
        let g0 = CoefficientList::new(g0_coeffs.clone());

        let mut m_coeffs = vec![F::ZERO; m_poly_size];
        for (coeff_idx, (g0_coeff, &msk_coeff)) in
            g0_coeffs.iter().zip(msk_coeffs.iter()).enumerate()
        {
            m_coeffs[2 * coeff_idx] = *g0_coeff;
            m_coeffs[2 * coeff_idx + 1] = msk_coeff;
        }
        let m_poly = CoefficientList::new(m_coeffs);

        let g_hats = (0..num_witness_variables)
            .map(|_| {
                let coeffs = (0..blinding_poly_size)
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
        }
    }

    /// Batch-evaluate all blinding polynomials at multiple gamma points using
    /// fused univariate Horner evaluation.
    ///
    /// For each gamma, evaluates msk, g₀, and all ĝⱼ in a single pass per gamma
    /// point, avoiding intermediate per-polynomial allocation vectors.
    ///
    /// Returns a Vec of `BlindingEvaluations` (one per gamma point), in the same
    /// order as the input gammas.
    pub fn batch_evaluate(
        &self,
        gammas: &[F],
        masking_challenge: F,
    ) -> Vec<BlindingEvaluations<F>> {
        use crate::algebra::univariate_evaluate;

        // Evaluate all blinding polynomials at a single gamma point.
        // This fuses msk, g₀, and ĝⱼ evaluations per-gamma, avoiding
        // μ+2 intermediate Vec<F> allocations of size |gammas|.
        let eval_at = |&gamma: &F| -> BlindingEvaluations<F> {
            let msk_val = univariate_evaluate(self.msk.coeffs(), gamma);
            let g0_val = univariate_evaluate(self.g0_hat.coeffs(), gamma);
            let m_eval = g0_val - masking_challenge * msk_val;
            let g_hat_evals = self
                .g_hats
                .iter()
                .map(|g_hat| univariate_evaluate(g_hat.coeffs(), gamma))
                .collect();
            BlindingEvaluations {
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

/// Collect interleaved references to all blinding polynomials in batch order:
/// `[M₁, ĝ₁₁, …, ĝ₁μ, M₂, ĝ₂₁, …, ĝ₂μ, …]`
///
/// Used by both committer (batch-commit) and prover (batch-prove).
pub(crate) fn interleave_blinding_poly_refs<'a, F: FftField>(
    m_polys: &'a [CoefficientList<F::BasePrimeField>],
    g_hats: &'a [Vec<CoefficientList<F::BasePrimeField>>],
) -> Vec<&'a CoefficientList<F::BasePrimeField>> {
    let num_polys = m_polys.len();
    let num_witness_vars = g_hats.first().map_or(0, |g_hat_list| g_hat_list.len());
    let mut refs = Vec::with_capacity(num_polys * (1 + num_witness_vars));
    for (m_poly, g_hat_list) in m_polys.iter().zip(g_hats) {
        refs.push(m_poly);
        for g_hat in g_hat_list {
            refs.push(g_hat);
        }
    }
    refs
}

/// Compute per-polynomial claims from blinding evaluations.
///
/// m_claim = Σᵢ τ₂ⁱ · m(γᵢ, ρ)
/// g_hat_j_claim = Σᵢ τ₂ⁱ · ĝⱼ(pow(γᵢ))
pub(crate) fn compute_per_polynomial_claims<F: FftField>(
    blinding_evals: &[BlindingEvaluations<F>],
    query_batching_challenge: F,
) -> (F, Vec<F>) {
    let num_g_hats = blinding_evals
        .first()
        .map_or(0, |eval| eval.g_hat_evals.len());

    let mut m_claim = F::ZERO;
    let mut g_hat_claims = vec![F::ZERO; num_g_hats];
    let mut batching_power = F::ONE;

    for eval in blinding_evals {
        m_claim += batching_power * eval.m_eval;
        for (g_hat_idx, &g_hat_eval) in eval.g_hat_evals.iter().enumerate() {
            g_hat_claims[g_hat_idx] += batching_power * g_hat_eval;
        }
        batching_power *= query_batching_challenge;
    }

    (m_claim, g_hat_claims)
}

/// Construct the weight function for the blinding polynomial WHIR sumcheck:
///
///   w(z, t) = eq(-masking_challenge, t) · [Σᵢ query_batching_challenge^i · eq(pow(γᵢ), z)]
///
/// Returns a `Weights::Linear` on (num_blinding_variables + 1) variables.
pub(crate) fn construct_batched_eq_weights<F: FftField>(
    blinding_evals: &[BlindingEvaluations<F>],
    masking_challenge: F,
    query_batching_challenge: F,
    num_blinding_variables: usize,
) -> Weights<F> {
    let neg_masking = -masking_challenge;
    let z_size = 1 << num_blinding_variables;
    let weight_size = 1 << (num_blinding_variables + 1);

    // Precompute query batching challenge powers
    let batching_powers =
        crate::algebra::geometric_sequence(query_batching_challenge, blinding_evals.len());

    // For each γᵢ, compute eq(pow(γᵢ), z) for all z ∈ {0,1}^ℓ using the
    // O(2^ℓ) butterfly expansion in MultilinearPoint::eq_weights(), then
    // accumulate query_batching_challenge^i · eq(pow(γᵢ), ·) into a single batched_eq vector.
    #[cfg(feature = "parallel")]
    let batched_eq: Vec<F> = {
        use rayon::prelude::*;
        blinding_evals
            .par_iter()
            .zip(batching_powers.par_iter())
            .fold(
                || vec![F::ZERO; z_size],
                |mut acc, (eval, &batching_power)| {
                    let eq_vals = MultilinearPoint::expand_from_univariate(
                        eval.gamma,
                        num_blinding_variables,
                    )
                    .eq_weights();
                    for (acc_elem, eq_val) in acc.iter_mut().zip(eq_vals) {
                        *acc_elem += batching_power * eq_val;
                    }
                    acc
                },
            )
            .reduce(
                || vec![F::ZERO; z_size],
                |mut merged_acc, partial_acc| {
                    for (merged_elem, partial_elem) in merged_acc.iter_mut().zip(partial_acc) {
                        *merged_elem += partial_elem;
                    }
                    merged_acc
                },
            )
    };
    #[cfg(not(feature = "parallel"))]
    let batched_eq: Vec<F> = {
        let mut batched = vec![F::ZERO; z_size];
        for (eval, &batching_power) in blinding_evals.iter().zip(batching_powers.iter()) {
            let eq_vals =
                MultilinearPoint::expand_from_univariate(eval.gamma, num_blinding_variables)
                    .eq_weights();
            for (acc_elem, &eq_val) in batched.iter_mut().zip(eq_vals.iter()) {
                *acc_elem += batching_power * eq_val;
            }
        }
        batched
    };

    // Build weight evaluations on {0,1}^(ℓ+1)
    // w(z, t) = eq(-masking_challenge, t) × batched_eq[z]
    // eq(-masking_challenge, 0) = 1 + masking_challenge
    // eq(-masking_challenge, 1) = -masking_challenge
    let eq_neg_masking_at_0 = F::ONE - neg_masking; // = 1 + masking_challenge
    let eq_neg_masking_at_1 = neg_masking; // = -masking_challenge

    let mut weight_evals = vec![F::ZERO; weight_size];
    for (z_idx, &beq_z) in batched_eq.iter().enumerate() {
        weight_evals[z_idx * 2] = eq_neg_masking_at_0 * beq_z; // t = 0
        weight_evals[z_idx * 2 + 1] = eq_neg_masking_at_1 * beq_z; // t = 1
    }

    Weights::linear(EvaluationsList::new(weight_evals))
}

/// Blinding polynomial evaluations at a single query point γ
#[derive(Clone, Debug)]
pub struct BlindingEvaluations<F> {
    /// The query point γ
    pub gamma: F,

    /// m(γ,ρ) = M(pow(γ), -ρ)
    pub m_eval: F,

    /// [ĝ₁(pow(γ)), ..., ĝμ(pow(γ))]
    pub g_hat_evals: Vec<F>,
}

impl<F: FftField> BlindingEvaluations<F> {
    /// Compute the blinding polynomial value h(γ) (without the masking_challenge·f̂ term).
    ///
    /// h(γ) = m(γ,masking) + Σᵢ blinding^i · γ^(2^(i-1)) · ĝᵢ(pow(γ))
    pub fn compute_h_value(&self, blinding_challenge: F) -> F {
        let mut value = self.m_eval;

        let mut blinding_power = blinding_challenge;
        let mut gamma_power = self.gamma;

        for &g_hat_eval in &self.g_hat_evals {
            value += blinding_power * gamma_power * g_hat_eval;
            blinding_power *= blinding_challenge;
            gamma_power = gamma_power.square();
        }

        value
    }
}
