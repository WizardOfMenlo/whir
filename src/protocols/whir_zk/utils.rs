use ark_ff::FftField;

use crate::algebra::{
    polynomials::{CoefficientList, EvaluationsList, MultilinearPoint},
    Weights,
};

/// Collect interleaved references to all blinding polynomials in batch order:
/// `[M₁, ĝ₁₁, …, ĝ₁μ, M₂, ĝ₂₁, …, ĝ₂μ, …]`
///
/// Used by both committer (batch-commit) and prover (batch-prove).
pub fn interleave_blinding_poly_refs<'a, F: FftField>(
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
pub fn compute_per_polynomial_claims<F: FftField>(
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
pub fn construct_batched_eq_weights<F: FftField>(
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
