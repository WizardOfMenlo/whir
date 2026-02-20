use ark_ff::{AdditiveGroup, FftField};
use ark_std::{
    rand::{CryptoRng, RngCore},
    UniformRand,
};

#[cfg(not(feature = "parallel"))]
use crate::algebra::scalar_mul_add;
use crate::algebra::{
    embedding::Basefield,
    linear_form::{Covector, Evaluate, LinearForm},
    univariate_evaluate,
};

/// Blinding evaluations at one round-0 point `gamma in Gamma`.
///
/// Values are represented in the doc-faithful `beq((pow(gamma), -rho), .)` basis:
/// - `m_eval = M(gamma, -rho)`,
/// - `g_hat_evals[j] = g_hat_j(gamma, -rho)` (with the committed `ell+1` embedding).
#[derive(Clone, Debug)]
pub struct BlindingEvaluations<F: FftField> {
    pub gamma: F,
    pub m_eval: F,
    pub g_hat_evals: Vec<F>,
}

impl<F: FftField> BlindingEvaluations<F> {
    /// Compute
    /// `h(gamma) = m(gamma,-rho) + Sum_j beta^j * gamma^(2^(j-1)) * g_hat_j(gamma,-rho)`.
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

/// Prover-side blinding polynomial family.
///
/// Shape follows the protocol notation:
/// - `m_poly(z, t)` on `ell + 1` variables by interleaving `(g0_hat, msk)`,
/// - `g_hats[j](z)` on `ell` variables.
#[derive(Clone, Debug)]
pub struct BlindingPolynomials<F: FftField> {
    pub m_poly: Vec<F::BasePrimeField>,
    pub g_hats: Vec<Vec<F::BasePrimeField>>,
}

impl<F: FftField> BlindingPolynomials<F> {
    pub fn sample<R: RngCore + CryptoRng>(
        rng: &mut R,
        num_blinding_variables: usize,
        num_witness_variables: usize,
    ) -> Self {
        let half_size = 1usize << num_blinding_variables;

        let msk = (0..half_size)
            .map(|_| F::BasePrimeField::rand(rng))
            .collect::<Vec<_>>();
        let g0_hat = (0..half_size)
            .map(|_| F::BasePrimeField::rand(rng))
            .collect::<Vec<_>>();
        let m_poly: Vec<_> = g0_hat
            .iter()
            .zip(&msk)
            .flat_map(|(&g, &m)| [g, m])
            .collect();
        let g_hats = (0..num_witness_variables)
            .map(|_| {
                (0..half_size)
                    .map(|_| F::BasePrimeField::rand(rng))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        Self { m_poly, g_hats }
    }

    /// Build committed vectors `[M, ĝ_1^emb, ..., ĝ_mu^emb]` on `ell+1` variables.
    ///
    /// This matches the blinding WHIR commitment layout used by prover and verifier.
    pub fn layout_vectors(&self) -> Vec<Vec<F::BasePrimeField>> {
        std::iter::once(self.m_poly.clone())
            .chain(self.embedded_g_hats())
            .collect()
    }

    /// Precompute `ell+1` embeddings of all `g_hat` vectors.
    pub fn embedded_g_hats(&self) -> Vec<Vec<F::BasePrimeField>> {
        self.g_hats
            .iter()
            .map(|g_hat| embed_to_ell_plus_one::<F>(g_hat))
            .collect()
    }

    #[cfg(test)]
    pub fn evaluate_at(&self, gamma: F, masking_challenge: F) -> BlindingEvaluations<F> {
        let embedded_g_hats = self.embedded_g_hats();
        self.evaluate_at_with_embedded(gamma, masking_challenge, &embedded_g_hats)
    }

    #[cfg(test)]
    pub fn evaluate_at_with_embedded(
        &self,
        gamma: F,
        masking_challenge: F,
        embedded_g_hats: &[Vec<F::BasePrimeField>],
    ) -> BlindingEvaluations<F> {
        debug_assert_eq!(self.m_poly.len() % 2, 0, "m_poly length must be even");
        let half_size = self.m_poly.len() / 2;
        debug_assert!(
            self.g_hats.iter().all(|g_hat| g_hat.len() == half_size),
            "all g_hats must have length {half_size}"
        );
        debug_assert_eq!(
            embedded_g_hats.len(),
            self.g_hats.len(),
            "embedded_g_hats count mismatch"
        );
        let num_blinding_variables = half_size.ilog2() as usize;
        let beq_covector = beq_covector_at_gamma(gamma, masking_challenge, num_blinding_variables);
        self.evaluate_with_covector(&beq_covector, embedded_g_hats, gamma)
    }

    /// Evaluate blinding polynomials using a precomputed `beq` covector.
    ///
    /// Callers that share one `beq_covector` across multiple polynomials at the
    /// same gamma avoid recomputing the `eq_weights_at_gamma` tensor product for
    /// each polynomial separately.
    pub fn evaluate_with_covector(
        &self,
        beq_covector: &Covector<F>,
        embedded_g_hats: &[Vec<F::BasePrimeField>],
        gamma: F,
    ) -> BlindingEvaluations<F> {
        debug_assert_eq!(embedded_g_hats.len(), self.g_hats.len());
        let embedding = Basefield::<F>::new();
        let m_eval = beq_covector.evaluate(&embedding, &self.m_poly);
        let g_hat_evals = embedded_g_hats
            .iter()
            .map(|g_hat_embedded| beq_covector.evaluate(&embedding, g_hat_embedded))
            .collect();
        BlindingEvaluations {
            gamma,
            m_eval,
            g_hat_evals,
        }
    }
}

/// Embed an `ell`-variable coefficient vector into `ell+1` variables by
/// interleaving zeros: `[c0, c1, ...] -> [c0, 0, c1, 0, ...]`.
///
/// This effectively treats the original polynomial as depending only on the
/// even-indexed variable, placing it in the correct position within the
/// `(ell+1)`-variable commitment layout used by the blinding WHIR.
fn embed_to_ell_plus_one<F: FftField>(coeffs: &[F::BasePrimeField]) -> Vec<F::BasePrimeField> {
    coeffs
        .iter()
        .flat_map(|&c| [c, F::BasePrimeField::ZERO])
        .collect()
}

/// Recombine a single combined claim from already tau2-batched
/// per-vector claims `(m_claim, g_hat_claims)` using tau1-batching.
pub fn combine_claim_from_components<F: FftField>(m_claim: F, g_hat_claims: &[F], tau1: F) -> F {
    // Factor 2 arises because the combined polynomial is
    //   C(tau1) = m + 2 * tau1 * Sum_j tau1^j * g_hat_j
    // The `2` compensates for the embedding interleave: g_hat coefficients
    // occupy only even slots in the (ell+1)-variable layout, halving the
    // inner product with the beq covector.
    const G_HAT_RECOMBINATION_SCALE: u64 = 2;
    m_claim + F::from(G_HAT_RECOMBINATION_SCALE) * tau1 * univariate_evaluate(g_hat_claims, tau1)
}

/// Build the combined claims and batched blinding subproof claims from
/// per-polynomial `(m_claim, g_hat_claims)`.
///
/// Returns `(combined_claims, batched_subproof_claims)` where the subproof
/// layout is `[m_0, g_0_0, ..., g_0_{mu-1}, m_1, g_1_0, ...]`.
pub fn build_combined_and_subproof_claims<F: FftField>(
    m_claims: &[F],
    g_hat_claims_per_poly: &[&[F]],
    tau1: F,
) -> (Vec<F>, Vec<F>) {
    let num_polynomials = m_claims.len();
    debug_assert_eq!(g_hat_claims_per_poly.len(), num_polynomials);
    let num_witness_variables = g_hat_claims_per_poly.first().map_or(0, |s| s.len());
    let mut combined_claims = Vec::with_capacity(num_polynomials);
    let mut subproof_claims = Vec::with_capacity(num_polynomials * (1 + num_witness_variables));
    for (poly_idx, &m_claim) in m_claims.iter().enumerate() {
        let g_hat_slice = g_hat_claims_per_poly[poly_idx];
        subproof_claims.push(m_claim);
        subproof_claims.extend_from_slice(g_hat_slice);
        combined_claims.push(combine_claim_from_components(m_claim, g_hat_slice, tau1));
    }
    (combined_claims, subproof_claims)
}

/// Build blinding-side linear forms as `[beq_weights, w_folded_weights...]`.
pub fn build_blinding_forms<'a, F: FftField>(
    beq_weights: &'a Covector<F>,
    w_folded_weights: &'a [Covector<F>],
) -> Vec<&'a dyn LinearForm<F>> {
    std::iter::once(beq_weights as &dyn LinearForm<F>)
        .chain(w_folded_weights.iter().map(|wf| wf as &dyn LinearForm<F>))
        .collect()
}

/// Fold a full-size weight to the masking period `2^(ℓ+1)`.
///
/// For a weight `w` on `μ` variables and period `P = 2^(ℓ+1)`:
///   `w_folded[j] = Σ_{i ≡ j (mod P)} w[i]`
///
/// The inner product `⟨w_folded, m_poly⟩` equals `⟨w, periodic_extension(m_poly)⟩`,
/// i.e. the masking contribution `M_eval` used for evaluation binding.
#[cfg_attr(
    feature = "tracing",
    tracing::instrument(skip_all, name = "fold_weight_to_mask_size")
)]
pub fn fold_weight_to_mask_size<F: FftField>(
    weight: &dyn crate::algebra::linear_form::LinearForm<F>,
    num_witness_variables: usize,
    num_blinding_variables: usize,
) -> Covector<F> {
    use std::any::Any;

    let mask_size = 1usize << (num_blinding_variables + 1);

    // Fast path: borrow the vector directly if the weight is already a Covector.
    let borrowed = (weight as &dyn Any).downcast_ref::<Covector<F>>();
    let owned;
    let vector = match borrowed {
        Some(cov) => &cov.vector,
        None => {
            owned = Covector::from(weight);
            &owned.vector
        }
    };
    debug_assert_eq!(vector.len(), 1usize << num_witness_variables);

    fold_vector_to_mask_size(vector, mask_size)
}

fn fold_vector_to_mask_size<F: FftField>(vector: &[F], mask_size: usize) -> Covector<F> {
    #[cfg(feature = "parallel")]
    let folded = {
        use rayon::prelude::*;
        vector
            .par_chunks(mask_size)
            .fold(
                || vec![F::ZERO; mask_size],
                |mut acc, chunk| {
                    for (j, &v) in chunk.iter().enumerate() {
                        acc[j] += v;
                    }
                    acc
                },
            )
            .reduce(
                || vec![F::ZERO; mask_size],
                |mut a, b| {
                    for (j, &v) in b.iter().enumerate() {
                        a[j] += v;
                    }
                    a
                },
            )
    };

    #[cfg(not(feature = "parallel"))]
    let folded = {
        let mut f = vec![F::ZERO; mask_size];
        for (i, &v) in vector.iter().enumerate() {
            f[i % mask_size] += v;
        }
        f
    };

    Covector::new(folded)
}

/// Build the single batched beq linear form used by the blinding subproof:
/// `Sum_i tau2^i * beq((pow(gamma_i), -rho), .)`.
#[cfg_attr(
    feature = "tracing",
    tracing::instrument(skip_all, name = "construct_batched_eq_weights")
)]
pub fn construct_batched_eq_weights_from_gammas<F: FftField>(
    gammas: &[F],
    masking_challenge: F,
    tau2: F,
    num_blinding_variables: usize,
) -> Covector<F> {
    let weight_size = 1 << (num_blinding_variables + 1);

    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;

        // Pre-compute tau2 powers: tau2^0, tau2^1, ..., tau2^(n-1)
        let tau2_powers: Vec<F> = {
            let mut powers = Vec::with_capacity(gammas.len());
            let mut p = F::ONE;
            for _ in 0..gammas.len() {
                powers.push(p);
                p *= tau2;
            }
            powers
        };

        // Each gamma's eq_weights computation is independent — parallelise across gammas
        // and reduce by element-wise summation.
        // The (acc, buf) tuple lets each rayon task reuse one scratch buffer across its gammas.
        let (weight_evals, _) = gammas
            .par_iter()
            .zip(tau2_powers.par_iter())
            .fold(
                || (vec![F::ZERO; weight_size], vec![F::ZERO; weight_size]),
                |(mut acc, mut buf), (&gamma, &power)| {
                    fill_eq_weights_at_gamma(
                        &mut buf,
                        gamma,
                        masking_challenge,
                        num_blinding_variables,
                    );
                    for (a, &b) in acc.iter_mut().zip(buf.iter()) {
                        *a += power * b;
                    }
                    (acc, buf)
                },
            )
            .reduce(
                || (vec![F::ZERO; weight_size], vec![F::ZERO; weight_size]),
                |(mut a, buf_a), (b, _)| {
                    for (ai, &bi) in a.iter_mut().zip(b.iter()) {
                        *ai += bi;
                    }
                    (a, buf_a)
                },
            );

        Covector::new(weight_evals)
    }

    #[cfg(not(feature = "parallel"))]
    {
        let mut weight_evals = vec![F::ZERO; weight_size];
        let mut buf = vec![F::ZERO; weight_size];
        let mut batching_power = F::ONE;
        for &gamma in gammas {
            fill_eq_weights_at_gamma(&mut buf, gamma, masking_challenge, num_blinding_variables);
            scalar_mul_add(&mut weight_evals, batching_power, &buf);
            batching_power *= tau2;
        }
        Covector::new(weight_evals)
    }
}

/// Build `beq((pow(gamma), -rho), .)` as a covector over `ell+1` variables.
#[cfg(test)]
pub fn beq_covector_at_gamma<F: FftField>(
    gamma: F,
    masking_challenge: F,
    num_blinding_variables: usize,
) -> Covector<F> {
    Covector::new(eq_weights_at_gamma(
        gamma,
        masking_challenge,
        num_blinding_variables,
    ))
}

#[cfg(test)]
fn eq_weights_at_gamma<F: FftField>(
    gamma: F,
    masking_challenge: F,
    num_blinding_variables: usize,
) -> Vec<F> {
    let num_total_variables = num_blinding_variables + 1;
    let size = 1usize << num_total_variables;
    let mut weights = vec![F::ZERO; size];
    fill_eq_weights_at_gamma(
        &mut weights,
        gamma,
        masking_challenge,
        num_blinding_variables,
    );
    weights
}

/// In-place version of [`eq_weights_at_gamma`] that fills an existing buffer.
///
/// `buf` must have length `>= 2^(num_blinding_variables + 1)`.  Reusing a
/// single buffer across gamma iterations avoids one 8 KiB allocation per point.
pub fn fill_eq_weights_at_gamma<F: FftField>(
    buf: &mut [F],
    gamma: F,
    masking_challenge: F,
    num_blinding_variables: usize,
) {
    // beq((pow(gamma), -rho), .) on ell+1 variables via iterative tensor expansion.
    // O(2^(ell+1)) muls instead of O(ell * 2^ell) from per-point eq_poly + tensor_product.
    let num_total_variables = num_blinding_variables + 1; // +1 for the masking variable
    let size = 1usize << num_total_variables;
    debug_assert!(buf.len() >= size);

    for w in &mut buf[..size] {
        *w = F::ZERO;
    }
    buf[0] = F::ONE;

    // Variables 0..ell: squaring-ladder basis (gamma, gamma^2, gamma^4, ...)
    let mut gamma_power = gamma;
    for i in 0..num_blinding_variables {
        for j in (0..1usize << i).rev() {
            buf[2 * j + 1] = buf[j] * gamma_power;
            buf[2 * j] = buf[j] - buf[2 * j + 1];
        }
        gamma_power = gamma_power.square();
    }

    // Last variable (ell): fixed to -masking_challenge.
    let neg_rho = -masking_challenge;
    let half = 1usize << num_blinding_variables;
    for j in (0..half).rev() {
        buf[2 * j + 1] = buf[j] * neg_rho;
        buf[2 * j] = buf[j] - buf[2 * j + 1];
    }
}

/// Half-size beq weights: only the first `ell` squaring-ladder variables,
/// WITHOUT the final `-rho` expansion.
///
/// `buf` must have length `>= 2^num_blinding_variables`.
///
/// The full-size beq satisfies `beq_full[2j] = beq_half[j] * (1+rho)` and
/// `beq_full[2j+1] = beq_half[j] * (-rho)`, so any inner product with a
/// vector whose odd entries are zero reduces to a half-size mixed dot
/// multiplied by `(1+rho)`.
pub fn fill_eq_weights_at_gamma_half<F: FftField>(
    buf: &mut [F],
    gamma: F,
    num_blinding_variables: usize,
) {
    let size = 1usize << num_blinding_variables;
    debug_assert!(buf.len() >= size);

    for w in &mut buf[..size] {
        *w = F::ZERO;
    }
    buf[0] = F::ONE;

    let mut gamma_power = gamma;
    for i in 0..num_blinding_variables {
        for j in (0..1usize << i).rev() {
            buf[2 * j + 1] = buf[j] * gamma_power;
            buf[2 * j] = buf[j] - buf[2 * j + 1];
        }
        gamma_power = gamma_power.square();
    }
}

#[cfg(test)]
mod tests {
    use ark_ff::Field;

    use super::*;
    use crate::algebra::{embedding::Basefield, fields::Field64_2};

    fn batch_with_challenge<F: FftField, I: IntoIterator<Item = F>>(values: I, challenge: F) -> F {
        let mut acc = F::ZERO;
        let mut power = F::ONE;
        for value in values {
            acc += power * value;
            power *= challenge;
        }
        acc
    }

    fn compute_per_polynomial_claims<F: FftField>(
        blinding_evals: &[BlindingEvaluations<F>],
        tau2: F,
    ) -> (F, Vec<F>) {
        let num_g_hats = blinding_evals
            .first()
            .map_or(0, |eval| eval.g_hat_evals.len());
        let m_claim = batch_with_challenge(blinding_evals.iter().map(|eval| eval.m_eval), tau2);
        let mut g_hat_claims = vec![F::ZERO; num_g_hats];
        for (g_hat_idx, claim) in g_hat_claims.iter_mut().enumerate() {
            *claim = batch_with_challenge(
                blinding_evals
                    .iter()
                    .map(|eval| eval.g_hat_evals[g_hat_idx]),
                tau2,
            );
        }
        (m_claim, g_hat_claims)
    }

    /// Convenience wrapper for tests that have already materialized [`BlindingEvaluations`].
    fn construct_batched_eq_weights<F: FftField>(
        blinding_evals: &[BlindingEvaluations<F>],
        masking_challenge: F,
        tau2: F,
        num_blinding_variables: usize,
    ) -> Covector<F> {
        let gammas = blinding_evals.iter().map(|e| e.gamma).collect::<Vec<_>>();
        construct_batched_eq_weights_from_gammas(
            &gammas,
            masking_challenge,
            tau2,
            num_blinding_variables,
        )
    }

    type EF = Field64_2;

    #[test]
    fn test_compute_h_value_uses_square_ladder() {
        let gamma = EF::from(5u64);
        let beta = EF::from(7u64);
        let eval = BlindingEvaluations {
            gamma,
            m_eval: EF::from(11u64),
            g_hat_evals: vec![EF::from(13u64), EF::from(17u64), EF::from(19u64)],
        };
        let expected = eval.m_eval
            + beta * gamma * eval.g_hat_evals[0]
            + beta.square() * gamma.square() * eval.g_hat_evals[1]
            + beta.pow([3]) * gamma.pow([4]) * eval.g_hat_evals[2];
        assert_eq!(eval.compute_h_value(beta), expected);
    }

    #[test]
    fn test_evaluate_at_matches_beq_layout_openings() {
        let num_blinding_variables = 3usize;
        let num_witness_variables = 2usize;
        let half = 1usize << num_blinding_variables;
        let msk = (0..half)
            .map(|i| <EF as Field>::BasePrimeField::from((i as u64) + 2))
            .collect::<Vec<_>>();
        let g0_hat = (0..half)
            .map(|i| <EF as Field>::BasePrimeField::from((3 * i as u64) + 1))
            .collect::<Vec<_>>();
        let mut m_poly = vec![<EF as Field>::BasePrimeField::from(0u64); 2 * half];
        for i in 0..half {
            m_poly[2 * i] = g0_hat[i];
            m_poly[2 * i + 1] = msk[i];
        }
        let g_hats = (0..num_witness_variables)
            .map(|j| {
                (0..half)
                    .map(|i| <EF as Field>::BasePrimeField::from((i as u64) + (j as u64) + 5))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let polys = BlindingPolynomials::<EF> { m_poly, g_hats };
        let gamma = EF::from(9u64);
        let rho = EF::from(4u64);
        let eval = polys.evaluate_at(gamma, rho);

        let beq = eq_weights_at_gamma(gamma, rho, num_blinding_variables);
        let covector = Covector::new(beq);
        let embedding = Basefield::<EF>::new();
        let layout_vectors = polys.layout_vectors();

        assert_eq!(
            eval.m_eval,
            covector.evaluate(&embedding, &layout_vectors[0])
        );
        for (g_idx, g_eval) in eval.g_hat_evals.iter().enumerate() {
            assert_eq!(
                *g_eval,
                covector.evaluate(&embedding, &layout_vectors[1 + g_idx])
            );
        }
    }

    #[test]
    fn test_construct_batched_eq_weights_matches_per_gamma_accumulation() {
        let gammas = [EF::from(3u64), EF::from(6u64), EF::from(10u64)];
        let rho = EF::from(2u64);
        let tau2 = EF::from(5u64);
        let num_blinding_variables = 3usize;
        let bevals = gammas
            .into_iter()
            .map(|gamma| BlindingEvaluations {
                gamma,
                m_eval: EF::ZERO,
                g_hat_evals: vec![],
            })
            .collect::<Vec<_>>();

        let batched = construct_batched_eq_weights(&bevals, rho, tau2, num_blinding_variables);
        let mut expected = vec![EF::ZERO; 1 << (num_blinding_variables + 1)];
        let mut power = EF::ONE;
        for gamma in gammas {
            let per_gamma = eq_weights_at_gamma(gamma, rho, num_blinding_variables);
            for (acc, value) in expected.iter_mut().zip(per_gamma) {
                *acc += power * value;
            }
            power *= tau2;
        }
        assert_eq!(batched.vector, expected);
    }

    #[test]
    fn test_combined_claim_matches_opening_side_recomposition() {
        let num_blinding_variables = 3usize;
        let num_witness_variables = 2usize;
        let half = 1usize << num_blinding_variables;
        let msk = (0..half)
            .map(|i| <EF as Field>::BasePrimeField::from((i as u64) + 7))
            .collect::<Vec<_>>();
        let g0_hat = (0..half)
            .map(|i| <EF as Field>::BasePrimeField::from((5 * i as u64) + 2))
            .collect::<Vec<_>>();
        let mut m_poly = vec![<EF as Field>::BasePrimeField::from(0u64); 2 * half];
        for i in 0..half {
            m_poly[2 * i] = g0_hat[i];
            m_poly[2 * i + 1] = msk[i];
        }
        let g_hats = (0..num_witness_variables)
            .map(|j| {
                (0..half)
                    .map(|i| <EF as Field>::BasePrimeField::from((i as u64) + (2 * j as u64) + 11))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let polys = BlindingPolynomials::<EF> { m_poly, g_hats };

        let gammas = [EF::from(3u64), EF::from(9u64), EF::from(12u64)];
        let rho = EF::from(4u64);
        let tau1 = EF::from(6u64);
        let tau2 = EF::from(5u64);
        let blinding_evals = gammas
            .into_iter()
            .map(|gamma| polys.evaluate_at(gamma, rho))
            .collect::<Vec<_>>();

        let direct_claim =
            blinding_evals
                .iter()
                .enumerate()
                .fold(EF::ZERO, |acc, (gamma_idx, eval)| {
                    let tau2_power = tau2.pow([gamma_idx as u64]);
                    let inner = eval
                        .g_hat_evals
                        .iter()
                        .enumerate()
                        .fold(EF::ZERO, |inner_acc, (j, g_eval)| {
                            inner_acc + tau1.pow([(j + 1) as u64]) * *g_eval
                        });
                    acc + tau2_power * (eval.m_eval + EF::from(2u64) * inner)
                });
        let (m_claim, g_hat_claims) = compute_per_polynomial_claims(&blinding_evals, tau2);
        let recomposed_claim = combine_claim_from_components(m_claim, &g_hat_claims, tau1);
        assert_eq!(direct_claim, recomposed_claim);
    }
}
