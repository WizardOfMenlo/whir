use ark_ff::{AdditiveGroup, FftField};
use ark_std::{
    rand::{CryptoRng, RngCore},
    UniformRand,
};

use crate::algebra::{
    linear_form::{Covector, Evaluate},
    tensor_product, MultilinearPoint,
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
        for (&gamma_power, &g_hat_eval) in gamma_pow_ladder(self.gamma, self.g_hat_evals.len())
            .iter()
            .zip(&self.g_hat_evals)
        {
            value += blinding_power * gamma_power * g_hat_eval;
            blinding_power *= blinding_challenge;
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
        let full_size = half_size * 2;

        let msk = (0..half_size)
            .map(|_| F::BasePrimeField::rand(rng))
            .collect::<Vec<_>>();
        let g0_hat = (0..half_size)
            .map(|_| F::BasePrimeField::rand(rng))
            .collect::<Vec<_>>();
        let mut m_poly = vec![F::BasePrimeField::ZERO; full_size];
        for i in 0..half_size {
            m_poly[2 * i] = g0_hat[i];
            m_poly[2 * i + 1] = msk[i];
        }
        let g_hats = (0..num_witness_variables)
            .map(|j| {
                let _ = j;
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
        let mut vectors = Vec::with_capacity(1 + self.g_hats.len());
        vectors.push(self.m_poly.clone());
        for g_hat in &self.g_hats {
            vectors.push(embed_to_ell_plus_one::<F>(g_hat));
        }
        vectors
    }

    pub fn evaluate_at(&self, gamma: F, masking_challenge: F) -> BlindingEvaluations<F> {
        use crate::algebra::embedding::Basefield;
        debug_assert_eq!(self.m_poly.len() % 2, 0);
        let half_size = self.m_poly.len() / 2;
        debug_assert!(self.g_hats.iter().all(|g_hat| g_hat.len() == half_size));
        let num_blinding_variables = half_size.ilog2() as usize;
        let beq_weights = eq_weights_at_gamma(
            gamma,
            masking_challenge,
            num_blinding_variables,
            self.m_poly.len(),
        );
        // Evaluate all blinding polynomials with the same beq(gamma, -rho) linear form
        // that is later used in the batched blinding WHIR opening.
        let beq_covector = Covector::new(beq_weights);
        let embedding = Basefield::<F>::new();
        let m_eval = beq_covector.evaluate(&embedding, &self.m_poly);
        let g_hat_evals = self
            .g_hats
            .iter()
            .map(|g_hat| {
                let g_hat_embedded = embed_to_ell_plus_one::<F>(g_hat);
                beq_covector.evaluate(&embedding, &g_hat_embedded)
            })
            .collect();
        BlindingEvaluations {
            gamma,
            m_eval,
            g_hat_evals,
        }
    }
}

fn embed_to_ell_plus_one<F: FftField>(coeffs: &[F::BasePrimeField]) -> Vec<F::BasePrimeField> {
    let mut out = vec![F::BasePrimeField::ZERO; coeffs.len() * 2];
    for (i, &c) in coeffs.iter().enumerate() {
        out[2 * i] = c;
    }
    out
}

pub fn compute_per_polynomial_claims<F: FftField>(
    blinding_evals: &[BlindingEvaluations<F>],
    tau2: F,
) -> (F, Vec<F>) {
    // Outer batching over Gamma with tau2: Sum_i tau2^i * eval_i.
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

pub fn batch_with_challenge<F: FftField, I>(values: I, challenge: F) -> F
where
    I: IntoIterator<Item = F>,
{
    let mut acc = F::ZERO;
    let mut power = F::ONE;
    for value in values {
        acc += power * value;
        power *= challenge;
    }
    acc
}

/// Recompose the same doc-style combined claim from already tau2-batched
/// per-vector claims `(m_claim, g_hat_claims)`.
pub fn recombine_doc_claim_from_components<F: FftField>(
    m_claim: F,
    g_hat_claims: &[F],
    tau1: F,
) -> F {
    let mut inner_g_claim = F::ZERO;
    let mut tau1_power = tau1;
    for &g_hat_claim in g_hat_claims {
        inner_g_claim += tau1_power * g_hat_claim;
        tau1_power *= tau1;
    }
    m_claim + (F::from(2u64) * inner_g_claim)
}

pub fn construct_batched_eq_weights<F: FftField>(
    blinding_evals: &[BlindingEvaluations<F>],
    masking_challenge: F,
    tau2: F,
    num_blinding_variables: usize,
) -> Covector<F> {
    // Build the single batched beq linear form used by the blinding subproof:
    // Sum_i tau2^i * beq((pow(gamma_i), -rho), .).
    let weight_size = 1 << (num_blinding_variables + 1);
    let mut weight_evals = vec![F::ZERO; weight_size];
    let mut batching_power = F::ONE;
    for eval in blinding_evals {
        let per_gamma = eq_weights_at_gamma(
            eval.gamma,
            masking_challenge,
            num_blinding_variables,
            weight_size,
        );
        for (acc_elem, &eq_val) in weight_evals.iter_mut().zip(per_gamma.iter()) {
            *acc_elem += batching_power * eq_val;
        }
        batching_power *= tau2;
    }
    Covector::new(weight_evals)
}

fn multilinear_pow_point<F: FftField>(gamma: F, num_blinding_variables: usize) -> Vec<F> {
    let mut point = Vec::with_capacity(num_blinding_variables);
    let mut power = gamma;
    for _ in 0..num_blinding_variables {
        point.push(power);
        power = power.square();
    }
    point
}

fn gamma_pow_ladder<F: FftField>(gamma: F, count: usize) -> Vec<F> {
    let mut powers = Vec::with_capacity(count);
    let mut power = gamma;
    for _ in 0..count {
        powers.push(power);
        power = power.square();
    }
    powers
}

fn eq_weights_at_gamma<F: FftField>(
    gamma: F,
    masking_challenge: F,
    num_blinding_variables: usize,
    output_size: usize,
) -> Vec<F> {
    // `z` coordinates are powers of gamma: (gamma, gamma^2, gamma^4, ...).
    // The last variable is fixed to `-rho`, giving beq((z, -rho), ·).
    let z_point = multilinear_pow_point(gamma, num_blinding_variables);
    let z_eq = MultilinearPoint(z_point).eq_weights();
    let eq_neg_masking_at_0 = F::ONE + masking_challenge;
    let eq_neg_masking_at_1 = -masking_challenge;
    let beq_weights = tensor_product(&z_eq, &[eq_neg_masking_at_0, eq_neg_masking_at_1]);
    debug_assert_eq!(output_size, beq_weights.len());
    beq_weights
}

#[cfg(test)]
mod tests {
    use ark_ff::Field;

    use super::*;
    use crate::algebra::{embedding::Basefield, fields::Field64_2};

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

        let beq = eq_weights_at_gamma(gamma, rho, num_blinding_variables, polys.m_poly.len());
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
            let per_gamma = eq_weights_at_gamma(gamma, rho, num_blinding_variables, expected.len());
            for (acc, value) in expected.iter_mut().zip(per_gamma) {
                *acc += power * value;
            }
            power *= tau2;
        }
        assert_eq!(batched.vector, expected);
    }

    #[test]
    fn test_doc_combined_claim_matches_opening_side_recomposition() {
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
        let recomposed_claim = recombine_doc_claim_from_components(m_claim, &g_hat_claims, tau1);
        assert_eq!(direct_claim, recomposed_claim);
    }
}
