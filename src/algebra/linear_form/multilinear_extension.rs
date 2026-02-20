use ark_ff::Field;

use super::{Evaluate, LinearForm};
use crate::{
    algebra::{eval_eq, mixed_multilinear_extend, Embedding},
    utils::zip_strict,
};

/// Multilinear extension evaluation as a linear form $𝔽^n → 𝔽$.
///
/// Given a multilinear function $f ∈ 𝔽^(≤ 1)[X_0,…,X_(k-1)]$ represented by a vector $v ∈ 𝔽^n$ with
/// $n = 2^k$ using the boolean hypercube evaluation basis such that $v_i = f( bits(i) )$ where
/// $bits: ℕ → {0,1}^k$ is the little-endian binary decomposition, then this linear form will
/// evaluate to $f(x)$ for some fixed point $x ∈ 𝔽^k$.
///
pub struct MultilinearExtension<F: Field> {
    pub point: Vec<F>,
}

impl<F: Field> MultilinearExtension<F> {
    pub const fn new(point: Vec<F>) -> Self {
        Self { point }
    }
}

impl<F: Field> LinearForm<F> for MultilinearExtension<F> {
    fn size(&self) -> usize {
        1 << self.point.len()
    }

    fn deferred(&self) -> bool {
        false
    }

    fn mle_evaluate(&self, point: &[F]) -> F {
        zip_strict(&self.point, point).fold(F::ONE, |acc, (&l, &r)| {
            acc * (l * r + (F::ONE - l) * (F::ONE - r))
        })
    }

    fn accumulate(&self, accumulator: &mut [F], scalar: F) {
        eval_eq(accumulator, &self.point, scalar);
    }

    /// Avoids materializing the full 2^μ weight vector by exploiting the factorization:
    ///
    ///   w_folded[j] = Σ_{i ≡ j (mod 2^fold_vars)} eq(α, binary(i))
    ///               = eq(α[μ-fold_vars .. μ], binary(j))
    ///
    /// The inner sum over high-bit combinations equals 1 by the normalization identity
    /// Σ_k eq(β, k) = 1, leaving only the low-coordinate factor.
    /// In the `eval_eq` MSB convention the low bits of index j correspond to the *last*
    /// coordinates of the evaluation point, so we use the suffix `α[μ-fold_vars..]`.
    fn fold_to_size(&self, fold_vars: usize) -> Option<Vec<F>> {
        let mu = self.point.len();
        if fold_vars > mu {
            return None;
        }
        let size = 1 << fold_vars;
        let mut result = vec![F::ZERO; size];
        eval_eq(&mut result, &self.point[mu - fold_vars..], F::ONE);
        Some(result)
    }
}

impl<M: Embedding> Evaluate<M> for MultilinearExtension<M::Target> {
    fn evaluate(&self, embedding: &M, vector: &[M::Source]) -> M::Target {
        mixed_multilinear_extend(embedding, vector, &self.point)
    }
}
