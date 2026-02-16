use ark_ff::Field;

use super::{Evaluate, LinearForm};
use crate::{
    algebra::{eval_eq, linear_form::UnivariateEvaluation, mixed_multilinear_extend, Embedding},
    utils::zip_strict,
};

/// Multilinear polynomial evaluation as a linear form $𝔽^n → 𝔽$.
///
/// Given a multilinear function $f ∈ 𝔽^(≤ 1)[X_0,…,X_(k-1)]$ represented by a vector $v ∈ 𝔽^n$ with
/// $n = 2^k$ using the boolean hypercube evaluation basis such that $v_i = f( bits(i) )$ where
/// $bits: ℕ → {0,1}^k$ is the little-endian binary decomposition, then this linear form will
/// evaluate to $f(x)$ for some fixed point $x ∈ 𝔽^k$.
///
pub struct MultilinearEvaluation<F: Field> {
    pub point: Vec<F>,
}

impl<F: Field> LinearForm<F> for MultilinearEvaluation<F> {
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
}

impl<M: Embedding> Evaluate<M> for MultilinearEvaluation<M::Target> {
    fn evaluate(&self, embedding: &M, vector: &[M::Source]) -> M::Target {
        mixed_multilinear_extend(embedding, vector, &self.point)
    }
}

impl<F: Field> MultilinearEvaluation<F> {
    pub const fn new(point: Vec<F>) -> Self {
        Self { point }
    }

    /// A [`UnivariateEvaluation`] can be represented as a [`MultilinearEvaluation`].
    pub fn from_univariate(univariate: &UnivariateEvaluation<F>) -> Self {
        assert!(univariate.size().is_power_of_two());
        let num_variables = univariate.size().trailing_zeros() as usize;
        let mut point = Vec::with_capacity(num_variables);
        let mut pow2k = univariate.point;
        for _ in 0..num_variables {
            point.push(pow2k);
            pow2k.square_in_place(); // Compute x^(2^k) at each step
        }
        point.reverse(); // Match big-endian convention used by multilinear evaluations.
        Self { point }
    }
}
