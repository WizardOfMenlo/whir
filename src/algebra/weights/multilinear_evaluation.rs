use ark_ff::Field;

use super::{Evaluate, Weights};
use crate::{
    algebra::{eval_eq, mixed_multilinear_extend, weights::UnivariateEvaluation, Embedding},
    utils::zip_strict,
};

/// Weights vector to represent multilinear polynomial evaluation.
///
/// Specifically the weights are $w_i = prod_j x_j^bit(i, j)$ where $bit(i, j)$
/// represents wheter the binary decomposition of $i$ has a $2^j$ term.
pub struct MultilinearEvaluation<F: Field> {
    pub point: Vec<F>,
}

impl<F: Field> Weights<F> for MultilinearEvaluation<F> {
    fn deferred(&self) -> bool {
        false
    }

    fn size(&self) -> usize {
        1 << self.point.len()
    }

    fn mle_evaluate(&self, point: &[F]) -> F {
        zip_strict(&self.point, point).fold(F::ONE, |acc, (&l, &r)| {
            acc * (l * r + (F::ONE - l) * (F::ONE - r))
        })
    }

    fn accumulate(&self, accumulator: &mut [F], scalar: F) {
        eval_eq(accumulator, &self.point, scalar)
    }
}

impl<M: Embedding> Evaluate<M> for MultilinearEvaluation<M::Target> {
    fn evaluate(&self, embedding: &M, vector: &[M::Source]) -> M::Target {
        mixed_multilinear_extend(embedding, vector, &self.point)
    }
}

impl<F: Field> MultilinearEvaluation<F> {
    pub fn new(point: Vec<F>) -> Self {
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
