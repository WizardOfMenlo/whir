use ark_ff::Field;

use super::Weights;
use crate::{
    algebra::{polynomials::MultilinearPoint, weights::UnivariateEvaluation, Embedding},
    utils::zip_strict,
};

/// Weights vector to represent multilinear polynomial evaluation.
///
/// Specifically the weights are $w_i = prod_j x_j^bit(i, j)$ where $bit(i, j)$
/// represents wheter the binary decomposition of $i$ has a $2^j$ term.
pub struct MultilinearEvaluation<M: Embedding> {
    pub embedding: M,
    pub point: Vec<M::Target>,
}

impl<M: Embedding> Weights<M> for MultilinearEvaluation<M> {
    fn embedding(&self) -> &M {
        &self.embedding
    }

    fn deferred(&self) -> bool {
        false
    }

    fn size(&self) -> usize {
        1 << self.point.len()
    }

    fn mle_evaluate(&self, point: MultilinearPoint<M::Target>) -> <M as Embedding>::Target {
        zip_strict(&self.point, &point.0).fold(M::Target::ONE, |acc, (&l, &r)| {
            acc * (l * r + (M::Target::ONE - l) * (M::Target::ONE - r))
        })
    }

    fn inner_product(&self, vector: &[M::Source]) -> M::Target {
        todo!()
    }

    fn accumulate(&self, accumulator: &mut [M::Target], scalar: M::Target) {
        todo!()
    }
}

impl<M: Embedding> MultilinearEvaluation<M> {
    pub fn new(point: Vec<M::Target>) -> Self
    where
        M: Default,
    {
        Self {
            embedding: M::default(),
            point,
        }
    }

    /// A [`UnivariateEvaluation`] can be represented as a [`MultilinearEvaluation`].
    pub fn from_univariate(univariate: &UnivariateEvaluation<M>) -> Self {
        assert!(univariate.size().is_power_of_two());
        let num_variables = univariate.size().trailing_zeros() as usize;
        let mut point = Vec::with_capacity(num_variables);
        let mut pow2k = univariate.point;
        for _ in 0..num_variables {
            point.push(pow2k);
            pow2k.square_in_place(); // Compute x^(2^k) at each step
        }
        Self {
            embedding: univariate.embedding.clone(),
            point,
        }
    }
}
