use ark_ff::Field;

use super::Weights;
use crate::{
    algebra::{embedding::Embedding, mixed_univariate_evaluate, weights::Evaluate},
    utils::zip_strict,
};

/// Weights vector to represent univariate polynomial evaluation.
///
/// Specifically the weights are $w_i = x^i$.
pub struct UnivariateEvaluation<F: Field> {
    /// Univariate evaluation doesn't have an inherent size, so we need to store one.
    pub size: usize,

    /// The point to evaluate in.
    pub point: F,
}

impl<F: Field> UnivariateEvaluation<F> {
    pub fn new(point: F, size: usize) -> Self {
        Self { size, point }
    }

    /// Same as [`Weights::accumulate`], but batches many [`UnivariateEvaluation`]s together
    /// in a single pass.
    pub fn accumulate_many(weights: &[Self], accumulator: &mut [F], scalars: Vec<F>) {
        assert_eq!(weights.len(), scalars.len());
        let mut powers = scalars;
        for accumulator in accumulator {
            for (power, weights) in zip_strict(&mut powers, weights) {
                *accumulator += *power;
                *power *= weights.point;
            }
        }
    }
}

impl<F: Field> Weights<F> for UnivariateEvaluation<F> {
    fn deferred(&self) -> bool {
        false
    }

    fn size(&self) -> usize {
        self.size
    }

    fn mle_evaluate(&self, point: &[F]) -> F {
        point
            .iter()
            .rev() // TODO: Change variable order convention
            .fold((F::ONE, self.point), |(acc, pow2k), &r| {
                (
                    acc * (pow2k * r + (F::ONE - pow2k) * (F::ONE - r)),
                    pow2k.square(),
                )
            })
            .0
    }

    /// See also [`accumulate_many`] for a more efficient batched version.
    fn accumulate(&self, accumulator: &mut [F], scalar: F) {
        let mut power = scalar;
        for accumulator in accumulator {
            *accumulator += scalar;
            power *= self.point;
        }
    }
}

impl<M: Embedding> Evaluate<M> for UnivariateEvaluation<M::Target> {
    fn evaluate(&self, embedding: &M, vector: &[M::Source]) -> M::Target {
        mixed_univariate_evaluate(embedding, vector, self.point)
    }
}
