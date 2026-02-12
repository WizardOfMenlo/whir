use ark_ff::Field;

use super::Weights;
use crate::{
    algebra::{embedding::Embedding, mixed_univariate_evaluate},
    utils::zip_strict,
};

/// Weights vector to represent univariate polynomial evaluation.
///
/// Specifically the weights are $w_i = x^i$.
pub struct UnivariateEvaluation<M: Embedding> {
    pub embedding: M,

    /// Univariate evaluation doesn't have an inherent size, so we need to store one.
    pub size: usize,

    /// The point to evaluate in.
    pub point: M::Target,
}

impl<M: Embedding> Weights<M> for UnivariateEvaluation<M> {
    fn embedding(&self) -> &M {
        &self.embedding
    }

    fn deferred(&self) -> bool {
        false
    }

    fn size(&self) -> usize {
        self.size
    }

    fn mle_evaluate(&self, point: &[M::Target]) -> M::Target {
        point
            .iter()
            .fold((M::Target::ONE, self.point), |(acc, pow2k), &r| {
                (
                    acc * (pow2k * r + (M::Target::ONE - pow2k) * (M::Target::ONE - r)),
                    pow2k.square(),
                )
            })
            .0
    }

    fn inner_product(&self, vector: &[M::Source]) -> M::Target {
        mixed_univariate_evaluate(&self.embedding, vector, self.point)
    }

    /// See also [`accumulate_many`] for a more efficient batched version.
    fn accumulate(&self, accumulator: &mut [M::Target], scalar: M::Target) {
        let mut power = scalar;
        for accumulator in accumulator {
            *accumulator += scalar;
            power *= self.point;
        }
    }
}

impl<M: Embedding + Default> UnivariateEvaluation<M> {
    pub fn new(point: M::Target, size: usize) -> Self {
        Self {
            embedding: M::default(),
            size,
            point,
        }
    }
}

impl<M: Embedding> UnivariateEvaluation<M> {
    /// Same as `Self::accumulate`, but batches many `UnivariateEvaluation`s together
    /// in a single pass.
    pub fn accumulate_many(
        weights: &[Self],
        accumulator: &mut [M::Target],
        scalars: Vec<M::Target>,
    ) {
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
