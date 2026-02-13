use ark_ff::Field;

use super::Weights;
use crate::{
    algebra::{
        embedding::Embedding, eval_eq, mixed_univariate_evaluate, polynomials::MultilinearPoint,
        weights::Evaluate,
    },
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
    pub fn accumulate_many(weights: &[Self], accumulator: &mut [F], scalars: &[F]) {
        for (weights, &scalar) in zip_strict(weights, scalars) {
            weights.accumulate(accumulator, scalar);
        }

        /*
        assert_eq!(weights.len(), scalars.len());
        let mut powers = scalars.to_vec();
        for accumulator in accumulator {
            for (power, weights) in zip_strict(&mut powers, weights) {
                *accumulator += *power;
                *power *= weights.point;
            }
        }
        */
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
        let point = MultilinearPoint::expand_from_univariate(
            self.point,
            self.size.trailing_zeros() as usize,
        );

        eval_eq(accumulator, &point.0, scalar);

        // let mut power = scalar;
        // for accumulator in accumulator {
        //     *accumulator += power;
        //     power *= self.point;
        // }
    }
}

impl<M: Embedding> Evaluate<M> for UnivariateEvaluation<M::Target> {
    // Evaluate a vector in coefficient form.
    fn evaluate(&self, embedding: &M, vector: &[M::Source]) -> M::Target {
        mixed_univariate_evaluate(embedding, vector, self.point)
    }
}
