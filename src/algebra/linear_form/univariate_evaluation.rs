use ark_ff::Field;

use super::LinearForm;
use crate::algebra::{
    embedding::Embedding, geometric_accumulate, linear_form::Evaluate, mixed_univariate_evaluate,
};

/// Linear form to represent univariate polynomial evaluation.
///
/// Given a vector $v ∈ 𝔽^n$ it computes $sum_i v_i · x^i$ for some fixed $x$.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct UnivariateEvaluation<F: Field> {
    /// Univariate evaluation doesn't have an inherent size, so we need to store one.
    pub size: usize,

    /// The point $x ∈ 𝔽$ to evaluate on.
    pub point: F,
}

impl<F: Field> UnivariateEvaluation<F> {
    pub const fn new(point: F, size: usize) -> Self {
        Self { size, point }
    }

    /// Batched version of [`LinearForm::accumulate`] for many [`UnivariateEvaluation`]s.
    pub fn accumulate_many(evaluators: &[Self], accumulator: &mut [F], scalars: &[F]) {
        assert_eq!(evaluators.len(), scalars.len());
        let Some(size) = evaluators.first().map(|e| e.size) else {
            return;
        };
        assert_eq!(accumulator.len(), size);
        for evaluator in evaluators {
            assert_eq!(evaluator.size, size);
        }
        let points = evaluators.iter().map(|e| e.point).collect::<Vec<F>>();
        let scalars = scalars.to_vec();
        geometric_accumulate(accumulator, scalars, &points);
    }
}

impl<F: Field> LinearForm<F> for UnivariateEvaluation<F> {
    fn size(&self) -> usize {
        self.size
    }

    fn mle_evaluate(&self, point: &[F]) -> F {
        // Multilinear extension of (1, x, x^2, ..) = ⨂_i (1, x^2^i).
        let mut x2i = self.point;
        let mut result = F::ONE;
        for &r in point.iter().rev() {
            // TODO: Why rev?
            result *= (F::ONE - r) + r * x2i;
            x2i.square_in_place();
        }
        result
    }

    /// See also [`Self::accumulate_many`] for a more efficient batched version.
    fn accumulate(&self, accumulator: &mut [F], scalar: F) {
        assert_eq!(accumulator.len(), self.size);
        let mut power = scalar;
        for entry in accumulator {
            *entry += power;
            power *= self.point;
        }
    }
}

impl<M: Embedding> Evaluate<M> for UnivariateEvaluation<M::Target> {
    fn evaluate(&self, embedding: &M, vector: &[M::Source]) -> M::Target {
        mixed_univariate_evaluate(embedding, vector, self.point)
    }
}
