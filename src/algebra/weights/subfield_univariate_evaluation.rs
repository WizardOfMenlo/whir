use ark_ff::Field;

use super::Weights;
use crate::{
    algebra::{
        embedding::{self, Embedding},
        mixed_univariate_evaluate, univariate_evaluate,
        weights::{Evaluate, UnivariateEvaluation},
    },
    utils::zip_strict,
};

/// Weights vector to represent univariate polynomial evaluation.
///
/// Identical to [`UnivariateEvaluation`], but optimized for when the evaluation point
/// is in a subfield.
pub struct SubfieldUnivariateEvaluation<M: Embedding> {
    pub embedding: M,

    /// Univariate evaluation doesn't have an inherent size, so we need to store one.
    pub size: usize,

    /// The point to evaluate in.
    pub point: M::Source,
}

impl<M: Embedding> SubfieldUnivariateEvaluation<M> {
    pub fn new(embedding: &M, point: M::Source, size: usize) -> Self {
        Self {
            embedding: embedding.clone(),
            size,
            point,
        }
    }

    /// Lift to an evaluation over a target field point
    pub fn lift(&self) -> UnivariateEvaluation<M::Target> {
        UnivariateEvaluation::new(self.embedding.map(self.point), self.size)
    }

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
                *power = weights.embedding.mixed_mul(*power, weights.point);
            }
        }
    }
}

impl<M: Embedding> Weights<M::Target> for SubfieldUnivariateEvaluation<M> {
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
                    acc * (self.embedding.mixed_mul(r, pow2k)
                        + self
                            .embedding
                            .mixed_mul(M::Target::ONE - r, M::Source::ONE - pow2k)),
                    pow2k.square(),
                )
            })
            .0
    }

    /// See also [`accumulate_many`] for a more efficient batched version.
    fn accumulate(&self, accumulator: &mut [M::Target], scalar: M::Target) {
        let mut power = scalar;
        for accumulator in accumulator {
            *accumulator += scalar;
            power = self.embedding.mixed_mul(power, self.point);
        }
    }
}

/// Evaluate over a vector in the subfield.
impl<M: Embedding> Evaluate<M> for SubfieldUnivariateEvaluation<M> {
    fn evaluate(&self, embedding: &M, vector: &[M::Source]) -> M::Target {
        assert_eq!(&self.embedding, embedding);
        self.embedding.map(univariate_evaluate(vector, self.point))
    }
}

/// Evaluate over a tower.
///
/// For e.g. M31 we want $𝔽 ↪︎ 𝔽^2 ↪︎ 𝔽^6$ where the evaluation point is in $𝔽^2$ and the vector in $𝔽^n$.
impl<N: Embedding, M: Embedding<Source = N::Target>> Evaluate<embedding::Compose<N, M>>
    for SubfieldUnivariateEvaluation<M>
{
    fn evaluate(&self, embedding: &embedding::Compose<N, M>, vector: &[N::Source]) -> M::Target {
        assert_eq!(embedding.outer(), &self.embedding);
        self.embedding.map(mixed_univariate_evaluate(
            embedding.inner(),
            vector,
            self.point,
        ))
    }
}
