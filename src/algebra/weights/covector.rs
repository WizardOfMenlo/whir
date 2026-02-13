use ark_ff::Field;

use super::{Evaluate, Weights};
use crate::algebra::{
    mixed_dot, multilinear_extend, ntt::wavelet_transform, scalar_mul_add, Embedding,
};

/// Weights as an explicit (co)vector in the field.
pub struct Covector<F: Field> {
    pub deferred: bool,
    pub vector: Vec<F>,
}

impl<F: Field> Weights<F> for Covector<F> {
    fn deferred(&self) -> bool {
        self.deferred
    }

    fn size(&self) -> usize {
        self.vector.len()
    }

    fn mle_evaluate(&self, point: &[F]) -> F {
        multilinear_extend(&self.vector, point)
    }

    fn accumulate(&self, accumulator: &mut [F], scalar: F) {
        scalar_mul_add(accumulator, scalar, &self.vector);
    }
}

impl<F: Field> Covector<F> {
    pub const fn new(weights: Vec<F>) -> Self {
        Self {
            deferred: true,
            vector: weights,
        }
    }

    /// Any [`Weights<F>`] vector can be converted to a [`Covector<F>`].
    pub fn from(weights: &impl Weights<F>) -> Self {
        let mut vector = vec![F::ZERO; weights.size()];
        weights.accumulate(&mut vector, F::ONE);
        Self {
            deferred: true,
            vector,
        }
    }
}

impl<M: Embedding> Evaluate<M> for Covector<M::Target> {
    fn evaluate(&self, embedding: &M, vector: &[M::Source]) -> M::Target {
        assert_eq!(self.vector.len(), vector.len());
        let mut evals = vector.to_vec();
        wavelet_transform(&mut evals);
        mixed_dot(embedding, &self.vector, &evals)
    }
}
