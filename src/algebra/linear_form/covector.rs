use ark_ff::Field;

use super::{Evaluate, LinearForm};
use crate::algebra::{
    mixed_dot, multilinear_extend, ntt::wavelet_transform, scalar_mul_add, Embedding,
};

/// Linear form as an explicit covector over the field.
pub struct Covector<F: Field> {
    pub deferred: bool,
    pub vector: Vec<F>,
}

impl<F: Field> LinearForm<F> for Covector<F> {
    fn size(&self) -> usize {
        self.vector.len()
    }

    fn deferred(&self) -> bool {
        self.deferred
    }

    fn mle_evaluate(&self, point: &[F]) -> F {
        multilinear_extend(&self.vector, point)
    }

    fn accumulate(&self, accumulator: &mut [F], scalar: F) {
        scalar_mul_add(accumulator, scalar, &self.vector);
    }
}

impl<F: Field> Covector<F> {
    pub const fn new(vector: Vec<F>) -> Self {
        Self {
            deferred: true,
            vector,
        }
    }

    /// Any [`LinearForm<F>`] can be converted to a [`Covector<F>`].
    pub fn from(linear_form: &dyn LinearForm<F>) -> Self {
        let mut vector = vec![F::ZERO; linear_form.size()];
        linear_form.accumulate(&mut vector, F::ONE);
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
