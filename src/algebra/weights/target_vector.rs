use ark_ff::{AdditiveGroup, Field};

use super::Weights;
use crate::algebra::{
    embedding::{Embedding, Identity},
    mixed_dot, mixed_scalar_mul_add,
    polynomials::multilinear_extend,
};

/// Weights as an explicit vector in the target field.
pub struct TargetVector<M: Embedding> {
    pub deferred: bool,
    pub embedding: M,
    pub vector: Vec<M::Target>,
}

impl<M: Embedding> Weights<M> for TargetVector<M> {
    fn embedding(&self) -> &M {
        &self.embedding
    }

    fn deferred(&self) -> bool {
        self.deferred
    }

    fn size(&self) -> usize {
        self.vector.len()
    }

    fn mle_evaluate(&self, point: &[M::Target]) -> M::Target {
        multilinear_extend(&self.vector, &point)
    }

    fn inner_product(&self, vector: &[M::Source]) -> M::Target {
        mixed_dot(&self.embedding, &self.vector, &vector)
    }

    fn accumulate(&self, accumulator: &mut [M::Target], scalar: M::Target) {
        mixed_scalar_mul_add(
            &Identity::<M::Target>::new(),
            accumulator,
            scalar,
            &self.vector,
        );
    }
}

impl<M: Embedding + Clone> TargetVector<M> {
    /// Any [`Weights`] vector can be converted to a [`TargetVector`].
    pub fn from(weights: &impl Weights<M>) -> Self {
        let mut vector = vec![M::Target::ZERO; weights.size()];
        weights.accumulate(&mut vector, M::Target::ONE);
        Self {
            embedding: weights.embedding().clone(),
            deferred: true,
            vector,
        }
    }
}

impl<M: Embedding + Default> TargetVector<M> {
    pub fn new(weights: Vec<M::Target>) -> Self {
        Self {
            embedding: M::default(),
            deferred: true,
            vector: weights,
        }
    }
}
