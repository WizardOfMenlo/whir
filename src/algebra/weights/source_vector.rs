use super::Weights;
use crate::algebra::{
    dot,
    embedding::Embedding,
    mixed_scalar_mul_add,
    polynomials::{mixed_eval_multilinear, MultilinearPoint},
};

/// Weights as an explicit vector in the source field.
pub struct SourceVector<M: Embedding> {
    pub deffered: bool,
    pub embedding: M,
    pub vector: Vec<M::Source>,
}

impl<M: Embedding> Weights<M> for SourceVector<M> {
    fn embedding(&self) -> &M {
        &self.embedding
    }

    fn deferred(&self) -> bool {
        self.deffered
    }

    fn size(&self) -> usize {
        self.vector.len()
    }

    fn mle_evaluate(&self, point: MultilinearPoint<M::Target>) -> M::Target {
        mixed_eval_multilinear(&self.embedding, &self.vector, &point.0)
    }

    fn inner_product(&self, vector: &[M::Source]) -> M::Target {
        self.embedding.map(dot(&self.vector, &vector))
    }

    fn accumulate(&self, accumulator: &mut [M::Target], scalar: M::Target) {
        mixed_scalar_mul_add(&self.embedding, accumulator, scalar, &self.vector);
    }
}

impl<M: Embedding + Default> SourceVector<M> {
    pub fn new(weights: Vec<M::Source>) -> Self {
        Self {
            embedding: M::default(),
            deffered: true,
            vector: weights,
        }
    }
}
