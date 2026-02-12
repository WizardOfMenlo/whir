mod matrix_program;
mod multilinear_evaluation;
mod source_vector;
mod target_vector;
mod univariate_evaluation;

use static_assertions::assert_obj_safe;

pub use self::{
    source_vector::SourceVector, target_vector::TargetVector,
    univariate_evaluation::UnivariateEvaluation,
};
use crate::algebra::{
    embedding::{self, Embedding},
    fields,
};

/// Represents a weight vector used in WHIR openings.
pub trait Weights<M: Embedding> {
    /// The embedding this weights vector uses.
    fn embedding(&self) -> &M;

    /// Indicate if the verifier should evaluate this directly or defer it to the caller.
    fn deferred(&self) -> bool;

    /// The number of coefficients in the vector.
    fn size(&self) -> usize;

    /// Evaluate the weights vector as a multi-linear extension in a random point.
    ///
    /// Used by the verifier for non-deferred weights.
    ///
    /// Should be such that evaluating on the boolean hypercube (TODO: In what order?)
    /// recovers the weights vector.
    ///
    /// See e.g. <https://eprint.iacr.org/2024/1103.pdf>
    fn mle_evaluate(&self, point: &[M::Target]) -> M::Target;

    /// Compute the inner product with a vector.
    ///
    /// Only used by the prover. (and then only for testing?)
    fn inner_product(&self, vector: &[M::Source]) -> M::Target;

    /// Accumulate the scaled weights.
    ///
    /// This can also be used to retrieve the concrete weight vector (see [`TargetVector::from`]).
    ///
    /// Only used by the prover.
    fn accumulate(&self, accumulator: &mut [M::Target], scalar: M::Target);
}

assert_obj_safe!(Weights<embedding::Identity<fields::Field64>>);
