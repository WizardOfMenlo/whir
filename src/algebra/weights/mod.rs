mod covector;
// mod matrix_program;
// mod multilinear_evaluation;
// mod source_vector;
mod subfield_univariate_evaluation;
mod univariate_evaluation;

use ark_ff::Field;
use static_assertions::assert_obj_safe;

pub use self::{
    covector::Covector, subfield_univariate_evaluation::SubfieldUnivariateEvaluation,
    univariate_evaluation::UnivariateEvaluation,
};
use crate::algebra::{
    embedding::{self, Embedding},
    fields,
};

/// Represents a linear functional used in WHIR openings.
///
/// It is some linear functional $𝔽^n → 𝔽$.
pub trait Weights<F: Field> {
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
    fn mle_evaluate(&self, point: &[F]) -> F;

    /// Accumulate the scaled weights *in evaluation basis*.
    ///
    /// This can also be used to retrieve the concrete weight vector (see [`TargetVector::from`]).
    ///
    /// Only used by the prover.
    fn accumulate(&self, accumulator: &mut [F], scalar: F);
}

pub trait Evaluate<M: Embedding>: Weights<M::Target> {
    fn evaluate(&self, embedding: &M, vector: &[M::Source]) -> M::Target;
}

assert_obj_safe!(Weights<fields::Field64>);

assert_obj_safe!(Evaluate<embedding::Basefield<fields::Field64_2>>);
