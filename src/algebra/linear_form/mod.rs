//! Linear Forms that commited vectors can be openened against.

mod covector;
mod multilinear_evaluation;
mod subfield_univariate_evaluation;
mod univariate_evaluation;

use ark_ff::Field;
use static_assertions::assert_obj_safe;

pub use self::{
    covector::Covector, multilinear_evaluation::MultilinearEvaluation,
    subfield_univariate_evaluation::SubfieldUnivariateEvaluation,
    univariate_evaluation::UnivariateEvaluation,
};
use crate::algebra::{
    embedding::{self, Embedding},
    fields,
};

/// Represents a linear function $𝔽^n → 𝔽$ used in WHIR openings.
///
/// Note that the trait does not contain a method to actually evaluate the linear form, for that
/// see the [`Evaluate`] trait.
pub trait LinearForm<F: Field> {
    /// The dimension of the domain of this linear form.
    fn size(&self) -> usize;

    /// Indicate if the verifier should evaluate this directly or defer it to the caller.
    ///
    /// If this returns `true`, the verifier will not call [`mle_evaluate`] but instead `verify`
    /// will return `(point, value)` pairs and it becomes **the callers responsibility** to verify
    /// `self.mle_evaluate(point) == value`. This allows the verifier to use more efficient means
    /// than direct evaluation (e.g. the Spartan Spark protocol).
    fn deferred(&self) -> bool;

    /// Evaluate the linear form as a multi-linear extension in a random point.
    ///
    /// Specifically the computed value should be the linear functional applied to vector given by
    /// the tensor product $a = a_0 ⊗ a_1 ⊗ ⋯ ⊗ a_(k-1)$ where $k ≥ ⌈log_2 size⌉$ and
    /// $a_i = (1-point_i, point_i)$ truncated to `self.size()`.
    ///
    /// If `self.deferred() == false` it is called by the verifier, otherwise it is called by
    /// the prover.
    fn mle_evaluate(&self, point: &[F]) -> F;

    /// Accumulate the covector representation of the linear form.
    ///
    /// Take $w ∈ 𝔽^n$ such that evaluating the linear form on $v ∈ 𝔽^n$ equals the inner
    /// product $⟨w,v⟩$. Then this function computes $accumulator_i += scalar · w_i$.
    ///
    /// This function is only called by the prover.
    fn accumulate(&self, accumulator: &mut [F], scalar: F);
}

/// A linear form that can be evaluated on a subfield vector.
///
/// In particular [`Evaluate<Identity<F>>`] allows evaluating the linear form
/// on its native field.
pub trait Evaluate<M: Embedding>: LinearForm<M::Target> {
    /// Evaluate the linear form on a subfield vector.
    ///
    /// - `self` is a linear form $𝔽^n → 𝔽$.
    /// - `embedding` is an embedding $𝔾 → 𝔽$.
    /// - `vector` is a vector in $𝔾^n$.
    ///
    /// *TODO*. This is actually false at the moment, it takes `vector` in coefficient basis.
    ///
    fn evaluate(&self, embedding: &M, vector: &[M::Source]) -> M::Target;
}

assert_obj_safe!(LinearForm<fields::Field64>);

assert_obj_safe!(Evaluate<embedding::Basefield<fields::Field64_2>>);
