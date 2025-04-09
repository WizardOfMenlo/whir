use std::{fmt::Debug, ops::Index};

use ark_ff::Field;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
#[cfg(feature = "tracing")]
use tracing::instrument;

use crate::poly_utils::{evals::EvaluationsList, multilinear::MultilinearPoint};

/// Represents a weight function used in polynomial evaluations.
///
/// A `Weights<F>` instance allows evaluating or accumulating weighted contributions
/// to a multilinear polynomial stored in evaluation form. It supports two modes:
///
/// - Evaluation mode: Represents an equality constraint at a specific `MultilinearPoint<F>`.
/// - Linear mode: Represents a set of per-corner weights stored as `EvaluationsList<F>`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "F: CanonicalSerialize + CanonicalDeserialize")]
pub enum Weights<F> {
    /// Represents a weight function that enforces equality constraints at a specific point.
    Evaluation { point: MultilinearPoint<F> },
    /// Represents a weight function defined as a precomputed set of evaluations.
    Linear { weight: EvaluationsList<F> },
}

impl<F: Field> Weights<F> {
    /// Constructs a weight in evaluation mode, enforcing an equality constraint at `point`.
    ///
    /// Given a multilinear polynomial `p(X)`, this weight evaluates:
    ///
    /// \begin{equation}
    /// w(X) = eq_{z}(X)
    /// \end{equation}
    ///
    /// where `eq_z(X)` is the Lagrange interpolation polynomial enforcing `X = z`.
    pub const fn evaluation(point: MultilinearPoint<F>) -> Self {
        Self::Evaluation { point }
    }

    /// Constructs a weight in linear mode, applying a set of precomputed weights.
    ///
    /// This mode allows applying a function `w(X)` stored in `EvaluationsList<F>`:
    ///
    /// \begin{equation}
    /// w(X) = \sum_{i} w_i \cdot X_i
    /// \end{equation}
    ///
    /// where `w_i` are the predefined weight values for each corner of the hypercube.
    pub const fn linear(weight: EvaluationsList<F>) -> Self {
        Self::Linear { weight }
    }

    /// Returns the number of variables involved in the weight function.
    pub fn num_variables(&self) -> usize {
        match self {
            Self::Evaluation { point } => point.num_variables(),
            Self::Linear { weight } => weight.num_variables(),
        }
    }

    /// Accumulates the contribution of the weight function into `accumulator`, scaled by `factor`.
    ///
    /// - In evaluation mode, updates `accumulator` using an equality constraint.
    /// - In linear mode, scales the weight function by `factor` and accumulates it.
    ///
    /// Given a weight function `w(X)` and a factor `λ`, this updates `accumulator` as:
    ///
    /// \begin{equation}
    /// a(X) \gets a(X) + \lambda \cdot w(X)
    /// \end{equation}
    ///
    /// where `a(X)` is the accumulator polynomial.
    ///
    /// **Precondition:**
    /// `accumulator.num_variables()` must match `self.num_variables()`.
    #[cfg_attr(feature = "tracing", instrument(skip_all, fields(num_variables = self.num_variables())))]
    pub fn accumulate(&self, accumulator: &mut EvaluationsList<F>, factor: F) {
        use crate::utils::eval_eq;

        assert_eq!(accumulator.num_variables(), self.num_variables());
        match self {
            Self::Evaluation { point } => {
                eval_eq(&point.0, accumulator.evals_mut(), factor);
            }
            Self::Linear { weight } => {
                #[cfg(feature = "parallel")]
                accumulator
                    .evals_mut()
                    .par_iter_mut()
                    .enumerate()
                    .for_each(|(corner, acc)| {
                        *acc += factor * weight.index(corner);
                    });

                #[cfg(not(feature = "parallel"))]
                accumulator
                    .evals_mut()
                    .iter_mut()
                    .enumerate()
                    .for_each(|(corner, acc)| {
                        *acc += factor * weight.index(corner);
                    });
            }
        }
    }

    /// Computes the weighted sum of a polynomial `p(X)` under the current weight function.
    ///
    /// - In linear mode, computes the inner product between the polynomial values and weights:
    ///
    /// \begin{equation}
    /// \sum_{i} p_i \cdot w_i
    /// \end{equation}
    ///
    /// - In evaluation mode, evaluates `p(X)` at the equality constraint point.
    ///
    /// **Precondition:**
    /// If `self` is in linear mode, `poly.num_variables()` must match `weight.num_variables()`.
    #[cfg_attr(feature = "tracing", instrument(skip_all, fields(num_variables = self.num_variables())))]
    pub fn weighted_sum(&self, poly: &EvaluationsList<F>) -> F {
        match self {
            Self::Linear { weight } => {
                assert_eq!(poly.num_variables(), weight.num_variables());
                #[cfg(not(feature = "parallel"))]
                {
                    poly.evals()
                        .iter()
                        .zip(weight.evals().iter())
                        .map(|(p, w)| *p * *w)
                        .sum()
                }
                #[cfg(feature = "parallel")]
                {
                    poly.evals()
                        .par_iter()
                        .zip(weight.evals().par_iter())
                        .map(|(p, w)| *p * *w)
                        .sum()
                }
            }
            Self::Evaluation { point } => poly.eval_extension(point),
        }
    }
}

/// Represents a system of weighted polynomial constraints.
///
/// Each constraint enforces a relationship between a `Weights<F>` function and a target sum.
/// Constraints can be combined using a random challenge into a single aggregated polynomial.
///
/// **Mathematical Definition:**
/// Given constraints:
///
/// \begin{equation}
/// w_1(X) = s_1, \quad w_2(X) = s_2, \quad \dots, \quad w_k(X) = s_k
/// \end{equation}
///
/// The combined polynomial under challenge $\gamma$ is:
///
/// \begin{equation}
/// W(X) = w_1(X) + \gamma w_2(X) + \gamma^2 w_3(X) + \dots + \gamma^{k-1} w_k(X)
/// \end{equation}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "F: CanonicalSerialize + CanonicalDeserialize")]
pub struct Statement<F> {
    /// Number of variables defining the polynomial space.
    num_variables: usize,
    /// Constraints represented as pairs `(w(X), s)`, where
    /// - `w(X)` is a weighted polynomial function
    /// - `s` is the expected sum.
    pub constraints: Vec<Constraint<F>>,
}

/// A constraint as a weight function and a target sum.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "F: CanonicalSerialize + CanonicalDeserialize")]
pub struct Constraint<F> {
    pub weights: Weights<F>,
    #[serde(with = "crate::ark_serde")]
    pub sum: F,
}

impl<F: Field> Statement<F> {
    /// Creates an empty `Statement<F>` for polynomials with `num_variables` variables.
    pub const fn new(num_variables: usize) -> Self {
        Self {
            num_variables,
            constraints: Vec::new(),
        }
    }

    /// Returns the number of variables defining the polynomial space.
    pub const fn num_variables(&self) -> usize {
        self.num_variables
    }

    /// Adds a constraint `(w(X), s)` to the system.
    ///
    /// **Precondition:**
    /// The number of variables in `w(X)` must match `self.num_variables`.
    pub fn add_constraint(&mut self, weights: Weights<F>, sum: F) {
        assert_eq!(weights.num_variables(), self.num_variables());
        self.constraints.push(Constraint { weights, sum });
    }

    /// Inserts a constraint `(w(X), s)` at the front of the system.
    pub fn add_constraint_in_front(&mut self, weights: Weights<F>, sum: F) {
        assert_eq!(weights.num_variables(), self.num_variables());
        self.constraints.insert(0, Constraint { weights, sum });
    }

    /// Inserts multiple constraints at the front of the system.
    pub fn add_constraints_in_front(&mut self, constraints: Vec<(Weights<F>, F)>) {
        for (weights, _) in &constraints {
            assert_eq!(weights.num_variables(), self.num_variables());
        }
        self.constraints.splice(
            0..0,
            constraints
                .into_iter()
                .map(|(weights, sum)| Constraint { weights, sum }),
        );
    }

    /// Combines all constraints into a single aggregated polynomial using a challenge.
    ///
    /// Given a random challenge $\gamma$, the new polynomial is:
    ///
    /// \begin{equation}
    /// W(X) = w_1(X) + \gamma w_2(X) + \gamma^2 w_3(X) + \dots + \gamma^{k-1} w_k(X)
    /// \end{equation}
    ///
    /// with the combined sum:
    ///
    /// \begin{equation}
    /// S = s_1 + \gamma s_2 + \gamma^2 s_3 + \dots + \gamma^{k-1} s_k
    /// \end{equation}
    ///
    /// **Returns:**
    /// - `EvaluationsList<F>`: The combined polynomial `W(X)`.
    /// - `F`: The combined sum `S`.
    #[cfg_attr(feature = "tracing", instrument(skip_all, fields(num_variables = self.num_variables(), num_constraints = self.constraints.len())))]
    pub fn combine(&self, challenge: F) -> (EvaluationsList<F>, F) {
        let evaluations_vec = vec![F::ZERO; 1 << self.num_variables];
        let mut combined_evals = EvaluationsList::new(evaluations_vec);
        let (combined_sum, _) = self.constraints.iter().fold(
            (F::ZERO, F::ONE),
            |(mut acc_sum, gamma_pow), constraint| {
                constraint
                    .weights
                    .accumulate(&mut combined_evals, gamma_pow);
                acc_sum += constraint.sum * gamma_pow;
                (acc_sum, gamma_pow * challenge)
            },
        );

        (combined_evals, combined_sum)
    }
}

/// Represents a verifier's interpretation of weighted polynomial constraints.
///
/// Used in the verification phase to process and evaluate constraints in a simplified form.
/// Can either:
/// - Directly evaluate at a given `point`
/// - Represent a linear weight with an optional precomputed term.
///
/// **Mathematical definition:**
/// - If `w(X)` is evaluated at a fixed point $p$, we store only $p$.
/// - If `w(X)` is a linear combination, we track the number of variables and an optional
///   precomputed term.
#[derive(Clone, Debug)]
pub enum VerifierWeights<F> {
    /// Direct evaluation at a specific point $p$.
    Evaluation { point: MultilinearPoint<F> },
    /// Linear weight representation over `num_variables` variables.
    /// May store a precomputed term for efficiency.
    Linear {
        num_variables: usize,
        term: Option<F>,
    },
}

impl<F: Field> VerifierWeights<F> {
    /// Constructs an evaluation weight at a fixed point.
    pub const fn evaluation(point: MultilinearPoint<F>) -> Self {
        Self::Evaluation { point }
    }

    /// Constructs a linear weight representation.
    ///
    /// - `num_variables`: The number of variables in the polynomial space.
    /// - `term`: An optional precomputed term for efficiency.
    pub const fn linear(num_variables: usize, term: Option<F>) -> Self {
        Self::Linear {
            num_variables,
            term,
        }
    }

    /// Returns the number of variables in the weight.
    ///
    /// - For an evaluation weight, this is the number of variables in `point`.
    /// - For a linear weight, this is explicitly stored.
    pub fn num_variables(&self) -> usize {
        match self {
            Self::Evaluation { point } => point.num_variables(),
            Self::Linear { num_variables, .. } => *num_variables,
        }
    }

    /// Computes the weight function evaluation under a given randomness.
    ///
    /// - In evaluation mode, it computes the equality polynomial `eq_poly_outside` at the provided
    ///   `folding_randomness`, enforcing the constraint at a specific point.
    /// - In linear mode, it returns the precomputed term if available.
    ///
    /// **Mathematical Definition:**
    /// - If `w(X)` is an evaluation weight at `p`, then:
    ///
    /// \begin{equation}
    /// w(X) = eq_p(X)
    /// \end{equation}
    ///
    /// where `eq_p(X)` is the Lagrange interpolation polynomial enforcing `X = p`.
    ///
    /// - If `w(X)` is a linear weight, it simply returns the stored `term`.
    ///
    /// **Precondition:**
    /// - If `self` is in linear mode, `term` must be `Some(F)`, otherwise the behavior is
    ///   undefined.
    pub fn compute(&self, folding_randomness: &MultilinearPoint<F>) -> F {
        match self {
            Self::Evaluation { point } => point.eq_poly_outside(folding_randomness),
            Self::Linear { term, .. } => term.unwrap(),
        }
    }
}

/// Represents a verifier's constraint system in a statement.
///
/// This structure is used to verify a given statement by storing and processing
/// a list of constraints. Each constraint consists of:
/// - `VerifierWeights<F>`: A weight applied to a polynomial.
/// - `F`: The expected sum (result of applying the constraint).
///
/// **Mathematical Formulation:**
/// Given a set of constraints:
///
/// \begin{equation}
/// \sum_{i} w_i(X) \cdot p_i(X) = s_i
/// \end{equation}
///
/// This struct stores and organizes these constraints for efficient verification.
#[derive(Clone, Debug, Default)]
pub struct StatementVerifier<F> {
    /// The number of variables in the statement.
    num_variables: usize,
    /// The list of constraints in the form `(weights, sum)`.
    pub constraints: Vec<(VerifierWeights<F>, F)>,
}

impl<F: Field> StatementVerifier<F> {
    /// Creates a new statement verifier for a given number of variables.
    const fn new(num_variables: usize) -> Self {
        Self {
            num_variables,
            constraints: Vec::new(),
        }
    }

    /// Returns the number of variables in the statement.
    pub const fn num_variables(&self) -> usize {
        self.num_variables
    }

    /// Adds a new constraint `(weights, sum)` to the verifier.
    ///
    /// Ensures that the constraint has the correct number of variables.
    pub fn add_constraint(&mut self, weights: VerifierWeights<F>, sum: F) {
        assert_eq!(weights.num_variables(), self.num_variables());
        self.constraints.push((weights, sum));
    }

    /// Inserts a constraint `(weights, sum)` at the front of the constraint list.
    pub fn add_constraint_in_front(&mut self, weights: VerifierWeights<F>, sum: F) {
        assert_eq!(weights.num_variables(), self.num_variables());
        self.constraints.insert(0, (weights, sum));
    }

    /// Inserts multiple constraints at the front of the constraint list.
    pub fn add_constraints_in_front(&mut self, constraints: Vec<(VerifierWeights<F>, F)>) {
        for (weights, _) in &constraints {
            assert_eq!(weights.num_variables(), self.num_variables());
        }
        self.constraints.splice(0..0, constraints);
    }

    /// Converts a `Statement<F>` into a `StatementVerifier<F>`, mapping `Weights<F>` into
    /// `VerifierWeights<F>`.
    ///
    /// This is used during the verification phase to simplify constraint handling.
    pub fn from_statement(statement: &Statement<F>) -> Self {
        let mut verifier = Self::new(statement.num_variables());
        for constraint in &statement.constraints {
            match &constraint.weights {
                Weights::Linear { weight, .. } => {
                    verifier.add_constraint(
                        VerifierWeights::linear(weight.num_variables(), None),
                        constraint.sum,
                    );
                }
                Weights::Evaluation { point } => {
                    verifier
                        .add_constraint(VerifierWeights::evaluation(point.clone()), constraint.sum);
                }
            }
        }
        verifier
    }
}

#[cfg(test)]
mod tests {
    use ark_ff::AdditiveGroup;

    use super::*;
    use crate::{crypto::fields::Field64, utils::eval_eq};

    #[test]
    fn test_weights_evaluation() {
        // Define a point in the multilinear space
        let point = MultilinearPoint(vec![Field64::ONE, Field64::ZERO]);
        let weight = Weights::evaluation(point);

        // The number of variables in the weight should match the number of variables in the point
        assert_eq!(weight.num_variables(), 2);
    }

    #[test]
    fn test_weights_linear() {
        // Define a list of evaluation values
        let evals = EvaluationsList::new(vec![
            Field64::ONE,
            Field64::from(2),
            Field64::from(3),
            Field64::from(3),
        ]);
        let weight = Weights::linear(evals);

        // The number of variables in the weight should match the number of variables in evals
        assert_eq!(weight.num_variables(), 2);
    }

    #[test]
    fn test_weighted_sum_evaluation() {
        // Define polynomial evaluations at different points
        let e0 = Field64::from(3);
        let e1 = Field64::from(5);
        let evals = EvaluationsList::new(vec![e0, e1]);

        // Define an evaluation weight at a specific point
        let point = MultilinearPoint(vec![Field64::ONE]);
        let weight = Weights::evaluation(point);

        // Expected result: polynomial evaluation at the given point
        let expected = e1;

        assert_eq!(weight.weighted_sum(&evals), expected);
    }

    #[test]
    fn test_weighted_sum_linear() {
        // Define polynomial evaluations
        let e0 = Field64::ONE;
        let e1 = Field64::from(2);
        let evals = EvaluationsList::new(vec![e0, e1]);

        // Define linear weights
        let w0 = Field64::from(2);
        let w1 = Field64::from(3);
        let weight_list = EvaluationsList::new(vec![w0, w1]);
        let weight = Weights::linear(weight_list);

        // Compute expected result manually
        //
        // \begin{equation}
        // \sum_{i} e_i \cdot w_i = e_0 \cdot w_0 + e_1 \cdot w_1
        // \end{equation}
        let expected = e0 * w0 + e1 * w1;

        assert_eq!(weight.weighted_sum(&evals), expected);
    }

    #[test]
    fn test_accumulate_linear() {
        // Initialize an empty accumulator
        let mut accumulator = EvaluationsList::new(vec![Field64::ZERO, Field64::ZERO]);

        // Define weights
        let w0 = Field64::from(2);
        let w1 = Field64::from(3);
        let weight_list = EvaluationsList::new(vec![w0, w1]);
        let weight = Weights::linear(weight_list);

        // Define a multiplication factor
        let factor = Field64::from(4);

        // Accumulate weighted values
        weight.accumulate(&mut accumulator, factor);

        // Expected result:
        //
        // \begin{equation}
        // acc_i = factor \cdot w_i
        // \end{equation}
        let expected = vec![
            w0 * factor, // 2 * 4 = 8
            w1 * factor, // 3 * 4 = 12
        ];

        assert_eq!(accumulator.evals(), &expected);
    }

    #[test]
    fn test_accumulate_evaluation() {
        // Initialize an empty accumulator
        let mut accumulator = EvaluationsList::new(vec![Field64::ZERO, Field64::ZERO]);

        // Define an evaluation point
        let point = MultilinearPoint(vec![Field64::ONE]);
        let weight = Weights::evaluation(point.clone());

        // Define a multiplication factor
        let factor = Field64::from(5);

        // Accumulate weighted values
        weight.accumulate(&mut accumulator, factor);

        // Compute expected result manually
        let mut expected = vec![Field64::ZERO, Field64::ZERO];
        eval_eq(&point.0, &mut expected, factor);

        assert_eq!(accumulator.evals(), &expected);
    }

    #[test]
    fn test_statement_combine() {
        // Create a new statement with 1 variable
        let mut statement = Statement::new(1);

        // Define weights
        let w0 = Field64::from(3);
        let w1 = Field64::from(5);
        let weight_list = EvaluationsList::new(vec![w0, w1]);
        let weight = Weights::linear(weight_list);

        // Define sum constraint
        let sum = Field64::from(7);
        statement.add_constraint(weight, sum);

        // Define a challenge factor
        let challenge = Field64::from(2);

        // Compute combined evaluations and sum
        let (combined_evals, combined_sum) = statement.combine(challenge);

        // Expected evaluations should match the accumulated weights
        let expected_combined_evals = vec![
            w0, // 3
            w1, // 5
        ];

        // Expected sum remains unchanged since there is only one constraint
        let expected_combined_sum = sum;

        assert_eq!(combined_evals.evals(), &expected_combined_evals);
        assert_eq!(combined_sum, expected_combined_sum);
    }

    #[test]
    fn test_statement_verifier_from_statement() {
        // Create a new statement with 2 variables
        let mut statement = Statement::new(2);

        // Define weights
        let w0 = Field64::from(3);
        let w1 = Field64::from(4);
        let w2 = Field64::from(5);
        let w3 = Field64::from(6);
        let weight_list = EvaluationsList::new(vec![w0, w1, w2, w3]);
        let weight = Weights::linear(weight_list);

        // Define sum constraint
        let sum = Field64::from(10);
        statement.add_constraint(weight, sum);

        // Convert statement to verifier format
        let verifier = StatementVerifier::from_statement(&statement);

        // Ensure verifier retains the same number of variables
        assert_eq!(verifier.num_variables(), statement.num_variables());

        // Ensure the constraint count matches
        assert_eq!(verifier.constraints.len(), 1);
    }

    #[test]
    fn test_statement_with_multiple_constraints() {
        // Create a new statement with 2 variables
        let mut statement = Statement::new(2);

        // Define weights for first constraint (2 variables => 4 evaluations)
        let w0 = Field64::from(1);
        let w1 = Field64::from(2);
        let w2 = Field64::from(3);
        let w3 = Field64::from(4);
        let weight_list1 = EvaluationsList::new(vec![w0, w1, w2, w3]);
        let weight1 = Weights::linear(weight_list1);

        // Define weights for second constraint (also 2 variables => 4 evaluations)
        let w4 = Field64::from(5);
        let w5 = Field64::from(6);
        let w6 = Field64::from(7);
        let w7 = Field64::from(8);
        let weight_list2 = EvaluationsList::new(vec![w4, w5, w6, w7]);
        let weight2 = Weights::linear(weight_list2);

        // Define sum constraints
        let sum1 = Field64::from(5);
        let sum2 = Field64::from(7);

        // Ensure both weight lists match the expected number of variables
        assert_eq!(weight1.num_variables(), 2);
        assert_eq!(weight2.num_variables(), 2);

        // Add constraints to the statement
        statement.add_constraint(weight1, sum1);
        statement.add_constraint(weight2, sum2);

        // Define a challenge factor
        let challenge = Field64::from(2);

        // Compute combined evaluations and sum
        let (combined_evals, combined_sum) = statement.combine(challenge);

        // Expected evaluations:
        //
        // \begin{equation}
        // combined = weight_1 + challenge \cdot weight_2
        // \end{equation}
        let expected_combined_evals = vec![
            w0 + challenge * w4, // 1 + 2 * 5 = 11
            w1 + challenge * w5, // 2 + 2 * 6 = 14
            w2 + challenge * w6, // 3 + 2 * 7 = 17
            w3 + challenge * w7, // 4 + 2 * 8 = 20
        ];

        // Expected sum:
        //
        // \begin{equation}
        // S_{combined} = S_1 + challenge \cdot S_2
        // \end{equation}
        let expected_combined_sum = sum1 + challenge * sum2; // 5 + 2 * 7 = 19

        assert_eq!(combined_evals.evals(), &expected_combined_evals);
        assert_eq!(combined_sum, expected_combined_sum);
    }

    #[test]
    fn test_compute_evaluation_weight() {
        // Define an evaluation weight at a specific point
        let point = MultilinearPoint(vec![Field64::from(3)]);
        let weight = VerifierWeights::evaluation(point.clone());

        // Define a randomness point for folding
        let folding_randomness = MultilinearPoint(vec![Field64::from(2)]);

        // Expected result is the evaluation of eq_poly_outside at the given randomness
        let expected = point.eq_poly_outside(&folding_randomness);

        assert_eq!(weight.compute(&folding_randomness), expected);
    }

    #[test]
    fn test_compute_linear_weight_with_term() {
        // Define a linear weight with a precomputed term
        let term = Field64::from(7);
        let weight = VerifierWeights::linear(2, Some(term));

        // Folding randomness should have no effect in linear mode
        let folding_randomness = MultilinearPoint(vec![Field64::from(3), Field64::from(4)]);

        // Expected result is the stored term
        assert_eq!(weight.compute(&folding_randomness), term);
    }

    #[test]
    #[should_panic]
    fn test_compute_linear_weight_without_term() {
        // Define a linear weight without a precomputed term
        let weight = VerifierWeights::linear(2, None);

        // Folding randomness is irrelevant in this case
        let folding_randomness = MultilinearPoint(vec![Field64::from(3), Field64::from(4)]);

        // This should panic due to an attempt to unwrap a None value
        weight.compute(&folding_randomness);
    }

    #[test]
    #[allow(clippy::redundant_clone)]
    fn test_compute_evaluation_weight_identity() {
        // Define an evaluation weight at a specific point
        let point = MultilinearPoint(vec![Field64::ONE, Field64::ZERO]);

        // Folding randomness is the same as the point itself
        let folding_randomness = point.clone();
        let weight = VerifierWeights::evaluation(point.clone());

        // Expected result should be identity for equality polynomial
        let expected = point.eq_poly_outside(&folding_randomness);
        assert_eq!(weight.compute(&folding_randomness), expected);
    }
}
