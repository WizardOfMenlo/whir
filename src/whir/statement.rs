use std::{fmt::Debug, ops::Index};

use ark_ff::Field;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
#[cfg(feature = "tracing")]
use tracing::instrument;

use crate::poly_utils::{
    coeffs::CoefficientList,
    evals::{geometric_till, EvaluationsList},
    multilinear::MultilinearPoint,
};

/// Represents a weight function used in polynomial evaluations.
///
/// A `Weights<F>` instance allows evaluating or accumulating weighted contributions
/// to a multilinear polynomial stored in evaluation form. It supports two modes:
///
/// - Evaluation mode: Represents an equality constraint at a specific `MultilinearPoint<F>`.
/// - Linear mode: Represents a set of per-corner weights stored as `EvaluationsList<F>`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "F: Field + CanonicalSerialize + CanonicalDeserialize")]
pub enum Weights<F> {
    /// Represents a weight function that enforces equality constraints at a specific point.
    Evaluation { point: MultilinearPoint<F> },
    /// Represents a weight function defined as a precomputed set of evaluations.
    Linear { weight: EvaluationsList<F> },
    /// Represents a weight function which is a geometric progression.
    Geometric {
        #[serde(with = "crate::ark_serde::field")]
        /// The multiplicative factor of the geometric progression.
        a: F,
        /// The number of terms in the geometric progression, post which all terms are zero.
        n: usize,
        /// Represents the geometric progression as a set of evaluations.
        weight: EvaluationsList<F>,
    },
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

    /// Construct weights for a univariate evaluation
    pub fn univariate(point: F, size: usize) -> Self {
        Self::Evaluation {
            point: MultilinearPoint::expand_from_univariate(point, size),
        }
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

    /// Similar to linear mode, but the weights are stored in a geometric progression.
    pub const fn geometric(a: F, n: usize, weight: EvaluationsList<F>) -> Self {
        Self::Geometric { a, n, weight }
    }

    /// Returns the number of variables involved in the weight function.
    pub fn num_variables(&self) -> usize {
        match self {
            Self::Evaluation { point } => point.num_variables(),
            Self::Linear { weight } | Self::Geometric { weight, .. } => weight.num_variables(),
        }
    }

    /// Evaluate the weighted sum with a polynomial in coefficient form.
    pub fn evaluate(&self, poly: &CoefficientList<F>) -> F {
        assert_eq!(self.num_variables(), poly.num_variables());
        match self {
            Self::Evaluation { point } => poly.evaluate(point),
            Self::Linear { weight } | Self::Geometric { weight, .. } => {
                let poly: EvaluationsList<F> = poly.clone().into();

                // We intentionally avoid parallel iterators here because this function is only called by the verifier,
                // which is assumed to run on a lightweight device.
                weight
                    .evals()
                    .iter()
                    .zip(poly.evals())
                    .map(|(&w, &p)| w * p)
                    .sum()
            }
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
            Self::Linear { weight } | Self::Geometric { weight, .. } => {
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
            Self::Linear { weight } | Self::Geometric { weight, .. } => {
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

    /// Computes the weight function evaluation under a given randomness.
    pub fn compute(&self, folding_randomness: &MultilinearPoint<F>) -> F {
        match self {
            Self::Evaluation { point } => point.eq_poly_outside(folding_randomness),
            Self::Linear { weight } => weight.eval_extension(folding_randomness),
            Self::Geometric { a, n, .. } => geometric_till(*a, *n, &folding_randomness.0),
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
#[serde(bound = "F: Field + CanonicalSerialize + CanonicalDeserialize")]
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
#[serde(bound = "F: Field + CanonicalSerialize + CanonicalDeserialize")]
pub struct Constraint<F> {
    pub weights: Weights<F>,

    #[serde(with = "crate::ark_serde::field")]
    pub sum: F,

    /// When set, the weight evaluation will not be checked by the WHIR verifier,
    /// but instead deferred to the caller.
    ///
    /// The whir verification will be done using a prover provided hint of the evaluation.
    pub defer_evaluation: bool,
}

impl<F: Field> Constraint<F> {
    /// Verify if a polynomial (in coefficient form) satisfies the constraint.
    pub fn verify(&self, poly: &CoefficientList<F>) -> bool {
        self.weights.evaluate(poly) == self.sum
    }
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
        let defer_evaluation = match &weights {
            Weights::Evaluation { .. } => false,
            Weights::Linear { .. } | Weights::Geometric { .. } => true,
        };
        self.constraints.push(Constraint {
            weights,
            sum,
            defer_evaluation,
        });
    }

    /// Inserts a constraint `(w(X), s)` at the front of the system.
    pub fn add_constraint_in_front(&mut self, weights: Weights<F>, sum: F) {
        assert_eq!(weights.num_variables(), self.num_variables());
        let defer_evaluation = match &weights {
            Weights::Evaluation { .. } => false,
            Weights::Linear { .. } | Weights::Geometric { .. } => true,
        };
        self.constraints.insert(
            0,
            Constraint {
                weights,
                sum,
                defer_evaluation,
            },
        );
    }

    /// Inserts multiple constraints at the front of the system.
    pub fn add_constraints_in_front(&mut self, constraints: Vec<(Weights<F>, F)>) {
        for (weights, _) in &constraints {
            assert_eq!(weights.num_variables(), self.num_variables());
        }
        self.constraints.splice(
            0..0,
            constraints.into_iter().map(|(weights, sum)| {
                let defer_evaluation = match &weights {
                    Weights::Evaluation { .. } => false,
                    Weights::Linear { .. } | Weights::Geometric { .. } => true,
                };
                Constraint {
                    weights,
                    sum,
                    defer_evaluation,
                }
            }),
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
    fn test_weights_geometric() {
        // Define a geometric progression
        let a = Field64::from(2);
        let n = 2;
        let evals = EvaluationsList::new(vec![Field64::ONE, a, a * a, Field64::ZERO]);
        let weight = Weights::geometric(a, n, evals);

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
    fn test_accumulate_geometric() {
        // Initialize an empty accumulator
        let mut accumulator = EvaluationsList::new(vec![
            Field64::from(2),
            Field64::from(3),
            Field64::from(4),
            Field64::from(5),
        ]);

        // Define weights
        let a = Field64::from(2);
        let n = 3;
        let weight_list = EvaluationsList::new(vec![Field64::ONE, a, a * a, Field64::ZERO]);
        let weight = Weights::geometric(a, n, weight_list);

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
            Field64::from(2) + Field64::ONE * factor, // 2 + 1 * 4 = 6
            Field64::from(3) + a * factor,            // 3 + 2 * 4 = 11
            Field64::from(4) + a * a * factor,        // 4 + 2 * 2 * 4 = 20
            Field64::from(5),                         // 5 + 0 * 4 = 5
        ];

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

        // Define weights for third constraint (also 2 variables => 4 evaluations)
        let a = Field64::from(2);
        let n = 3;
        let weight_list3 = EvaluationsList::new(vec![Field64::ONE, a, a * a, Field64::ZERO]);
        let weight3 = Weights::geometric(a, n, weight_list3);

        // Define sum constraints
        let sum1 = Field64::from(5);
        let sum2 = Field64::from(7);
        let sum3 = Field64::from(9);

        // Ensure both weight lists match the expected number of variables
        assert_eq!(weight1.num_variables(), 2);
        assert_eq!(weight2.num_variables(), 2);
        assert_eq!(weight3.num_variables(), 2);

        // Add constraints to the statement
        statement.add_constraint(weight1, sum1);
        statement.add_constraint(weight2, sum2);
        statement.add_constraint(weight3, sum3);

        // Define a challenge factor
        let challenge = Field64::from(2);

        // Compute combined evaluations and sum
        let (combined_evals, combined_sum) = statement.combine(challenge);

        // Expected evaluations:
        //
        // \begin{equation}
        // combined = weight_1 + challenge \cdot weight_2 + challenge^2 \cdot weight_3
        // \end{equation}
        let expected_combined_evals = vec![
            w0 + challenge * w4 + challenge * challenge * Field64::ONE, // 1 + 2 * 5 + 2^2 * 1 = 15
            w1 + challenge * w5 + challenge * challenge * a,            // 2 + 2 * 6 + 2^2 * 2 = 22
            w2 + challenge * w6 + challenge * challenge * a * a, // 3 + 2 * 7 + 2^2 * 2 * 2 = 33
            w3 + challenge * w7 + challenge * challenge * Field64::ZERO, // 4 + 2 * 8 + 2^2 * 0 = 20
        ];
        // Expected sum:
        //
        // \begin{equation}
        // S_{combined} = S_1 + challenge \cdot S_2
        // \end{equation}
        let expected_combined_sum = sum1 + challenge * sum2 + challenge * challenge * sum3; // 5 + 2 * 7 + 2^2 * 9 = 19 + 36 = 55

        assert_eq!(combined_evals.evals(), &expected_combined_evals);
        assert_eq!(combined_sum, expected_combined_sum);
    }

    #[test]
    fn test_compute_evaluation_weight() {
        // Define an evaluation weight at a specific point
        let point = MultilinearPoint(vec![Field64::from(3)]);
        let weight = Weights::evaluation(point.clone());

        // Define a randomness point for folding
        let folding_randomness = MultilinearPoint(vec![Field64::from(2)]);

        // Expected result is the evaluation of eq_poly_outside at the given randomness
        let expected = point.eq_poly_outside(&folding_randomness);

        assert_eq!(weight.compute(&folding_randomness), expected);
    }

    #[test]
    #[allow(clippy::redundant_clone)]
    fn test_compute_evaluation_weight_identity() {
        // Define an evaluation weight at a specific point
        let point = MultilinearPoint(vec![Field64::ONE, Field64::ZERO]);

        // Folding randomness is the same as the point itself
        let folding_randomness = point.clone();
        let weight = Weights::evaluation(point.clone());

        // Expected result should be identity for equality polynomial
        let expected = point.eq_poly_outside(&folding_randomness);
        assert_eq!(weight.compute(&folding_randomness), expected);
    }

    #[test]
    fn test_compute_geometric_weight() {
        // Define a geometric progression
        let a = Field64::from(2);
        let n = 3;
        let weight_list = EvaluationsList::new(vec![Field64::ONE, a, a * a, Field64::ZERO]);
        let weight = Weights::geometric(a, n, weight_list.clone());

        // Define a randomness point for folding
        let folding_randomness = MultilinearPoint(vec![Field64::from(2), Field64::from(3)]);

        // Expected result is the evaluation of the geometric progression at the given randomness using geometric_till
        let expected = weight_list.eval_extension(&folding_randomness);
        assert_eq!(weight.compute(&folding_randomness), expected);
    }
}
