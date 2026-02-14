use std::{fmt::Debug, ops::Index};

use ark_ff::Field;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
#[cfg(feature = "tracing")]
use tracing::instrument;

use crate::algebra::{
    embedding::{Embedding, Identity},
    mixed_dot,
    polynomials::{geometric_till, CoefficientList, EvaluationsList, MultilinearPoint},
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
        weight: EvaluationsList<F>, // TODO: Why do we need this?
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
    pub const fn num_variables(&self) -> usize {
        match self {
            Self::Evaluation { point } => point.num_variables(),
            Self::Linear { weight } | Self::Geometric { weight, .. } => weight.num_variables(),
        }
    }

    pub const fn deferred(&self) -> bool {
        matches!(self, Self::Linear { .. })
    }

    pub fn mixed_evaluate<M>(&self, embedding: &M, poly: &CoefficientList<M::Source>) -> M::Target
    where
        M: Embedding<Target = F>,
    {
        assert_eq!(self.num_variables(), poly.num_variables());
        match self {
            Self::Evaluation { point } => poly.mixed_evaluate(embedding, point),
            Self::Linear { weight } | Self::Geometric { weight, .. } => {
                let poly: EvaluationsList<M::Source> = poly.clone().into();
                mixed_dot(embedding, weight.evals(), poly.evals())
            }
        }
    }

    /// Evaluate the weighted sum with a polynomial in coefficient form.
    pub fn evaluate(&self, poly: &CoefficientList<F>) -> F {
        self.mixed_evaluate(&Identity::new(), poly)
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
    #[cfg_attr(feature = "tracing", instrument(level = "debug", skip_all, fields(num_variables = self.num_variables())))]
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

    /// Computes the weight function evaluation under a given randomness.
    pub fn compute(&self, folding_randomness: &MultilinearPoint<F>) -> F {
        match self {
            Self::Evaluation { point } => point.eq_poly_outside(folding_randomness),
            Self::Linear { weight } => weight.eval_extension(folding_randomness),
            Self::Geometric { a, n, .. } => geometric_till(*a, *n, &folding_randomness.0),
        }
    }
}

#[cfg(test)]
mod tests {
    use ark_ff::AdditiveGroup;

    use super::*;
    use crate::{algebra::fields::Field64, utils::eval_eq};

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

        assert_eq!(weight.evaluate(&evals.to_coeffs()), expected);
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

        assert_eq!(weight.evaluate(&evals.to_coeffs()), expected);
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

    #[test]
    fn test_protocol() {
        // ── Step 1: Create a CoefficientList (4 variables, 16 coefficients) ──
        let coeffs = CoefficientList::new(vec![
            Field64::ONE,
            Field64::ONE,
            Field64::ONE,
            Field64::ONE,
            Field64::ONE,
            Field64::ONE,
            Field64::ONE,
            Field64::ONE,
            Field64::ONE,
            Field64::ONE,
            Field64::ONE,
            Field64::ONE,
            Field64::ONE,
            Field64::ONE,
            Field64::ONE,
            Field64::ONE,
        ]);
        println!("coeffs: {:?}\n", coeffs);

        // ── Step 2: Evaluate at several MultilinearPoints ──
        let evaluation_points = vec![
            MultilinearPoint(vec![
                Field64::ONE,
                Field64::ZERO,
                Field64::ZERO,
                Field64::ZERO,
            ]),
            MultilinearPoint(vec![
                Field64::ZERO,
                Field64::ONE,
                Field64::ZERO,
                Field64::ZERO,
            ]),
            MultilinearPoint(vec![
                Field64::ZERO,
                Field64::ZERO,
                Field64::ONE,
                Field64::ZERO,
            ]),
            MultilinearPoint(vec![
                Field64::ZERO,
                Field64::ZERO,
                Field64::ZERO,
                Field64::ONE,
            ]),
        ];
        println!("evaluation_points: {:?}\n", evaluation_points);
        let weights = evaluation_points
            .iter()
            .map(|point| Weights::evaluation(point.clone()))
            .collect::<Vec<_>>();
        println!("weights: {:?}\n", weights);
        let evaluations = evaluation_points
            .iter()
            .map(|point| coeffs.mixed_evaluate(&Identity::new(), point))
            .collect::<Vec<_>>();
        println!("evaluations: {:?}\n", evaluations);

        // ── Step 3: Convert CoefficientList → EvaluationsList → CoefficientList ──
        // CoefficientList → EvaluationsList (via wavelet transform)
        let evals = EvaluationsList::from(coeffs.clone());
        println!("evals (hypercube evaluations): {:?}\n", evals);

        // EvaluationsList → CoefficientList (via inverse wavelet transform)
        let coeffs_roundtrip = evals.to_coeffs();
        println!("coeffs_roundtrip: {:?}\n", coeffs_roundtrip);

        // Verify round-trip: coeffs → evals → coeffs gives back the same polynomial
        assert_eq!(
            coeffs.coeffs(),
            coeffs_roundtrip.coeffs(),
            "Round-trip CoefficientList → EvaluationsList → CoefficientList must be identity"
        );

        // Both representations should evaluate to the same values at any point
        for point in &evaluation_points {
            let from_coeffs = coeffs.evaluate(point);
            let from_evals = evals.evaluate(point);
            assert_eq!(
                from_coeffs, from_evals,
                "CoefficientList and EvaluationsList must agree at {:?}",
                point
            );
        }
        println!("✓ Round-trip and evaluation consistency verified\n");

        // ── Step 4: Verify fold_in_place matches fold ──
        // fold() creates a new polynomial; fold_in_place() mutates in-place.
        // After folding f(X₀, X₁, X₂, X₃) at (r₀, r₁), we get g(X₂, X₃) = f(X₂, X₃, r₀, r₁).
        let folding_randomness = MultilinearPoint(vec![Field64::from(3u64), Field64::from(7u64)]);

        // fold() — allocating version
        let folded = coeffs.fold(&folding_randomness);
        println!("folded (via fold):          {:?}", folded);

        // fold_in_place() — in-place version
        let mut coeffs_mut = coeffs.clone();
        coeffs_mut.fold_in_place(&folding_randomness);
        println!("folded (via fold_in_place): {:?}", coeffs_mut);

        // They must produce identical results
        assert_eq!(
            folded.coeffs(),
            coeffs_mut.coeffs(),
            "fold() and fold_in_place() must produce the same polynomial"
        );
        println!("✓ fold and fold_in_place match\n");

        // ── Step 5: Verify folded polynomial is consistent with full evaluation ──
        // g(a, b) should equal f(a, b, r₀, r₁) for any (a, b)
        let eval_point = MultilinearPoint(vec![Field64::from(5u64), Field64::from(11u64)]);
        println!("eval_point: {:?}\n", eval_point);
        let full_point = MultilinearPoint(vec![
            eval_point.0[0],
            eval_point.0[1],
            folding_randomness.0[0],
            folding_randomness.0[1],
        ]);
        println!("full_point: {:?}\n", full_point);
        let folded_eval = folded.evaluate(&eval_point);
        println!("folded poly: {:?}\n", folded);
        let full_eval = coeffs.evaluate(&full_point);
        println!("full poly: {:?}\n", coeffs);

        println!("folded_eval: {:?}\n", folded_eval);
        println!("full_eval: {:?}\n", full_eval);
        assert_eq!(
            folded_eval, full_eval,
            "f.fold(r).evaluate(a) must equal f.evaluate(a || r)"
        );
        println!(
            "✓ folded.evaluate({:?}) == coeffs.evaluate({:?}) == {:?}\n",
            eval_point.0, full_point.0, folded_eval
        );
    }
}
