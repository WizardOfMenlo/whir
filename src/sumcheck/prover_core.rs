use super::proof::SumcheckPolynomial;
use crate::{
    poly_utils::{
        coeffs::CoefficientList, evals::EvaluationsList, multilinear::MultilinearPoint,
        sequential_lag_poly::LagrangePolynomialIterator,
    },
    utils::base_decomposition,
};
use ark_ff::Field;

/// Implements the sumcheck protocol for verifying multivariate polynomial evaluations.
///
/// This struct allows us to:
/// - Convert a polynomial from coefficient representation to evaluation form.
/// - Apply equality constraints using lagrange interpolation.
/// - Compute the sumcheck polynomial efficiently.
///
/// Given a multilinear polynomial `p(X1, ..., Xn)`, we compute:
///
/// \begin{equation}
/// v(X_1, ..., X_n) = p(X_1, ..., X_n) \cdot \sum_{i} \epsilon_i \cdot eq_{z_i}(X)
/// \end{equation}
///
/// where:
/// - `eq_z(X)` is the equality polynomial evaluating whether `X = z`.
/// - `\epsilon_i` is a random scalar weighting each constraint.
#[derive(Debug)]
pub struct SumcheckCore<F> {
    /// Evaluations of the polynomial `p(X)`.
    evaluation_of_p: EvaluationsList<F>,
    /// Evaluations of equality constraint polynomial `\sum_{i} \epsilon_i eq_{z_i}(X)`.
    evaluation_of_equality: EvaluationsList<F>,
    /// Number of variables `n` in the multilinear polynomial.
    num_variables: usize,
}

impl<F> SumcheckCore<F>
where
    F: Field,
{
    /// Creates a new `SumcheckCore` instance.
    ///
    /// This function:
    /// 1. Converts `coeffs` from coefficient form into evaluation form.
    /// 2. Initializes an empty equality evaluation table.
    /// 3. Applies equality constraints if provided.
    pub fn new(
        coeffs: CoefficientList<F>,
        points: &[MultilinearPoint<F>],
        combination_randomness: &[F],
    ) -> Self {
        assert_eq!(points.len(), combination_randomness.len());
        let num_variables = coeffs.num_variables();

        let mut prover = Self {
            evaluation_of_p: coeffs.into(),
            evaluation_of_equality: EvaluationsList::new(vec![F::ZERO; 1 << num_variables]),
            num_variables,
        };

        // Add equality constraints if provided
        if !points.is_empty() {
            prover.add_new_equality(points, combination_randomness);
        }

        prover
    }

    /// Computes the sumcheck polynomial for a given folding factor.
    ///
    /// The sumcheck polynomial is computed as:
    ///
    /// \begin{equation}
    /// S(X) = \sum_{\beta} p(\beta) \cdot eq(\beta)
    /// \end{equation}
    ///
    /// where `\beta` are evaluation points in `{0,1,2}^k` (k = folding factor).
    pub fn compute_sumcheck_polynomial(&self, folding_factor: usize) -> SumcheckPolynomial<F> {
        assert!(self.num_variables >= folding_factor);

        let two = F::ONE + F::ONE;
        let num_evaluation_points = 3_usize.pow(folding_factor as u32);
        let suffix_len = 1 << folding_factor;
        let prefix_len = (1 << self.num_variables) / suffix_len;

        // Precompute evaluation points in `{0,1,2}^folding_factor`.
        let evaluation_points: Vec<_> = (0..num_evaluation_points)
            .map(|point| {
                MultilinearPoint(
                    base_decomposition(point, 3, folding_factor)
                        .into_iter()
                        .map(|v| [F::ZERO, F::ONE, two][v as usize])
                        .collect(),
                )
            })
            .collect();

        let mut evaluations = vec![F::ZERO; num_evaluation_points];

        // Compute evaluations efficiently
        for beta_prefix in 0..prefix_len {
            let start_idx = beta_prefix * suffix_len;
            let end_idx = start_idx + suffix_len;

            let left_poly =
                EvaluationsList::new(self.evaluation_of_p.evals()[start_idx..end_idx].to_vec());
            let right_poly = EvaluationsList::new(
                self.evaluation_of_equality.evals()[start_idx..end_idx].to_vec(),
            );

            // Accumulate evaluations over `{0,1,2}^folding_factor`.
            evaluation_points
                .iter()
                .enumerate()
                .for_each(|(point, eval_point)| unsafe {
                    *evaluations.get_unchecked_mut(point) +=
                        left_poly.evaluate(eval_point) * right_poly.evaluate(eval_point);
                });
        }

        SumcheckPolynomial::new(evaluations, folding_factor)
    }

    /// Adds new equality constraints.
    ///
    /// Given a set of points `z_i` and corresponding randomness `\epsilon_i`, computes:
    ///
    /// \begin{equation}
    /// v(X) = \sum_{i} \epsilon_i eq_{z_i}(X)
    /// \end{equation}
    ///
    /// where `eq_{z_i}(X)` is the Lagrange equality polynomial.
    pub fn add_new_equality(
        &mut self,
        points: &[MultilinearPoint<F>],
        combination_randomness: &[F],
    ) {
        assert_eq!(combination_randomness.len(), points.len());
        points
            .iter()
            .zip(combination_randomness)
            .for_each(|(point, &rand)| {
                LagrangePolynomialIterator::from(point).for_each(|(prefix, lag)| unsafe {
                    *self
                        .evaluation_of_equality
                        .evals_mut()
                        .get_unchecked_mut(prefix.0) += rand * lag;
                });
            });
    }

    // When the folding randomness arrives, compress the table accordingly (adding the new points)
    pub fn compress(
        &mut self,
        folding_factor: usize,
        combination_randomness: F, // Scale the initial point
        folding_randomness: &MultilinearPoint<F>,
    ) {
        assert_eq!(folding_randomness.n_variables(), folding_factor);
        assert!(self.num_variables >= folding_factor);

        let suffix_len = 1 << folding_factor;
        let prefix_len = (1 << self.num_variables) / suffix_len;
        let mut evaluations_of_p = Vec::with_capacity(prefix_len);
        let mut evaluations_of_eq = Vec::with_capacity(prefix_len);

        // Compress the table
        for beta_prefix in 0..prefix_len {
            let indexes: Vec<_> = (0..suffix_len)
                .map(|beta_suffix| suffix_len * beta_prefix + beta_suffix)
                .collect();

            let left_poly =
                EvaluationsList::new(indexes.iter().map(|&i| self.evaluation_of_p[i]).collect());
            let right_poly = EvaluationsList::new(
                indexes
                    .iter()
                    .map(|&i| self.evaluation_of_equality[i])
                    .collect(),
            );

            evaluations_of_p.push(left_poly.evaluate(folding_randomness));
            evaluations_of_eq
                .push(combination_randomness * right_poly.evaluate(folding_randomness));
        }

        // Update
        self.num_variables -= folding_factor;
        self.evaluation_of_p = EvaluationsList::new(evaluations_of_p);
        self.evaluation_of_equality = EvaluationsList::new(evaluations_of_eq);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto::fields::Field64;
    use ark_ff::AdditiveGroup;

    #[test]
    fn test_sumcheck_core_initialization() {
        // Polynomial with 2 variables: f(X1, X2) = 1 + 2*X1 + 3*X2 + 4*X1*X2
        //
        // Evaluations are lexicographically ordered over {0,1}^2:
        // f(0,0) = 1, f(0,1) = 3, f(1,0) = 2, f(1,1) = 4
        let c1 = Field64::from(1);
        let c2 = Field64::from(3);
        let c3 = Field64::from(2);
        let c4 = Field64::from(4);

        let coeffs = CoefficientList::new(vec![
            c1, // f(0,0)
            c2, // f(0,1)
            c3, // f(1,0)
            c4, // f(1,1)
        ]);

        let points = vec![];
        let combination_randomness = vec![];

        let prover = SumcheckCore::new(coeffs, &points, &combination_randomness);

        // Expected evaluation table after wavelet transform:
        //
        // Original coefficients:   [1, 3, 2, 4]
        //
        // Applying wavelet transform:
        // [1, 1 + 3, 2 + 1, 1 + 3 + 2 + 4]
        let expected_evaluation_of_p = vec![c1, c1 + c2, c1 + c3, c1 + c3 + c2 + c4];

        assert_eq!(prover.evaluation_of_p.evals(), &expected_evaluation_of_p);

        // Since we initialized with no equality constraints:
        // `evaluation_of_equality` should remain filled with zeros.
        assert_eq!(
            prover.evaluation_of_equality.evals(),
            &vec![Field64::ZERO; 4]
        );
    }

    #[test]
    fn test_sumcheck_core_initialization_single_variable() {
        // Polynomial with 1 variable: f(X1) = 1 + 2*X1
        //
        // Evaluations over {0,1}:
        // f(0) = 1, f(1) = 3
        let c1 = Field64::from(1);
        let c2 = Field64::from(3);

        let coeffs = CoefficientList::new(vec![c1, c2]);

        let points = vec![];
        let combination_randomness = vec![];

        let prover = SumcheckCore::new(coeffs, &points, &combination_randomness);

        // Expected evaluation table after wavelet transform:
        //
        // Original coefficients:   [1, 3]
        //
        // Applying wavelet transform:
        // [1, 1 + 3] = [1, 4]
        let expected_evaluation_of_p = vec![c1, c1 + c2];

        assert_eq!(prover.evaluation_of_p.evals(), &expected_evaluation_of_p);
        assert_eq!(
            prover.evaluation_of_equality.evals(),
            &vec![Field64::ZERO; 2]
        );
    }

    #[test]
    fn test_sumcheck_core_initialization_three_variables() {
        // Polynomial with 3 variables: f(X1, X2, X3) = 1 + 2*X1 + 3*X2 + 4*X3 + 5*X1*X2 + 6*X1*X3 +
        // 7*X2*X3 + 8*X1*X2*X3
        //
        // Evaluations over {0,1}^3:
        // f(0,0,0) = 1
        // f(0,0,1) = 4
        // f(0,1,0) = 3
        // f(0,1,1) = 7
        // f(1,0,0) = 2
        // f(1,0,1) = 6
        // f(1,1,0) = 5
        // f(1,1,1) = 8
        let c1 = Field64::from(1);
        let c2 = Field64::from(4);
        let c3 = Field64::from(3);
        let c4 = Field64::from(7);
        let c5 = Field64::from(2);
        let c6 = Field64::from(6);
        let c7 = Field64::from(5);
        let c8 = Field64::from(8);

        let coeffs = CoefficientList::new(vec![c1, c2, c3, c4, c5, c6, c7, c8]);

        let points = vec![];
        let combination_randomness = vec![];

        let prover = SumcheckCore::new(coeffs, &points, &combination_randomness);

        // Expected evaluation table after wavelet transform:
        //
        // Original coefficients:   [1, 4, 3, 7, 2, 6, 5, 8]
        //
        // Applying wavelet transform:
        // Step 1:
        // [1, 1+4, 3+1, 1+4+3+7, 2+1, 1+4+2+6, 3+1+2+5, 1+4+3+7+2+6+5+8]
        let expected_evaluation_of_p = vec![
            c1,
            c1 + c2,
            c1 + c3,
            c1 + c2 + c3 + c4,
            c1 + c5,
            c1 + c2 + c5 + c6,
            c1 + c3 + c5 + c7,
            c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8,
        ];

        assert_eq!(prover.evaluation_of_p.evals(), &expected_evaluation_of_p);
        assert_eq!(
            prover.evaluation_of_equality.evals(),
            &vec![Field64::ZERO; 8]
        );
    }

    #[test]
    fn test_sumcheck_core_with_equality_constraints() {
        // Polynomial with 2 variables: f(X1, X2) = 1 + 2*X1 + 3*X2 + 4*X1*X2
        //
        // Evaluations over {0,1}^2 in lexicographic order:
        // f(0,0) = 1, f(0,1) = 3, f(1,0) = 2, f(1,1) = 4
        let c1 = Field64::from(1);
        let c2 = Field64::from(3);
        let c3 = Field64::from(2);
        let c4 = Field64::from(4);

        let coeffs = CoefficientList::new(vec![c1, c2, c3, c4]);

        // Introduce an equality constraint at point (X1, X2) = (1,0) with randomness 2
        let point = MultilinearPoint(vec![Field64::ONE, Field64::ZERO]);
        let combination_randomness = vec![Field64::from(2)];

        let prover = SumcheckCore::new(coeffs, &[point], &combination_randomness);

        // Expected evaluation_of_p after wavelet transform:
        // Wavelet transform applied to coefficients:
        //
        // Original coefficients:   [1, 3, 2, 4]
        //
        // Applying wavelet transform:
        // [1, 1 + 3, 1 + 2, 1 + 2 + 3 + 4]
        let expected_evaluation_of_p = vec![c1, c1 + c2, c1 + c3, c1 + c3 + c2 + c4];

        assert_eq!(prover.evaluation_of_p.evals(), &expected_evaluation_of_p);

        // Compute expected equality polynomial evaluations manually:
        let eq_weight = Field64::from(2);

        // Given constraint point (1,0), compute eq_poly(X1, X2) at each binary point:
        let f_00 = eq_weight * Field64::from(0); // f(0,0) += 0
        let f_01 = eq_weight * Field64::from(0); // f(0,1) += 0
        let f_10 = eq_weight * Field64::from(1); // f(1,0) += 2
        let f_11 = eq_weight * Field64::from(0); // f(1,1) += 0

        let expected_evaluation_of_equality = vec![f_00, f_01, f_10, f_11];

        assert_eq!(
            prover.evaluation_of_equality.evals(),
            &expected_evaluation_of_equality
        );
    }

    #[test]
    fn test_compute_sumcheck_polynomial_basic() {
        // Polynomial with 2 variables: f(X1, X2) = c1 + c2*X1 + c3*X2 + c4*X1*X2
        let c1 = Field64::from(1);
        let c2 = Field64::from(2);
        let c3 = Field64::from(3);
        let c4 = Field64::from(4);

        let coeffs = CoefficientList::new(vec![c1, c3, c2, c4]);

        let prover = SumcheckCore::new(coeffs, &[], &[]);
        let sumcheck_poly = prover.compute_sumcheck_polynomial(1);

        // Since no equality constraints, sumcheck_poly should be **zero**
        let expected_evaluations = vec![Field64::ZERO; 3];

        assert_eq!(sumcheck_poly.evaluations(), &expected_evaluations);
    }

    #[test]
    fn test_compute_sumcheck_polynomial_with_equality_constraints() {
        let two = Field64::ONE + Field64::ONE;

        // Define a polynomial: f(X1, X2) = c1 + c2*X1 + c3*X2 + c4*X1*X2
        let c1 = Field64::from(1);
        let c2 = Field64::from(2);
        let c3 = Field64::from(3);
        let c4 = Field64::from(4);

        let coeffs = CoefficientList::new(vec![c1, c2, c3, c4]);

        // Introduce an equality constraint at point (X1, X2) = (1,0) with randomness 2
        let point = MultilinearPoint(vec![Field64::ONE, Field64::ZERO]);
        let combination_randomness = vec![Field64::from(2)];

        let prover = SumcheckCore::new(coeffs, &[point], &combination_randomness);
        let sumcheck_poly = prover.compute_sumcheck_polynomial(1);

        // Expected evaluation table after wavelet transform:
        let ep_00 = c1; // f(0,0)
        let ep_01 = c1 + c2; // f(0,1)
        let ep_10 = c1 + c3; // f(1,0)
        let ep_11 = c1 + c3 + c2 + c4; // f(1,1)

        // Compute equality polynomial evaluations for each binary point
        // Given constraint point (X1, X2) = (1,0) with randomness 2:
        // eq_poly(X1, X2) = 2 * (X1 - 1) * (X2 - 0)
        let f_00 = Field64::ZERO; // f(0,0) unaffected
        let f_01 = Field64::ZERO; // f(0,1) unaffected
        let f_10 = two; // f(1,0) modified by the constraint
        let f_11 = Field64::ZERO; // f(1,1) unaffected

        // Compute expected sumcheck evaluations for {0,1,2}
        // Evaluating at X1 ∈ {0,1,2}
        let e0 = EvaluationsList::new(vec![ep_00, ep_01])
            .evaluate(&MultilinearPoint(vec![Field64::ZERO]))
            * EvaluationsList::new(vec![f_00, f_01])
                .evaluate(&MultilinearPoint(vec![Field64::ZERO]))
            + EvaluationsList::new(vec![ep_10, ep_11])
                .evaluate(&MultilinearPoint(vec![Field64::ZERO]))
                * EvaluationsList::new(vec![f_10, f_11])
                    .evaluate(&MultilinearPoint(vec![Field64::ZERO]));

        let e1 = EvaluationsList::new(vec![ep_00, ep_01])
            .evaluate(&MultilinearPoint(vec![Field64::ONE]))
            * EvaluationsList::new(vec![f_00, f_01])
                .evaluate(&MultilinearPoint(vec![Field64::ONE]))
            + EvaluationsList::new(vec![ep_10, ep_11])
                .evaluate(&MultilinearPoint(vec![Field64::ONE]))
                * EvaluationsList::new(vec![f_10, f_11])
                    .evaluate(&MultilinearPoint(vec![Field64::ONE]));

        let e2 = EvaluationsList::new(vec![ep_00, ep_01]).evaluate(&MultilinearPoint(vec![two]))
            * EvaluationsList::new(vec![f_00, f_01]).evaluate(&MultilinearPoint(vec![two]))
            + EvaluationsList::new(vec![ep_10, ep_11]).evaluate(&MultilinearPoint(vec![two]))
                * EvaluationsList::new(vec![f_10, f_11]).evaluate(&MultilinearPoint(vec![two]));

        let expected_evaluations = vec![e0, e1, e2];

        assert_eq!(sumcheck_poly.evaluations(), &expected_evaluations);
    }

    #[test]
    fn test_compute_sumcheck_polynomial_with_multiple_equality_constraints() {
        let two = Field64::ONE + Field64::ONE;

        // Define a polynomial with three variables:
        // f(X1, X2, X3) = c1 + c2*X1 + c3*X2 + c4*X3 + c5*X1*X2 + c6*X1*X3 + c7*X2*X3 + c8*X1*X2*X3
        let c1 = Field64::from(1);
        let c2 = Field64::from(2);
        let c3 = Field64::from(3);
        let c4 = Field64::from(4);
        let c5 = Field64::from(5);
        let c6 = Field64::from(6);
        let c7 = Field64::from(7);
        let c8 = Field64::from(8);

        let coeffs = CoefficientList::new(vec![c1, c2, c3, c4, c5, c6, c7, c8]);

        // Introduce multiple equality constraints:
        // Constraint 1: (X1, X2, X3) = (1,0,1) with randomness 2
        // Constraint 2: (X1, X2, X3) = (0,1,0) with randomness 3
        let point1 = MultilinearPoint(vec![Field64::ONE, Field64::ZERO, Field64::ONE]);
        let point2 = MultilinearPoint(vec![Field64::ZERO, Field64::ONE, Field64::ZERO]);
        let combination_randomness = vec![Field64::from(2), Field64::from(3)];

        let prover = SumcheckCore::new(coeffs, &[point1, point2], &combination_randomness);
        let sumcheck_poly = prover.compute_sumcheck_polynomial(1);

        // Expected evaluation table after wavelet transform:
        let ep_000 = c1;
        let ep_001 = c1 + c2;
        let ep_010 = c1 + c3;
        let ep_011 = c1 + c2 + c3 + c4;
        let ep_100 = c1 + c5;
        let ep_101 = c1 + c2 + c5 + c6;
        let ep_110 = c1 + c3 + c5 + c7;
        let ep_111 = c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8;

        // Compute equality polynomial evaluations manually
        // Constraint 1 (1,0,1) → modifies f(1,0,1)
        // Constraint 2 (0,1,0) → modifies f(0,1,0)
        let f_000 = Field64::ZERO;
        let f_001 = Field64::ZERO;
        let f_010 = Field64::from(3); // Modified by constraint 2
        let f_011 = Field64::ZERO;
        let f_100 = Field64::ZERO;
        let f_101 = Field64::from(2); // Modified by constraint 1
        let f_110 = Field64::ZERO;
        let f_111 = Field64::ZERO;

        // Compute expected sumcheck evaluations for {0,1,2}
        let e0 = EvaluationsList::new(vec![ep_000, ep_001])
            .evaluate(&MultilinearPoint(vec![Field64::ZERO]))
            * EvaluationsList::new(vec![f_000, f_001])
                .evaluate(&MultilinearPoint(vec![Field64::ZERO]))
            + EvaluationsList::new(vec![ep_010, ep_011])
                .evaluate(&MultilinearPoint(vec![Field64::ZERO]))
                * EvaluationsList::new(vec![f_010, f_011])
                    .evaluate(&MultilinearPoint(vec![Field64::ZERO]))
            + EvaluationsList::new(vec![ep_100, ep_101])
                .evaluate(&MultilinearPoint(vec![Field64::ZERO]))
                * EvaluationsList::new(vec![f_100, f_101])
                    .evaluate(&MultilinearPoint(vec![Field64::ZERO]))
            + EvaluationsList::new(vec![ep_110, ep_111])
                .evaluate(&MultilinearPoint(vec![Field64::ZERO]))
                * EvaluationsList::new(vec![f_110, f_111])
                    .evaluate(&MultilinearPoint(vec![Field64::ZERO]));

        let e1 = EvaluationsList::new(vec![ep_000, ep_001])
            .evaluate(&MultilinearPoint(vec![Field64::ONE]))
            * EvaluationsList::new(vec![f_000, f_001])
                .evaluate(&MultilinearPoint(vec![Field64::ONE]))
            + EvaluationsList::new(vec![ep_010, ep_011])
                .evaluate(&MultilinearPoint(vec![Field64::ONE]))
                * EvaluationsList::new(vec![f_010, f_011])
                    .evaluate(&MultilinearPoint(vec![Field64::ONE]))
            + EvaluationsList::new(vec![ep_100, ep_101])
                .evaluate(&MultilinearPoint(vec![Field64::ONE]))
                * EvaluationsList::new(vec![f_100, f_101])
                    .evaluate(&MultilinearPoint(vec![Field64::ONE]))
            + EvaluationsList::new(vec![ep_110, ep_111])
                .evaluate(&MultilinearPoint(vec![Field64::ONE]))
                * EvaluationsList::new(vec![f_110, f_111])
                    .evaluate(&MultilinearPoint(vec![Field64::ONE]));

        let e2 = EvaluationsList::new(vec![ep_000, ep_001]).evaluate(&MultilinearPoint(vec![two]))
            * EvaluationsList::new(vec![f_000, f_001]).evaluate(&MultilinearPoint(vec![two]))
            + EvaluationsList::new(vec![ep_010, ep_011]).evaluate(&MultilinearPoint(vec![two]))
                * EvaluationsList::new(vec![f_010, f_011]).evaluate(&MultilinearPoint(vec![two]))
            + EvaluationsList::new(vec![ep_100, ep_101]).evaluate(&MultilinearPoint(vec![two]))
                * EvaluationsList::new(vec![f_100, f_101]).evaluate(&MultilinearPoint(vec![two]))
            + EvaluationsList::new(vec![ep_110, ep_111]).evaluate(&MultilinearPoint(vec![two]))
                * EvaluationsList::new(vec![f_110, f_111]).evaluate(&MultilinearPoint(vec![two]));

        let expected_evaluations = vec![e0, e1, e2];

        assert_eq!(sumcheck_poly.evaluations(), &expected_evaluations);
    }
}
