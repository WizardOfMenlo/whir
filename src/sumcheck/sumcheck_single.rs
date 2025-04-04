use ark_ff::Field;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use spongefish::{
    codecs::arkworks_algebra::{FieldToUnitSerialize, UnitToField},
    ProofResult,
};
use spongefish_pow::{PoWChallenge, PowStrategy};
#[cfg(feature = "tracing")]
use tracing::{instrument, span, Level};

use super::SumcheckPolynomial;
use crate::{
    poly_utils::{coeffs::CoefficientList, evals::EvaluationsList, multilinear::MultilinearPoint},
    utils::eval_eq,
    whir::statement::Statement,
};

/// Implements the single-round sumcheck protocol for verifying a multilinear polynomial evaluation.
///
/// This struct is responsible for:
/// - Transforming a polynomial from coefficient representation into evaluation form.
/// - Constructing and evaluating weighted constraints.
/// - Computing the sumcheck polynomial, which is a quadratic polynomial in a single variable.
///
/// Given a multilinear polynomial `p(X1, ..., Xn)`, the sumcheck polynomial is computed as:
///
/// \begin{equation}
/// h(X) = \sum_b p(b, X) \cdot w(b, X)
/// \end{equation}
///
/// where:
/// - `b` ranges over evaluation points in `{0,1,2}^k` (with `k=1` in this implementation).
/// - `w(b, X)` represents generic weights applied to `p(b, X)`.
/// - The result `h(X)` is a quadratic polynomial in `X`.
///
/// The sumcheck protocol ensures that the claimed sum is correct.
#[derive(Clone, Debug)]
pub struct SumcheckSingle<F> {
    /// Evaluations of the polynomial `p(X)`.
    evaluation_of_p: EvaluationsList<F>,
    /// Evaluations of the weight polynomial used for enforcing constraints.
    weights: EvaluationsList<F>,
    /// Accumulated sum incorporating weighted constraints.
    sum: F,
}

impl<F> SumcheckSingle<F>
where
    F: Field,
{
    /// Constructs a new `SumcheckSingle` instance from polynomial coefficients.
    ///
    /// This function:
    /// - Converts `coeffs` into evaluation form.
    /// - Initializes an empty constraint table.
    /// - Applies weighted constraints if provided.
    ///
    /// The provided `Statement` encodes constraints that contribute to the final sumcheck equation.
    pub fn new(
        coeffs: CoefficientList<F>,
        statement: &Statement<F>,
        combination_randomness: F,
    ) -> Self {
        let (weights, sum) = statement.combine(combination_randomness);
        Self {
            evaluation_of_p: coeffs.into(),
            weights,
            sum,
        }
    }

    /// Returns the number of variables in the polynomial.
    pub const fn num_variables(&self) -> usize {
        self.evaluation_of_p.num_variables()
    }

    /// Computes the sumcheck polynomial `h(X)`, which is quadratic.
    ///
    /// The sumcheck polynomial is given by:
    ///
    /// \begin{equation}
    /// h(X) = \sum_b p(b, X) \cdot w(b, X)
    /// \end{equation}
    ///
    /// where:
    /// - `b` represents points in `{0,1,2}^1`.
    /// - `w(b, X)` are the generic weights applied to `p(b, X)`.
    /// - `h(X)` is a quadratic polynomial.
    #[cfg_attr(feature = "tracing", instrument(skip_all, fields(size = self.evaluation_of_p.num_evals())))]
    pub fn compute_sumcheck_polynomial(&self) -> SumcheckPolynomial<F> {
        assert!(self.num_variables() >= 1);

        #[cfg(feature = "parallel")]
        let (c0, c2) = self
            .evaluation_of_p
            .evals()
            .par_chunks_exact(2)
            .zip(self.weights.evals().par_chunks_exact(2))
            .map(|(p_at, eq_at)| {
                // Convert evaluations to coefficients for the linear fns p and eq.
                let (p_0, p_1) = (p_at[0], p_at[1] - p_at[0]);
                let (eq_0, eq_1) = (eq_at[0], eq_at[1] - eq_at[0]);

                // Now we need to add the contribution of p(x) * eq(x)
                (p_0 * eq_0, p_1 * eq_1)
            })
            .reduce(
                || (F::ZERO, F::ZERO),
                |(a0, a2), (b0, b2)| (a0 + b0, a2 + b2),
            );

        #[cfg(not(feature = "parallel"))]
        let (c0, c2) = self
            .evaluation_of_p
            .evals()
            .chunks_exact(2)
            .zip(self.weights.evals().chunks_exact(2))
            .map(|(p_at, eq_at)| {
                // Convert evaluations to coefficients for the linear fns p and eq.
                let (p_0, p_1) = (p_at[0], p_at[1] - p_at[0]);
                let (eq_0, eq_1) = (eq_at[0], eq_at[1] - eq_at[0]);

                // Now we need to add the contribution of p(x) * eq(x)
                (p_0 * eq_0, p_1 * eq_1)
            })
            .fold((F::ZERO, F::ZERO), |(a0, a2), (b0, b2)| (a0 + b0, a2 + b2));

        // Use the fact that self.sum = p(0) + p(1) = 2 * coeff_0 + coeff_1 + coeff_2
        let c1 = self.sum - c0.double() - c2;

        // Evaluate the quadratic polynomial at 0, 1, 2
        let eval_0 = c0;
        let eval_1 = c0 + c1 + c2;
        let eval_2 = eval_1 + c1 + c2 + c2.double();

        SumcheckPolynomial::new(vec![eval_0, eval_1, eval_2], 1)
    }

    /// Implements `folding_factor` rounds of the sumcheck protocol.
    ///
    /// The sumcheck protocol progressively reduces the number of variables in a multilinear
    /// polynomial. At each step, a quadratic polynomial is derived and verified.
    ///
    /// Given a polynomial \( p(X_1, \dots, X_n) \), this function iteratively applies the
    /// transformation:
    ///
    /// \begin{equation}
    /// h(X) = \sum_b p(b, X) \cdot w(b, X)
    /// \end{equation}
    ///
    /// where:
    /// - \( b \) are points in \{0,1,2\}.
    /// - \( w(b, X) \) represents generic weights applied to \( p(b, X) \).
    /// - \( h(X) \) is a quadratic polynomial in \( X \).
    ///
    /// This function:
    /// - Samples random values to progressively reduce the polynomial.
    /// - Applies proof-of-work grinding if required.
    /// - Returns the sampled folding randomness values used in each reduction step.
    #[cfg_attr(feature = "tracing", instrument(skip(self, prover_state)))]
    pub fn compute_sumcheck_polynomials<S, ProverState>(
        &mut self,
        prover_state: &mut ProverState,
        folding_factor: usize,
        pow_bits: f64,
    ) -> ProofResult<MultilinearPoint<F>>
    where
        ProverState: FieldToUnitSerialize<F> + UnitToField<F> + PoWChallenge,
        S: PowStrategy,
    {
        let mut res = Vec::with_capacity(folding_factor);

        for _ in 0..folding_factor {
            let sumcheck_poly = self.compute_sumcheck_polynomial();
            prover_state.add_scalars(sumcheck_poly.evaluations())?;
            let [folding_randomness] = prover_state.challenge_scalars()?;
            res.push(folding_randomness);

            // Do PoW if needed
            if pow_bits > 0. {
                #[cfg(feature = "tracing")]
                let _span = span!(Level::INFO, "challenge_pow", pow_bits).entered();
                prover_state.challenge_pow::<S>(pow_bits)?;
            }

            self.compress(F::ONE, &folding_randomness.into(), &sumcheck_poly);
        }

        res.reverse();
        Ok(MultilinearPoint(res))
    }

    /// Adds new weighted constraints to the polynomial.
    ///
    /// This function updates the weight evaluations and sum by incorporating new constraints.
    ///
    /// Given points `z_i`, weights `ε_i`, and evaluation values `f(z_i)`, it updates:
    ///
    /// \begin{equation}
    /// w(X) = w(X) + \sum \epsilon_i \cdot w_{z_i}(X)
    /// \end{equation}
    ///
    /// and updates the sum as:
    ///
    /// \begin{equation}
    /// S = S + \sum \epsilon_i \cdot f(z_i)
    /// \end{equation}
    ///
    /// where `w_{z_i}(X)` represents the constraint encoding at point `z_i`.
    #[cfg_attr(feature = "tracing", instrument(skip_all))]
    pub fn add_new_equality(
        &mut self,
        points: &[MultilinearPoint<F>],
        evaluations: &[F],
        combination_randomness: &[F],
    ) {
        assert_eq!(combination_randomness.len(), points.len());
        assert_eq!(combination_randomness.len(), evaluations.len());

        // Accumulate the sum while applying all constraints simultaneously
        points
            .iter()
            .zip(combination_randomness.iter().zip(evaluations.iter()))
            .for_each(|(point, (&rand, &eval))| {
                eval_eq(&point.0, self.weights.evals_mut(), rand);
                self.sum += rand * eval;
            });
    }

    /// Compresses the polynomial and weight evaluations by reducing the number of variables.
    ///
    /// Given a multilinear polynomial `p(X1, ..., Xn)`, this function eliminates `X1` using the
    /// folding randomness `r`:
    /// \begin{equation}
    ///     p'(X_2, ..., X_n) = (1 - r) \cdot p(0, X_2, ..., X_n) + r \cdot p(1, X_2, ..., X_n)
    /// \end{equation}
    ///
    /// The same transformation applies to the weights `w(X)`, and the sum is updated as:
    ///
    /// \begin{equation}
    ///     S' = \rho \cdot h(r)
    /// \end{equation}
    ///
    /// where `h(r)` is the sumcheck polynomial evaluated at `r`, and `\rho` is
    /// `combination_randomness`.
    ///
    /// # Effects
    /// - Shrinks `p(X)` and `w(X)` by half.
    /// - Fixes `X_1 = r`, reducing the dimensionality.
    /// - Updates `sum` using `sumcheck_poly`.
    #[cfg_attr(feature = "tracing", instrument(skip_all, fields(size = self.evaluation_of_p.num_evals())))]
    pub fn compress(
        &mut self,
        combination_randomness: F,
        folding_randomness: &MultilinearPoint<F>,
        sumcheck_poly: &SumcheckPolynomial<F>,
    ) {
        assert_eq!(folding_randomness.num_variables(), 1);
        assert!(self.num_variables() >= 1);

        let randomness = folding_randomness.0[0];

        let fold_chunk = |slice: &[F]| -> F { (slice[1] - slice[0]) * randomness + slice[0] };

        #[cfg(feature = "parallel")]
        let (evaluations_of_p, evaluations_of_eq) = {
            // Threshold below which sequential computation is faster
            //
            // This was chosen based on experiments with the `compress` function.
            // It is possible that the threshold can be tuned further.
            const PARALLEL_THRESHOLD: usize = 4096;

            if self.evaluation_of_p.evals().len() >= PARALLEL_THRESHOLD
                && self.weights.evals().len() >= PARALLEL_THRESHOLD
            {
                rayon::join(
                    || {
                        self.evaluation_of_p
                            .evals()
                            .par_chunks_exact(2)
                            .map(fold_chunk)
                            .collect()
                    },
                    || {
                        self.weights
                            .evals()
                            .par_chunks_exact(2)
                            .map(fold_chunk)
                            .collect()
                    },
                )
            } else {
                (
                    self.evaluation_of_p
                        .evals()
                        .chunks_exact(2)
                        .map(fold_chunk)
                        .collect(),
                    self.weights
                        .evals()
                        .chunks_exact(2)
                        .map(fold_chunk)
                        .collect(),
                )
            }
        };

        #[cfg(not(feature = "parallel"))]
        let (evaluations_of_p, evaluations_of_eq) = (
            self.evaluation_of_p
                .evals()
                .chunks_exact(2)
                .map(fold_chunk)
                .collect(),
            self.weights
                .evals()
                .chunks_exact(2)
                .map(fold_chunk)
                .collect(),
        );

        // Update internal state
        self.evaluation_of_p = EvaluationsList::new(evaluations_of_p);
        self.weights = EvaluationsList::new(evaluations_of_eq);
        self.sum = combination_randomness * sumcheck_poly.evaluate_at_point(folding_randomness);
    }
}

#[cfg(test)]
mod tests {
    use ark_ff::AdditiveGroup;
    use spongefish::{
        codecs::arkworks_algebra::FieldDomainSeparator, DefaultHash, DomainSeparator,
    };
    use spongefish_pow::{blake3::Blake3PoW, PoWDomainSeparator};

    use super::*;
    use crate::{
        crypto::fields::Field64 as F,
        poly_utils::{
            coeffs::CoefficientList, lagrange_iterator::LagrangePolynomialIterator,
            multilinear::MultilinearPoint,
        },
        whir::statement::Weights,
    };

    #[test]
    fn test_sumcheck_folding_factor_1() {
        let eval_point = MultilinearPoint(vec![F::from(10), F::from(11)]);
        let polynomial =
            CoefficientList::new(vec![F::from(1), F::from(5), F::from(10), F::from(14)]);

        let claimed_value = polynomial.evaluate(&eval_point);

        let eval = polynomial.evaluate(&eval_point);
        let mut statement = Statement::new(eval_point.num_variables());
        let weights = Weights::evaluation(eval_point);
        statement.add_constraint(weights, eval);

        let mut prover = SumcheckSingle::new(polynomial, &statement, F::from(1));

        let poly_1 = prover.compute_sumcheck_polynomial();

        // First, check that is sums to the right value over the hypercube
        assert_eq!(poly_1.sum_over_boolean_hypercube(), claimed_value);

        let combination_randomness = F::from(100_101);
        let folding_randomness = MultilinearPoint(vec![F::from(4999)]);

        prover.compress(combination_randomness, &folding_randomness, &poly_1);

        let poly_2 = prover.compute_sumcheck_polynomial();

        assert_eq!(
            poly_2.sum_over_boolean_hypercube(),
            combination_randomness * poly_1.evaluate_at_point(&folding_randomness)
        );
    }

    #[test]
    fn test_sumcheck_weighted_folding_factor_1() {
        let eval_point = MultilinearPoint(vec![F::from(10), F::from(11)]);
        let polynomial =
            CoefficientList::new(vec![F::from(1), F::from(5), F::from(10), F::from(14)]);

        let claimed_value = polynomial.evaluate(&eval_point);

        let eval = polynomial.evaluate(&eval_point);

        let mut statement = Statement::new(eval_point.num_variables());
        let weights = Weights::evaluation(eval_point);
        statement.add_constraint(weights, eval);

        let mut prover = SumcheckSingle::new(polynomial, &statement, F::from(1));

        let poly_1 = prover.compute_sumcheck_polynomial();
        // First, check that is sums to the right value over the hypercube
        assert_eq!(poly_1.sum_over_boolean_hypercube(), claimed_value);

        let combination_randomness = F::from(100_101);
        let folding_randomness = MultilinearPoint(vec![F::from(4999)]);

        prover.compress(combination_randomness, &folding_randomness, &poly_1);

        let poly_2 = prover.compute_sumcheck_polynomial();

        assert_eq!(
            poly_2.sum_over_boolean_hypercube(),
            combination_randomness * poly_1.evaluate_at_point(&folding_randomness)
        );
    }

    #[test]
    fn test_eval_eq() {
        let eval = vec![F::from(3), F::from(5)];
        let mut out = vec![F::ZERO; 4];
        eval_eq(&eval, &mut out, F::ONE);

        let point = MultilinearPoint(eval);
        let mut expected = vec![F::ZERO; 4];
        for (prefix, lag) in LagrangePolynomialIterator::from(&point) {
            expected[prefix.0] = lag;
        }

        assert_eq!(&out, &expected);
    }

    #[test]
    fn test_sumcheck_single_initialization() {
        // Polynomial with 2 variables: f(X1, X2) = 1 + 2*X1 + 3*X2 + 4*X1*X2
        let c1 = F::from(1);
        let c2 = F::from(2);
        let c3 = F::from(3);
        let c4 = F::from(4);

        let coeffs = CoefficientList::new(vec![c1, c2, c3, c4]);
        let statement = Statement::new(2);

        let prover = SumcheckSingle::new(coeffs, &statement, F::ONE);

        // Expected evaluation table after wavelet transform
        let expected_evaluation_of_p = vec![c1, c1 + c2, c1 + c3, c1 + c2 + c3 + c4];

        assert_eq!(prover.evaluation_of_p.evals(), &expected_evaluation_of_p);
        assert_eq!(prover.weights.evals(), &vec![F::ZERO; 4]);
        assert_eq!(prover.sum, F::ZERO);
        assert_eq!(prover.num_variables(), 2);
    }

    #[test]
    fn test_sumcheck_single_one_variable() {
        // Polynomial with 1 variable: f(X1) = 1 + 3*X1
        let c1 = F::from(1);
        let c2 = F::from(3);

        // Convert the polynomial into coefficient form
        let coeffs = CoefficientList::new(vec![c1, c2]);

        // Create an empty statement (no equality constraints)
        let statement = Statement::new(1);

        // Instantiate the Sumcheck prover
        let prover = SumcheckSingle::new(coeffs, &statement, F::ONE);

        // Expected evaluations of the polynomial in evaluation form
        let expected_evaluation_of_p = vec![c1, c1 + c2];

        assert_eq!(prover.evaluation_of_p.evals(), &expected_evaluation_of_p);
        assert_eq!(prover.weights.evals(), &vec![F::ZERO; 2]);
        assert_eq!(prover.sum, F::ZERO);
        assert_eq!(prover.num_variables(), 1);
    }

    #[test]
    fn test_sumcheck_single_three_variables() {
        // Polynomial with 3 variables:
        // f(X1, X2, X3) = 1 + 2*X1 + 3*X2 + 4*X1*X2 + 5*X3 + 6*X1*X3 + 7*X2*X3 + 8*X1*X2*X3
        let c1 = F::from(1);
        let c2 = F::from(2);
        let c3 = F::from(3);
        let c4 = F::from(4);
        let c5 = F::from(5);
        let c6 = F::from(6);
        let c7 = F::from(7);
        let c8 = F::from(8);

        // Convert the polynomial into coefficient form
        let coeffs = CoefficientList::new(vec![c1, c2, c3, c4, c5, c6, c7, c8]);

        // Create an empty statement (no equality constraints)
        let statement = Statement::new(3);

        // Instantiate the Sumcheck prover
        let prover = SumcheckSingle::new(coeffs, &statement, F::ONE);

        // Expected evaluations of the polynomial in evaluation form
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
        assert_eq!(prover.weights.evals(), &vec![F::ZERO; 8]);
        assert_eq!(prover.sum, F::ZERO);
        assert_eq!(prover.num_variables(), 3);
    }

    #[test]
    fn test_sumcheck_single_with_equality_constraints() {
        // Define a polynomial with 2 variables:
        // f(X1, X2) = 1 + 2*X1 + 3*X2 + 4*X1*X2
        let c1 = F::from(1);
        let c2 = F::from(2);
        let c3 = F::from(3);
        let c4 = F::from(4);

        // Convert the polynomial into coefficient form
        let coeffs = CoefficientList::new(vec![c1, c2, c3, c4]);

        // Create a statement and introduce an equality constraint at (X1, X2) = (1,0)
        let mut statement = Statement::new(2);
        let point = MultilinearPoint(vec![F::ONE, F::ZERO]);
        let weights = Weights::evaluation(point);
        let eval = F::from(5);
        statement.add_constraint(weights, eval);

        // Instantiate the Sumcheck prover
        let prover = SumcheckSingle::new(coeffs, &statement, F::ONE);

        // Expected sum update: sum = 5
        assert_eq!(prover.sum, eval);

        // Expected evaluation table after wavelet transform
        let expected_evaluation_of_p = vec![c1, c1 + c2, c1 + c3, c1 + c2 + c3 + c4];
        assert_eq!(prover.evaluation_of_p.evals(), &expected_evaluation_of_p);
        assert_eq!(prover.num_variables(), 2);
    }

    #[test]
    fn test_sumcheck_single_multiple_constraints() {
        // Define a polynomial with 3 variables:
        // f(X1, X2, X3) = c1 + c2*X1 + c3*X2 + c4*X3 + c5*X1*X2 + c6*X1*X3 + c7*X2*X3 + c8*X1*X2*X3
        let c1 = F::from(1);
        let c2 = F::from(2);
        let c3 = F::from(3);
        let c4 = F::from(4);
        let c5 = F::from(5);
        let c6 = F::from(6);
        let c7 = F::from(7);
        let c8 = F::from(8);

        // Convert the polynomial into coefficient form
        let coeffs = CoefficientList::new(vec![c1, c2, c3, c4, c5, c6, c7, c8]);

        // Create a statement and introduce multiple equality constraints
        let mut statement = Statement::new(3);

        // Constraints: (X1, X2, X3) = (1,0,1) with weight 2, (0,1,0) with weight 3
        let point1 = MultilinearPoint(vec![F::ONE, F::ZERO, F::ONE]);
        let point2 = MultilinearPoint(vec![F::ZERO, F::ONE, F::ZERO]);

        let weights1 = Weights::evaluation(point1);
        let weights2 = Weights::evaluation(point2);

        let eval1 = F::from(5);
        let eval2 = F::from(4);

        statement.add_constraint(weights1, eval1);
        statement.add_constraint(weights2, eval2);

        // Instantiate the Sumcheck prover
        let prover = SumcheckSingle::new(coeffs, &statement, F::ONE);

        // Expected sum update: sum = (5) + (4)
        let expected_sum = eval1 + eval2;
        assert_eq!(prover.sum, expected_sum);
    }

    #[test]
    fn test_compute_sumcheck_polynomial_basic() {
        // Polynomial with 2 variables: f(X1, X2) = c1 + c2*X1 + c3*X2 + c4*X1*X2
        let c1 = F::from(1);
        let c2 = F::from(2);
        let c3 = F::from(3);
        let c4 = F::from(4);

        // Convert the polynomial into coefficient form
        let coeffs = CoefficientList::new(vec![c1, c2, c3, c4]);

        // Create an empty statement (no constraints)
        let statement = Statement::new(2);

        // Instantiate the Sumcheck prover
        let prover = SumcheckSingle::new(coeffs, &statement, F::ONE);
        let sumcheck_poly = prover.compute_sumcheck_polynomial();

        // Since no equality constraints, sumcheck_poly should be **zero**
        let expected_evaluations = vec![F::ZERO; 3];
        assert_eq!(sumcheck_poly.evaluations(), &expected_evaluations);
    }

    #[test]
    fn test_compute_sumcheck_polynomial_with_equality_constraints() {
        // Define a multilinear polynomial with two variables:
        // f(X1, X2) = c1 + c2*X1 + c3*X2 + c4*X1*X2
        // This polynomial is represented in coefficient form.
        let c1 = F::from(1);
        let c2 = F::from(2);
        let c3 = F::from(3);
        let c4 = F::from(4);

        // Convert the polynomial into coefficient list representation
        let coeffs = CoefficientList::new(vec![c1, c2, c3, c4]);

        // Create a statement and introduce an equality constraint at (X1, X2) = (1,0)
        // The constraint enforces that f(1,0) must evaluate to 5 with weight 2.
        let mut statement = Statement::new(2);
        let point = MultilinearPoint(vec![F::ONE, F::ZERO]);
        let weights = Weights::evaluation(point);
        let eval = F::from(5);
        statement.add_constraint(weights, eval);

        // Instantiate the Sumcheck prover with the polynomial and equality constraints
        let prover = SumcheckSingle::new(coeffs, &statement, F::ONE);
        let sumcheck_poly = prover.compute_sumcheck_polynomial();

        // The constraint directly contributes to the sum, hence sum = 5
        assert_eq!(prover.sum, eval);

        // Compute the polynomial evaluations at the four possible binary inputs
        let ep_00 = c1; // f(0,0) = c1
        let ep_01 = c1 + c2; // f(0,1) = c1 + c2
        let ep_10 = c1 + c3; // f(1,0) = c1 + c3
        let ep_11 = c1 + c3 + c2 + c4; // f(1,1) = c1 + c3 + c2 + c4

        // Compute the evaluations of the equality constraint polynomial at each binary input
        // Given that the constraint is at (1,0) with weight 2, the equality function is:
        //
        // \begin{equation}
        // eq(X1, X2) = 2 * (X1 - 1) * (X2 - 0)
        // \end{equation}
        let f_00 = F::ZERO; // eq(0,0) = 0
        let f_01 = F::ZERO; // eq(0,1) = 0
        let f_10 = F::ONE; // eq(1,0) = 1
        let f_11 = F::ZERO; // eq(1,1) = 0

        // Compute the coefficients of the sumcheck polynomial S(X)
        let e0 = ep_00 * f_00 + ep_10 * f_10; // Constant term (X = 0)
        let e2 = (ep_01 - ep_00) * (f_01 - f_00) + (ep_11 - ep_10) * (f_11 - f_10); // Quadratic coefficient
        let e1 = prover.sum - e0.double() - e2; // Middle coefficient using sum rule

        // Compute the evaluations of the sumcheck polynomial at X ∈ {0,1,2}
        let eval_0 = e0;
        let eval_1 = e0 + e1 + e2;
        let eval_2 = eval_1 + e1 + e2 + e2.double();
        let expected_evaluations = vec![eval_0, eval_1, eval_2];

        // Ensure that the computed sumcheck polynomial matches expectations
        assert_eq!(sumcheck_poly.evaluations(), &expected_evaluations);
    }

    #[test]
    fn test_compute_sumcheck_polynomial_with_equality_constraints_3vars() {
        // Define a multilinear polynomial with three variables:
        // f(X1, X2, X3) = c1 + c2*X1 + c3*X2 + c4*X3 + c5*X1*X2 + c6*X1*X3 + c7*X2*X3 + c8*X1*X2*X3
        let c1 = F::from(1);
        let c2 = F::from(2);
        let c3 = F::from(3);
        let c4 = F::from(4);
        let c5 = F::from(5);
        let c6 = F::from(6);
        let c7 = F::from(7);
        let c8 = F::from(8);

        // Convert the polynomial into coefficient form
        let coeffs = CoefficientList::new(vec![c1, c2, c3, c4, c5, c6, c7, c8]);

        // Create a statement and introduce an equality constraint at (X1, X2, X3) = (1,0,1)
        let mut statement = Statement::new(3);
        let point = MultilinearPoint(vec![F::ONE, F::ZERO, F::ONE]);
        let weights = Weights::evaluation(point);
        let eval = F::from(5);
        statement.add_constraint(weights, eval);

        // Instantiate the Sumcheck prover with the polynomial and equality constraints
        let prover = SumcheckSingle::new(coeffs, &statement, F::ONE);
        let sumcheck_poly = prover.compute_sumcheck_polynomial();

        // Expected sum update: sum = 5
        assert_eq!(prover.sum, eval);

        // Compute polynomial evaluations at the eight possible binary inputs
        let ep_000 = c1; // f(0,0,0)
        let ep_001 = c1 + c2; // f(0,0,1)
        let ep_010 = c1 + c3; // f(0,1,0)
        let ep_011 = c1 + c2 + c3 + c4; // f(0,1,1)
        let ep_100 = c1 + c5; // f(1,0,0)
        let ep_101 = c1 + c2 + c5 + c6; // f(1,0,1)
        let ep_110 = c1 + c3 + c5 + c7; // f(1,1,0)
        let ep_111 = c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8; // f(1,1,1)

        // Compute the evaluations of the equality constraint polynomial at each binary input
        // Given that the constraint is at (1,0,1) with weight 2, the equality function is:
        //
        // \begin{equation}
        // eq(X1, X2, X3) = 2 * (X1 - 1) * (X2 - 0) * (X3 - 1)
        // \end{equation}
        let f_000 = F::ZERO; // eq(0,0,0) = 0
        let f_001 = F::ZERO; // eq(0,0,1) = 0
        let f_010 = F::ZERO; // eq(0,1,0) = 0
        let f_011 = F::ZERO; // eq(0,1,1) = 0
        let f_100 = F::ZERO; // eq(1,0,0) = 0
        let f_101 = F::ONE; // eq(1,0,1) = 1
        let f_110 = F::ZERO; // eq(1,1,0) = 0
        let f_111 = F::ZERO; // eq(1,1,1) = 0

        // Compute the coefficients of the sumcheck polynomial S(X)
        let e0 = ep_000 * f_000 + ep_010 * f_010 + ep_100 * f_100 + ep_110 * f_110; // Contribution at X = 0
        let e2 = (ep_001 - ep_000) * (f_001 - f_000)
            + (ep_011 - ep_010) * (f_011 - f_010)
            + (ep_101 - ep_100) * (f_101 - f_100)
            + (ep_111 - ep_110) * (f_111 - f_110); // Quadratic coefficient
        let e1 = prover.sum - e0.double() - e2; // Middle coefficient using sum rule

        // Compute sumcheck polynomial evaluations at {0,1,2}
        let eval_0 = e0;
        let eval_1 = e0 + e1 + e2;
        let eval_2 = eval_1 + e1 + e2 + e2.double();
        let expected_evaluations = vec![eval_0, eval_1, eval_2];

        // Assert that computed sumcheck polynomial matches expectations
        assert_eq!(sumcheck_poly.evaluations(), &expected_evaluations);
    }

    #[test]
    fn test_add_new_equality_single_constraint() {
        // Polynomial with 2 variables:
        // f(X1, X2) = c1 + c2*X1 + c3*X2 + c4*X1*X2
        let c1 = F::from(1);
        let c2 = F::from(2);
        let c3 = F::from(3);
        let c4 = F::from(4);
        let coeffs = CoefficientList::new(vec![c1, c2, c3, c4]);

        // Create an empty statement (no constraints initially)
        let statement = Statement::new(2);

        // Instantiate the Sumcheck prover
        let mut prover = SumcheckSingle::new(coeffs, &statement, F::ONE);

        // Add a single constraint at (X1, X2) = (1,0) with weight 2
        let point = MultilinearPoint(vec![F::ONE, F::ZERO]);
        let weight = F::from(2);

        // Compute f(1,0) **without simplifications**
        //
        // f(1,0) = c1 + c2*X1 + c3*X2 + c4*X1*X2
        //        = c1 + c2*(1) + c3*(0) + c4*(1)*(0)
        let eval = c1 + c2 * F::ONE + c3 * F::ZERO + c4 * F::ONE * F::ZERO;

        prover.add_new_equality(&[point.clone()], &[eval], &[weight]);

        // Compute expected sum explicitly:
        //
        // sum = weight * eval
        //     = (2 * (c1 + c2*(1) + c3*(0) + c4*(1)*(0)))
        //
        let expected_sum = weight * eval;
        assert_eq!(prover.sum, expected_sum);

        // Compute the expected weight updates:
        // The equality function at point (X1, X2) = (1,0) updates the weights.
        let mut expected_weights = vec![F::ZERO; 4];
        eval_eq(&point.0, &mut expected_weights, weight);

        assert_eq!(prover.weights.evals(), &expected_weights);
    }

    #[test]
    fn test_add_new_equality_multiple_constraints() {
        // Polynomial with 3 variables:
        // f(X1, X2, X3) = c1 + c2*X1 + c3*X2 + c4*X1*X2 + c5*X3 + c6*X1*X3 + c7*X2*X3 + c8*X1*X2*X3
        let c1 = F::from(1);
        let c2 = F::from(2);
        let c3 = F::from(3);
        let c4 = F::from(4);
        let c5 = F::from(5);
        let c6 = F::from(6);
        let c7 = F::from(7);
        let c8 = F::from(8);
        let coeffs = CoefficientList::new(vec![c1, c2, c3, c4, c5, c6, c7, c8]);

        // Create an empty statement (no constraints initially)
        let statement = Statement::new(3);

        // Instantiate the Sumcheck prover
        let mut prover = SumcheckSingle::new(coeffs, &statement, F::ONE);

        // Add constraints at (X1, X2, X3) = (1,0,1) with weight 2 and (0,1,0) with weight 3
        let point1 = MultilinearPoint(vec![F::ONE, F::ZERO, F::ONE]);
        let point2 = MultilinearPoint(vec![F::ZERO, F::ONE, F::ZERO]);

        let weight1 = F::from(2);
        let weight2 = F::from(3);

        // Compute f(1,0,1) using the polynomial definition:
        //
        // f(1,0,1) = c1 + c2*X1 + c3*X2 + c4*X1*X2 + c5*X3 + c6*X1*X3 + c7*X2*X3 + c8*X1*X2*X3
        //          = c1 + c2*(1) + c3*(0) + c4*(1)*(0) + c5*(1) + c6*(1)*(1) + c7*(0)*(1) +
        // c8*(1)*(0)*(1)
        let eval1 = c1
            + c2 * F::ONE
            + c3 * F::ZERO
            + c4 * F::ONE * F::ZERO
            + c5 * F::ONE
            + c6 * F::ONE * F::ONE
            + c7 * F::ZERO * F::ONE
            + c8 * F::ONE * F::ZERO * F::ONE;

        // Compute f(0,1,0) using the polynomial definition:
        //
        // f(0,1,0) = c1 + c2*X1 + c3*X2 + c4*X1*X2 + c5*X3 + c6*X1*X3 + c7*X2*X3 + c8*X1*X2*X3
        //          = c1 + c2*(0) + c3*(1) + c4*(0)*(1) + c5*(0) + c6*(0)*(0) + c7*(1)*(0) +
        // c8*(0)*(1)*(0)
        let eval2 = c1
            + c2 * F::ZERO
            + c3 * F::ONE
            + c4 * F::ZERO * F::ONE
            + c5 * F::ZERO
            + c6 * F::ZERO * F::ZERO
            + c7 * F::ONE * F::ZERO
            + c8 * F::ZERO * F::ONE * F::ZERO;

        prover.add_new_equality(
            &[point1.clone(), point2.clone()],
            &[eval1, eval2],
            &[weight1, weight2],
        );

        // Compute the expected sum manually:
        //
        // sum = (weight1 * eval1) + (weight2 * eval2)
        let expected_sum = weight1 * eval1 + weight2 * eval2;
        assert_eq!(prover.sum, expected_sum);

        // Expected weight updates
        let mut expected_weights = vec![F::ZERO; 8];
        eval_eq(&point1.0, &mut expected_weights, weight1);
        eval_eq(&point2.0, &mut expected_weights, weight2);

        assert_eq!(prover.weights.evals(), &expected_weights);
    }

    #[test]
    fn test_add_new_equality_with_zero_weight() {
        let c1 = F::from(1);
        let c2 = F::from(2);
        let coeffs = CoefficientList::new(vec![c1, c2]);

        let statement = Statement::new(1);
        let mut prover = SumcheckSingle::new(coeffs, &statement, F::ONE);

        let point = MultilinearPoint(vec![F::ONE]);
        let weight = F::ZERO;
        let eval = F::from(5);

        prover.add_new_equality(&[point], &[eval], &[weight]);

        // The sum should remain unchanged since the weight is zero
        assert_eq!(prover.sum, F::ZERO);

        // The weights should remain unchanged
        let expected_weights = vec![F::ZERO; 2];
        assert_eq!(prover.weights.evals(), &expected_weights);
    }

    #[test]
    fn test_compress_basic() {
        // Polynomial with 2 variables:
        // f(X1, X2) = c1 + c2*X1 + c3*X2 + c4*X1*X2
        let c1 = F::from(1);
        let c2 = F::from(2);
        let c3 = F::from(3);
        let c4 = F::from(4);
        let coeffs = CoefficientList::new(vec![c1, c2, c3, c4]);

        // Create an empty statement (no constraints initially)
        let statement = Statement::new(2);

        // Instantiate the Sumcheck prover
        let mut prover = SumcheckSingle::new(coeffs, &statement, F::ONE);

        // Define random values for compression
        let combination_randomness = F::from(3);
        let folding_randomness = MultilinearPoint(vec![F::from(2)]);

        // Compute sumcheck polynomial manually:
        let sumcheck_poly = prover.compute_sumcheck_polynomial();

        // Apply compression
        prover.compress(combination_randomness, &folding_randomness, &sumcheck_poly);

        // Compute expected evaluations after compression
        //
        // Compression follows the formula:
        //
        // p'(X2) = (p(X1=1, X2) - p(X1=0, X2)) * r + p(X1=0, X2)
        //
        // where r = folding_randomness
        let r = folding_randomness.0[0];

        let eval_00 = c1; // f(0,0) = c1
        let eval_01 = c1 + c3; // f(0,1) = c1 + c3
        let eval_10 = c1 + c2; // f(1,0) = c1 + c2
        let eval_11 = c1 + c2 + c3 + c4; // f(1,1) = c1 + c2 + c3 + c4

        // Compute new evaluations after compression:
        let compressed_eval_0 = (eval_10 - eval_00) * r + eval_00;
        let compressed_eval_1 = (eval_11 - eval_01) * r + eval_01;

        let expected_compressed_evaluations = vec![compressed_eval_0, compressed_eval_1];

        assert_eq!(
            prover.evaluation_of_p.evals(),
            &expected_compressed_evaluations
        );

        // Compute the expected sum update:
        //
        // sum' = combination_randomness * sumcheck_poly.evaluate_at_point(folding_randomness)
        let expected_sum =
            combination_randomness * sumcheck_poly.evaluate_at_point(&folding_randomness);
        assert_eq!(prover.sum, expected_sum);

        // Check weights after compression
        let weight_0 = prover.weights.evals()[0]; // w(X1=0, X2=0)
        let weight_1 = prover.weights.evals()[1]; // w(X1=0, X2=1)

        // Compute compressed weights
        let compressed_weight_0 = weight_0; // No change as X1=0 remains
        let compressed_weight_1 = weight_1; // No change as X1=0 remains

        // The expected compressed weights after applying the transformation
        let expected_compressed_weights = vec![compressed_weight_0, compressed_weight_1];

        assert_eq!(prover.weights.evals(), &expected_compressed_weights);
    }

    #[test]
    fn test_compress_three_variables() {
        // Polynomial with 3 variables:
        // f(X1, X2, X3) = c1 + c2*X1 + c3*X2 + c4*X1*X2 + c5*X3 + c6*X1*X3 + c7*X2*X3 + c8*X1*X2*X3
        let c1 = F::from(1);
        let c2 = F::from(2);
        let c3 = F::from(3);
        let c4 = F::from(4);
        let c5 = F::from(5);
        let c6 = F::from(6);
        let c7 = F::from(7);
        let c8 = F::from(8);
        let coeffs = CoefficientList::new(vec![c1, c2, c3, c4, c5, c6, c7, c8]);

        // Create an empty statement (no constraints initially)
        let statement = Statement::new(3);

        // Instantiate the Sumcheck prover
        let mut prover = SumcheckSingle::new(coeffs, &statement, F::ONE);

        // Define random values for compression
        let combination_randomness = F::from(2);
        let folding_randomness = MultilinearPoint(vec![F::from(3)]);

        // Compute sumcheck polynomial manually:
        let sumcheck_poly = prover.compute_sumcheck_polynomial();

        // Apply compression
        prover.compress(combination_randomness, &folding_randomness, &sumcheck_poly);

        // Compute expected evaluations after compression
        //
        // Compression formula:
        //
        // p'(X2, X3) = (p(X1=1, X2, X3) - p(X1=0, X2, X3)) * r + p(X1=0, X2, X3)
        //
        // where r = folding_randomness
        let r = folding_randomness.0[0];

        let eval_000 = c1;
        let eval_001 = c1 + c5;
        let eval_010 = c1 + c3;
        let eval_011 = c1 + c3 + c5 + c7;
        let eval_100 = c1 + c2;
        let eval_101 = c1 + c2 + c5 + c6;
        let eval_110 = c1 + c2 + c3 + c4;
        let eval_111 = c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8;

        // Compute compressed evaluations
        let compressed_eval_00 = (eval_100 - eval_000) * r + eval_000;
        let compressed_eval_01 = (eval_101 - eval_001) * r + eval_001;
        let compressed_eval_10 = (eval_110 - eval_010) * r + eval_010;
        let compressed_eval_11 = (eval_111 - eval_011) * r + eval_011;

        let expected_compressed_evaluations = vec![
            compressed_eval_00,
            compressed_eval_10,
            compressed_eval_01,
            compressed_eval_11,
        ];

        assert_eq!(
            prover.evaluation_of_p.evals(),
            &expected_compressed_evaluations
        );

        // Compute the expected sum update:
        let expected_sum =
            combination_randomness * sumcheck_poly.evaluate_at_point(&folding_randomness);
        assert_eq!(prover.sum, expected_sum);

        // Check weights after compression
        //
        // Compression formula:
        //
        // w'(X2, X3) = (w(X1=1, X2, X3) - w(X1=0, X2, X3)) * r + w(X1=0, X2, X3)
        //
        let r = folding_randomness.0[0];

        let weight_00 = prover.weights.evals()[0];
        let weight_01 = prover.weights.evals()[1];
        let weight_10 = prover.weights.evals()[2];
        let weight_11 = prover.weights.evals()[3];

        // Apply the same compression rule
        let compressed_weight_00 = (weight_10 - weight_00) * r + weight_00;
        let compressed_weight_01 = (weight_11 - weight_01) * r + weight_01;

        // The compressed weights should match expected values
        let expected_compressed_weights = vec![
            compressed_weight_00,
            compressed_weight_01,
            compressed_weight_00,
            compressed_weight_01,
        ];

        assert_eq!(prover.weights.evals(), &expected_compressed_weights);
    }

    #[test]
    fn test_compress_with_zero_randomness() {
        // Polynomial with 2 variables:
        // f(X1, X2) = c1 + c2*X1 + c3*X2 + c4*X1*X2
        let c1 = F::from(1);
        let c2 = F::from(2);
        let c3 = F::from(3);
        let c4 = F::from(4);
        let coeffs = CoefficientList::new(vec![c1, c2, c3, c4]);

        // Create an empty statement (no constraints initially)
        let statement = Statement::new(2);

        // Instantiate the Sumcheck prover
        let mut prover = SumcheckSingle::new(coeffs, &statement, F::ONE);

        // Define zero folding randomness
        let combination_randomness = F::from(2);
        let folding_randomness = MultilinearPoint(vec![F::ZERO]);

        // Compute sumcheck polynomial manually:
        let sumcheck_poly = prover.compute_sumcheck_polynomial();

        // Apply compression
        prover.compress(combination_randomness, &folding_randomness, &sumcheck_poly);

        // Since folding randomness is zero, the compressed evaluations should be:
        //
        // p'(X2) = (p(X1=1, X2) - p(X1=0, X2)) * 0 + p(X1=0, X2)
        //        = p(X1=0, X2)
        let expected_compressed_evaluations = vec![c1, c1 + c3];

        assert_eq!(
            prover.evaluation_of_p.evals(),
            &expected_compressed_evaluations
        );

        // Compute the expected sum update:
        let expected_sum =
            combination_randomness * sumcheck_poly.evaluate_at_point(&folding_randomness);
        assert_eq!(prover.sum, expected_sum);

        // Check weights after compression
        //
        // Compression formula:
        //
        // w'(X2) = (w(X1=1, X2) - w(X1=0, X2)) * 0 + w(X1=0, X2)
        //        = w(X1=0, X2)
        //
        // Since `r = 0`, this means the weights remain the same as `X1=0` slice.

        let weight_0 = prover.weights.evals()[0]; // w(X1=0, X2=0)
        let weight_1 = prover.weights.evals()[1]; // w(X1=0, X2=1)

        let expected_compressed_weights = vec![weight_0, weight_1];

        assert_eq!(prover.weights.evals(), &expected_compressed_weights);
    }

    #[test]
    fn test_compute_sumcheck_polynomials_basic_case() {
        // Polynomial with 1 variable: f(X1) = 1 + 2*X1
        let c1 = F::from(1);
        let c2 = F::from(2);
        let coeffs = CoefficientList::new(vec![c1, c2]);

        // Create a statement with no equality constraints
        let statement = Statement::new(1);

        // Instantiate the Sumcheck prover
        let mut prover = SumcheckSingle::new(coeffs, &statement, F::ONE);

        // Domain separator setup
        // Step 1: Initialize domain separator with a context label
        let domsep: DomainSeparator<DefaultHash> = DomainSeparator::new("test");

        // Step 2: Register the fact that we’re about to absorb 3 field elements
        let domsep = <DomainSeparator as FieldDomainSeparator<F>>::add_scalars(domsep, 3, "test");

        // Step 3: Sample 1 challenge scalar from the transcript
        let domsep =
            <DomainSeparator as FieldDomainSeparator<F>>::challenge_scalars(domsep, 1, "test");

        // Convert the domain separator to a prover state
        let mut prover_state = domsep.to_prover_state();

        let folding_factor = 1; // Minimum folding factor
        let pow_bits = 0.; // No grinding

        // Compute sumcheck polynomials
        let result = prover
            .compute_sumcheck_polynomials::<Blake3PoW, _>(
                &mut prover_state,
                folding_factor,
                pow_bits,
            )
            .unwrap();

        // The result should contain `folding_factor` elements
        assert_eq!(result.0.len(), folding_factor);
    }

    #[test]
    fn test_compute_sumcheck_polynomials_with_multiple_folding_factors() {
        // Polynomial with 2 variables: f(X1, X2) = 1 + 2*X1 + 3*X2 + 4*X1*X2
        let c1 = F::from(1);
        let c2 = F::from(2);
        let c3 = F::from(3);
        let c4 = F::from(3);
        let coeffs = CoefficientList::new(vec![c1, c2, c3, c4]);

        let statement = Statement::new(2);
        let mut prover = SumcheckSingle::new(coeffs, &statement, F::ONE);

        let folding_factor = 2; // Increase folding factor
        let pow_bits = 1.; // Minimal grinding

        // Setup the domain separator
        let mut domsep: DomainSeparator<DefaultHash> = DomainSeparator::new("test");

        // For each folding round, we must absorb values, sample challenge, and apply PoW
        for _ in 0..folding_factor {
            // Absorb 3 field elements (evaluations of sumcheck polynomial)
            domsep = <DomainSeparator as FieldDomainSeparator<F>>::add_scalars(domsep, 3, "tag");

            // Sample 1 challenge scalar from the Fiat-Shamir transcript
            domsep =
                <DomainSeparator as FieldDomainSeparator<F>>::challenge_scalars(domsep, 1, "tag");

            // Apply optional PoW grinding to ensure randomness
            domsep = domsep.challenge_pow("tag");
        }

        // Convert the domain separator to a prover state
        let mut prover_state = domsep.to_prover_state();

        let result = prover
            .compute_sumcheck_polynomials::<Blake3PoW, _>(
                &mut prover_state,
                folding_factor,
                pow_bits,
            )
            .unwrap();

        // Ensure we get `folding_factor` sampled randomness values
        assert_eq!(result.0.len(), folding_factor);
    }

    #[test]
    fn test_compute_sumcheck_polynomials_with_three_variables() {
        // Multilinear polynomial with 3 variables:
        // f(X1, X2, X3) = 1 + 2*X1 + 3*X2 + 4*X3 + 5*X1*X2 + 6*X1*X3 + 7*X2*X3 + 8*X1*X2*X3
        let c1 = F::from(1);
        let c2 = F::from(2);
        let c3 = F::from(3);
        let c4 = F::from(4);
        let c5 = F::from(5);
        let c6 = F::from(6);
        let c7 = F::from(7);
        let c8 = F::from(8);
        let coeffs = CoefficientList::new(vec![c1, c2, c3, c4, c5, c6, c7, c8]);

        let statement = Statement::new(3);
        let mut prover = SumcheckSingle::new(coeffs, &statement, F::ONE);

        let folding_factor = 3;
        let pow_bits = 2.;

        // Setup the domain separator
        let mut domsep: DomainSeparator<DefaultHash> = DomainSeparator::new("test");

        // Register interactions with the transcript for each round
        for _ in 0..folding_factor {
            // Absorb 3 field values (sumcheck evaluations at X = 0, 1, 2)
            domsep = <DomainSeparator as FieldDomainSeparator<F>>::add_scalars(domsep, 3, "tag");

            // Sample 1 field challenge (folding randomness)
            domsep =
                <DomainSeparator as FieldDomainSeparator<F>>::challenge_scalars(domsep, 1, "tag");

            // Apply challenge PoW (grinding) to enhance soundness
            domsep = domsep.challenge_pow("tag");
        }

        // Convert the domain separator to a prover state
        let mut prover_state = domsep.to_prover_state();

        let result = prover
            .compute_sumcheck_polynomials::<Blake3PoW, _>(
                &mut prover_state,
                folding_factor,
                pow_bits,
            )
            .unwrap();

        assert_eq!(result.0.len(), folding_factor);
    }

    #[test]
    fn test_compute_sumcheck_polynomials_edge_case_zero_folding() {
        // Multilinear polynomial with 2 variables:
        // f(X1, X2) = 1 + 2*X1 + 3*X2 + 4*X1*X2
        let c1 = F::from(1);
        let c2 = F::from(2);
        let c3 = F::from(3);
        let c4 = F::from(4);
        let coeffs = CoefficientList::new(vec![c1, c2, c3, c4]);

        let statement = Statement::new(2);
        let mut prover = SumcheckSingle::new(coeffs, &statement, F::ONE);

        let folding_factor = 0; // Edge case: No folding
        let pow_bits = 1.;

        // No domain separator logic needed since we don't fold
        let domsep: DomainSeparator<DefaultHash> = DomainSeparator::new("test");
        let mut prover_state = domsep.to_prover_state();

        let result = prover
            .compute_sumcheck_polynomials::<Blake3PoW, _>(
                &mut prover_state,
                folding_factor,
                pow_bits,
            )
            .unwrap();

        assert_eq!(result.0.len(), 0);
    }
}
