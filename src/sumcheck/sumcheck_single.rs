use super::SumcheckPolynomial;
use crate::{
    poly_utils::{coeffs::CoefficientList, evals::EvaluationsList, multilinear::MultilinearPoint},
    whir::statement::Statement,
};

use ark_ff::Field;
use nimue::{
    plugins::ark::{FieldChallenges, FieldWriter},
    ProofResult,
};
use nimue_pow::{PoWChallenge, PowStrategy};

#[cfg(feature = "parallel")]
use rayon::{join, prelude::*};

/// Sumcheck instance for a single weighted sum of an MLE.
pub struct SumcheckSingle<F> {
    // The evaluation of p
    evaluation_of_p: EvaluationsList<F>,
    weights: EvaluationsList<F>,
    sum: F,
}

impl<F> SumcheckSingle<F>
where
    F: Field,
{
    // Get the coefficient of polynomial p, statements and combination randomness element
    // and initialise tables of polynomial evaluations and a random linear combination of
    // statements using the combination randomness element
    pub fn new(
        coeffs: CoefficientList<F>,
        statement: &Statement<F>,
        combination_randomness_gen: F,
    ) -> Self {
        let (weights, sum) = statement.combine(combination_randomness_gen);

        SumcheckSingle {
            evaluation_of_p: coeffs.into(),
            weights,
            sum,
        }
    }

    pub fn num_variables(&self) -> usize {
        self.evaluation_of_p.num_variables()
    }

    /// Compute the polynomial that represents the sum in the first variable.
    #[cfg(not(feature = "parallel"))]
    pub fn compute_sumcheck_polynomial(&self) -> SumcheckPolynomial<F> {
        assert!(self.num_variables >= 1);

        // Compute coefficients of the quadratic result polynomial
        let eval_p_iter = self.evaluation_of_p.evals().chunks_exact(2);
        let eval_eq_iter = self.weights.evals().chunks_exact(2);
        let (c0, c2) = eval_p_iter
            .zip(eval_eq_iter)
            .map(|(p_at, eq_at)| {
                // Convert evaluations to coefficients for the linear fns p and eq.
                let (p_0, p_1) = (p_at[0], p_at[1] - p_at[0]);
                let (eq_0, eq_1) = (eq_at[0], eq_at[1] - eq_at[0]);

                // Now we need to add the contribution of p(x) * eq(x)
                (p_0 * eq_0, p_1 * eq_1)
            })
            .reduce(|(a0, a2), (b0, b2)| (a0 + b0, a2 + b2))
            .unwrap_or((F::ZERO, F::ZERO));

        // Use the fact that self.sum = p(0) + p(1) = 2 * c0 + c1 + c2
        let c1 = self.sum - c0.double() - c2;

        // Evaluate the quadratic polynomial at 0, 1, 2
        let eval_0 = c0;
        let eval_1 = c0 + c1 + c2;
        let eval_2 = eval_1 + c1 + c2 + c2.double();

        SumcheckPolynomial::new(vec![eval_0, eval_1, eval_2], 1)
    }

    /// Compute the polynomial that represents the sum in the first variable.
    #[cfg(feature = "parallel")]
    pub fn compute_sumcheck_polynomial(&self) -> SumcheckPolynomial<F> {
        assert!(self.num_variables() >= 1);

        // Compute coefficients of the quadratic result polynomial
        let eval_p_iter = self.evaluation_of_p.evals().par_chunks_exact(2);
        let eval_eq_iter = self.weights.evals().par_chunks_exact(2);
        let (c0, c2) = eval_p_iter
            .zip(eval_eq_iter)
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

        // Use the fact that self.sum = p(0) + p(1) = 2 * coeff_0 + coeff_1 + coeff_2
        let c1 = self.sum - c0.double() - c2;

        // Evaluate the quadratic polynomial at 0, 1, 2
        let eval_0 = c0;
        let eval_1 = c0 + c1 + c2;
        let eval_2 = eval_1 + c1 + c2 + c2.double();

        SumcheckPolynomial::new(vec![eval_0, eval_1, eval_2], 1)
    }

    /// Do `folding_factor` rounds of sumcheck, and return the proof.
    pub fn compute_sumcheck_polynomials<S, Merlin>(
        &mut self,
        merlin: &mut Merlin,
        folding_factor: usize,
        pow_bits: f64,
    ) -> ProofResult<MultilinearPoint<F>>
    where
        Merlin: FieldWriter<F> + FieldChallenges<F> + PoWChallenge,
        S: PowStrategy,
    {
        let mut res = Vec::with_capacity(folding_factor);

        for _ in 0..folding_factor {
            let sumcheck_poly = self.compute_sumcheck_polynomial();
            merlin.add_scalars(sumcheck_poly.evaluations())?;
            let [folding_randomness]: [F; 1] = merlin.challenge_scalars()?;
            res.push(folding_randomness);

            // Do PoW if needed
            if pow_bits > 0. {
                merlin.challenge_pow::<S>(pow_bits)?;
            }

            self.compress(F::ONE, &folding_randomness.into(), &sumcheck_poly);
        }

        res.reverse();
        Ok(MultilinearPoint(res))
    }

    /// Adds new equality constraints to the polynomial.
    ///
    /// Computes:
    ///
    /// ```text
    /// eq(X) = ∑ ε_i * eq_{z_i}(X)
    /// ```
    ///
    /// where:
    /// - `ε_i` are weighting factors.
    /// - `eq_{z_i}(X)` is the equality polynomial ensuring `X = z_i`.
    pub fn add_new_equality(
        &mut self,
        points: &[MultilinearPoint<F>],
        combination_randomness: &[F],
        evaluations: &[F],
    ) {
        assert_eq!(combination_randomness.len(), points.len());
        assert_eq!(combination_randomness.len(), evaluations.len());
for (point, rand) in points.iter().zip(combination_randomness) {
            // TODO: We might want to do all points simultaneously so we
            // do only a single pass over the data.
            Self::eval_eq(&point.0, self.weights.evals_mut(), *rand);
        }
        // Update the sum
        for (rand, eval) in combination_randomness.iter().zip(evaluations.iter()) {
            self.sum += *rand * eval;
        }
    }

    // When the folding randomness arrives, compress the table accordingly (adding the new points)
    #[cfg(not(feature = "parallel"))]
    pub fn compress(
        &mut self,
        combination_randomness: F, // Scale the initial point
        folding_randomness: &MultilinearPoint<F>,
        sumcheck_poly: &SumcheckPolynomial<F>,
    ) {
        assert_eq!(folding_randomness.n_variables(), 1);
        assert!(self.num_variables >= 1);

        let randomness = folding_randomness.0[0];
        let evaluations_of_p = self
            .evaluation_of_p
            .evals()
            .chunks_exact(2)
            .map(|at| (at[1] - at[0]) * randomness + at[0])
            .collect();
        let evaluations_of_eq = self
            .weights
            .evals()
            .chunks_exact(2)
            .map(|at| (at[1] - at[0]) * randomness + at[0])
            .collect();

        // Update
        self.num_variables -= 1;
        self.evaluation_of_p = EvaluationsList::new(evaluations_of_p);
        self.weights = EvaluationsList::new(evaluations_of_eq);
        self.sum = combination_randomness * sumcheck_poly.evaluate_at_point(folding_randomness);
    }

    #[cfg(feature = "parallel")]
    pub fn compress(
        &mut self,
        combination_randomness: F, // Scale the initial point
        folding_randomness: &MultilinearPoint<F>,
        sumcheck_poly: &SumcheckPolynomial<F>,
    ) {
        assert_eq!(folding_randomness.num_variables(), 1);
        assert!(self.num_variables() >= 1);

        let randomness = folding_randomness.0[0];
        let (evaluations_of_p, evaluations_of_eq) = join(
            || {
                self.evaluation_of_p
                    .evals()
                    .par_chunks_exact(2)
                    .map(|at| (at[1] - at[0]) * randomness + at[0])
                    .collect()
            },
            || {
                self.weights
                    .evals()
                    .par_chunks_exact(2)
                    .map(|at| (at[1] - at[0]) * randomness + at[0])
                    .collect()
            },
        );

        // Update
        self.evaluation_of_p = EvaluationsList::new(evaluations_of_p);
        self.weights = EvaluationsList::new(evaluations_of_eq);
        self.sum = combination_randomness * sumcheck_poly.evaluate_at_point(folding_randomness);
    }
}

// Evaluate the eq function on for a given point on the hypercube, and add
// the result multiplied by the scalar to the output.
#[cfg(not(feature = "parallel"))]
fn eval_eq<F: Field>(eval: &[F], out: &mut [F], scalar: F) {
    debug_assert_eq!(out.len(), 1 << eval.len());
    if let Some((&x, tail)) = eval.split_first() {
        let (low, high) = out.split_at_mut(out.len() / 2);
        let s1 = scalar * x;
        let s0 = scalar - s1;
        Self::eval_eq(tail, low, s0);
        Self::eval_eq(tail, high, s1);
    } else {
        out[0] += scalar;
    }
}

/// Computes the equality polynomial evaluations efficiently.
///
/// Given an evaluation point vector `eval`, the function computes
/// the equality polynomial recursively using the formula:
///
/// ```text
/// eq(X) = ∏ (1 - X_i + 2X_i z_i)
/// ```
///
/// where `z_i` are the constraint points.
#[cfg(feature = "parallel")]
fn eval_eq<F: Field>(eval: &[F], out: &mut [F], scalar: F) {
    const PARALLEL_THRESHOLD: usize = 10;

    // Ensure that the output buffer size is correct:
    // It should be of size `2^n`, where `n` is the number of variables.
    debug_assert_eq!(out.len(), 1 << eval.len());

    // Base case: When there are no more variables to process, update the final value.
    if let Some((&x, tail)) = eval.split_first() {
        // Divide the output buffer into two halves: one for `X_i = 0` and one for `X_i = 1`
        let (low, high) = out.split_at_mut(out.len() / 2);

        // Compute weight updates for the two branches:
        // - `s0` corresponds to the case when `X_i = 0`
        // - `s1` corresponds to the case when `X_i = 1`
        //
        // Mathematically, this follows the recurrence:
        // ```text
        // eq_{X1, ..., Xn}(X) = (1 - X_1) * eq_{X2, ..., Xn}(X) + X_1 * eq_{X2, ..., Xn}(X)
        // ```
        let s1 = scalar * x; // Contribution when `X_i = 1`
        let s0 = scalar - s1; // Contribution when `X_i = 0`

        // Use parallel execution if the number of remaining variables is large.
        if tail.len() > PARALLEL_THRESHOLD {
            join(|| eval_eq(tail, low, s0), || eval_eq(tail, high, s1));
        } else {
            eval_eq(tail, low, s0);
            eval_eq(tail, high, s1);
        }
    } else {
        // Leaf case: Add the accumulated scalar to the final output slot.
        out[0] += scalar;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        crypto::fields::Field64 as F,
        poly_utils::{
            coeffs::CoefficientList, multilinear::MultilinearPoint,
            sequential_lag_poly::LagrangePolynomialIterator,
        },
        whir::statement::Weights,
    };
    use ark_ff::AdditiveGroup;

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

        let combination_randomness = F::from(100101);
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

        let claimed_value: ark_ff::Fp<ark_ff::MontBackend<crate::crypto::fields::FConfig64, 1>, 1> =
            polynomial.evaluate(&eval_point);

        let eval = polynomial.evaluate(&eval_point);

        let mut statement = Statement::new(eval_point.num_variables());
        let weights = Weights::evaluation(eval_point);
        statement.add_constraint(weights.clone(), eval);

        let mut prover = SumcheckSingle::new(polynomial, &statement, F::from(1));

        let poly_1 = prover.compute_sumcheck_polynomial();
        // First, check that is sums to the right value over the hypercube
        assert_eq!(poly_1.sum_over_boolean_hypercube(), claimed_value);

        let combination_randomness = F::from(100101);
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
        dbg!(&out);

        let point = MultilinearPoint(eval.clone());
        let mut expected = vec![F::ZERO; 4];
        for (prefix, lag) in LagrangePolynomialIterator::from(&point) {
            expected[prefix.0] = lag;
        }
        dbg!(&expected);

        assert_eq!(&out, &expected);
    }
}
