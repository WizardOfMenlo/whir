use super::SumcheckPolynomial;
use crate::{poly_utils::{coeffs::CoefficientList, evals::EvaluationsList, MultilinearPoint}, whir::statement::Statement};
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
    // Get the coefficient of polynomial p and a list of points
    // and initialises the table of the initial polynomial
    // v(X_1, ..., X_n) = p(X_1, ... X_n) * (epsilon_1 eq_z_1(X) + epsilon_2 eq_z_2(X) ...)
    pub fn new(coeffs: CoefficientList<F>) -> Self {
        let weights = EvaluationsList::new(vec![F::ZERO; 1 << coeffs.num_variables()]);
           
        SumcheckSingle {
            evaluation_of_p: coeffs.into(),
            weights,
            sum: F::ZERO,
        }
    }

    pub fn num_variables(&self) -> usize {
        self.evaluation_of_p.num_variables()
    }
    #[cfg(not(feature = "parallel"))]
    pub fn add_weighted_sum(
        &mut self,
        statement: &Statement<F>,
        combination_randomness_gen : F
    ) {
        assert_eq!(statement.num_variables(), self.num_variables());
        (self.weights, self.sum) = statement.combine(combination_randomness_gen);
    }
    
    #[cfg(feature = "parallel")]
    pub fn add_weighted_sum(
        &mut self,
        statement: &Statement<F>,
        combination_randomness_gen : F
    ) {
        assert_eq!(statement.num_variables(), self.num_variables());
        (self.weights, self.sum) = statement.combine_parallel(combination_randomness_gen);
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

    // Evaluate the eq function on for a given point on the hypercube, and add
    // the result multiplied by the scalar to the output.
    #[cfg(not(feature = "parallel"))]
    fn eval_eq(eval: &[F], out: &mut [F], scalar: F) {
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

    // Evaluate the eq function on a given point on the hypercube, and add
    // the result multiplied by the scalar to the output.
    #[cfg(feature = "parallel")]
    fn eval_eq(eval: &[F], out: &mut [F], scalar: F) {
        const PARALLEL_THRESHOLD: usize = 10;
        debug_assert_eq!(out.len(), 1 << eval.len());
        if let Some((&x, tail)) = eval.split_first() {
            let (low, high) = out.split_at_mut(out.len() / 2);
            // Update scalars using a single mul. Note that this causes a data dependency,
            // so for small fields it might be better to use two muls.
            // This data dependency should go away once we implement parallel point evaluation.
            let s1 = scalar * x;
            let s0 = scalar - s1;
            if tail.len() > PARALLEL_THRESHOLD {
                join(
                    || Self::eval_eq(tail, low, s0),
                    || Self::eval_eq(tail, high, s1),
                );
            } else {
                Self::eval_eq(tail, low, s0);
                Self::eval_eq(tail, high, s1);
            }
        } else {
            out[0] += scalar;
        }
    }

    pub fn add_new_equality(
        &mut self,
        points: &[MultilinearPoint<F>],
        evaluations: &[F],
        combination_randomness: &[F],
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        crypto::fields::Field64 as F,
        poly_utils::{
            coeffs::CoefficientList, sequential_lag_poly::LagrangePolynomialIterator,
            MultilinearPoint,
        }, whir::statement::Weights,
    };
    use ark_ff::AdditiveGroup;

    #[test]
    fn test_sumcheck_folding_factor_1() {
        let eval_point = MultilinearPoint(vec![F::from(10), F::from(11)]);
        let polynomial =
            CoefficientList::new(vec![F::from(1), F::from(5), F::from(10), F::from(14)]);

        let claimed_value = polynomial.evaluate(&eval_point);

        let eval = polynomial.evaluate(&eval_point);
        let mut prover = SumcheckSingle::new(polynomial);
        prover.add_new_equality(&[eval_point], &[eval], &[F::from(1)]);

        let poly_1 = prover.compute_sumcheck_polynomial();

        // First, check that is sums to the right value over the hypercube
        assert_eq!(poly_1.sum_over_hypercube(), claimed_value);

        let combination_randomness = F::from(100101);
        let folding_randomness = MultilinearPoint(vec![F::from(4999)]);

        prover.compress(combination_randomness, &folding_randomness, &poly_1);

        let poly_2 = prover.compute_sumcheck_polynomial();

        assert_eq!(
            poly_2.sum_over_hypercube(),
            combination_randomness * poly_1.evaluate_at_point(&folding_randomness)
        );
    }



    #[test]
    fn test_sumcheck_weighted_folding_factor_1() {
        let eval_point = MultilinearPoint(vec![F::from(10), F::from(11)]);
        let polynomial =
            CoefficientList::new(vec![F::from(1), F::from(5), F::from(10), F::from(14)]);

        let claimed_value: ark_ff::Fp<ark_ff::MontBackend<crate::crypto::fields::FConfig64, 1>, 1> = polynomial.evaluate(&eval_point);

        let eval = polynomial.evaluate(&eval_point);
        let mut prover = SumcheckSingle::new(polynomial);

        let mut statement = Statement::new(eval_point.num_variables());
        let weights = Weights::evaluation(eval_point);
        statement.add_constraint(weights.clone(), eval);

        prover.add_weighted_sum(&statement, F::from(1));

        let poly_1 = prover.compute_sumcheck_polynomial();
        // First, check that is sums to the right value over the hypercube
        assert_eq!(poly_1.sum_over_hypercube(), claimed_value);

        let combination_randomness = F::from(100101);
        let folding_randomness = MultilinearPoint(vec![F::from(4999)]);

        prover.compress(combination_randomness, &folding_randomness, &poly_1);

        let poly_2 = prover.compute_sumcheck_polynomial();

        assert_eq!(
            poly_2.sum_over_hypercube(),
            combination_randomness * poly_1.evaluate_at_point(&folding_randomness)
        );
    }

    #[test]
    fn test_eval_eq() {
        let eval = vec![F::from(3), F::from(5)];
        let mut out = vec![F::ZERO; 4];
        SumcheckSingle::eval_eq(&eval, &mut out, F::ONE);
        dbg!(&out);

        let point = MultilinearPoint(eval.clone());
        let mut expected = vec![F::ZERO; 4];
        for (prefix, lag) in LagrangePolynomialIterator::new(&point) {
            expected[prefix.0] = lag;
        }
        dbg!(&expected);

        assert_eq!(&out, &expected);
    }
}
