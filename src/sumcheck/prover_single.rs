use ark_ff::Field;

use crate::poly_utils::{
    coeffs::CoefficientList, evals::EvaluationsList, lag_poly::LagrangePolynomialIterator,
    MultilinearPoint,
};

use super::proof::SumcheckPolynomial;

pub struct SumcheckSingle<F> {
    // The evaluation of p
    evaluation_of_p: EvaluationsList<F>,
    evaluation_of_equality: EvaluationsList<F>,
    num_variables: usize,
}

impl<F> SumcheckSingle<F>
where
    F: Field,
{
    // Get the coefficient of polynomial p and a list of points
    // and initialises the table of the initial polynomial
    // v(X_1, ..., X_n) = p(X_1, ... X_n) * (epsilon_1 eq_z_1(X) + epsilon_2 eq_z_2(X) ...)
    pub fn new(
        coeffs: CoefficientList<F>,
        points: &[MultilinearPoint<F>],
        combination_randomness: &[F],
    ) -> Self {
        assert_eq!(points.len(), combination_randomness.len());
        let num_variables = coeffs.num_variables();

        let mut prover = SumcheckSingle {
            evaluation_of_p: coeffs.into(),
            evaluation_of_equality: EvaluationsList::new(vec![F::ZERO; 1 << num_variables]),
            num_variables,
        };

        prover.add_new_equality(points, combination_randomness);
        prover
    }

    pub fn compute_sumcheck_polynomial(&self) -> SumcheckPolynomial<F> {
        let two = F::ONE + F::ONE; // Enlightening

        assert!(self.num_variables >= 1);

        let prefix_len = 1 << (self.num_variables - 1);

        let mut sum_0 = F::ZERO;
        let mut sum_1 = F::ZERO;
        let mut sum_2 = F::ZERO;

        for beta_prefix in 0..prefix_len {
            let eval_of_p_0 = self.evaluation_of_p[2 * beta_prefix];
            let eval_of_p_1 = self.evaluation_of_p[2 * beta_prefix + 1];

            let p_0 = eval_of_p_0;
            let p_1 = eval_of_p_1 - eval_of_p_0;

            let eval_of_eq_0 = self.evaluation_of_equality[2 * beta_prefix];
            let eval_of_eq_1 = self.evaluation_of_equality[2 * beta_prefix + 1];

            let w_0 = eval_of_eq_0;
            let w_1 = eval_of_eq_1 - eval_of_eq_0;

            sum_0 += p_0 * w_0;
            sum_1 += w_1 * p_0 + w_0 * p_1;
            sum_2 += p_1 * w_1;
        }

        let eval_0 = sum_0;
        let eval_1 = sum_0 + sum_1 + sum_2;
        let eval_2 = sum_0 + two * sum_1 + two * two * sum_2;

        SumcheckPolynomial::new(vec![eval_0, eval_1, eval_2], 1)
    }

    pub fn add_new_equality(
        &mut self,
        points: &[MultilinearPoint<F>],
        combination_randomness: &[F],
    ) {
        assert_eq!(combination_randomness.len(), points.len());
        for (point, rand) in points.iter().zip(combination_randomness) {
            for (prefix, lag) in LagrangePolynomialIterator::new(point) {
                self.evaluation_of_equality.evals_mut()[prefix.0] += *rand * lag;
            }
        }
    }

    // When the folding randomness arrives, compress the table accordingly (adding the new points)
    pub fn compress(
        &mut self,
        combination_randomness: F, // Scale the initial point
        folding_randomness: &MultilinearPoint<F>,
    ) {
        assert_eq!(folding_randomness.n_variables(), 1);
        assert!(self.num_variables >= 1);

        let randomness = folding_randomness.0[0];
        let randomness_bar = F::ONE - randomness;

        let prefix_len = 1 << (self.num_variables - 1);
        let mut evaluations_of_p = Vec::with_capacity(prefix_len);
        let mut evaluations_of_eq = Vec::with_capacity(prefix_len);

        // Compress the table
        for beta_prefix in 0..prefix_len {
            let eval_of_p_0 = self.evaluation_of_p[2 * beta_prefix];
            let eval_of_p_1 = self.evaluation_of_p[2 * beta_prefix + 1];
            let eval_of_p = eval_of_p_0 * randomness_bar + eval_of_p_1 * randomness;

            let eval_of_eq_0 = self.evaluation_of_equality[2 * beta_prefix];
            let eval_of_eq_1 = self.evaluation_of_equality[2 * beta_prefix + 1];
            let eval_of_eq = eval_of_eq_0 * randomness_bar + eval_of_eq_1 * randomness;

            evaluations_of_p.push(eval_of_p);
            evaluations_of_eq.push(combination_randomness * eval_of_eq);
        }

        // Update
        self.num_variables -= 1;
        self.evaluation_of_p = EvaluationsList::new(evaluations_of_p);
        self.evaluation_of_equality = EvaluationsList::new(evaluations_of_eq);
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        crypto::fields::Field64,
        poly_utils::{coeffs::CoefficientList, MultilinearPoint},
    };

    use super::SumcheckSingle;

    type F = Field64;

    #[test]
    fn test_sumcheck_folding_factor_1() {
        let eval_point = MultilinearPoint(vec![F::from(10), F::from(11)]);
        let polynomial =
            CoefficientList::new(vec![F::from(1), F::from(5), F::from(10), F::from(14)]);

        let claimed_value = polynomial.evaluate(&eval_point);

        let mut prover = SumcheckSingle::new(polynomial, &[eval_point], &[F::from(1)]);

        let poly_1 = prover.compute_sumcheck_polynomial();

        // First, check that is sums to the right value over the hypercube
        assert_eq!(poly_1.sum_over_hypercube(), claimed_value);

        let combination_randomness = F::from(100101);
        let folding_randomness = MultilinearPoint(vec![F::from(4999)]);

        prover.compress(combination_randomness, &folding_randomness);

        let poly_2 = prover.compute_sumcheck_polynomial();

        assert_eq!(
            poly_2.sum_over_hypercube(),
            combination_randomness * poly_1.evaluate_at_point(&folding_randomness)
        );
    }
}
