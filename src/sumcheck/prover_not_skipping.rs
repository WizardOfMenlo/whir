use ark_ff::{Field, PrimeField};
use nimue::{
    plugins::ark::{FieldChallenges, FieldIOPattern, FieldWriter},
    IOPattern, Merlin, ProofResult,
};

use crate::poly_utils::{coeffs::CoefficientList, MultilinearPoint};

use super::prover_single::SumcheckSingle;

pub trait SumcheckNotSkippingIOPattern<F: Field> {
    fn add_sumcheck(self, folding_factor: usize) -> Self;
}

impl<F> SumcheckNotSkippingIOPattern<F> for IOPattern
where
    F: Field,
    IOPattern: FieldIOPattern<F>,
{
    fn add_sumcheck(mut self, folding_factor: usize) -> Self {
        for _ in 0..folding_factor {
            self = self
                .add_scalars(3, "sumcheck_poly")
                .challenge_scalars(1, "folding_randomness");
        }
        self
    }
}

pub struct SumcheckProverNotSkipping<F> {
    sumcheck_prover: SumcheckSingle<F>,
}

impl<F> SumcheckProverNotSkipping<F>
where
    F: Field + PrimeField,
{
    // Get the coefficient of polynomial p and a list of points
    // and initialises the table of the initial polynomial
    // v(X_1, ..., X_n) = p(X_1, ... X_n) * (epsilon_1 eq_z_1(X) + epsilon_2 eq_z_2(X) ...)
    pub fn new(
        coeffs: CoefficientList<F>,
        points: &[MultilinearPoint<F>],
        combination_randomness: &[F],
    ) -> Self {
        Self {
            sumcheck_prover: SumcheckSingle::new(coeffs, points, combination_randomness),
        }
    }

    pub fn compute_sumcheck_polynomials(
        &mut self,
        merlin: &mut Merlin,
        folding_factor: usize,
    ) -> ProofResult<MultilinearPoint<F>> {
        let mut res = Vec::with_capacity(folding_factor);

        for _ in 0..folding_factor {
            let sumcheck_poly = self.sumcheck_prover.compute_sumcheck_polynomial();
            merlin.add_scalars(sumcheck_poly.evaluations())?;
            let [folding_randomness]: [F; 1] = merlin.challenge_scalars()?;
            res.push(folding_randomness);

            self.sumcheck_prover
                .compress(F::ONE, &folding_randomness.into());
        }

        res.reverse();
        Ok(MultilinearPoint(res))
    }

    pub fn add_new_equality(
        &mut self,
        points: &[MultilinearPoint<F>],
        combination_randomness: &[F],
    ) {
        self.sumcheck_prover
            .add_new_equality(points, combination_randomness)
    }
}

#[cfg(test)]
mod tests {
    use ark_ff::Field;
    use nimue::{
        plugins::ark::{FieldChallenges, FieldIOPattern, FieldReader},
        IOPattern, ProofResult,
    };

    use crate::{
        crypto::fields::Field64,
        poly_utils::{coeffs::CoefficientList, eq_poly_outside, MultilinearPoint},
        sumcheck::{proof::SumcheckPolynomial, prover_not_skipping::SumcheckProverNotSkipping},
    };

    type F = Field64;

    #[test]
    fn test_e2e_short() -> ProofResult<()> {
        let num_variables = 2;
        let folding_factor = 2;
        let polynomial = CoefficientList::new((0..1 << num_variables).map(F::from).collect());

        // Initial stuff
        let ood_point = MultilinearPoint::expand_from_univariate(F::from(42), num_variables);
        let statement_point = MultilinearPoint::expand_from_univariate(F::from(97), num_variables);

        // All the randomness
        let [epsilon_1, epsilon_2] = [F::from(15), F::from(32)];

        fn add_sumcheck_io_pattern<F>() -> IOPattern
        where
            F: Field,
            IOPattern: FieldIOPattern<F>,
        {
            IOPattern::new("test")
                .add_scalars(3, "sumcheck_poly")
                .challenge_scalars(1, "folding_randomness")
                .add_scalars(3, "sumcheck_poly")
                .challenge_scalars(1, "folding_randomness")
        }

        let iopattern = add_sumcheck_io_pattern::<F>();

        // Prover part
        let mut merlin = iopattern.to_merlin();
        let mut prover = SumcheckProverNotSkipping::new(
            polynomial.clone(),
            &[ood_point.clone(), statement_point.clone()],
            &[epsilon_1, epsilon_2],
        );

        let folding_randomness_1 =
            prover.compute_sumcheck_polynomials(&mut merlin, folding_factor)?;

        // Compute the answers
        let folded_poly_1 = polynomial.fold(&folding_randomness_1);

        let ood_answer = polynomial.evaluate(&ood_point);
        let statement_answer = polynomial.evaluate(&statement_point);

        // Verifier part
        let mut arthur = iopattern.to_arthur(merlin.transcript());
        let sumcheck_poly_11: [F; 3] = arthur.next_scalars()?;
        let sumcheck_poly_11 = SumcheckPolynomial::new(sumcheck_poly_11.to_vec(), 1);
        let [folding_randomness_11]: [F; 1] = arthur.challenge_scalars()?;
        let sumcheck_poly_12: [F; 3] = arthur.next_scalars()?;
        let sumcheck_poly_12 = SumcheckPolynomial::new(sumcheck_poly_12.to_vec(), 1);
        let [folding_randomness_12]: [F; 1] = arthur.challenge_scalars()?;

        assert_eq!(
            sumcheck_poly_11.sum_over_hypercube(),
            epsilon_1 * ood_answer + epsilon_2 * statement_answer
        );

        assert_eq!(
            sumcheck_poly_12.sum_over_hypercube(),
            sumcheck_poly_11.evaluate_at_point(&folding_randomness_11.into())
        );

        let full_folding = MultilinearPoint(vec![folding_randomness_12, folding_randomness_11]);

        let eval_coeff = folded_poly_1.coeffs()[0];
        assert_eq!(
            sumcheck_poly_12.evaluate_at_point(&folding_randomness_12.into()),
            eval_coeff
                * (epsilon_1 * eq_poly_outside(&full_folding, &ood_point)
                    + epsilon_2 * eq_poly_outside(&full_folding, &statement_point))
        );

        Ok(())
    }

    #[test]
    fn test_e2e() -> ProofResult<()> {
        let num_variables = 4;
        let folding_factor = 2;
        let polynomial = CoefficientList::new((0..1 << num_variables).map(F::from).collect());

        // Initial stuff
        let ood_point = MultilinearPoint::expand_from_univariate(F::from(42), num_variables);
        let statement_point = MultilinearPoint::expand_from_univariate(F::from(97), num_variables);

        // All the randomness
        let [epsilon_1, epsilon_2] = [F::from(15), F::from(32)];
        let fold_point = MultilinearPoint(vec![F::from(31), F::from(15)]);
        let combination_randomness = vec![F::from(1000)];

        fn add_sumcheck_io_pattern<F>() -> IOPattern
        where
            F: Field,
            IOPattern: FieldIOPattern<F>,
        {
            IOPattern::new("test")
                .add_scalars(3, "sumcheck_poly")
                .challenge_scalars(1, "folding_randomness")
                .add_scalars(3, "sumcheck_poly")
                .challenge_scalars(1, "folding_randomness")
                .add_scalars(3, "sumcheck_poly")
                .challenge_scalars(1, "folding_randomness")
                .add_scalars(3, "sumcheck_poly")
                .challenge_scalars(1, "folding_randomness")
        }

        let iopattern = add_sumcheck_io_pattern::<F>();

        // Prover part
        let mut merlin = iopattern.to_merlin();
        let mut prover = SumcheckProverNotSkipping::new(
            polynomial.clone(),
            &[ood_point.clone(), statement_point.clone()],
            &[epsilon_1, epsilon_2],
        );

        let folding_randomness_1 =
            prover.compute_sumcheck_polynomials(&mut merlin, folding_factor)?;
        prover.add_new_equality(&[fold_point.clone()], &combination_randomness);
        let folding_randomness_2 =
            prover.compute_sumcheck_polynomials(&mut merlin, folding_factor)?;

        // Compute the answers
        let folded_poly_1 = polynomial.fold(&folding_randomness_1);
        let folded_poly_2 = folded_poly_1.fold(&folding_randomness_2);

        let ood_answer = polynomial.evaluate(&ood_point);
        let statement_answer = polynomial.evaluate(&statement_point);
        let fold_answer = folded_poly_1.evaluate(&fold_point);

        // Verifier part
        let mut arthur = iopattern.to_arthur(merlin.transcript());
        let sumcheck_poly_11: [F; 3] = arthur.next_scalars()?;
        let sumcheck_poly_11 = SumcheckPolynomial::new(sumcheck_poly_11.to_vec(), 1);
        let [folding_randomness_11]: [F; 1] = arthur.challenge_scalars()?;
        let sumcheck_poly_12: [F; 3] = arthur.next_scalars()?;
        let sumcheck_poly_12 = SumcheckPolynomial::new(sumcheck_poly_12.to_vec(), 1);
        let [folding_randomness_12]: [F; 1] = arthur.challenge_scalars()?;
        let sumcheck_poly_21: [F; 3] = arthur.next_scalars()?;
        let sumcheck_poly_21 = SumcheckPolynomial::new(sumcheck_poly_21.to_vec(), 1);
        let [folding_randomness_21]: [F; 1] = arthur.challenge_scalars()?;
        let sumcheck_poly_22: [F; 3] = arthur.next_scalars()?;
        let sumcheck_poly_22 = SumcheckPolynomial::new(sumcheck_poly_22.to_vec(), 1);
        let [folding_randomness_22]: [F; 1] = arthur.challenge_scalars()?;

        assert_eq!(
            sumcheck_poly_11.sum_over_hypercube(),
            epsilon_1 * ood_answer + epsilon_2 * statement_answer
        );

        assert_eq!(
            sumcheck_poly_12.sum_over_hypercube(),
            sumcheck_poly_11.evaluate_at_point(&folding_randomness_11.into())
        );

        assert_eq!(
            sumcheck_poly_21.sum_over_hypercube(),
            sumcheck_poly_12.evaluate_at_point(&folding_randomness_12.into())
                + combination_randomness[0] * fold_answer
        );

        assert_eq!(
            sumcheck_poly_22.sum_over_hypercube(),
            sumcheck_poly_21.evaluate_at_point(&folding_randomness_21.into())
        );

        let full_folding = MultilinearPoint(vec![
            folding_randomness_22,
            folding_randomness_21,
            folding_randomness_12,
            folding_randomness_11,
        ]);

        let partial_folding = MultilinearPoint(vec![folding_randomness_22, folding_randomness_21]);

        let eval_coeff = folded_poly_2.coeffs()[0];
        assert_eq!(
            sumcheck_poly_22.evaluate_at_point(&folding_randomness_22.into()),
            eval_coeff
                * ((epsilon_1 * eq_poly_outside(&full_folding, &ood_point)
                    + epsilon_2 * eq_poly_outside(&full_folding, &statement_point))
                    + combination_randomness[0] * eq_poly_outside(&partial_folding, &fold_point))
        );

        Ok(())
    }
}
