use super::{evals::EvaluationsList, hypercube::BinaryHypercubePoint, MultilinearPoint};
use ark_ff::Field;
use ark_poly::{univariate::DensePolynomial, DenseUVPolynomial, Polynomial};

#[derive(Debug, Clone)]
pub struct CoefficientList<F> {
    coeffs: Vec<F>,
    num_variables: usize,
}

impl<F> CoefficientList<F>
where
    F: Field,
{
    pub fn evaluate_hypercube(&self, point: BinaryHypercubePoint) -> F {
        assert_eq!(self.coeffs.len(), 1 << self.num_variables);
        assert!(point.0 < (1 << self.num_variables));
        // TODO: Optimized implementation
        self.evaluate(&MultilinearPoint::from_binary_hypercube_point(
            point,
            self.num_variables,
        ))
    }

    pub fn evaluate(&self, point: &MultilinearPoint<F>) -> F {
        assert_eq!(self.num_variables, point.n_variables());

        /*
        use crate::poly_utils::streaming_evaluation_helper::TermPolynomialIterator;
        let mut sum = F::ZERO;
        for (b, term) in TermPolynomialIterator::new(point) {
            sum += self.coeffs[b.0] * term;
        }
        */

        Self::eval(self.coeffs(), &point.0, F::ONE)
    }

    fn eval(coeff: &[F], eval: &[F], scalar: F) -> F {
        debug_assert_eq!(coeff.len(), 1 << eval.len());
        if let Some((&x, tail)) = eval.split_first() {
            let (low, high) = coeff.split_at(coeff.len() / 2);
            let a = Self::eval(low, tail, scalar);
            let b = Self::eval(high, tail, scalar * x);
            a + b
        } else {
            scalar * coeff[0]
        }
    }

    #[cfg(not(feature = "parallel"))]
    fn eval_extension<E: Field<BasePrimeField = F>>(coeff: &[F], eval: &[E], scalar: E) -> E {
        debug_assert_eq!(coeff.len(), 1 << eval.len());
        if let Some((&x, tail)) = eval.split_first() {
            let (low, high) = coeff.split_at(coeff.len() / 2);
            let a = Self::eval_extension(low, tail, scalar);
            let b = Self::eval_extension(high, tail, scalar * x);
            a + b
        } else {
            scalar.mul_by_base_prime_field(&coeff[0])
        }
    }

    #[cfg(feature = "parallel")]
    fn eval_extension<E: Field<BasePrimeField = F>>(coeff: &[F], eval: &[E], scalar: E) -> E {
        const PARALLEL_THRESHOLD: usize = 10;
        debug_assert_eq!(coeff.len(), 1 << eval.len());
        if let Some((&x, tail)) = eval.split_first() {
            let (low, high) = coeff.split_at(coeff.len() / 2);
            if tail.len() > PARALLEL_THRESHOLD {
                let (a, b) = rayon::join(
                    || Self::eval_extension(low, tail, scalar),
                    || Self::eval_extension(high, tail, scalar * x),
                );
                a + b
            } else {
                Self::eval_extension(low, tail, scalar)
                    + Self::eval_extension(high, tail, scalar * x)
            }
        } else {
            scalar.mul_by_base_prime_field(&coeff[0])
        }
    }

    pub fn evaluate_at_extension<E: Field<BasePrimeField = F>>(
        &self,
        point: &MultilinearPoint<E>,
    ) -> E {
        assert_eq!(self.num_variables, point.n_variables());
        Self::eval_extension(&self.coeffs, &point.0, E::ONE)
    }

    pub fn evaluate_at_univariate(&self, points: &[F]) -> Vec<F> {
        let univariate = DensePolynomial::from_coefficients_slice(&self.coeffs);
        points
            .iter()
            .map(|point| univariate.evaluate(point))
            .collect()
    }
}

impl<F> CoefficientList<F> {
    pub fn new(coeffs: Vec<F>) -> Self {
        let len = coeffs.len();
        assert!(len.is_power_of_two());
        let num_variables = len.ilog2();

        CoefficientList {
            coeffs,
            num_variables: num_variables as usize,
        }
    }

    pub fn coeffs(&self) -> &[F] {
        &self.coeffs
    }

    pub fn num_variables(&self) -> usize {
        self.num_variables
    }

    pub fn num_coeffs(&self) -> usize {
        self.coeffs.len()
    }

    // Map to the corresponding polynomial in the extension field
    pub fn to_extension<E: Field<BasePrimeField = F>>(self) -> CoefficientList<E> {
        CoefficientList::new(
            self.coeffs
                .into_iter()
                .map(E::from_base_prime_field)
                .collect(),
        )
    }
}

fn eval_multivariate<F: Field>(coeffs: &[F], point: &[F]) -> F {
    debug_assert_eq!(coeffs.len(), 1 << point.len());
    if let Some((x, tail)) = point.split_first() {
        let (low, hi) = coeffs.split_at(coeffs.len() / 2);
        let a = eval_multivariate(low, tail);
        let b = eval_multivariate(hi, tail);
        a + b * x
    } else {
        coeffs[0]
    }
}

impl<F> CoefficientList<F>
where
    F: Field,
{
    pub fn fold(&self, folding_randomness: &MultilinearPoint<F>) -> Self {
        let folding_factor = folding_randomness.n_variables();
        let coeffs = self
            .coeffs
            .chunks_exact(1 << folding_factor)
            .map(|coeffs| eval_multivariate(coeffs, &folding_randomness.0))
            .collect();

        CoefficientList {
            coeffs,
            num_variables: self.num_variables() - folding_factor,
        }
    }
}

impl<F> From<CoefficientList<F>> for DensePolynomial<F>
where
    F: Field,
{
    fn from(value: CoefficientList<F>) -> Self {
        DensePolynomial::from_coefficients_vec(value.coeffs)
    }
}

impl<F> From<DensePolynomial<F>> for CoefficientList<F>
where
    F: Field,
{
    fn from(value: DensePolynomial<F>) -> Self {
        CoefficientList::new(value.coeffs)
    }
}

impl<F> From<CoefficientList<F>> for EvaluationsList<F>
where
    F: Field,
{
    fn from(value: CoefficientList<F>) -> Self {
        let mut evals = value.coeffs;
        let num_coeffs = evals.len();
        let num_variables = value.num_variables;

        for var in 0..num_variables {
            let step = 1 << var;
            for i in (0..num_coeffs).step_by(step * 2) {
                for j in 0..step {
                    if i + j + step < num_coeffs {
                        let sum = evals[i + j] + evals[i + j + step];
                        evals[i + j + step] = sum;
                    }
                }
            }
        }

        EvaluationsList::new(evals)
    }
}

/* Previous recursive version
impl<F> From<CoefficientList<F>> for EvaluationsList<F>
where
    F: Field,
{
    fn from(value: CoefficientList<F>) -> Self {
        let num_coeffs = value.num_coeffs();
        // Base case
        if num_coeffs == 1 {
            return EvaluationsList::new(value.coeffs);
        }

        let half_coeffs = num_coeffs / 2;

        // Left is polynomial with last variable set to 0
        let mut left = Vec::with_capacity(half_coeffs);

        // Right is polynomial with last variable set to 1
        let mut right = Vec::with_capacity(half_coeffs);

        for i in 0..half_coeffs {
            left.push(value.coeffs[2 * i]);
            right.push(value.coeffs[2 * i] + value.coeffs[2 * i + 1]);
        }

        let left_poly = CoefficientList {
            coeffs: left,
            num_variables: value.num_variables - 1,
        };
        let right_poly = CoefficientList {
            coeffs: right,
            num_variables: value.num_variables - 1,
        };

        // Compute evaluation of right and left
        let left_eval = EvaluationsList::from(left_poly);
        let right_eval = EvaluationsList::from(right_poly);

        // Combine
        let mut evaluation_list = Vec::with_capacity(num_coeffs);
        for i in 0..half_coeffs {
            evaluation_list.push(left_eval[i]);
            evaluation_list.push(right_eval[i]);
        }

        EvaluationsList::new(evaluation_list)
    }
}
*/

#[cfg(test)]
mod tests {
    use ark_poly::{univariate::DensePolynomial, Polynomial};

    use crate::{
        crypto::fields::Field64,
        poly_utils::{coeffs::CoefficientList, evals::EvaluationsList, MultilinearPoint},
    };

    type F = Field64;

    #[test]
    fn test_evaluation_conversion() {
        let coeffs = vec![F::from(22), F::from(05), F::from(10), F::from(97)];
        let coeffs_list = CoefficientList::new(coeffs.clone());

        let evaluations = EvaluationsList::from(coeffs_list);

        assert_eq!(evaluations[0], coeffs[0]);
        assert_eq!(evaluations[1], coeffs[0] + coeffs[1]);
        assert_eq!(evaluations[2], coeffs[0] + coeffs[2]);
        assert_eq!(
            evaluations[3],
            coeffs[0] + coeffs[1] + coeffs[2] + coeffs[3]
        );
    }

    #[test]
    fn test_folding() {
        let coeffs = vec![F::from(22), F::from(05), F::from(00), F::from(00)];
        let coeffs_list = CoefficientList::new(coeffs);

        let alpha = F::from(100);
        let beta = F::from(32);

        let folded = coeffs_list.fold(&MultilinearPoint(vec![beta]));

        assert_eq!(
            coeffs_list.evaluate(&MultilinearPoint(vec![alpha, beta])),
            folded.evaluate(&MultilinearPoint(vec![alpha]))
        )
    }

    #[test]
    fn test_folding_and_evaluation() {
        let num_variables = 10;
        let coeffs = (0..(1 << num_variables)).map(F::from).collect();
        let coeffs_list = CoefficientList::new(coeffs);

        let randomness: Vec<_> = (0..num_variables).map(|i| F::from(35 * i as u64)).collect();
        for k in 0..num_variables {
            let fold_part = randomness[0..k].to_vec();
            let eval_part = randomness[k..randomness.len()].to_vec();

            let fold_random = MultilinearPoint(fold_part.clone());
            let eval_point = MultilinearPoint([eval_part.clone(), fold_part].concat());

            let folded = coeffs_list.fold(&fold_random);
            assert_eq!(
                folded.evaluate(&MultilinearPoint(eval_part)),
                coeffs_list.evaluate(&eval_point)
            );
        }
    }

    #[test]
    fn test_evaluation_mv() {
        let polynomial = vec![
            F::from(0),
            F::from(1),
            F::from(2),
            F::from(3),
            F::from(4),
            F::from(5),
            F::from(6),
            F::from(7),
            F::from(8),
            F::from(9),
            F::from(10),
            F::from(11),
            F::from(12),
            F::from(13),
            F::from(14),
            F::from(15),
        ];

        let mv_poly = CoefficientList::new(polynomial);
        let uv_poly: DensePolynomial<_> = mv_poly.clone().into();

        let eval_point = F::from(4999);
        assert_eq!(
            uv_poly.evaluate(&F::from(1)),
            F::from((0..=15).sum::<u32>())
        );
        assert_eq!(
            uv_poly.evaluate(&eval_point),
            mv_poly.evaluate(&MultilinearPoint::expand_from_univariate(eval_point, 4))
        )
    }
}
