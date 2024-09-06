use std::ops::AddAssign;

use super::{evals::EvaluationsList, hypercube::BinaryHypercubePoint, MultilinearPoint};
use ark_ff::Field;
use ark_poly::{univariate::DensePolynomial, DenseUVPolynomial, Polynomial};
#[cfg(feature = "parallel")]
use {
    rayon::{join, prelude::*},
    std::mem::size_of,
};

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
        eval_multivariate(&self.coeffs, &point.0)
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

// Multivariate evaluation in coefficient form.
fn eval_multivariate<F: Field>(coeffs: &[F], point: &[F]) -> F {
    debug_assert_eq!(coeffs.len(), 1 << point.len());
    match point {
        [] => coeffs[0],
        [x] => coeffs[0] + coeffs[1] * x,
        [x0, x1] => {
            let b0 = coeffs[0] + coeffs[1] * x1;
            let b1 = coeffs[2] + coeffs[3] * x1;
            b0 + b1 * x0
        }
        [x0, x1, x2] => {
            let b00 = coeffs[0] + coeffs[1] * x2;
            let b01 = coeffs[2] + coeffs[3] * x2;
            let b10 = coeffs[4] + coeffs[5] * x2;
            let b11 = coeffs[6] + coeffs[7] * x2;
            let b0 = b00 + b01 * x1;
            let b1 = b10 + b11 * x1;
            b0 + b1 * x0
        }
        [x0, x1, x2, x3] => {
            let b000 = coeffs[0] + coeffs[1] * x3;
            let b001 = coeffs[2] + coeffs[3] * x3;
            let b010 = coeffs[4] + coeffs[5] * x3;
            let b011 = coeffs[6] + coeffs[7] * x3;
            let b100 = coeffs[8] + coeffs[9] * x3;
            let b101 = coeffs[10] + coeffs[11] * x3;
            let b110 = coeffs[12] + coeffs[13] * x3;
            let b111 = coeffs[14] + coeffs[15] * x3;
            let b00 = b000 + b001 * x2;
            let b01 = b010 + b011 * x2;
            let b10 = b100 + b101 * x2;
            let b11 = b110 + b111 * x2;
            let b0 = b00 + b01 * x1;
            let b1 = b10 + b11 * x1;
            b0 + b1 * x0
        }
        [x, tail @ ..] => {
            let (b0t, b1t) = coeffs.split_at(coeffs.len() / 2);
            #[cfg(not(feature = "parallel"))]
            let (b0t, b1t) = (eval_multivariate(b0t, tail), eval_multivariate(b1t, tail));
            #[cfg(feature = "parallel")]
            let (b0t, b1t) = {
                let work_size: usize = (1 << 15) / size_of::<F>();
                if coeffs.len() > work_size {
                    join(
                        || eval_multivariate(b0t, tail),
                        || eval_multivariate(b1t, tail),
                    )
                } else {
                    (eval_multivariate(b0t, tail), eval_multivariate(b1t, tail))
                }
            };
            b0t + b1t * x
        }
    }
}

impl<F> CoefficientList<F>
where
    F: Field,
{
    pub fn fold(&self, folding_randomness: &MultilinearPoint<F>) -> Self {
        let folding_factor = folding_randomness.n_variables();
        #[cfg(not(feature = "parallel"))]
        let coeffs = self
            .coeffs
            .chunks_exact(1 << folding_factor)
            .map(|coeffs| eval_multivariate(coeffs, &folding_randomness.0))
            .collect();
        #[cfg(feature = "parallel")]
        let coeffs = self
            .coeffs
            .par_chunks_exact(1 << folding_factor)
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
        wavelet_transform(&mut evals);
        EvaluationsList::new(evals)
    }
}

fn wavelet_transform<F>(values: &mut [F])
where
    F: for<'a> AddAssign<&'a F>,
{
    debug_assert!(values.len().is_power_of_two());
    eprintln!("wavelet_transform {}", values.len().trailing_zeros());
    for r in 0..values.len().trailing_zeros() {
        for coeffs in values.chunks_mut(1 << (r + 1)) {
            let (left, right) = coeffs.split_at_mut(1 << r);
            for (left, right) in left.iter().zip(right.iter_mut()) {
                *right += left;
            }
        }
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
