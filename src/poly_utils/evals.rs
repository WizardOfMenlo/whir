use std::ops::Index;

use ark_ff::Field;
use rayon::prelude::*;

use super::{sequential_lag_poly::LagrangePolynomialIterator, MultilinearPoint};

/// An EvaluationsList models a multi-linear polynomial f in `num_variables`
/// unknowns, stored via their evaluations at {0,1}^{num_variables}
///
/// `evals` stores the evaluation in lexicographic order.
#[derive(Debug, Clone)]
pub struct EvaluationsList<F> {
    evals: Vec<F>,
    num_variables: usize,
}

impl<F> EvaluationsList<F>
where
    F: Field,
{
    /// Constructs a EvaluationList from the given vector `eval` of evaluations.
    ///
    /// The provided `evals` is supposed to be the list of evaluations, where the ordering of evaluation points in {0,1}^n
    /// is lexicographic.
    pub fn new(evals: Vec<F>) -> Self {
        let len = evals.len();
        assert!(len.is_power_of_two());
        let num_variables = len.ilog2();

        EvaluationsList {
            evals,
            num_variables: num_variables as usize,
        }
    }

    /// evaluate the polynomial at `point`
    pub fn evaluate(&self, point: &MultilinearPoint<F>) -> F {
        if let Some(point) = point.to_hypercube() {
            return self.evals[point.0];
        }

        let mut sum = F::ZERO;
        for (b, lag) in LagrangePolynomialIterator::new(point) {
            sum += lag * self.evals[b.0]
        }

        sum
    }
    
    fn eval_multilinear(&self, evals: &[F], point: &[F]) -> F {
        debug_assert_eq!(evals.len(), 1 << point.len());
        let one = F::one();
        match point {
            [] => evals[0],
            [x] => evals[0] * (one - *x) + evals[1] * *x,
            [x0, x1] => {
                let a0 = evals[0] * (one - *x1) + evals[1] * *x1;
                let a1 = evals[2] * (one - *x1) + evals[3] * *x1;
                a0 * (one - *x0) + a1 * *x0
            }
            [x0, x1, x2] => {
                let a00 = evals[0] * (one - *x2) + evals[1] * *x2;
                let a01 = evals[2] * (one - *x2) + evals[3] * *x2;
                let a10 = evals[4] * (one - *x2) + evals[5] * *x2;
                let a11 = evals[6] * (one - *x2) + evals[7] * *x2;
                let a0 = a00 * (one - *x1) + a01 * *x1;
                let a1 = a10 * (one - *x1) + a11 * *x1;
                a0 * (one - *x0) + a1 * *x0
            }
            [x0, x1, x2, x3] => {
                let a000 = evals[0] * (one - *x3) + evals[1] * *x3;
                let a001 = evals[2] * (one - *x3) + evals[3] * *x3;
                let a010 = evals[4] * (one - *x3) + evals[5] * *x3;
                let a011 = evals[6] * (one - *x3) + evals[7] * *x3;
                let a100 = evals[8] * (one - *x3) + evals[9] * *x3;
                let a101 = evals[10] * (one - *x3) + evals[11] * *x3;
                let a110 = evals[12] * (one - *x3) + evals[13] * *x3;
                let a111 = evals[14] * (one - *x3) + evals[15] * *x3;
                let a00 = a000 * (one - *x2) + a001 * *x2;
                let a01 = a010 * (one - *x2) + a011 * *x2;
                let a10 = a100 * (one - *x2) + a101 * *x2;
                let a11 = a110 * (one - *x2) + a111 * *x2;
                let a0 = a00 * (one - *x1) + a01 * *x1;
                let a1 = a10 * (one - *x1) + a11 * *x1;
                a0 * (one - *x0) + a1 * *x0
            }
            [x, tail @ ..] => {
                let (f0, f1) = evals.split_at(evals.len() / 2);
                // let mid = evals.len() / 2;
                 #[cfg(not(feature = "parallel"))]
                let (f0, f1) = (
                    self.eval_multilinear(f0, tail),
                    self.eval_multilinear(f1, tail),
                );
                #[cfg(feature = "parallel")]
                let (f0, f1) = {
                    let work_size: usize = (1 << 15) / std::mem::size_of::<F>();
                    if evals.len() > work_size {
                        rayon::join(
                            || self.eval_multilinear(f0, tail),
                            || self.eval_multilinear(f1, tail),
                        )
                    } else {
                        (
                            self.eval_multilinear(f0, tail),
                            self.eval_multilinear(f1, tail),
                        )
                    }
                };
                f0 * (one - *x) + f1 * *x
            }
        }
    }

    pub fn eval_extension(&self, point: &MultilinearPoint<F>) -> F {
        if let Some(point) = point.to_hypercube() {
            return self.evals[point.0];
        }
        self.eval_multilinear(&self.evals, &point.0)
    }

    pub fn evals(&self) -> &[F] {
        &self.evals
    }

    pub fn evals_mut(&mut self) -> &mut [F] {
        &mut self.evals
    }

    pub fn num_evals(&self) -> usize {
        self.evals.len()
    }

    pub fn num_variables(&self) -> usize {
        self.num_variables
    }
}

impl<F> Index<usize> for EvaluationsList<F> {
    type Output = F;
    fn index(&self, index: usize) -> &Self::Output {
        &self.evals[index]
    }
}

#[cfg(test)]
mod tests {
    use crate::poly_utils::hypercube::BinaryHypercube;

    use super::*;
    use ark_ff::*;

    type F = crate::crypto::fields::Field64;

    #[test]
    fn test_evaluation() {
        let evaluations_vec = vec![F::ZERO, F::ONE, F::ZERO, F::ONE];
        let evals = EvaluationsList::new(evaluations_vec.clone());

        for i in BinaryHypercube::new(2) {
            assert_eq!(
                evaluations_vec[i.0],
                evals.evaluate(&MultilinearPoint::from_binary_hypercube_point(i, 2))
            );
        }
    }
}
