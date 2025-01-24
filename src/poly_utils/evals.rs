use std::ops::Index;

use ark_ff::Field;

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
