use ark_ff::Field;

use crate::{
    poly_utils::{eq_poly3, MultilinearPoint},
    utils::base_decomposition,
};

// Stored in evaluation form
#[derive(Debug, Clone)]
pub struct SumcheckPolynomial<F> {
    n_variables: usize, // number of variables;
    // evaluations has length 3^{n_variables}
    // The order in which it is stored is such that evaluations[i]
    // corresponds to the evaluation at utils::base_decomposition(i, 3, n_variables),
    // which performs (big-endian) ternary decomposition.
    // (in other words, the ordering is lexicographic wrt the evaluation point)
    evaluations: Vec<F>, // Each of our polynomials will be in F^{<3}[X_1, \dots, X_k],
                         // so it us uniquely determined by it's evaluations over {0, 1, 2}^k
}

impl<F> SumcheckPolynomial<F>
where
    F: Field,
{
    pub const fn new(evaluations: Vec<F>, n_variables: usize) -> Self {
        Self {
            evaluations,
            n_variables,
        }
    }

    /// Returns the vector of evaluations at {0,1,2}^n_variables of the polynomial f
    /// in the following order: [f(0,0,..,0), f(0,0,..,1), f(0,0,...,2), f(0,0,...,1,0), ...]
    /// (i.e. lexicographic wrt. to the evaluation points.
    pub fn evaluations(&self) -> &[F] {
        &self.evaluations
    }

    // TODO(Gotti): Rename to sum_over_binary_hypercube for clarity?
    // TODO(Gotti): Make more efficient; the base_decomposition and filtering is unneccessary.

    /// Returns the sum of evaluations of f, when summed only over {0,1}^n_variables
    ///
    /// (and not over {0,1,2}^n_variable)
    pub fn sum_over_hypercube(&self) -> F {
        let num_evaluation_points = 3_usize.pow(self.n_variables as u32);

        let mut sum = F::ZERO;
        for point in 0..num_evaluation_points {
            if base_decomposition(point, 3, self.n_variables)
                .into_iter()
                .all(|v| matches!(v, 0 | 1))
            {
                sum += self.evaluations[point];
            }
        }

        sum
    }

    /// Returns the sum of evaluations of f, when summed only over {0,1}^n_variables
    /// Avoids enumerating 3^n, instead only iterates 2^n
    pub fn sum_over_binary_hypercube(&self) -> F {
        let binary_points = 2_usize.pow(self.n_variables as u32);
        let mut sum = F::ZERO;

        for point in 0..binary_points {
            let ternary_index = self.binary_to_ternary_index(point);
            sum += self.evaluations[ternary_index];
        }
        sum
    }

    /// Converts a binary index (0..2^n) to its corresponding ternary index (0..3^n).
    fn binary_to_ternary_index(&self, binary_index: usize) -> usize {
        let mut ternary_index = 0;
        let mut factor = 3_usize.pow((self.n_variables - 1) as u32);

        // Read bits from the most significant to the least, assigning them to descending powers of 3
        for i in 0..self.n_variables {
            let shift = self.n_variables - 1 - i;
            let bit = (binary_index >> shift) & 1;
            ternary_index += bit * factor;
            if i < self.n_variables - 1 {
                factor /= 3;
            }
        }

        ternary_index
    }

    /// evaluates the polynomial at an arbitrary point, not neccessarily in {0,1,2}^n_variables.
    ///
    /// We assert that point.n_variables() == self.n_variables
    pub fn evaluate_at_point(&self, point: &MultilinearPoint<F>) -> F {
        assert!(point.n_variables() == self.n_variables);
        let num_evaluation_points = 3_usize.pow(self.n_variables as u32);

        let mut evaluation = F::ZERO;

        for index in 0..num_evaluation_points {
            evaluation += self.evaluations[index] * eq_poly3(point, index);
        }

        evaluation
    }
}

#[cfg(test)]
mod tests {
    use crate::{crypto::fields::Field64, poly_utils::MultilinearPoint, utils::base_decomposition};

    use super::SumcheckPolynomial;
    use std::time::Instant;

    type F = Field64;

    #[test]
    fn test_evaluation() {
        let num_variables = 2;

        let num_evaluation_points = 3_usize.pow(num_variables as u32);
        let evaluations = (0..num_evaluation_points as u64).map(F::from).collect();

        let poly = SumcheckPolynomial::new(evaluations, num_variables);

        for i in 0..num_evaluation_points {
            let decomp = base_decomposition(i, 3, num_variables);
            let point = MultilinearPoint(decomp.into_iter().map(F::from).collect());
            assert_eq!(poly.evaluate_at_point(&point), poly.evaluations()[i]);
        }
    }

    #[test]
    fn test_sum_over_hypercube_correctness_and_bench() {
        let n = 6;
        let evaluations: Vec<F> = (0..(3_usize.pow(n as u32)) as u64).map(F::from).collect();
        let poly = SumcheckPolynomial::new(evaluations, n);

        let sum_orig = poly.sum_over_hypercube();
        let sum_improved = poly.sum_over_binary_hypercube();
        assert_eq!(sum_orig, sum_improved);

        let loop_count = 10;

        let start = Instant::now();
        for _ in 0..loop_count {
            let _ = poly.sum_over_hypercube();
        }
        let dur_orig = start.elapsed();

        let start = Instant::now();
        for _ in 0..loop_count {
            let _ = poly.sum_over_binary_hypercube();
        }
        let dur_improved = start.elapsed();

        println!("  sum_over_hypercube (original) total time: {:?}", dur_orig);
        println!("  sum_over_hypercube_improved     total time: {:?}", dur_improved);
    }
}
