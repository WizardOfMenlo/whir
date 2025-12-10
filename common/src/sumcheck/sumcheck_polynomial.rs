use ark_ff::Field;

use crate::poly_utils::multilinear::MultilinearPoint;

/// Represents a polynomial stored in evaluation form over a ternary domain {0,1,2}^n.
///
/// This structure is uniquely determined by its evaluations over the ternary hypercube.
/// The order of storage follows big-endian lexicographic ordering with respect to the
/// evaluation points.
///
/// Given `n_variables`, the number of stored evaluations is `3^n_variables`.
#[derive(Debug, Clone)]
pub struct SumcheckPolynomial<F> {
    /// Number of variables in the polynomial (defines the dimension of the evaluation domain).
    num_variables: usize,
    /// Vector of function evaluations at points in `{0,1,2}^n_variables`, stored in lexicographic
    /// order.
    evaluations: Vec<F>,
}

impl<F> SumcheckPolynomial<F>
where
    F: Field,
{
    /// Creates a new sumcheck polynomial with `n_variables` variables.
    ///
    /// # Parameters:
    /// - `evaluations`: A vector of function values evaluated on `{0,1,2}^n_variables`.
    /// - `n_variables`: The number of variables (determines the evaluation domain size).
    ///
    /// The vector `evaluations` **must** have a length of `3^n_variables`.
    pub const fn new(evaluations: Vec<F>, num_variables: usize) -> Self {
        Self {
            num_variables,
            evaluations,
        }
    }

    /// Returns the vector of stored evaluations.
    ///
    /// The order follows lexicographic ordering of the ternary hypercube `{0,1,2}^n_variables`:
    ///
    /// ```ignore
    /// evaluations[i] = f(x_1, x_2, ..., x_n)  where (x_1, ..., x_n) ∈ {0,1,2}^n
    /// ```
    #[allow(clippy::missing_const_for_fn)]
    pub fn evaluations(&self) -> &[F] {
        &self.evaluations
    }

    /// Computes the sum of function values over the Boolean hypercube `{0,1}^n_variables`.
    ///
    /// Instead of summing over all `3^n` evaluations, this method only sums over points where all
    /// coordinates are 0 or 1.
    ///
    /// Mathematically, this computes:
    /// ```ignore
    /// sum = ∑ h(x_1, ..., x_n)  where  (x_1, ..., x_n) ∈ {0,1}^n
    /// ```
    pub fn sum_over_boolean_hypercube(&self) -> F {
        (0..(1 << self.num_variables))
            .map(|point| self.evaluations[self.binary_to_ternary_index(point)])
            .sum()
    }

    /// Converts a binary index `(0..2^n)` to its corresponding ternary index `(0..3^n)`.
    ///
    /// This maps a Boolean hypercube `{0,1}^n` to the ternary hypercube `{0,1,2}^n`.
    ///
    /// Given a binary index:
    /// ```ignore
    /// binary_index = b_{n-1} b_{n-2} ... b_0  (in bits)
    /// ```
    /// The corresponding **ternary index** is computed as:
    /// ```ignore
    /// ternary_index = b_0 * 3^0 + b_1 * 3^1 + ... + b_{n-1} * 3^{n-1}
    /// ```
    ///
    /// # Example:
    /// ```ignore
    /// binary index 0b11  (3 in decimal)  →  ternary index 4
    /// binary index 0b10  (2 in decimal)  →  ternary index 3
    /// binary index 0b01  (1 in decimal)  →  ternary index 1
    /// binary index 0b00  (0 in decimal)  →  ternary index 0
    /// ```
    fn binary_to_ternary_index(&self, mut binary_index: usize) -> usize {
        let mut ternary_index = 0;
        let mut factor = 1;

        for _ in 0..self.num_variables {
            ternary_index += (binary_index & 1) * factor;
            // Move to next bit
            binary_index >>= 1;
            // Increase ternary place value
            factor *= 3;
        }

        ternary_index
    }

    /// Evaluates the polynomial at an arbitrary point in the domain `{0,1,2}^n`.
    ///
    /// Given an interpolation point `point ∈ F^n`, this computes:
    /// ```ignore
    /// f(point) = ∑ evaluations[i] * eq_poly3(i)
    /// ```
    /// where `eq_poly3(i)` is the Lagrange basis polynomial at index `i` in `{0,1,2}^n`.
    ///
    /// This allows evaluating the polynomial at non-discrete inputs beyond `{0,1,2}^n`.
    ///
    /// # Constraints:
    /// - The input `point` must have `n_variables` dimensions.
    pub fn evaluate_at_point(&self, point: &MultilinearPoint<F>) -> F {
        assert_eq!(point.num_variables(), self.num_variables);
        self.evaluations
            .iter()
            .enumerate()
            .map(|(i, &eval)| eval * point.eq_poly3(i))
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use ark_ff::AdditiveGroup;

    use super::*;
    use crate::crypto::fields::Field64;

    #[test]
    fn test_binary_to_ternary_index() {
        let poly = SumcheckPolynomial::new(vec![Field64::ZERO; 9], 2);

        // Binary indices: 0, 1, 2, 3 (for 2 variables: {00, 01, 10, 11})
        // Corresponding ternary indices: 0, 1, 3, 4
        assert_eq!(poly.binary_to_ternary_index(0b00), 0);
        assert_eq!(poly.binary_to_ternary_index(0b01), 1);
        assert_eq!(poly.binary_to_ternary_index(0b10), 3);
        assert_eq!(poly.binary_to_ternary_index(0b11), 4);
    }

    #[test]
    fn test_binary_to_ternary_index_three_vars() {
        let poly = SumcheckPolynomial::new(vec![Field64::ZERO; 27], 3);

        // Check conversion for all binary points in {0,1}^3
        assert_eq!(poly.binary_to_ternary_index(0b000), 0);
        assert_eq!(poly.binary_to_ternary_index(0b001), 1);
        assert_eq!(poly.binary_to_ternary_index(0b010), 3);
        assert_eq!(poly.binary_to_ternary_index(0b011), 4);
        assert_eq!(poly.binary_to_ternary_index(0b100), 9);
        assert_eq!(poly.binary_to_ternary_index(0b101), 10);
        assert_eq!(poly.binary_to_ternary_index(0b110), 12);
        assert_eq!(poly.binary_to_ternary_index(0b111), 13);
    }

    #[test]
    fn test_sum_over_boolean_hypercube_single_var() {
        // Test case for a single variable (n_variables = 1)
        // Function values at {0,1,2}: f(0) = 3, f(1) = 5, f(2) = 7
        let evaluations = vec![
            Field64::from(3), // f(0)
            Field64::from(5), // f(1)
            Field64::from(7), // f(2)
        ];
        let poly = SumcheckPolynomial::new(evaluations, 1);

        // Sum over {0,1}: f(0) + f(1)
        let expected_sum = Field64::from(3) + Field64::from(5);
        assert_eq!(poly.sum_over_boolean_hypercube(), expected_sum);
    }

    #[test]
    fn test_sum_over_boolean_hypercube() {
        // Define a simple function f such that:
        // f(0,0) = 1, f(0,1) = 2, f(0,2) = 3
        // f(1,0) = 4, f(1,1) = 5, f(1,2) = 6
        // f(2,0) = 7, f(2,1) = 8, f(2,2) = 9
        let evaluations: Vec<_> = (1..=9).map(Field64::from).collect();
        let poly = SumcheckPolynomial::new(evaluations, 2);

        // Sum over {0,1}^2: f(0,0) + f(0,1) + f(1,0) + f(1,1)
        let expected_sum =
            Field64::from(1) + Field64::from(2) + Field64::from(4) + Field64::from(5);
        let computed_sum = poly.sum_over_boolean_hypercube();
        assert_eq!(computed_sum, expected_sum);
    }

    #[test]
    fn test_sum_over_boolean_hypercube_three_vars() {
        // Test case for three variables (n_variables = 3)
        // Evaluations indexed lexicographically in {0,1,2}^3:
        //
        // f(0,0,0) = 1  f(0,0,1) = 2  f(0,0,2) = 3
        // f(0,1,0) = 4  f(0,1,1) = 5  f(0,1,2) = 6
        // f(0,2,0) = 7  f(0,2,1) = 8  f(0,2,2) = 9
        //
        // f(1,0,0) = 10 f(1,0,1) = 11 f(1,0,2) = 12
        // f(1,1,0) = 13 f(1,1,1) = 14 f(1,1,2) = 15
        // f(1,2,0) = 16 f(1,2,1) = 17 f(1,2,2) = 18
        //
        // f(2,0,0) = 19 f(2,0,1) = 20 f(2,0,2) = 21
        // f(2,1,0) = 22 f(2,1,1) = 23 f(2,1,2) = 24
        // f(2,2,0) = 25 f(2,2,1) = 26 f(2,2,2) = 27
        let evaluations: Vec<_> = (1..=27).map(Field64::from).collect();
        let poly = SumcheckPolynomial::new(evaluations, 3);

        // Sum over {0,1}^3
        let expected_sum = Field64::from(1)
            + Field64::from(2)
            + Field64::from(4)
            + Field64::from(5)
            + Field64::from(10)
            + Field64::from(11)
            + Field64::from(13)
            + Field64::from(14);

        assert_eq!(poly.sum_over_boolean_hypercube(), expected_sum);
    }

    #[test]
    fn test_evaluate_at_point() {
        // Define a function f where evaluations are hardcoded:
        // f(0,0) = 1, f(0,1) = 2, f(0,2) = 3
        // f(1,0) = 4, f(1,1) = 5, f(1,2) = 6
        // f(2,0) = 7, f(2,1) = 8, f(2,2) = 9
        let evaluations: Vec<_> = (1..=9).map(Field64::from).collect();
        let poly = SumcheckPolynomial::new(evaluations, 2);

        // Define an evaluation point (0.5, 0.5) as an interpolation between {0,1,2}^2
        let point = MultilinearPoint(vec![Field64::from(1) / Field64::from(2); 2]);

        let result = poly.evaluate_at_point(&point);

        // Compute the expected result using the full weighted sum:
        let expected_value = Field64::from(1) * point.eq_poly3(0)
            + Field64::from(2) * point.eq_poly3(1)
            + Field64::from(3) * point.eq_poly3(2)
            + Field64::from(4) * point.eq_poly3(3)
            + Field64::from(5) * point.eq_poly3(4)
            + Field64::from(6) * point.eq_poly3(5)
            + Field64::from(7) * point.eq_poly3(6)
            + Field64::from(8) * point.eq_poly3(7)
            + Field64::from(9) * point.eq_poly3(8);

        assert_eq!(result, expected_value);
    }

    #[test]
    fn test_evaluate_at_point_three_vars() {
        // Define a function with three variables
        let evaluations: Vec<_> = (1..=27).map(Field64::from).collect();
        let poly = SumcheckPolynomial::new(evaluations, 3);

        // Define an interpolation point (1/2, 1/2, 1/2) in {0,1,2}^3
        let point = MultilinearPoint(vec![Field64::from(1) / Field64::from(2); 3]);

        // Compute expected evaluation:
        let expected_value = (0..27)
            .map(|i| poly.evaluations[i] * point.eq_poly3(i))
            .sum::<Field64>();

        let computed_value = poly.evaluate_at_point(&point);
        assert_eq!(computed_value, expected_value);
    }
}
