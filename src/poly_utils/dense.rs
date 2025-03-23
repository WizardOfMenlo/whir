use ark_ff::Field;

/// A univariate polynomial represented in coefficient form.
///
/// The coefficient of `x^i` is stored at index `i`.
///
/// Designed for verifier use: avoids parallelism by enforcing sequential Horner evaluation.
/// The verifier should be run on a cheap device.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct WhirDensePolynomial<F: Field> {
    /// The coefficient of `x^i` is stored at location `i` in `self.coeffs`.
    pub coeffs: Vec<F>,
}

impl<F: Field> WhirDensePolynomial<F> {
    /// Constructs a new polynomial from a list of coefficients.
    pub(crate) fn from_coefficients_slice(coeffs: &[F]) -> Self {
        Self::from_coefficients_vec(coeffs.to_vec())
    }

    /// Constructs a new polynomial from a list of coefficients.
    pub(crate) fn from_coefficients_vec(coeffs: Vec<F>) -> Self {
        let mut result = Self { coeffs };
        // While there are zeros at the end of the coefficient vector, pop them off.
        result.truncate_leading_zeros();
        // Check that either the coefficients vec is empty or that the last coeff is
        // non-zero.
        assert!(result.coeffs.last().is_none_or(|coeff| !coeff.is_zero()));
        result
    }

    fn truncate_leading_zeros(&mut self) {
        while self.coeffs.last().is_some_and(|c| c.is_zero()) {
            self.coeffs.pop();
        }
    }

    /// Checks if the given polynomial is zero.
    fn is_zero(&self) -> bool {
        self.coeffs.is_empty() || self.coeffs.iter().all(|coeff| coeff.is_zero())
    }

    /// Evaluates `self` at the given `point` in `Self::Point`.
    pub fn evaluate(&self, point: &F) -> F {
        if self.is_zero() {
            return F::ZERO;
        } else if point.is_zero() {
            return self.coeffs[0];
        }
        self.horner_evaluate(point)
    }

    // Horner's method for polynomial evaluation
    fn horner_evaluate(&self, point: &F) -> F {
        self.coeffs.iter().rfold(F::zero(), move |result, coeff| result * point + coeff)
    }
}

#[cfg(test)]
mod tests {
    use ark_ff::{AdditiveGroup, Zero};

    use super::*;
    use crate::crypto::fields::Field64;

    #[test]
    fn test_zero_polynomial() {
        // A zero polynomial has no coefficients
        let poly = WhirDensePolynomial::<Field64>::from_coefficients_vec(vec![]);
        assert!(poly.is_zero());
        assert_eq!(poly.evaluate(&Field64::from(42)), Field64::zero());
    }

    #[test]
    fn test_constant_polynomial() {
        // Polynomial: f(x) = 7
        let c0 = Field64::from(7);
        let poly = WhirDensePolynomial::from_coefficients_vec(vec![c0]);

        // f(0)
        assert_eq!(poly.evaluate(&Field64::zero()), c0);
        // f(1)
        assert_eq!(poly.evaluate(&Field64::ONE), c0);
        // f(42)
        assert_eq!(poly.evaluate(&Field64::from(42)), c0);
    }

    #[test]
    fn test_linear_polynomial() {
        // Polynomial: f(x) = 3 + 4x
        let c0 = Field64::from(3);
        let c1 = Field64::from(4);
        let poly = WhirDensePolynomial::from_coefficients_vec(vec![c0, c1]);

        // f(0)
        assert_eq!(poly.evaluate(&Field64::zero()), c0);
        // f(1)
        assert_eq!(poly.evaluate(&Field64::ONE), c0 + c1 * Field64::ONE);
        // f(2)
        assert_eq!(poly.evaluate(&Field64::from(2)), c0 + c1 * Field64::from(2));
    }

    #[test]
    fn test_quadratic_polynomial() {
        // Polynomial: f(x) = 2 + 0x + 5x²
        let c0 = Field64::from(2);
        let c1 = Field64::from(0);
        let c2 = Field64::from(5);
        let poly = WhirDensePolynomial::from_coefficients_vec(vec![c0, c1, c2]);

        // f(0)
        assert_eq!(poly.evaluate(&Field64::zero()), c0);
        // f(1)
        assert_eq!(poly.evaluate(&Field64::ONE), c0 + c2);
        // f(2)
        assert_eq!(poly.evaluate(&Field64::from(2)), c0 + c2 * Field64::from(4));
    }

    #[test]
    fn test_cubic_polynomial() {
        // Polynomial: f(x) = 1 + 2x + 3x² + 4x³
        let c0 = Field64::from(1);
        let c1 = Field64::from(2);
        let c2 = Field64::from(3);
        let c3 = Field64::from(4);
        let poly = WhirDensePolynomial::from_coefficients_vec(vec![c0, c1, c2, c3]);

        // f(0)
        assert_eq!(poly.evaluate(&Field64::zero()), c0);
        // f(1)
        assert_eq!(poly.evaluate(&Field64::ONE), c0 + c1 + c2 + c3);

        // f(2)
        assert_eq!(
            poly.evaluate(&Field64::from(2)),
            c0 + c1 * Field64::from(2) + c2 * Field64::from(4) + c3 * Field64::from(8)
        );
    }

    #[test]
    fn test_leading_zeros_trimmed() {
        // Polynomial: f(x) = 1 + 2x, with trailing zeroes
        let c0 = Field64::from(1);
        let c1 = Field64::from(2);
        let poly =
            WhirDensePolynomial::from_coefficients_vec(vec![c0, c1, Field64::ZERO, Field64::ZERO]);

        // Should be trimmed to degree 1
        assert_eq!(poly.coeffs.len(), 2);
        assert_eq!(poly.evaluate(&Field64::from(3)), c0 + c1 * Field64::from(3));
    }

    #[test]
    fn test_is_zero_various_cases() {
        let zero_poly = WhirDensePolynomial::<Field64>::from_coefficients_vec(vec![]);
        assert!(zero_poly.is_zero());

        let zero_poly_all_zeros =
            WhirDensePolynomial::<Field64>::from_coefficients_vec(vec![Field64::ZERO; 5]);
        assert!(zero_poly_all_zeros.is_zero());

        let non_zero_poly =
            WhirDensePolynomial::<Field64>::from_coefficients_vec(vec![Field64::ONE]);
        assert!(!non_zero_poly.is_zero());
    }
}
