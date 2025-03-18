use super::hypercube::BinaryHypercubePoint;
use ark_ff::Field;
use rand::Rng;
use rand::{distributions::Standard, prelude::Distribution, RngCore};

/// A point `(x_1, ..., x_n)` in `F^n` for some field `F`.
///
/// Often, `x_i` are binary. If strictly binary, `BinaryHypercubePoint` is used.
#[derive(Default, Debug, Clone, PartialEq, Eq)]
pub struct MultilinearPoint<F>(pub Vec<F>);

impl<F> MultilinearPoint<F>
where
    F: Field,
{
    /// Returns the number of variables (dimension `n`).
    #[inline]
    pub fn num_variables(&self) -> usize {
        self.0.len()
    }

    /// Converts a `BinaryHypercubePoint` (bit representation) into a `MultilinearPoint`.
    ///
    /// This maps each bit in the binary integer to `F::ONE` (1) or `F::ZERO` (0) in the field.
    ///
    /// Given `point = b_{n-1} ... b_1 b_0` (big-endian), it produces:
    /// ```ignore
    /// [b_{n-1}, b_{n-2}, ..., b_1, b_0]
    /// ```
    pub fn from_binary_hypercube_point(point: BinaryHypercubePoint, num_variables: usize) -> Self {
        Self(
            (0..num_variables)
                .rev()
                .map(|i| {
                    if (point.0 >> i) & 1 == 1 {
                        F::ONE
                    } else {
                        F::ZERO
                    }
                })
                .collect(),
        )
    }

    /// Converts `MultilinearPoint` to a `BinaryHypercubePoint`, assuming values are binary.
    ///
    /// The point is interpreted as a binary number:
    /// ```ignore
    /// b_{n-1} * 2^{n-1} + b_{n-2} * 2^{n-2} + ... + b_1 * 2^1 + b_0 * 2^0
    /// ```
    /// Returns `None` if any coordinate is non-binary.
    pub fn to_hypercube(&self) -> Option<BinaryHypercubePoint> {
        self.0
            .iter()
            .try_fold(0, |acc, &coord| {
                if coord == F::ZERO {
                    Some(acc << 1)
                } else if coord == F::ONE {
                    Some((acc << 1) | 1)
                } else {
                    None
                }
            })
            .map(BinaryHypercubePoint)
    }

    /// Converts a univariate evaluation point into a multilinear one.
    ///
    /// Uses the bijection:
    /// ```ignore
    /// f(x_1, ..., x_n) <-> g(y) := f(y^(2^(n-1)), ..., y^4, y^2, y)
    /// ```
    /// Meaning:
    /// ```ignore
    /// x_1^i_1 * ... * x_n^i_n <-> y^i
    /// ```
    /// where `(i_1, ..., i_n)` is the **big-endian** binary decomposition of `i`.
    ///
    /// Reversing the order ensures the **big-endian** convention.
    pub fn expand_from_univariate(point: F, num_variables: usize) -> Self {
        let mut res = Vec::with_capacity(num_variables);
        let mut cur = point;

        for _ in 0..num_variables {
            res.push(cur);
            cur = cur * cur; // Compute y^(2^k) at each step
        }

        res.reverse();
        Self(res)
    }

    /// Computes the equality polynomial `eq(c, p)`, where `p` is binary.
    ///
    /// The **equality polynomial** is defined as:
    /// ```ignore
    /// eq(c, p) = ∏ (c_i * p_i + (1 - c_i) * (1 - p_i))
    /// ```
    /// which evaluates to `1` if `c == p`, and `0` otherwise.
    ///
    /// `p` is interpreted as a **big-endian** binary number.
    pub fn eq_poly(&self, mut point: BinaryHypercubePoint) -> F {
        let n_variables = self.num_variables();
        assert!(*point < (1 << n_variables)); // Ensure correct length

        let mut acc = F::ONE;

        for val in self.0.iter().rev() {
            let b = *point % 2;
            acc *= if b == 1 { *val } else { F::ONE - *val };
            *point >>= 1;
        }

        acc
    }

    /// Computes `eq(c, p)`, where `p` is a general `MultilinearPoint` (not necessarily binary).
    ///
    /// The **equality polynomial** for two `MultilinearPoint`s `c` and `p` is:
    /// ```ignore
    /// eq(c, p) = ∏ (c_i * p_i + (1 - c_i) * (1 - p_i))
    /// ```
    /// which evaluates to `1` if `c == p`, and `0` otherwise.
    pub fn eq_poly_outside(&self, point: &Self) -> F {
        assert_eq!(self.num_variables(), point.num_variables());
        self.0.iter().zip(&point.0).fold(F::ONE, |acc, (&l, &r)| {
            acc * (l * r + (F::ONE - l) * (F::ONE - r))
        })
    }

    /// Computes `eq3(c, p)`, the **equality polynomial** for `{0,1,2}^n`.
    ///
    /// `p` is interpreted as a **big-endian** ternary number.
    ///
    /// `eq3(c, p)` is the unique polynomial of **degree ≤ 2** in each variable,
    /// such that:
    /// ```ignore
    /// eq3(c, p) = 1  if c == p
    ///           = 0  otherwise
    /// ```
    /// Uses precomputed values to reduce redundant operations.
    pub fn eq_poly3(&self, mut point: usize) -> F {
        let two = F::ONE + F::ONE;
        let two_inv = two.inverse().unwrap();

        let n_variables = self.num_variables();
        assert!(point < 3usize.pow(n_variables as u32));

        let mut acc = F::ONE;

        // Iterate in **little-endian** order and adjust using big-endian convention.
        for &val in self.0.iter().rev() {
            let val_minus_one = val - F::ONE;
            let val_minus_two = val - two;

            acc *= match point % 3 {
                0 => val_minus_one * val_minus_two * two_inv, // (val - 1)(val - 2) / 2
                1 => val * val_minus_two * (-F::ONE),         // val (val - 2)(-1)
                2 => val * val_minus_one * two_inv,           // val (val - 1) / 2
                _ => unreachable!(),
            };
            point /= 3;
        }

        acc
    }
}

impl<F> MultilinearPoint<F>
where
    Standard: Distribution<F>,
{
    pub fn rand(rng: &mut impl RngCore, num_variables: usize) -> Self {
        Self((0..num_variables).map(|_| rng.gen()).collect())
    }
}

impl<F> From<F> for MultilinearPoint<F> {
    fn from(value: F) -> Self {
        Self(vec![value])
    }
}

#[cfg(test)]
#[allow(
    clippy::identity_op,
    clippy::cast_sign_loss,
    clippy::erasing_op,
    clippy::should_panic_without_expect
)]
mod tests {
    use ark_ff::AdditiveGroup;

    use crate::crypto::fields::Field64;

    use super::*;

    #[test]
    fn test_n_variables() {
        let point =
            MultilinearPoint::<Field64>(vec![Field64::from(1), Field64::from(0), Field64::from(1)]);
        assert_eq!(point.num_variables(), 3);
    }

    #[test]
    fn test_from_binary_hypercube_point_all_zeros() {
        let num_variables = 5;
        // Represents (0,0,0,0,0)
        let binary_point = BinaryHypercubePoint(0);
        let ml_point =
            MultilinearPoint::<Field64>::from_binary_hypercube_point(binary_point, num_variables);

        let expected = vec![Field64::ZERO; num_variables];
        assert_eq!(ml_point.0, expected);
    }

    #[test]
    fn test_from_binary_hypercube_point_all_ones() {
        let num_variables = 4;
        // Represents (1,1,1,1)
        let binary_point = BinaryHypercubePoint((1 << num_variables) - 1);
        let ml_point =
            MultilinearPoint::<Field64>::from_binary_hypercube_point(binary_point, num_variables);

        let expected = vec![Field64::ONE; num_variables];
        assert_eq!(ml_point.0, expected);
    }

    #[test]
    fn test_from_binary_hypercube_point_mixed_bits() {
        let num_variables = 6;
        // Represents (1,0,1,0,1,0)
        let binary_point = BinaryHypercubePoint(0b10_1010);
        let ml_point =
            MultilinearPoint::<Field64>::from_binary_hypercube_point(binary_point, num_variables);

        let expected = vec![
            Field64::ONE,
            Field64::ZERO,
            Field64::ONE,
            Field64::ZERO,
            Field64::ONE,
            Field64::ZERO,
        ];
        assert_eq!(ml_point.0, expected);
    }

    #[test]
    fn test_from_binary_hypercube_point_truncation() {
        let num_variables = 3;
        // Should only use last 3 bits (101)
        let binary_point = BinaryHypercubePoint(0b10101);
        let ml_point =
            MultilinearPoint::<Field64>::from_binary_hypercube_point(binary_point, num_variables);

        let expected = vec![Field64::ONE, Field64::ZERO, Field64::ONE];
        assert_eq!(ml_point.0, expected);
    }

    #[test]
    fn test_from_binary_hypercube_point_expansion() {
        let num_variables = 8;
        // Represents (0,0,0,0,1,0,1,0)
        let binary_point = BinaryHypercubePoint(0b1010);
        let ml_point =
            MultilinearPoint::<Field64>::from_binary_hypercube_point(binary_point, num_variables);

        let expected = vec![
            Field64::ZERO,
            Field64::ZERO,
            Field64::ZERO,
            Field64::ZERO,
            Field64::ONE,
            Field64::ZERO,
            Field64::ONE,
            Field64::ZERO,
        ];
        assert_eq!(ml_point.0, expected);
    }

    #[test]
    fn test_to_hypercube_all_zeros() {
        let point = MultilinearPoint(vec![
            Field64::ZERO,
            Field64::ZERO,
            Field64::ZERO,
            Field64::ZERO,
        ]);
        assert_eq!(point.to_hypercube(), Some(BinaryHypercubePoint(0)));
    }

    #[test]
    fn test_to_hypercube_all_ones() {
        let point = MultilinearPoint(vec![Field64::ONE, Field64::ONE, Field64::ONE, Field64::ONE]);
        assert_eq!(point.to_hypercube(), Some(BinaryHypercubePoint(0b1111)));
    }

    #[test]
    fn test_to_hypercube_mixed_bits() {
        let point = MultilinearPoint(vec![
            Field64::ONE,
            Field64::ZERO,
            Field64::ONE,
            Field64::ZERO,
        ]);
        assert_eq!(point.to_hypercube(), Some(BinaryHypercubePoint(0b1010)));
    }

    #[test]
    fn test_to_hypercube_single_bit() {
        let point = MultilinearPoint(vec![Field64::ONE]);
        assert_eq!(point.to_hypercube(), Some(BinaryHypercubePoint(1)));

        let point = MultilinearPoint(vec![Field64::ZERO]);
        assert_eq!(point.to_hypercube(), Some(BinaryHypercubePoint(0)));
    }

    #[test]
    fn test_to_hypercube_large_binary_number() {
        let point = MultilinearPoint(vec![
            Field64::ONE,
            Field64::ONE,
            Field64::ZERO,
            Field64::ONE,
            Field64::ZERO,
            Field64::ONE,
            Field64::ONE,
            Field64::ZERO,
        ]);
        assert_eq!(
            point.to_hypercube(),
            Some(BinaryHypercubePoint(0b1101_0110))
        );
    }

    #[test]
    fn test_to_hypercube_non_binary_values() {
        let invalid_value = Field64::from(2);
        let point = MultilinearPoint(vec![Field64::ONE, invalid_value, Field64::ZERO]);
        assert_eq!(point.to_hypercube(), None);
    }

    #[test]
    fn test_to_hypercube_empty_vector() {
        let point = MultilinearPoint::<Field64>(vec![]);
        assert_eq!(point.to_hypercube(), Some(BinaryHypercubePoint(0)));
    }

    #[test]
    fn test_expand_from_univariate_single_variable() {
        let point = Field64::from(3);
        let expanded = MultilinearPoint::expand_from_univariate(point, 1);

        // For n = 1, we expect [y]
        assert_eq!(expanded.0, vec![point]);
    }

    #[test]
    fn test_expand_from_univariate_two_variables() {
        let point = Field64::from(2);
        let expanded = MultilinearPoint::expand_from_univariate(point, 2);

        // For n = 2, we expect [y^2, y]
        let expected = vec![point * point, point];
        assert_eq!(expanded.0, expected);
    }

    #[test]
    fn test_expand_from_univariate_three_variables() {
        let point = Field64::from(5);
        let expanded = MultilinearPoint::expand_from_univariate(point, 3);

        // For n = 3, we expect [y^4, y^2, y]
        let expected = vec![point.pow([4]), point.pow([2]), point];
        assert_eq!(expanded.0, expected);
    }

    #[test]
    fn test_expand_from_univariate_large_variables() {
        let point = Field64::from(7);
        let expanded = MultilinearPoint::expand_from_univariate(point, 5);

        // For n = 5, we expect [y^16, y^8, y^4, y^2, y]
        let expected = vec![
            point.pow([16]),
            point.pow([8]),
            point.pow([4]),
            point.pow([2]),
            point,
        ];
        assert_eq!(expanded.0, expected);
    }

    #[test]
    fn test_expand_from_univariate_identity() {
        let point = Field64::ONE;
        let expanded = MultilinearPoint::expand_from_univariate(point, 4);

        // Since 1^k = 1 for all k, the result should be [1, 1, 1, 1]
        let expected = vec![Field64::ONE; 4];
        assert_eq!(expanded.0, expected);
    }

    #[test]
    fn test_expand_from_univariate_zero() {
        let point = Field64::ZERO;
        let expanded = MultilinearPoint::expand_from_univariate(point, 4);

        // Since 0^k = 0 for all k, the result should be [0, 0, 0, 0]
        let expected = vec![Field64::ZERO; 4];
        assert_eq!(expanded.0, expected);
    }

    #[test]
    fn test_expand_from_univariate_empty() {
        let point = Field64::from(9);
        let expanded = MultilinearPoint::expand_from_univariate(point, 0);

        // No variables should return an empty vector
        assert_eq!(expanded.0, vec![]);
    }

    #[test]
    fn test_expand_from_univariate_powers_correctness() {
        let point = Field64::from(3);
        let expanded = MultilinearPoint::expand_from_univariate(point, 6);

        // For n = 6, we expect [y^32, y^16, y^8, y^4, y^2, y]
        let expected = vec![
            point.pow([32]),
            point.pow([16]),
            point.pow([8]),
            point.pow([4]),
            point.pow([2]),
            point,
        ];
        assert_eq!(expanded.0, expected);
    }

    #[test]
    fn test_eq_poly_all_zeros() {
        // Multilinear point (0,0,0,0)
        let ml_point = MultilinearPoint(vec![Field64::ZERO; 4]);
        let binary_point = BinaryHypercubePoint(0b0000);

        // eq_poly should evaluate to 1 since c_i = p_i = 0
        assert_eq!(ml_point.eq_poly(binary_point), Field64::ONE);
    }

    #[test]
    fn test_eq_poly_all_ones() {
        // Multilinear point (1,1,1,1)
        let ml_point = MultilinearPoint(vec![Field64::ONE; 4]);
        let binary_point = BinaryHypercubePoint(0b1111);

        // eq_poly should evaluate to 1 since c_i = p_i = 1
        assert_eq!(ml_point.eq_poly(binary_point), Field64::ONE);
    }

    #[test]
    fn test_eq_poly_mixed_bits_match() {
        // Multilinear point (1,0,1,0)
        let ml_point = MultilinearPoint(vec![
            Field64::ONE,
            Field64::ZERO,
            Field64::ONE,
            Field64::ZERO,
        ]);
        let binary_point = BinaryHypercubePoint(0b1010);

        // eq_poly should evaluate to 1 since c_i = p_i for all i
        assert_eq!(ml_point.eq_poly(binary_point), Field64::ONE);
    }

    #[test]
    fn test_eq_poly_mixed_bits_mismatch() {
        // Multilinear point (1,0,1,0)
        let ml_point = MultilinearPoint(vec![
            Field64::ONE,
            Field64::ZERO,
            Field64::ONE,
            Field64::ZERO,
        ]);
        let binary_point = BinaryHypercubePoint(0b1100); // Differs at second bit

        // eq_poly should evaluate to 0 since there is at least one mismatch
        assert_eq!(ml_point.eq_poly(binary_point), Field64::ZERO);
    }

    #[test]
    fn test_eq_poly_single_variable_match() {
        // Multilinear point (1)
        let ml_point = MultilinearPoint(vec![Field64::ONE]);
        let binary_point = BinaryHypercubePoint(0b1);

        // eq_poly should evaluate to 1 since c_1 = p_1 = 1
        assert_eq!(ml_point.eq_poly(binary_point), Field64::ONE);
    }

    #[test]
    fn test_eq_poly_single_variable_mismatch() {
        // Multilinear point (1)
        let ml_point = MultilinearPoint(vec![Field64::ONE]);
        let binary_point = BinaryHypercubePoint(0b0);

        // eq_poly should evaluate to 0 since c_1 != p_1
        assert_eq!(ml_point.eq_poly(binary_point), Field64::ZERO);
    }

    #[test]
    fn test_eq_poly_large_binary_number_match() {
        // Multilinear point (1,1,0,1,0,1,1,0)
        let ml_point = MultilinearPoint(vec![
            Field64::ONE,
            Field64::ONE,
            Field64::ZERO,
            Field64::ONE,
            Field64::ZERO,
            Field64::ONE,
            Field64::ONE,
            Field64::ZERO,
        ]);
        let binary_point = BinaryHypercubePoint(0b1101_0110);

        // eq_poly should evaluate to 1 since c_i = p_i for all i
        assert_eq!(ml_point.eq_poly(binary_point), Field64::ONE);
    }

    #[test]
    fn test_eq_poly_large_binary_number_mismatch() {
        // Multilinear point (1,1,0,1,0,1,1,0)
        let ml_point = MultilinearPoint(vec![
            Field64::ONE,
            Field64::ONE,
            Field64::ZERO,
            Field64::ONE,
            Field64::ZERO,
            Field64::ONE,
            Field64::ONE,
            Field64::ZERO,
        ]);
        let binary_point = BinaryHypercubePoint(0b1101_0111); // Last bit differs

        // eq_poly should evaluate to 0 since there is a mismatch
        assert_eq!(ml_point.eq_poly(binary_point), Field64::ZERO);
    }

    #[test]
    fn test_eq_poly_empty_vector() {
        // Empty Multilinear Point
        let ml_point = MultilinearPoint::<Field64>(vec![]);
        let binary_point = BinaryHypercubePoint(0);

        // eq_poly should evaluate to 1 since both are trivially equal
        assert_eq!(ml_point.eq_poly(binary_point), Field64::ONE);
    }

    #[test]
    fn test_eq_poly_outside_all_zeros() {
        let ml_point1 = MultilinearPoint(vec![Field64::ZERO; 4]);
        let ml_point2 = MultilinearPoint(vec![Field64::ZERO; 4]);

        assert_eq!(ml_point1.eq_poly_outside(&ml_point2), Field64::ONE);
    }

    #[test]
    fn test_eq_poly_outside_all_ones() {
        let ml_point1 = MultilinearPoint(vec![Field64::ONE; 4]);
        let ml_point2 = MultilinearPoint(vec![Field64::ONE; 4]);

        assert_eq!(ml_point1.eq_poly_outside(&ml_point2), Field64::ONE);
    }

    #[test]
    fn test_eq_poly_outside_mixed_match() {
        let ml_point1 = MultilinearPoint(vec![
            Field64::ONE,
            Field64::ZERO,
            Field64::ONE,
            Field64::ZERO,
        ]);
        let ml_point2 = MultilinearPoint(vec![
            Field64::ONE,
            Field64::ZERO,
            Field64::ONE,
            Field64::ZERO,
        ]);

        assert_eq!(ml_point1.eq_poly_outside(&ml_point2), Field64::ONE);
    }

    #[test]
    fn test_eq_poly_outside_mixed_mismatch() {
        let ml_point1 = MultilinearPoint(vec![
            Field64::ONE,
            Field64::ZERO,
            Field64::ONE,
            Field64::ZERO,
        ]);
        let ml_point2 = MultilinearPoint(vec![
            Field64::ZERO,
            Field64::ONE,
            Field64::ZERO,
            Field64::ONE,
        ]);

        assert_eq!(ml_point1.eq_poly_outside(&ml_point2), Field64::ZERO);
    }

    #[test]
    fn test_eq_poly_outside_single_variable_match() {
        let ml_point1 = MultilinearPoint(vec![Field64::ONE]);
        let ml_point2 = MultilinearPoint(vec![Field64::ONE]);

        assert_eq!(ml_point1.eq_poly_outside(&ml_point2), Field64::ONE);
    }

    #[test]
    fn test_eq_poly_outside_single_variable_mismatch() {
        let ml_point1 = MultilinearPoint(vec![Field64::ONE]);
        let ml_point2 = MultilinearPoint(vec![Field64::ZERO]);

        assert_eq!(ml_point1.eq_poly_outside(&ml_point2), Field64::ZERO);
    }

    #[test]
    fn test_eq_poly_outside_large_match() {
        let ml_point1 = MultilinearPoint(vec![
            Field64::ONE,
            Field64::ONE,
            Field64::ZERO,
            Field64::ONE,
            Field64::ZERO,
            Field64::ONE,
            Field64::ONE,
            Field64::ZERO,
        ]);
        let ml_point2 = MultilinearPoint(vec![
            Field64::ONE,
            Field64::ONE,
            Field64::ZERO,
            Field64::ONE,
            Field64::ZERO,
            Field64::ONE,
            Field64::ONE,
            Field64::ZERO,
        ]);

        assert_eq!(ml_point1.eq_poly_outside(&ml_point2), Field64::ONE);
    }

    #[test]
    fn test_eq_poly_outside_large_mismatch() {
        let ml_point1 = MultilinearPoint(vec![
            Field64::ONE,
            Field64::ONE,
            Field64::ZERO,
            Field64::ONE,
            Field64::ZERO,
            Field64::ONE,
            Field64::ONE,
            Field64::ZERO,
        ]);
        let ml_point2 = MultilinearPoint(vec![
            Field64::ONE,
            Field64::ONE,
            Field64::ZERO,
            Field64::ONE,
            Field64::ZERO,
            Field64::ONE,
            Field64::ONE,
            Field64::ONE, // Last bit differs
        ]);

        assert_eq!(ml_point1.eq_poly_outside(&ml_point2), Field64::ZERO);
    }

    #[test]
    fn test_eq_poly_outside_empty_vector() {
        let ml_point1 = MultilinearPoint::<Field64>(vec![]);
        let ml_point2 = MultilinearPoint::<Field64>(vec![]);

        assert_eq!(ml_point1.eq_poly_outside(&ml_point2), Field64::ONE);
    }

    #[test]
    #[should_panic]
    fn test_eq_poly_outside_different_lengths() {
        let ml_point1 = MultilinearPoint(vec![Field64::ONE, Field64::ZERO]);
        let ml_point2 = MultilinearPoint(vec![Field64::ONE, Field64::ZERO, Field64::ONE]);

        // Should panic because lengths do not match
        ml_point1.eq_poly_outside(&ml_point2);
    }

    #[test]
    fn test_eq_poly3_all_zeros() {
        let ml_point = MultilinearPoint(vec![Field64::ZERO; 4]);
        // (0,0,0,0) in base 3 = 0 * 3^3 + 0 * 3^2 + 0 * 3^1 + 0 * 3^0 = 0
        let ternary_point = 0;

        assert_eq!(ml_point.eq_poly3(ternary_point), Field64::ONE);
    }

    #[test]
    fn test_eq_poly3_all_ones() {
        let ml_point = MultilinearPoint(vec![Field64::ONE; 4]);
        // (1,1,1,1) in base 3 = 1 * 3^3 + 1 * 3^2 + 1 * 3^1 + 1 * 3^0
        let ternary_point = 1 * 3_i32.pow(3) + 1 * 3_i32.pow(2) + 1 * 3_i32.pow(1) + 1;

        assert_eq!(ml_point.eq_poly3(ternary_point as usize), Field64::ONE);
    }

    #[test]
    fn test_eq_poly3_all_twos() {
        let two = Field64::ONE + Field64::ONE;
        let ml_point = MultilinearPoint(vec![two; 4]);
        // (2,2,2,2) in base 3 = 2 * 3^3 + 2 * 3^2 + 2 * 3^1 + 2 * 3^0
        let ternary_point = 2 * 3_i32.pow(3) + 2 * 3_i32.pow(2) + 2 * 3_i32.pow(1) + 2;

        assert_eq!(ml_point.eq_poly3(ternary_point as usize), Field64::ONE);
    }

    #[test]
    fn test_eq_poly3_mixed_match() {
        let two = Field64::ONE + Field64::ONE;
        let ml_point = MultilinearPoint(vec![two, Field64::ONE, Field64::ZERO, Field64::ONE]);
        // (2,1,0,1) in base 3 = 2 * 3^3 + 1 * 3^2 + 0 * 3^1 + 1 * 3^0
        let ternary_point = 2 * 3_i32.pow(3) + 1 * 3_i32.pow(2) + 0 * 3_i32.pow(1) + 1;

        assert_eq!(ml_point.eq_poly3(ternary_point as usize), Field64::ONE);
    }

    #[test]
    fn test_eq_poly3_mixed_mismatch() {
        let two = Field64::ONE + Field64::ONE;
        let ml_point = MultilinearPoint(vec![two, Field64::ONE, Field64::ZERO, Field64::ONE]);
        // (2,2,0,1) differs at the second coordinate
        let ternary_point = 2 * 3_i32.pow(3) + 2 * 3_i32.pow(2) + 0 * 3_i32.pow(1) + 1;

        assert_eq!(ml_point.eq_poly3(ternary_point as usize), Field64::ZERO);
    }

    #[test]
    fn test_eq_poly3_single_variable_match() {
        let ml_point = MultilinearPoint(vec![Field64::ONE]);
        // (1) in base 3 = 1
        let ternary_point = 1;

        assert_eq!(ml_point.eq_poly3(ternary_point), Field64::ONE);
    }

    #[test]
    fn test_eq_poly3_single_variable_mismatch() {
        let ml_point = MultilinearPoint(vec![Field64::ONE]);
        // (2) in base 3 = 2
        let ternary_point = 2;

        assert_eq!(ml_point.eq_poly3(ternary_point), Field64::ZERO);
    }

    #[test]
    fn test_eq_poly3_large_match() {
        let two = Field64::ONE + Field64::ONE;
        let ml_point = MultilinearPoint(vec![
            two,
            Field64::ONE,
            Field64::ZERO,
            Field64::ONE,
            Field64::ZERO,
            two,
            Field64::ONE,
        ]);
        // (2,1,0,1,0,2,1) in base 3 = 2 * 3^6 + 1 * 3^5 + 0 * 3^4 + 1 * 3^3 + 0 * 3^2 + 2 * 3^1 + 1
        // * 3^0
        let ternary_point = 2 * 3_i32.pow(6)
            + 1 * 3_i32.pow(5)
            + 0 * 3_i32.pow(4)
            + 1 * 3_i32.pow(3)
            + 0 * 3_i32.pow(2)
            + 2 * 3_i32.pow(1)
            + 1;

        assert_eq!(ml_point.eq_poly3(ternary_point as usize), Field64::ONE);
    }

    #[test]
    fn test_eq_poly3_large_mismatch() {
        let two = Field64::ONE + Field64::ONE;
        let ml_point = MultilinearPoint(vec![
            two,
            Field64::ONE,
            Field64::ZERO,
            Field64::ONE,
            Field64::ZERO,
            two,
            Field64::ONE,
        ]);
        // (2,1,0,1,1,2,1) differs at the fifth coordinate
        let ternary_point = 2 * 3_i32.pow(6)
            + 1 * 3_i32.pow(5)
            + 0 * 3_i32.pow(4)
            + 1 * 3_i32.pow(3)
            + 1 * 3_i32.pow(2)
            + 2 * 3_i32.pow(1)
            + 1;

        assert_eq!(ml_point.eq_poly3(ternary_point as usize), Field64::ZERO);
    }

    #[test]
    fn test_eq_poly3_empty_vector() {
        let ml_point = MultilinearPoint::<Field64>(vec![]);
        let ternary_point = 0;

        assert_eq!(ml_point.eq_poly3(ternary_point), Field64::ONE);
    }

    #[test]
    #[should_panic]
    fn test_eq_poly3_invalid_ternary_value() {
        let ml_point = MultilinearPoint(vec![Field64::ONE, Field64::ZERO]);
        let ternary_point = 9; // Invalid ternary representation (not in {0,1,2})

        ml_point.eq_poly3(ternary_point);
    }

    #[test]
    fn test_equality() {
        let point = MultilinearPoint(vec![Field64::from(0), Field64::from(0)]);
        assert_eq!(point.eq_poly(BinaryHypercubePoint(0b00)), Field64::from(1));
        assert_eq!(point.eq_poly(BinaryHypercubePoint(0b01)), Field64::from(0));
        assert_eq!(point.eq_poly(BinaryHypercubePoint(0b10)), Field64::from(0));
        assert_eq!(point.eq_poly(BinaryHypercubePoint(0b11)), Field64::from(0));

        let point = MultilinearPoint(vec![Field64::from(1), Field64::from(0)]);
        assert_eq!(point.eq_poly(BinaryHypercubePoint(0b00)), Field64::from(0));
        assert_eq!(point.eq_poly(BinaryHypercubePoint(0b01)), Field64::from(0));
        assert_eq!(point.eq_poly(BinaryHypercubePoint(0b10)), Field64::from(1));
        assert_eq!(point.eq_poly(BinaryHypercubePoint(0b11)), Field64::from(0));
    }

    #[test]
    #[allow(clippy::cognitive_complexity)]
    fn test_equality3() {
        let point = MultilinearPoint(vec![Field64::from(0), Field64::from(0)]);

        assert_eq!(point.eq_poly3(0), Field64::from(1));
        assert_eq!(point.eq_poly3(1), Field64::from(0));
        assert_eq!(point.eq_poly3(2), Field64::from(0));
        assert_eq!(point.eq_poly3(3), Field64::from(0));
        assert_eq!(point.eq_poly3(4), Field64::from(0));
        assert_eq!(point.eq_poly3(5), Field64::from(0));
        assert_eq!(point.eq_poly3(6), Field64::from(0));
        assert_eq!(point.eq_poly3(7), Field64::from(0));
        assert_eq!(point.eq_poly3(8), Field64::from(0));

        let point = MultilinearPoint(vec![Field64::from(1), Field64::from(0)]);

        assert_eq!(point.eq_poly3(0), Field64::from(0));
        assert_eq!(point.eq_poly3(1), Field64::from(0));
        assert_eq!(point.eq_poly3(2), Field64::from(0));
        assert_eq!(point.eq_poly3(3), Field64::from(1)); // 3 corresponds to ternary (1,0)
        assert_eq!(point.eq_poly3(4), Field64::from(0));
        assert_eq!(point.eq_poly3(5), Field64::from(0));
        assert_eq!(point.eq_poly3(6), Field64::from(0));
        assert_eq!(point.eq_poly3(7), Field64::from(0));
        assert_eq!(point.eq_poly3(8), Field64::from(0));

        let point = MultilinearPoint(vec![Field64::from(0), Field64::from(2)]);

        assert_eq!(point.eq_poly3(0), Field64::from(0));
        assert_eq!(point.eq_poly3(1), Field64::from(0));
        assert_eq!(point.eq_poly3(2), Field64::from(1)); // 2 corresponds to ternary (0,2)
        assert_eq!(point.eq_poly3(3), Field64::from(0));
        assert_eq!(point.eq_poly3(4), Field64::from(0));
        assert_eq!(point.eq_poly3(5), Field64::from(0));
        assert_eq!(point.eq_poly3(6), Field64::from(0));
        assert_eq!(point.eq_poly3(7), Field64::from(0));
        assert_eq!(point.eq_poly3(8), Field64::from(0));

        let point = MultilinearPoint(vec![Field64::from(2), Field64::from(2)]);

        assert_eq!(point.eq_poly3(0), Field64::from(0));
        assert_eq!(point.eq_poly3(1), Field64::from(0));
        assert_eq!(point.eq_poly3(2), Field64::from(0));
        assert_eq!(point.eq_poly3(3), Field64::from(0));
        assert_eq!(point.eq_poly3(4), Field64::from(0));
        assert_eq!(point.eq_poly3(5), Field64::from(0));
        assert_eq!(point.eq_poly3(6), Field64::from(0));
        assert_eq!(point.eq_poly3(7), Field64::from(0));
        assert_eq!(point.eq_poly3(8), Field64::from(1)); // 8 corresponds to ternary (2,2)
    }
}
