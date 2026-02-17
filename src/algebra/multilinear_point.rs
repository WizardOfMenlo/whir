use ark_ff::Field;
use ark_std::rand::{distributions::Standard, prelude::Distribution, Rng, RngCore};

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
    pub const fn num_variables(&self) -> usize {
        self.0.len()
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
    pub(crate) fn eq_poly(&self, mut point: usize) -> F {
        let n_variables = self.num_variables();
        assert!(point < (1 << n_variables)); // Ensure correct length

        let mut acc = F::ONE;

        for val in self.0.iter().rev() {
            let b = point % 2;
            acc *= if b == 1 { *val } else { F::ONE - *val };
            point >>= 1;
        }

        acc
    }

    /// Computes eq(c, p) on the hypercube for all p.
    pub(crate) fn eq_weights(&self) -> Vec<F> {
        (0..1 << self.0.len())
            .map(|point| self.eq_poly(point))
            .collect()
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
    use ark_std::rand::thread_rng;

    use super::*;
    use crate::algebra::fields::Field64;

    #[test]
    fn test_n_variables() {
        let point =
            MultilinearPoint::<Field64>(vec![Field64::from(1), Field64::from(0), Field64::from(1)]);
        assert_eq!(point.num_variables(), 3);
    }

    #[test]
    fn test_eq_poly_all_zeros() {
        // Multilinear point (0,0,0,0)
        let ml_point = MultilinearPoint(vec![Field64::ZERO; 4]);
        let binary_point = 0b0000;

        // eq_poly should evaluate to 1 since c_i = p_i = 0
        assert_eq!(ml_point.eq_poly(binary_point), Field64::ONE);
    }

    #[test]
    fn test_eq_poly_all_ones() {
        // Multilinear point (1,1,1,1)
        let ml_point = MultilinearPoint(vec![Field64::ONE; 4]);
        let binary_point = 0b1111;

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
        let binary_point = 0b1010;

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
        let binary_point = 0b1100; // Differs at second bit

        // eq_poly should evaluate to 0 since there is at least one mismatch
        assert_eq!(ml_point.eq_poly(binary_point), Field64::ZERO);
    }

    #[test]
    fn test_eq_poly_single_variable_match() {
        // Multilinear point (1)
        let ml_point = MultilinearPoint(vec![Field64::ONE]);
        let binary_point = 0b1;

        // eq_poly should evaluate to 1 since c_1 = p_1 = 1
        assert_eq!(ml_point.eq_poly(binary_point), Field64::ONE);
    }

    #[test]
    fn test_eq_poly_single_variable_mismatch() {
        // Multilinear point (1)
        let ml_point = MultilinearPoint(vec![Field64::ONE]);
        let binary_point = 0b0;

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
        let binary_point = 0b1101_0110;

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
        let binary_point = 0b1101_0111; // Last bit differs

        // eq_poly should evaluate to 0 since there is a mismatch
        assert_eq!(ml_point.eq_poly(binary_point), Field64::ZERO);
    }

    #[test]
    fn test_eq_poly_empty_vector() {
        // Empty Multilinear Point
        let ml_point = MultilinearPoint::<Field64>(vec![]);
        let binary_point = 0;

        // eq_poly should evaluate to 1 since both are trivially equal
        assert_eq!(ml_point.eq_poly(binary_point), Field64::ONE);
    }

    #[test]
    fn test_equality() {
        let point = MultilinearPoint(vec![Field64::from(0), Field64::from(0)]);
        assert_eq!(point.eq_poly(0b00), Field64::from(1));
        assert_eq!(point.eq_poly(0b01), Field64::from(0));
        assert_eq!(point.eq_poly(0b10), Field64::from(0));
        assert_eq!(point.eq_poly(0b11), Field64::from(0));

        let point = MultilinearPoint(vec![Field64::from(1), Field64::from(0)]);
        assert_eq!(point.eq_poly(0b00), Field64::from(0));
        assert_eq!(point.eq_poly(0b01), Field64::from(0));
        assert_eq!(point.eq_poly(0b10), Field64::from(1));
        assert_eq!(point.eq_poly(0b11), Field64::from(0));
    }

    #[test]
    fn test_multilinear_point_rand_not_all_same() {
        const K: usize = 20; // Number of trials
        const N: usize = 10; // Number of variables

        let mut rng = thread_rng();

        let mut all_same_count = 0;

        for _ in 0..K {
            let point = MultilinearPoint::<Field64>::rand(&mut rng, N);
            let first = point.0[0];

            // Check if all coordinates are the same as the first one
            if point.0.iter().all(|&x| x == first) {
                all_same_count += 1;
            }
        }

        // If all K trials are completely uniform, the RNG is suspicious
        assert!(
            all_same_count < K,
            "rand generated uniform points in all {K} trials"
        );
    }
}
