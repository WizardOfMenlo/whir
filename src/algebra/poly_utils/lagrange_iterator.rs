// NOTE: This is the one from Blendy

use ark_ff::Field;

use super::{hypercube::BinaryHypercubePoint, multilinear::MultilinearPoint};

/// Iterator for evaluating the Lagrange polynomial over the hypercube `{0,1}^n`.
///
/// This efficiently computes values of the equality polynomial at every binary point.
///
/// Given a multilinear point `(c_1, ..., c_n)`, it iterates over all binary vectors `(x_1, ...,
/// x_n)` and computes:
///
/// \begin{equation}
/// y = \prod_{i=1}^{n} \left( x_i c_i + (1 - x_i) (1 - c_i) \right)
/// \end{equation}
///
/// This means `y = eq_poly(c, x)`, where `eq_poly` is the **equality polynomial**.
///
/// # Properties
/// - **Precomputed negations**: We store `1 - c_i` to avoid recomputation.
#[derive(Debug)]
pub struct LagrangePolynomialIterator<F> {
    /// The last binary point output (`None` before the first step).
    last_position: Option<usize>,
    /// The point `(c_1, ..., c_n)` stored in **reverse order** for efficient access.
    point: Vec<F>,
    /// Precomputed values `1 - c_i` stored in **reverse order**.
    point_negated: Vec<F>,
    /// Stack containing **partial products**:
    /// - Before first iteration: `[1, y_1, y_1 y_2, ..., y_1 ... y_n]`
    /// - After each iteration: updated values corresponding to the next `x`
    stack: Vec<F>,
    /// The number of variables `n` (i.e., dimension of the hypercube).
    num_variables: usize,
}

impl<F: Field> From<&MultilinearPoint<F>> for LagrangePolynomialIterator<F> {
    /// Initializes the iterator from a multilinear point `(c_1, ..., c_n)`.
    ///
    /// # Initialization:
    /// - Stores `c_i` in reverse order for **efficient bit processing**.
    /// - Precomputes `1 - c_i` for each coordinate to avoid recomputation.
    /// - Constructs a **stack** for incremental computation.
    fn from(multilinear_point: &MultilinearPoint<F>) -> Self {
        let num_variables = multilinear_point.num_variables();

        // Clone the original point (c_1, ..., c_n)
        let mut point = multilinear_point.0.clone();

        // Compute point_negated = (1 - c_1, ..., 1 - c_n)
        let mut point_negated: Vec<_> = point.iter().map(|&x| F::ONE - x).collect();

        // Compute the stack of partial products (1, (1 - c_1), ..., ∏_{i=1}^{n} (1 - c_i))
        let mut stack = Vec::with_capacity(num_variables + 1);
        let mut running_product = F::ONE;
        stack.push(running_product); // stack[0] = 1

        for &neg in &point_negated {
            running_product *= neg;
            stack.push(running_product);
        }

        // Reverse the point and its negation for bit-friendly access
        point.reverse();
        point_negated.reverse();

        Self {
            last_position: None,
            point,
            point_negated,
            stack,
            num_variables,
        }
    }
}

impl<F: Field> Iterator for LagrangePolynomialIterator<F> {
    type Item = (BinaryHypercubePoint, F);
    /// Computes the next `(x, y)` pair where `y = eq_poly(c, x)`.
    ///
    /// - The first iteration **outputs** `(0, y_1 ... y_n)`, where `y_i = (1 - c_i)`.
    /// - Subsequent iterations **update** `y` using binary code ordering, minimizing recomputations.
    fn next(&mut self) -> Option<Self::Item> {
        // a) Check if this is the first iteration
        if self.last_position.is_none() {
            // Initialize last position
            self.last_position = Some(0);
            // Return the top of the stack
            return Some((BinaryHypercubePoint(0), *self.stack.last().unwrap()));
        }

        // b) Check if in the last iteration we finished iterating
        if self.last_position.unwrap() + 1 >= 1 << self.num_variables {
            return None;
        }

        // c) Everything else, first get bit diff
        let last_position = self.last_position.unwrap();
        let next_position = last_position + 1;
        let bit_diff = last_position ^ next_position;

        // Determine the shared prefix of the most significant bits
        let low_index_of_prefix = (bit_diff + 1).trailing_zeros() as usize;

        // Discard any stack values outside of this prefix
        self.stack.truncate(self.stack.len() - low_index_of_prefix);

        // Iterate up to this prefix computing lag poly correctly
        for bit_index in (0..low_index_of_prefix).rev() {
            let last_element = self.stack.last().unwrap();
            let next_bit = (next_position & (1 << bit_index)) != 0;
            self.stack.push(if next_bit {
                *last_element * self.point[bit_index]
            } else {
                *last_element * self.point_negated[bit_index]
            });
        }

        // Don't forget to update the last position
        self.last_position = Some(next_position);

        // Return the top of the stack
        Some((
            BinaryHypercubePoint(next_position),
            *self.stack.last().unwrap(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use ark_ff::AdditiveGroup;

    use super::*;
    use crate::algebra::{
        fields::Field64,
        poly_utils::{hypercube::BinaryHypercubePoint, multilinear::MultilinearPoint},
    };

    type F = Field64;

    #[test]
    fn test_blendy() {
        let one = F::from(1);
        let (a, b) = (F::from(2), F::from(3));
        let point_1 = MultilinearPoint(vec![a, b]);

        let mut lag_iterator = LagrangePolynomialIterator::from(&point_1);

        assert_eq!(
            lag_iterator.next().unwrap(),
            (BinaryHypercubePoint(0), (one - a) * (one - b))
        );
        assert_eq!(
            lag_iterator.next().unwrap(),
            (BinaryHypercubePoint(1), (one - a) * b)
        );
        assert_eq!(
            lag_iterator.next().unwrap(),
            (BinaryHypercubePoint(2), a * (one - b))
        );
        assert_eq!(
            lag_iterator.next().unwrap(),
            (BinaryHypercubePoint(3), a * b)
        );
        assert_eq!(lag_iterator.next(), None);
    }

    #[test]
    fn test_blendy_2() {
        let point = MultilinearPoint(vec![F::from(12), F::from(13), F::from(32)]);

        let mut last_b = None;
        for (b, lag) in LagrangePolynomialIterator::from(&point) {
            assert_eq!(point.eq_poly(b), lag);
            assert!(b.0 < 1 << 3);
            last_b = Some(b);
        }
        assert_eq!(last_b, Some(BinaryHypercubePoint(7)));
    }

    #[test]
    fn test_blendy_3() {
        let point = MultilinearPoint(vec![
            F::from(414_151),
            F::from(109_849_018),
            F::from(33_184_190),
            F::from(33_184_190),
            F::from(33_184_190),
        ]);

        let mut last_b = None;
        for (b, lag) in LagrangePolynomialIterator::from(&point) {
            assert_eq!(point.eq_poly(b), lag);
            last_b = Some(b);
        }
        assert_eq!(last_b, Some(BinaryHypercubePoint(31)));
    }

    #[test]
    fn test_lagrange_iterator_single_variable() {
        let point = MultilinearPoint(vec![F::from(3)]);
        let mut iter = LagrangePolynomialIterator::from(&point);

        // Expected values: (0, 1 - p) and (1, p)
        assert_eq!(
            iter.next(),
            Some((BinaryHypercubePoint(0), F::ONE - point.0[0]))
        );
        assert_eq!(iter.next(), Some((BinaryHypercubePoint(1), point.0[0])));
        assert_eq!(iter.next(), None); // No more elements should be present
    }

    #[test]
    fn test_lagrange_iterator_two_variables() {
        let (a, b) = (F::from(2), F::from(3));
        let point = MultilinearPoint(vec![a, b]);
        let mut iter = LagrangePolynomialIterator::from(&point);

        // Expected values based on binary enumeration (big-endian)
        assert_eq!(
            iter.next(),
            Some((BinaryHypercubePoint(0b00), (F::ONE - a) * (F::ONE - b)))
        );
        assert_eq!(
            iter.next(),
            Some((BinaryHypercubePoint(0b01), (F::ONE - a) * b))
        );
        assert_eq!(
            iter.next(),
            Some((BinaryHypercubePoint(0b10), a * (F::ONE - b)))
        );
        assert_eq!(iter.next(), Some((BinaryHypercubePoint(0b11), a * b)));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_lagrange_iterator_all_zeros() {
        let point = MultilinearPoint(vec![F::ZERO; 3]);
        let mut iter = LagrangePolynomialIterator::from(&point);

        // Expect all outputs to be 1 when x is all zeros and 0 otherwise
        assert_eq!(iter.next(), Some((BinaryHypercubePoint(0b000), F::ONE)));
        assert_eq!(iter.next(), Some((BinaryHypercubePoint(0b001), F::ZERO)));
        assert_eq!(iter.next(), Some((BinaryHypercubePoint(0b010), F::ZERO)));
        assert_eq!(iter.next(), Some((BinaryHypercubePoint(0b011), F::ZERO)));
        assert_eq!(iter.next(), Some((BinaryHypercubePoint(0b100), F::ZERO)));
        assert_eq!(iter.next(), Some((BinaryHypercubePoint(0b101), F::ZERO)));
        assert_eq!(iter.next(), Some((BinaryHypercubePoint(0b110), F::ZERO)));
        assert_eq!(iter.next(), Some((BinaryHypercubePoint(0b111), F::ZERO)));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_lagrange_iterator_all_ones() {
        let point = MultilinearPoint(vec![F::ONE; 3]);
        let mut iter = LagrangePolynomialIterator::from(&point);

        // Expect all outputs to be 1 when x is all ones and 0 otherwise
        assert_eq!(iter.next(), Some((BinaryHypercubePoint(0b000), F::ZERO)));
        assert_eq!(iter.next(), Some((BinaryHypercubePoint(0b001), F::ZERO)));
        assert_eq!(iter.next(), Some((BinaryHypercubePoint(0b010), F::ZERO)));
        assert_eq!(iter.next(), Some((BinaryHypercubePoint(0b011), F::ZERO)));
        assert_eq!(iter.next(), Some((BinaryHypercubePoint(0b100), F::ZERO)));
        assert_eq!(iter.next(), Some((BinaryHypercubePoint(0b101), F::ZERO)));
        assert_eq!(iter.next(), Some((BinaryHypercubePoint(0b110), F::ZERO)));
        assert_eq!(iter.next(), Some((BinaryHypercubePoint(0b111), F::ONE)));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_lagrange_iterator_mixed_values() {
        let (a, b, c) = (F::from(2), F::from(3), F::from(4));
        let point = MultilinearPoint(vec![a, b, c]);
        let mut iter = LagrangePolynomialIterator::from(&point);

        // Verify correctness against eq_poly function
        for (b, lag) in LagrangePolynomialIterator::from(&point) {
            assert_eq!(point.eq_poly(b), lag);
        }

        // Ensure the iterator completes all 2^3 = 8 elements
        let mut count = 0;
        while iter.next().is_some() {
            count += 1;
        }
        assert_eq!(count, 8);
    }

    #[test]
    fn test_lagrange_iterator_four_variables() {
        let point = MultilinearPoint(vec![F::from(1), F::from(2), F::from(3), F::from(4)]);
        let mut iter = LagrangePolynomialIterator::from(&point);

        // Ensure the iterator completes all 2^4 = 16 elements
        let mut count = 0;
        while iter.next().is_some() {
            count += 1;
        }
        assert_eq!(count, 16);
    }

    #[test]
    fn test_lagrange_iterator_correct_order() {
        let point = MultilinearPoint(vec![F::from(5), F::from(7)]);
        let mut iter = LagrangePolynomialIterator::from(&point);

        // Expect values in **binary order**: 0b00, 0b01, 0b10, 0b11
        let expected_order = [0b00, 0b01, 0b10, 0b11];

        for &expected in &expected_order {
            let (b, _) = iter.next().unwrap();
            assert_eq!(b.0, expected);
        }

        // Ensure no extra values are generated
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_lagrange_iterator_output_count() {
        let num_vars = 5;
        let point = MultilinearPoint(vec![F::from(3); num_vars]);
        let iter = LagrangePolynomialIterator::from(&point);

        // The iterator should yield exactly 2^num_vars elements
        assert_eq!(iter.count(), 1 << num_vars);
    }

    #[test]
    fn test_lagrange_iterator_empty() {
        let point = MultilinearPoint::<F>(vec![]);
        let mut iter = LagrangePolynomialIterator::from(&point);

        // Only a single output should be generated
        assert_eq!(iter.next(), Some((BinaryHypercubePoint(0), F::ONE)));
        assert_eq!(iter.next(), None);
    }
}
