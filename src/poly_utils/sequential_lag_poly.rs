// NOTE: This is the one from Blendy

use ark_ff::Field;

use super::{hypercube::BinaryHypercubePoint, MultilinearPoint};

/// There is an alternative (possibly more efficient) implementation that iterates over the x in Gray code ordering.

///
/// LagrangePolynomialIterator for a given multilinear n-dimensional `point` iterates over pairs (x, y)
/// where x ranges over all possible {0,1}^n
/// and y equals the product y_1 * ... * y_n where
///
/// y_i = point[i] if x_i == 1
/// y_i = 1-point[i] if x_i == 0
///
/// This means that y == eq_poly(point, x)
pub struct LagrangePolynomialIterator<F: Field> {
    last_position: Option<usize>, // the previously output BinaryHypercubePoint (encoded as usize). None before the first output.
    point: Vec<F>, // stores a copy of the `point` given when creating the iterator. For easier(?) bit-fiddling, we store in in reverse order.
    point_negated: Vec<F>, // stores the precomputed values 1-point[i] in the same ordering as point.
    /// stack Stores the n+1 values (in order) 1, y_1, y_1*y_2, y_1*y_2*y_3, ..., y_1*...*y_n for the previously output y.
    /// Before the first iteration (if last_position == None), it stores the values for the next (i.e. first) output instead.
    stack: Vec<F>,
    num_variables: usize, // dimension
}

impl<F: Field> LagrangePolynomialIterator<F> {
    pub fn new(point: &MultilinearPoint<F>) -> Self {
        let num_variables = point.0.len();

        // Initialize a stack with capacity for messages/ message_hats and the identity element
        let mut stack: Vec<F> = Vec::with_capacity(point.0.len() + 1);
        stack.push(F::ONE);

        let mut point = point.0.clone();
        let mut point_negated: Vec<_> = point.iter().map(|x| F::ONE - *x).collect();
        // Iterate over the message_hats, update the running product, and push it onto the stack
        let mut running_product: F = F::ONE;
        for point_neg in &point_negated {
            running_product *= point_neg;
            stack.push(running_product);
        }

        point.reverse();
        point_negated.reverse();

        // Return
        Self {
            num_variables,
            point,
            point_negated,
            stack,
            last_position: None,
        }
    }
}

impl<F: Field> Iterator for LagrangePolynomialIterator<F> {
    type Item = (BinaryHypercubePoint, F);
    // Iterator implementation for the struct
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
            let next_bit: bool = (next_position & (1 << bit_index)) != 0;
            self.stack.push(match next_bit {
                true => *last_element * self.point[bit_index],
                false => *last_element * self.point_negated[bit_index],
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
    use crate::{
        crypto::fields::Field64,
        poly_utils::{eq_poly, hypercube::BinaryHypercubePoint, MultilinearPoint},
    };

    use super::LagrangePolynomialIterator;

    type F = Field64;

    #[test]
    fn test_blendy() {
        let one = F::from(1);
        let (a, b) = (F::from(2), F::from(3));
        let point_1 = MultilinearPoint(vec![a, b]);

        let mut lag_iterator = LagrangePolynomialIterator::new(&point_1);

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
        for (b, lag) in LagrangePolynomialIterator::new(&point) {
            assert_eq!(eq_poly(&point, b), lag);
            assert!(b.0 < 1 << 3);
            last_b = Some(b);
        }
        assert_eq!(last_b, Some(BinaryHypercubePoint(7)));
    }

    #[test]
    fn test_blendy_3() {
        let point = MultilinearPoint(vec![
            F::from(414151),
            F::from(109849018),
            F::from(033184190),
            F::from(033184190),
            F::from(033184190),
        ]);

        let mut last_b = None;
        for (b, lag) in LagrangePolynomialIterator::new(&point) {
            assert_eq!(eq_poly(&point, b), lag);
            last_b = Some(b);
        }
        assert_eq!(last_b, Some(BinaryHypercubePoint(31)));
    }
}
