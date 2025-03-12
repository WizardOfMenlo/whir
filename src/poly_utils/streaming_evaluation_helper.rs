// NOTE: This is the one from Blendy adapted for streaming evals

use ark_ff::Field;

use super::{hypercube::BinaryHypercubePoint, multilinear::MultilinearPoint};

pub struct TermPolynomialIterator<F: Field> {
    last_position: Option<usize>,
    point: Vec<F>,
    stack: Vec<F>,
    num_variables: usize,
}

impl<F: Field> TermPolynomialIterator<F> {
    pub fn new(point: &MultilinearPoint<F>) -> Self {
        let num_variables = point.0.len();

        // Initialize a stack with capacity for messages/ message_hats and the identity element
        let stack: Vec<F> = vec![F::ONE; point.0.len() + 1];

        let mut point = point.0.clone();

        point.reverse();

        // Return
        Self {
            num_variables,
            point,
            stack,
            last_position: None,
        }
    }
}

impl<F: Field> Iterator for TermPolynomialIterator<F> {
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
            self.stack.push(if next_bit {
                *last_element * self.point[bit_index]
            } else {
                *last_element
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
