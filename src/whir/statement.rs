use crate::poly_utils::{evals::EvaluationsList, MultilinearPoint};
use ark_ff::Field;
use std::fmt::Debug;

/// Weights for a linear constraint.
pub trait Weights<F: Field>: Debug {
    fn num_variables(&self) -> usize;

    /// Evaluate the MLE of the Weights at an arbitrary point.
    fn evaluate_mle(&self, point: &MultilinearPoint<F>) -> F;

    /// Return the weight associated with a particular corner.
    /// `corner` must be < 2^num_variables.
    fn evaluate_cube(&self, corner: usize) -> F {
        let point = MultilinearPoint(
            (0..self.num_variables())
                .map(|b| {
                    if (corner >> b) & 1 == 1 {
                        F::ONE
                    } else {
                        F::ZERO
                    }
                })
                .collect::<Vec<_>>(),
        );
        self.evaluate_mle(&point)
    }

    /// Accumulate the weights on the hypercube.
    /// The default implementation repeatedly calls `evaluate_cube`.
    fn accumulate(&self, accumulator: &mut EvaluationsList<F>, factor: F) {
        assert_eq!(accumulator.num_variables(), self.num_variables());
        for (corner, acc) in accumulator.evals_mut().iter_mut().enumerate() {
            *acc += factor * self.evaluate_cube(corner);
        }
    }

    /// Compute the weighted sum with the given evaluations.
    /// The default implementation repeatedly calls `evaluate_cube`.
    fn weighted_sum(&self, poly: &EvaluationsList<F>) -> F {
        assert_eq!(poly.num_variables(), self.num_variables());
        let mut sum = F::ZERO;
        for (corner, poly) in poly.evals().iter().enumerate() {
            sum += self.evaluate_cube(corner) * poly;
        }
        sum
    }
}

/// A statement for the prover to prove. Statements are a collection of linear constraints.
#[derive(Debug, Default)]
pub struct Statement<F: Field> {
    num_variables: usize,
    constraints: Vec<(Box<dyn Weights<F>>, F)>,
}

#[derive(Clone, Debug)]
pub struct EvaluationWeights<F: Field> {
    point: MultilinearPoint<F>,
}

impl<F: Field> Statement<F> {
    pub fn new(num_variables: usize) -> Self {
        Self {
            num_variables,
            ..Self::default()
        }
    }

    pub fn num_variables(&self) -> usize {
        self.num_variables
    }

    pub fn add_constraint(&mut self, weights: Box<dyn Weights<F>>, sum: F) {
        assert_eq!(weights.num_variables(), self.num_variables());
        self.constraints.push((weights, sum));
    }

    /// Combine all linear constraints into a single dense linear constraint.
    pub fn combine(&self, challenge: F) -> (EvaluationsList<F>, F) {
        todo!()
    }
}

impl<F: Field> Weights<F> for EvaluationWeights<F> {
    fn num_variables(&self) -> usize {
        self.point.num_variables()
    }

    fn evaluate_mle(&self, point: &MultilinearPoint<F>) -> F {
        assert_eq!(point.num_variables(), self.point.num_variables());
        let mut acc = F::ONE;
        for (&l, &r) in self.point.0.iter().zip(&point.0) {
            acc *= l * r + (F::ONE - l) * (F::ONE - r);
        }
        acc
    }
}
