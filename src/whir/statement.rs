use crate::poly_utils::{eq_poly_outside, evals::EvaluationsList, MultilinearPoint};
use ark_ff::Field;
use std::fmt::Debug;

#[derive(Clone, Debug)]
pub enum Weights<F: Field> {
    Evaluation {
        point: MultilinearPoint<F>,
    },
    Linear {
        weight: EvaluationsList<F>,
    },
    LinearVerifier {
        num_variables: usize,
        term: F,
    },
}

impl<F: Field> Weights<F> {
    pub fn evaluation(point: MultilinearPoint<F>) -> Self {
        Self::Evaluation { point }
    }

    pub fn linear(weight: EvaluationsList<F>) -> Self {
        Self::Linear { weight }
    }

    pub fn linear_verifier(num_variables: usize, term: F) -> Self {
        Self::LinearVerifier { num_variables, term }
    }

    pub fn num_variables(&self) -> usize {
        match self {
            Self::Evaluation { point } => point.num_variables(),
            Self::Linear { weight } => weight.num_variables(),
            Self::LinearVerifier { num_variables, .. } => *num_variables,
        }
    }

    pub fn ev_linear(&self, weight: &EvaluationsList<F>, point: &MultilinearPoint<F>) -> F {
        assert_eq!(point.num_variables(), weight.num_variables());
        weight.eval_extension(&point)
    }

    pub fn ev_regular(&self, eval_point: &MultilinearPoint<F>, point: &MultilinearPoint<F>) -> F {
        assert_eq!(point.num_variables(), eval_point.num_variables());
        let mut acc = F::ONE;
        for (&l, &r) in eval_point.0.iter().zip(&point.0) {
            if acc == F::ZERO {
                return F::ZERO;
            } 
            acc *= l * r + (F::ONE - l) * (F::ONE - r);
        }
        acc
    }

    pub fn evaluate_mle(&self, point: &MultilinearPoint<F>) -> F {
        match self {
            Self::Evaluation { point: eval_point } => {
                self.ev_regular(eval_point, point)
            },
            Self::Linear { weight } => {
                self.ev_linear(weight, point)
            },
            Self::LinearVerifier { term, .. } => *term,
        }
    }

    fn convert_to_multilinear_point(&self, corner: usize) -> MultilinearPoint<F> {
        let mut bits = (0..self.num_variables())
            .map(|b| {
                if (corner >> b) & 1 == 1 {
                    F::ONE
                } else {
                    F::ZERO
                }
            })
            .collect::<Vec<_>>();
        bits.reverse();
        MultilinearPoint(bits)
    }

    pub fn evaluate_cube(&self, corner: usize) -> F {
        let point = self.convert_to_multilinear_point(corner);
        self.evaluate_mle(&point)
    }

    pub fn accumulate(&self, accumulator: &mut EvaluationsList<F>, factor: F) {
        assert_eq!(accumulator.num_variables(), self.num_variables());
        for (corner, acc) in accumulator.evals_mut().iter_mut().enumerate() {
            *acc += factor * self.evaluate_cube(corner);
        }
    }

    pub fn weighted_sum(&self, poly: &EvaluationsList<F>) -> F {
        match self {
            Self::Linear { weight } => {
                assert_eq!(poly.num_variables(), weight.num_variables());
                let mut sum = F::ZERO;
                for (corner, poly) in poly.evals().iter().enumerate() {
                    let point = self.convert_to_multilinear_point(corner);
                    sum += weight.eval_extension(&point) * poly;
                }
                sum
            },
            Self::LinearVerifier { term, .. } => *term,
            _ => {
                assert_eq!(poly.num_variables(), self.num_variables());
                let mut sum = F::ZERO;
                for (corner, poly) in poly.evals().iter().enumerate() {
                    sum += self.evaluate_cube(corner) * poly;
                }
                sum
            }
        }
    }

    pub fn compute(&self, folding_randomness: &MultilinearPoint<F>) -> F {
        match self {
            Self::Evaluation { point } => eq_poly_outside(point, folding_randomness),
            Self::LinearVerifier { term, .. } => *term,
            _ => F::ZERO,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Statement<F: Field> {
    num_variables: usize,
    pub constraints: Vec<(Weights<F>, F)>,
}

#[derive(Clone, Debug, Default)]
pub struct StatementVerifier<F: Field> {
    num_variables: usize,
    pub constraints: Vec<(Option<F>, F)>,
}

impl<F: Field> Statement<F> {
    pub fn new(num_variables: usize) -> Self {
        Self {
            num_variables,
            constraints: Vec::new(),
        }
    }

    pub fn num_variables(&self) -> usize {
        self.num_variables
    }

    pub fn add_constraint(&mut self, weights: Weights<F>, sum: F) {
        assert_eq!(weights.num_variables(), self.num_variables());
        self.constraints.push((weights, sum));
    }

    pub fn add_constraint_in_front(&mut self, weights: Weights<F>, sum: F) {
        assert_eq!(weights.num_variables(), self.num_variables());
        self.constraints.insert(0, (weights, sum));
    }

    pub fn add_constraints_in_front(&mut self, constraints: Vec<(Weights<F>, F)>) {
        for (weights, _) in &constraints {
            assert_eq!(weights.num_variables(), self.num_variables());
        }
        self.constraints.splice(0..0, constraints);
    }

    pub fn combine(&self, challenge: F) -> (EvaluationsList<F>, F) {
        let evaluations_vec = vec![F::ZERO; 1 << self.num_variables];
        let mut combined_evals = EvaluationsList::new(evaluations_vec);
        let mut combined_sum = F::ZERO;

        let mut challenge_power = F::ONE;

        for (weights, sum) in &self.constraints {
            weights.accumulate(&mut combined_evals, challenge_power);
            combined_sum += *sum * challenge_power;
            challenge_power *= challenge;
        }

        (combined_evals, combined_sum)
    }
}

impl<F: Field> StatementVerifier<F> {
    pub fn new(num_variables: usize) -> Self {
        Self {
            num_variables,
            constraints: Vec::new(),
        }
    }
    
    pub fn num_variables(&self) -> usize {
        self.num_variables
    }

    pub fn add_constraint(&mut self, term: Option<F>, sum: F) {
        self.constraints.push((term, sum));
    }
}