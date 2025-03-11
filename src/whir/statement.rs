use crate::poly_utils::{eq_poly_outside, evals::EvaluationsList, MultilinearPoint};
use ark_ff::Field;
use std::{fmt::Debug, ops::Index};
use rayon::prelude::*;

#[derive(Clone, Debug)]
pub enum Weights<F: Field> {
    Evaluation {
        point: MultilinearPoint<F>,
    },
    Linear {
        weight: EvaluationsList<F>,
    }
}

impl<F: Field> Weights<F> {
    pub fn evaluation(point: MultilinearPoint<F>) -> Self {
        Self::Evaluation { point }
    }

    pub fn linear(weight: EvaluationsList<F>) -> Self {
        Self::Linear { weight }
    }

    pub fn num_variables(&self) -> usize {
        match self {
            Self::Evaluation { point } => point.num_variables(),
            Self::Linear { weight } => weight.num_variables(),
        }
    }

    #[cfg(not(feature = "parallel"))]
    pub fn accumulate(&self, accumulator: &mut EvaluationsList<F>, factor: F) {
        match self {
            Weights::Evaluation { point } => {
                for (prefix, lag) in LagrangePolynomialIterator::new(point) {
                    accumulator.evals_mut()[prefix.0] += factor * lag;
                }
            }
            Weights::Linear { weight } => {
                accumulator.evals_mut().par_iter_mut().enumerate().for_each(|(corner, acc)| {
                    *acc += factor * weight.index(corner);
                });
            }
        }
    }


    #[cfg(feature = "parallel")]
    fn eval_eq(eval: &[F], out: &mut [F], scalar: F) {
        use rayon::join;

        const PARALLEL_THRESHOLD: usize = 10;
        debug_assert_eq!(out.len(), 1 << eval.len());
        if let Some((&x, tail)) = eval.split_first() {
            let (low, high) = out.split_at_mut(out.len() / 2);
            // Update scalars using a single mul. Note that this causes a data dependency,
            // so for small fields it might be better to use two muls.
            // This data dependency should go away once we implement parallel point evaluation.
            let s1 = scalar * x;
            let s0 = scalar - s1;
            if tail.len() > PARALLEL_THRESHOLD {
                join(
                    || Self::eval_eq(tail, low, s0),
                    || Self::eval_eq(tail, high, s1),
                );
            } else {
                Self::eval_eq(tail, low, s0);
                Self::eval_eq(tail, high, s1);
            }
        } else {
            out[0] += scalar;
        }
    }

    #[cfg(feature = "parallel")]
    pub fn accumulate(&self, accumulator: &mut EvaluationsList<F>, factor: F) {
        assert_eq!(accumulator.num_variables(), self.num_variables());
        match self {
            Weights::Evaluation { point } => {
                Self::eval_eq(&point.0, accumulator.evals_mut(), factor);
            }
            Weights::Linear { weight } => {
                accumulator.evals_mut().par_iter_mut().enumerate().for_each(|(corner, acc)| {
                    *acc += factor * weight.index(corner);
                });
            }
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
        MultilinearPoint(
            bits
        )
    }

    fn evaluate_cube(&self, corner: usize) -> F {
        let point = self.convert_to_multilinear_point(corner);
        self.evaluate_mle(&point)
    }

    fn evaluate_mle(&self, point: &MultilinearPoint<F>) -> F {
        match self {
            Weights::Evaluation { point: p } => {
                assert_eq!(point.num_variables(), p.num_variables());
                let mut acc = F::ONE;
                for (&l, &r) in p.0.iter().zip(&point.0) {
                    acc *= l * r + (F::ONE - l) * (F::ONE - r);
                }
                acc
            },
            _ => F::ZERO,
        }
    }

    pub fn weighted_sum(&self, poly: &EvaluationsList<F>) -> F {
        match self {
            Self::Linear { weight } => {
                assert_eq!(poly.num_variables(), weight.num_variables());
                #[cfg(not(feature = "parallel"))]
                {
                    let mut sum = F::ZERO;
                    for (corner, poly) in poly.evals().iter().enumerate() {
                        sum += *weight.index(corner) * poly;
                    }
                    sum
                }
                #[cfg(feature = "parallel")]
                {
                    poly.evals()
                        .par_iter()
                        .enumerate()
                        .map(|(corner, poly)| *weight.index(corner) * *poly)
                        .sum()
                }
            },
            Self::Evaluation { point } => {
                let mut sum = F::ZERO;
                for (corner, poly) in poly.evals().iter().enumerate() {
                    sum += self.evaluate_cube(corner) * poly;
                }
                sum
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct Statement<F: Field> {
    num_variables: usize,
    pub constraints: Vec<(Weights<F>, F)>,
}

#[derive(Clone, Debug)]
pub enum VerifierWeights<F: Field> {
    Evaluation {
        point: MultilinearPoint<F>,
    },
    Linear {
        num_variables: usize,
        term: Option<F>,
    },
}

impl<F: Field> VerifierWeights<F> {
    pub fn evaluation(point: MultilinearPoint<F>) -> Self {
        Self::Evaluation { point }
    }

    pub fn linear(num_variables: usize, term: Option<F>) -> Self {
        Self::Linear { num_variables, term }
    }

    pub fn num_variables(&self) -> usize {
        match self {
            Self::Evaluation { point } => point.num_variables(),
            Self::Linear{ num_variables, .. } => *num_variables,
        }
    }
    pub fn compute(&self, folding_randomness: &MultilinearPoint<F>) -> F {
        match self {
            Self::Evaluation { point } => eq_poly_outside(point, folding_randomness),
            Self::Linear { term, .. } => term.unwrap(),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct StatementVerifier<F: Field> {
    num_variables: usize,
    pub constraints: Vec<(VerifierWeights<F>, F)>,
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
    fn new(num_variables: usize) -> Self {
        Self {
            num_variables,
            constraints: Vec::new(),
        }
    }
    
    pub fn num_variables(&self) -> usize {
        self.num_variables
    }

    pub fn add_constraint(&mut self, weights: VerifierWeights<F>, sum: F) {
        assert_eq!(weights.num_variables(), self.num_variables());
        self.constraints.push((weights, sum));
    }

    pub fn add_constraint_in_front(&mut self, weights: VerifierWeights<F>, sum: F) {
        assert_eq!(weights.num_variables(), self.num_variables());
        self.constraints.insert(0, (weights, sum));
    }

    pub fn add_constraints_in_front(&mut self, constraints: Vec<(VerifierWeights<F>, F)>) {
        for (weights, _) in &constraints {
            assert_eq!(weights.num_variables(), self.num_variables());
        }
        self.constraints.splice(0..0, constraints);
    }

}
impl<F: Field> StatementVerifier<F> {
    pub fn from_statement(statement: &Statement<F>) -> Self {
        let mut verifier = StatementVerifier::new(statement.num_variables());
        for (weights, sum) in &statement.constraints {
            match weights {
                Weights::Linear { weight, .. } => {
                    let weights = VerifierWeights::linear(weight.num_variables(), None);
                    verifier.add_constraint(weights.clone(), *sum);
                }
                Weights::Evaluation { point } => {
                    let weights = VerifierWeights::evaluation(point.clone());
                    verifier.add_constraint(weights.clone(), *sum);
                }
            }
        }
        verifier
    }
}
