use crate::poly_utils::{eq_poly_outside, evals::EvaluationsList, MultilinearPoint};
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
        let point = self.convert_to_multilinear_point(corner);
        self.evaluate_mle(&point)
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

    /// Accumulate the weights on the hypercube.
    /// The default implementation repeatedly calls `evaluate_cube`.
    fn accumulate(&self, accumulator: &mut EvaluationsList<F>, factor: F) {
        assert_eq!(accumulator.num_variables(), self.num_variables());
        for (corner, acc) in accumulator.evals_mut().iter_mut().enumerate() {
            println!("corner {:?} acc before {:?}", corner, acc);
            *acc += factor * self.evaluate_cube(corner);
            println!("corner {:?} acc after {:?}", corner, acc);
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

    fn get_point_if_evaluation(&self) -> Option<MultilinearPoint<F>> {
        None
    }

    fn get_statement_for_verifier(&self, point: &MultilinearPoint<F>) -> Option<AffineClaimVerifier<F>> {
        None
    }

    fn compute(&self, value: F, point: &MultilinearPoint<F>) -> F {
        F::ZERO
    }

    fn box_clone(&self) -> Box<dyn Weights<F>>;

}

impl<F: Field> Clone for Statement<F> {
    fn clone(&self) -> Self {
        let mut new_constraints = Vec::with_capacity(self.constraints.len());
        for (weights, sum) in &self.constraints {
            new_constraints.push((weights.box_clone(), *sum));
        }
        Self {
            num_variables: self.num_variables,
            constraints: new_constraints,
        }
    }
}
/// A statement for the prover to prove. Statements are a collection of linear constraints.
#[derive(Debug, Default)]
pub struct Statement<F: Field> {
    num_variables: usize,
    pub constraints: Vec<(Box<dyn Weights<F>>, F)>,
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

    pub fn add_constraint_in_front(&mut self, weights: Box<dyn Weights<F>>, sum: F) {
        assert_eq!(weights.num_variables(), self.num_variables());
        self.constraints.insert(0, (weights, sum));
    }

    /// Combine all linear constraints into a single dense linear constraint.
    pub fn combine(&self, challenge: F) -> (EvaluationsList<F>, F) {
        let evaluations_vec = vec![F::ZERO; 1 << self.num_variables];
        let mut combined_evals = EvaluationsList::new(evaluations_vec);
        let mut combined_sum = F::ZERO;

        let mut challenge_power = F::ONE;
        println!("combined_evals {:?}", combined_evals);

        for (weights, sum) in &self.constraints {
            println!("weights for accumulation {:?}", weights);
            weights.accumulate(&mut combined_evals, challenge_power);

            println!("combined_evals {:?}", combined_evals);
            println!("sum {:?} challenge_power {:?}", sum, challenge_power);
            combined_sum += *sum * challenge_power;

            challenge_power *= challenge;
        }
        println!("combined sum {:?}", combined_sum);

        (combined_evals, combined_sum)
    }

    fn compute(&self, value: F, folding_randomness: &MultilinearPoint<F>) -> F {
        F::ONE
    }
}

impl<F: Field> EvaluationWeights<F> {
    pub fn new(point: MultilinearPoint<F>) -> Self {
        Self {
            point
        }
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

    fn get_point_if_evaluation(&self) -> Option<MultilinearPoint<F>> {
        Some(self.point.clone())
    }
    
    fn box_clone(&self) -> Box<dyn Weights<F>> {
        Box::new(self.clone())
    }

    fn compute(&self, value: F, folding_randomness: &MultilinearPoint<F>) -> F {
        println!("In EvaluationWeights value {:?} point {:?} folding {:?}", value, self.point, folding_randomness);
        value * eq_poly_outside(&self.point, &folding_randomness)
    }
}

#[derive(Clone, Debug)]
pub struct AffineClaimWeights<F: Field> {
    input: EvaluationsList<F>,
    constant_term: EvaluationsList<F>,
    linear_term: EvaluationsList<F>,
}

impl<F: Field> AffineClaimWeights<F> {
    pub fn new(input: EvaluationsList<F>, constant_term: EvaluationsList<F>, linear_term: EvaluationsList<F>) -> Self {
        assert!(input.num_variables() == constant_term.num_variables());
        assert!(constant_term.num_variables() == linear_term.num_variables());
        Self {
            input,
            constant_term,
            linear_term
        }
    }
}

impl<F: Field> Weights<F> for AffineClaimWeights<F> {
    fn num_variables(&self) -> usize {
        self.input.num_variables()
    }

    fn evaluate_mle(&self, point: &MultilinearPoint<F>) -> F {
        assert_eq!(point.num_variables(), self.input.num_variables());
        let mut acc = F::ZERO;
        for l in point.0.clone() {
            acc += self.linear_term.evaluate(point) * self.input.evaluate(point) + self.constant_term.evaluate(point) * l
        }
        acc
    }

    fn get_point_if_evaluation(&self) -> Option<MultilinearPoint<F>> {
        None
    }
    
    fn box_clone(&self) -> Box<dyn Weights<F>> {
        Box::new(self.clone())
    }

    fn weighted_sum(&self, poly: &EvaluationsList<F>) -> F {
        assert_eq!(poly.num_variables(), self.num_variables());
        let mut sum = F::ZERO;
        for (corner, poly) in poly.evals().iter().enumerate() {
            let point = self.convert_to_multilinear_point(corner);
            println!("input {:?} constant {:?}  linear term {:?} poly {:?}", self.input.evaluate(&point), self.constant_term.evaluate(&point), self.linear_term.evaluate(&point), poly);
            sum += self.input.evaluate(&point) * self.constant_term.evaluate(&point) + self.linear_term.evaluate(&point) * poly;
        }
        sum
    }

    fn get_statement_for_verifier(&self, point: &MultilinearPoint<F>) -> Option<AffineClaimVerifier<F>> {
        Some(AffineClaimVerifier::new(self.input.clone(), self.constant_term.evaluate(point), self.linear_term.evaluate(point)))
    }
}

#[derive(Clone, Debug)]
pub struct AffineClaimVerifier<F: Field> {
    input: EvaluationsList<F>,
    constant_term: F,
    linear_term: F,
}

impl<F: Field> AffineClaimVerifier<F> {
    pub fn new(input: EvaluationsList<F>, constant_term: F, linear_term: F) -> Self {
        Self {
            input,
            constant_term,
            linear_term,
        }
    }
}


impl<F: Field> Weights<F> for AffineClaimVerifier<F> {
    fn num_variables(&self) -> usize {
        self.input.num_variables()
    }

    fn evaluate_mle(&self, point: &MultilinearPoint<F>) -> F {
        assert_eq!(point.num_variables(), self.input.num_variables());
        let mut acc = F::ZERO;
        for l in point.0.clone() {
            acc += self.linear_term * self.input.evaluate(point) + self.constant_term * l
        }
        acc
    }

    fn get_point_if_evaluation(&self) -> Option<MultilinearPoint<F>> {
        None
    }
    
    fn box_clone(&self) -> Box<dyn Weights<F>> {
        Box::new(self.clone())
    }

    fn weighted_sum(&self, poly: &EvaluationsList<F>) -> F {
        assert_eq!(poly.num_variables(), self.num_variables());
        let mut sum = F::ZERO;
        for (corner, poly) in poly.evals().iter().enumerate() {
            let point = self.convert_to_multilinear_point(corner);
            println!("input {:?} constant {:?}  linear term {:?} poly {:?}", self.input.evaluate(&point), self.constant_term, self.linear_term, poly);
            sum += self.input.evaluate(&point) * self.constant_term + self.linear_term * poly;
        }
        sum
    }

    fn get_statement_for_verifier(&self, point: &MultilinearPoint<F>) -> Option<AffineClaimVerifier<F>> {
        Some(AffineClaimVerifier::new(self.input.clone(), self.constant_term.clone(), self.linear_term.clone()))
    }

    fn compute(&self, value: F, folding_randomness: &MultilinearPoint<F>) -> F {
        println!("In AffineClaimVerifier constant {:?} input.evaluate {:?} linear {:?}, final_value {:?}", self.constant_term, folding_randomness, self.linear_term, value);

        self.constant_term * self.input.evaluate(folding_randomness) + self.linear_term * value
    }
}