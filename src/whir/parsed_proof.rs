use ark_ff::Field;

use super::stir_evaluations::StirEvalContext;
use crate::{
    poly_utils::{coeffs::CoefficientList, multilinear::MultilinearPoint},
    sumcheck::SumcheckPolynomial,
};

/// Represents a single folding round in the WHIR protocol.
///
/// This structure enables recursive compression and verification of a Reed–Solomon
/// proximity test under algebraic constraints.
#[derive(Default, Debug, Clone)]
pub(crate) struct ParsedRound<F> {
    /// Folding randomness vector used in this round.
    pub(crate) folding_randomness: MultilinearPoint<F>,
    /// Out-of-domain query points.
    pub(crate) ood_points: Vec<F>,
    /// OOD answers at each query point for each polynomial.
    pub(crate) ood_answers: Vec<Vec<F>>,
    /// Indexes of STIR constraint polynomials used in this round.
    pub(crate) stir_challenges_indexes: Vec<usize>,
    /// STIR constraint evaluation points.
    pub(crate) stir_challenges_points: Vec<F>,
    /// Answers to the STIR constraints at each evaluation point for each polynomial.
    /// [polynomial][coset index][index in coset]
    pub(crate) stir_challenges_answers: Vec<Vec<Vec<F>>>,
    /// Initial random coefficients used to combine polynomials.
    pub(crate) polynomial_randomness: Vec<F>,
    /// Initial random coefficients used to combine constraints.
    pub(crate) constraint_randomness: Vec<F>,
    /// Sumcheck messages and challenge values for verifying correctness.
    pub(crate) sumcheck_rounds: Vec<(SumcheckPolynomial<F>, F)>,
    /// Inverse of the domain generator used in this round.
    pub(crate) domain_gen_inv: F,
}

/// Represents a fully parsed and structured WHIR proof.
///
/// The structure is designed to support recursive verification and evaluation
/// of folded functions under STIR-style constraints.
#[derive(Default, Clone)]
pub(crate) struct ParsedProof<F> {
    /// Initial random coefficients used to combine polynomials before folding.
    pub(crate) initial_polynomial_randomness: Vec<F>,
    /// Initial random coefficients used to combine constraints before folding.
    pub(crate) initial_constraint_randomness: Vec<F>,
    /// Initial sumcheck messages and challenges for the first constraint.
    pub(crate) initial_sumcheck_rounds: Vec<(SumcheckPolynomial<F>, F)>,
    /// All folding rounds, each reducing the problem dimension.
    pub(crate) rounds: Vec<ParsedRound<F>>,
    /// Inverse of the domain generator used in the final round.
    pub(crate) final_domain_gen_inv: F,
    /// Indexes of the final constraint polynomials.
    pub(crate) final_randomness_indexes: Vec<usize>,
    /// Evaluation points for the final constraint polynomials.
    pub(crate) final_randomness_points: Vec<F>,
    /// Evaluation results of the final constraints.
    pub(crate) final_randomness_answers: Vec<Vec<F>>,
    /// Folding randomness used in the final recursive step.
    pub(crate) final_folding_randomness: MultilinearPoint<F>,
    /// Final sumcheck proof for verifying the last constraint.
    pub(crate) final_sumcheck_rounds: Vec<(SumcheckPolynomial<F>, F)>,
    /// Challenge vector used to evaluate the last polynomial.
    pub(crate) final_sumcheck_randomness: MultilinearPoint<F>,
    /// Coefficients of the final small polynomial.
    pub(crate) final_coefficients: CoefficientList<F>,
    /// Evaluation values of the statement being proven at a random point.
    pub(crate) statement_values_at_random_point: Vec<F>,
}

impl<F: Field> ParsedProof<F> {
    /// Computes all intermediate fold evaluations using prover-assisted folding.
    ///
    /// For each round, this evaluates the STIR answers as multilinear polynomials
    /// at the provided folding randomness point. This simulates what the verifier
    /// would receive in a sound recursive sumcheck-based proximity test.
    ///
    /// Returns:
    /// - A vector of vectors, where each inner vector contains the evaluated result
    ///   of each multilinear polynomial at its corresponding folding point.
    pub fn compute_folds_helped(&self) -> Vec<Vec<F>> {
        // Closure to apply folding evaluation logic.
        let evaluate_answers = |answers: &[Vec<F>], randomness: &MultilinearPoint<F>| {
            let mut out = Vec::with_capacity(answers.len());
            StirEvalContext::ProverHelps {
                folding_randomness: randomness,
            }
            .evaluate(answers, &mut out);
            out
        };

        todo!()
        // let mut result: Vec<_> = self
        //     .rounds
        //     .iter()
        //     .map(|round| evaluate_answers(round.stir_challenges_answers, &round.folding_randomness))
        //     .collect();

        // result.push(evaluate_answers(
        //     &self.final_randomness_answers,
        //     &self.final_folding_randomness,
        // ));
        // result
    }
}

#[cfg(test)]
mod tests {
    use ark_ff::AdditiveGroup;

    use super::*;
    use crate::crypto::fields::Field64;

    #[test]
    fn test_compute_folds_helped_basic_case() {
        // Define a simple coefficient list with four values
        // This represents a polynomial over `{X1, X2}`
        let stir_challenges_answers = vec![
            Field64::from(1), // f(0,0)
            Field64::from(2), // f(0,1)
            Field64::from(3), // f(1,0)
            Field64::from(4), // f(1,1)
        ];

        // Define a simple coefficient list with four values
        // This represents a polynomial over `{X1, X2}`
        let final_randomness_answers = vec![
            Field64::from(5), // f(0,0)
            Field64::from(6), // f(0,1)
            Field64::from(7), // f(1,0)
            Field64::from(8), // f(1,1)
        ];

        // The folding randomness values `(5,6)` will be applied to interpolate the polynomial.
        // This means we are evaluating the polynomial at `X1=5, X2=6`.
        let folding_randomness = MultilinearPoint(vec![Field64::from(5), Field64::from(6)]);

        // Final folding randomness values `(55,66)` will be applied to compute the last fold.
        // This means we are evaluating the polynomial at `X1=55, X2=66`.
        let final_folding_randomness = MultilinearPoint(vec![Field64::from(55), Field64::from(66)]);

        let single_round = ParsedRound {
            folding_randomness,
            stir_challenges_answers: vec![vec![stir_challenges_answers]],
            ..Default::default()
        };

        let proof = ParsedProof {
            rounds: vec![single_round],
            final_folding_randomness,
            final_randomness_answers: vec![final_randomness_answers],
            ..Default::default()
        };

        let folds = proof.compute_folds_helped();

        // Expected first-round evaluation:
        // f(5,6) = 1 + 2(6) + 3(5) + 4(5)(6) = 148
        let expected_rounds = vec![CoefficientList::new(vec![
            Field64::from(1),
            Field64::from(2),
            Field64::from(3),
            Field64::from(4),
        ])
        .evaluate(&MultilinearPoint(vec![Field64::from(5), Field64::from(6)]))];

        // Expected final round evaluation:
        // f(55,66) = 5 + 6(66) + 7(55) + 8(55)(66) = 14718
        let expected_final_round = vec![CoefficientList::new(vec![
            Field64::from(5),
            Field64::from(6),
            Field64::from(7),
            Field64::from(8),
        ])
        .evaluate(&MultilinearPoint(vec![
            Field64::from(55),
            Field64::from(66),
        ]))];

        assert_eq!(folds, vec![expected_rounds, expected_final_round]);
    }

    #[test]
    fn test_compute_folds_helped_single_variable() {
        let stir_challenges_answers = vec![
            Field64::from(2), // f(0)
            Field64::from(5), // f(1)
        ];

        let folding_randomness = MultilinearPoint(vec![Field64::from(3)]); // Evaluating at X1=3

        let single_round = ParsedRound {
            folding_randomness,
            stir_challenges_answers: vec![vec![stir_challenges_answers]],
            ..Default::default()
        };

        let proof = ParsedProof {
            rounds: vec![single_round],
            final_folding_randomness: MultilinearPoint(vec![Field64::from(7)]), /* Evaluating at X1=7 */
            final_randomness_answers: vec![vec![Field64::from(8), Field64::from(10)]],
            ..Default::default()
        };

        let folds = proof.compute_folds_helped();

        // Compute expected evaluation at X1=3:
        // f(3) = 2 + 5(3) = 17
        let expected_rounds = vec![
            CoefficientList::new(vec![Field64::from(2), Field64::from(5)])
                .evaluate(&MultilinearPoint(vec![Field64::from(3)])),
        ];

        // Compute expected final round evaluation at X1=7:
        // f(7) = 8 + 10(7) = 78
        let expected_final_round = vec![CoefficientList::new(vec![
            Field64::from(8),
            Field64::from(10),
        ])
        .evaluate(&MultilinearPoint(vec![Field64::from(7)]))];

        assert_eq!(folds, vec![expected_rounds, expected_final_round]);
    }

    #[test]
    fn test_compute_folds_helped_all_zeros() {
        let stir_challenges_answers = vec![Field64::ZERO; 4];

        let proof = ParsedProof {
            rounds: vec![ParsedRound {
                folding_randomness: MultilinearPoint(vec![Field64::from(4), Field64::from(5)]),
                stir_challenges_answers: vec![vec![stir_challenges_answers.clone()]],
                ..Default::default()
            }],
            final_folding_randomness: MultilinearPoint(vec![Field64::from(10), Field64::from(20)]),
            final_randomness_answers: vec![stir_challenges_answers],
            ..Default::default()
        };

        let folds = proof.compute_folds_helped();

        // Since all coefficients are zero, every evaluation must be zero.
        assert_eq!(folds, vec![vec![Field64::ZERO], vec![Field64::ZERO]]);
    }
}
