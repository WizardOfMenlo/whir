use ark_ff::FftField;

use super::stir_evaluations::StirEvalContext;
use crate::{
    poly_utils::{coeffs::CoefficientList, multilinear::MultilinearPoint},
    sumcheck::SumcheckPolynomial,
};

#[derive(Default, Debug, Clone)]
pub(crate) struct ParsedRound<F> {
    pub(crate) folding_randomness: MultilinearPoint<F>,
    pub(crate) ood_points: Vec<F>,
    pub(crate) ood_answers: Vec<F>,
    pub(crate) stir_challenges_indexes: Vec<usize>,
    pub(crate) stir_challenges_points: Vec<F>,
    pub(crate) stir_challenges_answers: Vec<Vec<F>>,
    pub(crate) combination_randomness: Vec<F>,
    pub(crate) sumcheck_rounds: Vec<(SumcheckPolynomial<F>, F)>,
    pub(crate) domain_gen_inv: F,
}

#[derive(Default, Clone)]
pub(crate) struct ParsedProof<F> {
    pub(crate) initial_combination_randomness: Vec<F>,
    pub(crate) initial_sumcheck_rounds: Vec<(SumcheckPolynomial<F>, F)>,
    pub(crate) rounds: Vec<ParsedRound<F>>,
    pub(crate) final_domain_gen_inv: F,
    pub(crate) final_randomness_indexes: Vec<usize>,
    pub(crate) final_randomness_points: Vec<F>,
    pub(crate) final_randomness_answers: Vec<Vec<F>>,
    pub(crate) final_folding_randomness: MultilinearPoint<F>,
    pub(crate) final_sumcheck_rounds: Vec<(SumcheckPolynomial<F>, F)>,
    pub(crate) final_sumcheck_randomness: MultilinearPoint<F>,
    pub(crate) final_coefficients: CoefficientList<F>,
    pub(crate) statement_values_at_random_point: Vec<F>,
}

impl<F: FftField> ParsedProof<F> {
    pub fn compute_folds_helped(&self) -> Vec<Vec<F>> {
        let mut result = Vec::with_capacity(self.rounds.len() + 1);

        for round in &self.rounds {
            let mut evals = Vec::with_capacity(round.stir_challenges_answers.len());

            let stir_evals_context =
                StirEvalContext::ProverHelps { folding_randomness: &round.folding_randomness };

            stir_evals_context.evaluate(&round.stir_challenges_answers, &mut evals);
            result.push(evals);
        }

        // Add final round
        let mut final_evals = Vec::with_capacity(self.final_randomness_answers.len());

        let stir_evals_context =
            StirEvalContext::ProverHelps { folding_randomness: &self.final_folding_randomness };
        stir_evals_context.evaluate(&self.final_randomness_answers, &mut final_evals);
        result.push(final_evals);
        result
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
            stir_challenges_answers: vec![stir_challenges_answers],
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
        .evaluate(&MultilinearPoint(vec![Field64::from(55), Field64::from(66)]))];

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
            stir_challenges_answers: vec![stir_challenges_answers],
            ..Default::default()
        };

        let proof = ParsedProof {
            rounds: vec![single_round],
            final_folding_randomness: MultilinearPoint(vec![Field64::from(7)]), /* Evaluating at
                                                                                 * X1=7 */
            final_randomness_answers: vec![vec![Field64::from(8), Field64::from(10)]],
            ..Default::default()
        };

        let folds = proof.compute_folds_helped();

        // Compute expected evaluation at X1=3:
        // f(3) = 2 + 5(3) = 17
        let expected_rounds = vec![CoefficientList::new(vec![Field64::from(2), Field64::from(5)])
            .evaluate(&MultilinearPoint(vec![Field64::from(3)]))];

        // Compute expected final round evaluation at X1=7:
        // f(7) = 8 + 10(7) = 78
        let expected_final_round =
            vec![CoefficientList::new(vec![Field64::from(8), Field64::from(10)])
                .evaluate(&MultilinearPoint(vec![Field64::from(7)]))];

        assert_eq!(folds, vec![expected_rounds, expected_final_round]);
    }

    #[test]
    fn test_compute_folds_helped_all_zeros() {
        let stir_challenges_answers = vec![Field64::ZERO; 4];

        let proof = ParsedProof {
            rounds: vec![ParsedRound {
                folding_randomness: MultilinearPoint(vec![Field64::from(4), Field64::from(5)]),
                stir_challenges_answers: vec![stir_challenges_answers.clone()],
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
