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
    /// OOD answers at each query point.
    pub(crate) ood_answers: Vec<F>,
    /// Indexes of STIR constraint polynomials used in this round.
    pub(crate) stir_challenges_indexes: Vec<usize>,
    /// STIR constraint evaluation points.
    pub(crate) stir_challenges_points: Vec<F>,
    /// Answers to the STIR constraints at each evaluation point.
    pub(crate) stir_challenges_answers: Vec<Vec<F>>,
    /// Randomness used to linearly combine constraints.
    pub(crate) combination_randomness: Vec<F>,
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
    /// Initial random coefficients used to combine constraints before folding.
    pub(crate) initial_combination_randomness: Vec<F>,
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

    /// Deferred constraint evaluations.
    pub(crate) deferred: Vec<F>,
}
