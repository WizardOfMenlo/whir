use crate::poly_utils::{coeffs::CoefficientList, multilinear::MultilinearPoint};

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
    /// STIR constraint evaluation points.
    pub(crate) stir_challenges_points: Vec<F>,
    /// Randomness used to linearly combine constraints.
    pub(crate) combination_randomness: Vec<F>,
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
    /// Folding randomness used in the final recursive step.
    pub(crate) final_folding_randomness: MultilinearPoint<F>,
    /// Challenge vector used to evaluate the last polynomial.
    pub(crate) final_sumcheck_randomness: MultilinearPoint<F>,
    /// Coefficients of the final small polynomial.
    pub(crate) final_coefficients: CoefficientList<F>,

    /// Deferred constraint evaluations.
    pub(crate) deferred: Vec<F>,
}
