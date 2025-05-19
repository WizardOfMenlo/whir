use crate::poly_utils::multilinear::MultilinearPoint;

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
