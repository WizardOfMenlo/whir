use ark_ff::Field;
use ark_poly::univariate::DensePolynomial;

/// Represents a single folding round in the STIR protocol.
///
/// This structure enables recursive compression and verification of a Reed–Solomon
/// proximity test under algebraic constraints.
#[derive(Debug, Clone)]
pub(crate) struct ParsedRound<F> {
    /// Folding randomness vector used in this round.
    pub(crate) r_fold: F,
    /// Out-of-domain query points.
    pub(crate) ood_points: Vec<F>,
    /// OOD answers at each query point.
    pub(crate) ood_evals: Vec<F>,
    /// Indexes of STIR constraint polynomials used in this round.
    pub(crate) r_shift_indexes: Vec<usize>,
    /// STIR constraint evaluation points.
    pub(crate) r_shift_points: Vec<F>,
    /// Answers to the STIR constraints at each evaluation point.
    pub(crate) r_shift_virtual_evals: Vec<Vec<F>>,
    /// Randomness used to linearly combine constraints.
    pub(crate) r_comb: F,
    /// Generator of L at round.
    pub(crate) domain_gen: F,
    /// Inverse of generator of L at round.
    pub(crate) domain_gen_inv: F,
    /// Offset of L at round.
    pub(crate) domain_offset: F,
    /// Inverse of offset of L at round.
    pub(crate) domain_offset_inv: F,
    /// Size of L^k at round.
    pub(crate) folded_domain_size: usize,
}

/// Represents a fully parsed and structured STIR proof.
///
/// The structure is designed to support recursive verification and evaluation
/// of folded functions under STIR-style constraints.
#[derive(Clone)]
pub(crate) struct ParsedProof<F: Field> {
    /// All folding rounds, each reducing the problem dimension.
    pub(crate) rounds: Vec<ParsedRound<F>>,
    /// Domain generator used in the final round.
    pub(crate) final_domain_gen: F,
    /// Inverse of the domain generator used in the final round.
    pub(crate) final_domain_gen_inv: F,
    /// Domain offset used in the final round.
    pub(crate) final_domain_offset: F,
    /// Inverse of the domain offset used in the final round.
    pub(crate) final_domain_offset_inv: F,
    /// Indexes of the final stir queries.
    pub(crate) final_r_shift_indexes: Vec<usize>,
    /// Evaluation points of the final stir queries.
    pub(crate) final_r_shift_points: Vec<F>,
    /// Evaluation results of the final stir queries.
    pub(crate) final_r_shift_virtual_evals: Vec<Vec<F>>,
    /// Folding randomness used in the final recursive step.
    pub(crate) final_r_fold: F,
    /// Coefficients of the final small polynomial.
    pub(crate) p_poly: DensePolynomial<F>,
}
