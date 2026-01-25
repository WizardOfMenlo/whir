use crate::{algebra::poly_utils::coeffs::CoefficientList, protocols::matrix_commit};

pub mod reader;
pub mod writer;

pub use reader::CommitmentReader;
pub use writer::CommitmentWriter;

/// Represents the commitment and evaluation data for a polynomial.
///
/// This structure holds all necessary components to verify a commitment,
/// including the polynomial itself, the Merkle tree used for commitment,
/// and out-of-domain (OOD) evaluations.
#[derive(Clone)]
#[allow(clippy::struct_field_names)]
pub struct Witness<F> {
    /// The committed polynomial in coefficient form. In case of batching, its
    /// the batched polynomial, i.e., the weighted sum of polynomials in
    /// batching data.
    pub(crate) polynomial: CoefficientList<F>,

    /// The witness to matrix commitment of the matrix containing the polynomial
    /// evaluations.
    pub(crate) matrix_witness: matrix_commit::Witness,

    /// The leaves of the Merkle tree, derived from folded polynomial
    /// evaluations. In case of batching, its the merkle leaves of the batched
    /// tree. These leaves are computed as the weighted sum leaf values in the
    /// batching_data.
    pub(crate) merkle_leaves: Vec<F>,

    /// Out-of-domain challenge points used for polynomial verification.
    pub(crate) ood_points: Vec<F>,

    /// The corresponding polynomial evaluations at the OOD challenge points.
    pub(crate) ood_answers: Vec<F>,

    /// The batching randomness. If there's no batching, this value is zero.
    pub batching_randomness: F,
}
