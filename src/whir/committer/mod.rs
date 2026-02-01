use crate::{algebra::poly_utils::coeffs::CoefficientList, protocols::irs_commit};

mod reader;
mod writer;

use ark_ff::{FftField, Field};
pub use reader::CommitmentReader;
pub use writer::CommitmentWriter;

/// Represents the commitment and evaluation data for a polynomial.
///
/// This structure holds all necessary components to verify a commitment,
/// including the polynomial itself, the Merkle tree used for commitment,
/// and out-of-domain (OOD) evaluations.
#[derive(Clone)]
#[allow(clippy::struct_field_names)]
pub struct Witness<F: FftField> {
    /// The committed polynomial in coefficient form in the base field.
    pub(crate) polynomials: Vec<CoefficientList<F::BasePrimeField>>,

    /// The witness to matrix commitment of the matrix containing the polynomial
    /// evaluations.
    pub(crate) witness: irs_commit::Witness<F::BasePrimeField, F>,
}

/// Commitment parsed by the verifier from verifier's FS context.
#[derive(Clone, Debug)]
pub struct ParsedCommitment<F: Field> {
    pub commitment: irs_commit::Commitment<F>,
}
