use crate::{
    algebra::{dot, geometric_sequence, poly_utils::coeffs::CoefficientList},
    protocols::irs_commit::{self, Evaluations},
    whir::statement::Weights,
};

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
    /// The committed polynomial in coefficient form in the extended field.
    /// In case of batching, its
    /// the batched polynomial, i.e., the weighted sum of polynomials in
    /// batching data.
    // TODO: Keep unbatched polynomials.
    pub(crate) polynomial: CoefficientList<F>,

    /// The witness to matrix commitment of the matrix containing the polynomial
    /// evaluations.
    pub(crate) witness: irs_commit::Witness<F::BasePrimeField, F>,

    /// The batching randomness. If there's no batching, this value is zero.
    // TODO: Move to prover opening step.
    pub batching_randomness: F,
}

/// Commitment parsed by the verifier from verifier's FS context.
#[derive(Clone)]
pub struct ParsedCommitment<F: Field> {
    pub commitment: irs_commit::Commitment<F>,
    pub batching_randomness: F,
}

pub fn constraints<F: Field>(
    evals: &Evaluations<F>,
    batching_randomness: F,
    num_variables: usize,
) -> Vec<(Weights<F>, F)> {
    let num_points = evals.points.len();
    let num_polynomials = evals.matrix.len() / num_points;
    let weights = geometric_sequence(batching_randomness, num_polynomials);
    evals
        .points
        .iter()
        .copied()
        .zip(evals.matrix.chunks_exact(num_polynomials))
        .map(|(point, evals)| {
            (
                Weights::univariate(point, num_variables),
                dot(&weights, evals),
            )
        })
        .collect()
}
