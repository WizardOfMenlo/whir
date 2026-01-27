use crate::{
    algebra::{dot, geometric_sequence, poly_utils::coeffs::CoefficientList},
    protocols::irs_commit,
    whir::statement::{Constraint, Weights},
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
    pub(crate) polynomial_size: usize,
    pub(crate) num_polynomials: usize,

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
    pub(crate) polynomial_size: usize,
    pub(crate) num_polynomials: usize,
    pub commitment: irs_commit::Commitment<F>,
    pub batching_randomness: F,
}

impl<F: FftField> Witness<F> {
    /// Return the matrix in the extension field (currentlyneeded for prover).
    pub fn matrix(&self) -> Vec<F> {
        self.witness
            .matrix
            .iter()
            .copied()
            .map(F::from_base_prime_field)
            .collect()
    }

    pub fn ood_points(&self) -> Vec<F> {
        self.witness
            .out_of_domain()
            .0
            .iter()
            .map(|(point, _)| point)
            .copied()
            .collect()
    }

    pub fn oods_constraints(&self) -> Vec<(Weights<F>, F)> {
        let batch_weights = geometric_sequence(self.batching_randomness, self.num_polynomials);
        self.witness
            .out_of_domain()
            .constraints(&batch_weights, self.polynomial_size)
            .collect()
    }
}

impl<F: FftField> ParsedCommitment<F> {
    /// Return constraints for OODS
    pub fn oods_constraints(&self) -> Vec<Constraint<F>> {
        let batch_weights = geometric_sequence(self.batching_randomness, self.num_polynomials);
        self.commitment
            .out_of_domain()
            .constraints(&batch_weights, self.polynomial_size)
            .map(|(weights, sum)| Constraint {
                weights,
                sum,
                defer_evaluation: false,
            })
            .collect::<Vec<_>>()
    }
}
