use ark_crypto_primitives::merkle_tree::{Config as MerkleConfig, MerkleTree};
use ark_ff::Field;

use crate::poly_utils::coeffs::CoefficientList;

pub mod reader;
pub mod writer;

pub use reader::CommitmentReader;
pub use writer::CommitmentWriter;

use super::statement::{Statement, Weights};

/// Represents the commitment and evaluation data for a set of polynomials.
///
/// This structure holds all necessary components to verify a commitment,
/// including the polynomials themselves, the Merkle tree used for commitment,
/// and out-of-domain (OOD) evaluations.
pub struct Witness<F: Field, M: MerkleConfig> {
    /// The committed polynomials in coefficient form.
    pub(crate) polynomials: Vec<CoefficientList<F>>,

    /// The Merkle tree constructed from the polynomial evaluations.
    /// A commitment creates one Merkle tree commiting to one or more polynomials.
    pub(crate) merkle_tree: MerkleTree<M>,

    /// The leaves of the Merkle tree, derived from folded polynomial
    /// evaluations.
    /// For multiple polynomials the leaves are interleaved.
    pub(crate) merkle_leaves: Vec<F>,

    /// Out-of-domain challenge points used for polynomial verification.
    /// The same point is used for all polynomials.
    pub(crate) ood_points: Vec<F>,

    /// The corresponding polynomial evaluations at the OOD challenge points.
    /// One for each polynomial and OOD point.
    pub(crate) ood_answers: Vec<F>,
}

impl<F: Field, M: MerkleConfig> Witness<F, M> {
    pub fn num_polynomials(&self) -> usize {
        self.polynomials.len()
    }

    pub fn num_variables(&self) -> usize {
        assert!(!self.polynomials.is_empty());
        self.polynomials[0].num_variables()
    }

    /// Returns the [`Statement`] for asserting the oods evaluations of each polynomial.
    pub fn oods_constraints(&self) -> Vec<Statement<F>> {
        self.ood_answers
            .chunks_exact(self.num_polynomials())
            .map(|answers| {
                let mut statement = Statement::new(self.num_variables());
                self.ood_points
                    .iter()
                    .map(|x| Weights::eval_univariate(self.num_variables(), *x))
                    .zip(answers)
                    .for_each(|(weights, answer)| statement.add_constraint(weights, *answer));
                statement
            })
            .collect()
    }
}
