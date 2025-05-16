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
    /// The same points are used for all polynomials.
    pub(crate) ood_points: Vec<F>,

    /// The corresponding polynomial evaluations at the OOD challenge points.
    /// One for each polynomial and OOD point.
    pub(crate) ood_answers: Vec<F>,
}

/// Represents the commitment data for a set of polynomials.
#[derive(Clone)]
pub struct Commitment<F: Field, D: Sized> {
    /// The number of variables each multilinear polynomial has.
    pub(crate) num_variables: usize,

    /// The root of the Merkle tree committing to all the polynomials.
    pub(crate) root: D,

    /// Out-of-domain challenge points used for polynomial verification.
    /// The same points are used for all polynomials.
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

    /// Total number of constraints on all polynomials.
    pub fn num_constraints(&self) -> usize {
        self.ood_answers.len()
    }

    pub fn coset_size(&self) -> usize {
        1_usize << (self.merkle_tree.height() - 1)
    }

    /// Returns the [`Statement`] for asserting the oods evaluations of each polynomial.
    pub fn oods_constraints(&self) -> Vec<Statement<F>> {
        Commitment::from(self).oods_constraints()
    }

    /// Return an iterator over the `index`-th cosets of the polynomials
    pub fn cosets(&self, index: usize) -> impl Iterator<Item = &[F]> {
        let leaf_size = self.coset_size() * self.num_polynomials();
        let cosets = &self.merkle_leaves[index * leaf_size..(index + 1) * leaf_size];
        debug_assert_eq!(cosets.len() % self.coset_size(), 0);
        cosets.chunks_exact(self.coset_size())
    }
}

impl<F: Field, D: Sized> Commitment<F, D> {
    pub fn num_polynomials(&self) -> usize {
        self.ood_answers.len() / self.ood_points.len()
    }

    pub fn num_variables(&self) -> usize {
        self.num_variables
    }

    /// Total number of constraints on all polynomials.
    pub fn num_constraints(&self) -> usize {
        self.ood_answers.len()
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

impl<F, M, D> From<&Witness<F, M>> for Commitment<F, D>
where
    F: Field,
    M: MerkleConfig<InnerDigest = D>,
{
    fn from(value: &Witness<F, M>) -> Self {
        Self {
            num_variables: value.num_variables(),
            root: value.merkle_tree.root(),
            ood_points: value.ood_points.clone(),
            ood_answers: value.ood_answers.clone(),
        }
    }
}
