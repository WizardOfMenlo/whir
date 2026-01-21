use ark_crypto_primitives::merkle_tree::{Config, MerkleTree};

use crate::poly_utils::coeffs::CoefficientList;

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
pub struct Witness<F, MerkleConfig>
where
    MerkleConfig: Config,
{
    /// The committed polynomial in coefficient form. In case of batching, its
    /// the batched polynomial, i.e., the weighted sum of polynomials in
    /// batching data.
    pub(crate) polynomial: CoefficientList<F>,

    /// The Merkle tree constructed from the polynomial evaluations. In case of
    /// batching, it's the merkle tree of the batched polynomial.
    pub(crate) merkle_tree: MerkleTree<MerkleConfig>,

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

impl<F, MerkleConfig> Witness<F, MerkleConfig>
where
    MerkleConfig: Config,
{
    pub fn merkle_leaves_size(&self) -> usize {
        self.merkle_leaves.len() * std::mem::size_of::<F>()
    }

    pub fn polynomial_size(&self) -> usize {
        self.polynomial.num_coeffs() * std::mem::size_of::<F>()
    }

    pub fn merkle_tree_estimated_size(&self) -> usize {
        let height = self.merkle_tree.height();
        let num_leaves = 1usize << (height - 1);
        let num_non_leaves = num_leaves - 1;
        num_leaves * std::mem::size_of::<MerkleConfig::LeafDigest>()
            + num_non_leaves * std::mem::size_of::<MerkleConfig::InnerDigest>()
    }

    pub fn ood_size(&self) -> usize {
        (self.ood_points.len() + self.ood_answers.len()) * std::mem::size_of::<F>()
    }

    /// Clears the merkle_leaves to free memory. Must call `regenerate_merkle_leaves`
    /// before using this Witness in prove operations.
    pub fn clear_merkle_leaves(&mut self) {
        self.merkle_leaves = Vec::new();
    }

    /// Returns whether merkle_leaves have been cleared and need regeneration.
    pub fn needs_merkle_leaves(&self) -> bool {
        self.merkle_leaves.is_empty()
    }

    /// Sets the merkle_leaves. Used when regenerating after clearing.
    pub fn set_merkle_leaves(&mut self, leaves: Vec<F>) {
        self.merkle_leaves = leaves;
    }
}
