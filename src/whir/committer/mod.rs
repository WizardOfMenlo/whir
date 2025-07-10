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
    /// The committed polynomial in coefficient form.
    pub(crate) polynomial: CoefficientList<F>,
    /// The Merkle tree constructed from the polynomial evaluations.
    pub(crate) merkle_tree: MerkleTree<MerkleConfig>,
    /// The leaves of the Merkle tree, derived from folded polynomial evaluations.
    pub(crate) merkle_leaves: Vec<F>,
    /// Out-of-domain challenge points used for polynomial verification.
    pub(crate) ood_points: Vec<F>,
    /// The corresponding polynomial evaluations at the OOD challenge points.
    pub(crate) ood_answers: Vec<F>,
}
