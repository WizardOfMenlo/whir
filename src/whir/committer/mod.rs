use ark_crypto_primitives::merkle_tree::{Config, MerkleTree};

use crate::poly_utils::coeffs::CoefficientList;

pub mod reader;
pub mod writer;

pub use reader::CommitmentReader;
pub use writer::CommitmentWriter;

///
/// Represents the individual leaves  and merkle tree associated
/// with the polynomial.
///
#[derive(Clone)]
pub(crate) struct BatchingData<F, MerkleConfig>
where
    MerkleConfig: Config,
{
    pub(crate) merkle_tree: MerkleTree<MerkleConfig>,
    pub(crate) merkle_leaves: Vec<F>,
}

/// Represents the commitment and evaluation data for a polynomial.
///
/// This structure holds all necessary components to verify a commitment,
/// including the polynomial itself, the Merkle tree used for commitment,
/// and out-of-domain (OOD) evaluations.
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

    /// The individual commitments of each batching polynomial
    pub(crate) batching_data: Vec<BatchingData<F, MerkleConfig>>,

    /// The batching randomness. If there's no batching, this value is zero.
    pub(crate) batching_randomness: F,
}

impl<F, MerkleConfig> From<Witness<F, MerkleConfig>> for BatchingData<F, MerkleConfig>
where
    MerkleConfig: Config,
{
    fn from(value: Witness<F, MerkleConfig>) -> Self {
        Self {
            merkle_tree: value.merkle_tree,
            merkle_leaves: value.merkle_leaves,
        }
    }
}
