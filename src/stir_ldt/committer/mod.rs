use ark_crypto_primitives::merkle_tree::{Config, MerkleTree};
use ark_ff::Field;
use ark_poly::univariate::DensePolynomial;

pub mod reader;
pub mod writer;

pub use reader::CommitmentReader;
pub use writer::CommitmentWriter;

pub struct Witness<F, MerkleConfig>
where
    F: Field,
    MerkleConfig: Config,
{
    /// The committed polynomial in coefficient form.
    pub(crate) polynomial: DensePolynomial<F>,
    /// The Merkle tree constructed from the polynomial evaluations.
    pub(crate) merkle_tree: MerkleTree<MerkleConfig>,
    /// The leaves of the Merkle tree, derived from folded polynomial evaluations.
    pub(crate) merkle_leaves: Vec<F>,
}
