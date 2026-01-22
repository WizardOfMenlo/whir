use ark_ff::Field;
use ark_std::rand::{CryptoRng, RngCore};
use serde::{Deserialize, Serialize};
use spongefish::{DuplexSpongeInterface, NargSerialize, VerificationError, VerificationResult};

use crate::{
    ensure,
    hash::{self, Hash},
    protocols::merkle_tree,
    transcript::{FieldConfig, ProtocolId, ProverMessage, ProverState, VerifierState},
};

/// Commits row-wise to a matrix of field elements using a merkle tree.
#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
#[serde(bound = "F: Field")]
pub struct Config<F: Field> {
    /// Field used for the matrix elements.
    pub field: FieldConfig<F>,

    /// The number of columns in the matrix.
    pub num_cols: usize,

    /// The hash function used for rows of the matrix.
    pub leaf_hash_id: ProtocolId,

    /// Merkle tree configuration for the matrix.
    pub merkle_tree: merkle_tree::Config,
}

pub type Witness = merkle_tree::Witness;

pub type Commitment = merkle_tree::Commitment;

impl<F: Field> Config<F> {
    pub fn new(num_rows: usize, num_cols: usize) -> Self {
        let field = FieldConfig::new();

        // Select a leaf hash function.
        let leaf_size = field.size_bytes() * num_cols;
        let leaf_hash_id = if leaf_size <= 32 {
            hash::COPY
        } else if hash::Blake3::supports_size(leaf_size) {
            hash::BLAKE3
        } else {
            hash::SHA2
        };

        Self {
            field,
            num_cols,
            leaf_hash_id,
            merkle_tree: merkle_tree::Config::new(num_rows),
        }
    }

    pub fn num_rows(&self) -> usize {
        self.merkle_tree.num_leaves
    }

    pub fn leaf_size_bytes(&self) -> usize {
        self.field.size_bytes() * self.num_cols
    }

    /// Commit the matrix (in row-major order).
    #[cfg_attr(feature = "tracing", instrument(skip(prover_state, matrix), fields(size = matrix.len(), engine)))]
    pub fn commit<H, R>(&self, prover_state: &mut ProverState<H, R>, matrix: &[F]) -> Witness
    where
        F: NargSerialize,
        H: DuplexSpongeInterface,
        R: RngCore + CryptoRng,
        Hash: ProverMessage<[H::U]>,
    {
        assert_eq!(matrix.len(), self.num_rows() * self.num_cols);

        let engine = hash::ENGINES
            .retrieve(self.leaf_hash_id)
            .expect("Failed to retrieve hash engine");
        #[cfg(feature = "tracing")]
        tracing::Span::current().record("engine", &engine.name());

        // Compute leaf hashes
        let mut leaves = Vec::with_capacity(self.merkle_tree.num_nodes());
        leaves.resize(self.merkle_tree.num_leaves, Hash::default());
        hash_rows(&*engine, matrix, &mut leaves[..self.num_rows()]);

        // Commit the leaf hashes
        self.merkle_tree.commit(prover_state, leaves)
    }

    pub fn receive_commitment<H>(
        &self,
        verifier_state: &mut VerifierState<H>,
    ) -> VerificationResult<Commitment>
    where
        H: DuplexSpongeInterface,
        Hash: ProverMessage<[H::U]>,
    {
        self.merkle_tree.receive_commitment(verifier_state)
    }

    /// Opens the commitment at the provided row indices.
    ///
    /// Indices can be in any order and may contain duplicates. The row values are not provided by
    /// this protocol, it is up to the caller to provide them to the verifier.
    #[cfg_attr(feature = "tracing", instrument(skip(prover_state, witness, leaves), fields(num_indices = indices.len())))]
    pub fn open<H, R>(
        &self,
        prover_state: &mut ProverState<H, R>,
        witness: &Witness,
        indices: &[usize],
    ) where
        H: DuplexSpongeInterface,
        R: RngCore + CryptoRng,
        Hash: ProverMessage<[H::U]>,
    {
        self.merkle_tree.open(prover_state, witness, indices);
    }

    /// Verifies the commitment at the provided row indices.
    ///
    /// Indices can be in any order and may contain duplicates. The row values are not provided by
    /// this protocol, it is up to the caller to provide them to the verifier.
    #[cfg_attr(feature = "tracing", instrument(skip(verifier_state, commitment, indices, matrix), fields(engine, num_indices = indices.len())))]
    pub fn verify<H>(
        &self,
        verifier_state: &mut VerifierState<H>,
        commitment: Commitment,
        indices: &[usize],
        matrix: &[F],
    ) -> VerificationResult<()>
    where
        F: NargSerialize,
        H: DuplexSpongeInterface,
        Hash: ProverMessage<[H::U]>,
    {
        ensure!(
            matrix.len() == self.num_cols * indices.len(),
            VerificationError
        );

        let engine = hash::ENGINES
            .retrieve(self.leaf_hash_id)
            .ok_or(VerificationError)?;
        #[cfg(feature = "tracing")]
        tracing::Span::current().record("engine", &engine.name());

        let mut leaf_hashes = vec![Hash::default(); indices.len()];
        hash_rows(&*engine, matrix, &mut leaf_hashes);
        self.merkle_tree
            .verify(verifier_state, commitment, indices, &leaf_hashes)
    }
}

#[cfg(not(feature = "parallel"))]
fn hash_rows<F: Field + NargSerialize>(
    engine: &dyn hash::Engine,
    matrix: &[F],
    mut out: &mut [Hash],
) {
    let cols = matrix.len() / out.len();
    let encoded_size = FieldConfig::<F>::new().size_bytes();
    let message_size = encoded_size * cols;
    let batch_size = engine.preferred_batch_size();

    let mut buffer = Vec::with_capacity(batch_size * cols * encoded_size);
    for chunk in matrix.chunks(batch_size * cols) {
        buffer.clear();
        for value in chunk {
            value.serialize_into_narg(&mut buffer);
        }
        let count = chunk.len() / cols;
        engine.hash_many(message_size, &buffer, &mut out[..count]);
        out = &mut out[count..];
    }
}

#[cfg(feature = "parallel")]
fn hash_rows<F: Field + NargSerialize>(
    engine: &dyn hash::Engine,
    matrix: &[F],
    mut out: &mut [Hash],
) {
    use crate::utils::workload_size;

    if matrix.len() > workload_size::<F>() && out.len() > engine.preferred_batch_size() {
        let split = (out.len() / 2).next_multiple_of(engine.preferred_batch_size());
        let m_split = (matrix.len() / out.len()) * split;
        let (mat_a, mat_b) = matrix.split_at(m_split);
        let (out_a, out_b) = out.split_at_mut(split);
        rayon::join(
            || hash_rows(engine, mat_a, out_a),
            || hash_rows(engine, mat_b, out_b),
        );
    } else {
        let cols = matrix.len() / out.len();
        let encoded_size = FieldConfig::<F>::new().size_bytes();
        let message_size = encoded_size * cols;
        let batch_size = engine.preferred_batch_size();

        use ark_ff::PrimeField;
        let modulus_bytes = (F::BasePrimeField::MODULUS_BIT_SIZE as usize + 7) / 8;

        let mut buffer = Vec::with_capacity(batch_size * cols * encoded_size);
        for chunk in matrix.chunks(batch_size * cols) {
            buffer.clear();
            for value in chunk {
                for coeff in value.to_base_prime_field_elements() {
                    use zerocopy::IntoBytes;

                    let bigint = coeff.into_bigint();
                    let limbs = bigint.as_ref();
                    let bytes = &limbs.as_bytes()[..modulus_bytes];
                    buffer.extend_from_slice(bytes);
                }

                // value.serialize_into_narg(&mut buffer);
            }
            let count = chunk.len() / cols;
            engine.hash_many(message_size, &buffer, &mut out[..count]);
            out = &mut out[count..];
        }
    }
}
