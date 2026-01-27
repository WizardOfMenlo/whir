//! Protocol for committing to rows of a matrix of some type <code>T: [Encodable]</code>.

use ark_ff::{Field, PrimeField};
use ark_std::rand::{CryptoRng, RngCore};
use serde::{Deserialize, Serialize};
use spongefish::{DuplexSpongeInterface, VerificationError, VerificationResult};
use static_assertions::assert_obj_safe;
#[cfg(feature = "tracing")]
use tracing::instrument;
use zerocopy::{Immutable, IntoBytes};

use crate::{
    ensure,
    hash::{self, Hash},
    protocols::merkle_tree,
    transcript::{ProtocolId, ProverMessage, ProverState, VerifierState},
    type_info::{Type, TypeInfo},
    utils::workload_size,
};

/// Trait for types that can be encoded into a byte slice.
pub trait Encodable {
    /// Exact size in bytes this type will encode in.
    fn encoded_size() -> usize;

    /// Encoder for this type.
    ///
    /// During leaf hashing, an encoder will be created per thread and re-used
    /// as much as possible.
    fn encoder() -> Box<dyn Encoder<Self>>;
}

/// Object-safe encoder for types that implement [`Encodable`].
pub trait Encoder<T> {
    /// Returns true if the encoder uses an internal buffer.
    ///
    /// When true the input sizes will be chunked to limit the
    /// required buffer size.
    fn is_buffered(&self) -> bool;

    /// Encodes a slice of values into a byte slice.
    ///
    /// The byte slice must be exactly [`values.len()`] times [`T::encoded_size()`] bytes long.
    ///
    /// It can be a reference to an internal buffer in the encoder, or a reference to the values.
    fn encode<'e, 'd>(&'e mut self, values: &'d [T]) -> &'e [u8]
    where
        'd: 'e;
}

assert_obj_safe!(Encoder<()>);

/// Encoder for [`ark_ff::Field`]s.
///
/// It encodes reduced values in little-endian byte order using the minimum number
/// of bytes that fit all field elements. For extensions,  the coefficients are
/// encoded in order of increasing degree.
pub struct ArkFieldEncoder(Vec<u8>);

/// Encoder for types that implement [`zerocopy`]'s [`Immutable`] and [`IntoBytes`].
///
/// This is the fastest encoder available for types that support it. To use it you
/// need to implement [`Encodable`] as follows:
///
/// ```
/// # use zerocopy::{Immutable, IntoBytes};
/// # use whir::protocols::matrix_commit::{Encoder, Encodable, ZeroCopyEncoder};
/// #[derive(Immutable, IntoBytes)]
/// pub struct MyType(u64);
///
/// impl Encodable for MyType {
///   fn encoded_size() -> usize {
///       std::mem::size_of::<Self>()
///   }
///
///   fn encoder() -> Box<dyn Encoder<Self>> {
///       Box::new(ZeroCopyEncoder)
///   }
/// }
///
/// ```
///
pub struct ZeroCopyEncoder;

/// Configuration for the matrix commit protocol.
///
/// Commits row-wise to a matrix of field elements using a merkle tree.
#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
#[serde(bound = "T: TypeInfo")]
pub struct Config<T>
where
    T: TypeInfo + Encodable + Send + Sync,
{
    /// Field used for the matrix elements.
    pub element_type: Type<T>,

    /// The number of columns in the matrix.
    pub num_cols: usize,

    /// The hash function used for rows of the matrix.
    pub leaf_hash_id: ProtocolId,

    /// Merkle tree configuration for the matrix.
    pub merkle_tree: merkle_tree::Config,
}

pub type Witness = merkle_tree::Witness;

pub type Commitment = merkle_tree::Commitment;

/// Encode [`ark_ff::Field`]s using [`ArkFieldEncoder`].
impl<F: Field> Encodable for F {
    fn encoded_size() -> usize {
        let base_bytes = (F::BasePrimeField::MODULUS_BIT_SIZE as usize).div_ceil(8);
        base_bytes * F::extension_degree() as usize
    }

    fn encoder() -> Box<dyn Encoder<Self>> {
        Box::new(ArkFieldEncoder(Vec::new()))
    }
}

impl<F: Field> Encoder<F> for ArkFieldEncoder {
    fn is_buffered(&self) -> bool {
        true
    }

    fn encode<'e, 'd>(&'e mut self, values: &'d [F]) -> &'e [u8]
    where
        'd: 'e,
    {
        let base_bytes = (F::BasePrimeField::MODULUS_BIT_SIZE as usize).div_ceil(8);
        self.0.clear();
        for value in values {
            for coeff in value.to_base_prime_field_elements() {
                // Convert to regular reduced form (e.g. not Montgomery encoded).
                let bigint = coeff.into_bigint();
                // Get the limbs of the bigint in little-endian order.
                let limbs = bigint.as_ref();
                // Get the bytes of the limbs in little-endian order, and strip trailing zeros.
                #[cfg(not(target_endian = "little"))]
                compile_error!("This crate requires a little-endian target.");
                let bytes = &limbs.as_bytes()[..base_bytes];
                self.0.extend_from_slice(bytes);
            }
        }
        self.0.as_ref()
    }
}

impl<T: Immutable + IntoBytes> Encoder<T> for ZeroCopyEncoder {
    fn is_buffered(&self) -> bool {
        false
    }

    fn encode<'e, 'd>(&'e mut self, values: &'d [T]) -> &'e [u8]
    where
        'd: 'e,
    {
        values.as_bytes()
    }
}

impl<T: TypeInfo + Encodable + Send + Sync> Config<T> {
    /// Create a new matrix commit configuration with the recommended hash function.
    pub fn new(num_rows: usize, num_cols: usize) -> Self {
        // Select a leaf hash function.
        let leaf_size = T::encoded_size() * num_cols;
        let leaf_hash_id = if leaf_size <= 32 {
            hash::COPY
        } else if hash::Blake3::supports_size(leaf_size) {
            hash::BLAKE3
        } else {
            hash::SHA2
        };

        Self {
            element_type: Type::new(),
            num_cols,
            leaf_hash_id,
            merkle_tree: merkle_tree::Config::new(num_rows),
        }
    }

    pub fn with_hash(hash_id: ProtocolId, num_rows: usize, num_cols: usize) -> Self {
        Self {
            element_type: Type::new(),
            num_cols,
            leaf_hash_id: hash_id,
            merkle_tree: merkle_tree::Config::with_hash(hash_id, num_rows),
        }
    }

    pub const fn num_rows(&self) -> usize {
        self.merkle_tree.num_leaves
    }

    /// Commit the matrix (in row-major order).
    #[cfg_attr(
        feature = "tracing",
        instrument(skip(self, prover_state, matrix), fields(size = matrix.len(), engine))
    )]
    pub fn commit<H, R>(&self, prover_state: &mut ProverState<H, R>, matrix: &[T]) -> Witness
    where
        H: DuplexSpongeInterface,
        R: RngCore + CryptoRng,
        Hash: ProverMessage<[H::U]>,
    {
        assert_eq!(matrix.len(), self.num_rows() * self.num_cols);

        let engine = hash::ENGINES
            .retrieve(self.leaf_hash_id)
            .expect("Failed to retrieve hash engine");
        #[cfg(feature = "tracing")]
        tracing::Span::current().record("engine", &*engine.name());

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
    #[cfg_attr(
        feature = "tracing",
        instrument(skip(self, prover_state, witness), fields(num_indices = indices.len()))
    )]
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
    #[cfg_attr(
        feature = "tracing",
        instrument(
            skip(self, verifier_state, commitment, indices, matrix),
            fields(engine, num_indices = indices.len())
        )
    )]
    pub fn verify<H>(
        &self,
        verifier_state: &mut VerifierState<H>,
        commitment: Commitment,
        indices: &[usize],
        matrix: &[T],
    ) -> VerificationResult<()>
    where
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
        tracing::Span::current().record("engine", &*engine.name());

        let mut leaf_hashes = vec![Hash::default(); indices.len()];
        hash_rows(&*engine, matrix, &mut leaf_hashes);
        self.merkle_tree
            .verify(verifier_state, commitment, indices, &leaf_hashes)
    }
}

#[cfg(not(feature = "parallel"))]
fn hash_rows<T: Encodable + Send + Sync>(
    engine: &dyn hash::Engine,
    matrix: &[T],
    out: &mut [Hash],
) {
    hash_rows_serial(engine, matrix, out);
}

#[cfg(feature = "parallel")]
fn hash_rows<T: Encodable + Send + Sync>(
    engine: &dyn hash::Engine,
    matrix: &[T],
    out: &mut [Hash],
) {
    let Some(cols) = matrix.len().checked_div(out.len()) else {
        // Empty input and output
        return;
    };
    if matrix.len() > workload_size::<T>() && out.len() > engine.preferred_batch_size() {
        // Large task, split into parallel threads.
        let split = (out.len() / 2).next_multiple_of(engine.preferred_batch_size());
        let (mat_a, mat_b) = matrix.split_at(split * cols);
        let (out_a, out_b) = out.split_at_mut(split);
        rayon::join(
            || hash_rows(engine, mat_a, out_a),
            || hash_rows(engine, mat_b, out_b),
        );
    } else {
        hash_rows_serial(engine, matrix, out);
    }
}

fn hash_rows_serial<T: Encodable + Send + Sync>(
    engine: &dyn hash::Engine,
    matrix: &[T],
    out: &mut [Hash],
) {
    assert!(matrix.len().is_multiple_of(out.len()));
    let Some(cols) = matrix.len().checked_div(out.len()) else {
        // Empty input and output
        return;
    };
    let message_size = T::encoded_size() * cols;
    let mut encoder = T::encoder();
    if encoder.is_buffered() {
        // Buffered encoder, find some optimal size.
        let target = workload_size::<u8>() / 8;
        let batch_size = (target / message_size).next_multiple_of(engine.preferred_batch_size());
        assert!(batch_size >= 1);
        for (matrix, out) in matrix
            .chunks(batch_size * cols)
            .zip(out.chunks_mut(batch_size))
        {
            let bytes = encoder.encode(matrix);
            engine.hash_many(message_size, bytes, out);
        }
    } else {
        // Unbuffered, encode everything in one go
        let bytes = encoder.encode(matrix);
        engine.hash_many(message_size, bytes, out);
    }
}
