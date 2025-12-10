use ark_crypto_primitives::merkle_tree::Config;
use ark_ff::FftField;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use itertools::Itertools;
use spongefish::{
    codecs::arkworks_algebra::{FieldToUnitSerialize, UnitToField},
    ProofResult, UnitToBytes,
};
#[cfg(feature = "tracing")]
use tracing::instrument;

use crate::{parameters::DeduplicationStrategy, poly_utils::multilinear::MultilinearPoint};

///
/// A utility function to compute the response to OOD challenge and add it to
/// the transcript. The OOD challenge should have already been sampled and added
/// to the transcript before this call.
///
#[cfg_attr(feature = "tracing", instrument(skip(prover_state, evaluate_fn)))]
pub(crate) fn compute_ood_response<F, ProverState, E>(
    prover_state: &mut ProverState,
    ood_points: &[F],
    num_variables: usize,
    evaluate_fn: E,
) -> ProofResult<Vec<F>>
where
    F: FftField,
    ProverState: FieldToUnitSerialize<F>,
    E: Fn(&MultilinearPoint<F>) -> F,
{
    let num_samples = ood_points.len();
    let mut ood_answers = Vec::<F>::with_capacity(ood_points.len());

    if num_samples > 0 {
        // Evaluate the function at each OOD point
        ood_answers.extend(ood_points.iter().map(|ood_point| {
            evaluate_fn(&MultilinearPoint::expand_from_univariate(
                *ood_point,
                num_variables,
            ))
        }));

        // Commit the answers to the narg_string
        prover_state.add_scalars(&ood_answers)?;
    }

    Ok(ood_answers)
}

/// A utility function to sample Out-of-Domain (OOD) points and evaluate them
///
/// This operates on the prover side.
#[cfg_attr(feature = "tracing", instrument(skip(prover_state, evaluate_fn)))]
pub fn sample_ood_points<F, ProverState, E>(
    prover_state: &mut ProverState,
    num_samples: usize,
    num_variables: usize,
    evaluate_fn: E,
) -> ProofResult<(Vec<F>, Vec<F>)>
where
    F: FftField,
    ProverState: FieldToUnitSerialize<F> + UnitToField<F>,
    E: Fn(&MultilinearPoint<F>) -> F,
{
    let mut ood_points = vec![F::ZERO; num_samples];
    let ood_answers = if num_samples > 0 {
        // Generate OOD points from ProverState randomness
        prover_state.fill_challenge_scalars(&mut ood_points)?;
        compute_ood_response(prover_state, &ood_points, num_variables, evaluate_fn)?
    } else {
        vec![]
    };

    Ok((ood_points, ood_answers))
}

/// Generates a list of unique challenge queries within a folded domain.
///
/// Given a `domain_size` and `folding_factor`, this function:
/// - Computes the folded domain size: `folded_domain_size = domain_size / 2^folding_factor`.
/// - Derives query indices from random narg_string bytes.
/// - Deduplicates indices while preserving order.
pub fn get_challenge_stir_queries<T>(
    domain_size: usize,
    folding_factor: usize,
    num_queries: usize,
    narg_string: &mut T,
    deduplication_strategy: &DeduplicationStrategy,
) -> ProofResult<Vec<usize>>
where
    T: UnitToBytes,
{
    let folded_domain_size = domain_size >> folding_factor;
    // Compute required bytes per index: `domain_size_bytes = ceil(log2(folded_domain_size) / 8)`
    let domain_size_bytes = ((folded_domain_size * 2 - 1).ilog2() as usize).div_ceil(8);

    // Allocate space for query bytes
    let mut queries = vec![0u8; num_queries * domain_size_bytes];
    narg_string.fill_challenge_bytes(&mut queries)?;

    // Convert bytes into indices in **one efficient pass**
    let mut indices = queries
        .chunks_exact(domain_size_bytes)
        .map(|chunk| {
            chunk.iter().fold(0usize, |acc, &b| (acc << 8) | b as usize) % folded_domain_size
        })
        .collect_vec();

    match deduplication_strategy {
        DeduplicationStrategy::Enabled => {
            indices.sort_unstable();
            indices.dedup();
        }
        DeduplicationStrategy::Disabled => {}
    }

    Ok(indices)
}

pub trait DigestToUnitSerialize<MerkleConfig: Config> {
    fn add_digest(&mut self, digest: MerkleConfig::InnerDigest) -> ProofResult<()>;
}

pub trait DigestToUnitDeserialize<MerkleConfig: Config> {
    fn read_digest(&mut self) -> ProofResult<MerkleConfig::InnerDigest>;
}

pub fn rlc_batched_leaves<F: ark_ff::Field>(
    leaves: Vec<Vec<F>>,
    fold_size: usize,
    batch_size: usize,
    batching_randomness: F,
) -> Vec<Vec<F>> {
    leaves
        .into_iter()
        .map(|leaf| {
            assert_eq!(leaf.len(), batch_size * fold_size);
            let mut out = vec![F::ZERO; fold_size];
            let mut pow = F::ONE;
            for block in leaf.chunks_exact(fold_size).take(batch_size) {
                for (o, v) in out.iter_mut().zip(block) {
                    *o += pow * *v;
                }
                pow *= batching_randomness;
            }
            out
        })
        .collect()
}

pub trait HintSerialize {
    fn hint<T: CanonicalSerialize>(&mut self, hint: &T) -> ProofResult<()>;
}

pub trait HintDeserialize {
    fn hint<T: CanonicalDeserialize>(&mut self) -> ProofResult<T>;
}

#[cfg(test)]
mod tests {
    use spongefish::DomainSeparatorMismatch;

    use super::*;

    struct MockTranscript {
        data: Vec<u8>,
        index: usize,
    }

    impl UnitToBytes for MockTranscript {
        fn fill_challenge_bytes(
            &mut self,
            buffer: &mut [u8],
        ) -> Result<(), DomainSeparatorMismatch> {
            if self.index + buffer.len() > self.data.len() {
                return Err(String::from("Invalid input").into());
            }
            buffer.copy_from_slice(&self.data[self.index..self.index + buffer.len()]);
            self.index += buffer.len();
            Ok(())
        }
    }

    #[test]
    fn test_challenge_stir_queries_single_byte_indices() {
        // Case where `domain_size_bytes = 1`, meaning all indices are computed using a single byte.
        let domain_size = 256;
        let folding_factor = 1;
        let num_queries = 5;

        // Mock narg_string with fixed bytes (ensuring reproducibility)
        let narg_string_data = vec![
            0x01, 0x23, 0x45, 0x67, 0x89, // Query 1
            0xAB, 0xCD, 0xEF, 0x12, 0x34, // Query 2
            0x56, 0x78, 0x9A, 0xBC, 0xDE, // Query 3
            0xF0, 0x11, 0x22, 0x33, 0x44, // Query 4
            0x55, 0x66, 0x77, 0x88, 0x99, // Query 5
        ];
        let mut narg_string = MockTranscript {
            data: narg_string_data,
            index: 0,
        };

        let result = get_challenge_stir_queries(
            domain_size,
            folding_factor,
            num_queries,
            &mut narg_string,
            &DeduplicationStrategy::Enabled,
        )
        .unwrap();

        let folded_domain_size = 128; // domain_size / 2

        // Manually computed expected indices
        let index_0 = 0x01 % folded_domain_size;
        let index_1 = 0x23 % folded_domain_size;
        let index_2 = 0x45 % folded_domain_size;
        let index_3 = 0x67 % folded_domain_size;
        let index_4 = 0x89 % folded_domain_size;

        let mut expected_indices = vec![index_0, index_1, index_2, index_3, index_4];
        expected_indices.sort_unstable();

        assert_eq!(
            result, expected_indices,
            "Mismatch in computed indices for domain_size_bytes = 1"
        );
    }

    #[test]
    fn test_challenge_stir_queries_two_byte_indices() {
        // Case where `domain_size_bytes = 2`, meaning indices are computed using two bytes.
        let domain_size = 65536; // 2^16
        let folding_factor = 3; // 2^3 = 8
        let num_queries = 5;

        // Expected `folded_domain_size = 65536 / 8 = 8192`
        let narg_string_data = vec![
            0x01, 0x23, 0x45, 0x67, 0x89, // Query 1
            0xAB, 0xCD, 0xEF, 0x12, 0x34, // Query 2
            0x56, 0x78, 0x9A, 0xBC, 0xDE, // Query 3
            0xF0, 0x11, 0x22, 0x33, 0x44, // Query 4
            0x55, 0x66, 0x77, 0x88, 0x99, // Query 5
        ];
        let mut narg_string = MockTranscript {
            data: narg_string_data,
            index: 0,
        };

        let result = get_challenge_stir_queries(
            domain_size,
            folding_factor,
            num_queries,
            &mut narg_string,
            &DeduplicationStrategy::Enabled,
        )
        .unwrap();

        let folded_domain_size = 8192; // 65536 / 8

        // Manually computed expected indices using two bytes per index
        let index_0 = ((0x01 << 8) | 0x23) % folded_domain_size;
        let index_1 = ((0x45 << 8) | 0x67) % folded_domain_size;
        let index_2 = ((0x89 << 8) | 0xAB) % folded_domain_size;
        let index_3 = ((0xCD << 8) | 0xEF) % folded_domain_size;
        let index_4 = ((0x12 << 8) | 0x34) % folded_domain_size;

        let mut expected_indices = vec![index_0, index_1, index_2, index_3, index_4];
        expected_indices.sort_unstable();

        assert_eq!(
            result, expected_indices,
            "Mismatch in computed indices for domain_size_bytes = 2"
        );
    }

    #[test]
    fn test_challenge_stir_queries_three_byte_indices() {
        // Case where `domain_size_bytes = 3`, meaning indices are computed using three bytes.
        let domain_size = 2usize.pow(24); // 16,777,216
        let folding_factor = 4; // 2^4 = 16
        let num_queries = 4;

        // Expected `folded_domain_size = 2^24 / 16 = 2^20 = 1,048,576`
        let narg_string_data = vec![
            0x12, 0x34, 0x56, // Query 1
            0x78, 0x9A, 0xBC, // Query 2
            0xDE, 0xF0, 0x11, // Query 3
            0x22, 0x33, 0x44, // Query 4
        ];
        let mut narg_string = MockTranscript {
            data: narg_string_data,
            index: 0,
        };

        let result = get_challenge_stir_queries(
            domain_size,
            folding_factor,
            num_queries,
            &mut narg_string,
            &DeduplicationStrategy::Enabled,
        )
        .unwrap();

        let folded_domain_size = 1_048_576; // 2^20

        // Manually computed expected indices using three bytes per index
        let index_0 = ((0x12 << 16) | (0x34 << 8) | 0x56) % folded_domain_size;
        let index_1 = ((0x78 << 16) | (0x9A << 8) | 0xBC) % folded_domain_size;
        let index_2 = ((0xDE << 16) | (0xF0 << 8) | 0x11) % folded_domain_size;
        let index_3 = ((0x22 << 16) | (0x33 << 8) | 0x44) % folded_domain_size;

        let mut expected_indices = vec![index_0, index_1, index_2, index_3];
        expected_indices.sort_unstable();

        assert_eq!(
            result, expected_indices,
            "Mismatch in computed indices for domain_size_bytes = 3"
        );
    }

    #[test]
    fn test_challenge_stir_queries_duplicate_indices() {
        // Case where the function should deduplicate indices
        let domain_size = 128;
        let folding_factor = 0;
        let num_queries = 5;

        // Mock narg_string where some indices will collide
        let narg_string_data = vec![
            0x20, 0x40, 0x20, 0x60, 0x40, // Duplicate indices 0x20 and 0x40
        ];
        let mut narg_string = MockTranscript {
            data: narg_string_data,
            index: 0,
        };

        let result = get_challenge_stir_queries(
            domain_size,
            folding_factor,
            num_queries,
            &mut narg_string,
            &DeduplicationStrategy::Enabled,
        )
        .unwrap();

        let folded_domain_size = 128;

        // Manually computed expected indices, ensuring duplicates are removed
        let mut expected_indices = vec![
            0x20 % folded_domain_size,
            0x40 % folded_domain_size,
            0x60 % folded_domain_size,
        ];
        expected_indices.sort_unstable();

        assert_eq!(
            result, expected_indices,
            "Mismatch in computed indices for deduplication test"
        );
    }
}
