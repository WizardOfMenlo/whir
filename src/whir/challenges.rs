//! Extension traits for [`spongefish`] to generate challenges of specific forms.
// TODO: Maybe upstream to spongefish?

use ark_ff::Field;
use spongefish::{codecs::arkworks_algebra::UnitToField, ProofResult, UnitToBytes};

/// Helper functions for randomness generation on prover/verifier state.
pub trait ChallengeField<F: Field> {
    /// Generate a vector of uniform random field elements.
    fn challenge_vec(&mut self, len: usize) -> ProofResult<Vec<F>>;

    /// Generate a geometric sequence of 1, r, r^2, ..., r^(len-1)  where r is uniform a random field element.
    ///
    /// This requires fewer invocations of the sponge function than [`challenge_vec`].
    fn challenge_geometric_sequence(&mut self, out: &mut [F]) -> ProofResult<()>;

    /// Allocating helper for [`challenge_geometric_sequence`].
    fn challenge_geometric_vec(&mut self, len: usize) -> ProofResult<Vec<F>> {
        let mut res = vec![F::ZERO; len];
        self.challenge_geometric_sequence(&mut res)?;
        Ok(res)
    }
}

pub trait ChallengeIndices {
    /// Generates a list of sorted unique challenge indices.
    /// Indices are in range 0..`range`.
    /// Returns at most `len` values (potentially less due to deduplication).
    fn challenge_indices(&mut self, range: usize, len: usize) -> ProofResult<Vec<usize>>;
}

impl<F: Field, S: UnitToField<F>> ChallengeField<F> for S {
    fn challenge_vec(&mut self, len: usize) -> ProofResult<Vec<F>> {
        let mut res = vec![F::ZERO; len];
        if !res.is_empty() {
            self.fill_challenge_scalars(&mut res);
        }
        Ok(res)
    }

    fn challenge_geometric_sequence(&mut self, out: &mut [F]) -> ProofResult<()> {
        if out.len() <= 1 {
            out.fill(F::ONE);
            return Ok(());
        }
        let [base] = self.challenge_scalars()?;
        out[0] = F::ONE;
        out[1] = base;
        let mut acc = base;
        for out in &mut out[2..] {
            acc *= self.unit_to_field();
            *out = acc;
        }
        Ok(())
    }
}

impl<S: UnitToBytes> ChallengeIndices for S {
    fn challenge_indices(&mut self, range: usize, len: usize) -> ProofResult<Vec<usize>> {
        // TODO: check this
        let domain_size_bytes = ((range * 2 - 1).ilog2() as usize).div_ceil(8);

        // Allocate space for query bytes
        let mut queries = vec![0u8; len * domain_size_bytes];
        self.fill_challenge_bytes(&mut queries)?;

        // Convert bytes into indices in **one efficient pass**
        let indices = queries
            .chunks_exact(domain_size_bytes)
            .map(|chunk| chunk.iter().fold(0usize, |acc, &b| (acc << 8) | b as usize) % range)
            .sorted_unstable()
            .dedup()
            .collect_vec();

        Ok(indices)
    }
}

#[cfg(test)]
mod tests {
    use spongefish::DomainSeparatorMismatch;

    use super::*;
    use crate::whir::challenges::ChallengeIndices;

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

        let result = narg_string
            .challenge_indices(domain_size >> folding_factor, num_queries)
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

        let result = narg_string
            .challenge_indices(domain_size >> folding_factor, num_queries)
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

        let result = narg_string
            .challenge_indices(domain_size >> folding_factor, num_queries)
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

        let result = narg_string
            .challenge_indices(domain_size >> folding_factor, num_queries)
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
