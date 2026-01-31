//! Produce challenge indices from a transcript.

use crate::transcript::{Decoding, VerifierMessage};

/// Generate a set of indices for challenges.
pub fn challenge_indices<T>(
    transcript: &mut T,
    num_leaves: usize,
    count: usize,
    deduplicate: bool,
) -> Vec<usize>
where
    T: VerifierMessage,
    u8: Decoding<[T::U]>,
{
    if count == 0 {
        return Vec::new();
    }
    assert!(
        num_leaves.is_power_of_two(),
        "Number of leaves must be a power of two for unbiased results."
    );
    if num_leaves == 1 {
        // `size_bytes` would be zero, making `chunks_exact` panic.
        return if deduplicate { vec![0] } else { vec![0; count] };
    }

    // Calculate the required bytes of entropy
    // TODO: Round total to bytes, instead of per index.
    let size_bytes = (num_leaves.ilog2() as usize).div_ceil(8);

    // Get required entropy bits.
    let entropy: Vec<u8> = (0..count * size_bytes)
        .map(|_| transcript.verifier_message())
        .collect();

    // Convert bytes into indices
    let mut indices = entropy
        .chunks_exact(size_bytes)
        .map(|chunk| chunk.iter().fold(0usize, |acc, &b| (acc << 8) | b as usize) % num_leaves)
        .collect::<Vec<usize>>();

    // Sort and deduplicate indices if requested
    if deduplicate {
        indices.sort_unstable();
        indices.dedup();
    }
    indices
}
