use crate::ntt::transpose;
use ark_ff::Field;
use std::collections::BTreeSet;

// checks whether the given number n is a power of two.
pub const fn is_power_of_two(n: usize) -> bool {
    n != 0 && n.is_power_of_two()
}

// TODO(Gotti): n_bits is a misnomer if base > 2. Should be n_limbs or sth.
// Also, should the behaviour for value >= base^n_bits be specified as part of the API or asserted not to happen?
// Currently, we compute the decomposition of value % (base^n_bits).

/// Decomposes `value` into its base-`base` representation with `n_bits` digits.
/// The result follows big-endian order:
/// ```ignore
/// value = v[0] * base^(n_bits-1) + v[1] * base^(n_bits-2) + ... + v[n_bits-1] * 1
/// ```
/// where each `v[i]` is in `0..base`. Always returns exactly `n_bits` digits (padded with 0).
pub fn base_decomposition(mut value: usize, base: u8, n_bits: usize) -> Vec<u8> {
    debug_assert!(base > 1, "Base must be at least 2");

    // Preallocate a vector with zeroes, ensuring it's exactly `n_bits` long
    let mut result = vec![0u8; n_bits];

    // Compute the decomposition in reverse order (avoids shifting later)
    for digit in result.iter_mut().rev() {
        *digit = (value % (base as usize)) as u8;
        value /= base as usize;
    }

    result
}

// Gotti: Consider renaming this function. The name sounds like it's a PRG.
// TODO (Gotti): Check that ordering is actually correct at point of use (everything else is big-endian).

/// expand_randomness outputs the vector [1, base, base^2, base^3, ...] of length len.
pub fn expand_randomness<F: Field>(base: F, len: usize) -> Vec<F> {
    let mut res = Vec::with_capacity(len);
    let mut acc = F::ONE;
    for _ in 0..len {
        res.push(acc);
        acc *= base;
    }

    res
}

/// Deduplicates AND orders a vector
pub fn dedup<T: Ord>(v: impl IntoIterator<Item = T>) -> Vec<T> {
    Vec::from_iter(BTreeSet::from_iter(v))
}

// FIXME(Gotti): comment does not match what function does (due to mismatch between folding_factor and folding_factor_exp)
// Also, k should be defined: k = evals.len() / 2^{folding_factor}, I guess.

/// Takes the vector of evaluations (assume that evals[i] = f(omega^i))
/// and folds them into a vector of such that folded_evals[i] = [f(omega^(i + k * j)) for j in 0..folding_factor]
pub fn stack_evaluations<F: Field>(mut evals: Vec<F>, folding_factor: usize) -> Vec<F> {
    let folding_factor_exp = 1 << folding_factor;
    assert!(evals.len() % folding_factor_exp == 0);
    let size_of_new_domain = evals.len() / folding_factor_exp;

    // interpret evals as (folding_factor_exp x size_of_new_domain)-matrix and transpose in-place
    transpose(&mut evals, folding_factor_exp, size_of_new_domain);
    evals
}

#[cfg(test)]
mod tests {
    use crate::utils::base_decomposition;

    use super::{is_power_of_two, stack_evaluations};

    #[test]
    fn test_evaluations_stack() {
        use crate::crypto::fields::Field64 as F;

        let num = 256;
        let folding_factor = 3;
        let fold_size = 1 << folding_factor;
        assert_eq!(num % fold_size, 0);
        let evals: Vec<_> = (0..num as u64).map(F::from).collect();

        let stacked = stack_evaluations(evals, folding_factor);
        assert_eq!(stacked.len(), num);

        for (i, fold) in stacked.chunks_exact(fold_size).enumerate() {
            assert_eq!(fold.len(), fold_size);
            for (j, &f) in fold.iter().enumerate().take(fold_size) {
                assert_eq!(f, F::from((i + j * num / fold_size) as u64));
            }
        }
    }

    #[test]
    fn test_is_power_of_two() {
        assert!(!is_power_of_two(0));
        assert!(is_power_of_two(1));
        assert!(is_power_of_two(2));
        assert!(!is_power_of_two(3));
        assert!(!is_power_of_two(usize::MAX));
    }

    #[test]
    fn test_base_decomposition_binary() {
        // Base-2 decomposition (big-endian, padded to n_bits)
        // 11 in binary (6-bit representation): 001011
        assert_eq!(base_decomposition(0b1011, 2, 6), vec![0, 0, 1, 0, 1, 1]);

        // 5 in binary (4-bit representation): 0101
        assert_eq!(base_decomposition(5, 2, 4), vec![0, 1, 0, 1]);

        // 10 in binary (4-bit representation): 1010
        assert_eq!(base_decomposition(10, 2, 4), vec![1, 0, 1, 0]);

        // 0 in binary (4-bit representation): 0000
        assert_eq!(base_decomposition(0, 2, 4), vec![0, 0, 0, 0]);
    }

    #[test]
    fn test_base_decomposition_ternary() {
        // Base-3 decomposition (big-endian, padded to n_bits)
        // 15 in base-3 (3-digit representation): 120
        // Computation:
        // 15 / 3 = 5 remainder 0
        // 5 / 3  = 1 remainder 2
        // 1 / 3  = 0 remainder 1
        // Result = [1, 2, 0]
        assert_eq!(base_decomposition(15, 3, 3), vec![1, 2, 0]);

        // 8 in base-3 (4-digit representation): 0022
        // Computation:
        // 8 / 3 = 2 remainder 2
        // 2 / 3 = 0 remainder 2
        // Result = [0, 0, 2, 2] (padded to 4 bits)
        assert_eq!(base_decomposition(8, 3, 4), vec![0, 0, 2, 2]);
    }

    #[test]
    fn test_base_decomposition_large_values() {
        // Base-5 decomposition (4-digit representation of 123)
        // Computation:
        // 123 / 5 = 24 remainder 3
        // 24 / 5  = 4 remainder 4
        // 4 / 5   = 0 remainder 4
        // Result = [0, 4, 4, 3] (padded to 4 bits)
        assert_eq!(base_decomposition(123, 5, 4), vec![0, 4, 4, 3]);

        // Base-7 decomposition (5-digit representation of 100)
        // Computation:
        // 100 / 7 = 14 remainder 2
        // 14 / 7  = 2 remainder 0
        // 2 / 7   = 0 remainder 2
        // Result = [0, 0, 2, 0, 2] (padded to 5 bits)
        assert_eq!(base_decomposition(100, 7, 5), vec![0, 0, 2, 0, 2]);
    }

    #[test]
    fn test_base_decomposition_padding() {
        // Ensure correct padding when value is smaller than max base^n_bits
        // Base-4 decomposition (5 in base-4 with 5-bit representation)
        // 5 in base-4: 11 → padded to [0, 0, 0, 1, 1]
        assert_eq!(base_decomposition(5, 4, 5), vec![0, 0, 0, 1, 1]);

        // 2 in base-3 (4-bit representation): 0002
        assert_eq!(base_decomposition(2, 3, 4), vec![0, 0, 0, 2]);

        // 0 in base-5 (4-bit representation): 0000
        assert_eq!(base_decomposition(0, 5, 4), vec![0, 0, 0, 0]);
    }

    #[test]
    fn test_base_decomposition_edge_cases() {
        // Edge case: Maximum value within n_bits range
        // Base-2 decomposition of 15 in 4-bit representation: 1111
        assert_eq!(base_decomposition(15, 2, 4), vec![1, 1, 1, 1]);

        // Edge case: Maximum value within n_bits range for base-3
        // Base-3 decomposition of 26 in 3-bit representation (26 mod 3^3)
        // 26 in base-3: 222 → padded to 3 bits
        assert_eq!(base_decomposition(26, 3, 3), vec![2, 2, 2]);

        // Edge case: Maximum value that can fit in n_bits
        // Base-4 decomposition of 63 in 3-bit representation (63 mod 4^3)
        // 63 in base-4: 333 → padded to 3 bits
        assert_eq!(base_decomposition(63, 4, 3), vec![3, 3, 3]);
    }

    #[test]
    fn test_base_decomposition_truncation_behavior() {
        // Ensure truncation when value exceeds `base^n_bits`
        // (Undocumented but current behavior: computes modulo base^n_bits)
        //
        // Base-3 decomposition of (15 + 81) with 3-bit representation
        // (15 + 81) = 96, which is equivalent to (15 mod 27) = 15
        // 15 in base-3: 120
        assert_eq!(base_decomposition(15 + 81, 3, 3), vec![1, 2, 0]);

        // Base-2 decomposition of (20 + 16) with 4-bit representation
        // (20 + 16) = 36, which is equivalent to (20 mod 16) = 4
        // 4 in base-2: 0100
        assert_eq!(base_decomposition(20 + 16, 2, 4), vec![0, 1, 0, 0]);
    }

    #[test]
    #[should_panic]
    fn test_base_decomposition_invalid_base() {
        // Base cannot be 0 or 1 (should panic)
        base_decomposition(10, 0, 5);
    }
}
