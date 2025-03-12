use crate::ntt::transpose;
use ark_ff::Field;
use std::collections::BTreeSet;

// TODO(Gotti): n_bits is a misnomer if base > 2. Should be n_limbs or sth.
// Also, should the behaviour for value >= base^n_bits be specified as part of the API or asserted not to happen?
// Currently, we compute the decomposition of value % (base^n_bits).

/// decomposes value into its big-endian base-ary decomposition, meaning we return a vector v, s.t.
///
/// value = v[0]*base^(n_bits-1) + v[1] * base^(n_bits-2) + ... + v[n_bits-1] * 1,
/// where each v[i] is in 0..base.
/// The returned vector always has length exactly n_bits (we pad with leading zeros);
pub fn base_decomposition(value: usize, base: u8, n_bits: usize) -> Vec<u8> {
    // Initialize the result vector with zeros of the specified length
    let mut result = vec![0u8; n_bits];

    // Create a mutable copy of the value for computation
    // Note: We could just make the local passed-by-value argument `value` mutable, but this is clearer.
    let mut value = value;

    // Compute the base decomposition
    for i in 0..n_bits {
        result[n_bits - 1 - i] = (value % (base as usize)) as u8;
        value /= base as usize;
    }
    // TODO: Should we assert!(value == 0) here to check that the orginally passed `value` is < base^n_bits ?

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

/// Stacks evaluations by grouping them into cosets and transposing in-place.
///
/// Given `evals[i] = f(ω^i)`, reorganizes values into `2^folding_factor` cosets.
/// The transformation follows:
///
/// ```ignore
/// stacked[i, j] = f(ω^(i + j * (N / 2^folding_factor)))
/// ```
///
/// The input length must be a multiple of `2^folding_factor`.
pub fn stack_evaluations<F: Field>(mut evals: Vec<F>, folding_factor: usize) -> Vec<F> {
    let folding_factor_exp = 1 << folding_factor;
    assert!(evals.len() % folding_factor_exp == 0);
    let size_of_new_domain = evals.len() / folding_factor_exp;

    // Interpret evals as (folding_factor_exp x size_of_new_domain)-matrix and transpose in-place
    transpose(&mut evals, folding_factor_exp, size_of_new_domain);
    evals
}

#[cfg(test)]
mod tests {
    use crate::{crypto::fields::Field64, utils::base_decomposition};

    use super::stack_evaluations;

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
    fn test_stack_evaluations_basic() {
        // Basic test with 8 elements and folding factor of 2 (groups of 4)
        let evals: Vec<_> = (0..8).map(Field64::from).collect();
        let folding_factor = 2;

        let stacked = stack_evaluations(evals.clone(), folding_factor);

        // Check that the length remains unchanged after transformation
        assert_eq!(stacked.len(), evals.len());

        // Original matrix before stacking (4 rows, 2 columns):
        // 0  1
        // 2  3
        // 4  5
        // 6  7
        //
        // After transposition:
        // 0  2  4  6
        // 1  3  5  7
        let expected: Vec<_> = vec![0, 2, 4, 6, 1, 3, 5, 7]
            .into_iter()
            .map(Field64::from)
            .collect();

        assert_eq!(stacked, expected);
    }

    #[test]
    fn test_stack_evaluations_power_of_two() {
        // Test with 16 elements and a folding factor of 3 (groups of 8)
        let evals: Vec<_> = (0..16).map(Field64::from).collect();
        let folding_factor = 3;

        let stacked = stack_evaluations(evals.clone(), folding_factor);

        // Ensure the length remains unchanged
        assert_eq!(stacked.len(), evals.len());

        // Original matrix (8 rows, 2 columns):
        //  0   1
        //  2   3
        //  4   5
        //  6   7
        //  8   9
        // 10  11
        // 12  13
        // 14  15
        //
        // After stacking:
        //  0   2   4   6   8  10  12  14
        //  1   3   5   7   9  11  13  15
        let expected: Vec<_> = vec![0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15]
            .into_iter()
            .map(Field64::from)
            .collect();

        assert_eq!(stacked, expected);
    }

    #[test]
    fn test_stack_evaluations_identity_case() {
        // When folding_factor is 0, the function should return the input unchanged
        let evals: Vec<_> = (0..4).map(Field64::from).collect();
        let folding_factor = 0;

        let stacked = stack_evaluations(evals.clone(), folding_factor);

        // Expected result: No change
        assert_eq!(stacked, evals);
    }

    #[test]
    fn test_stack_evaluations_large_case() {
        // Test with 32 elements and a folding factor of 4 (groups of 16)
        let evals: Vec<_> = (0..32).map(Field64::from).collect();
        let folding_factor = 4;

        let stacked = stack_evaluations(evals.clone(), folding_factor);

        // Ensure the length remains unchanged
        assert_eq!(stacked.len(), evals.len());

        // Original matrix before stacking (16 rows, 2 columns):
        //  0   1
        //  2   3
        //  4   5
        //  6   7
        //  8   9
        // 10  11
        // 12  13
        // 14  15
        // 16  17
        // 18  19
        // 20  21
        // 22  23
        // 24  25
        // 26  27
        // 28  29
        // 30  31
        //
        // After stacking:
        //  0   2   4   6   8  10  12  14  16  18  20  22  24  26  28  30
        //  1   3   5   7   9  11  13  15  17  19  21  23  25  27  29  31
        let expected: Vec<_> = vec![
            0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 1, 3, 5, 7, 9, 11, 13, 15,
            17, 19, 21, 23, 25, 27, 29, 31,
        ]
        .into_iter()
        .map(Field64::from)
        .collect();

        assert_eq!(stacked, expected);
    }

    #[test]
    #[should_panic]
    fn test_stack_evaluations_invalid_size() {
        let evals: Vec<_> = (0..10).map(Field64::from).collect();
        let folding_factor = 2; // folding size = 4, but 10 is not divisible by 4
        stack_evaluations(evals, folding_factor);
    }

    #[test]
    fn test_base_decomposition() {
        assert_eq!(base_decomposition(0b1011, 2, 6), vec![0, 0, 1, 0, 1, 1]);
        assert_eq!(base_decomposition(15, 3, 3), vec![1, 2, 0]);
        // check truncation: This checks the current (undocumented) behaviour (compute modulo base^number_of_limbs) works as believed.
        // If we actually specify the API to have a different behaviour, this test should change.
        assert_eq!(base_decomposition(15 + 81, 3, 3), vec![1, 2, 0]);
    }
}
