use crate::ntt::transpose;
use ark_ff::Field;
use std::collections::BTreeSet;

// checks whether the given number n is a power of two.
//
// You must not call this with n == 0.
pub fn is_power_of_two(n: usize) -> bool {
    // BUG: This underflows if n == 0, causing a panic on release builds and returns true on debug build.
    n & (n - 1) == 0
}

// performs big-endian binary decomposition of value and returns the result.
//
// The returned vector v starts with the big-endian bits of value and always has exactly n_bits many elements.
// n_bits must be at must usize::BITS. If it is strictly smaller, the relevant higher-order bits of value are ignored.
pub fn to_binary(value: usize, n_bits: usize) -> Vec<bool> {
    // Ensure that n is within the bounds of the input integer type
    assert!(n_bits <= usize::BITS as usize);
    let mut result = vec![false; n_bits];
    for i in 0..n_bits {
        result[n_bits - 1 - i] = (value & (1 << i)) != 0;
    }
    result
}


// TODO: n_bits is a misnomer if base > 2. Should be n_limbs or sth.

// decomposes value into its base-ary decomposition, meaning we return a vector v, s.t.
//
// value = v[0] + v[1] * base + v[2] * base^2 + ... + v[n_bits-1] * base^(n_bits-1),
// where each v[i] is in 0..base.
// The returned vector always has length exactly n_bits (we pad with leading zeros);
// if value >= base^n_bits, we truncate, effectively computing value % (base^n_bits)
pub fn base_decomposition(value: usize, base: u8, n_bits: usize) -> Vec<u8> {
    // Initialize the result vector with zeros of the specified length
    let mut result = vec![0u8; n_bits];

    // Create a mutable copy of the value for computation
    let mut value = value;

    // Compute the base decomposition
    for i in 0..n_bits {
        result[n_bits - 1 - i] = (value % (base as usize)) as u8;
        value /= base as usize;
    }

    result
}

pub fn expand_randomness<F: Field>(base: F, len: usize) -> Vec<F> {
    let mut res = Vec::with_capacity(len);
    let mut acc = F::ONE;
    for _ in 0..len {
        res.push(acc);
        acc *= base;
    }

    res
}

// Deduplicates AND orders a vector
pub fn dedup<T: Ord>(v: impl IntoIterator<Item = T>) -> Vec<T> {
    Vec::from_iter(BTreeSet::from_iter(v))
}

// Takes the vector of evaluations (assume that evals[i] = f(omega^i))
// and folds them into a vector of such that folded_evals[i] = [f(omega^(i + k * j)) for j in 0..folding_factor]
pub fn stack_evaluations<F: Field>(mut evals: Vec<F>, folding_factor: usize) -> Vec<F> {
    let folding_factor_exp = 1 << folding_factor;
    assert!(evals.len() % folding_factor_exp == 0);
    let size_of_new_domain = evals.len() / folding_factor_exp;
    transpose(&mut evals, folding_factor_exp, size_of_new_domain);
    evals
}

#[cfg(test)]
mod tests {
    use super::{stack_evaluations, to_binary};

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
            for j in 0..fold_size {
                assert_eq!(fold[j], F::from((i + j * num / fold_size) as u64));
            }
        }
    }

    #[test]
    fn test_to_binary() {
        assert_eq!(to_binary(0b10111, 5), vec![true,false,true,true,true]);
        assert_eq!(to_binary(0b11001, 2), vec![false, true]);  // truncate
        assert_eq!(to_binary(1, 0), vec![]);
        assert_eq!(to_binary(0,0), vec![]);
    }
}