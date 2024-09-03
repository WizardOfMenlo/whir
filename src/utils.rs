use std::collections::BTreeSet;

use ark_ff::Field;

pub fn is_power_of_two(n: usize) -> bool {
    n & (n - 1) == 0
}

pub fn to_binary(value: usize, n_bits: usize) -> Vec<bool> {
    // Ensure that n is within the bounds of the input integer type
    assert!(n_bits <= usize::BITS as usize);
    let mut result = vec![false; n_bits];
    for i in 0..n_bits {
        result[n_bits - 1 - i] = (value & (1 << i)) != 0;
    }
    result
}

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
pub fn stack_evaluations<F: Copy>(evals: Vec<F>, folding_factor: usize) -> Vec<F> {
    let folding_factor_exp = 1 << folding_factor;
    assert!(evals.len() % folding_factor_exp == 0);
    let size_of_new_domain = evals.len() / folding_factor_exp;

    let mut stacked_evaluations = Vec::with_capacity(evals.len());
    for i in 0..size_of_new_domain {
        for j in 0..folding_factor_exp {
            stacked_evaluations.push(evals[i + j * size_of_new_domain]);
        }
    }

    stacked_evaluations
}

#[cfg(test)]
mod tests {
    use super::stack_evaluations;

    #[test]
    fn test_evaluations_stack() {
        let num = 256;
        let folding_factor = 3;
        let evals: Vec<_> = (0..num).collect();

        let stacked = stack_evaluations(evals, folding_factor);

        assert_eq!(stacked.len(), num / (1 << folding_factor));

        for i in 0..stacked.len() {
            assert_eq!(stacked[i].len(), 1 << folding_factor);
            for j in 0..(1 << folding_factor) {
                assert_eq!(stacked[i][j], i + j * num / (1 << folding_factor));
            }
        }
    }
}
