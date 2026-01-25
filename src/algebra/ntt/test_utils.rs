use ark_ff::FftField;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::*;

pub(super) fn expand_from_coeff<F: FftField>(coeffs: &[F], expansion: usize) -> Vec<F> {
    let engine = cooley_tukey::NttEngine::<F>::new_from_cache();
    let expanded_size = coeffs.len() * expansion;
    let mut result = Vec::with_capacity(expanded_size);
    // Note: We can also zero-extend the coefficients and do a larger NTT.
    // But this is more efficient.

    // Do coset NTT.
    let root = engine.root(expanded_size);
    result.extend_from_slice(coeffs);
    #[cfg(not(feature = "parallel"))]
    for i in 1..expansion {
        let root = root.pow([i as u64]);
        let mut offset = F::ONE;
        result.extend(coeffs.iter().map(|x| {
            let val = *x * offset;
            offset *= root;
            val
        }));
    }
    #[cfg(feature = "parallel")]
    result.par_extend((1..expansion).into_par_iter().flat_map(|i| {
        let root_i = root.pow([i as u64]);
        coeffs
            .par_iter()
            .enumerate()
            .map_with(F::ZERO, move |root_j, (j, coeff)| {
                if root_j.is_zero() {
                    *root_j = root_i.pow([j as u64]);
                } else {
                    *root_j *= root_i;
                }
                *coeff * *root_j
            })
    }));

    ntt_batch(&mut result, coeffs.len());
    transpose(&mut result, expansion, coeffs.len());
    result
}

/// Applies a folding transformation to evaluation vectors in-place.
///
/// Performs reshaping, inverse NTTs, and applies coset + scaling correction.
///
/// The evaluations are grouped into `2^folding_factor` blocks of size `N / 2^folding_factor`.
/// For each group, the function performs the following:
///
/// 1. Transpose: reshape layout to enable independent processing of each sub-coset.
/// 2. Inverse NTT: convert each sub-coset from evaluation to coefficient form (no 1/N scaling).
/// 3. Scale correction:
///    Each output is multiplied by:
///
///    ```ignore
///    size_inv * (domain_gen_inv^i)^j
///    ```
///
///    where:
///      - `size_inv = 1 / 2^folding_factor`
///      - `i` is the subdomain index
///      - `j` is the index within the block
///
/// # Panics
/// Panics if the input size is not divisible by `2^folding_factor`.
pub fn transform_evaluations<F: FftField>(
    evals: &mut [F],
    domain_gen_inv: F,
    folding_factor: usize,
) {
    // Compute the number of sub-cosets = 2^folding_factor
    let folding_factor_exp = 1 << folding_factor;

    // Ensure input is divisible by folding factor
    assert!(evals.len().is_multiple_of(folding_factor_exp));

    // Number of rows (one per subdomain)
    let size_of_new_domain = evals.len() / folding_factor_exp;

    // Step 1: Reshape via transposition
    transpose(evals, folding_factor_exp, size_of_new_domain);

    // Step 2: Apply inverse NTTs
    intt_batch(evals, folding_factor_exp);

    // Step 3: Apply scaling to match the desired domain layout
    // Each value is scaled by: size_inv * offset^j
    let size_inv = F::from(folding_factor_exp as u64).inverse().unwrap();
    #[cfg(not(feature = "parallel"))]
    {
        let mut coset_offset_inv = F::ONE;
        for answers in evals.chunks_exact_mut(folding_factor_exp) {
            let mut scale = size_inv;
            for v in answers.iter_mut() {
                *v *= scale;
                scale *= coset_offset_inv;
            }
            coset_offset_inv *= domain_gen_inv;
        }
    }
    #[cfg(feature = "parallel")]
    evals
        .par_chunks_exact_mut(folding_factor_exp)
        .enumerate()
        .for_each_with(F::ZERO, |offset, (i, answers)| {
            if *offset == F::ZERO {
                *offset = domain_gen_inv.pow([i as u64]);
            } else {
                *offset *= domain_gen_inv;
            }
            let mut scale = size_inv;
            for v in answers.iter_mut() {
                *v *= scale;
                scale *= &*offset;
            }
        });
}
