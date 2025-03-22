use crate::ntt::{intt_batch, transpose};
use crate::parameters::FoldType;
use ark_ff::{FftField, Field};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Computes the folded value of a function evaluated on a coset.
///
/// This function applies a recursive folding transformation to a given set of function
/// evaluations on a coset, progressively reducing the number of evaluations while incorporating
/// randomness and coset transformations. The folding process is performed `folding_factor` times,
/// halving the number of evaluations at each step.
///
/// Mathematical Formulation:
/// Given an initial evaluation vector:
/// \begin{equation}
/// f(x) = [f_0, f_1, ..., f_{2^m - 1}]
/// \end{equation}
///
/// Each folding step computes:
/// \begin{equation}
/// g_i = \frac{f_i + f_{i + N/2} + r \cdot (f_i - f_{i + N/2}) \cdot (o^{-1} \cdot g^{-i})}{2}
/// \end{equation}
///
/// where:
/// - \( r \) is the folding randomness
/// - \( o^{-1} \) is the inverse coset offset
/// - \( g^{-i} \) is the inverse generator raised to index \( i \)
/// - The function is recursively applied until the vector reduces to size 1.
pub fn compute_fold<F: Field>(
    answers: &[F],
    folding_randomness: &[F],
    mut coset_offset_inv: F,
    mut coset_gen_inv: F,
    two_inv: F,
    folding_factor: usize,
) -> F {
    let mut answers = answers.to_vec();

    // Perform the folding process `folding_factor` times.
    for rec in 0..folding_factor {
        let r = folding_randomness[folding_randomness.len() - 1 - rec];
        let offset = answers.len() / 2;
        let mut coset_index_inv = F::ONE;

        // Compute the new folded values, iterating over the first half of `answers`.
        for i in 0..offset {
            let f0 = answers[i];
            let f1 = answers[i + offset];
            let point_inv = coset_offset_inv * coset_index_inv;

            let left = f0 + f1;
            let right = point_inv * (f0 - f1);

            // Apply the folding transformation with randomness
            answers[i] = two_inv * (left + r * right);
            coset_index_inv *= coset_gen_inv;
        }

        // Reduce answers to half its size without allocating a new vector
        answers.truncate(offset);

        // Update for next iteration
        coset_offset_inv *= coset_offset_inv;
        coset_gen_inv *= coset_gen_inv;
    }

    answers[0]
}

/// Applies a folding transformation to evaluation vectors in-place.
///
/// This is used to prepare a set of evaluations for a sumcheck-style polynomial folding,
/// supporting two modes:
///
/// - `FoldType::Naive`: applies only the reshaping step (transposition).
/// - `FoldType::ProverHelps`: performs reshaping, inverse NTTs, and applies coset + scaling correction.
///
/// The evaluations are grouped into `2^folding_factor` blocks of size `N / 2^folding_factor`.
/// For each group, the function performs the following (if `ProverHelps`):
///
/// 1. Transpose: reshape layout to enable independent processing of each sub-coset.
/// 2. Inverse NTT: convert each sub-coset from evaluation to coefficient form (no 1/N scaling).
/// 3. Scale correction:
///    Each output is multiplied by:
///
///    ```
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
    fold_type: FoldType,
    _domain_gen: F,
    domain_gen_inv: F,
    folding_factor: usize,
) {
    // Compute the number of sub-cosets = 2^folding_factor
    let folding_factor_exp = 1 << folding_factor;

    // Ensure input is divisible by folding factor
    assert!(evals.len() % folding_factor_exp == 0);

    // Number of rows (one per subdomain)
    let size_of_new_domain = evals.len() / folding_factor_exp;

    match fold_type {
        FoldType::Naive => {
            // Simply transpose into column-major form: shape = [folding_factor_exp × size_of_new_domain]
            transpose(evals, folding_factor_exp, size_of_new_domain);
        }
        FoldType::ProverHelps => {
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
    }
}

#[cfg(test)]
mod tests {
    use super::{compute_fold, transform_evaluations};
    use crate::{
        crypto::fields::Field64,
        ntt::transpose,
        parameters::FoldType,
        poly_utils::{coeffs::CoefficientList, multilinear::MultilinearPoint},
    };
    use ark_ff::{AdditiveGroup, FftField, Field};

    type F = Field64;

    #[test]
    fn test_folding() {
        // Number of variables in the multilinear polynomial (5 → 32 coefficients)
        let num_variables = 5;
        let num_coeffs = 1 << num_variables;

        // Total size of the evaluation domain
        let domain_size = 256;

        // Folding factor determines how many evaluations to fold at once (2^3 = 8)
        let folding_factor = 3;
        let folding_factor_exp = 1 << folding_factor;

        // Create a simple multilinear polynomial f(x₀, ..., x₄) = ∑ xᵢ
        let poly = CoefficientList::new((0..num_coeffs).map(F::from).collect());

        // Get the primitive root of unity for the domain
        let root_of_unity = F::get_root_of_unity(domain_size).unwrap();

        // Pick a specific coset index to evaluate
        let index = 15;
        // Folding randomness vector `r = [0, 1, 2]`
        let folding_randomness: Vec<_> = (0..folding_factor as u64).map(F::from).collect();

        // Compute coset offset = ω^index
        let coset_offset = root_of_unity.pow([index]);
        // Compute coset generator = ω^{N / 2^m}
        let coset_gen = root_of_unity.pow([domain_size / folding_factor_exp]);

        // Evaluate the polynomial at the points in the coset: γ * g^i
        let poly_eval: Vec<_> = (0..folding_factor_exp)
            .map(|i| {
                poly.evaluate(&MultilinearPoint::expand_from_univariate(
                    coset_offset * coset_gen.pow([i]),
                    num_variables,
                ))
            })
            .collect();

        // Fold the evaluations down to a single value using the compute_fold routine
        let fold_value = compute_fold(
            &poly_eval,
            &folding_randomness,
            coset_offset.inverse().unwrap(),
            coset_gen.inverse().unwrap(),
            F::from(2).inverse().unwrap(),
            folding_factor,
        );

        // Compute the expected value by folding the polynomial, then evaluating it at ω^{8·index}
        let truth_value = poly.fold(&MultilinearPoint(folding_randomness)).evaluate(
            &MultilinearPoint::expand_from_univariate(
                root_of_unity.pow([folding_factor_exp * index]),
                2,
            ),
        );

        // The folded value should match the evaluation of the folded polynomial
        assert_eq!(fold_value, truth_value);
    }

    #[test]
    fn test_folding_optimised() {
        // Number of variables in the multilinear polynomial (2⁵ = 32 values)
        let num_variables = 5;
        let num_coeffs = 1 << num_variables;

        // Set the domain size and folding factor (folds into groups of 8)
        let domain_size = 256;
        let folding_factor = 3;
        let folding_factor_exp: u64 = 1 << folding_factor;

        // Define the polynomial as f(x) = x for x in 0..32
        let poly = CoefficientList::new((0..num_coeffs).map(F::from).collect());

        // Get root of unity and its inverse
        let root_of_unity = F::get_root_of_unity(domain_size).unwrap();
        let root_of_unity_inv = root_of_unity.inverse().unwrap();

        // Randomness used in folding (e.g., r = [0, 1, 2])
        let folding_randomness: Vec<_> = (0..folding_factor as u64).map(F::from).collect();

        // Evaluate polynomial on all domain points: ω^0, ω^1, ..., ω^{255}
        let mut domain_evaluations: Vec<_> = (0..domain_size)
            .map(|w| root_of_unity.pow([w]))
            .map(|point| {
                poly.evaluate(&MultilinearPoint::expand_from_univariate(
                    point,
                    num_variables,
                ))
            })
            .collect();

        // Clone original evaluations to simulate the "unprocessed" version
        let mut unprocessed = domain_evaluations.clone();

        // Stack the evaluations: reshape into matrix of size (256 / 8) × 8 = 32 × 8
        transpose(
            &mut unprocessed,
            folding_factor_exp as usize,
            domain_evaluations.len() / folding_factor_exp as usize,
        );

        // Transform the evaluations into the "ProverHelps" format
        transform_evaluations(
            &mut domain_evaluations,
            crate::parameters::FoldType::ProverHelps,
            root_of_unity,
            root_of_unity_inv,
            folding_factor,
        );

        // Number of cosets = domain_size / folding_factor_exp
        let num = domain_size / folding_factor_exp;

        // Compute inverse of coset generator: ω^{-num}
        let coset_gen_inv = root_of_unity_inv.pow([num]);

        // For each coset (row in the transposed matrix)...
        for index in 0..num {
            // Compute inverse offset: ω^{-index}
            let offset_inv = root_of_unity_inv.pow([index]);

            // Slice the unprocessed chunk from the full evaluation table
            let span =
                (index * folding_factor_exp) as usize..((index + 1) * folding_factor_exp) as usize;

            // Compute folded value manually using compute_fold on unprocessed input
            let answer_unprocessed = compute_fold(
                &unprocessed[span.clone()],
                &folding_randomness,
                offset_inv,
                coset_gen_inv,
                F::from(2).inverse().unwrap(),
                folding_factor,
            );

            // Compute folded value using the processed evaluation and standard evaluation
            let answer_processed = CoefficientList::new(domain_evaluations[span].to_vec())
                .evaluate(&MultilinearPoint(folding_randomness.clone()));

            // Assert that both answers match
            assert_eq!(answer_processed, answer_unprocessed);
        }
    }

    #[test]
    fn test_compute_fold_single_layer() {
        // Folding a vector of size 2: f(x) = [1, 3]
        let f0 = F::from(1);
        let f1 = F::from(3);
        let answers = vec![f0, f1];

        let r = F::from(2); // folding randomness
        let folding_randomness = vec![r];

        let coset_offset_inv = F::from(5); // arbitrary inverse offset
        let coset_gen_inv = F::from(7); // arbitrary generator inverse
        let two_inv = F::from(2).inverse().unwrap();

        // g = (f0 + f1 + r * (f0 - f1) * coset_offset_inv) / 2
        // Here coset_index_inv = 1
        // => left = f0 + f1
        // => right = r * (f0 - f1) * coset_offset_inv
        // => g = (left + right) / 2
        let expected = two_inv * (f0 + f1 + r * (f0 - f1) * coset_offset_inv);

        let result = compute_fold(
            &answers,
            &folding_randomness,
            coset_offset_inv,
            coset_gen_inv,
            two_inv,
            1,
        );
        assert_eq!(result, expected);
    }

    #[test]
    fn test_compute_fold_two_layers() {
        // Define the input evaluations: f(x) = [f00, f01, f10, f11]
        let f00 = F::from(1);
        let f01 = F::from(2);
        let f10 = F::from(3);
        let f11 = F::from(4);

        // Create the input vector for folding
        let answers = vec![f00, f01, f10, f11];

        // Folding randomness used in each layer (innermost first)
        let r0 = F::from(5); // randomness for layer 1 (first fold)
        let r1 = F::from(7); // randomness for layer 2 (second fold)
        let folding_randomness = vec![r1, r0]; // reversed because fold reads from the back

        // Precompute constants
        let two_inv = F::from(2).inverse().unwrap(); // 1/2 used in folding formula
        let coset_offset_inv = F::from(9); // offset⁻¹
        let coset_gen_inv = F::from(3); // generator⁻¹

        // --- First layer of folding ---

        // Fold the pair [f00, f10] using coset_index_inv = 1
        // left = f00 + f10
        let g0_left = f00 + f10;

        // right = (f00 - f10) * coset_offset_inv * coset_index_inv
        // where coset_index_inv = 1
        let g0_right = coset_offset_inv * (f00 - f10);

        // g0 = (left + r0 * right) / 2
        let g0 = two_inv * (g0_left + r0 * g0_right);

        // Fold the pair [f01, f11] using coset_index_inv = coset_gen_inv
        let coset_index_inv_1 = coset_gen_inv;

        // left = f01 + f11
        let g1_left = f01 + f11;

        // right = (f01 - f11) * coset_offset_inv * coset_index_inv_1
        let g1_right = coset_offset_inv * coset_index_inv_1 * (f01 - f11);

        // g1 = (left + r0 * right) / 2
        let g1 = two_inv * (g1_left + r0 * g1_right);

        // --- Second layer of folding ---

        // Update the coset offset for next layer: offset⁻¹ → offset⁻¹²
        let next_coset_offset_inv = coset_offset_inv * coset_offset_inv;

        // Fold the pair [g0, g1] using coset_index_inv = 1
        // left = g0 + g1
        let g_final_left = g0 + g1;

        // right = (g0 - g1) * next_coset_offset_inv
        let g_final_right = next_coset_offset_inv * (g0 - g1);

        // Final folded value
        let expected = two_inv * (g_final_left + r1 * g_final_right);

        // Compute using the actual implementation
        let result = compute_fold(
            &answers,
            &folding_randomness,
            coset_offset_inv,
            coset_gen_inv,
            two_inv,
            2,
        );

        // Assert that the result matches the manually computed expected value
        assert_eq!(result, expected);
    }

    #[test]
    fn test_compute_fold_with_zero_randomness() {
        // Inputs: f(x) = [f0, f1]
        let f0 = F::from(6);
        let f1 = F::from(2);
        let answers = vec![f0, f1];

        let r = F::ZERO;
        let folding_randomness = vec![r];

        let two_inv = F::from(2).inverse().unwrap();
        let coset_offset_inv = F::from(10);
        let coset_gen_inv = F::from(3);

        let left = f0 + f1;
        // with r = 0, this simplifies to (f0 + f1) / 2
        let expected = two_inv * left;

        let result = compute_fold(
            &answers,
            &folding_randomness,
            coset_offset_inv,
            coset_gen_inv,
            two_inv,
            1,
        );

        assert_eq!(result, expected);
    }

    #[test]
    fn test_compute_fold_all_zeros() {
        // All values are zero: f(x) = [0, 0, ..., 0]
        let answers = vec![F::ZERO; 8];
        let folding_randomness = vec![F::from(3); 3];
        let two_inv = F::from(2).inverse().unwrap();
        let coset_offset_inv = F::from(4);
        let coset_gen_inv = F::from(7);

        // each fold step is (0 + 0 + r * (0 - 0) * _) / 2 = 0
        let expected = F::ZERO;

        let result = compute_fold(
            &answers,
            &folding_randomness,
            coset_offset_inv,
            coset_gen_inv,
            two_inv,
            3,
        );

        assert_eq!(result, expected);
    }

    #[test]
    fn test_transform_evaluations_naive_manual() {
        // Input: 2×2 matrix (folding_factor = 1 → folding_factor_exp = 2)
        // Stored row-major as: [a00, a01, a10, a11]
        // We expect it to transpose to: [a00, a10, a01, a11]

        let folding_factor = 1;
        let mut evals = vec![
            F::from(1), // a00
            F::from(2), // a01
            F::from(3), // a10
            F::from(4), // a11
        ];

        // Expected transpose: column-major layout
        let expected = vec![
            F::from(1), // a00
            F::from(3), // a10
            F::from(2), // a01
            F::from(4), // a11
        ];

        // Naive transpose — only reshuffling the data
        transform_evaluations(&mut evals, FoldType::Naive, F::ONE, F::ONE, folding_factor);

        assert_eq!(evals, expected);
    }

    #[test]
    #[allow(clippy::cast_sign_loss)]
    fn test_transform_evaluations_prover_helps() {
        // Setup:
        // - 2×2 matrix: 2 cosets (rows), each with 2 points
        // - folding_factor = 1 → folding_factor_exp = 2
        let folding_factor = 1;
        let folding_factor_exp = 1 << folding_factor;

        // Domain generator and its inverse (arbitrary but consistent)
        let domain_gen = F::from(4);
        let domain_gen_inv = domain_gen.inverse().unwrap();

        // Row-major input:
        //   row 0: [a0, a1]
        //   row 1: [b0, b1]
        let a0 = F::from(1);
        let a1 = F::from(3);
        let b0 = F::from(2);
        let b1 = F::from(4);
        let mut evals = vec![a0, a1, b0, b1];

        // Step 1: Transpose (rows become columns)
        // After transpose: [a0, b0, a1, b1]
        let t0 = a0;
        let t1 = b0;
        let t2 = a1;
        let t3 = b1;

        // Step 2: Inverse NTT on each row (length 2) without scaling
        //
        // For [x0, x1], inverse NTT without scaling is:
        //     [x0 + x1, x0 - x1]
        //
        // Row 0: [a0, b0] → [a0 + b0, a0 - b0]
        let intt0 = t0 + t1;
        let intt1 = t0 - t1;

        // Row 1: [a1, b1] → [a1 + b1, a1 - b1]
        let intt2 = t2 + t3;
        let intt3 = t2 - t3;

        // Step 3: Apply scaling
        //
        // Each row is scaled by:
        //    v[j] *= size_inv * (coset_offset_inv)^j
        //
        // For row 0, offset = 1 (coset_offset_inv^j = 1)
        // For row 1, offset = domain_gen_inv
        let size_inv = F::from(folding_factor_exp as u64).inverse().unwrap();

        let expected = vec![
            intt0 * size_inv * F::ONE,
            intt1 * size_inv * F::ONE,
            intt2 * size_inv * F::ONE, // first entry of row 1, scale = 1
            intt3 * size_inv * domain_gen_inv, // second entry of row 1
        ];

        // Run transform
        transform_evaluations(
            &mut evals,
            FoldType::ProverHelps,
            domain_gen,
            domain_gen_inv,
            folding_factor,
        );

        // Validate output
        assert_eq!(evals, expected);
    }

    #[test]
    #[should_panic]
    fn test_transform_evaluations_invalid_length() {
        let mut evals = vec![F::from(1); 6]; // Not a power of 2
        transform_evaluations(&mut evals, FoldType::Naive, F::ONE, F::ONE, 2);
    }
}
