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
            answers[i] =
                two_inv * (left + folding_randomness[folding_randomness.len() - 1 - rec] * right);
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

pub fn transform_evaluations<F: FftField>(
    evals: &mut [F],
    fold_type: FoldType,
    _domain_gen: F,
    domain_gen_inv: F,
    folding_factor: usize,
) {
    let folding_factor_exp = 1 << folding_factor;
    assert!(evals.len() % folding_factor_exp == 0);
    let size_of_new_domain = evals.len() / folding_factor_exp;

    match fold_type {
        FoldType::Naive => {
            // Perform only the stacking step (transpose)
            transpose(evals, folding_factor_exp, size_of_new_domain);
        }
        FoldType::ProverHelps => {
            // Perform stacking (transpose)
            transpose(evals, folding_factor_exp, size_of_new_domain);

            // Batch inverse NTTs
            intt_batch(evals, folding_factor_exp);

            // Apply coset and size correction.
            // Stacked evaluation at i is f(B_l) where B_l = w^i * <w^n/k>
            let size_inv = F::from(folding_factor_exp as u64).inverse().unwrap();
            #[cfg(not(feature = "parallel"))]
            {
                let mut coset_offset_inv = F::ONE;
                for answers in stacked_evaluations.chunks_exact_mut(folding_size as usize) {
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
    use ark_ff::{FftField, Field};

    use crate::{
        crypto::fields::Field64,
        ntt::transpose,
        poly_utils::{coeffs::CoefficientList, multilinear::MultilinearPoint},
    };

    use super::{compute_fold, transform_evaluations};

    type F = Field64;

    #[test]
    fn test_folding() {
        let num_variables = 5;
        let num_coeffs = 1 << num_variables;

        let domain_size = 256;
        let folding_factor = 3; // We fold in 8
        let folding_factor_exp = 1 << folding_factor;

        let poly = CoefficientList::new((0..num_coeffs).map(F::from).collect());

        let root_of_unity = F::get_root_of_unity(domain_size).unwrap();

        let index = 15;
        let folding_randomness: Vec<_> = (0..folding_factor).map(|i| F::from(i as u64)).collect();

        let coset_offset = root_of_unity.pow([index]);
        let coset_gen = root_of_unity.pow([domain_size / folding_factor_exp]);

        // Evaluate the polynomial on the coset
        let poly_eval: Vec<_> = (0..folding_factor_exp)
            .map(|i| {
                poly.evaluate(&MultilinearPoint::expand_from_univariate(
                    coset_offset * coset_gen.pow([i]),
                    num_variables,
                ))
            })
            .collect();

        let fold_value = compute_fold(
            &poly_eval,
            &folding_randomness,
            coset_offset.inverse().unwrap(),
            coset_gen.inverse().unwrap(),
            F::from(2).inverse().unwrap(),
            folding_factor,
        );

        let truth_value = poly.fold(&MultilinearPoint(folding_randomness)).evaluate(
            &MultilinearPoint::expand_from_univariate(
                root_of_unity.pow([folding_factor_exp * index]),
                2,
            ),
        );

        assert_eq!(fold_value, truth_value);
    }

    #[test]
    fn test_folding_optimised() {
        let num_variables = 5;
        let num_coeffs = 1 << num_variables;

        let domain_size = 256;
        let folding_factor = 3; // We fold in 8
        let folding_factor_exp: u64 = 1 << folding_factor;

        let poly = CoefficientList::new((0..num_coeffs).map(F::from).collect());

        let root_of_unity = F::get_root_of_unity(domain_size).unwrap();
        let root_of_unity_inv = root_of_unity.inverse().unwrap();

        let folding_randomness: Vec<_> = (0..folding_factor).map(|i| F::from(i as u64)).collect();

        // Evaluate the polynomial on the domain
        let mut domain_evaluations: Vec<_> = (0..domain_size)
            .map(|w| root_of_unity.pow([w]))
            .map(|point| {
                poly.evaluate(&MultilinearPoint::expand_from_univariate(
                    point,
                    num_variables,
                ))
            })
            .collect();

        let mut unprocessed = domain_evaluations.clone();
        transpose(
            &mut unprocessed,
            folding_factor_exp as usize,
            domain_evaluations.len() / folding_factor_exp as usize,
        );

        transform_evaluations(
            &mut domain_evaluations,
            crate::parameters::FoldType::ProverHelps,
            root_of_unity,
            root_of_unity_inv,
            folding_factor,
        );

        let num = domain_size / folding_factor_exp;
        let coset_gen_inv = root_of_unity_inv.pow([num]);

        for index in 0..num {
            let offset_inv = root_of_unity_inv.pow([index]);
            let span =
                (index * folding_factor_exp) as usize..((index + 1) * folding_factor_exp) as usize;

            let answer_unprocessed = compute_fold(
                &unprocessed[span.clone()],
                &folding_randomness,
                offset_inv,
                coset_gen_inv,
                F::from(2).inverse().unwrap(),
                folding_factor,
            );

            let answer_processed = CoefficientList::new(domain_evaluations[span].to_vec())
                .evaluate(&MultilinearPoint(folding_randomness.clone()));

            assert_eq!(answer_processed, answer_unprocessed);
        }
    }
}
