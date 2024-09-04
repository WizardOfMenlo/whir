use ark_ff::{FftField, Field};
use ark_poly::{Evaluations, Radix2EvaluationDomain};

use crate::crypto::ntt::ntt_batch;
use crate::parameters::FoldType;

// Given the evaluation of f on the coset specified by coset_offset * <coset_gen>
// Compute the fold on that point
pub fn compute_fold<F: Field>(
    answers: &[F],
    folding_randomness: &[F],
    mut coset_offset_inv: F,
    mut coset_gen_inv: F,
    two_inv: F,
    folding_factor: usize,
) -> F {
    let mut answers = answers.to_vec();

    // We recursively compute the fold, rec is where it is
    for rec in 0..folding_factor {
        let offset = answers.len() / 2;
        let mut new_answers = vec![F::ZERO; offset];
        let mut coset_index_inv = F::ONE;
        for i in 0..offset {
            let f_value_0 = answers[i];
            let f_value_1 = answers[i + offset];
            let point_inv = coset_offset_inv * coset_index_inv;

            let left = f_value_0 + f_value_1;
            let right = point_inv * (f_value_0 - f_value_1);

            new_answers[i] =
                two_inv * (left + folding_randomness[folding_randomness.len() - 1 - rec] * right);
            coset_index_inv *= coset_gen_inv;
        }
        answers = new_answers;

        // Update for next one
        coset_offset_inv = coset_offset_inv * coset_offset_inv;
        coset_gen_inv = coset_gen_inv * coset_gen_inv;
    }

    answers[0]
}

pub fn restructure_evaluations<F: FftField>(
    mut stacked_evaluations: Vec<F>,
    fold_type: FoldType,
    domain_gen: F,
    domain_gen_inv: F,
    folding_factor: usize,
) -> Vec<F> {
    let folding_size = 1_u64 << folding_factor;
    assert_eq!(stacked_evaluations.len() % (folding_size as usize), 0);
    eprintln!("{} x {}", stacked_evaluations.len(), folding_size);
    match fold_type {
        FoldType::Naive => stacked_evaluations,
        FoldType::ProverHelps => {
            // Stacked evaluation at i is f(B_l) where B_l = w^i * <w^n/k>
            let gen_scale = stacked_evaluations.len() / folding_size as usize; // n/2^k
            let coset_generator = domain_gen.pow(&[gen_scale as u64]);
            let coset_generator_inv = domain_gen_inv.pow(&[gen_scale as u64]);
            let size_as_field_element = F::from(folding_size);
            let size_inv = size_as_field_element.inverse().unwrap();

            ntt_batch(&mut stacked_evaluations, folding_size as usize);

            let mut coset_offset = F::ONE;
            let mut coset_offset_inv = F::ONE;
            for answers in stacked_evaluations.chunks_exact_mut(folding_size as usize) {
                let domain = Radix2EvaluationDomain {
                    size: folding_size,
                    log_size_of_group: folding_factor as u32,
                    size_as_field_element,
                    group_gen: coset_generator,
                    group_gen_inv: coset_generator_inv,
                    offset: coset_offset,
                    offset_inv: coset_offset_inv,
                    size_inv,
                    offset_pow_size: coset_offset.pow([folding_size]),
                };

                let evaluations = Evaluations::from_vec_and_domain(answers.to_vec(), domain);
                let mut interp = evaluations.interpolate().coeffs;
                interp.resize(folding_size as usize, F::ZERO);

                answers.copy_from_slice(&interp);

                coset_offset *= domain_gen;
                coset_offset_inv *= domain_gen_inv;
            }
            stacked_evaluations
        }
    }
}

#[cfg(test)]
mod tests {
    use ark_ff::{FftField, Field};

    use crate::{
        crypto::fields::Field64,
        poly_utils::{coeffs::CoefficientList, MultilinearPoint},
        utils::stack_evaluations,
    };

    use super::{compute_fold, restructure_evaluations};

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
        let folding_factor_exp = 1 << folding_factor;

        let poly = CoefficientList::new((0..num_coeffs).map(F::from).collect());

        let root_of_unity = F::get_root_of_unity(domain_size).unwrap();
        let root_of_unity_inv = root_of_unity.inverse().unwrap();

        let folding_randomness: Vec<_> = (0..folding_factor).map(|i| F::from(i as u64)).collect();

        // Evaluate the polynomial on the domain
        let domain_evaluations: Vec<_> = (0..domain_size)
            .map(|w| root_of_unity.pow([w as u64]))
            .map(|point| {
                poly.evaluate(&MultilinearPoint::expand_from_univariate(
                    point,
                    num_variables,
                ))
            })
            .collect();

        let unprocessed = stack_evaluations(domain_evaluations, folding_factor);

        let processed = restructure_evaluations(
            unprocessed.clone(),
            crate::parameters::FoldType::ProverHelps,
            root_of_unity,
            root_of_unity_inv,
            folding_factor,
        );

        let num = domain_size / folding_factor_exp;
        let coset_gen_inv = root_of_unity_inv.pow(&[num]);

        for index in 0..num {
            let offset_inv = root_of_unity_inv.pow(&[index]);

            let answer_unprocessed = compute_fold(
                &unprocessed[index as usize],
                &folding_randomness,
                offset_inv,
                coset_gen_inv,
                F::from(2).inverse().unwrap(),
                folding_factor,
            );

            let answer_processed = CoefficientList::new(processed[index as usize].to_vec())
                .evaluate(&MultilinearPoint(folding_randomness.clone()));

            assert_eq!(answer_processed, answer_unprocessed);
        }
    }
}
