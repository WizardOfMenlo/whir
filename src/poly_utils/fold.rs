use ark_ff::Field;

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

#[cfg(test)]
mod tests {
    use ark_ff::{FftField, Field};

    use crate::{
        crypto::fields::Field64,
        poly_utils::{coeffs::CoefficientList, MultilinearPoint},
    };

    use super::compute_fold;

    type F = Field64;

    #[test]
    fn test_folding() {
        let num_variables = 5;
        let num_coeffs = 1 << num_variables;

        let domain_size = 256;
        let folding_factor = 3; // We fold in 8
        let folding_factor_exp = 1 << folding_factor;

        let poly = CoefficientList::new((0..num_coeffs).map(|i| F::from(i)).collect());

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
}
