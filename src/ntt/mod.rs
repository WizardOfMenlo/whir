//! NTT and related algorithms.

mod cooley_tukey;
mod matrix;

#[cfg(test)]
pub(crate) mod test_utils;

mod transpose;
mod utils;
mod wavelet;

use std::{fmt::Debug, sync::LazyLock};

use ark_ff::FftField;
use static_assertions::assert_obj_safe;
#[cfg(feature = "tracing")]
use tracing::instrument;

use self::matrix::MatrixMut;
pub use self::{
    cooley_tukey::{generator, intt, intt_batch, ntt, ntt_batch},
    transpose::transpose,
    wavelet::{inverse_wavelet_transform, wavelet_transform},
};
use crate::type_map::{self, TypeMap};

pub static NTT: LazyLock<TypeMap<NttFamily>> = LazyLock::new(|| TypeMap::default());

#[derive(Default)]
pub struct NttFamily;

impl type_map::Family for NttFamily {
    type Dyn<F: 'static> = dyn ReedSolomon<F>;
}

/// Trait for replacing the default Reed Solomon encoding ([`RSDefault`]) with an specialised Reed Solomon encoder for the FFTField and BasePrimeField.
pub trait ReedSolomon<F>: Debug + Send + Sync {
    fn interleaved_encode(
        &self,
        interleaved_coeffs: &[F],
        expansion: usize,
        fold_factor: usize,
    ) -> Vec<F>;
}

assert_obj_safe!(ReedSolomon<crate::crypto::fields::Field256>);

// pub fn interleaved_rs_encode<F>(
//     interleaved_coeffs: &[F],
//     expansion: usize,
//     fold_factor: usize,
// ) -> Vec<F> {
//     let engine = NTT.get::<F>().expect("Unsupported field");
//     engine.interleaved_encode(interleaved_coeffs, expansion, fold_factor)
// }

///
/// RS encode interleaved data `interleaved_coeffs` at the rate
/// 1/`expansion`, where 2^`fold_factor` elements are interleaved
/// together.
///
/// This function computes the RS-code for each interleaved message and
/// outputs the interleaved alphabets in the same order as the input.
///
#[cfg_attr(feature = "tracing", instrument(skip(interleaved_coeffs), fields(size = interleaved_coeffs.len())))]
pub fn interleaved_rs_encode<F: FftField>(
    coeffs: &[F],
    expansion: usize,
    interleaving_depth: usize,
) -> Vec<F> {
    assert!(expansion > 0);
    assert!(coeffs.len().is_multiple_of(interleaving_depth));
    let expanded_size = coeffs.len() * expansion;

    // 1. Create zero-padded message of appropriate size
    let mut result = vec![F::zero(); expanded_size];
    result[..coeffs.len()].copy_from_slice(coeffs);

    let rows = expanded_size / interleaving_depth;
    let columns = interleaving_depth;
    //
    // 2. Convert from column-major (interleaved form) to row-major
    //    representation.
    //

    // TODO: Might be useful to keep the transposed data for future use.
    transpose(&mut result, rows, columns);
    ntt_batch(&mut result, rows);
    transpose(&mut result, columns, rows);
    result
}

/// Tag to select the built-in Reed Solomon Encoding
#[derive(Clone, Copy, Debug)]
pub struct RSDefault;

impl<F: FftField> ReedSolomon<F> for RSDefault {
    fn interleaved_encode(
        &self,
        interleaved_coeffs: &[F],
        expansion: usize,
        fold_factor: usize,
    ) -> Vec<F> {
        interleaved_rs_encode(interleaved_coeffs, expansion, 1 << fold_factor)
    }
}

#[cfg(test)]
mod tests {
    use ark_ff::Field;

    use super::*;
    use crate::{crypto::fields::Field64, ntt::cooley_tukey::NttEngine};

    #[test]
    fn test_expand_from_coeff_size_2() {
        let engine = NttEngine::<Field64>::new_from_fftfield();

        let c0 = Field64::from(1);
        let c1 = Field64::from(2);
        let coeffs = vec![c0, c1];
        let expansion = 2;

        let omega = engine.root(4);

        // Expansion of the coefficient vector
        //
        // The expansion factor is 2, so we extend the original coefficients as follows:
        //
        //   f0 = c0
        //   f1 = c1
        //   f2 = c0 * ω⁰ = c0
        //   f3 = c1 * ω¹ = c1 * ω
        //
        // Using c0 = 1, c1 = 2, and ω as the generator:

        let f0 = c0;
        let f1 = c1;
        let f2 = c0 * omega.pow([0]);
        let f3 = c1 * omega.pow([1]);

        // Compute the expected NTT
        //
        // The NTT for a size-2 batch follows:
        //
        //   F(0) = f0 + f1
        //   F(1) = f0 - f1
        //
        // We apply this to both pairs (f0, f1) and (f2, f3):
        //
        //   F(0) = f0 + f1
        //   F(1) = f0 - f1
        //
        //   F(2) = f2 + f3
        //   F(3) = f2 - f3
        //
        // Now using the omega-based approach:

        let expected_f0 = f0 + f1;
        let expected_f1 = f0 - f1;
        let expected_f2 = f2 + f3;
        let expected_f3 = f2 - f3;

        // The expected NTT result should be in transposed order:
        let expected_values_transposed = vec![expected_f0, expected_f2, expected_f1, expected_f3];

        let computed_values = test_utils::expand_from_coeff(&coeffs, expansion);
        assert_eq!(computed_values, expected_values_transposed);
    }

    #[test]
    fn test_expand_from_coeff_size_4() {
        let engine = NttEngine::<Field64>::new_from_fftfield();

        let c0 = Field64::from(1);
        let c1 = Field64::from(2);
        let c2 = Field64::from(3);
        let c3 = Field64::from(4);
        let coeffs = vec![c0, c1, c2, c3];
        let expansion = 4;

        let omega = engine.root(16);

        // Manual expansion of the coefficient vector
        //
        // The expansion factor is 4, so we extend the original coefficients into 16 values:
        //
        //   f0  = c0
        //   f1  = c1
        //   f2  = c2
        //   f3  = c3
        //   f4  = c0 * ω⁰  = c0
        //   f5  = c1 * ω¹  = c1 * ω
        //   f6  = c2 * ω²  = c2 * ω²
        //   f7  = c3 * ω³  = c3 * ω³
        //   f8  = c0 * ω⁰  = c0
        //   f9  = c1 * ω²  = c1 * ω²
        //   f10 = c2 * ω⁴  = c2 * ω⁴
        //   f11 = c3 * ω⁶  = c3 * ω⁶
        //   f12 = c0 * ω⁰  = c0
        //   f13 = c1 * ω³  = c1 * ω³
        //   f14 = c2 * ω⁶  = c2 * ω⁶
        //   f15 = c3 * ω⁹  = c3 * ω⁹
        //
        // With c0 = 1, c1 = 2, c2 = 3, c3 = 4, and ω as the generator:

        let f0 = c0;
        let f1 = c1;
        let f2 = c2;
        let f3 = c3;

        let f4 = c0 * omega.pow([1]).pow([0]);
        let f5 = c1 * omega.pow([1]).pow([1]);
        let f6 = c2 * omega.pow([1]).pow([2]);
        let f7 = c3 * omega.pow([1]).pow([3]);

        let f8 = c0 * omega.pow([2]).pow([0]);
        let f9 = c1 * omega.pow([2]).pow([1]);
        let f10 = c2 * omega.pow([2]).pow([2]);
        let f11 = c3 * omega.pow([2]).pow([3]);

        let f12 = c0 * omega.pow([3]).pow([0]);
        let f13 = c1 * omega.pow([3]).pow([1]);
        let f14 = c2 * omega.pow([3]).pow([2]);
        let f15 = c3 * omega.pow([3]).pow([3]);

        // Compute the expected NTT manually using omega powers
        //
        // We process the values in **four chunks of four elements**, following the radix-2
        // butterfly structure.

        let omega = engine.root(4);

        let omega1 = omega; // ω
        let omega2 = omega * omega; // ω²
        let omega3 = omega * omega2; // ω³
        let omega4 = omega * omega3; // ω⁴

        // Chunk 1 (f0 to f3)
        let expected_f0 = f0 + f1 + f2 + f3;
        let expected_f1 = f0 + f1 * omega1 + f2 * omega2 + f3 * omega3;
        let expected_f2 = f0 + f1 * omega2 + f2 * omega4 + f3 * omega2;
        let expected_f3 = f0 + f1 * omega3 + f2 * omega2 + f3 * omega1;

        // Chunk 2 (f4 to f7)
        let expected_f4 = f4 + f5 + f6 + f7;
        let expected_f5 = f4 + f5 * omega1 + f6 * omega2 + f7 * omega3;
        let expected_f6 = f4 + f5 * omega2 + f6 * omega4 + f7 * omega2;
        let expected_f7 = f4 + f5 * omega3 + f6 * omega2 + f7 * omega1;

        // Chunk 3 (f8 to f11)
        let expected_f8 = f8 + f9 + f10 + f11;
        let expected_f9 = f8 + f9 * omega1 + f10 * omega2 + f11 * omega3;
        let expected_f10 = f8 + f9 * omega2 + f10 * omega4 + f11 * omega2;
        let expected_f11 = f8 + f9 * omega3 + f10 * omega2 + f11 * omega1;

        // Chunk 4 (f12 to f15)
        let expected_f12 = f12 + f13 + f14 + f15;
        let expected_f13 = f12 + f13 * omega1 + f14 * omega2 + f15 * omega3;
        let expected_f14 = f12 + f13 * omega2 + f14 * omega4 + f15 * omega2;
        let expected_f15 = f12 + f13 * omega3 + f14 * omega2 + f15 * omega1;

        // Ensure correct NTT ordering
        let expected_values_transposed = vec![
            expected_f0,
            expected_f4,
            expected_f8,
            expected_f12,
            expected_f1,
            expected_f5,
            expected_f9,
            expected_f13,
            expected_f2,
            expected_f6,
            expected_f10,
            expected_f14,
            expected_f3,
            expected_f7,
            expected_f11,
            expected_f15,
        ];

        let computed_values = test_utils::expand_from_coeff(&coeffs, expansion);
        assert_eq!(computed_values, expected_values_transposed);
    }

    #[test]
    fn test_interleaved_rs_encode() {
        use ark_poly::{EvaluationDomain, GeneralEvaluationDomain};
        use ark_std::UniformRand;

        let mut rng = ark_std::test_rng();
        let count = 1 << 20;
        let expansion = 4;
        let folding_factor = 6;

        let eval_domain = GeneralEvaluationDomain::<Field64>::new(count * expansion).unwrap();

        let poly: Vec<_> = (0..count).map(|_| Field64::rand(&mut rng)).collect();

        // Compute things the old way
        let mut expected = test_utils::expand_from_coeff(&poly, expansion);
        test_utils::transform_evaluations(
            &mut expected,
            eval_domain.group_gen_inv(),
            folding_factor,
        );

        let rs = RSDefault;

        // Compute things the new way
        let interleaved_ntt = rs.interleaved_encode(&poly, expansion, folding_factor);
        assert_eq!(expected, interleaved_ntt);
    }
}
