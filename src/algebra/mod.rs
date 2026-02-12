pub mod domain;
pub mod embedding;
pub mod fields;
pub mod ntt;
pub mod polynomials;
pub mod sumcheck;
mod weights;

use ark_ff::{AdditiveGroup, FftField, Field};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use self::embedding::Embedding;
pub use self::weights::Weights;
#[cfg(feature = "parallel")]
use crate::utils::workload_size;

pub fn geometric_sequence<F: Field>(base: F, length: usize) -> Vec<F> {
    let mut result = Vec::with_capacity(length);
    let mut current = F::ONE;
    for _ in 0..length {
        result.push(current);
        current *= base;
    }
    result
}

pub fn dot<F: Field>(a: &[F], b: &[F]) -> F {
    mixed_dot(&embedding::Identity::new(), a, b)
}

pub fn tensor_product<F: Field>(a: &[F], b: &[F]) -> Vec<F> {
    let mut result = Vec::with_capacity(a.len() * b.len());
    for &x in a {
        for &y in b {
            result.push(x * y);
        }
    }
    result
}

pub fn univariate_evaluate<F: Field>(coefficients: &[F], point: F) -> F {
    mixed_univariate_evaluate(&embedding::Identity::new(), coefficients, point)
}

/// Lift a vector to an embedding.
pub fn lift<M: Embedding>(embedding: &M, source: &[M::Source]) -> Vec<M::Target> {
    #[cfg(not(feature = "parallel"))]
    let result = source.iter().map(|c| embedding.map(*c)).collect();

    #[cfg(feature = "parallel")]
    let result = source.par_iter().map(|c| embedding.map(*c)).collect();

    result
}

/// Scalar-mul add (same-field AXPY)
///
/// accumulator[i] += weight * vector[i]
pub fn scalar_mul_add<F: Field>(accumulator: &mut [F], weight: F, vector: &[F]) {
    mixed_scalar_mul_add(&embedding::Identity::new(), accumulator, weight, vector);
}

/// Mixed scalar-mul add
///
/// accumulator[i] += weight * vector[i]
pub fn mixed_scalar_mul_add<M: Embedding>(
    embedding: &M,
    accumulator: &mut [M::Target],
    weight: M::Target,
    vector: &[M::Source],
) {
    assert_eq!(accumulator.len(), vector.len());
    for (accumulator, value) in accumulator.iter_mut().zip(vector) {
        *accumulator += embedding.mixed_mul(weight, *value);
    }
}

/// Mixed field univariate Horner evaluation.
pub fn mixed_univariate_evaluate<M: Embedding>(
    embedding: &M,
    coefficients: &[M::Source],
    point: M::Target,
) -> M::Target {
    #[cfg(feature = "parallel")]
    if coefficients.len() > workload_size::<M::Source>() {
        let half = coefficients.len() / 2;
        let (low, high) = coefficients.split_at(half);
        let (low, high) = rayon::join(
            || mixed_univariate_evaluate(embedding, low, point),
            || mixed_univariate_evaluate(embedding, high, point),
        );
        return low + high * point.pow([half as u64]);
    }

    let Some(mut acc) = coefficients.last().map(|c| embedding.map(*c)) else {
        return M::Target::ZERO;
    };
    for &c in coefficients.iter().rev().skip(1) {
        acc *= point;
        acc = embedding.mixed_add(acc, c);
    }
    acc
}

pub fn mixed_dot<F: Field, G: Field>(
    embedding: &impl Embedding<Source = F, Target = G>,
    a: &[G],
    b: &[F],
) -> G {
    assert_eq!(a.len(), b.len());

    #[cfg(not(feature = "parallel"))]
    let result = a
        .iter()
        .zip(b)
        .map(|(a, b)| embedding.mixed_mul(*a, *b))
        .sum();

    #[cfg(feature = "parallel")]
    let result = a
        .par_iter()
        .zip(b)
        .map(|(a, b)| embedding.mixed_mul(*a, *b))
        .sum();

    result
}

/// Project an extension field element to its base prime field component.
///
/// Panics if the element does not lie in the base prime subfield.
#[inline]
pub fn project_to_base<F: Field>(val: F) -> F::BasePrimeField {
    val.to_base_prime_field_elements()
        .next()
        .expect("element should lie in base prime subfield")
}

/// Project every element of an extension-field slice to the base prime field.
///
/// Panics if any element does not lie in the base prime subfield.
pub fn project_all_to_base<F: FftField>(coeffs: &[F]) -> Vec<F::BasePrimeField> {
    #[cfg(feature = "parallel")]
    {
        coeffs.par_iter().map(|c| project_to_base(*c)).collect()
    }
    #[cfg(not(feature = "parallel"))]
    {
        coeffs.iter().map(|&c| project_to_base(c)).collect()
    }
}

/// Element-wise add a base-field slice with a (possibly shorter) extension-field
/// slice projected to base field.
///
/// Computes `result[i] = base[i] + project_to_base(ext[i])` for `i < ext.len()`,
/// and `result[i] = base[i]` for `i >= ext.len()`.
///
/// Each element of `ext` must lie in the base prime subfield.
pub fn add_base_with_projection<F: FftField>(
    base: &[F::BasePrimeField],
    ext_addend: &[F],
) -> Vec<F::BasePrimeField> {
    debug_assert!(
        ext_addend.len() <= base.len(),
        "ext_addend ({}) must not exceed base ({})",
        ext_addend.len(),
        base.len(),
    );
    let ext_len = ext_addend.len();

    #[cfg(feature = "parallel")]
    {
        (0..base.len())
            .into_par_iter()
            .map(|i| {
                if i < ext_len {
                    base[i] + project_to_base(ext_addend[i])
                } else {
                    base[i]
                }
            })
            .collect()
    }
    #[cfg(not(feature = "parallel"))]
    {
        (0..base.len())
            .map(|i| {
                if i < ext_len {
                    base[i] + project_to_base(ext_addend[i])
                } else {
                    base[i]
                }
            })
            .collect()
    }
}
