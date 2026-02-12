pub mod domain;
pub mod embedding;
pub mod fields;
mod multilinear;
pub mod ntt;
mod old_weights;
pub mod polynomials;
pub mod sumcheck;
pub mod weights;

use ark_ff::{AdditiveGroup, Field};
pub use multilinear::{eval_eq, mixed_multilinear_extend, multilinear_extend};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use self::embedding::Embedding;
pub use self::old_weights::OldWeights;
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

/// Lift a vector to an embedding.
pub fn lift<M: Embedding>(embedding: &M, source: &[M::Source]) -> Vec<M::Target> {
    #[cfg(not(feature = "parallel"))]
    let result = source.iter().map(|c| embedding.map(*c)).collect();

    #[cfg(feature = "parallel")]
    let result = source.par_iter().map(|c| embedding.map(*c)).collect();

    result
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

pub fn univariate_evaluate<F: Field>(coefficients: &[F], point: F) -> F {
    mixed_univariate_evaluate(&embedding::Identity::new(), coefficients, point)
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
