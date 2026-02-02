pub mod domain;
pub mod embedding;
pub mod fields;
pub mod ntt;
pub mod poly_utils;

use ark_ff::{AdditiveGroup, Field};

use self::embedding::Embedding;
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
    a.iter()
        .zip(b.iter())
        .map(|(a, b)| embedding.mixed_mul(*a, *b))
        .sum()
}
