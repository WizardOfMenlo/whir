pub mod domain;
pub mod embedding;
pub mod fields;
pub mod ntt;
pub mod poly_utils;

use ark_ff::Field;

use self::embedding::Embedding;

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
    assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
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
    coefficients.iter().rev().skip(1).fold(
        coefficients.last().copied().unwrap_or(F::ZERO),
        |acc, &coeff| acc * point + coeff,
    )
}

/// Mixed field univariate Horner evaluation.
pub fn mixed_univariate_evaluate<F: Field, G: Field>(
    embedding: &impl Embedding<Source = F, Target = G>,
    coefficients: &[F],
    point: G,
) -> G {
    coefficients.iter().rev().skip(1).fold(
        coefficients.last().map_or(G::ZERO, |f| embedding.map(*f)),
        |acc, &coeff| embedding.mixed_add(acc * point, coeff),
    )
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
