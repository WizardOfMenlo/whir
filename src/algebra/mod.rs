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
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// Mixed field univariate Horner evaluation.
pub fn mixed_univariate_evaluate<F: Field, G: Field>(
    embedding: &impl Embedding<Source = F, Target = G>,
    coefficients: &[F],
    point: G,
) -> G {
    coefficients.iter().rev().fold(G::ZERO, |acc, &coeff| {
        embedding.mixed_add(acc * point, coeff)
    })
}

pub fn mixed_dot<F: Field, G: Field>(
    embedding: &impl Embedding<Source = F, Target = G>,
    a: &[G],
    b: &[F],
) -> G {
    a.iter()
        .zip(b.iter())
        .map(|(a, b)| embedding.mixed_mul(*a, *b))
        .sum()
}
