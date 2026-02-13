use ark_ff::Field;
#[cfg(feature = "parallel")]
use rayon::{join, prelude::*};

use crate::algebra::embedding::Embedding;
#[cfg(feature = "parallel")]
use crate::utils::workload_size;

/// Computes the constant and quadratic coefficient of the sumcheck polynomial.
pub fn compute_sumcheck_polynomial<F: Field>(a: &[F], b: &[F]) -> (F, F) {
    assert_eq!(a.len(), b.len());

    #[cfg(not(feature = "parallel"))]
    let result = a
        .chunks_exact(2)
        .zip(b.chunks_exact(2))
        .map(|(p_at, eq_at)| {
            // Convert evaluations to coefficients for the linear fns p and eq.
            let (p_0, p_1) = (p_at[0], p_at[1] - p_at[0]);
            let (eq_0, eq_1) = (eq_at[0], eq_at[1] - eq_at[0]);

            // Now we need to add the contribution of p(x) * eq(x)
            (p_0 * eq_0, p_1 * eq_1)
        })
        .fold((F::ZERO, F::ZERO), |(a0, a2), (b0, b2)| (a0 + b0, a2 + b2));

    #[cfg(feature = "parallel")]
    let result = a
        .par_chunks_exact(2)
        .zip(b.par_chunks_exact(2))
        .map(|(p_at, eq_at)| {
            // Convert evaluations to coefficients for the linear fns p and eq.
            let (p_0, p_1) = (p_at[0], p_at[1] - p_at[0]);
            let (eq_0, eq_1) = (eq_at[0], eq_at[1] - eq_at[0]);

            // Now we need to add the contribution of p(x) * eq(x)
            (p_0 * eq_0, p_1 * eq_1)
        })
        .reduce(
            || (F::ZERO, F::ZERO),
            |(a0, a2), (b0, b2)| (a0 + b0, a2 + b2),
        );

    result
}

/// Folds evaluations by linear interpolation at the given weight.
pub fn fold<F: Field>(weight: F, values: &[F]) -> Vec<F> {
    assert!(values.len().is_multiple_of(2));

    #[cfg(not(feature = "parallel"))]
    let result = values
        .chunks_exact(2)
        .map(|w| (w[1] - w[0]) * weight + w[0])
        .collect();

    #[cfg(feature = "parallel")]
    let result = values
        .par_chunks_exact(2)
        .map(|w| (w[1] - w[0]) * weight + w[0])
        .collect();

    result
}

/// Evaluate a coefficient vector at a multilinear point in the target field.
pub fn mixed_eval<M: Embedding>(
    embedding: &M,
    coeff: &[M::Source],
    eval: &[M::Target],
    scalar: M::Target,
) -> M::Target {
    debug_assert_eq!(coeff.len(), 1 << eval.len());

    if let Some((&x, tail)) = eval.split_first() {
        let (low, high) = coeff.split_at(coeff.len() / 2);

        #[cfg(feature = "parallel")]
        if low.len() > workload_size::<M::Source>() {
            let (a, b) = join(
                || mixed_eval(embedding, low, tail, scalar),
                || mixed_eval(embedding, high, tail, scalar * x),
            );
            return a + b;
        }

        mixed_eval(embedding, low, tail, scalar) + mixed_eval(embedding, high, tail, scalar * x)
    } else {
        embedding.mixed_mul(scalar, coeff[0])
    }
}
