use ark_ff::Field;

/// Compute
#[cfg(feature = "parallel")]
pub fn compute_sumcheck_polynomial<F: Field>(a: &[F], b: &[F]) -> (F, F) {
    use rayon::prelude::*;

    assert_eq!(a.len(), b.len());
    a.par_chunks_exact(2)
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
        )
}

#[cfg(not(feature = "parallel"))]
pub fn compute_sumcheck_polynomial<F: Field>(a: &[F], b: &[F]) -> (F, F) {
    assert_eq!(a.len(), b.len());
    a.chunks_exact(2)
        .zip(b.chunks_exact(2))
        .map(|(p_at, eq_at)| {
            // Convert evaluations to coefficients for the linear fns p and eq.
            let (p_0, p_1) = (p_at[0], p_at[1] - p_at[0]);
            let (eq_0, eq_1) = (eq_at[0], eq_at[1] - eq_at[0]);

            // Now we need to add the contribution of p(x) * eq(x)
            (p_0 * eq_0, p_1 * eq_1)
        })
        .fold((F::ZERO, F::ZERO), |(a0, a2), (b0, b2)| (a0 + b0, a2 + b2))
}

pub fn fold<F: Field>(weight: F, values: &[F]) -> Vec<F> {
    assert!(values.len().is_multiple_of(2));
    values
        .chunks_exact(2)
        .map(|w| (w[1] - w[0]) * weight + w[0])
        .collect()
}
