use ark_ff::Field;
#[cfg(feature = "parallel")]
use rayon::join;

use crate::algebra::embedding::Embedding;
#[cfg(feature = "parallel")]
use crate::utils::workload_size;

/// Computes the constant and quadratic coefficient of the sumcheck polynomial.
pub fn compute_sumcheck_polynomial<F: Field>(a: &[F], b: &[F]) -> (F, F) {
    assert_eq!(a.len(), b.len());
    let half = a.len() / 2;

    fn recurse<F: Field>(a0: &[F], a1: &[F], b0: &[F], b1: &[F]) -> (F, F) {
        debug_assert_eq!(a0.len(), a1.len());
        debug_assert_eq!(b0.len(), b1.len());
        debug_assert_eq!(a0.len(), b0.len());

        #[cfg(feature = "parallel")]
        if a0.len() * 4 > workload_size::<F>() {
            let mid = a0.len() / 2;
            let (a0l, a0r) = a0.split_at(mid);
            let (a1l, a1r) = a1.split_at(mid);
            let (b0l, b0r) = b0.split_at(mid);
            let (b1l, b1r) = b1.split_at(mid);
            let (left, right) = join(
                || recurse(a0l, a1l, b0l, b1l),
                || recurse(a0r, a1r, b0r, b1r),
            );
            return (left.0 + right.0, left.1 + right.1);
        }

        let mut acc0 = F::ZERO;
        let mut acc2 = F::ZERO;
        for ((&p0, &p1), (&eq0, &eq1)) in a0.iter().zip(a1).zip(b0.iter().zip(b1)) {
            acc0 += p0 * eq0;
            acc2 += (p1 - p0) * (eq1 - eq0);
        }
        (acc0, acc2)
    }

    let (a0, a1) = a.split_at(half);
    let (b0, b1) = b.split_at(half);
    recurse(a0, a1, b0, b1)
}

/// Folds evaluations by linear interpolation at the given weight, in place.
pub fn fold<F: Field>(values: &mut Vec<F>, weight: F) {
    assert!(values.len().is_multiple_of(2));
    let half = values.len() / 2;

    fn recurse<F: Field>(low: &mut [F], high: &[F], weight: F) {
        #[cfg(feature = "parallel")]
        if low.len() > workload_size::<F>() {
            let split = low.len() / 2;
            let (ll, lr) = low.split_at_mut(split);
            let (hl, hr) = high.split_at(split);
            rayon::join(|| recurse(ll, hl, weight), || recurse(lr, hr, weight));
            return;
        }

        for (low, high) in low.iter_mut().zip(high) {
            *low += (*high - *low) * weight;
        }
    }

    let (low, high) = values.split_at_mut(half);
    recurse(low, high, weight);
    values.truncate(half);
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
