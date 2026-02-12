mod coeffs;
mod evals;
pub mod fold;
pub mod hypercube;
pub mod lagrange_iterator;
mod multilinear;

use ark_ff::Field;

pub use self::{coeffs::CoefficientList, evals::EvaluationsList, multilinear::MultilinearPoint};
use crate::algebra::embedding::{Embedding, Identity};

/// Evaluate the multi-linear extension of `evals` in `point`.
pub fn eval_multilinear<F: Field>(evals: &[F], point: &[F]) -> F {
    mixed_eval_multilinear(&Identity::<F>::new(), evals, point)
}

/// Evaluate the multi-linear extension of `evals` in `point`.
pub fn mixed_eval_multilinear<M: Embedding>(
    embedding: &M,
    evals: &[M::Source],
    point: &[M::Target],
) -> M::Target {
    assert_eq!(evals.len(), 1 << point.len());

    // Helper to compute (a + (b - a) * c) efficiently with a, b in source field.
    let mixed = |a, b, c| embedding.mixed_add(embedding.mixed_mul(c, b - a), a);

    match point {
        [] => embedding.map(evals[0]),
        [x] => mixed(evals[0], evals[1], *x),
        [x0, x1] => {
            let a0 = mixed(evals[0], evals[1], *x1);
            let a1 = mixed(evals[2], evals[3], *x1);
            a0 + (a1 - a0) * *x0
        }
        [x0, x1, x2] => {
            let a00 = mixed(evals[0], evals[1], *x2);
            let a01 = mixed(evals[2], evals[3], *x2);
            let a10 = mixed(evals[4], evals[5], *x2);
            let a11 = mixed(evals[6], evals[7], *x2);
            let a0 = a00 + (a01 - a00) * *x1;
            let a1 = a10 + (a11 - a10) * *x1;
            a0 + (a1 - a0) * *x0
        }
        [x0, x1, x2, x3] => {
            let a000 = mixed(evals[0], evals[1], *x3);
            let a001 = mixed(evals[2], evals[3], *x3);
            let a010 = mixed(evals[4], evals[5], *x3);
            let a011 = mixed(evals[6], evals[7], *x3);
            let a100 = mixed(evals[8], evals[9], *x3);
            let a101 = mixed(evals[10], evals[11], *x3);
            let a110 = mixed(evals[12], evals[13], *x3);
            let a111 = mixed(evals[14], evals[15], *x3);
            let a00 = a000 + (a001 - a000) * *x2;
            let a01 = a010 + (a011 - a010) * *x2;
            let a10 = a100 + (a101 - a100) * *x2;
            let a11 = a110 + (a111 - a110) * *x2;
            let a0 = a00 + (a01 - a00) * *x1;
            let a1 = a10 + (a11 - a10) * *x1;
            a0 + (a1 - a0) * *x0
        }
        [x, tail @ ..] => {
            let (f0, f1) = evals.split_at(evals.len() / 2);
            #[cfg(not(feature = "parallel"))]
            let (f0, f1) = (
                mixed_eval_multilinear(embedding, f0, tail),
                mixed_eval_multilinear(embedding, f1, tail),
            );
            #[cfg(feature = "parallel")]
            let (f0, f1) = {
                use crate::utils::workload_size;
                if evals.len() > workload_size::<M::Source>() {
                    rayon::join(
                        || mixed_eval_multilinear(embedding, f0, tail),
                        || mixed_eval_multilinear(embedding, f1, tail),
                    )
                } else {
                    (
                        mixed_eval_multilinear(embedding, f0, tail),
                        mixed_eval_multilinear(embedding, f1, tail),
                    )
                }
            };
            f0 + (f1 - f0) * *x
        }
    }
}

/// Evaluates the mle of a polynomial from evaluations in a geometric progression.
///
/// The evaluation list is of the form [1,a,a^2,a^3,...,a^{n-1},0,...,0]
/// a is the base of the geometric progression.
/// n is the number of non-zero terms in the progression.
pub fn geometric_till<F: Field>(mut a: F, n: usize, x: &[F]) -> F {
    let k = x.len();
    assert!(n > 0 && n < (1 << k));
    let mut borrow_0 = F::one();
    let mut borrow_1 = F::zero();
    for (i, &x) in x.iter().rev().enumerate() {
        let bn = ((n - 1) >> i) & 1;
        let b0 = F::one() - x;
        let b1 = a * x;
        (borrow_0, borrow_1) = if bn == 0 {
            (b0 * borrow_0, (b0 + b1) * borrow_1 + b1 * borrow_0)
        } else {
            ((b0 + b1) * borrow_0 + b0 * borrow_1, b1 * borrow_1)
        };
        a = a.square();
    }
    borrow_0
}
