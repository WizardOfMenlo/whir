mod coeffs;
mod evals;
pub mod fold;
pub mod hypercube;
pub mod lagrange_iterator;
mod multilinear;

use ark_ff::Field;

pub use self::{coeffs::CoefficientList, evals::EvaluationsList, multilinear::MultilinearPoint};

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
