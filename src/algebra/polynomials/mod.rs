mod coeffs;
mod evals;
pub mod fold;
pub mod hypercube;
pub mod lagrange_iterator;
mod multilinear;

use ark_ff::Field;

pub use self::{coeffs::CoefficientList, evals::EvaluationsList, multilinear::MultilinearPoint};

/// Spot-check that an `EvaluationsList` matches a `CoefficientList` at a few
/// deterministic boolean hypercube points, without cloning the full polynomial.
///
/// Checks indices 0, n/3, 2n/3, and n-1 (where n = 2^num_variables).
/// Each check is O(n) via `evaluate`, giving O(n) total instead of the O(n log n)
/// wavelet transform + O(n) allocation that a full comparison would require.
///
/// All call sites use `debug_assert!`, so this is dead-code-eliminated in
/// release builds.
pub fn spot_check_evals_eq<F: Field>(
    eval_list: &EvaluationsList<F>,
    coefficients: &CoefficientList<F>,
) -> bool {
    use hypercube::BinaryHypercubePoint;
    let evals = eval_list.evals();
    let n = evals.len();
    let num_vars = coefficients.num_variables();
    for &idx in &[0, n / 3, 2 * n / 3, n - 1] {
        let point =
            MultilinearPoint::from_binary_hypercube_point(BinaryHypercubePoint(idx), num_vars);
        let expected = coefficients.evaluate(&point);
        if evals[idx] != expected {
            return false;
        }
    }
    true
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
