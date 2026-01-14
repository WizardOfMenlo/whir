use std::ops::Index;

use ark_ff::Field;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use serde::{Deserialize, Serialize};

use super::{lagrange_iterator::LagrangePolynomialIterator, multilinear::MultilinearPoint};

/// Represents a multilinear polynomial `f` in `num_variables` unknowns, stored via its evaluations
/// over the hypercube `{0,1}^{num_variables}`.
///
/// The vector `evals` contains function evaluations at **lexicographically ordered** points.
#[derive(Default, Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "F: CanonicalSerialize + CanonicalDeserialize")]
pub struct EvaluationsList<F> {
    /// Stores evaluations in **lexicographic order**.
    #[serde(with = "crate::ark_serde")]
    evals: Vec<F>,
    /// Number of variables in the multilinear polynomial.
    /// Ensures `evals.len() = 2^{num_variables}`.
    num_variables: usize,
}

impl<F> EvaluationsList<F>
where
    F: Field,
{
    /// Constructs an `EvaluationsList` from a given vector of evaluations.
    ///
    /// - The `evals` vector must have a **length that is a power of two** since it represents
    ///   evaluations over an `n`-dimensional binary hypercube.
    /// - The ordering of evaluation points follows **lexicographic order**.
    ///
    /// **Mathematical Constraint:**
    /// If `evals.len() = 2^n`, then `num_variables = n`, ensuring correct indexing.
    ///
    /// **Panics:**
    /// - If `evals.len()` is **not** a power of two.
    pub fn new(evals: Vec<F>) -> Self {
        let len = evals.len();
        assert!(
            len.is_power_of_two(),
            "Evaluation list length must be a power of two."
        );

        Self {
            evals,
            num_variables: len.ilog2() as usize,
        }
    }

    /// Evaluates the polynomial at a given multilinear point.
    ///
    /// - If `point` belongs to the binary hypercube `{0,1}^n`, we directly return the precomputed
    ///   evaluation.
    /// - Otherwise, we **reconstruct** the evaluation using Lagrange interpolation.
    ///
    /// Mathematical definition:
    /// Given evaluations `f(x)` stored in `evals`, we compute:
    ///
    /// ```ignore
    /// f(p) = Σ_{x ∈ {0,1}^n} eq(x, p) * f(x)
    /// ```
    ///
    /// where `eq(x, p)` is the Lagrange basis polynomial.
    pub fn evaluate(&self, point: &MultilinearPoint<F>) -> F {
        if let Some(binary_index) = point.to_hypercube() {
            return self.evals[binary_index.0];
        }

        self.evals
            .iter()
            .zip(LagrangePolynomialIterator::from(point))
            .map(|(eval, (_, lag))| *eval * lag)
            .sum()
    }

    pub fn eval_extension(&self, point: &MultilinearPoint<F>) -> F {
        if let Some(point) = point.to_hypercube() {
            return self.evals[point.0];
        }
        eval_multilinear(&self.evals, &point.0)
    }

    /// Returns an immutable reference to the evaluations vector.
    #[allow(clippy::missing_const_for_fn)]
    pub fn evals(&self) -> &[F] {
        &self.evals
    }

    /// Returns a mutable reference to the evaluations vector.
    #[allow(clippy::missing_const_for_fn)]
    pub fn evals_mut(&mut self) -> &mut [F] {
        &mut self.evals
    }

    /// Returns the total number of stored evaluations.
    ///
    /// Mathematical Invariant:
    /// ```ignore
    /// num_evals = 2^{num_variables}
    /// ```
    pub fn num_evals(&self) -> usize {
        self.evals.len()
    }

    /// Returns the number of variables in the multilinear polynomial.
    pub const fn num_variables(&self) -> usize {
        self.num_variables
    }

    pub fn to_coeffs(&self) -> crate::poly_utils::coeffs::CoefficientList<F> {
        let mut coeffs = self.evals.clone();
        crate::ntt::inverse_wavelet_transform(&mut coeffs);
        crate::poly_utils::coeffs::CoefficientList::new(coeffs)
    }
}

impl<F> Index<usize> for EvaluationsList<F> {
    type Output = F;

    fn index(&self, index: usize) -> &Self::Output {
        &self.evals[index]
    }
}

fn eval_multilinear<F: Field>(evals: &[F], point: &[F]) -> F {
    debug_assert_eq!(evals.len(), 1 << point.len());
    match point {
        [] => evals[0],
        [x] => evals[0] + (evals[1] - evals[0]) * *x,
        #[cfg(not(feature = "small-stack"))]
        [x0, x1] => {
            let a0 = evals[0] + (evals[1] - evals[0]) * *x1;
            let a1 = evals[2] + (evals[3] - evals[2]) * *x1;
            a0 + (a1 - a0) * *x0
        }
        #[cfg(not(feature = "small-stack"))]
        [x0, x1, x2] => {
            let a00 = evals[0] + (evals[1] - evals[0]) * *x2;
            let a01 = evals[2] + (evals[3] - evals[2]) * *x2;
            let a10 = evals[4] + (evals[5] - evals[4]) * *x2;
            let a11 = evals[6] + (evals[7] - evals[6]) * *x2;
            let a0 = a00 + (a01 - a00) * *x1;
            let a1 = a10 + (a11 - a10) * *x1;
            a0 + (a1 - a0) * *x0
        }
        #[cfg(not(feature = "small-stack"))]
        [x0, x1, x2, x3] => {
            let a000 = evals[0] + (evals[1] - evals[0]) * *x3;
            let a001 = evals[2] + (evals[3] - evals[2]) * *x3;
            let a010 = evals[4] + (evals[5] - evals[4]) * *x3;
            let a011 = evals[6] + (evals[7] - evals[6]) * *x3;
            let a100 = evals[8] + (evals[9] - evals[8]) * *x3;
            let a101 = evals[10] + (evals[11] - evals[10]) * *x3;
            let a110 = evals[12] + (evals[13] - evals[12]) * *x3;
            let a111 = evals[14] + (evals[15] - evals[14]) * *x3;
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
            let (f0, f1) = (eval_multilinear(f0, tail), eval_multilinear(f1, tail));
            #[cfg(feature = "parallel")]
            let (f0, f1) = {
                let work_size: usize = (1 << 15) / std::mem::size_of::<F>();
                if evals.len() > work_size {
                    rayon::join(|| eval_multilinear(f0, tail), || eval_multilinear(f1, tail))
                } else {
                    (eval_multilinear(f0, tail), eval_multilinear(f1, tail))
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

#[cfg(test)]
#[allow(clippy::should_panic_without_expect)]
mod tests {
    use ark_ff::AdditiveGroup;

    use super::*;
    use crate::{crypto::fields::Field64, poly_utils::hypercube::BinaryHypercube};

    #[test]
    fn test_new_evaluations_list() {
        let evals = vec![Field64::ZERO, Field64::ONE, Field64::ZERO, Field64::ONE];
        let evaluations_list = EvaluationsList::new(evals.clone());

        assert_eq!(evaluations_list.num_evals(), evals.len());
        assert_eq!(evaluations_list.num_variables(), 2);
        assert_eq!(evaluations_list.evals(), &evals);
    }

    #[test]
    #[should_panic]
    fn test_new_evaluations_list_invalid_length() {
        // Length is not a power of two, should panic
        let _ = EvaluationsList::new(vec![Field64::ONE, Field64::ZERO, Field64::ONE]);
    }

    #[test]
    fn test_indexing() {
        let evals = vec![
            Field64::from(1),
            Field64::from(2),
            Field64::from(3),
            Field64::from(4),
        ];
        let evaluations_list = EvaluationsList::new(evals.clone());

        assert_eq!(evaluations_list[0], evals[0]);
        assert_eq!(evaluations_list[1], evals[1]);
        assert_eq!(evaluations_list[2], evals[2]);
        assert_eq!(evaluations_list[3], evals[3]);
    }

    #[test]
    #[should_panic]
    fn test_index_out_of_bounds() {
        let evals = vec![Field64::ZERO, Field64::ONE, Field64::ZERO, Field64::ONE];
        let evaluations_list = EvaluationsList::new(evals);

        let _ = evaluations_list[4]; // Index out of range, should panic
    }

    #[test]
    fn test_mutability_of_evals() {
        let mut evals = EvaluationsList::new(vec![
            Field64::ZERO,
            Field64::ONE,
            Field64::ZERO,
            Field64::ONE,
        ]);

        assert_eq!(evals.evals()[1], Field64::ONE);

        evals.evals_mut()[1] = Field64::from(5);

        assert_eq!(evals.evals()[1], Field64::from(5));
    }

    #[test]
    fn test_evaluate_on_hypercube_points() {
        let evaluations_vec = vec![Field64::ZERO, Field64::ONE, Field64::ZERO, Field64::ONE];
        let evals = EvaluationsList::new(evaluations_vec.clone());

        for i in BinaryHypercube::new(2) {
            assert_eq!(
                evaluations_vec[i.0],
                evals.evaluate(&MultilinearPoint::from_binary_hypercube_point(i, 2))
            );
        }
    }

    #[test]
    fn test_evaluate_on_non_hypercube_points() {
        let evals = EvaluationsList::new(vec![
            Field64::from(1),
            Field64::from(2),
            Field64::from(3),
            Field64::from(4),
        ]);

        let point = MultilinearPoint(vec![Field64::from(2), Field64::from(3)]);

        let result = evals.evaluate(&point);

        // The result should be computed using Lagrange interpolation.
        let expected = LagrangePolynomialIterator::from(&point)
            .map(|(b, lag)| lag * evals[b.0])
            .sum();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_evaluate_edge_cases() {
        let e1 = Field64::from(7);
        let e2 = Field64::from(8);
        let e3 = Field64::from(9);
        let e4 = Field64::from(10);

        let evals = EvaluationsList::new(vec![e1, e2, e3, e4]);

        // Evaluating at a binary hypercube point should return the direct value
        assert_eq!(
            evals.evaluate(&MultilinearPoint(vec![Field64::ZERO, Field64::ZERO])),
            e1
        );
        assert_eq!(
            evals.evaluate(&MultilinearPoint(vec![Field64::ZERO, Field64::ONE])),
            e2
        );
        assert_eq!(
            evals.evaluate(&MultilinearPoint(vec![Field64::ONE, Field64::ZERO])),
            e3
        );
        assert_eq!(
            evals.evaluate(&MultilinearPoint(vec![Field64::ONE, Field64::ONE])),
            e4
        );
    }

    #[test]
    fn test_num_evals() {
        let evals = EvaluationsList::new(vec![
            Field64::ONE,
            Field64::ZERO,
            Field64::ONE,
            Field64::ZERO,
        ]);
        assert_eq!(evals.num_evals(), 4);
    }

    #[test]
    fn test_num_variables() {
        let evals = EvaluationsList::new(vec![
            Field64::ONE,
            Field64::ZERO,
            Field64::ONE,
            Field64::ZERO,
        ]);
        assert_eq!(evals.num_variables(), 2);
    }

    #[test]
    fn test_eval_extension_on_hypercube_points() {
        let evals = vec![
            Field64::from(1),
            Field64::from(2),
            Field64::from(3),
            Field64::from(4),
        ];
        let eval_list = EvaluationsList::new(evals.clone());

        for i in BinaryHypercube::new(2) {
            assert_eq!(
                eval_list.eval_extension(&MultilinearPoint::from_binary_hypercube_point(i, 2)),
                evals[i.0]
            );
        }
    }

    #[test]
    fn test_eval_extension_on_non_hypercube_points() {
        let evals = EvaluationsList::new(vec![
            Field64::from(1),
            Field64::from(2),
            Field64::from(3),
            Field64::from(4),
        ]);

        let point = MultilinearPoint(vec![Field64::from(2), Field64::from(3)]);

        let result = evals.eval_extension(&point);

        // Expected result using `eval_multilinear`
        let expected = eval_multilinear(evals.evals(), &point.0);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_eval_multilinear_1d() {
        let a = Field64::from(5);
        let b = Field64::from(10);
        let evals = vec![a, b];

        // Evaluate at midpoint `x = 1/2`
        let x = Field64::from(1) / Field64::from(2);
        let expected = a + (b - a) * x;

        assert_eq!(eval_multilinear(&evals, &[x]), expected);
    }

    #[test]
    fn test_eval_multilinear_2d() {
        let a = Field64::from(1);
        let b = Field64::from(2);
        let c = Field64::from(3);
        let d = Field64::from(4);

        // The evaluations are stored in lexicographic order for (x, y)
        // f(0,0) = a, f(0,1) = c, f(1,0) = b, f(1,1) = d
        let evals = vec![a, b, c, d];

        // Evaluate at `(x, y) = (1/2, 1/2)`
        let x = Field64::from(1) / Field64::from(2);
        let y = Field64::from(1) / Field64::from(2);

        // Interpolation formula:
        // f(x, y) = (1-x)(1-y) * f(0,0) + (1-x)y * f(0,1) + x(1-y) * f(1,0) + xy * f(1,1)
        let expected = (Field64::ONE - x) * (Field64::ONE - y) * a
            + (Field64::ONE - x) * y * c
            + x * (Field64::ONE - y) * b
            + x * y * d;

        assert_eq!(eval_multilinear(&evals, &[x, y]), expected);
    }

    #[test]
    fn test_eval_multilinear_3d() {
        let a = Field64::from(1);
        let b = Field64::from(2);
        let c = Field64::from(3);
        let d = Field64::from(4);
        let e = Field64::from(5);
        let f = Field64::from(6);
        let g = Field64::from(7);
        let h = Field64::from(8);

        // The evaluations are stored in lexicographic order for (x, y, z)
        // f(0,0,0) = a, f(0,0,1) = c, f(0,1,0) = b, f(0,1,1) = e
        // f(1,0,0) = d, f(1,0,1) = f, f(1,1,0) = g, f(1,1,1) = h
        let evals = vec![a, b, c, e, d, f, g, h];

        let x = Field64::from(1) / Field64::from(3);
        let y = Field64::from(1) / Field64::from(3);
        let z = Field64::from(1) / Field64::from(3);

        // Using trilinear interpolation formula:
        let expected = (Field64::ONE - x) * (Field64::ONE - y) * (Field64::ONE - z) * a
            + (Field64::ONE - x) * (Field64::ONE - y) * z * c
            + (Field64::ONE - x) * y * (Field64::ONE - z) * b
            + (Field64::ONE - x) * y * z * e
            + x * (Field64::ONE - y) * (Field64::ONE - z) * d
            + x * (Field64::ONE - y) * z * f
            + x * y * (Field64::ONE - z) * g
            + x * y * z * h;

        assert_eq!(eval_multilinear(&evals, &[x, y, z]), expected);
    }

    #[test]
    fn test_eval_multilinear_4d() {
        let a = Field64::from(1);
        let b = Field64::from(2);
        let c = Field64::from(3);
        let d = Field64::from(4);
        let e = Field64::from(5);
        let f = Field64::from(6);
        let g = Field64::from(7);
        let h = Field64::from(8);
        let i = Field64::from(9);
        let j = Field64::from(10);
        let k = Field64::from(11);
        let l = Field64::from(12);
        let m = Field64::from(13);
        let n = Field64::from(14);
        let o = Field64::from(15);
        let p = Field64::from(16);

        // Evaluations stored in lexicographic order for (x, y, z, w)
        let evals = vec![a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p];

        let x = Field64::from(1) / Field64::from(2);
        let y = Field64::from(2) / Field64::from(3);
        let z = Field64::from(1) / Field64::from(4);
        let w = Field64::from(3) / Field64::from(5);

        // Quadlinear interpolation formula
        let expected =
            (Field64::ONE - x) * (Field64::ONE - y) * (Field64::ONE - z) * (Field64::ONE - w) * a
                + (Field64::ONE - x) * (Field64::ONE - y) * (Field64::ONE - z) * w * b
                + (Field64::ONE - x) * (Field64::ONE - y) * z * (Field64::ONE - w) * c
                + (Field64::ONE - x) * (Field64::ONE - y) * z * w * d
                + (Field64::ONE - x) * y * (Field64::ONE - z) * (Field64::ONE - w) * e
                + (Field64::ONE - x) * y * (Field64::ONE - z) * w * f
                + (Field64::ONE - x) * y * z * (Field64::ONE - w) * g
                + (Field64::ONE - x) * y * z * w * h
                + x * (Field64::ONE - y) * (Field64::ONE - z) * (Field64::ONE - w) * i
                + x * (Field64::ONE - y) * (Field64::ONE - z) * w * j
                + x * (Field64::ONE - y) * z * (Field64::ONE - w) * k
                + x * (Field64::ONE - y) * z * w * l
                + x * y * (Field64::ONE - z) * (Field64::ONE - w) * m
                + x * y * (Field64::ONE - z) * w * n
                + x * y * z * (Field64::ONE - w) * o
                + x * y * z * w * p;

        // Validate against the function output
        assert_eq!(eval_multilinear(&evals, &[x, y, z, w]), expected);
    }
}
