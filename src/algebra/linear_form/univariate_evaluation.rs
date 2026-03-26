use ark_ff::Field;

use super::LinearForm;
use crate::algebra::{
    embedding::Embedding, geometric_accumulate, linear_form::Evaluate, mixed_univariate_evaluate,
};

/// Linear form to represent univariate polynomial evaluation.
///
/// Given a vector $v ∈ 𝔽^n$ it computes $sum_i v_i · x^i$ for some fixed $x$.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct UnivariateEvaluation<F: Field> {
    /// Univariate evaluation doesn't have an inherent size, so we need to store one.
    pub size: usize,

    /// The point $x ∈ 𝔽$ to evaluate on.
    pub point: F,
}

impl<F: Field> UnivariateEvaluation<F> {
    pub const fn new(point: F, size: usize) -> Self {
        Self { size, point }
    }

    /// Batched version of [`LinearForm::accumulate`] for many [`UnivariateEvaluation`]s.
    pub fn accumulate_many(evaluators: &[Self], accumulator: &mut [F], scalars: &[F]) {
        assert_eq!(evaluators.len(), scalars.len());
        let Some(size) = evaluators.first().map(|e| e.size) else {
            return;
        };
        assert_eq!(accumulator.len(), size);
        for evaluator in evaluators {
            assert_eq!(evaluator.size, size);
        }
        let points = evaluators.iter().map(|e| e.point).collect::<Vec<F>>();
        let scalars = scalars.to_vec();
        geometric_accumulate(accumulator, scalars, &points);
    }
}

/// Lagrange basis polynomial L_index(point) on {0,1}^n (MSB-first).
fn lagrange_basis_single<F: Field>(point: &[F], index: usize) -> F {
    let n = point.len();
    point.iter().enumerate().fold(F::ONE, |acc, (j, &r)| {
        if (index >> (n - 1 - j)) & 1 == 1 {
            acc * r
        } else {
            acc * (F::ONE - r)
        }
    })
}

impl<F: Field> LinearForm<F> for UnivariateEvaluation<F> {
    fn size(&self) -> usize {
        self.size
    }

    fn mle_evaluate(&self, point: &[F]) -> F {
        let k = self.size.trailing_zeros() as usize;
        let extra = point.len().saturating_sub(k);

        if extra == 0 {
            // Power-of-2 path: MLE of (1, x, x^2, ..) = ⊗_i (1, x^{2^i}).
            let mut x2i = self.point;
            let mut result = F::ONE;
            for &r in point.iter().rev() {
                result *= (F::ONE - r) + r * x2i;
                x2i.square_in_place();
            }
            return result;
        }

        // Smooth path: size = 2^k * odd where odd = 3^b * 13^c.
        let odd = self.size >> k;
        let leading = &point[..extra];
        let trailing = &point[extra..];

        let mut x2i = self.point;
        let mut inner = F::ONE;
        for &r in trailing.iter().rev() {
            inner *= (F::ONE - r) + r * x2i;
            x2i.square_in_place();
        }

        let x_pow_2k = x2i;
        let mut outer = F::ZERO;
        let mut x_h = F::ONE;
        for h in 0..odd {
            outer += lagrange_basis_single(leading, h) * x_h;
            x_h *= x_pow_2k;
        }

        outer * inner
    }

    /// See also [`Self::accumulate_many`] for a more efficient batched version.
    fn accumulate(&self, accumulator: &mut [F], scalar: F) {
        assert_eq!(accumulator.len(), self.size);
        let mut power = scalar;
        for entry in accumulator {
            *entry += power;
            power *= self.point;
        }
    }
}

impl<M: Embedding> Evaluate<M> for UnivariateEvaluation<M::Target> {
    fn evaluate(&self, embedding: &M, vector: &[M::Source]) -> M::Target {
        mixed_univariate_evaluate(embedding, vector, self.point)
    }
}
