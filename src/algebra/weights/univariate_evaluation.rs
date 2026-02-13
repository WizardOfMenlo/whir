use ark_ff::Field;

use super::Weights;
use crate::{
    algebra::{embedding::Embedding, mixed_univariate_evaluate, weights::Evaluate},
    utils::zip_strict,
};

/// Weights vector to represent univariate polynomial evaluation.
///
/// Specifically the weights are $w_i = x^i$.
pub struct UnivariateEvaluation<F: Field> {
    /// Univariate evaluation doesn't have an inherent size, so we need to store one.
    pub size: usize,

    /// The point to evaluate in.
    pub point: F,
}

impl<F: Field> UnivariateEvaluation<F> {
    pub fn new(point: F, size: usize) -> Self {
        Self { size, point }
    }

    /// Same as [`Weights::accumulate`], but batches many [`UnivariateEvaluation`]s together
    /// in a single pass.
    pub fn accumulate_many(weights: &[Self], accumulator: &mut [F], scalars: &[F]) {
        for (weights, &scalar) in zip_strict(weights, scalars) {
            weights.accumulate(accumulator, scalar);
        }

        /*
        assert_eq!(weights.len(), scalars.len());
        let mut powers = scalars.to_vec();
        for accumulator in accumulator {
            for (power, weights) in zip_strict(&mut powers, weights) {
                *accumulator += *power;
                *power *= weights.point;
            }
        }
        */
    }
}

impl<F: Field> Weights<F> for UnivariateEvaluation<F> {
    fn deferred(&self) -> bool {
        false
    }

    fn size(&self) -> usize {
        self.size
    }

    fn mle_evaluate(&self, point: &[F]) -> F {
        point
            .iter()
            .rev() // TODO: Change variable order convention
            .fold((F::ONE, self.point), |(acc, pow2k), &r| {
                (
                    acc * (pow2k * r + (F::ONE - pow2k) * (F::ONE - r)),
                    pow2k.square(),
                )
            })
            .0
    }

    /// See also [`accumulate_many`] for a more efficient batched version.
    fn accumulate(&self, accumulator: &mut [F], scalar: F) {
        assert_eq!(accumulator.len(), self.size);
        assert!(self.size.is_power_of_two());
        let num_variables = self.size.trailing_zeros() as usize;
        let mut point = Vec::with_capacity(num_variables);
        let mut pow2k = self.point;
        point.push(pow2k);
        for _ in 1..num_variables {
            pow2k.square_in_place();
            point.push(pow2k);
        }
        point.reverse();

        eval_eq(accumulator, point.as_slice(), scalar);
    }
}

impl<M: Embedding> Evaluate<M> for UnivariateEvaluation<M::Target> {
    // Evaluate a vector in coefficient form.
    fn evaluate(&self, embedding: &M, vector: &[M::Source]) -> M::Target {
        mixed_univariate_evaluate(embedding, vector, self.point)
    }
}

fn eval_eq<F: Field>(accumulator: &mut [F], point: &[F], scalar: F) {
    assert_eq!(accumulator.len(), 1 << point.len());
    if let [x0, xs @ ..] = point {
        let (acc_0, acc_1) = accumulator.split_at_mut(1 << xs.len());
        let s1 = scalar * x0; // Contribution when `X_i = 1`
        let s0 = scalar - s1; // Contribution when `X_i = 0`

        #[cfg(feature = "parallel")]
        {
            use crate::utils::workload_size;
            if acc_0.len() > workload_size::<F>() {
                rayon::join(|| eval_eq(acc_0, xs, s0), || eval_eq(acc_1, xs, s1));
                return;
            }
        }
        eval_eq(acc_0, xs, s0);
        eval_eq(acc_1, xs, s1);
    } else {
        accumulator[0] += scalar;
    }
}
