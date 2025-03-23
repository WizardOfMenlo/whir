use ark_ff::FftField;

use crate::{
    parameters::FoldingFactor,
    poly_utils::{coeffs::CoefficientList, fold::compute_fold, multilinear::MultilinearPoint},
};

/// Unified context for evaluating STIR queries during a WHIR proof round.
///
/// This enum captures the two strategies used in STIR evaluation:
///
/// - `Naive`: In this mode, the verifier evaluates the folded polynomial using the original oracle
///   evaluations over a coset of the domain.
///
/// - `ProverHelps`: In this mode, the prover provides coefficients directly, and evaluation reduces
///   to computing `f(𝑟)` via multilinear interpolation.
///
/// These modes are unified in this enum to simplify dispatch and centralize logic.
pub(crate) enum StirEvalContext<'a, F: FftField> {
    /// Naive evaluation strategy.
    ///
    /// Requires detailed information about the current folding round and domain.
    Naive {
        /// Total size of the evaluation domain before folding.
        domain_size: usize,
        /// Multiplicative inverse of the domain generator (used for coset offset computation).
        domain_gen_inv: F,
        /// The current round number; used to determine the folding factor.
        round: usize,
        /// The indices corresponding to which cosets in the domain to evaluate.
        stir_challenges_indexes: &'a [usize],
        /// Folding factor configuration, varies per round.
        folding_factor: &'a FoldingFactor,
        /// The folding randomness vector `𝑟`, used in multilinear evaluation.
        folding_randomness: &'a MultilinearPoint<F>,
    },

    /// ProverHelps evaluation strategy.
    ///
    /// Only requires access to the folding randomness, as the prover sends coefficients.
    ProverHelps {
        /// The folding randomness vector `𝑟`.
        folding_randomness: &'a MultilinearPoint<F>,
    },
}

impl<F: FftField> StirEvalContext<'_, F> {
    /// Computes STIR evaluations based on the context strategy.
    ///
    /// # Naive Strategy
    ///
    /// In this mode, we compute the folded evaluation over a coset:
    ///
    /// ```ignore
    /// f_folded(α) = ∑_{i=0}^{2^m - 1} f(ωᵢ · γ) · eᵢ(r)
    /// ```
    ///
    /// where:
    /// - `ωᵢ` are powers of the coset generator,
    /// - `γ` is the coset offset (determined by the query index),
    /// - `r` is the folding randomness vector,
    /// - `eᵢ(r)` are multilinear basis polynomials.
    ///
    /// # ProverHelps Strategy
    ///
    /// In this mode, the prover sends coefficients directly, and we compute:
    ///
    /// ```ignore
    /// f(r) = evaluate(coeffs, r)
    /// ```
    ///
    /// # Arguments
    ///
    /// - `answers`: Oracle values — either raw evaluations (naive) or preprocessed coefficients
    ///   (prover helps).
    /// - `stir_evaluations`: Output vector where the results will be appended.
    pub(crate) fn evaluate(&self, answers: &[Vec<F>], stir_evaluations: &mut Vec<F>) {
        match self {
            Self::Naive {
                domain_size,
                domain_gen_inv,
                round,
                stir_challenges_indexes,
                folding_factor,
                folding_randomness,
            } => {
                // The number of elements in each coset = 2^m for this round
                let coset_domain_size = 1 << folding_factor.at_round(*round);

                // Inverse of the coset generator: w^(N / 2^m), where N is domain_size
                let coset_generator_inv =
                    domain_gen_inv.pow([(domain_size / coset_domain_size) as u64]);

                // Precompute inverse of 2
                let two_inv = F::from(2).inverse().unwrap();

                // Evaluate folded values for each challenge index
                stir_evaluations.extend(stir_challenges_indexes.iter().zip(answers).map(
                    |(index, answers)| {
                        // Compute coset offset: w^index
                        let coset_offset_inv = domain_gen_inv.pow([*index as u64]);

                        // Fold evaluations using randomness `r` and coset info
                        compute_fold(
                            answers,
                            &folding_randomness.0,
                            coset_offset_inv,
                            coset_generator_inv,
                            two_inv,
                            folding_factor.at_round(*round),
                        )
                    },
                ));
            }
            Self::ProverHelps { folding_randomness } => {
                // Directly evaluate each list of coefficients at `r`
                stir_evaluations.extend(answers.iter().map(|answers| {
                    CoefficientList::new(answers.clone()).evaluate(folding_randomness)
                }));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use ark_ff::{AdditiveGroup, Field};

    use super::*;
    use crate::crypto::fields::Field64;

    #[test]
    fn test_stir_eval_prover_helps_basic() {
        // A degree-2 multilinear polynomial over two variables:
        // f(x, y) = 1 + 2y + 3x + 4xy
        let c0 = Field64::from(1);
        let c1 = Field64::from(2);
        let c2 = Field64::from(3);
        let c3 = Field64::from(4);

        let coeffs = vec![
            c0, // f(0,0)
            c1, // f(0,1)
            c2, // f(1,0)
            c3, // f(1,1)
        ];

        // Evaluate at point (5, 6)
        let r0 = Field64::from(5);
        let r1 = Field64::from(6);
        let r = MultilinearPoint(vec![r0, r1]);

        // f(r0, r1)
        let expected = { c0 + c1 * r1 + c2 * r0 + c3 * r0 * r1 };

        let mut evals = Vec::new();

        let context = StirEvalContext::ProverHelps { folding_randomness: &r };
        context.evaluate(&[coeffs], &mut evals);

        assert_eq!(evals, vec![expected]);
    }

    #[test]
    fn test_stir_eval_prover_helps_single_variable() {
        // A single-variable linear polynomial: f(x) = 2 + 5x
        let c0 = Field64::from(2);
        let c1 = Field64::from(5);
        let coeffs = vec![c0, c1];

        let r0 = Field64::from(3);
        let r = MultilinearPoint(vec![r0]); // Evaluate at x = 3

        let expected = c0 + c1 * r0;

        let mut evals = Vec::new();
        let context = StirEvalContext::ProverHelps { folding_randomness: &r };
        context.evaluate(&[coeffs], &mut evals);

        assert_eq!(evals, vec![expected]);
    }

    #[test]
    fn test_stir_eval_prover_helps_zero_polynomial() {
        // Zero polynomial of any degree should evaluate to zero at any point
        let coeffs = vec![Field64::ZERO; 8];
        let r = MultilinearPoint(vec![Field64::from(10), Field64::from(20), Field64::from(30)]);

        let mut evals = Vec::new();
        let context = StirEvalContext::ProverHelps { folding_randomness: &r };
        context.evaluate(&[coeffs], &mut evals);

        assert_eq!(evals, vec![Field64::ZERO]);
    }

    #[test]
    fn test_stir_eval_prover_helps_multiple_points() {
        // Test several different coefficient sets with same randomness
        let coeffs1 = vec![Field64::from(1), Field64::from(0)]; // f(x) = 1
        let coeffs2 = vec![Field64::from(0), Field64::from(1)]; // f(x) = x

        let r = MultilinearPoint(vec![Field64::from(7)]); // Evaluate at x = 7

        let mut evals = Vec::new();
        let context = StirEvalContext::ProverHelps { folding_randomness: &r };
        context.evaluate(&[coeffs1, coeffs2], &mut evals);

        // f1(7) = 1, f2(7) = 7
        assert_eq!(evals, vec![Field64::from(1), Field64::from(7)]);
    }

    #[test]
    fn test_stir_eval_naive_single_variable() {
        // A multilinear polynomial with one variable: f(x) = 2 + 3x
        let f0 = Field64::from(2); // f(0)
        let f1 = Field64::from(5); // f(1)
        let answers = vec![vec![f0, f1]];

        // We will fold over this coset using folding_factor = 1
        let folding_factor = FoldingFactor::Constant(1);
        let round = 0;

        // Domain size must be a multiple of 2^folding_factor
        let domain_size = 2;
        let domain_gen = Field64::from(7);
        let domain_gen_inv = domain_gen.inverse().unwrap();

        // Only one index = 1 → offset = domain_gen^1 = 7 → offset⁻¹ = domain_gen_inv
        let stir_challenges_indexes = [1];

        // Folding randomness `r = [4]`
        let r = Field64::from(4);
        let folding_randomness = MultilinearPoint(vec![r]);

        // Manually compute expected using the fold formula:
        // folding step:
        //   g = (f0 + f1 + r * (f0 - f1) * offset⁻¹ * g⁰⁻¹) / 2
        //     = (f0 + f1 + r * (f0 - f1) * offset⁻¹) / 2
        let two_inv = Field64::from(2).inverse().unwrap();
        let diff = f0 - f1;
        let offset_inv = domain_gen_inv.pow([1]);
        let left = f0 + f1;
        let right = r * diff * offset_inv;
        let expected = two_inv * (left + right);

        let context = StirEvalContext::Naive {
            domain_size,
            domain_gen_inv,
            round,
            stir_challenges_indexes: &stir_challenges_indexes,
            folding_factor: &folding_factor,
            folding_randomness: &folding_randomness,
        };

        let mut evals = Vec::new();
        context.evaluate(&answers, &mut evals);

        assert_eq!(evals, vec![expected]);
    }
}
