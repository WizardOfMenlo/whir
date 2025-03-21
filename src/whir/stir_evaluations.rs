use crate::{
    parameters::FoldingFactor,
    poly_utils::{coeffs::CoefficientList, fold::compute_fold, multilinear::MultilinearPoint},
};
use ark_ff::FftField;

/// Unified context for evaluating STIR queries during a WHIR proof round.
///
/// This enum captures the two strategies used in STIR evaluation:
///
/// - `Naive`: In this mode, the verifier evaluates the folded polynomial using
///   the original oracle evaluations over a coset of the domain.
///
/// - `ProverHelps`: In this mode, the prover provides coefficients directly,
///   and evaluation reduces to computing `f(𝑟)` via multilinear interpolation.
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
    /// - `answers`: Oracle values — either raw evaluations (naive) or preprocessed coefficients (prover helps).
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
