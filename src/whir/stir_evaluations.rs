use crate::{
    parameters::FoldingFactor,
    poly_utils::{coeffs::CoefficientList, fold::compute_fold, multilinear::MultilinearPoint},
};
use ark_ff::FftField;

/// Context for evaluating STIR queries during a WHIR proof round.
///
/// This holds all the information needed to compute evaluations
/// at queried points.
pub(crate) struct StirEvalContext<'a, F: FftField> {
    /// The domain size before folding.
    pub(crate) domain_size: Option<usize>,
    /// The domain used for the evaluations.
    pub(crate) domain_gen_inv: Option<F>,
    /// The folding randomness vector $\vec{r}$.
    pub(crate) folding_randomness: &'a MultilinearPoint<F>,
    /// The current round number `i`, which determines the folding factor.
    pub(crate) round: Option<usize>,
}

impl<F: FftField> StirEvalContext<'_, F> {
    /// Computes STIR evaluations using the naive folding strategy.
    ///
    /// This method computes, for each query, the folded value of a multilinear
    /// polynomial using its evaluations on a coset:
    ///
    ///   $f_{\text{folded}}(\alpha) = \sum_{i=0}^{2^m - 1} f(\omega_i \cdot \gamma) \cdot e_i(\vec{r})$
    ///
    /// where:
    /// - $\omega_i$ are powers of the coset generator (a root of unity),
    /// - $\gamma$ is the coset offset for that query,
    /// - $\vec{r}$ is the folding randomness,
    /// - $e_i(\vec{r})$ is the multilinear basis polynomial.
    ///
    /// The result is appended into `stir_evaluations`.
    pub(crate) fn stir_evaluations_naive(
        &self,
        stir_challenges_indexes: &[usize],
        answers: &[Vec<F>],
        folding_factor: &FoldingFactor,
        stir_evaluations: &mut Vec<F>,
    ) where
        F: FftField,
    {
        let round = self.round.unwrap();
        let domain_size = self.domain_size.unwrap();
        let domain_gen_inv = self.domain_gen_inv.unwrap();

        let coset_domain_size = 1 << folding_factor.at_round(round);
        let coset_generator_inv = domain_gen_inv.pow([(domain_size / coset_domain_size) as u64]);
        let two_inv = F::from(2).inverse().unwrap();

        stir_evaluations.extend(stir_challenges_indexes.iter().zip(answers).map(
            |(index, answers)| {
                let coset_offset_inv = domain_gen_inv.pow([*index as u64]);

                compute_fold(
                    answers,
                    &self.folding_randomness.0,
                    coset_offset_inv,
                    coset_generator_inv,
                    two_inv,
                    folding_factor.at_round(round),
                )
            },
        ));
    }

    /// Computes STIR evaluations using the "Prover Helps" strategy.
    ///
    /// Assumes the oracle values sent by the prover are already linearly
    /// transformed into coefficient form. The evaluation becomes simply:
    ///
    ///   $f(\vec{r}) = \text{eval}(\text{coeffs}, \vec{r})$
    ///
    /// where the folding randomness $\vec{r}$ is applied directly to the
    /// preprocessed coefficients.
    pub(crate) fn stir_evaluations_prover_helps(
        &self,
        answers: &[Vec<F>],
        stir_evaluations: &mut Vec<F>,
    ) {
        stir_evaluations.extend(answers.iter().map(|answers| {
            CoefficientList::new(answers.clone()).evaluate(self.folding_randomness)
        }));
    }
}
