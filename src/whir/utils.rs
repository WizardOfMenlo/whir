use ark_ff::FftField;
use spongefish::{
    codecs::arkworks_algebra::{FieldToUnit, UnitToField},
    ProofResult,
};

use crate::poly_utils::multilinear::MultilinearPoint;

/// A utility function to sample Out-of-Domain (OOD) points and evaluate them
///
/// This operates on the prover side.
pub(crate) fn sample_ood_points<F, ProverState, E>(
    prover_state: &mut ProverState,
    num_samples: usize,
    num_variables: usize,
    evaluate_fn: E,
) -> ProofResult<(Vec<F>, Vec<F>)>
where
    F: FftField,
    ProverState: FieldToUnit<F> + UnitToField<F>,
    E: Fn(&MultilinearPoint<F>) -> F,
{
    let mut ood_points = vec![F::ZERO; num_samples];
    let mut ood_answers = Vec::with_capacity(num_samples);

    if num_samples > 0 {
        // Generate OOD points from ProverState randomness
        prover_state.fill_challenge_scalars(&mut ood_points)?;

        // Evaluate the function at each OOD point
        ood_answers.extend(ood_points.iter().map(|ood_point| {
            evaluate_fn(&MultilinearPoint::expand_from_univariate(
                *ood_point,
                num_variables,
            ))
        }));

        // Commit the answers to the narg_string
        prover_state.add_scalars(&ood_answers)?;
    }

    Ok((ood_points, ood_answers))
}
