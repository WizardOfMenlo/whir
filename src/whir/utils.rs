use ark_crypto_primitives::merkle_tree::{Config, MultiPath};
use ark_ff::FftField;
use itertools::Itertools;
use spongefish::{
    codecs::arkworks_algebra::{FieldToUnitSerialize, UnitToField},
    ProofResult,
};
#[cfg(feature = "tracing")]
use tracing::instrument;

use crate::poly_utils::multilinear::MultilinearPoint;

///
/// A utility function to compute the response to OOD challenge and add it to
/// the transcript. The OOD challenge should have already been sampled and added
/// to the transcript before this call.
///
#[cfg_attr(feature = "tracing", instrument(skip(prover_state, evaluate_fn)))]
pub(crate) fn compute_ood_response<F, ProverState, E>(
    prover_state: &mut ProverState,
    ood_points: &[F],
    num_variables: usize,
    evaluate_fn: E,
) -> ProofResult<Vec<F>>
where
    F: FftField,
    ProverState: FieldToUnitSerialize<F>,
    E: Fn(&MultilinearPoint<F>) -> F,
{
    let num_samples = ood_points.len();
    let mut ood_answers = Vec::<F>::with_capacity(ood_points.len());

    if num_samples > 0 {
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

    Ok(ood_answers)
}

/// A utility function to sample Out-of-Domain (OOD) points and evaluate them
///
/// This operates on the prover side.
#[cfg_attr(feature = "tracing", instrument(skip(prover_state, evaluate_fn)))]
pub(crate) fn sample_ood_points<F, ProverState, E>(
    prover_state: &mut ProverState,
    num_samples: usize,
    num_variables: usize,
    evaluate_fn: E,
) -> ProofResult<(Vec<F>, Vec<F>)>
where
    F: FftField,
    ProverState: FieldToUnitSerialize<F> + UnitToField<F>,
    E: Fn(&MultilinearPoint<F>) -> F,
{
    let mut ood_points = vec![F::ZERO; num_samples];
    let mut ood_answers = vec![F::ZERO; num_samples];

    if num_samples > 0 {
        // Generate OOD points from ProverState randomness
        prover_state.fill_challenge_scalars(&mut ood_points)?;
        ood_answers = compute_ood_response(prover_state, &ood_points, num_variables, evaluate_fn)?;
    }

    Ok((ood_points, ood_answers))
}

pub trait DigestToUnitSerialize<MerkleConfig: Config> {
    fn add_digest(&mut self, digest: MerkleConfig::InnerDigest) -> ProofResult<()>;
}

pub trait DigestToUnitDeserialize<MerkleConfig: Config> {
    fn read_digest(&mut self) -> ProofResult<MerkleConfig::InnerDigest>;
}

///
/// Compute the combined stir queries in-place
///
pub(crate) fn fma_stir_queries<F: ark_ff::Field>(
    scale_factor: F,
    stir_queries: &[Vec<F>],
    folded_queries: &mut [Vec<F>],
) {
    for (folded_vec, stir_vec) in folded_queries.iter_mut().zip(stir_queries.iter()) {
        folded_vec
            .iter_mut()
            .zip(stir_vec.iter())
            .for_each(|(fused_values, value)| *fused_values += *value * scale_factor);
    }
}

#[allow(dead_code)] // Only used with debug build
pub(crate) fn validate_stir_queries<F, MerkleConfig>(
    batching_randomness: &F,
    expected_stir_value: &[Vec<F>],
    stir_list: &[(MultiPath<MerkleConfig>, Vec<Vec<F>>)],
) -> bool
where
    F: ark_ff::Field,
    MerkleConfig: Config<Leaf = [F]>,
{
    if stir_list.is_empty() {
        return true;
    }

    let (basic_indexes, mut computed_stir) = stir_list.first().unwrap().clone();
    let mut multiplier = *batching_randomness;

    for (multi_path, stir_queries) in &stir_list[1..] {
        if !multi_path
            .leaf_indexes
            .iter()
            .zip(basic_indexes.leaf_indexes.iter())
            .all(|(lhs, rhs)| lhs == rhs)
        {
            return false;
        }

        fma_stir_queries(multiplier, stir_queries.as_slice(), &mut computed_stir);
        multiplier *= batching_randomness;
    }

    let result = expected_stir_value == computed_stir;
    result
}
