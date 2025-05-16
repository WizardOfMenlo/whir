use ark_crypto_primitives::merkle_tree::Config;
use ark_ff::FftField;
use spongefish::{
    codecs::arkworks_algebra::{FieldToUnitDeserialize, UnitToField},
    BytesToUnitDeserialize, ProofResult, UnitToBytes,
};

use super::Commitment;
use crate::whir::{
    challenges::ChallengeField, parameters::WhirConfig, utils::DigestToUnitDeserialize,
};

///
///  Commitment parsed by the verifier from verifier's FS context.
///
///

// TODO: It would be nice to have a Reader trait in spongefish instead.
pub struct CommitmentReader<'a, F, MerkleConfig, PowStrategy>(
    &'a WhirConfig<F, MerkleConfig, PowStrategy>,
)
where
    F: FftField,
    MerkleConfig: Config;

impl<'a, F, MerkleConfig, PowStrategy> CommitmentReader<'a, F, MerkleConfig, PowStrategy>
where
    F: FftField,
    MerkleConfig: Config<Leaf = [F]>,
    PowStrategy: spongefish_pow::PowStrategy,
{
    pub const fn new(params: &'a WhirConfig<F, MerkleConfig, PowStrategy>) -> Self {
        Self(params)
    }

    pub fn parse_commitment<VerifierState>(
        &self,
        verifier_state: &mut VerifierState,
    ) -> ProofResult<Commitment<F, MerkleConfig::InnerDigest>>
    where
        VerifierState: UnitToBytes
            + FieldToUnitDeserialize<F>
            + UnitToField<F>
            + DigestToUnitDeserialize<MerkleConfig>
            + BytesToUnitDeserialize,
    {
        // Read single root for Merkle tree comitting to multiple polynomials.
        let root = verifier_state.read_digest()?;

        // Read single set of Out of Domain Sampling challenge points
        let ood_points = verifier_state.challenge_vec(self.0.committment_ood_samples)?;

        // Read set of OODS evaluations for each polynomial in the batch.
        let mut ood_answers = vec![F::ZERO; self.0.batch_size * ood_points.len()];
        verifier_state.fill_next_scalars(&mut ood_answers)?;

        Ok(Commitment {
            num_variables: self.0.mv_parameters.num_variables,
            root,
            ood_points,
            ood_answers,
        })
    }
}
