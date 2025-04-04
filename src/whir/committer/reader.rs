use ark_crypto_primitives::merkle_tree::Config;
use ark_ff::FftField;
use spongefish::{
    codecs::arkworks_algebra::{FieldToUnitDeserialize, UnitToField},
    ProofResult,
};

use crate::whir::{parameters::WhirConfig, utils::DigestToUnitDeserialize, WhirCommitmentData};

#[derive(Clone)]
pub struct ParsedCommitment<F, D> {
    pub root: D,
    pub ood_points: Vec<F>,
    pub ood_answers: Vec<F>,
}

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
    ) -> ProofResult<ParsedCommitment<F, MerkleConfig::InnerDigest>>
    where
        VerifierState:
            FieldToUnitDeserialize<F> + UnitToField<F> + DigestToUnitDeserialize<MerkleConfig>,
    {
        let root = verifier_state.read_digest()?;

        let mut ood_points = vec![F::ZERO; self.0.committment_ood_samples];
        let mut ood_answers = vec![F::ZERO; self.0.committment_ood_samples];
        if self.0.committment_ood_samples > 0 {
            verifier_state.fill_challenge_scalars(&mut ood_points)?;
            verifier_state.fill_next_scalars(&mut ood_answers)?;
        }

        Ok(ParsedCommitment {
            root,
            ood_points,
            ood_answers,
        })
    }
}

impl<F, M: Config> WhirCommitmentData<F, M> for ParsedCommitment<F, M::InnerDigest> {
    fn committed_root(&self) -> &<M as Config>::InnerDigest {
        &self.root
    }

    fn ood_data(&self) -> (&[F], &[F]) {
        (&self.ood_points, &self.ood_answers)
    }

    fn batching_randomness(&self) -> Option<F> {
        None
    }
}
