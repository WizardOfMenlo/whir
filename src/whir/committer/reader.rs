use ark_crypto_primitives::merkle_tree::Config;
use ark_ff::{FftField, Field};
use spongefish::{
    codecs::arkworks_algebra::{FieldToUnitDeserialize, UnitToField},
    ProofResult,
};

use crate::whir::{
    parameters::WhirConfig,
    statement::{Constraint, Weights},
    utils::DigestToUnitDeserialize,
};

#[derive(Clone)]
pub struct ParsedCommitment<F, D> {
    pub num_variables: usize,
    pub root: D,
    pub ood_points: Vec<F>,
    pub ood_answers: Vec<F>,
}

impl<F: Field, D> ParsedCommitment<F, D> {
    pub fn parse<VerifierState, MerkleConfig>(
        verifier_state: &mut VerifierState,
        num_variables: usize,
        ood_samples: usize,
    ) -> ProofResult<ParsedCommitment<F, MerkleConfig::InnerDigest>>
    where
        MerkleConfig: Config<Leaf = [F]>,
        VerifierState:
            FieldToUnitDeserialize<F> + UnitToField<F> + DigestToUnitDeserialize<MerkleConfig>,
    {
        let root = verifier_state.read_digest()?;

        let mut ood_points = vec![F::ZERO; ood_samples];
        let mut ood_answers = vec![F::ZERO; ood_samples];
        if ood_samples > 0 {
            verifier_state.fill_challenge_scalars(&mut ood_points)?;
            verifier_state.fill_next_scalars(&mut ood_answers)?;
        }

        Ok(ParsedCommitment {
            num_variables,
            root,
            ood_points,
            ood_answers,
        })
    }

    /// Return constraints for OODS
    pub fn oods_constraints(&self) -> Vec<Constraint<F>> {
        self.ood_points
            .iter()
            .zip(&self.ood_answers)
            .map(|(&point, &eval)| Constraint {
                weights: Weights::univariate(point, self.num_variables),
                sum: eval,
                defer_evaluation: false,
            })
            .collect()
    }
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
        ParsedCommitment::<F, MerkleConfig::InnerDigest>::parse(
            verifier_state,
            self.0.mv_parameters.num_variables,
            self.0.committment_ood_samples,
        )
    }
}
