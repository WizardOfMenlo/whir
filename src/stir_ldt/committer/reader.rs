use ark_crypto_primitives::merkle_tree::Config;
use ark_ff::FftField;
use spongefish::{
    codecs::arkworks_algebra::{FieldToUnitDeserialize, UnitToField},
    ProofResult,
};

use crate::stir_ldt::parameters::StirConfig;
use crate::whir::utils::DigestToUnitDeserialize;

#[derive(Clone)]
pub struct ParsedCommitment<D> {
    pub root: D,
}

pub struct CommitmentReader<'a, F, MerkleConfig, PowStrategy>(
    // TODO: Refactor to use this.
    #[allow(unused)] &'a StirConfig<F, MerkleConfig, PowStrategy>,
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
    pub const fn new(params: &'a StirConfig<F, MerkleConfig, PowStrategy>) -> Self {
        Self(params)
    }

    pub fn parse_commitment<VerifierState>(
        &self,
        verifier_state: &mut VerifierState,
    ) -> ProofResult<ParsedCommitment<MerkleConfig::InnerDigest>>
    where
        VerifierState:
            FieldToUnitDeserialize<F> + UnitToField<F> + DigestToUnitDeserialize<MerkleConfig>,
    {
        let root = verifier_state.read_digest()?;

        Ok(ParsedCommitment { root })
    }
}
