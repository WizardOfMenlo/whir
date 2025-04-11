use ark_crypto_primitives::merkle_tree::Config;
use ark_ff::FftField;
use spongefish::ProofResult;

use crate::fs_utils::DigestToUnitDeserialize;

#[derive(Clone)]
pub struct ParsedCommitment<D> {
    pub root: D,
}

#[derive(Default)]
pub struct CommitmentReader {}

impl CommitmentReader {
    pub const fn new() -> Self {
        Self {}
    }

    pub fn parse_commitment<F, MerkleConfig, VerifierState>(
        &self,
        verifier_state: &mut VerifierState,
    ) -> ProofResult<ParsedCommitment<MerkleConfig::InnerDigest>>
    where
        F: FftField,
        MerkleConfig: Config<Leaf = [F]>,
        VerifierState: DigestToUnitDeserialize<MerkleConfig>,
    {
        let root = verifier_state.read_digest()?;

        Ok(ParsedCommitment { root })
    }
}
