use ark_crypto_primitives::merkle_tree::Config;
use ark_ff::FftField;
use spongefish::{
    codecs::arkworks_algebra::{FieldToUnitDeserialize, UnitToField},
    BytesToUnitDeserialize, ProofResult, UnitToBytes,
};

use crate::whir::{parameters::WhirConfig, utils::DigestToUnitDeserialize};

///
///  Commitment parsed by the verifier from verifier's FS context.
///
/// In case of single oracle: Root will have a single element and
/// batching_randomness will be F::ZERO
///
/// In case of multiple virtual oracles: Root will have as many Merkle roots as
/// the number of virtual oracles and the value of batching_randomness will
/// neither be F::ZERO nor F::ONE (w.h.p.).
///

#[derive(Clone)]
pub struct ParsedCommitment<F, D> {
    pub root: Vec<D>,
    pub ood_points: Vec<F>,
    pub ood_answers: Vec<F>,
    pub batching_randomness: F,
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
        VerifierState: UnitToBytes
            + FieldToUnitDeserialize<F>
            + UnitToField<F>
            + DigestToUnitDeserialize<MerkleConfig>
            + BytesToUnitDeserialize,
    {
        let mut batch_size_le_bytes: [u8; 4] = [0, 0, 0, 0];
        verifier_state.fill_next_bytes(&mut batch_size_le_bytes)?;
        let batch_size = u32::from_le_bytes(batch_size_le_bytes) as usize;

        if batch_size == 1 {
            self.parse_single(verifier_state)
        } else {
            self.parse_batched(batch_size, verifier_state)
        }
    }

    fn parse_single<VerifierState>(
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
            root: vec![root],
            ood_points,
            ood_answers,
            batching_randomness: F::ZERO,
        })
    }

    fn parse_batched<VerifierState>(
        &self,
        batch_size: usize,
        verifier_state: &mut VerifierState,
    ) -> ProofResult<ParsedCommitment<F, MerkleConfig::InnerDigest>>
    where
        VerifierState: UnitToBytes
            + FieldToUnitDeserialize<F>
            + UnitToField<F>
            + DigestToUnitDeserialize<MerkleConfig>
            + BytesToUnitDeserialize,
    {
        let mut roots = Vec::<MerkleConfig::InnerDigest>::with_capacity(batch_size);

        for _ in 0..batch_size {
            // Order is important here
            let root = verifier_state.read_digest()?;
            roots.push(root);
        }

        // Re-derive batching randomness

        let [batching_randomness] = verifier_state.challenge_scalars()?;

        Ok(ParsedCommitment {
            root: roots,
            batching_randomness,
            ood_points: vec![],
            ood_answers: vec![],
        })
    }
}

impl<F, D> ParsedCommitment<F, D> {
    pub(crate) fn ood_data(&self) -> (&[F], &[F]) {
        (&self.ood_points, &self.ood_answers)
    }
}
