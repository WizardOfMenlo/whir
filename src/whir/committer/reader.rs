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
///

#[derive(Clone)]
pub struct ParsedCommitment<F, D> {
    pub root: Vec<D>,
    pub ood_points: Vec<F>,
    pub ood_answers: Vec<Vec<F>>,
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
        let mut roots = Vec::<MerkleConfig::InnerDigest>::with_capacity(self.0.batch_size);

        for _ in 0..self.0.batch_size {
            let root = verifier_state.read_digest()?;
            roots.push(root);
        }

        let mut ood_points = vec![F::ZERO; self.0.committment_ood_samples];
        let mut ood_answers = Vec::with_capacity(self.0.batch_size);

        if self.0.committment_ood_samples > 0 {
            verifier_state.fill_challenge_scalars(&mut ood_points)?;
            for _ in 0..self.0.batch_size {
                let mut virt_answers = vec![F::ZERO; self.0.committment_ood_samples];
                verifier_state.fill_next_scalars(&mut virt_answers)?;
                ood_answers.push(virt_answers);
            }
        }

        let [batching_randomness] = if self.0.batch_size > 1 {
            verifier_state.challenge_scalars()?
        } else {
            [F::zero()]
        };

        Ok(ParsedCommitment {
            root: roots,
            batching_randomness,
            ood_points,
            ood_answers,
        })
    }
}

impl<F, D> ParsedCommitment<F, D>
where
    F: ark_ff::Field,
{
    pub(crate) fn ood_data(&self) -> (&[F], Vec<F>) {
        let mut multiplier = self.batching_randomness;

        let result = self
            .ood_answers
            .clone()
            .into_iter()
            .reduce(|result, this_round| {
                let this_multiplier = multiplier;
                multiplier *= self.batching_randomness;
                result
                    .into_iter()
                    .zip(this_round.into_iter())
                    .map(|(lhs, v)| lhs + (v * this_multiplier))
                    .collect()
            })
            .unwrap_or_default();
        (self.ood_points.as_slice(), result)
    }
}
