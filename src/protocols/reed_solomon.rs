//! Folding Reed-Solomon vector commitments.
//! Opening takes 0..n folding parameters, where n is the number of coefficients in the polynomial.

use std::{fmt::Debug, ops::Range, sync::Arc};

use ark_ff::{FftField, Field};
use ark_poly::{evaluations, EvaluationDomain, GeneralEvaluationDomain};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use proptest::result;
use spongefish::{
    codecs::arkworks::{ArkworksHintPattern, ArkworksHintProver, ArkworksHintVerifier},
    transcript::{Label, Length},
    Unit,
};
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

use crate::{
    ntt::expand_from_coeff,
    poly_utils::{coeffs::CoefficientList, fold::transform_evaluations},
    protocols::{
        challenge_indices::{ChallengeIndicesCommon, ChallengeIndicesPattern},
        merkle_tree::{self, MerkleTreePattern, MerkleTreeProver, MerkleTreeVerifier},
    },
};

#[derive(Debug, Clone)]
pub struct Config<F>
where
    F: FftField + CanonicalSerialize + CanonicalDeserialize,
{
    /// Merkle tree used for committing to the cosets.
    merkle_tree: merkle_tree::Config<F>,

    /// Number of coefficients in the polynomial to be committed.
    num_coefficients: usize,

    /// The domain over which the polynomial is evaluated as RS encoding.
    evaluation_domain: GeneralEvaluationDomain<F>,

    /// The number of folds to apply on opening.
    /// Can be zero to disable folding.
    num_folds: usize,

    /// The number of samples from the evaluation_domain used for opening.
    num_in_domain_samples: usize,
}

pub struct Witness<F>
where
    F: FftField + CanonicalSerialize + CanonicalDeserialize,
{
    merkle_tree: merkle_tree::Witness,
    evaluations: Vec<F>,
}

pub type Commitment = merkle_tree::Commitment;

pub trait ReedSolomonPattern<U>:
    MerkleTreePattern<U> + ChallengeIndicesPattern<U> + ArkworksHintPattern<U>
where
    U: Unit,
{
    fn reed_solomon_commit<F>(
        &mut self,
        label: impl Into<Label>,
        config: &Config<F>,
    ) -> Result<(), Self::Error>
    where
        F: FftField;

    /// Opens the RS commitment by random sampling from the evaluation domain.
    /// Returns the evaluation points and values.
    fn reed_solomon_open<F>(
        &mut self,
        label: impl Into<Label>,
        config: &Config<F>,
    ) -> Result<(), Self::Error>
    where
        F: FftField + CanonicalSerialize + CanonicalDeserialize;
}

pub trait ReedSolomonProver<U>:
    MerkleTreeProver<U> + ChallengeIndicesCommon<U> + ArkworksHintProver<U>
where
    U: Unit,
{
    fn reed_solomon_commit<F>(
        &mut self,
        label: impl Into<Label>,
        config: &Config<F>,
        polynomial: &CoefficientList<F>,
    ) -> Result<Witness<F>, Self::Error>
    where
        F: FftField + CanonicalSerialize + CanonicalDeserialize;

    /// Opens a Reed-Solomon commitment by sampling the polynomial at random points.
    /// Returns the evaluation points and values of the folded polynomial.
    fn reed_solomon_open<F>(
        &mut self,
        label: impl Into<Label>,
        config: &Config<F>,
        witness: &Witness<F>,
    ) -> Result<(Vec<F>, Vec<F>), Self::Error>
    where
        F: FftField + CanonicalSerialize + CanonicalDeserialize;

    fn reed_solomon_open_fold<F>(
        &mut self,
        label: impl Into<Label>,
        config: &Config<F>,
        witness: &Witness<F>,
        folds: &[F],
    ) -> Result<Vec<(F, F)>, Self::Error>
    where
        F: FftField + CanonicalSerialize + CanonicalDeserialize,
    {
        assert_eq!(folds.len(), config.num_folds);
        let (points, evaluations) = self.reed_solomon_open(label, config, witness)?;
        let result = points
            .into_iter()
            .zip(evaluations.into_iter().chunks_exact(config.coset_size))
            .map(|(point, coeffs)| (point, CoefficientList::new(coeffs).evaluate(folds)))
            .collect::<Vec<_>>();
        Ok(result)
    }

    /// Opens using folds in an extension of F, otherwise identical to [`reed_solomon_open`].
    fn reed_solomon_open_fold_extended<F, G>(
        &mut self,
        label: impl Into<Label>,
        config: &Config<F>,
        witness: &Witness<F>,
        folds: &[G],
    ) -> Result<Vec<(F, G)>, Self::Error>
    where
        F: FftField + CanonicalSerialize + CanonicalDeserialize,
        G: Field<BasePrimeField = F>,
    {
        assert_eq!(folds.len(), config.num_folds);
        let (points, evaluations) = self.reed_solomon_open(label, config, witness)?;
        let result = points
            .into_iter()
            .zip(evaluations.into_iter().chunks_exact(config.coset_size))
            .map(|(point, coeffs)| (point, CoefficientList::new(coeffs).evaluate(folds)))
            .collect::<Vec<_>>();
        Ok(result)
    }
}

pub trait ReedSolomonVerifier<'a, U>:
    MerkleTreeVerifier<'a, U> + ChallengeIndicesCommon<U> + ArkworksHintVerifier<'a, U>
where
    U: Unit,
{
}

impl<F> Config<F>
where
    F: FftField + CanonicalSerialize + CanonicalDeserialize,
{
    pub fn new(
        engine: Arc<dyn merkle_tree::Engine<F>>,
        num_coefficients: usize,
        expansion: usize,
        num_folds: usize,
        num_in_domain_samples: usize,
    ) -> Self {
        let evaluation_domain = GeneralEvaluationDomain::new(num_coefficients * expansion)
            .expect("Can not create evaluation domain of required size.");
        assert_eq!(
            evaluation_domain.size().trailing_zeros() < num_folds as u32,
            0,
            "Evaluation domain not a multiple of coset size."
        );
        let leaf_size = 1 << num_folds;
        let num_leaves = evaluation_domain.size() / leaf_size;
        let merkle_tree = merkle_tree::Config::new(engine, leaf_size, num_leaves);

        Self {
            merkle_tree,
            num_coefficients,
            evaluation_domain,
            num_folds,
            num_in_domain_samples,
        }
    }

    pub fn expansion(&self) -> usize {
        // Compute the expansion factor based on the evaluation domain size and polynomial length.
        self.evaluation_domain.size() / self.num_coefficients
    }

    pub fn coset_size(&self) -> usize {
        1 << self.num_folds
    }

    pub fn coset_range(&self, index: usize) -> Range<usize> {
        let coset_size = self.coset_size();
        let start = index * coset_size;
        let end = start + coset_size;
        start..end
    }

    pub fn is_valid_witness(&self, witness: &Witness<F>) -> bool {
        // Check if the witness evaluations match the expected size.
        witness.evaluations.len() == self.evaluation_domain.size()
            && self.merkle_tree.is_valid_witness(&witness.merkle_tree)
    }
}

impl<U, P> ReedSolomonPattern<U> for P
where
    U: Unit,
    P: ArkworksHintPattern<U> + ChallengeIndicesPattern<U> + MerkleTreePattern<U>,
{
    fn reed_solomon_commit<F>(
        &mut self,
        label: impl Into<Label>,
        config: &Config<F>,
    ) -> Result<(), Self::Error>
    where
        F: FftField + CanonicalSerialize + CanonicalDeserialize,
    {
        let label = label.into();
        self.begin_message::<Config<F>>(label.clone(), Length::Fixed(config.num_coefficients))?;
        self.merkle_tree_commit("merkle-tree", &config.merkle_tree)?;
        self.end_message::<Config<F>>(label.clone(), Length::Fixed(config.num_coefficients))
    }

    fn reed_solomon_open<F>(
        &mut self,
        label: impl Into<Label>,
        config: &Config<F>,
    ) -> Result<(), Self::Error>
    where
        F: FftField + CanonicalSerialize + CanonicalDeserialize,
    {
        let label = label.into();
        self.begin_hint::<Config<F>>(label.clone(), Length::Fixed(config.num_coefficients))?;
        self.challenge_indices(
            "evaluation-point-indices",
            config.evaluation_domain.size(),
            config.num_in_domain_samples,
        )?;
        self.hint_arkworks::<Vec<F>>("evaluations")?;
        self.merkle_tree_open(
            "merkle-tree",
            &config.merkle_tree,
            config.num_in_domain_samples,
        )?;
        self.end_hint::<Config<F>>(label.clone(), Length::Fixed(config.num_coefficients))
    }
}

impl<U, P> ReedSolomonProver<U> for P
where
    U: Unit,
    P: MerkleTreeProver<U> + ChallengeIndicesCommon<U> + ArkworksHintProver<U>,
{
    fn reed_solomon_commit<F>(
        &mut self,
        label: impl Into<Label>,
        config: &Config<F>,
        polynomial: &CoefficientList<F>,
    ) -> Result<Witness<F>, Self::Error>
    where
        F: FftField + CanonicalSerialize + CanonicalDeserialize,
    {
        let label = label.into();
        self.begin_message::<Config<F>>(label.clone(), Length::Fixed(config.num_coefficients))?;
        let mut evaluations = expand_from_coeff(polynomial.coeffs(), config.expansion());
        transform_evaluations(
            &mut evaluations,
            config.evaluation_domain.group_gen_inv(),
            config.coset_size,
        );
        let merkle_tree =
            self.merkle_tree_commit("merkle-tree", &config.merkle_tree, &evaluations)?;
        self.end_message::<Config<F>>(label.clone(), Length::Fixed(config.num_coefficients))?;
        Ok(Witness {
            merkle_tree,
            evaluations,
        })
    }

    fn reed_solomon_open<F>(
        &mut self,
        label: impl Into<Label>,
        config: &Config<F>,
        witness: &Witness<F>,
    ) -> Result<(Vec<F>, Vec<F>), Self::Error>
    where
        F: FftField + CanonicalSerialize + CanonicalDeserialize,
    {
        let label = label.into();
        self.begin_hint::<Config<F>>(label.clone(), Length::Fixed(config.num_coefficients))?;

        assert!(config.is_valid_witness(witness));

        // Compute in-domain challenge points.
        let indices = self.challenge_indices_vec(
            "evaluation-point-indices",
            config.evaluation_domain.size(),
            config.num_in_domain_samples,
        )?;
        let points = indices
            .iter()
            .map(|&i| config.evaluation_domain.element(i))
            .collect::<Vec<_>>();

        // Select the evaluations and open the Merkle tree.
        let evaluations = self.merkle_tree_open_with_leaves_ark(
            "merkle-tree",
            &config.merkle_tree,
            &witness.evaluations,
            &witness.merkle_tree,
            &indices,
        )?;
        self.end_hint::<Config<F>>(label.clone(), Length::Fixed(config.num_coefficients))?;
        Ok((points, evaluations))
    }
}

impl<'a, U, P> ReedSolomonVerifier<'a, U> for P
where
    U: Unit,
    P: MerkleTreeVerifier<'a, U> + ChallengeIndicesCommon<U> + ArkworksHintVerifier<'a, U>,
{
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use ark_ff::MontBackend;
    use sha3::Keccak256;
    use spongefish::{transcript::TranscriptRecorder, ProverState, VerifierState};

    use super::*;
    use crate::{
        crypto::fields::{FConfig64, Field64},
        protocols::merkle_tree::digest,
    };

    #[test]
    fn test_all_ops() -> Result<()> {
        type MyEngine =
            digest::DigestEngine<Keccak256, digest::ArkFieldUpdater<MontBackend<FConfig64, 1>, 1>>;
        let config = Config::new(Arc::new(MyEngine::new()), 64, 4, 2, 10);

        let mut pattern: TranscriptRecorder = TranscriptRecorder::new();
        pattern.reed_solomon_commit("1", &config)?;
        let pattern = pattern.finalize()?;
        eprintln!("{pattern}");

        let mut prover: ProverState = ProverState::from(&pattern);
        let proof = prover.finalize()?;
        assert_eq!(hex::encode(&proof), "");

        let mut verifier: VerifierState = VerifierState::new(pattern.into(), &proof);
        verifier.finalize()?;

        Ok(())
    }
}
