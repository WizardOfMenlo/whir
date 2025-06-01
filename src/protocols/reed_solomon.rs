//! Commit to a vector of field elements by encoding as a Reed-Solomon codeword.
//! Opening results in a sequence of [`Constraint`]s.

use std::{
    fmt::{Debug, Display},
    mem::swap,
    sync::Arc,
};

use ark_ff::FftField;
use ark_poly::{EvaluationDomain, GeneralEvaluationDomain};
use spongefish::{
    codecs::{
        ZeroCopyHintProver, ZeroCopyHintVerifier, ZeroCopyPattern, ZeroCopyProver, ZeroCopyVerifier,
    },
    transcript::{Label, Length},
    Unit,
};
use thiserror::Error;
use zerocopy::{transmute_ref, FromBytes, Immutable, IntoBytes, KnownLayout};

use crate::{
    ntt::expand_from_coeff,
    poly_utils::{coeffs::CoefficientList, fold::transform_evaluations},
    protocols::merkle_tree::{self, MerkleTreePattern, MerkleTreeProver, MerkleTreeVerifier},
};

#[derive(Debug, Clone)]
pub struct Config<F>
where
    F: FftField,
{
    /// Merkle tree used for committing to the cosets.
    merkle_tree: merkle_tree::Config<F>,

    /// Number of coefficients in the polynomial to be committed.
    num_coefficients: usize,

    /// The domain over which the polynomial is evaluated as RS encoding.
    evaluation_domain: GeneralEvaluationDomain<F>,

    /// The size of the coset
    coset_size: usize,

    /// The number of samples from the evaluation_domain used for opening.
    num_in_domain_samples: usize,
}

pub struct Witness(merkle_tree::Witness);

#[derive(
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Debug,
    Immutable,
    KnownLayout,
    FromBytes,
    IntoBytes,
)]
#[repr(transparent)]
pub struct Commitment(merkle_tree::Commitment);

pub trait ReedSolomonPattern<U>: ZeroCopyPattern<U>
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

    fn reed_solomon_open<F>(
        &mut self,
        label: impl Into<Label>,
        config: &Config<F>,
    ) -> Result<(), Self::Error>
    where
        F: FftField;
}

pub trait ReedSolomonProver<U>: ZeroCopyProver<U>
where
    U: Unit,
{
    fn reed_solomon_commit<F>(
        &mut self,
        label: impl Into<Label>,
        config: &Config<F>,
        polynomial: &CoefficientList<F>,
    ) -> Result<Witness, Self::Error>
    where
        F: FftField;
}

pub trait ReedSolomonVerifier<'a, U>: ZeroCopyVerifier<'a, U>
where
    U: Unit,
{
}

impl<F> Config<F>
where
    F: FftField,
{
    pub fn new(
        engine: Arc<dyn merkle_tree::Engine<F>>,
        num_coefficients: usize,
        expansion: usize,
        coset_size: usize,
        num_in_domain_samples: usize,
    ) -> Self {
        let evaluation_domain = GeneralEvaluationDomain::new(num_coefficients * expansion)
            .expect("Can not create evaluation domain of required size.");
        assert_eq!(
            evaluation_domain.size() % coset_size,
            0,
            "Evaluation domain not a multiple of coset size."
        );
        let leaf_size = F::default().compressed_size() * coset_size;
        let num_leaves = evaluation_domain.size() / coset_size;
        let merkle_tree = merkle_tree::Config::new(engine, leaf_size, num_leaves);

        Self {
            merkle_tree,
            num_coefficients,
            evaluation_domain,
            coset_size,
            num_in_domain_samples,
        }
    }

    pub fn expansion(&self) -> usize {
        // Compute the expansion factor based on the evaluation domain size and polynomial length.
        self.evaluation_domain.size() / self.num_coefficients
    }
}

impl Witness {}

impl<U, P> ReedSolomonPattern<U> for P
where
    U: Unit,
    P: ZeroCopyPattern<U>,
{
    fn reed_solomon_commit<F>(
        &mut self,
        label: impl Into<Label>,
        config: &Config<F>,
    ) -> Result<(), Self::Error>
    where
        F: FftField,
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
        F: FftField,
    {
        todo!()
    }
}

impl<U, P> ReedSolomonProver<U> for P
where
    U: Unit,
    P: ZeroCopyProver<U> + ZeroCopyHintProver<U>,
{
    fn reed_solomon_commit<F>(
        &mut self,
        label: impl Into<Label>,
        config: &Config<F>,
        polynomial: &CoefficientList<F>,
    ) -> Result<Witness, Self::Error>
    where
        F: FftField,
    {
        // Expand the polynomial coefficients into evaluations over the extended domain.
        let mut evals = expand_from_coeff(polynomial.coeffs(), config.expansion());
        transform_evaluations(
            &mut evals,
            config.evaluation_domain.group_gen_inv(),
            config.coset_size,
        );

        todo!()
    }
}

impl<'a, U, P> ReedSolomonVerifier<'a, U> for P
where
    U: Unit,
    P: ZeroCopyVerifier<'a, U> + ZeroCopyHintVerifier<'a, U>,
{
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use sha3::Keccak256;
    use spongefish::{transcript::TranscriptRecorder, ProverState, VerifierState};

    use super::*;
    use crate::{
        crypto::fields::Field256 as F,
        protocols::merkle_tree::digest::{ArkFieldUpdater, DigestEngine},
    };

    #[test]
    fn test_all_ops() -> Result<()> {
        let config = Config::<F>::new(
            Arc::new(ArkDigestEngine::<Keccak256>::default()),
            64,
            4,
            2,
            10,
        );

        let mut pattern: TranscriptRecorder = TranscriptRecorder::new();
        pattern.reed_solomon_commit("1", &config);
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
