//! Interleaved Reed-Solomon vector commitments.
//! Opening takes 0..n folding parameters, where n is the number of coefficients in the polynomial.

use std::{
    f64::consts::LOG2_10,
    fmt::{Debug, Display},
    ops::Range,
    str::FromStr,
    sync::Arc,
    usize,
};

use ark_ff::{FftField, Field};
use ark_poly::{EvaluationDomain, GeneralEvaluationDomain};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use serde::{Deserialize, Serialize};
use spongefish::{
    codecs::arkworks::{field, serialize},
    transcript::{self, Label, Length},
};

use crate::{
    ntt::expand_from_coeff,
    poly_utils::{
        coeffs::{eval_multivariate, eval_multivariate_extend},
        fold::transform_evaluations,
    },
    protocols::{challenge_indices, merkle_tree},
    utils::field_size,
};

/// Defines the soundness type for the proof system.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SoundnessType {
    /// Unique decoding guarantees a single valid witness.
    UniqueDecoding,
    /// Provable list decoding allows multiple valid witnesses but provides proof.
    ProvableList,
    /// Conjecture-based list decoding with no strict guarantees.
    ConjectureList,
}

pub use merkle_tree::VerifierError;

#[derive(Debug, Clone)]
pub struct Config<F>
where
    F: FftField + CanonicalSerialize + CanonicalDeserialize,
{
    /// Merkle tree used for committing to the cosets.
    merkle_tree: merkle_tree::Config<F>,

    /// Number of coefficients in the polynomial to be committed.
    size: usize,

    /// The domain over which the polynomial is evaluated as RS encoding.
    evaluation_domain: GeneralEvaluationDomain<F>,

    /// In both JB and CB theorems such as list-size only hold for proximity parameters slighly below the bound.
    /// E.g. in JB proximity gaps holds for every δ ∈ (0, 1 - √ρ).
    /// η is the distance between the chosen proximity parameter and the bound.
    /// I.e. in JB δ = 1 - √ρ - η and in CB δ = 1 - ρ - η.
    log2_eta: f64,

    /// The number of folds to apply on opening.
    /// Can be zero to disable folding.
    num_folds: usize,

    /// The number of samples from the evaluation_domain used for opening.
    num_in_domain_samples: usize,

    /// The number of samples from outside the evaluation domain used on commit.
    num_out_domain_samples: usize,
}

pub struct Witness<F>
where
    F: FftField + CanonicalSerialize + CanonicalDeserialize,
{
    merkle_tree: merkle_tree::Witness,

    // OPT: It might be more efficient to recompute the opened values.
    evaluations: Vec<F>,

    ood_samples: Vec<(F, F)>,
}

pub struct Commitment<F>
where
    F: FftField + CanonicalSerialize + CanonicalDeserialize,
{
    merkle_tree: merkle_tree::Commitment,
    ood_samples: Vec<(F, F)>,
}

pub trait Pattern {
    fn reed_solomon_commit<F>(&mut self, label: Label, config: &Config<F>)
    where
        F: FftField;

    /// Opens the RS commitment by random sampling from the evaluation domain.
    /// Returns the evaluation points and values.
    fn reed_solomon_open<F>(&mut self, label: Label, config: &Config<F>)
    where
        F: FftField + CanonicalSerialize + CanonicalDeserialize;
}

pub trait Prover {
    fn reed_solomon_commit<F>(
        &mut self,
        label: Label,
        config: &Config<F>,
        values: &[F],
    ) -> Witness<F>
    where
        F: FftField + CanonicalSerialize + CanonicalDeserialize;

    /// Opens a Reed-Solomon commitment by sampling the polynomial at in domain points.
    /// Returns the evaluation points and the coset polynomials.
    fn reed_solomon_open<F>(
        &mut self,
        label: Label,
        config: &Config<F>,
        witness: &Witness<F>,
    ) -> (Vec<F>, Vec<F>)
    where
        F: FftField + CanonicalSerialize + CanonicalDeserialize;

    /// Opens a Reed-Solomon commitment by sampling the polynomial at in domain points.
    /// Returns the evaluation points and the folded evaluations.
    fn reed_solomon_open_fold<F>(
        &mut self,
        label: Label,
        config: &Config<F>,
        witness: &Witness<F>,
        folds: &[F],
    ) -> Vec<(F, F)>
    where
        F: FftField + CanonicalSerialize + CanonicalDeserialize,
    {
        assert_eq!(folds.len(), config.num_folds);
        let (points, evaluations) = self.reed_solomon_open(label, config, witness);
        let result = points
            .into_iter()
            .zip(evaluations.chunks_exact(config.coset_size()))
            .map(|(point, coeffs)| (point, eval_multivariate(coeffs, folds)))
            .collect::<Vec<_>>();
        result
    }

    /// Opens using folds in an extension of F, otherwise identical to [`reed_solomon_open`].
    fn reed_solomon_open_fold_extended<F, G>(
        &mut self,
        label: Label,
        config: &Config<F>,
        witness: &Witness<F>,
        folds: &[G],
    ) -> Vec<(F, G)>
    where
        F: FftField + CanonicalSerialize + CanonicalDeserialize,
        G: Field<BasePrimeField = F>,
    {
        assert_eq!(folds.len(), config.num_folds);
        let (points, evaluations) = self.reed_solomon_open(label, config, witness);
        let result = points
            .into_iter()
            .zip(evaluations.chunks_exact(config.coset_size()))
            .map(|(point, coeffs)| (point, eval_multivariate_extend(coeffs, folds)))
            .collect::<Vec<_>>();
        result
    }
}

pub trait Verifier {
    fn reed_solomon_commit<F>(
        &mut self,
        label: Label,
        config: &Config<F>,
    ) -> Result<Commitment<F>, VerifierError>
    where
        F: FftField + CanonicalSerialize + CanonicalDeserialize;

    /// Opens a Reed-Solomon commitment by sampling the polynomial at in domain points.
    /// Returns the evaluation points and the coset polynomials.
    fn reed_solomon_open<F>(
        &mut self,
        label: Label,
        config: &Config<F>,
        witness: &Witness<F>,
    ) -> Result<(Vec<F>, Vec<F>), VerifierError>
    where
        F: FftField + CanonicalSerialize + CanonicalDeserialize;

    /// Opens a Reed-Solomon commitment by sampling the polynomial at in domain points.
    /// Returns the evaluation points and the folded evaluations.
    fn reed_solomon_open_fold<F>(
        &mut self,
        label: Label,
        config: &Config<F>,
        witness: &Witness<F>,
        folds: &[F],
    ) -> Result<Vec<(F, F)>, VerifierError>
    where
        F: FftField + CanonicalSerialize + CanonicalDeserialize,
    {
        assert_eq!(folds.len(), config.num_folds);
        let (points, evaluations) = self.reed_solomon_open(label, config, witness)?;
        let result = points
            .into_iter()
            .zip(evaluations.chunks_exact(config.coset_size()))
            .map(|(point, coeffs)| (point, eval_multivariate(coeffs, folds)))
            .collect::<Vec<_>>();
        Ok(result)
    }

    /// Opens using folds in an extension of F, otherwise identical to [`reed_solomon_open`].
    fn reed_solomon_open_fold_extended<F, G>(
        &mut self,
        label: Label,
        config: &Config<F>,
        witness: &Witness<F>,
        folds: &[G],
    ) -> Result<Vec<(F, G)>, VerifierError>
    where
        F: FftField + CanonicalSerialize + CanonicalDeserialize,
        G: Field<BasePrimeField = F>,
    {
        assert_eq!(folds.len(), config.num_folds);
        let (points, evaluations) = self.reed_solomon_open(label, config, witness)?;
        let result = points
            .into_iter()
            .zip(evaluations.chunks_exact(config.coset_size()))
            .map(|(point, coeffs)| (point, eval_multivariate_extend(coeffs, folds)))
            .collect::<Vec<_>>();
        Ok(result)
    }
}

impl Display for SoundnessType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::ProvableList => "ProvableList",
            Self::ConjectureList => "ConjectureList",
            Self::UniqueDecoding => "UniqueDecoding",
        })
    }
}

impl FromStr for SoundnessType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "ProvableList" => Ok(Self::ProvableList),
            "ConjectureList" => Ok(Self::ConjectureList),
            "UniqueDecoding" => Ok(Self::UniqueDecoding),
            _ => Err(format!("Invalid soundness specification: {s}")),
        }
    }
}

impl<F> Config<F>
where
    F: FftField + CanonicalSerialize + CanonicalDeserialize,
{
    pub fn new(
        engine: Arc<dyn merkle_tree::Engine<F>>,
        size: usize,
        expansion: usize,
        num_folds: usize,
        soundness_type: SoundnessType,
        protocol_security_level: usize,
    ) -> Self {
        assert!(size > 0, "Empty vectors not supported");
        let coset_size = 1 << num_folds;
        // Chcek if the field supports FFTs of required size.
        let evaluation_domain = GeneralEvaluationDomain::new(size * expansion)
            .expect("Can not create evaluation domain of required size.");
        assert_eq!(
            evaluation_domain.size() % coset_size,
            0,
            "Evaluation domain not a multiple of coset size."
        );

        // Merkle tree
        let num_leaves = evaluation_domain.size() / coset_size;
        let merkle_tree = merkle_tree::Config::new(engine, coset_size, num_leaves);

        // In and out of domain sampling configuration
        let mut result = Self {
            merkle_tree,
            size,
            evaluation_domain,
            num_folds,
            soundness_type,
            num_in_domain_samples: usize::MAX,
            num_out_domain_samples: usize::MAX,
        };
        result.compute_in_domain_samples(protocol_security_level);
        result.compute_out_domain_samples(protocol_security_level);
        result
    }

    pub fn expansion(&self) -> usize {
        // Compute the expansion factor based on the evaluation domain size and polynomial length.
        self.evaluation_domain.size() / self.size
    }

    pub fn num_cosets(&self) -> usize {
        self.evaluation_domain.size() >> self.num_folds
    }

    pub fn rate(&self) -> f64 {
        1.0 / self.expansion() as f64
    }

    pub fn log2_size(&self) -> f64 {
        (self.size as f64).log2()
    }

    pub fn num_samples(&self) -> usize {
        self.num_in_domain_samples + self.num_out_domain_samples
    }

    pub fn log2_inv_rate(&self) -> f64 {
        -self.rate().log2()
    }

    // TODO: Make part of config object.
    pub fn log2_eta(&self) -> f64 {
        // Original author left the following explanation:
        // > Ask me how I did this? At the time, only God and I knew. Now only God knows
        match self.soundness_type {
            SoundnessType::ProvableList => -(0.5 * self.log2_inv_rate() + LOG2_10 + 1.),
            SoundnessType::UniqueDecoding => 0.,
            SoundnessType::ConjectureList => -(self.log2_inv_rate() + 1.),
        }
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

    pub fn assert_valid_witness(&self, witness: &Witness<F>) {
        assert_eq!(witness.evaluations.len(), self.evaluation_domain.size());
        self.merkle_tree.assert_valid_witness(&witness.merkle_tree);
    }

    /// This is the bits of security of the in domain samples.
    pub fn in_domain_round_by_round_soundness(&self) -> f64 {
        let log_inv_rate = self.expansion() as f64;
        let num_queries = self.num_in_domain_samples as f64;
        match self.soundness_type {
            SoundnessType::UniqueDecoding => {
                let denom = -(0.5 * (1. + self.rate())).log2();
                num_queries * denom
            }
            SoundnessType::ProvableList => num_queries * 0.5 * log_inv_rate,
            SoundnessType::ConjectureList => num_queries * log_inv_rate,
        }
    }

    pub fn log2_list_size(&self) -> f64 {
        match self.soundness_type {
            SoundnessType::ConjectureList => {
                (self.log2_size() + self.log2_inv_rate()) - self.log2_eta()
            }
            SoundnessType::ProvableList => {
                let log_inv_sqrt_rate: f64 = self.log2_inv_rate() / 2.;
                log_inv_sqrt_rate - (1. + self.log2_eta())
            }
            SoundnessType::UniqueDecoding => 0.0,
        }
    }

    pub fn out_domain_round_by_round_soundness(&self) -> f64 {
        let oods = self.num_out_domain_samples as f64;
        let error = 2. * self.log2_list_size() + (self.log2_size() * oods);
        (oods * field_size::<F>().log2()) + 1. - error
    }

    /// This is the bits of security of the in and out of domain samples.
    pub fn round_by_round_soundness(&self) -> f64 {
        let log2_samples = (self.num_samples() as f64).log2();
        field_size::<F>().log2() - (log2_samples + self.log2_list_size() + 1.)
    }

    // Compute the proximity gaps term of the fold
    pub fn round_by_round_soundness_fold_prox_gaps(&self) -> f64 {
        // Recall, at each round we are only folding by two at a time
        let error = match self.soundness_type {
            SoundnessType::ConjectureList => {
                (self.log2_size() + self.log2_inv_rate()) - self.log2_eta()
            }
            SoundnessType::ProvableList => {
                LOG2_10 + 3.5 * self.log2_inv_rate() + 2. * self.log2_size()
            }
            SoundnessType::UniqueDecoding => self.log2_size() + self.log2_inv_rate(),
        };

        field_size::<F>().log2() - error
    }

    /// Compute the required number of in domain samples for a given
    /// set of assumptions and security
    fn compute_in_domain_samples(&mut self, protocol_security_level: usize) {
        self.num_in_domain_samples = (match self.soundness_type {
            SoundnessType::UniqueDecoding => {
                let denom = (0.5 * (1. + self.rate())).log2();
                -(protocol_security_level as f64) / denom
            }
            SoundnessType::ProvableList => {
                (2.0 * protocol_security_level as f64) / self.log2_inv_rate()
            }
            SoundnessType::ConjectureList => {
                (protocol_security_level as f64) / self.log2_inv_rate()
            }
        })
        .ceil() as usize;
    }

    fn compute_out_domain_samples(&mut self, protocol_security_level: usize) {
        let soundness = protocol_security_level as f64;
        self.num_out_domain_samples = 0;
        match self.soundness_type {
            SoundnessType::UniqueDecoding => {}
            _ => {
                while self.out_domain_round_by_round_soundness() < soundness {
                    self.num_out_domain_samples += 1;
                    if self.num_out_domain_samples > 64 {
                        panic!("Could not find an appropriate number of out of domain samples")
                    }
                }
            }
        }
    }
}

impl<F> Display for Config<F>
where
    F: FftField + CanonicalSerialize + CanonicalDeserialize,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Folding Reed-Solomon commitment (size = {}, rate = 1/{}, samples = {} + {}, folds = {}, soundness = {:.2} bits {})",
            self.size,
            self.expansion(),
            self.num_out_domain_samples,
            self.num_in_domain_samples,
            self.num_folds,
            self.round_by_round_soundness(),
            self.soundness_type,
        )
    }
}

impl<P> Pattern for P
where
    P: transcript::Pattern
        + serialize::Pattern
        + field::Pattern
        + challenge_indices::Pattern
        + merkle_tree::Pattern,
{
    fn reed_solomon_commit<F>(&mut self, label: Label, config: &Config<F>)
    where
        F: FftField + CanonicalSerialize + CanonicalDeserialize,
    {
        self.begin_protocol::<Config<F>>(label);
        self.merkle_tree_commit("evaluations", &config.merkle_tree);
        self.challenge_ark_fels::<F>("out-challenges", config.num_out_domain_samples);
        // TODO: Dedicated message for field values.
        self.message_arkworks::<Vec<F>>("out-answers");
        self.end_protocol::<Config<F>>(label);
    }

    fn reed_solomon_open<F>(&mut self, label: Label, config: &Config<F>)
    where
        F: FftField + CanonicalSerialize + CanonicalDeserialize,
    {
        self.begin_protocol::<Config<F>>(label);
        self.challenge_indices(
            "in-challenges",
            config.num_cosets(),
            config.num_in_domain_samples,
        );
        self.merkle_tree_open_with_leaves_ark(
            "in-answers",
            &config.merkle_tree,
            config.num_in_domain_samples,
        );
        self.end_protocol::<Config<F>>(label);
    }
}

impl<P> Prover for P
where
    P: transcript::Prover
        + serialize::Prover
        + field::Common
        + challenge_indices::Prover
        + merkle_tree::Prover,
{
    fn reed_solomon_commit<F>(
        &mut self,
        label: Label,
        config: &Config<F>,
        values: &[F],
    ) -> Witness<F>
    where
        F: FftField + CanonicalSerialize + CanonicalDeserialize,
    {
        self.begin_protocol::<Config<F>>(label);

        // Evaluate over evaluation_domain and commit to coset polynomials
        let mut evaluations = expand_from_coeff(values, config.expansion());
        transform_evaluations(
            &mut evaluations,
            config.evaluation_domain.group_gen_inv(),
            config.coset_size(),
        );
        let merkle_tree = self.merkle_tree_commit("evaluations", &config.merkle_tree, &evaluations);

        // Sample out of domains
        // TODO: Optionally sample from extension?
        let challenges =
            self.challenge_ark_fels_vec::<F>("out-challenges", config.num_out_domain_samples);
        // Batch Horner evaluation to get the answers.
        // OPT: parallelize over values
        let mut answers = vec![*values.last().unwrap(); challenges.len()];
        for &coeff in values.iter().rev().skip(1) {
            for (answer, &x) in answers.iter_mut().zip(challenges.iter()) {
                *answer *= x;
                *answer += coeff;
            }
        }
        self.message_arkworks("out-answers", &answers);
        let ood_samples = challenges.into_iter().zip(answers.into_iter()).collect();

        self.end_protocol::<Config<F>>(label);
        Witness {
            merkle_tree,
            evaluations,
            ood_samples,
        }
    }

    fn reed_solomon_open<F>(
        &mut self,
        label: Label,
        config: &Config<F>,
        witness: &Witness<F>,
    ) -> (Vec<F>, Vec<F>)
    where
        F: FftField + CanonicalSerialize + CanonicalDeserialize,
    {
        config.assert_valid_witness(witness);

        self.begin_protocol::<Config<F>>(label);

        // Compute in-domain challenge points.
        let indices = self.challenge_indices_vec(
            "in-challenges",
            config.num_cosets(),
            config.num_in_domain_samples,
        );
        let points = indices
            .iter()
            .map(|&i| config.evaluation_domain.element(i))
            .collect::<Vec<_>>();

        // OPT: How often do we get duplicate indices and should we optimize for this?
        // I don't think such an optimization needs to affect the transcript, for example
        // when we random-linear-combine the constraints we can handle duplicates as
        // (r_i + r_j) * constraint.
        // The leaves are provided as hints, so we can deduplicate those as well to reduce
        // proof size without affecting static control flow of the verifier (necessary for
        // recursion).

        // Select the evaluations and open the Merkle tree.
        let evaluations = self.merkle_tree_open_with_leaves_ark(
            "in-answers",
            &config.merkle_tree,
            &witness.evaluations,
            &witness.merkle_tree,
            &indices,
        );
        self.end_protocol::<Config<F>>(label);
        (points, evaluations)
    }
}

impl<'a, P> Verifier for P
where
    P: transcript::Verifier
        + serialize::Verifier
        + field::Common
        + challenge_indices::Verifier
        + merkle_tree::Verifier<'a>,
{
    fn reed_solomon_commit<F>(
        &mut self,
        label: Label,
        config: &Config<F>,
    ) -> Result<Commitment<F>, VerifierError>
    where
        F: FftField + CanonicalSerialize + CanonicalDeserialize,
    {
        self.begin_protocol::<Config<F>>(label);
        let merkle_tree = self.merkle_tree_commit("evaluations", &config.merkle_tree)?;
        let challenges =
            self.challenge_ark_fels_vec::<F>("out-challenges", config.num_out_domain_samples);
        // TODO: Dedicated pattern in field.
        let answers = self.message_arkworks::<Vec<F>>("out-answers").unwrap(); // TODO
        assert_eq!(answers.len(), config.num_out_domain_samples);
        let ood_samples = challenges.into_iter().zip(answers.into_iter()).collect();
        self.end_protocol::<Config<F>>(label);
        Ok(Commitment {
            merkle_tree,
            ood_samples,
        })
    }

    fn reed_solomon_open<F>(
        &mut self,
        label: Label,
        config: &Config<F>,
        witness: &Witness<F>,
    ) -> Result<(Vec<F>, Vec<F>), VerifierError>
    where
        F: FftField + CanonicalSerialize + CanonicalDeserialize,
    {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use std::iter;

    use anyhow::Result;
    use ark_ff::{AdditiveGroup, Field, MontBackend};
    use rand::Rng;
    use sha3::Keccak256;
    use spongefish::{transcript::PatternState, ProverState, VerifierState};

    use super::*;
    use crate::{
        crypto::fields::{FConfig64, Field64},
        protocols::merkle_tree::digest,
    };

    #[test]
    fn test_all_ops() -> Result<()> {
        type MyEngine =
            digest::DigestEngine<Keccak256, digest::ArkFieldUpdater<MontBackend<FConfig64, 1>, 1>>;
        let config = Config::new(
            Arc::new(MyEngine::new()),
            256,
            4,
            2,
            SoundnessType::ConjectureList,
            128,
        );
        eprintln!("{config}");

        let mut pattern: PatternState = PatternState::new();
        pattern.reed_solomon_commit("1", &config);
        pattern.reed_solomon_open("2", &config);
        let pattern = pattern.finalize();
        eprintln!("{pattern}");

        let mut values = (0..config.size)
            .map(|n| Field64::from(n as u64))
            .collect::<Vec<_>>();

        let mut prover: ProverState = ProverState::from(&pattern);
        let witness = prover.reed_solomon_commit("1", &config, &values);
        prover.reed_solomon_open("2", &config, &witness);
        let proof = prover.finalize();
        eprintln!("Proof size: {}", proof.len());
        assert_eq!(proof.len(), 5104);

        let mut verifier: VerifierState = VerifierState::new(pattern.into(), &proof);

        verifier.finalize();

        Ok(())
    }
}
