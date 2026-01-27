//! Interleaved Reed-Solomon Commitment Protocol

// COMMIT NOTES:
// Changes compared to previous version:
// - OODS answer are (poly,point) order, not (point,poly). This is for consistency with the in-domain samples.
// - Matrix commitment is over the subfield. This performs better when the subfield is smaller.

use std::fmt;

use ark_ff::{FftField, Field};
use ark_std::rand::{CryptoRng, RngCore};
use serde::{Deserialize, Serialize};
use spongefish::{Codec, Decoding, DuplexSpongeInterface, VerificationResult};
#[cfg(feature = "tracing")]
use tracing::instrument;

use crate::{
    algebra::{
        embedding::{Basefield, Embedding, Identity},
        mixed_dot, mixed_univariate_evaluate,
        ntt::{self, interleaved_rs_encode},
    },
    hash::Hash,
    protocols::matrix_commit,
    transcript::{ProverMessage, ProverState, VerifierMessage, VerifierState},
    type_info::{TypeInfo, Typed},
    verify,
};

/// Specialization of [`Config`] for commiting with identity embedding.
#[allow(type_alias_bounds)] // Bound is only to reference BasePrimeField.
pub type IdentityConfig<F: Field> = Config<F, F, Identity<F>>;

/// Specialization of [`Config`] for commiting over base fields
#[allow(type_alias_bounds)] // Bound is only to reference BasePrimeField.
pub type BasefieldConfig<F: Field> = Config<F::BasePrimeField, F, Basefield<F>>;

pub type Evaluations<F> = Vec<(F, Vec<F>)>;

/// Commit to polynomials over an fft-friendly field F
#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
#[serde(bound = "F: FftField, G: Field, M: Embedding<Source = F, Target = G>")]
pub struct Config<F, G = F, M = Identity<F>>
where
    F: FftField,
    G: Field,
    M: Embedding<Source = F, Target = G>,
{
    /// Embedding into a (larger) field used for weights and drawing challenges.
    pub embedding: Typed<M>,

    /// The number of polynomials to commit to in one operation.
    pub num_polynomials: usize,

    /// The number of coefficients in each polynomial.
    pub polynomial_size: usize,

    /// The Reed-Solomon expansion factor.
    pub expansion: usize,

    /// The number of independent codewords that are interleaved together.
    pub interleaving_depth: usize,

    /// The matrix commitment configuration.
    pub matrix_commit: matrix_commit::Config<F>,

    /// The number of in-domain samples.
    pub in_domain_samples: usize,

    /// The number of out-of-domain samples.
    pub out_domain_samples: usize,
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash, Default, Serialize, Deserialize)]
#[must_use]
pub struct Witness<F: FftField, G: Field> {
    pub matrix: Vec<F>,
    pub matrix_witness: matrix_commit::Witness,
    pub out_of_domain: Evaluations<G>,
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash, Default, Serialize, Deserialize)]
#[must_use]
pub struct Commitment<G: Field> {
    matrix_commitment: matrix_commit::Commitment,
    out_of_domain: Evaluations<G>,
}

impl<F, G, M> Config<F, G, M>
where
    F: FftField,
    G: Field,
    M: Embedding<Source = F, Target = G>,
{
    pub fn num_rows(&self) -> usize {
        self.matrix_commit.num_rows()
    }

    pub fn num_cols(&self) -> usize {
        self.matrix_commit.num_cols
    }

    pub fn size(&self) -> usize {
        self.matrix_commit.size()
    }

    pub fn embedding(&self) -> &M {
        &self.embedding
    }

    pub fn generator(&self) -> F {
        ntt::generator(self.num_rows()).expect("Subgroup of requested size not found")
    }

    /// Commit to one or more polynomials.
    ///
    /// Polynomials are given in coefficient form.
    #[cfg_attr(feature = "tracing", instrument(skip_all, fields(self = %self)))]
    pub fn commit<H, R>(
        &self,
        prover_state: &mut ProverState<H, R>,
        polynomials: &[&[F]],
    ) -> Witness<F, G>
    where
        H: DuplexSpongeInterface,
        R: RngCore + CryptoRng,
        G: Codec<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        assert!((self.polynomial_size * self.expansion).is_multiple_of(self.interleaving_depth));
        assert_eq!(polynomials.len(), self.num_polynomials);
        assert!(polynomials.iter().all(|p| p.len() == self.polynomial_size));

        // TODO: If only one polynomial, we can skip the copying.
        let mut matrix = vec![F::ZERO; self.size()];
        for (i, polynomial) in polynomials.iter().enumerate() {
            // Interleaved RS encode
            let evals = interleaved_rs_encode(polynomial, self.expansion, self.interleaving_depth);

            // Stack evaluations leaf-wise
            let dst = matrix.chunks_exact_mut(self.num_cols()).map(|leave| {
                leave
                    .chunks_exact_mut(self.interleaving_depth)
                    .nth(i)
                    .unwrap()
            });
            for (evals, leaves) in evals.chunks_exact(self.interleaving_depth).zip(dst) {
                leaves.copy_from_slice(evals);
            }
        }

        // Commit to the matrix
        let matrix_witness = self.matrix_commit.commit(prover_state, &matrix);

        // Handle out-of-domain points and values
        let oods_points: Vec<G> = prover_state.verifier_message_vec(self.out_domain_samples);
        let mut out_of_domain = Vec::with_capacity(self.out_domain_samples);
        for &point in &oods_points {
            let mut values = Vec::with_capacity(self.num_polynomials);
            for &polynomial in polynomials {
                let value = mixed_univariate_evaluate(&*self.embedding, polynomial, point);
                prover_state.prover_message(&value);
                values.push(value);
            }
            out_of_domain.push((point, values));
        }

        Witness {
            matrix,
            matrix_witness,
            out_of_domain,
        }
    }

    /// Receive a commitment to one or more polynomials.
    #[cfg_attr(feature = "tracing", instrument(skip_all, fields(self = %self)))]
    pub fn receive_commitment<H>(
        &self,
        verifier_state: &mut VerifierState<H>,
    ) -> VerificationResult<Commitment<G>>
    where
        H: DuplexSpongeInterface,
        Hash: ProverMessage<[H::U]>,
        G: Codec<[H::U]>,
    {
        let matrix_commitment = self.matrix_commit.receive_commitment(verifier_state)?;
        let oods_points: Vec<G> = verifier_state.verifier_message_vec(self.out_domain_samples);
        let mut out_of_domain = Vec::with_capacity(self.out_domain_samples);
        for point in oods_points {
            let mut answers = Vec::with_capacity(self.num_polynomials);
            for _ in 0..self.num_polynomials {
                answers.push(verifier_state.prover_message()?);
            }
            out_of_domain.push((point, answers));
        }
        Ok(Commitment {
            matrix_commitment,
            out_of_domain,
        })
    }

    /// Opens the commitment and returns the evaluations of the polynomials.
    ///
    /// Constraints are returned as a pairs of evaluation point and values for each polynomial.
    #[cfg_attr(feature = "tracing", instrument(skip_all, fields(self = %self)))]
    pub fn open<H, R>(
        &self,
        prover_state: &mut ProverState<H, R>,
        witness: &Witness<F, G>,
        weights: &[G],
    ) -> Evaluations<G>
    where
        H: DuplexSpongeInterface,
        R: RngCore + CryptoRng,
        u8: Decoding<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        assert_eq!(witness.matrix.len(), self.size());
        assert_eq!(witness.out_of_domain.len(), self.out_domain_samples);
        assert!(witness
            .out_of_domain
            .iter()
            .all(|o| o.1.len() == self.num_polynomials));
        assert_eq!(weights.len(), self.interleaving_depth);

        // Generate in-domain evaluations
        let mut evaluations = Vec::with_capacity(self.in_domain_samples);
        let generator = self.generator();
        let indices = challenge_indices(prover_state, self.num_rows(), self.in_domain_samples);
        let mut submatrix = Vec::with_capacity(self.in_domain_samples * self.num_cols());
        for &index in &indices {
            let row = &witness.matrix[index * self.num_cols()..(index + 1) * self.num_cols()];
            submatrix.extend_from_slice(row);

            // Linearly combine coefficients with weights
            evaluations.push((
                self.embedding.map(generator.pow([index as u64])),
                row.chunks_exact(self.interleaving_depth)
                    .map(|coeffs| mixed_dot(&*self.embedding, weights, coeffs))
                    .collect(),
            ));
        }
        prover_state.prover_hint_ark(&submatrix);
        self.matrix_commit
            .open(prover_state, &witness.matrix_witness, &indices);

        evaluations
    }

    /// Verifies an opening and returns the folded in-domain evaluations.
    ///
    /// **Note.** The verifier needs to separately verify the out-of-domain evaluations
    /// from [`Witness::out_of_domain()`]!
    #[cfg_attr(feature = "tracing", instrument(skip_all, fields(self = %self)))]
    pub fn verify<H>(
        &self,
        verifier_state: &mut VerifierState<H>,
        commitment: &Commitment<G>,
        weights: &[G],
    ) -> VerificationResult<Evaluations<G>>
    where
        H: DuplexSpongeInterface,
        u8: Decoding<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        verify!(commitment.out_of_domain.len() == self.out_domain_samples);
        verify!(commitment
            .out_of_domain
            .iter()
            .all(|o| o.1.len() == self.num_polynomials));

        // Get in-domain openings
        let indices = challenge_indices(verifier_state, self.num_rows(), self.in_domain_samples);
        let submatrix: Vec<F> = verifier_state.prover_hint_ark()?;
        self.matrix_commit.verify(
            verifier_state,
            commitment.matrix_commitment,
            &indices,
            &submatrix,
        )?;

        // Compute in-domain evaluations
        let mut evaluations = Vec::with_capacity(self.in_domain_samples);
        let generator = self.generator();
        let points = indices
            .into_iter()
            .map(|index| generator.pow([index as u64]));
        if self.num_cols() > 0 {
            for (point, row) in points.zip(submatrix.chunks_exact(self.num_cols())) {
                evaluations.push((
                    self.embedding.map(point),
                    row.chunks_exact(self.interleaving_depth)
                        .map(|coeffs| mixed_dot(&*self.embedding, weights, coeffs))
                        .collect(),
                ));
            }
        } else {
            // Degenerate cases
            verify!(self.num_polynomials == 0);
            evaluations.extend(points.map(|point| (self.embedding.map(point), Vec::new())));
        }

        Ok(evaluations)
    }
}

impl<G: Field> Commitment<G> {
    /// Returns the out-of-domain evaluations.
    pub fn out_of_domain(&self) -> &Evaluations<G> {
        &self.out_of_domain
    }
}

impl<F: FftField, G: Field> Witness<F, G> {
    /// Returns the out-of-domain evaluations.
    pub fn out_of_domain(&self) -> &Evaluations<G> {
        &self.out_of_domain
    }
}

impl<F, G, M> fmt::Display for Config<F, G, M>
where
    F: FftField,
    G: Field,
    M: Embedding<Source = F, Target = G> + TypeInfo + Serialize + for<'a> Deserialize<'a>,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "IRSCommit(count: {}, size: {}, interleaving: {}, samples: ({}, {}))",
            self.num_polynomials,
            self.polynomial_size,
            self.interleaving_depth,
            self.in_domain_samples,
            self.out_domain_samples
        )
    }
}

/// Generate a set of indices for challenges.
pub fn challenge_indices<T>(transcript: &mut T, num_leaves: usize, count: usize) -> Vec<usize>
where
    T: VerifierMessage,
    u8: Decoding<[T::U]>,
{
    if count == 0 {
        return Vec::new();
    }
    assert!(
        num_leaves.is_power_of_two(),
        "Number of leaves must be a power of two for unbiased results."
    );
    if num_leaves == 1 {
        return vec![0; count];
    }

    // Calculate the required bytes of entropy
    // TODO: Only round final result to bytes.
    let size_bytes = (num_leaves.ilog2() as usize).div_ceil(8);

    // Get required entropy bits.
    let entropy: Vec<u8> = (0..count * size_bytes)
        .map(|_| transcript.verifier_message())
        .collect();

    // Convert bytes into indices
    entropy
        .chunks_exact(size_bytes)
        .map(|chunk| chunk.iter().fold(0usize, |acc, &b| (acc << 8) | b as usize) % num_leaves)
        .collect::<Vec<usize>>()
}

#[cfg(test)]
mod tests {
    use ark_std::rand::{
        distributions::Standard, prelude::Distribution, rngs::StdRng, Rng, SeedableRng,
    };
    use proptest::{prelude::Strategy, proptest, sample::select, strategy::Just};
    use spongefish::{domain_separator, session};

    use super::*;
    use crate::{algebra::fields, transcript::codecs::U64};

    fn config<M: Embedding + Clone>(
        embedding: M,
        num_polynomials: usize,
        polynomial_size: usize,
        interleaving_depth: usize,
    ) -> impl Strategy<Value = Config<M::Source, M::Target, M>>
    where
        M::Source: FftField,
    {
        assert!(interleaving_depth != 0);
        assert!(polynomial_size.is_multiple_of(interleaving_depth));
        let base = polynomial_size / interleaving_depth;

        // Compute supported NTT domains for F
        let valid_expansions = (1..=30)
            .filter(|&n| ntt::generator::<M::Source>(base * n).is_some())
            .collect::<Vec<_>>();
        let expansion = select(valid_expansions);

        // Combine with a matrix commitment config
        let expansion_matrix = expansion.prop_flat_map(move |expansion| {
            (
                Just(expansion),
                matrix_commit::tests::config::<M::Source>(
                    polynomial_size * expansion / interleaving_depth,
                    interleaving_depth * num_polynomials,
                ),
            )
        });

        (expansion_matrix, 0_usize..=10, 0_usize..=10).prop_map(
            move |((expansion, matrix_commit), in_domain_samples, out_domain_samples)| Config {
                embedding: Typed::new(embedding.clone()),
                num_polynomials,
                polynomial_size,
                expansion,
                interleaving_depth,
                matrix_commit,
                in_domain_samples,
                out_domain_samples,
            },
        )
    }

    fn test<M: Embedding>(seed: u64, config: Config<M::Source, M::Target, M>)
    where
        M::Source: FftField + ProverMessage,
        M::Target: Codec,
        Standard: Distribution<M::Source>,
        Standard: Distribution<M::Target>,
    {
        crate::tests::init();

        // Pseudo-random Instance
        let instance = U64(seed);
        let ds = domain_separator!("whir::protocols::irs_commit")
            .session(session!("Test at {}:{}", file!(), line!()))
            .instance(&instance);
        let mut rng = StdRng::seed_from_u64(seed);
        let polynomials = (0..config.num_polynomials)
            .map(|_| {
                (0..config.polynomial_size)
                    .map(|_| rng.gen::<M::Source>())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let weights: Vec<M::Target> = (0..config.interleaving_depth).map(|_| rng.gen()).collect();
        let folded_polynomials: Vec<Vec<M::Target>> = polynomials
            .iter()
            .map(|polynomial| {
                polynomial
                    .chunks_exact(config.interleaving_depth)
                    .map(|coeffs| mixed_dot(config.embedding(), &weights, coeffs))
                    .collect()
            })
            .collect();

        // Prover
        let mut prover_state = ProverState::from(ds.std_prover());
        let witness = config.commit(
            &mut prover_state,
            &polynomials.iter().map(|p| p.as_slice()).collect::<Vec<_>>(),
        );
        assert_eq!(witness.out_of_domain().len(), config.out_domain_samples);
        for (point, evals) in witness.out_of_domain() {
            for (polynomial, expected) in polynomials.iter().zip(evals.iter()) {
                assert_eq!(
                    mixed_univariate_evaluate(config.embedding(), polynomial, *point),
                    *expected
                );
            }
        }
        let in_domain_evals = config.open(&mut prover_state, &witness, &weights);
        assert_eq!(in_domain_evals.len(), config.in_domain_samples);
        for (point, evals) in &in_domain_evals {
            for (polynomial, expected) in folded_polynomials.iter().zip(evals.iter()) {
                assert_eq!(
                    mixed_univariate_evaluate(&Identity::new(), polynomial, *point),
                    *expected
                );
            }
        }
        let proof = prover_state.proof();

        // Verifier
        let mut verifier_state =
            VerifierState::from(ds.std_verifier(&proof.narg_string), &proof.hints);
        let commitment = config.receive_commitment(&mut verifier_state).unwrap();
        assert_eq!(commitment.out_of_domain(), witness.out_of_domain());
        let verifier_in_domain_evals = config
            .verify(&mut verifier_state, &commitment, &weights)
            .unwrap();
        assert_eq!(verifier_in_domain_evals.len(), config.in_domain_samples);
        assert_eq!(&verifier_in_domain_evals, &in_domain_evals);
        verifier_state.check_eof().unwrap();
    }

    fn proptest<M: Embedding>(embedding: M)
    where
        M::Source: FftField + ProverMessage,
        M::Target: FftField + Codec,
        Standard: Distribution<M::Source>,
        Standard: Distribution<M::Target>,
    {
        let valid_sizes = (1..=1024)
            .filter(|&n| ntt::generator::<M::Source>(n).is_some())
            .collect::<Vec<_>>();
        let size = select(valid_sizes);

        let config = (0_usize..=3, size, 1_usize..=10).prop_flat_map(
            |(num_polynomials, size, interleaving_depth)| {
                config(
                    embedding.clone(),
                    num_polynomials,
                    size * interleaving_depth,
                    interleaving_depth,
                )
            },
        );
        proptest!(|(
            seed: u64,
            config in config,
        )| {
            test(seed, config);
        });
    }

    #[test]
    fn test_field64_1() {
        proptest(Identity::<fields::Field64>::new());
    }

    #[test]
    fn test_field64_2() {
        proptest(Identity::<fields::Field64_2>::new());
    }

    #[test]
    fn test_field64_3() {
        proptest(Identity::<fields::Field64_3>::new());
    }

    #[test]
    fn test_field128() {
        proptest(Identity::<fields::Field128>::new());
    }

    #[test]
    fn test_field192() {
        proptest(Identity::<fields::Field192>::new());
    }

    #[test]
    fn test_field256() {
        proptest(Identity::<fields::Field256>::new());
    }
}
