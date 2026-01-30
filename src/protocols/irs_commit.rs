//! Interleaved Reed-Solomon Commitment Protocol
//!
//! Commits to a `num_polynomials` by `polynomial_size` matrix over `F`.
//!
//! This will be reshaped into a `polynomial_size / interleaving_depth` by
//! `num_polynomials * interleaving_depth` matrix. Then each row is encoded
//! using an NTT friendly Reed-Solomon code to produce a `num_polynomials * interleaving_depth`
//! by `codeword_size` matrix. This matrix is committed using the [`matrix_commit`] protocol.
//!
//! After committing the encoded matrix, the protocol generates a random Reed-Solomon code of
//! length `out_domain_samples` over an extension field `G` of `F` and encodes the original
//! matrix using this code to produce a `num_polynomials` by `out_domain_samples` matrix over `G`.
//! Together, these two encoded matrices form a commitment to the original matrix.
//!
//! On opening the commitment, the protocol randomly selects `in_domain_samples` rows and opens
//! it using the [`matrix_commit`] protocol. Sampling is done with replacement, so may produce
//! fewer than `in_domain_samples` distinct rows. This produces `in_domain_samples` evaluation
//! points in `F` and `in_domain_samples` by `num_polynomials * interleaving_depth`.
//!
//! *To do:*:
//! - Consistently Reframe as vector commitment protocol (or, with batching, a matrix commitment protocol).
//! - Instead of `expansion` have `codeword_size` to allow non-integer expansion ratios.

// COMMIT NOTES:
// Changes compared to previous version:
// - OODS answer are (poly,point) order, not (point,poly). This is for consistency with the in-domain samples.
// - Matrix commitment is over the subfield. This performs better when the subfield is smaller.

use std::fmt;

use ark_ff::{FftField, Field};
use ark_std::rand::{CryptoRng, RngCore};
use serde::{Deserialize, Serialize};
#[cfg(feature = "tracing")]
use tracing::instrument;

use crate::{
    algebra::{
        embedding::{Basefield, Embedding, Identity},
        mixed_univariate_evaluate,
        ntt::{self, interleaved_rs_encode},
    },
    hash::Hash,
    protocols::matrix_commit,
    transcript::{
        Codec, Decoding, DuplexSpongeInterface, ProverMessage, ProverState, VerificationResult,
        VerifierMessage, VerifierState,
    },
    type_info::{TypeInfo, Typed},
    verify,
};

/// Specialization of [`Config`] for commiting with identity embedding.
#[allow(type_alias_bounds)] // Bound is only to reference BasePrimeField.
pub type IdentityConfig<F: Field> = Config<F, F, Identity<F>>;

/// Specialization of [`Config`] for commiting over base fields
#[allow(type_alias_bounds)] // Bound is only to reference BasePrimeField.
pub type BasefieldConfig<F: Field> = Config<F::BasePrimeField, F, Basefield<F>>;

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

    /// Whether to sort and deduplicate the in-domain samples.
    ///
    /// Deduplication can slightly reduce proof size and prover/verifier
    /// complexity, but it makes transcript pattern and control flow
    /// non-deterministic.
    pub deduplicate_in_domain: bool,
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

/// Interleaved Reed-Solomon code.
///
/// Used for out- and in-domain samples.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Serialize, Deserialize, Default)]
pub struct Evaluations<F> {
    /// Evaluation points for the RS code.
    pub points: Vec<F>,

    /// Matrix of codewords for each row.
    pub matrix: Vec<F>,
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
        // Validate config
        assert!((self.polynomial_size).is_multiple_of(self.interleaving_depth));
        assert_eq!(
            self.matrix_commit.num_rows(),
            (self.polynomial_size / self.interleaving_depth) * self.expansion
        );
        assert_eq!(
            self.matrix_commit.num_cols,
            self.num_polynomials * self.interleaving_depth
        );

        // Validate input
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
        let mut oods_matrix = Vec::with_capacity(self.out_domain_samples * self.num_polynomials);
        for &point in &oods_points {
            for &polynomial in polynomials {
                let value = mixed_univariate_evaluate(&*self.embedding, polynomial, point);
                prover_state.prover_message(&value);
                oods_matrix.push(value);
            }
        }

        Witness {
            matrix,
            matrix_witness,
            out_of_domain: Evaluations {
                points: oods_points,
                matrix: oods_matrix,
            },
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
        let oods_matrix =
            verifier_state.prover_messages_vec(self.out_domain_samples * self.num_polynomials)?;
        Ok(Commitment {
            matrix_commitment,
            out_of_domain: Evaluations {
                points: oods_points,
                matrix: oods_matrix,
            },
        })
    }

    /// Opens the commitment and returns the evaluations of the polynomials.
    ///
    /// Constraints are returned as a pair of evaluation point and values
    /// for each row.
    #[cfg_attr(feature = "tracing", instrument(skip_all, fields(self = %self)))]
    pub fn open<H, R>(
        &self,
        prover_state: &mut ProverState<H, R>,
        witness: &Witness<F, G>,
    ) -> Evaluations<F>
    where
        H: DuplexSpongeInterface,
        R: RngCore + CryptoRng,
        u8: Decoding<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        assert_eq!(witness.matrix.len(), self.size());
        assert_eq!(witness.out_of_domain.points.len(), self.out_domain_samples);
        assert_eq!(
            witness.out_of_domain.matrix.len(),
            self.out_domain_samples * self.num_polynomials
        );

        // Generate in-domain evaluations
        let indices = challenge_indices(
            prover_state,
            self.num_rows(),
            self.in_domain_samples,
            self.deduplicate_in_domain,
        );
        let mut submatrix = Vec::with_capacity(self.in_domain_samples * self.num_cols());
        for &index in &indices {
            let row = &witness.matrix[index * self.num_cols()..(index + 1) * self.num_cols()];
            submatrix.extend_from_slice(row);
        }
        prover_state.prover_hint_ark(&submatrix);
        self.matrix_commit
            .open(prover_state, &witness.matrix_witness, &indices);

        // Compute corresponding RS evaluation points
        let generator = self.generator();
        let points = indices
            .iter()
            .map(|&index| generator.pow([index as u64]))
            .collect();

        Evaluations {
            points,
            matrix: submatrix,
        }
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
    ) -> VerificationResult<Evaluations<F>>
    where
        H: DuplexSpongeInterface,
        u8: Decoding<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        verify!(commitment.out_of_domain.points.len() == self.out_domain_samples);
        verify!(
            commitment.out_of_domain.matrix.len() == self.num_polynomials * self.out_domain_samples
        );

        // Get in-domain openings
        let indices = challenge_indices(
            verifier_state,
            self.num_rows(),
            self.in_domain_samples,
            self.deduplicate_in_domain,
        );
        let submatrix: Vec<F> = verifier_state.prover_hint_ark()?;
        self.matrix_commit.verify(
            verifier_state,
            &commitment.matrix_commitment,
            &indices,
            &submatrix,
        )?;

        // Compute corresponding in-domain evaluation points
        let generator = self.generator();
        let points = indices
            .into_iter()
            .map(|index| generator.pow([index as u64]))
            .collect::<Vec<_>>();

        Ok(Evaluations {
            points,
            matrix: submatrix,
        })
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
pub fn challenge_indices<T>(
    transcript: &mut T,
    num_leaves: usize,
    count: usize,
    deduplicate: bool,
) -> Vec<usize>
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
        // `size_bytes` would be zero, making `chunks_exact` panic.
        return if deduplicate { vec![0] } else { vec![0; count] };
    }

    // Calculate the required bytes of entropy
    // TODO: Round total to bytes, instead of per index.
    let size_bytes = (num_leaves.ilog2() as usize).div_ceil(8);

    // Get required entropy bits.
    let entropy: Vec<u8> = (0..count * size_bytes)
        .map(|_| transcript.verifier_message())
        .collect();

    // Convert bytes into indices
    let mut indices = entropy
        .chunks_exact(size_bytes)
        .map(|chunk| chunk.iter().fold(0usize, |acc, &b| (acc << 8) | b as usize) % num_leaves)
        .collect::<Vec<usize>>();

    // Sort and deduplicate indices if requested
    if deduplicate {
        indices.sort_unstable();
        indices.dedup();
    }
    indices
}

#[cfg(test)]
mod tests {
    use ark_std::rand::{
        distributions::Standard, prelude::Distribution, rngs::StdRng, Rng, SeedableRng,
    };
    use proptest::{bool, prelude::Strategy, proptest, sample::select, strategy::Just};

    use super::*;
    use crate::{
        algebra::{
            embedding::{Compose, Frobenius},
            fields, univariate_evaluate,
        },
        transcript::{
            codecs::{Empty, U64},
            DomainSeparator,
        },
    };

    // Create a [`Strategy`] for generating [`irs_commit`] configurations.
    pub fn config<M: Embedding + Clone>(
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

        (expansion_matrix, 0_usize..=10, 0_usize..=10, bool::ANY).prop_map(
            move |(
                (expansion, matrix_commit),
                in_domain_samples,
                out_domain_samples,
                deduplicate_in_domain,
            )| Config {
                embedding: Typed::new(embedding.clone()),
                num_polynomials,
                polynomial_size,
                expansion,
                interleaving_depth,
                matrix_commit,
                in_domain_samples,
                out_domain_samples,
                deduplicate_in_domain,
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
        let ds = DomainSeparator::protocol(&config)
            .session(&format!("Test at {}:{}", file!(), line!()))
            .instance(&Empty);
        let mut rng = StdRng::seed_from_u64(seed);
        let polynomials = (0..config.num_polynomials)
            .map(|_| {
                (0..config.polynomial_size)
                    .map(|_| rng.gen::<M::Source>())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        // Prover
        let mut prover_state = ProverState::new_std(&ds);
        let witness = config.commit(
            &mut prover_state,
            &polynomials.iter().map(|p| p.as_slice()).collect::<Vec<_>>(),
        );
        assert_eq!(
            witness.out_of_domain().points.len(),
            config.out_domain_samples
        );
        assert_eq!(
            witness.out_of_domain().matrix.len(),
            config.out_domain_samples * config.num_polynomials
        );
        if config.num_polynomials > 0 {
            for (point, evals) in witness.out_of_domain().points.iter().zip(
                witness
                    .out_of_domain()
                    .matrix
                    .chunks_exact(config.num_polynomials),
            ) {
                for (polynomial, expected) in polynomials.iter().zip(evals.iter()) {
                    assert_eq!(
                        mixed_univariate_evaluate(config.embedding(), polynomial, *point),
                        *expected
                    );
                }
            }
        }
        let in_domain_evals = config.open(&mut prover_state, &witness);
        if config.deduplicate_in_domain {
            // Sorting is over index order, not points
            assert!(in_domain_evals.points.len() <= config.in_domain_samples);
            assert!({
                let mut unique = in_domain_evals.points.clone();
                unique.sort_unstable();
                unique.dedup();
                unique.len() == in_domain_evals.points.len()
            });
        } else {
            assert_eq!(in_domain_evals.points.len(), config.in_domain_samples);
        }
        assert_eq!(
            in_domain_evals.matrix.len(),
            in_domain_evals.points.len() * config.num_polynomials * config.interleaving_depth
        );
        if config.num_polynomials > 0 {
            for (point, evals) in in_domain_evals.points.iter().zip(
                in_domain_evals
                    .matrix
                    .chunks_exact(config.num_polynomials * config.interleaving_depth),
            ) {
                let expected_iter = polynomials.iter().flat_map(|poly| {
                    (0..config.interleaving_depth).map(|j| {
                        // coefficients at positions j, j+d, j+2d, ...
                        let coeffs: Vec<_> = poly
                            .iter()
                            .copied()
                            .skip(j)
                            .step_by(config.interleaving_depth)
                            .collect();
                        univariate_evaluate(&coeffs, *point)
                    })
                });
                for (expected, got) in expected_iter.zip(evals.iter()) {
                    assert_eq!(expected, *got);
                }
            }
        }
        let proof = prover_state.proof();

        // Verifier
        let mut verifier_state = VerifierState::new_std(&ds, &proof);
        let commitment = config.receive_commitment(&mut verifier_state).unwrap();
        assert_eq!(commitment.out_of_domain(), witness.out_of_domain());
        let verifier_in_domain_evals = config.verify(&mut verifier_state, &commitment).unwrap();
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
    #[ignore]
    fn test_field64_2() {
        proptest(Identity::<fields::Field64_2>::new());
    }

    #[test]
    #[ignore]
    fn test_field64_3() {
        proptest(Identity::<fields::Field64_3>::new());
    }

    #[test]
    #[ignore]
    fn test_field128() {
        proptest(Identity::<fields::Field128>::new());
    }

    #[test]
    #[ignore]
    fn test_field192() {
        proptest(Identity::<fields::Field192>::new());
    }

    #[test]
    #[ignore]
    fn test_field256() {
        proptest(Identity::<fields::Field256>::new());
    }

    #[test]
    fn test_basefield_field64_2() {
        proptest(Basefield::<fields::Field64_2>::new());
    }

    #[test]
    #[ignore]
    fn test_basefield_field64_3() {
        proptest(Basefield::<fields::Field64_3>::new());
    }

    #[test]
    #[ignore]
    fn test_base_frob_field64_3() {
        let embedding = Compose::new(Basefield::<fields::Field64_3>::new(), Frobenius::new(2));
        proptest(embedding);
    }
}
