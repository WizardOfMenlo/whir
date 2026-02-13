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
//! - Support mixed `num_polys` openings.

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
        dot,
        embedding::{Basefield, Embedding, Identity},
        lift, mixed_univariate_evaluate,
        ntt::{self, interleaved_rs_encode},
        weights::UnivariateEvaluation,
    },
    hash::Hash,
    protocols::{challenge_indices::challenge_indices, matrix_commit},
    transcript::{
        Codec, Decoding, DuplexSpongeInterface, ProverMessage, ProverState, VerificationResult,
        VerifierMessage, VerifierState,
    },
    type_info::{TypeInfo, Typed},
    utils::zip_strict,
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
    pub const fn num_rows(&self) -> usize {
        self.matrix_commit.num_rows()
    }

    pub const fn num_cols(&self) -> usize {
        self.matrix_commit.num_cols
    }

    pub const fn size(&self) -> usize {
        self.matrix_commit.size()
    }

    pub fn embedding(&self) -> &M {
        &self.embedding
    }

    pub fn generator(&self) -> F {
        ntt::generator(self.num_rows()).expect("Subgroup of requested size not found")
    }

    pub fn rate(&self) -> f64 {
        1.0 / self.expansion as f64
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
            for (evals, leaves) in zip_strict(evals.chunks_exact(self.interleaving_depth), dst) {
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
    ///
    /// When there are multiple openings, the evaluation matrices will
    /// be horizontally concatenated.
    #[cfg_attr(feature = "tracing", instrument(skip_all, fields(self = %self)))]
    pub fn open<H, R>(
        &self,
        prover_state: &mut ProverState<H, R>,
        witnesses: &[&Witness<F, G>],
    ) -> Evaluations<F>
    where
        H: DuplexSpongeInterface,
        R: RngCore + CryptoRng,
        u8: Decoding<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        for witness in witnesses {
            assert_eq!(witness.matrix.len(), self.size());
            assert_eq!(witness.out_of_domain.points.len(), self.out_domain_samples);
            assert_eq!(
                witness.out_of_domain.matrix.len(),
                self.out_domain_samples * self.num_polynomials
            );
        }

        // Get in-domain openings
        let (indices, points) = self.in_domain_challenges(prover_state);

        // For each commitment, send the selected rows to the verifier
        // and collect them in the evaluation matrix.
        let stride = witnesses.len() * self.num_cols();
        let mut matrix = vec![F::ZERO; indices.len() * stride];
        let mut submatrix = Vec::with_capacity(indices.len() * self.num_cols());
        let mut matrix_col_offset = 0;
        for witness in witnesses {
            submatrix.clear();
            for (point_index, &code_index) in indices.iter().enumerate() {
                let row = &witness.matrix
                    [code_index * self.num_cols()..(code_index + 1) * self.num_cols()];
                submatrix.extend_from_slice(row);

                let matrix_row = &mut matrix[point_index * stride..(point_index + 1) * stride];
                matrix_row[matrix_col_offset..matrix_col_offset + self.num_cols()]
                    .copy_from_slice(row);
            }
            prover_state.prover_hint_ark(&submatrix);
            self.matrix_commit
                .open(prover_state, &witness.matrix_witness, &indices);
            matrix_col_offset += self.num_cols();
        }

        Evaluations { points, matrix }
    }

    /// Verifies one or more openings and returns the in-domain evaluations.
    ///
    /// **Note.** This does not verify the out-of-domain evaluations.
    #[cfg_attr(feature = "tracing", instrument(skip_all, fields(self = %self)))]
    pub fn verify<H>(
        &self,
        verifier_state: &mut VerifierState<H>,
        commitments: &[&Commitment<G>],
    ) -> VerificationResult<Evaluations<F>>
    where
        H: DuplexSpongeInterface,
        u8: Decoding<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        for commitment in commitments {
            verify!(commitment.out_of_domain.points.len() == self.out_domain_samples);
            verify!(
                commitment.out_of_domain.matrix.len()
                    == self.num_polynomials * self.out_domain_samples
            );
        }

        // Get in-domain openings
        let (indices, points) = self.in_domain_challenges(verifier_state);

        // Receive (as a hint) a matrix of all the columns of all the commitments
        // corresponding to the in-domain opening rows.
        let stride = commitments.len() * self.num_cols();
        let mut matrix = vec![F::ZERO; indices.len() * stride];
        let mut matrix_col_offset = 0;
        for commitment in commitments {
            let submatrix: Vec<F> = verifier_state.prover_hint_ark()?;
            self.matrix_commit.verify(
                verifier_state,
                &commitment.matrix_commitment,
                &indices,
                &submatrix,
            )?;
            // Horizontally concatenate matrices.
            if stride != 0 && self.num_cols() != 0 {
                for (dst, src) in zip_strict(
                    matrix.chunks_exact_mut(stride),
                    submatrix.chunks_exact(self.num_cols()),
                ) {
                    dst[matrix_col_offset..matrix_col_offset + self.num_cols()]
                        .copy_from_slice(src);
                }
            }
            matrix_col_offset += self.num_cols();
        }
        Ok(Evaluations { points, matrix })
    }

    fn in_domain_challenges<T>(&self, transcript: &mut T) -> (Vec<usize>, Vec<F>)
    where
        T: VerifierMessage,
        u8: Decoding<[T::U]>,
    {
        // Get in-domain openings
        let indices = challenge_indices(
            transcript,
            self.num_rows(),
            self.in_domain_samples,
            self.deduplicate_in_domain,
        );

        // Compute corresponding in-domain evaluation points
        let generator = self.generator();
        let points = indices
            .iter()
            .map(|index| generator.pow([*index as u64]))
            .collect::<Vec<_>>();

        (indices, points)
    }
}

impl<G: Field> Commitment<G> {
    /// Returns the out-of-domain evaluations.
    pub const fn out_of_domain(&self) -> &Evaluations<G> {
        &self.out_of_domain
    }

    pub fn num_polynomials(&self) -> usize {
        self.out_of_domain().num_columns()
    }
}

impl<F: FftField, G: Field> Witness<F, G> {
    /// Returns the out-of-domain evaluations.
    pub const fn out_of_domain(&self) -> &Evaluations<G> {
        &self.out_of_domain
    }

    pub fn num_polynomials(&self) -> usize {
        self.out_of_domain().num_columns()
    }
}

impl<F: Field> Evaluations<F> {
    pub const fn num_points(&self) -> usize {
        self.points.len()
    }

    pub fn num_columns(&self) -> usize {
        self.matrix
            .len()
            .checked_div(self.num_points())
            .unwrap_or_default()
    }

    pub fn rows(&self) -> impl Iterator<Item = &[F]> {
        let cols = self.num_columns();
        (0..self.num_points()).map(move |i| &self.matrix[i * cols..(i + 1) * cols])
    }

    pub fn lift<M>(&self, embedding: &M) -> Evaluations<M::Target>
    where
        M: Embedding<Source = F>,
    {
        Evaluations {
            points: lift(embedding, &self.points),
            matrix: lift(embedding, &self.matrix),
        }
    }

    pub fn weights(&self, size: usize) -> impl '_ + Iterator<Item = UnivariateEvaluation<F>> {
        self.points
            .iter()
            .map(move |&point| UnivariateEvaluation::new(point, size))
    }

    pub fn values<'a>(&'a self, weights: &'a [F]) -> impl 'a + Iterator<Item = F> {
        self.rows().map(|row| dot(weights, row))
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
            "size {}×{}/{}",
            self.num_polynomials, self.polynomial_size, self.interleaving_depth,
        )?;
        if self.expansion.is_power_of_two() {
            write!(f, " rate 2⁻{}", self.expansion.ilog2() as usize,)?;
        } else {
            write!(f, " rate 1/{}", self.expansion,)?;
        }
        write!(
            f,
            " samples {} in- {} out-domain",
            self.in_domain_samples, self.out_domain_samples
        )
    }
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
        transcript::{codecs::U64, DomainSeparator},
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

    fn test<M: Embedding>(seed: u64, config: &Config<M::Source, M::Target, M>)
    where
        M::Source: FftField + ProverMessage,
        M::Target: Codec,
        Standard: Distribution<M::Source> + Distribution<M::Target>,
    {
        crate::tests::init();

        // Pseudo-random Instance
        let instance = U64(seed);
        let ds = DomainSeparator::protocol(config)
            .session(&format!("Test at {}:{}", file!(), line!()))
            .instance(&instance);
        let mut rng = StdRng::seed_from_u64(seed);
        let polynomials = (0..config.num_polynomials)
            .map(|_| {
                (0..config.polynomial_size)
                    .map(|_| rng.gen::<M::Source>())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        // TODO: Multiple commitments and openings.

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
            for (point, evals) in zip_strict(
                witness.out_of_domain().points.iter(),
                witness
                    .out_of_domain()
                    .matrix
                    .chunks_exact(config.num_polynomials),
            ) {
                for (polynomial, expected) in zip_strict(polynomials.iter(), evals.iter()) {
                    assert_eq!(
                        mixed_univariate_evaluate(config.embedding(), polynomial, *point),
                        *expected
                    );
                }
            }
        }
        let in_domain_evals = config.open(&mut prover_state, &[&witness]);
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
            let base = config.polynomial_size / config.interleaving_depth;
            for (point, evals) in zip_strict(
                &in_domain_evals.points,
                in_domain_evals
                    .matrix
                    .chunks_exact(config.num_polynomials * config.interleaving_depth),
            ) {
                let expected_iter = polynomials.iter().flat_map(|poly| {
                    (0..config.interleaving_depth).map(|j| {
                        // coefficients in the contiguous block for this interleaving index
                        let start = j * base;
                        let coeffs: Vec<_> = poly.iter().copied().skip(start).take(base).collect();
                        univariate_evaluate(&coeffs, *point)
                    })
                });
                for (expected, got) in zip_strict(expected_iter, evals.iter()) {
                    assert_eq!(expected, *got);
                }
            }
        }
        let proof = prover_state.proof();

        // Verifier
        let mut verifier_state = VerifierState::new_std(&ds, &proof);
        let commitment = config.receive_commitment(&mut verifier_state).unwrap();
        assert_eq!(commitment.out_of_domain(), witness.out_of_domain());
        let verifier_in_domain_evals = config.verify(&mut verifier_state, &[&commitment]).unwrap();
        assert_eq!(&verifier_in_domain_evals, &in_domain_evals);
        verifier_state.check_eof().unwrap();
    }

    fn proptest<M: Embedding>(embedding: &M)
    where
        M::Source: FftField + ProverMessage,
        M::Target: FftField + Codec,
        Standard: Distribution<M::Source> + Distribution<M::Target>,
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
            test(seed, &config);
        });
    }

    #[test]
    fn test_field64_1() {
        proptest(&Identity::<fields::Field64>::new());
    }

    #[test]
    #[ignore = "Somewhat expensive and redundant"]
    fn test_field64_2() {
        proptest(&Identity::<fields::Field64_2>::new());
    }

    #[test]
    #[ignore = "Somewhat expensive and redundant"]
    fn test_field64_3() {
        proptest(&Identity::<fields::Field64_3>::new());
    }

    #[test]
    #[ignore = "Somewhat expensive and redundant"]
    fn test_field128() {
        proptest(&Identity::<fields::Field128>::new());
    }

    #[test]
    #[ignore = "Somewhat expensive and redundant"]
    fn test_field192() {
        proptest(&Identity::<fields::Field192>::new());
    }

    #[test]
    #[ignore = "Somewhat expensive and redundant"]
    fn test_field256() {
        proptest(&Identity::<fields::Field256>::new());
    }

    #[test]
    fn test_basefield_field64_2() {
        proptest(&Basefield::<fields::Field64_2>::new());
    }

    #[test]
    #[ignore = "Somewhat expensive and redundant"]
    fn test_basefield_field64_3() {
        proptest(&Basefield::<fields::Field64_3>::new());
    }

    #[test]
    #[ignore = "Somewhat expensive and redundant"]
    fn test_base_frob_field64_3() {
        let embedding = Compose::new(Basefield::<fields::Field64_3>::new(), Frobenius::new(2));
        proptest(&embedding);
    }
}
