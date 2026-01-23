//! Interleaved Reed-Solomon Commitment Protocol

use std::fmt;

use ark_ff::{FftField, Field};
use ark_std::rand::{CryptoRng, RngCore};
use serde::{Deserialize, Serialize};
use spongefish::{Codec, Decoding, DuplexSpongeInterface, VerificationResult};
#[cfg(feature = "tracing")]
use tracing::instrument;

use crate::{
    hash::Hash,
    ntt::{self, interleaved_rs_encode},
    protocols::matrix_commit,
    transcript::{ProverMessage, ProverState, VerifierMessage, VerifierState},
    verify,
};

pub type Evaluations<F> = Vec<(F, Vec<F>)>;

#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
#[serde(bound = "F: FftField")]
pub struct Config<F: FftField> {
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
pub struct Witness<F: FftField> {
    matrix: Vec<F>,
    matrix_witness: matrix_commit::Witness,
    out_of_domain: Evaluations<F>,
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash, Default, Serialize, Deserialize)]
#[must_use]
pub struct Commitment<F: FftField> {
    matrix_commitment: matrix_commit::Commitment,
    out_of_domain: Evaluations<F>,
}

impl<F: FftField> Config<F> {
    pub fn num_rows(&self) -> usize {
        self.matrix_commit.num_rows()
    }

    pub fn num_cols(&self) -> usize {
        self.matrix_commit.num_cols
    }

    pub fn size(&self) -> usize {
        self.matrix_commit.size()
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
    ) -> Witness<F>
    where
        H: DuplexSpongeInterface,
        R: RngCore + CryptoRng,
        F: Codec<[H::U]>,
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
        let oods_points = prover_state.verifier_message_vec(self.out_domain_samples);
        let mut out_of_domain = Vec::with_capacity(self.out_domain_samples);
        for &point in &oods_points {
            dbg!(point);
            let mut values = Vec::with_capacity(self.num_polynomials);
            for &polynomial in polynomials {
                let value = univariate_evaluate(polynomial, point);
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
    ) -> VerificationResult<Commitment<F>>
    where
        H: DuplexSpongeInterface,
        Hash: ProverMessage<[H::U]>,
        F: Codec<[H::U]>,
    {
        let matrix_commitment = self.matrix_commit.receive_commitment(verifier_state)?;
        let oods_points: Vec<F> = verifier_state.verifier_message_vec(self.out_domain_samples);
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
        witness: &Witness<F>,
        weights: &[F],
    ) -> Evaluations<F>
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
                generator.pow([index as u64]),
                row.chunks_exact(self.interleaving_depth)
                    .map(|coeffs| dot(coeffs, weights))
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
        commitment: &Commitment<F>,
        weights: &[F],
    ) -> VerificationResult<Evaluations<F>>
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
                    point,
                    row.chunks_exact(self.interleaving_depth)
                        .map(|coeffs| dot(coeffs, weights))
                        .collect(),
                ));
            }
        } else {
            // Degenerate cases
            verify!(self.num_polynomials == 0);
            evaluations.extend(points.map(|point| (point, Vec::new())));
        }
        dbg!(&evaluations);

        Ok(evaluations)
    }
}

impl<F: FftField> Witness<F> {
    /// Returns the out-of-domain evaluations.
    pub fn out_of_domain(&self) -> &Evaluations<F> {
        &self.out_of_domain
    }
}

impl<F: FftField> Commitment<F> {
    /// Returns the out-of-domain evaluations.
    pub fn out_of_domain(&self) -> &Evaluations<F> {
        &self.out_of_domain
    }
}

impl<F: FftField> fmt::Display for Config<F> {
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

pub fn univariate_evaluate<F: Field>(coefficients: &[F], point: F) -> F {
    coefficients
        .iter()
        .rev()
        .fold(F::ZERO, |acc, &coeff| acc * point + coeff)
}

pub fn dot<F: Field>(a: &[F], b: &[F]) -> F {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use ark_std::rand::{
        distributions::Standard, prelude::Distribution, rngs::StdRng, Rng, SeedableRng,
    };
    use proptest::{prop_assume, proptest};
    use spongefish::{domain_separator, session};

    use super::*;
    use crate::{crypto::fields, transcript::codecs::Empty};

    fn test<F>(
        seed: u64,
        num_polynomials: usize,
        polynomial_size: usize,
        interleaving_depth: usize,
        expansion: usize,
        in_domain_samples: usize,
        out_domain_samples: usize,
    ) where
        F: FftField + Codec,
        Standard: Distribution<F>,
    {
        dbg!((
            seed,
            num_polynomials,
            polynomial_size,
            interleaving_depth,
            expansion,
            in_domain_samples,
            out_domain_samples
        ));
        crate::tests::init();
        let mut rng = StdRng::seed_from_u64(seed);

        assert!(interleaving_depth >= 1);

        // Config
        let ds = domain_separator!("whir::protocols::irs_commit")
            .session(session!("Test at {}:{}", file!(), line!()))
            .instance(&Empty);
        let config = Config {
            num_polynomials,
            polynomial_size,
            interleaving_depth,
            expansion,
            matrix_commit: matrix_commit::Config::new(
                polynomial_size * expansion / interleaving_depth,
                interleaving_depth * num_polynomials,
            ),
            in_domain_samples,
            out_domain_samples,
        };

        // Instance
        let polynomials = (0..num_polynomials)
            .map(|_| {
                (0..polynomial_size)
                    .map(|_| rng.gen::<F>())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let weights: Vec<F> = (0..interleaving_depth).map(|_| rng.gen::<F>()).collect();

        // Prover
        let mut prover_state = ProverState::from(ds.std_prover());
        let witness = config.commit(
            &mut prover_state,
            &polynomials.iter().map(|p| p.as_slice()).collect::<Vec<_>>(),
        );
        let in_domain_evals = config.open(&mut prover_state, &witness, &weights);
        let proof = prover_state.proof();
        assert_eq!(witness.out_of_domain().len(), out_domain_samples);
        assert_eq!(in_domain_evals.len(), in_domain_samples);

        // Verifier
        let mut verifier_state =
            VerifierState::from(ds.std_verifier(&proof.narg_string), &proof.hints);
        let commitment = config.receive_commitment(&mut verifier_state).unwrap();
        let verifier_in_domain_evals = config
            .verify(&mut verifier_state, &commitment, &weights)
            .unwrap();
        verifier_state.check_eof().unwrap();
        assert_eq!(commitment.out_of_domain().len(), out_domain_samples);
        assert_eq!(verifier_in_domain_evals.len(), in_domain_samples);
        assert_eq!(witness.out_of_domain(), commitment.out_of_domain());
        assert_eq!(&in_domain_evals, &verifier_in_domain_evals);

        // Check out-domain evaluations
        let out_domain_evals = commitment.out_of_domain();
        for (point, evals) in out_domain_evals {
            for (polynomial, expected) in polynomials.iter().zip(evals.iter()) {
                assert_eq!(univariate_evaluate(polynomial, *point), *expected);
            }
        }
        // Fold polynomials
        let polynomials: Vec<Vec<F>> = polynomials
            .into_iter()
            .map(|polynomial| {
                polynomial
                    .chunks_exact(interleaving_depth)
                    .map(|coeffs| dot(coeffs, &weights))
                    .collect()
            })
            .collect();
        // Check in-domain evaluations
        for (point, evals) in &in_domain_evals {
            for (polynomial, expected) in polynomials.iter().zip(evals.iter()) {
                assert_eq!(univariate_evaluate(polynomial, *point), *expected);
            }
        }
    }

    fn proptest<F>()
    where
        F: FftField + Codec,
        Standard: Distribution<F>,
    {
        let valid_domains = (0..100)
            .filter(|&n| ntt::generator::<F>(n).is_some())
            .collect::<Vec<_>>();
        dbg!(valid_domains);

        proptest!(|(
            seed: u64,
            num_polynomials in 0_usize..4,
            polynomial_size in 0_usize..10,
            interleaving_depth in 1_usize..10,
            expansion in 1_usize..10,
            in_domain_samples in 0_usize..10,
            out_domain_samples in 0_usize..10
        )| {
            // Polynomial size must be multiple of interleaving depth.
            let polynomial_size = polynomial_size * interleaving_depth;

            // F^* must have a subgroup of correct size.
            let domain_size = polynomial_size * expansion / interleaving_depth;
            prop_assume!(ntt::generator::<F>(domain_size).is_some());

            test::<F>(seed, num_polynomials, polynomial_size, interleaving_depth, expansion, in_domain_samples, out_domain_samples);
        });
    }

    #[test]
    fn test_field64_1() {
        proptest::<fields::Field64>();
    }

    #[test]
    fn test_field64_2() {
        proptest::<fields::Field64_2>();
    }

    #[test]
    fn test_field64_3() {
        proptest::<fields::Field64_3>();
    }

    #[test]
    fn test_field256() {
        proptest::<fields::Field256>();
    }
}
