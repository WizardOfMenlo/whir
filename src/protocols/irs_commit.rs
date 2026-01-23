//! Interleaved Reed-Solomon Commitment Protocol

use ark_ff::FftField;
use ark_poly::{EvaluationDomain, GeneralEvaluationDomain};
use ark_std::rand::{CryptoRng, RngCore};
use serde::{Deserialize, Serialize};
use spongefish::{Codec, Decoding, DuplexSpongeInterface, VerificationError, VerificationResult};

use crate::{
    ensure,
    hash::Hash,
    ntt::interleaved_rs_encode,
    poly_utils::{coeffs::CoefficientList, multilinear::MultilinearPoint},
    protocols::matrix_commit,
    transcript::{ProverMessage, ProverState, VerifierMessage, VerifierState},
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

    /// The matrix commitment configuration.
    pub matrix_commit: matrix_commit::Config<F>,

    /// The number of in-domain samples.
    pub in_domain_samples: usize,

    /// The number of out-of-domain samples.
    pub out_domain_samples: usize,
}

pub struct Witness<F: FftField> {
    matrix: Vec<F>,
    matrix_witness: matrix_commit::Witness,
    out_of_domain: Evaluations<F>,
}

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

    pub fn fold_size(&self) -> usize {
        assert!(self
            .matrix_commit
            .num_cols
            .is_multiple_of(self.num_polynomials));
        self.matrix_commit
            .num_cols
            .checked_div(self.num_polynomials)
            .unwrap_or_default()
    }

    pub fn num_folds(&self) -> usize {
        let fold_size = self.fold_size();
        assert!(fold_size.is_power_of_two());
        fold_size.trailing_zeros() as usize
    }

    /// Commit to one or more polynomials.
    #[cfg_attr(feature = "tracing", instrument(skip(prover_state, polynomials)))]
    pub fn commit<H, R>(
        &self,
        prover_state: &mut ProverState<H, R>,
        polynomials: &[&CoefficientList<F>],
    ) -> Witness<F>
    where
        H: DuplexSpongeInterface,
        R: RngCore + CryptoRng,
        F: Codec<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        assert_eq!(polynomials.len(), self.num_polynomials);
        assert!(polynomials
            .iter()
            .all(|p| p.num_coeffs() == self.polynomial_size));

        let mut matrix = vec![F::ZERO; self.size()];
        for (i, polynomial) in polynomials.iter().enumerate() {
            // RS encode
            let evals =
                interleaved_rs_encode(polynomial.coeffs(), self.expansion, self.num_folds());

            // Stack evaluations leaf-wise
            let dst = matrix
                .chunks_exact_mut(self.num_cols())
                .map(|leave| leave.chunks_exact_mut(self.fold_size()).nth(i).unwrap());
            for (evals, leaves) in evals.chunks_exact(self.fold_size()).zip(dst) {
                leaves.copy_from_slice(evals);
            }
        }

        // Commit to the matrix
        let matrix_witness = self.matrix_commit.commit(prover_state, &matrix);

        // Handle out-of-domain points and values
        let oods_points = prover_state.verifier_message_vec(self.out_domain_samples);
        let mut out_of_domain = Vec::with_capacity(self.out_domain_samples);
        for &point in &oods_points {
            let mut values = Vec::with_capacity(self.num_polynomials);
            for polynomial in polynomials {
                let point =
                    MultilinearPoint::expand_from_univariate(point, polynomial.num_variables());
                let value = polynomial.evaluate(&point);
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
        let out_of_domain = oods_points
            .into_iter()
            .map(|point| {
                (
                    point,
                    (0..self.num_polynomials)
                        .map(|_| verifier_state.verifier_message())
                        .collect::<Vec<F>>(),
                )
            })
            .collect();
        Ok(Commitment {
            matrix_commitment,
            out_of_domain,
        })
    }

    /// Opens the commitment and returns the constraints on the polynomials.
    ///
    /// Constraints are returned as a pair of evaluation point and value for each polynomial.
    #[cfg_attr(
        feature = "tracing",
        instrument(skip(prover_state, witness, folding_randomess))
    )]
    pub fn open<H, R>(
        &self,
        prover_state: &mut ProverState<H, R>,
        witness: &Witness<F>,
        folding_randomness: &MultilinearPoint<F>,
    ) -> Vec<(F, Vec<F>)>
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
        assert_eq!(folding_randomness.num_variables(), self.num_folds());

        // Add out-of-domain evaluations
        let mut evaluations = Vec::with_capacity(self.out_domain_samples + self.in_domain_samples);
        evaluations.extend_from_slice(&witness.out_of_domain);

        // Generate in-domain evaluations
        let indices = challenge_indices(prover_state, self.num_rows(), self.in_domain_samples);
        let mut submatrix = Vec::with_capacity(self.in_domain_samples * self.num_cols());
        for &index in &indices {
            let row = witness
                .matrix
                .chunks_exact(self.num_cols())
                .nth(index)
                .expect("Challenge index out of range");
            submatrix.extend_from_slice(row);

            // Compute the point correpsonding to the index.
            let backing_domain: GeneralEvaluationDomain<F> = todo!();
            let domain_scaled_gen = backing_domain.element(self.fold_size());
            let x = domain_scaled_gen.pow([index as u64]);

            // Evaluate each polynomial in that point.
            let values = Vec::with_capacity(self.num_polynomials);
            for polynomial in row.chunks_exact(self.fold_size()) {
                let polynomial = CoefficientList::new(polynomial.to_vec());
                let value = polynomial.evaluate(folding_randomness);
                values.push(value);
            }
            evaluations.push((x, values));
        }
        prover_state.prover_hint_ark(&submatrix);
        self.matrix_commit
            .open(prover_state, &witness.matrix_witness, &indices);

        evaluations
    }
}

/// Generate a set of indices for challenges.
pub fn challenge_indices<T>(transcript: &mut T, num_leaves: usize, count: usize) -> Vec<usize>
where
    T: VerifierMessage,
    u8: Decoding<[T::U]>,
{
    assert!(
        num_leaves.is_power_of_two(),
        "Number of leaves must be a power of two for unbiased results."
    );

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
