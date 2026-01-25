use ark_ff::FftField;
use ark_poly::EvaluationDomain;
use ark_std::rand::{CryptoRng, RngCore};
use spongefish::{Codec, DuplexSpongeInterface};
#[cfg(feature = "tracing")]
use tracing::instrument;

use super::Witness;
use crate::{
    algebra::poly_utils::coeffs::CoefficientList,
    hash::Hash,
    transcript::{ProverMessage, ProverState, VerifierMessage},
    whir::{
        parameters::WhirConfig,
        utils::{compute_ood_response, sample_ood_points},
    },
};

/// Responsible for committing polynomials using a Merkle-based scheme.
///
/// The `Committer` processes a polynomial, expands and folds its evaluations,
/// and constructs a Merkle tree from the resulting values.
///
/// It provides a commitment that can be used for proof generation and verification.
pub struct CommitmentWriter<F>
where
    F: FftField,
{
    pub(crate) config: WhirConfig<F>,
}

impl<F> CommitmentWriter<F>
where
    F: FftField,
{
    pub const fn new(config: WhirConfig<F>) -> Self {
        Self { config }
    }

    #[allow(clippy::too_many_lines)]
    #[cfg_attr(feature = "tracing", instrument(skip_all, fields(size = polynomials.first().unwrap().num_coeffs())))]
    pub fn commit_batch<H, R>(
        &self,
        prover_state: &mut ProverState<H, R>,
        polynomials: &[&CoefficientList<F::BasePrimeField>],
    ) -> Witness<F>
    where
        H: DuplexSpongeInterface,
        R: RngCore + CryptoRng,
        F: Codec<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        assert!(!polynomials.is_empty());
        assert_eq!(polynomials.len(), self.config.batch_size);

        let num_vars = polynomials.first().unwrap().num_variables();
        let num_coeffs = polynomials.first().unwrap().num_coeffs();

        for poly in polynomials {
            assert_eq!(poly.num_variables(), num_vars);
            assert_eq!(poly.num_coeffs(), num_coeffs);
        }

        let base_domain = self.config.starting_domain.base_domain.unwrap();
        let expansion = base_domain.size() / num_coeffs;
        let fold_size = 1 << self.config.folding_factor.at_round(0);

        let num_leaves = (polynomials[0].num_coeffs() * expansion) / fold_size;
        let stacked_leaf_size = fold_size * polynomials.len();
        let mut stacked_leaves = vec![F::zero(); num_leaves * stacked_leaf_size];

        for (poly_idx, poly) in polynomials.iter().enumerate() {
            let evals = self.config.basefield_reed_solomon.interleaved_encode(
                poly.coeffs(),
                expansion,
                self.config.folding_factor.at_round(0),
            );

            for (i, chunk) in evals.chunks_exact(fold_size).enumerate() {
                let start_dst = i * stacked_leaf_size + poly_idx * fold_size;
                for (j, &eval) in chunk.iter().enumerate() {
                    stacked_leaves[start_dst + j] = F::from_base_prime_field(eval);
                }
            }
        }

        // Commit to the matrix.
        let matrix_witness = self
            .config
            .initial_matrix_committer
            .commit(prover_state, &stacked_leaves);

        let (ood_points, first_answers) = sample_ood_points(
            prover_state.inner_mut(),
            self.config.committment_ood_samples,
            self.config.mv_parameters.num_variables,
            |point| polynomials[0].evaluate_at_extension(point),
        );
        let mut per_poly_ood_answers: Vec<Vec<F>> = Vec::with_capacity(polynomials.len());
        per_poly_ood_answers.push(first_answers);
        for poly in polynomials.iter().skip(1) {
            let answers = compute_ood_response(
                prover_state.inner_mut(),
                &ood_points,
                self.config.mv_parameters.num_variables,
                |point| poly.evaluate_at_extension(point),
            );
            per_poly_ood_answers.push(answers);
        }

        let batching_randomness = if polynomials.len() > 1 {
            prover_state.verifier_message()
        } else {
            F::zero()
        };

        if polynomials.len() == 1 {
            return Witness {
                polynomial: polynomials[0].clone().to_extension(),
                matrix_witness,
                merkle_leaves: stacked_leaves,
                ood_points,
                ood_answers: per_poly_ood_answers.remove(0),
                batching_randomness,
            };
        }

        let mut batched_poly = polynomials[0].clone().to_extension().into_coeffs();

        let mut multiplier = batching_randomness;
        for poly in polynomials.iter().skip(1) {
            for (dst, src) in batched_poly.iter_mut().zip(poly.coeffs()) {
                *dst += multiplier * F::from_base_prime_field(*src);
            }
            multiplier *= batching_randomness;
        }

        let polynomial = CoefficientList::new(batched_poly);

        let mut per_poly_ood_iter = per_poly_ood_answers.into_iter();
        let mut batched_ood_resp = per_poly_ood_iter.next().unwrap();

        let mut multiplier = batching_randomness;
        for answers in per_poly_ood_iter {
            for (dst, src) in batched_ood_resp.iter_mut().zip(&answers) {
                *dst += multiplier * src;
            }
            multiplier *= batching_randomness;
        }

        Witness {
            polynomial,
            matrix_witness,
            merkle_leaves: stacked_leaves,
            ood_points,
            ood_answers: batched_ood_resp,
            batching_randomness,
        }
    }

    #[cfg_attr(feature = "tracing", instrument(skip_all, fields(size = polynomial.num_coeffs())))]
    pub fn commit<H, R>(
        &self,
        prover_state: &mut ProverState<H, R>,
        polynomial: &CoefficientList<F::BasePrimeField>,
    ) -> Witness<F>
    where
        H: DuplexSpongeInterface,
        R: RngCore + CryptoRng,
        F: Codec<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        self.commit_batch(prover_state, &[polynomial])
    }
}

impl<F> Witness<F>
where
    F: FftField,
{
    /// Returns the batched polynomial
    pub const fn batched_poly(&self) -> &CoefficientList<F> {
        &self.polynomial
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use ark_ff::UniformRand;
    use spongefish::{domain_separator, session};

    use super::*;
    use crate::{
        algebra::{fields::Field64, ntt::RSDefault, poly_utils::multilinear::MultilinearPoint},
        hash,
        parameters::{FoldingFactor, MultivariateParameters, ProtocolParameters, SoundnessType},
        transcript::codecs::Empty,
    };

    #[test]
    fn test_basic_commitment() {
        // Define the field type and Merkle tree configuration.
        type F = Field64;

        let mut rng = ark_std::test_rng();

        // Set up Whir protocol parameters.
        let security_level = 100;
        let pow_bits = 20;
        let num_variables = 5;
        let starting_rate = 1;
        let folding_factor = 4;
        let first_round_folding_factor = 4;

        let whir_params = ProtocolParameters {
            initial_statement: true,
            security_level,
            pow_bits,
            folding_factor: FoldingFactor::ConstantFromSecondRound(
                first_round_folding_factor,
                folding_factor,
            ),
            soundness_type: SoundnessType::ConjectureList,
            starting_log_inv_rate: starting_rate,
            batch_size: 1,
            hash_id: hash::SHA2,
        };

        // Define multivariate parameters for the polynomial.
        let mv_params = MultivariateParameters::<F>::new(num_variables);
        let reed_solomon = Arc::new(RSDefault);
        let basefield_reed_solomon = reed_solomon.clone();
        let params = WhirConfig::<F>::new(
            reed_solomon,
            basefield_reed_solomon,
            mv_params,
            &whir_params,
        );

        // Generate a random polynomial with 32 coefficients.
        let polynomial = CoefficientList::new(vec![F::rand(&mut rng); 32]);

        // Set up the DomainSeparator and initialize a ProverState narg_string.
        let mut prover_state = domain_separator!("whir::comitter::writer")
            .session(session!("Test at {}:{}", file!(), line!()))
            .instance(&Empty)
            .std_prover()
            .into();

        // Run the Commitment Phase
        let committer = CommitmentWriter::new(params.clone());
        let witness = committer.commit(&mut prover_state, &polynomial);

        // Ensure Merkle leaves are correctly generated.
        assert!(
            !witness.merkle_leaves.is_empty(),
            "Merkle leaves should not be empty"
        );

        // Ensure OOD (out-of-domain) points are generated.
        assert!(
            !witness.ood_points.is_empty(),
            "OOD points should be generated"
        );

        // Validate the number of generated OOD points.
        assert_eq!(
            witness.ood_points.len(),
            params.committment_ood_samples,
            "OOD points count should match expected samples"
        );

        // Check that the matrix commitment is valid
        assert_eq!(
            witness.matrix_witness.num_nodes(),
            params.initial_matrix_committer.merkle_tree.num_nodes()
        );

        // Ensure polynomial data is correctly stored
        assert_eq!(
            witness.polynomial.coeffs().len(),
            polynomial.coeffs().len(),
            "Stored polynomial should have the correct number of coefficients"
        );

        // Check that OOD answers match expected evaluations
        for (i, ood_point) in witness.ood_points.iter().enumerate() {
            let expected_eval = polynomial.evaluate_at_extension(
                &MultilinearPoint::expand_from_univariate(*ood_point, num_variables),
            );
            assert_eq!(
                witness.ood_answers[i], expected_eval,
                "OOD answer at index {i} should match expected evaluation"
            );
        }
    }

    #[test]
    fn test_large_polynomial() {
        type F = Field64;

        let mut rng = ark_std::test_rng();

        let reed_solomon = Arc::new(RSDefault);
        let basefield_reed_solomon = reed_solomon.clone();
        let params = WhirConfig::<F>::new(
            reed_solomon,
            basefield_reed_solomon,
            MultivariateParameters::<F>::new(10),
            &ProtocolParameters {
                initial_statement: true,
                security_level: 100,
                pow_bits: 20,
                folding_factor: FoldingFactor::ConstantFromSecondRound(4, 4),
                soundness_type: SoundnessType::ConjectureList,
                starting_log_inv_rate: 1,
                batch_size: 1,
                hash_id: hash::BLAKE3,
            },
        );

        let polynomial = CoefficientList::new(vec![F::rand(&mut rng); 1024]); // Large polynomial
        let mut prover_state = ProverState::from(
            domain_separator!("🌪️")
                .session(session!("Test at {}:{}", file!(), line!()))
                .instance(&Empty)
                .std_prover(),
        );

        let committer = CommitmentWriter::new(params);
        let witness = committer.commit(&mut prover_state, &polynomial);

        // Expansion factor is 2
        assert_eq!(
            witness.merkle_leaves.len(),
            1024 * 2,
            "Merkle tree should have expected number of leaves"
        );
    }

    #[test]
    fn test_commitment_without_ood_samples() {
        type F = Field64;

        let mut rng = ark_std::test_rng();
        let reed_solomon = Arc::new(RSDefault);
        let basefield_reed_solomon = reed_solomon.clone();
        let mut params = WhirConfig::<F>::new(
            reed_solomon,
            basefield_reed_solomon,
            MultivariateParameters::<F>::new(5),
            &ProtocolParameters {
                initial_statement: true,
                security_level: 100,
                pow_bits: 20,
                folding_factor: FoldingFactor::ConstantFromSecondRound(4, 4),
                soundness_type: SoundnessType::ConjectureList,
                starting_log_inv_rate: 1,
                batch_size: 1,
                hash_id: hash::BLAKE3,
            },
        );

        params.committment_ood_samples = 0; // No OOD samples

        let polynomial = CoefficientList::new(vec![F::rand(&mut rng); 32]);
        let mut prover_state = ProverState::from(
            domain_separator!("🌪️")
                .session(session!("Test at {}:{}", file!(), line!()))
                .instance(&Empty)
                .std_prover(),
        );

        let committer = CommitmentWriter::new(params);
        let witness = committer.commit(&mut prover_state, &polynomial);

        assert!(
            witness.ood_points.is_empty(),
            "There should be no OOD points when committment_ood_samples is 0"
        );
    }
}
