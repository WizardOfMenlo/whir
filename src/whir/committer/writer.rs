use ark_ff::FftField;
use ark_std::rand::{CryptoRng, RngCore};
#[cfg(feature = "tracing")]
use tracing::instrument;

use super::Witness;
use crate::{
    algebra::{embedding::Embedding, geometric_sequence, poly_utils::coeffs::CoefficientList},
    hash::Hash,
    transcript::{Codec, DuplexSpongeInterface, ProverMessage, ProverState, VerifierMessage},
    whir::parameters::WhirConfig,
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
        let config = &self.config.initial_committer;
        let poly_refs = polynomials
            .iter()
            .map(|poly| poly.coeffs())
            .collect::<Vec<_>>();
        let witness = config.commit(prover_state, poly_refs.as_slice());

        // Computed batched extended polynomial
        let batching_randomness = if polynomials.len() > 1 {
            prover_state.verifier_message()
        } else {
            F::ZERO
        };
        let weights = geometric_sequence(batching_randomness, config.num_polynomials);
        let mut batched = vec![F::ZERO; config.polynomial_size];
        for (polynomial, weight) in polynomials.iter().zip(weights.iter()) {
            for (dst, src) in batched.iter_mut().zip(polynomial.coeffs()) {
                *dst += config.embedding.mixed_mul(*weight, *src);
            }
        }

        Witness {
            polynomial: CoefficientList::new(batched),
            witness,
            batching_randomness,
        }
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

    use super::*;
    use crate::{
        algebra::{fields::Field64, ntt::RSDefault, poly_utils::multilinear::MultilinearPoint},
        hash,
        parameters::{FoldingFactor, MultivariateParameters, ProtocolParameters, SoundnessType},
        transcript::{codecs::Empty, DomainSeparator},
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
        let ds = DomainSeparator::protocol(&module_path!())
            .session(&format!("Test at {}:{}", file!(), line!()))
            .instance(&Empty);
        let mut prover_state = ProverState::new_std(&ds);

        // Run the Commitment Phase
        let committer = CommitmentWriter::new(params.clone());
        let witness = committer.commit(&mut prover_state, &polynomial);

        // Ensure Merkle leaves are correctly generated.
        assert!(
            !witness.witness.matrix.is_empty(),
            "Merkle leaves should not be empty"
        );

        // Ensure OOD (out-of-domain) points are generated.
        assert!(
            !witness.witness.out_of_domain.points.is_empty(),
            "OOD points should be generated"
        );

        // Validate the number of generated OOD points.
        assert_eq!(
            witness.witness.out_of_domain.points.len(),
            params.committment_ood_samples,
            "OOD points count should match expected samples"
        );

        // Check that the matrix commitment is valid
        assert_eq!(
            witness.witness.matrix_witness.num_nodes(),
            params
                .initial_committer
                .matrix_commit
                .merkle_tree
                .num_nodes()
        );

        // Ensure polynomial data is correctly stored
        assert_eq!(
            witness.polynomial.coeffs().len(),
            polynomial.coeffs().len(),
            "Stored polynomial should have the correct number of coefficients"
        );

        // Check that OOD answers match expected evaluations
        for (i, ood_point) in witness.witness.out_of_domain.points.iter().enumerate() {
            let expected_eval = polynomial.evaluate_at_extension(
                &MultilinearPoint::expand_from_univariate(*ood_point, num_variables),
            );
            assert_eq!(
                witness.witness.out_of_domain.matrix[i], expected_eval,
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
        let ds = DomainSeparator::protocol(&module_path!())
            .session(&format!("Test at {}:{}", file!(), line!()))
            .instance(&Empty);
        let mut prover_state = ProverState::new_std(&ds);

        let committer = CommitmentWriter::new(params);
        let witness = committer.commit(&mut prover_state, &polynomial);

        // Expansion factor is 2
        assert_eq!(
            witness.witness.matrix.len(),
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

        params.initial_committer.out_domain_samples = 0; // No OOD samples

        let polynomial = CoefficientList::new(vec![F::rand(&mut rng); 32]);
        let ds = DomainSeparator::protocol(&module_path!())
            .session(&format!("Test at {}:{}", file!(), line!()))
            .instance(&Empty);
        let mut prover_state = ProverState::new_std(&ds);

        let committer = CommitmentWriter::new(params);
        let witness = committer.commit(&mut prover_state, &polynomial);

        assert!(
            witness.witness.out_of_domain.points.is_empty(),
            "There should be no OOD points when committment_ood_samples is 0"
        );
    }
}
