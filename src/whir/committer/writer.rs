use ark_crypto_primitives::merkle_tree::{Config, MerkleTree};
use ark_ff::FftField;
use ark_poly::EvaluationDomain;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use spongefish::{
    codecs::arkworks_algebra::{BytesToUnitSerialize, FieldToUnitSerialize, UnitToField},
    ProofResult,
};
#[cfg(feature = "tracing")]
use tracing::{instrument, span, Level};

use super::Witness;
use crate::{
    poly_utils::coeffs::CoefficientList,
    whir::{
        parameters::WhirConfig,
        utils::{compute_ood_response, sample_ood_points, DigestToUnitSerialize},
    },
};

/// Responsible for committing polynomials using a Merkle-based scheme.
///
/// The `Committer` processes a polynomial, expands and folds its evaluations,
/// and constructs a Merkle tree from the resulting values.
///
/// It provides a commitment that can be used for proof generation and verification.
pub struct CommitmentWriter<F, MerkleConfig, PowStrategy>
where
    F: FftField,
    MerkleConfig: Config,
{
    pub(crate) config: WhirConfig<F, MerkleConfig, PowStrategy>,
}

impl<F, MerkleConfig, PowStrategy> CommitmentWriter<F, MerkleConfig, PowStrategy>
where
    F: FftField,
    MerkleConfig: Config<Leaf = [F]>,
{
    pub const fn new(config: WhirConfig<F, MerkleConfig, PowStrategy>) -> Self {
        Self { config }
    }

    #[allow(clippy::too_many_lines)]
    #[cfg_attr(feature = "tracing", instrument(skip_all, fields(size = polynomials.first().unwrap().num_coeffs())))]
    pub fn commit_batch<ProverState>(
        &self,
        prover_state: &mut ProverState,
        polynomials: &[&CoefficientList<F::BasePrimeField>],
    ) -> ProofResult<Witness<F, MerkleConfig>>
    where
        ProverState: FieldToUnitSerialize<F>
            + BytesToUnitSerialize
            + DigestToUnitSerialize<MerkleConfig>
            + UnitToField<F>,
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

        #[cfg(not(feature = "parallel"))]
        let leafs_iter = stacked_leaves.chunks_exact(stacked_leaf_size);
        #[cfg(feature = "parallel")]
        let leafs_iter = stacked_leaves.par_chunks_exact(stacked_leaf_size);

        let merkle_tree = {
            #[cfg(feature = "tracing")]
            let _span = span!(Level::INFO, "MerkleTree::new", size = leafs_iter.len()).entered();
            MerkleTree::<MerkleConfig>::new(
                &self.config.leaf_hash_params,
                &self.config.two_to_one_params,
                leafs_iter,
            )
            .unwrap()
        };

        let root = merkle_tree.root();
        prover_state.add_digest(root)?;

        let (ood_points, first_answers) = sample_ood_points(
            prover_state,
            self.config.committment_ood_samples,
            self.config.mv_parameters.num_variables,
            |point| polynomials[0].evaluate_at_extension(point),
        )?;
        let mut per_poly_ood_answers: Vec<Vec<F>> = Vec::with_capacity(polynomials.len());
        per_poly_ood_answers.push(first_answers);
        for poly in polynomials.iter().skip(1) {
            let answers = compute_ood_response(
                prover_state,
                &ood_points,
                self.config.mv_parameters.num_variables,
                |point| poly.evaluate_at_extension(point),
            )?;
            per_poly_ood_answers.push(answers);
        }

        let [batching_randomness] = if polynomials.len() > 1 {
            prover_state.challenge_scalars()?
        } else {
            [F::zero()]
        };

        // Memory optimization: For batch_size=1, we can regenerate leaves from the polynomial.
        // For batch_size>1, the leaves are interleaved from multiple polynomials and cannot
        // be regenerated from the batched polynomial, so we must keep them.
        if polynomials.len() == 1 {
            drop(stacked_leaves);
            return Ok(Witness {
                polynomial: polynomials[0].clone().to_extension(),
                merkle_tree,
                merkle_leaves: Vec::new(),
                ood_points,
                ood_answers: per_poly_ood_answers.remove(0),
                batching_randomness,
            });
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

        Ok(Witness {
            polynomial,
            merkle_tree,
            merkle_leaves: stacked_leaves,
            ood_points,
            ood_answers: batched_ood_resp,
            batching_randomness,
        })
    }

    #[cfg_attr(feature = "tracing", instrument(skip_all, fields(size = polynomial.num_coeffs())))]
    pub fn commit<ProverState>(
        &self,
        prover_state: &mut ProverState,
        polynomial: &CoefficientList<F::BasePrimeField>,
    ) -> ProofResult<Witness<F, MerkleConfig>>
    where
        ProverState: FieldToUnitSerialize<F>
            + UnitToField<F>
            + DigestToUnitSerialize<MerkleConfig>
            + BytesToUnitSerialize,
    {
        self.commit_batch(prover_state, &[polynomial])
    }

    pub fn regenerate_merkle_leaves(
        &self,
        polynomials: &[&CoefficientList<F::BasePrimeField>],
    ) -> Vec<F> {
        assert!(!polynomials.is_empty());
        assert_eq!(polynomials.len(), self.config.batch_size);

        let num_coeffs = polynomials.first().unwrap().num_coeffs();
        let base_domain = self.config.starting_domain.base_domain.unwrap();
        let expansion = base_domain.size() / num_coeffs;
        let fold_size = 1 << self.config.folding_factor.at_round(0);

        let num_leaves = (num_coeffs * expansion) / fold_size;
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

        stacked_leaves
    }

    /// Regenerates merkle leaves from a batched polynomial (already in extension field).
    /// Use this to restore leaves after calling `Witness::clear_merkle_leaves()`.
    pub fn regenerate_merkle_leaves_from_batched(&self, polynomial: &CoefficientList<F>) -> Vec<F> {
        let num_coeffs = polynomial.num_coeffs();
        let base_domain = self.config.starting_domain.base_domain.unwrap();
        let expansion = base_domain.size() / num_coeffs;
        let fold_size = 1 << self.config.folding_factor.at_round(0);

        let num_leaves = (num_coeffs * expansion) / fold_size;
        let stacked_leaf_size = fold_size * self.config.batch_size;
        let mut stacked_leaves = vec![F::zero(); num_leaves * stacked_leaf_size];

        let evals = self.config.reed_solomon.interleaved_encode(
            polynomial.coeffs(),
            expansion,
            self.config.folding_factor.at_round(0),
        );

        for (i, chunk) in evals.chunks_exact(fold_size).enumerate() {
            let start_dst = i * stacked_leaf_size;
            for (j, &eval) in chunk.iter().enumerate() {
                stacked_leaves[start_dst + j] = eval;
            }
        }

        stacked_leaves
    }
}

impl<F, MerkleConfig> Witness<F, MerkleConfig>
where
    F: FftField,
    MerkleConfig: Config<Leaf = [F]>,
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
    use spongefish::DomainSeparator;
    use spongefish_pow::blake3::Blake3PoW;

    use super::*;
    use crate::{
        crypto::{
            fields::Field64,
            merkle_tree::{
                keccak::{KeccakCompress, KeccakLeafHash, KeccakMerkleTreeParams},
                parameters::default_config,
            },
        },
        ntt::RSDefault,
        parameters::{
            DeduplicationStrategy, FoldingFactor, MerkleProofStrategy, MultivariateParameters,
            ProtocolParameters, SoundnessType,
        },
        poly_utils::multilinear::MultilinearPoint,
        whir::domainsep::WhirDomainSeparator,
    };

    #[test]
    fn test_basic_commitment() {
        // Define the field type and Merkle tree configuration.
        type F = Field64;
        type MerkleConfig = KeccakMerkleTreeParams<F>;

        let mut rng = ark_std::test_rng();

        // Generate Merkle tree hash parameters
        let (leaf_hash_params, two_to_one_params) =
            default_config::<F, KeccakLeafHash<F>, KeccakCompress>(&mut rng);

        // Set up Whir protocol parameters.
        let security_level = 100;
        let pow_bits = 20;
        let num_variables = 5;
        let starting_rate = 1;
        let folding_factor = 4;
        let first_round_folding_factor = 4;

        let whir_params = ProtocolParameters::<MerkleConfig, Blake3PoW> {
            initial_statement: true,
            security_level,
            pow_bits,
            folding_factor: FoldingFactor::ConstantFromSecondRound(
                first_round_folding_factor,
                folding_factor,
            ),
            leaf_hash_params,
            two_to_one_params,
            soundness_type: SoundnessType::ConjectureList,
            _pow_parameters: std::marker::PhantomData,
            starting_log_inv_rate: starting_rate,
            batch_size: 1,
            deduplication_strategy: DeduplicationStrategy::Enabled,
            merkle_proof_strategy: MerkleProofStrategy::Compressed,
        };

        // Define multivariate parameters for the polynomial.
        let mv_params = MultivariateParameters::<F>::new(num_variables);
        let reed_solomon = Arc::new(RSDefault);
        let basefield_reed_solomon = reed_solomon.clone();
        let params = WhirConfig::<F, MerkleConfig, Blake3PoW>::new(
            reed_solomon,
            basefield_reed_solomon,
            mv_params,
            whir_params,
        );

        // Generate a random polynomial with 32 coefficients.
        let polynomial = CoefficientList::new(vec![F::rand(&mut rng); 32]);

        // Set up the DomainSeparator and initialize a ProverState narg_string.
        let domainsep = DomainSeparator::new("🌪️")
            .commit_statement(&params)
            .add_whir_proof(&params);
        let mut prover_state = domainsep.to_prover_state();

        // Run the Commitment Phase
        let committer = CommitmentWriter::new(params.clone());
        let witness = committer.commit(&mut prover_state, &polynomial).unwrap();

        assert!(
            witness.merkle_leaves.is_empty(),
            "Merkle leaves should be empty (regenerated on-demand for memory optimization)"
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

        // Check that the Merkle tree root is valid
        let root = witness.merkle_tree.root();
        assert_ne!(
            root.as_ref(),
            &[0u8; 32],
            "Merkle tree root should not be zero"
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
        type MerkleConfig = KeccakMerkleTreeParams<F>;

        let mut rng = ark_std::test_rng();
        let (leaf_hash_params, two_to_one_params) =
            default_config::<F, KeccakLeafHash<F>, KeccakCompress>(&mut rng);

        let reed_solomon = Arc::new(RSDefault);
        let basefield_reed_solomon = reed_solomon.clone();
        let params = WhirConfig::<F, MerkleConfig, Blake3PoW>::new(
            reed_solomon,
            basefield_reed_solomon,
            MultivariateParameters::<F>::new(10),
            ProtocolParameters {
                initial_statement: true,
                security_level: 100,
                pow_bits: 20,
                folding_factor: FoldingFactor::ConstantFromSecondRound(4, 4),
                leaf_hash_params,
                two_to_one_params,
                soundness_type: SoundnessType::ConjectureList,
                _pow_parameters: Default::default(),
                starting_log_inv_rate: 1,
                batch_size: 1,
                deduplication_strategy: DeduplicationStrategy::Enabled,
                merkle_proof_strategy: MerkleProofStrategy::Compressed,
            },
        );

        let polynomial = CoefficientList::new(vec![F::rand(&mut rng); 1024]); // Large polynomial
        let domainsep = DomainSeparator::new("🌪️").commit_statement(&params);
        let mut prover_state = domainsep.to_prover_state();

        let committer = CommitmentWriter::new(params);
        let witness = committer.commit(&mut prover_state, &polynomial).unwrap();

        assert!(
            witness.merkle_leaves.is_empty(),
            "Merkle leaves should be empty (regenerated on-demand)"
        );
    }

    #[test]
    fn test_commitment_without_ood_samples() {
        type F = Field64;
        type MerkleConfig = KeccakMerkleTreeParams<F>;

        let mut rng = ark_std::test_rng();
        let (leaf_hash_params, two_to_one_params) =
            default_config::<F, KeccakLeafHash<F>, KeccakCompress>(&mut rng);
        let reed_solomon = Arc::new(RSDefault);
        let basefield_reed_solomon = reed_solomon.clone();
        let mut params = WhirConfig::<F, MerkleConfig, Blake3PoW>::new(
            reed_solomon,
            basefield_reed_solomon,
            MultivariateParameters::<F>::new(5),
            ProtocolParameters {
                initial_statement: true,
                security_level: 100,
                pow_bits: 20,
                folding_factor: FoldingFactor::ConstantFromSecondRound(4, 4),
                leaf_hash_params,
                two_to_one_params,
                soundness_type: SoundnessType::ConjectureList,
                _pow_parameters: Default::default(),
                starting_log_inv_rate: 1,
                batch_size: 1,
                deduplication_strategy: DeduplicationStrategy::Enabled,
                merkle_proof_strategy: MerkleProofStrategy::Compressed,
            },
        );

        params.committment_ood_samples = 0; // No OOD samples

        let polynomial = CoefficientList::new(vec![F::rand(&mut rng); 32]);
        let domainsep = DomainSeparator::new("🌪️").commit_statement(&params);
        let mut prover_state = domainsep.to_prover_state();

        let committer = CommitmentWriter::new(params);
        let witness = committer.commit(&mut prover_state, &polynomial).unwrap();

        assert!(
            witness.ood_points.is_empty(),
            "There should be no OOD points when committment_ood_samples is 0"
        );
    }
}
