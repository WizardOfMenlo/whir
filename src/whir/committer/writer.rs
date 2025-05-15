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
    ntt::expand_from_coeff,
    poly_utils::{
        coeffs::CoefficientList, fold::transform_evaluations, multilinear::MultilinearPoint,
    },
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
pub struct CommitmentWriter<F, MerkleConfig, PowStrategy>(
    pub(crate) WhirConfig<F, MerkleConfig, PowStrategy>,
)
where
    F: FftField,
    MerkleConfig: Config;

impl<F, MerkleConfig, PowStrategy> CommitmentWriter<F, MerkleConfig, PowStrategy>
where
    F: FftField,
    MerkleConfig: Config<Leaf = [F]>,
{
    pub const fn new(config: WhirConfig<F, MerkleConfig, PowStrategy>) -> Self {
        Self(config)
    }

    /// Commits a polynomial using a Merkle-based commitment scheme.
    ///
    /// This function:
    /// - Expands polynomial coefficients to evaluations.
    /// - Applies folding and restructuring optimizations.
    /// - Converts evaluations to an extension field.
    /// - Constructs a Merkle tree from the evaluations.
    /// - Returns a `Witness` containing the commitment data.
    pub fn commit<ProverState>(
        &self,
        prover_state: &mut ProverState,
        polynomial: &CoefficientList<F::BasePrimeField>,
    ) -> ProofResult<Witness<F, MerkleConfig>>
    where
        ProverState: FieldToUnitSerialize<F> + UnitToField<F> + DigestToUnitSerialize<MerkleConfig>,
    {
        self.commit_many(prover_state, &[polynomial])
    }

    /// Commits multiple polynomials.
    ///
    /// This function:
    /// - Expands polynomial coefficients to evaluations.
    /// - Applies folding and restructuring optimizations.
    /// - Converts evaluations to an extension field.
    /// - Constructs a Merkle tree from the evaluations.
    /// - Returns a `Witness` containing the commitment data.
    #[cfg_attr(feature = "tracing", instrument(skip_all, fields(
        batch_size = self.0.batch_size,
        poly_size = self.0.mv_parameters.num_variables
    )))]
    pub fn commit_many<ProverState>(
        &self,
        prover_state: &mut ProverState,
        polynomials: &[&CoefficientList<F::BasePrimeField>],
    ) -> ProofResult<Witness<F, MerkleConfig>>
    where
        ProverState: FieldToUnitSerialize<F> + UnitToField<F> + DigestToUnitSerialize<MerkleConfig>,
    {
        // Get configuration
        let batch_size = self.0.batch_size;
        let num_variables = self.0.mv_parameters.num_variables;
        let num_coeff = 1 << num_variables;
        let base_domain = self.0.starting_domain.base_domain.unwrap();
        let expansion = base_domain.size() / num_coeff;
        let coset_size = 1 << self.0.folding_factor.at_round(0);
        let leaf_size = batch_size * coset_size;
        let ood_samples = self.0.committment_ood_samples;

        // Validate input
        assert_eq!(polynomials.len(), batch_size);
        for poly in polynomials {
            assert_eq!(poly.num_variables(), num_variables);
        }

        // Compute the leaf element values
        let mut merkle_leaves = vec![F::ZERO; batch_size * num_coeff * expansion];
        for (i, polynomial) in polynomials.iter().enumerate() {
            // Expand the polynomial coefficients into evaluations over the extended domain.
            let mut evals = expand_from_coeff(polynomial.coeffs(), expansion);

            // Transform cosets to coefficient form
            transform_evaluations(
                &mut evals,
                self.0.fold_optimisation,
                base_domain.group_gen(),
                base_domain.group_gen_inv(),
                self.0.folding_factor.at_round(0),
            );

            // Convert to extension field.
            // This is not necessary for the commit, but in further rounds
            // we will need the extension field. For simplicity we do it here too.
            // TODO: Commit to base field directly.
            let folded_evals = {
                #[cfg(feature = "tracing")]
                let _span = span!(Level::INFO, "evals_to_extension", size = evals.len());
                evals
                    .into_iter()
                    .map(F::from_base_prime_field)
                    .collect::<Vec<_>>()
            };

            // Write leaves to the correct locations in the final vector
            for (src, dst) in folded_evals
                .chunks_exact(coset_size)
                .zip(merkle_leaves.chunks_exact_mut(leaf_size))
            {
                dst[i * coset_size..(i + 1) * coset_size].copy_from_slice(src);
            }
        }

        // Chunk evaluations into leaves for Merkle tree construction.
        #[cfg(not(feature = "parallel"))]
        let leafs_iter = merkle_leaves.chunks_exact(leaf_size);
        #[cfg(feature = "parallel")]
        let leafs_iter = merkle_leaves.par_chunks_exact(leaf_size);

        // Construct a single Merkle tree with given hash parameters.
        let merkle_tree = {
            #[cfg(feature = "tracing")]
            let _span = span!(Level::INFO, "MerkleTree::new", size = leafs_iter.len()).entered();
            MerkleTree::<MerkleConfig>::new(
                &self.0.leaf_hash_params,
                &self.0.two_to_one_params,
                leafs_iter,
            )
            .unwrap()
        };

        // Retrieve the Merkle tree root and add it to the transcript.
        let root = merkle_tree.root();
        prover_state.add_digest(root)?;

        // Handle OOD (Out-Of-Domain) samples
        let (ood_points, ood_answers) = if ood_samples > 0 {
            // Create challenge points
            let mut ood_points = vec![F::zero(); ood_samples];
            prover_state.fill_challenge_scalars(&mut ood_points)?;

            // Compute OOD answers for each polynomial
            let mut ood_answers = Vec::with_capacity(ood_samples * batch_size);
            for ood_point in &ood_points {
                let extension = MultilinearPoint::expand_from_univariate(*ood_point, num_variables);
                for polynomial in polynomials {
                    let answer = polynomial.evaluate_at_extension(&extension);
                    ood_answers.push(answer);
                }
            }
            prover_state.add_scalars(&ood_answers)?;
            (ood_points, ood_answers)
        } else {
            (vec![], vec![])
        };

        // Return the witness containing the polynomial, Merkle tree, and OOD results.
        Ok(Witness {
            polynomials: polynomials
                .iter()
                .map(|poly| (*poly).clone().to_extension())
                .collect(),
            merkle_tree,
            merkle_leaves,
            ood_points,
            ood_answers,
        })
    }
}

#[cfg(test)]
mod tests {
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
        parameters::{
            FoldType, FoldingFactor, MultivariateParameters, ProtocolParameters, SoundnessType,
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
            fold_optimisation: FoldType::ProverHelps,
            _pow_parameters: std::marker::PhantomData,
            starting_log_inv_rate: starting_rate,
            batch_size: 1,
        };

        // Define multivariate parameters for the polynomial.
        let mv_params = MultivariateParameters::<F>::new(num_variables);
        let params = WhirConfig::<F, MerkleConfig, Blake3PoW>::new(mv_params, whir_params);

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

        // Check that the Merkle tree root is valid
        let root = witness.merkle_tree.root();
        assert_ne!(
            root.as_ref(),
            &[0u8; 32],
            "Merkle tree root should not be zero"
        );

        // Ensure polynomial data is correctly stored
        assert_eq!(
            witness.polynomials[0].coeffs().len(),
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

        let params = WhirConfig::<F, MerkleConfig, Blake3PoW>::new(
            MultivariateParameters::<F>::new(10),
            ProtocolParameters {
                initial_statement: true,
                security_level: 100,
                pow_bits: 20,
                folding_factor: FoldingFactor::ConstantFromSecondRound(4, 4),
                leaf_hash_params,
                two_to_one_params,
                soundness_type: SoundnessType::ConjectureList,
                fold_optimisation: FoldType::ProverHelps,
                _pow_parameters: Default::default(),
                starting_log_inv_rate: 1,
                batch_size: 1,
            },
        );

        let polynomial = CoefficientList::new(vec![F::rand(&mut rng); 1024]); // Large polynomial
        let domainsep = DomainSeparator::new("🌪️").commit_statement(&params);
        let mut prover_state = domainsep.to_prover_state();

        let committer = CommitmentWriter::new(params);
        let witness = committer.commit(&mut prover_state, &polynomial).unwrap();

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
        type MerkleConfig = KeccakMerkleTreeParams<F>;

        let mut rng = ark_std::test_rng();
        let (leaf_hash_params, two_to_one_params) =
            default_config::<F, KeccakLeafHash<F>, KeccakCompress>(&mut rng);

        let mut params = WhirConfig::<F, MerkleConfig, Blake3PoW>::new(
            MultivariateParameters::<F>::new(5),
            ProtocolParameters {
                initial_statement: true,
                security_level: 100,
                pow_bits: 20,
                folding_factor: FoldingFactor::ConstantFromSecondRound(4, 4),
                leaf_hash_params,
                two_to_one_params,
                soundness_type: SoundnessType::ConjectureList,
                fold_optimisation: FoldType::ProverHelps,
                _pow_parameters: Default::default(),
                starting_log_inv_rate: 1,
                batch_size: 1,
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
