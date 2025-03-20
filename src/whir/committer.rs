use super::parameters::WhirConfig;
use crate::whir::fs_utils::DigestWriter;
use crate::{
    ntt::expand_from_coeff, poly_utils::coeffs::CoefficientList,
    poly_utils::fold::transform_evaluations, utils::sample_ood_points,
};
use ark_crypto_primitives::merkle_tree::{Config, MerkleTree};
use ark_ff::FftField;
use ark_poly::EvaluationDomain;
use nimue::{
    plugins::ark::{FieldChallenges, FieldWriter},
    ByteWriter, ProofResult,
};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Represents the commitment and evaluation data for a polynomial.
///
/// This structure holds all necessary components to verify a commitment,
/// including the polynomial itself, the Merkle tree used for commitment,
/// and out-of-domain (OOD) evaluations.
pub struct Witness<F, MerkleConfig>
where
    MerkleConfig: Config,
{
    /// The committed polynomial in coefficient form.
    pub(crate) polynomial: CoefficientList<F>,
    /// The Merkle tree constructed from the polynomial evaluations.
    pub(crate) merkle_tree: MerkleTree<MerkleConfig>,
    /// The leaves of the Merkle tree, derived from folded polynomial evaluations.
    pub(crate) merkle_leaves: Vec<F>,
    /// Out-of-domain challenge points used for polynomial verification.
    pub(crate) ood_points: Vec<F>,
    /// The corresponding polynomial evaluations at the OOD challenge points.
    pub(crate) ood_answers: Vec<F>,
}

/// Responsible for committing polynomials using a Merkle-based scheme.
///
/// The `Committer` processes a polynomial, expands and folds its evaluations,
/// and constructs a Merkle tree from the resulting values.
///
/// It provides a commitment that can be used for proof generation and verification.
pub struct Committer<F, MerkleConfig, PowStrategy>(WhirConfig<F, MerkleConfig, PowStrategy>)
where
    F: FftField,
    MerkleConfig: Config;

impl<F, MerkleConfig, PowStrategy> Committer<F, MerkleConfig, PowStrategy>
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
    /// - Computes out-of-domain (OOD) challenge points and their evaluations.
    /// - Returns a `Witness` containing the commitment data.
    pub fn commit<Merlin>(
        &self,
        merlin: &mut Merlin,
        polynomial: CoefficientList<F::BasePrimeField>,
    ) -> ProofResult<Witness<F, MerkleConfig>>
    where
        Merlin: FieldWriter<F> + FieldChallenges<F> + ByteWriter + DigestWriter<MerkleConfig>,
    {
        // Retrieve the base domain, ensuring it is set.
        let base_domain = self.0.starting_domain.base_domain.unwrap();

        // Compute expansion factor based on the domain size and polynomial length.
        let expansion = base_domain.size() / polynomial.num_coeffs();

        // Expand the polynomial coefficients into evaluations over the extended domain.
        let mut evals = expand_from_coeff(polynomial.coeffs(), expansion);
        transform_evaluations(
            &mut evals,
            self.0.fold_optimisation,
            base_domain.group_gen(),
            base_domain.group_gen_inv(),
            self.0.folding_factor.at_round(0),
        );

        // Convert to extension field.
        // This is not necessary for the commit, but in further rounds
        // we will need the extension field. For symplicity we do it here too.
        // TODO: Commit to base field directly.
        let folded_evals = evals
            .into_iter()
            .map(F::from_base_prime_field)
            .collect::<Vec<_>>();

        // Determine leaf size based on folding factor.
        let fold_size = 1 << self.0.folding_factor.at_round(0);

        // Chunk evaluations into leaves for Merkle tree construction.
        #[cfg(not(feature = "parallel"))]
        let leafs_iter = folded_evals.chunks_exact(fold_size);
        #[cfg(feature = "parallel")]
        let leafs_iter = folded_evals.par_chunks_exact(fold_size);

        // Construct the Merkle tree with given hash parameters.
        let merkle_tree = MerkleTree::<MerkleConfig>::new(
            &self.0.leaf_hash_params,
            &self.0.two_to_one_params,
            leafs_iter,
        )
        .unwrap();

        // Retrieve the Merkle tree root and add it to the transcript.
        let root = merkle_tree.root();
        merlin.add_digest(root)?;

        // Handle OOD (Out-Of-Domain) samples
        let (ood_points, ood_answers) = sample_ood_points(
            merlin,
            self.0.committment_ood_samples,
            self.0.mv_parameters.num_variables,
            |point| polynomial.evaluate_at_extension(point),
        )?;

        // Return the witness containing the polynomial, Merkle tree, and OOD results.
        Ok(Witness {
            polynomial: polynomial.to_extension(),
            merkle_tree,
            merkle_leaves: folded_evals,
            ood_points,
            ood_answers,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto::fields::Field64;
    use crate::crypto::merkle_tree::keccak;
    use crate::parameters::{
        FoldType, FoldingFactor, MultivariateParameters, SoundnessType, WhirParameters,
    };
    use crate::poly_utils::multilinear::MultilinearPoint;
    use crate::whir::iopattern::WhirIOPattern;
    use ark_ff::UniformRand;
    use nimue::{DefaultHash, IOPattern};
    use nimue_pow::blake3::Blake3PoW;

    #[test]
    fn test_basic_commitment() {
        // Define the field type and Merkle tree configuration.
        type F = Field64;
        type MerkleConfig = keccak::MerkleTreeParams<F>;

        let mut rng = ark_std::test_rng();

        // Generate Merkle tree hash parameters
        let (leaf_hash_params, two_to_one_params) = keccak::default_config::<F>(&mut rng);

        // Set up Whir protocol parameters.
        let security_level = 100;
        let pow_bits = 20;
        let num_variables = 5;
        let starting_rate = 1;
        let folding_factor = 4;
        let first_round_folding_factor = 4;

        let whir_params = WhirParameters::<MerkleConfig, Blake3PoW> {
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
        };

        // Define multivariate parameters for the polynomial.
        let mv_params = MultivariateParameters::<F>::new(num_variables);
        let params = WhirConfig::<F, MerkleConfig, Blake3PoW>::new(mv_params, whir_params);

        // Generate a random polynomial with 32 coefficients.
        let polynomial = CoefficientList::new(vec![F::rand(&mut rng); 32]);

        // Set up the IOPattern and initialize a Merlin transcript.
        let io = IOPattern::<DefaultHash>::new("🌪️")
            .commit_statement(&params)
            .add_whir_proof(&params);
        let mut merlin = io.to_merlin();

        // Run the Commitment Phase
        let committer = Committer::new(params.clone());
        let witness = committer.commit(&mut merlin, polynomial.clone()).unwrap();

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
        type MerkleConfig = keccak::MerkleTreeParams<F>;

        let mut rng = ark_std::test_rng();
        let (leaf_hash_params, two_to_one_params) = keccak::default_config::<F>(&mut rng);

        let params = WhirConfig::<F, MerkleConfig, Blake3PoW>::new(
            MultivariateParameters::<F>::new(10),
            WhirParameters {
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
            },
        );

        let polynomial = CoefficientList::new(vec![F::rand(&mut rng); 1024]); // Large polynomial
        let io = IOPattern::<DefaultHash>::new("🌪️").commit_statement(&params);
        let mut merlin = io.to_merlin();

        let committer = Committer::new(params);
        let witness = committer.commit(&mut merlin, polynomial).unwrap();

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
        type MerkleConfig = keccak::MerkleTreeParams<F>;

        let mut rng = ark_std::test_rng();
        let (leaf_hash_params, two_to_one_params) = keccak::default_config::<F>(&mut rng);

        let mut params = WhirConfig::<F, MerkleConfig, Blake3PoW>::new(
            MultivariateParameters::<F>::new(5),
            WhirParameters {
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
            },
        );

        params.committment_ood_samples = 0; // No OOD samples

        let polynomial = CoefficientList::new(vec![F::rand(&mut rng); 32]);
        let io = IOPattern::<DefaultHash>::new("🌪️").commit_statement(&params);
        let mut merlin = io.to_merlin();

        let committer = Committer::new(params);
        let witness = committer.commit(&mut merlin, polynomial).unwrap();

        assert!(
            witness.ood_points.is_empty(),
            "There should be no OOD points when committment_ood_samples is 0"
        );
    }
}
