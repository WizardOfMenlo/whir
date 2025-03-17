use super::parameters::WhirConfig;
use crate::whir::fs_utils::DigestWriter;
use crate::{
    ntt::expand_from_coeff,
    poly_utils::{
        coeffs::CoefficientList, fold::restructure_evaluations, multilinear::MultilinearPoint,
    },
    utils,
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
        let evals = expand_from_coeff(polynomial.coeffs(), expansion);
        // TODO: `stack_evaluations` and `restructure_evaluations` are really in-place algorithms.
        // They also partially overlap and undo one another. We should merge them.
        let folded_evals = utils::stack_evaluations(evals, self.0.folding_factor.at_round(0));
        let folded_evals = restructure_evaluations(
            folded_evals,
            self.0.fold_optimisation,
            base_domain.group_gen(),
            base_domain.group_gen_inv(),
            self.0.folding_factor.at_round(0),
        );

        // Convert evaluations into the extension field, required for later rounds.
        // TODO: Commit to base field directly in the future.
        let folded_evals = folded_evals
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

        // Initialize out-of-domain (OOD) challenge points and evaluations.
        let mut ood_points = vec![F::ZERO; self.0.committment_ood_samples];
        let mut ood_answers = Vec::with_capacity(self.0.committment_ood_samples);

        // Generate OOD points and compute their evaluations.
        if self.0.committment_ood_samples > 0 {
            merlin.fill_challenge_scalars(&mut ood_points)?;
            ood_answers.extend(ood_points.iter().map(|ood_point| {
                polynomial.evaluate_at_extension(&MultilinearPoint::expand_from_univariate(
                    *ood_point,
                    self.0.mv_parameters.num_variables,
                ))
            }));
            merlin.add_scalars(&ood_answers)?;
        }

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
    use crate::whir::iopattern::WhirIOPattern;
    use ark_ff::UniformRand;
    use nimue::{DefaultHash, IOPattern};
    use nimue_pow::blake3::Blake3PoW;

    #[test]
    fn test_basic_commitment() {
        type F = Field64;
        type MerkleConfig = keccak::MerkleTreeParams<F>;

        let mut rng = ark_std::test_rng();

        // Generate Merkle tree hash parameters
        let (leaf_hash_params, two_to_one_params) = keccak::default_config::<F>(&mut rng);

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

        let mv_params = MultivariateParameters::<F>::new(num_variables);
        let params = WhirConfig::<F, MerkleConfig, Blake3PoW>::new(mv_params, whir_params);

        let polynomial = CoefficientList::new(vec![F::rand(&mut rng); 32]);

        let io = IOPattern::<DefaultHash>::new("🌪️")
            .commit_statement(&params)
            .add_whir_proof(&params);

        let mut merlin = io.to_merlin();

        // Step 1: Run the Commitment Phase
        let committer = Committer::new(params.clone());
        let witness = committer.commit(&mut merlin, polynomial.clone()).unwrap();

        // Assertions to validate the commitment
        assert!(
            !witness.merkle_leaves.is_empty(),
            "Merkle leaves should not be empty"
        );
        assert!(
            !witness.ood_points.is_empty(),
            "OOD points should be generated"
        );
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
}
