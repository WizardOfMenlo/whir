use ark_crypto_primitives::merkle_tree::{Config, LeafParam, MultiPath, TwoToOneParam};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

pub mod batching;
pub mod committer;
pub mod domainsep;
pub mod parameters;
pub mod parsed_proof;
pub mod prover;
pub mod statement;
pub mod stir_evaluations;
pub mod utils;
pub mod verifier;

///
/// Trait to abstract away committed data access by the verifier. Parsed
/// commitment data should implement this trait.
///
pub(crate) trait WhirCommitmentData<F, M>
where
    M: Config,
{
    /// Returns the committed out-of-domain points and answers
    fn ood_data(&self) -> (&[F], &[F]);

    ///
    /// returns the committed root in case of single root otherwise returns some
    /// dummy value. Verifier should use [RoundZeroProofValidator] trait to
    /// validate committed Merkle roots.
    ///
    fn committed_root(&self) -> &<M as Config>::InnerDigest;

    fn batching_randomness(&self) -> Option<F>;
}

///
/// Trait to validate the zero-th round (i.e., the round whose Merkle roots are
/// part of the commitment). [WhirProof] and [BatchedWhirProof] types implement
/// this trait.
///
pub(crate) trait RoundZeroProofValidator<C> {
    ///
    /// Merkle Tree configuration that was used to compute the hashes of leaf
    /// and inner nodes
    ///
    type MerkleConfig: Config;

    ///
    /// Given a commitment this method validates the first round of Merkle path
    /// against that committed data and checks that the stir indices match the
    /// indices in the tree.
    ///
    fn validate_first_round(
        &self,
        commitment: &C,
        stir_challenges_indexes: &[usize],
        leaf_hash: &LeafParam<Self::MerkleConfig>,
        inner_node_hash: &TwoToOneParam<Self::MerkleConfig>,
    ) -> bool;
}

///
/// Trait to abstract away access to Round by round proof data. If the zero-th
/// round of proof has multiple roots (as in case of batched proof), the
/// MultiPath returned is the multipath corresponding to first Merkle root (this
/// is useful for getting the indexes of STIR queries). For validating the
/// zero-th  round always use the methods of the trait [RoundZeroProofValidator]
/// which will ensure that each root in the Merkle tree has the same set of stir
/// query indices.
///
pub(crate) trait WhirProofRoundData<F, MerkleConfig>
where
    MerkleConfig: Config<Leaf = [F]>,
    F: Sized + Clone + CanonicalSerialize + CanonicalDeserialize,
{
    ///
    /// Returns the path to the root of the Merkle tree and the corresponding
    /// Leaf node values. If the WhirProof has multiple roots, then the
    /// MultiPath returned in is a dummy and inconsistent value. The Leaf node
    /// value is the batched value of the lead node.
    ///
    fn round_data(
        &self,
        whir_round: usize,
        batching_randomness: Option<F>,
    ) -> (MultiPath<MerkleConfig>, Vec<Vec<F>>);

    fn statement_values(&self) -> &[F];
}

// Only includes the authentication paths
#[derive(Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct WhirProof<MerkleConfig, F>
where
    MerkleConfig: Config<Leaf = [F]>,
    F: Sized + Clone + CanonicalSerialize + CanonicalDeserialize,
{
    pub merkle_paths: Vec<(MultiPath<MerkleConfig>, Vec<Vec<F>>)>,
    pub statement_values_at_random_point: Vec<F>,
}

impl<F, MerkleConfig> WhirProofRoundData<F, MerkleConfig> for WhirProof<MerkleConfig, F>
where
    MerkleConfig: Config<Leaf = [F]>,
    F: Sized + Clone + CanonicalSerialize + CanonicalDeserialize,
{
    fn round_data(
        &self,
        whir_round: usize,
        _: Option<F>,
    ) -> (MultiPath<MerkleConfig>, Vec<Vec<F>>) {
        assert!(
            whir_round < self.merkle_paths.len(),
            "Whir round number must the less than the number of entries"
        );
        self.merkle_paths[whir_round].clone()
    }

    fn statement_values(&self) -> &[F] {
        &self.statement_values_at_random_point
    }
}

pub fn whir_proof_size<MerkleConfig, F>(
    narg_string: &[u8],
    whir_proof: &WhirProof<MerkleConfig, F>,
) -> usize
where
    MerkleConfig: Config<Leaf = [F]>,
    F: Sized + Clone + CanonicalSerialize + CanonicalDeserialize,
{
    narg_string.len() + whir_proof.serialized_size(ark_serialize::Compress::Yes)
}

#[cfg(test)]
mod tests {
    use ark_ff::Field;
    use spongefish::DomainSeparator;
    use spongefish_pow::blake3::Blake3PoW;

    use crate::{
        crypto::{
            fields::Field64,
            merkle_tree::{
                blake3::{Blake3Compress, Blake3LeafHash, Blake3MerkleTreeParams},
                parameters::default_config,
            },
        },
        parameters::{
            FoldType, FoldingFactor, MultivariateParameters, SoundnessType, WhirParameters,
        },
        poly_utils::{
            coeffs::CoefficientList, evals::EvaluationsList, multilinear::MultilinearPoint,
        },
        whir::{
            committer::{CommitmentReader, CommitmentWriter},
            domainsep::WhirDomainSeparator,
            parameters::WhirConfig,
            prover::Prover,
            statement::{Statement, StatementVerifier, Weights},
            verifier::Verifier,
        },
    };

    /// Merkle tree configuration type for commitment layers.
    type MerkleConfig = Blake3MerkleTreeParams<F>;
    /// PoW strategy used for grinding challenges in Fiat-Shamir transcript.
    type PowStrategy = Blake3PoW;
    /// Field type used in the tests.
    type F = Field64;

    /// Run a complete WHIR STARK proof lifecycle: commit, prove, and verify.
    ///
    /// This function:
    /// - builds a multilinear polynomial with a specified number of variables,
    /// - constructs a STARK statement with constraints based on evaluations and linear relations,
    /// - commits to the polynomial using a Merkle-based commitment scheme,
    /// - generates a proof using the WHIR prover,
    /// - verifies the proof using the WHIR verifier.
    fn make_whir_things(
        num_variables: usize,
        folding_factor: FoldingFactor,
        num_points: usize,
        soundness_type: SoundnessType,
        pow_bits: usize,
        fold_type: FoldType,
    ) {
        // Number of coefficients in the multilinear polynomial (2^num_variables)
        let num_coeffs = 1 << num_variables;

        // Randomness source
        let mut rng = ark_std::test_rng();
        // Generate Merkle parameters: hash function and compression function
        let (leaf_hash_params, two_to_one_params) =
            default_config::<F, Blake3LeafHash<F>, Blake3Compress>(&mut rng);

        // Configure multivariate polynomial parameters
        let mv_params = MultivariateParameters::new(num_variables);

        // Configure the WHIR protocol parameters
        let whir_params = WhirParameters::<MerkleConfig, PowStrategy> {
            initial_statement: true,
            security_level: 32,
            pow_bits,
            folding_factor,
            leaf_hash_params,
            two_to_one_params,
            soundness_type,
            _pow_parameters: Default::default(),
            starting_log_inv_rate: 1,
            fold_optimisation: fold_type,
            enable_batching: false,
        };

        // Build global configuration from multivariate + protocol parameters
        let params = WhirConfig::new(mv_params, whir_params);

        // Define the multilinear polynomial: constant 1 across all inputs
        let polynomial = CoefficientList::new(vec![F::ONE; num_coeffs]);

        // Generate `num_points` random points in the multilinear domain
        let points: Vec<_> = (0..num_points)
            .map(|_| MultilinearPoint::rand(&mut rng, num_variables))
            .collect();

        // Initialize a statement with no constraints yet
        let mut statement = Statement::new(num_variables);

        // For each random point, evaluate the polynomial and create a constraint
        for point in &points {
            let eval = polynomial.evaluate(point);
            let weights = Weights::evaluation(point.clone());
            statement.add_constraint(weights, eval);
        }

        // Construct a coefficient vector for linear sumcheck constraint
        let input = CoefficientList::new((0..1 << num_variables).map(F::from).collect());

        // Define weights for linear combination
        let linear_claim_weight = Weights::linear(input.into());

        // Convert polynomial to extension field representation
        let poly = EvaluationsList::from(polynomial.clone().to_extension());

        // Compute the weighted sum of the polynomial (for sumcheck)
        let sum = linear_claim_weight.weighted_sum(&poly);

        // Add linear constraint to the statement
        statement.add_constraint(linear_claim_weight, sum);

        // Define the Fiat-Shamir domain separator for committing and proving
        let domainsep = DomainSeparator::new("🌪️")
            .commit_statement(&params)
            .add_whir_proof(&params);

        // Initialize the Merlin transcript from the domain separator
        let mut prover_state = domainsep.to_prover_state();

        // Create a commitment to the polynomial and generate auxiliary witness data
        let committer = CommitmentWriter::new(params.clone());
        let witness = committer.commit(&mut prover_state, polynomial).unwrap();

        // Instantiate the prover with the given parameters
        let prover = Prover(params.clone());

        // Extract verifier-side version of the statement (only public data)
        let statement_verifier = StatementVerifier::from_statement(&statement);

        // Generate a STARK proof for the given statement and witness
        let proof = prover.prove(&mut prover_state, statement, witness).unwrap();

        // Create a commitment reader
        let commitment_reader = CommitmentReader::new(&params);

        // Create a verifier with matching parameters
        let verifier = Verifier::new(&params);

        // Reconstruct verifier's view of the transcript using the DomainSeparator and prover's data
        let mut verifier_state = domainsep.to_verifier_state(prover_state.narg_string());

        // Parse the commitment
        let parsed_commitment = commitment_reader
            .parse_commitment(&mut verifier_state)
            .unwrap();

        // Verify that the generated proof satisfies the statement
        assert!(verifier
            .verify(
                &mut verifier_state,
                &parsed_commitment,
                &statement_verifier,
                &proof
            )
            .is_ok());
    }

    #[test]
    fn test_whir() {
        let folding_factors = [1, 2, 3, 4];
        let soundness_type = [
            SoundnessType::ConjectureList,
            SoundnessType::ProvableList,
            SoundnessType::UniqueDecoding,
        ];
        let fold_types = [FoldType::Naive, FoldType::ProverHelps];
        let num_points = [0, 1, 2];
        let pow_bits = [0, 5, 10];

        for folding_factor in folding_factors {
            let num_variables = folding_factor..=3 * folding_factor;
            for num_variable in num_variables {
                for fold_type in fold_types {
                    for num_points in num_points {
                        for soundness_type in soundness_type {
                            for pow_bits in pow_bits {
                                make_whir_things(
                                    num_variable,
                                    FoldingFactor::Constant(folding_factor),
                                    num_points,
                                    soundness_type,
                                    pow_bits,
                                    fold_type,
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}
