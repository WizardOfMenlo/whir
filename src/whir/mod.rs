pub mod committer;
pub mod domainsep;
pub mod parameters;
pub mod parsed_proof;
pub mod prover;
pub mod statement;
pub mod stir_evaluations;
pub mod utils;
pub mod verifier;

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
            FoldType, FoldingFactor, MultivariateParameters, ProtocolParameters, SoundnessType,
        },
        poly_utils::{
            coeffs::CoefficientList, evals::EvaluationsList, multilinear::MultilinearPoint,
        },
        utils::test_serde,
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
        let whir_params = ProtocolParameters::<MerkleConfig, PowStrategy> {
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
        };

        // Build global configuration from multivariate + protocol parameters
        let params = WhirConfig::new(mv_params, whir_params);

        // Test that the config is serializable
        eprintln!("{params:?}");
        test_serde(&params);

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

        // Test that the proof is serializable
        test_serde(&proof);

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
        verifier
            .verify(&mut verifier_state, &parsed_commitment, &statement_verifier)
            .unwrap();
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
