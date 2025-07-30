///
/// High level idea of batching multiple polynomials is as follows:
///
/// - Prover commits to multiple Merkle roots of evaluations of each polynomial
///
/// - Verifier samples a batching_randomness field element. For soundness
///   reasons, batching_randomness must be sampled after the Merkle roots have
///   been committed.
///
/// - Prover computes the weighted sum of each individual polynomial based on
///   batching_randomness to compute a single polynomial to proceed during
///   Sumcheck rounds and any other future round.
///
/// - In the first round of the proof, Prover includes the STIR queries from
///   each individual oracle. (The STIR query indexes should be the same for all
///   oracles.)
///
/// - During proof verification, for the first round, the Verifier validates
///   each individual STIR query in the proof against individual Merkle root.
///   Once these Merkle paths are validated, the Verifier re-derives
///   batching_randomness and combines the STIR responses using powers of
///   batching_randomness.
///
/// - After the first round, rest of the protocol proceeds as usual.
///
#[cfg(test)]
mod batching_tests {
    use ark_std::UniformRand;
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
        parameters::{FoldingFactor, MultivariateParameters, ProtocolParameters, SoundnessType},
        poly_utils::{
            coeffs::CoefficientList, evals::EvaluationsList, multilinear::MultilinearPoint,
        },
        whir::{
            committer::{reader::CommitmentReader, CommitmentWriter},
            domainsep::WhirDomainSeparator,
            parameters::WhirConfig,
            prover::Prover,
            statement::{Statement, Weights},
            verifier::Verifier,
        },
    };

    /// Merkle tree configuration type for commitment layers.
    type MerkleConfig = Blake3MerkleTreeParams<F>;
    /// PoW strategy used for grinding challenges in Fiat-Shamir transcript.
    type PowStrategy = Blake3PoW;
    /// Field type used in the tests.
    type F = Field64;

    fn random_poly(num_coefficients: usize) -> CoefficientList<F> {
        let mut store = Vec::<F>::with_capacity(num_coefficients);
        let mut rng = ark_std::rand::thread_rng();
        (0..num_coefficients).for_each(|_| store.push(F::rand(&mut rng)));

        CoefficientList::new(store)
    }

    /// Run a complete WHIR STARK proof lifecycle: commit, prove, and verify.
    ///
    /// This function:
    /// - builds a multilinear polynomial with a specified number of variables,
    /// - constructs a STARK statement with constraints based on evaluations and linear relations,
    /// - commits to the polynomial using a Merkle-based commitment scheme,
    /// - generates a proof using the WHIR prover,
    /// - verifies the proof using the WHIR verifier.
    fn make_batched_whir_things(
        batch_size: usize,
        num_variables: usize,
        folding_factor: FoldingFactor,
        num_points: usize,
        soundness_type: SoundnessType,
        pow_bits: usize,
    ) {
        println!("Test parameters: ");
        println!("  num_polynomials: {batch_size}");
        println!("  num_variables  : {num_variables}");
        println!("  folding_factor : {:?}", &folding_factor);
        println!("  num_points     : {num_points:?}");
        println!("  soundness_type : {soundness_type:?}");
        println!("  pow_bits       : {pow_bits}");

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
            batch_size,
        };

        // Build global configuration from multivariate + protocol parameters
        let params = WhirConfig::new(mv_params, whir_params);

        let mut poly_list = Vec::<CoefficientList<F>>::with_capacity(batch_size);

        (0..batch_size).for_each(|_| poly_list.push(random_poly(num_coeffs)));

        // Construct a coefficient vector for linear sumcheck constraint
        let weight_poly = CoefficientList::new((0..1 << num_variables).map(F::from).collect());

        // Generate `num_points` random points in the multilinear domain
        let points: Vec<_> = (0..num_points)
            .map(|_| MultilinearPoint::rand(&mut rng, num_variables))
            .collect();

        // Define the Fiat-Shamir IOPattern for committing and proving
        let io = DomainSeparator::new("🌪️")
            .commit_statement(&params)
            .add_whir_proof(&params);

        // println!("IO Domain Separator: {:?}", str::from_utf8(io.as_bytes()));
        // Initialize the Merlin transcript from the IOPattern
        let mut prover_state = io.to_prover_state();

        // Create a commitment to the polynomial and generate auxiliary witness data
        let committer = CommitmentWriter::new(params.clone());
        let batched_witness = committer
            .commit_batch(&mut prover_state, &poly_list)
            .unwrap();

        // Get the batched polynomial
        let batched_poly = batched_witness.batched_poly();

        // Initialize a statement with no constraints yet
        let mut statement = Statement::new(num_variables);

        // For each random point, evaluate the polynomial and create a constraint
        for point in &points {
            let eval = batched_poly.evaluate(point);
            let weights = Weights::evaluation(point.clone());
            statement.add_constraint(weights, eval);
        }

        // Define weights for linear combination
        let linear_claim_weight = Weights::linear(weight_poly.into());

        // Convert polynomial to extension field representation
        let poly = EvaluationsList::from(batched_poly.clone().to_extension());

        // Compute the weighted sum of the polynomial (for sumcheck)
        let sum = linear_claim_weight.weighted_sum(&poly);

        // Add linear constraint to the statement
        statement.add_constraint(linear_claim_weight, sum);

        // Instantiate the prover with the given parameters
        let prover = Prover(params.clone());

        // Extract verifier-side version of the statement (only public data)

        // Generate a STARK proof for the given statement and witness
        prover
            .prove(&mut prover_state, statement.clone(), batched_witness)
            .unwrap();

        // Create a verifier with matching parameters
        let verifier = Verifier::new(&params);

        // Reconstruct verifier's view of the transcript using the IOPattern and prover's data
        let mut verifier_state = io.to_verifier_state(prover_state.narg_string());

        // Create a commitment reader
        let commitment_reader = CommitmentReader::new(&params);

        let parsed_commitment = commitment_reader
            .parse_commitment(&mut verifier_state)
            .unwrap();

        // Verify that the generated proof satisfies the statement
        assert!(verifier
            .verify(&mut verifier_state, &parsed_commitment, &statement,)
            .is_ok());
    }

    #[test]
    fn test_whir() {
        let folding_factors = [1, 4];
        let soundness_type = [
            SoundnessType::ConjectureList,
            SoundnessType::ProvableList,
            SoundnessType::UniqueDecoding,
        ];
        let num_points = [0, 2];
        let pow_bits = [0, 10];

        for folding_factor in folding_factors {
            let num_variables = (2 * folding_factor)..=3 * folding_factor;
            for num_variable in num_variables {
                for num_points in num_points {
                    for soundness_type in soundness_type {
                        for pow_bits in pow_bits {
                            for batch_size in 1..=4 {
                                make_batched_whir_things(
                                    batch_size,
                                    num_variable,
                                    FoldingFactor::Constant(folding_factor),
                                    num_points,
                                    soundness_type,
                                    pow_bits,
                                    // FoldType::Naive,
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}
