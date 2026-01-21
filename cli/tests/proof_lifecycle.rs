mod tests {
    use std::sync::Arc;

    use ark_ff::Field;
    use spongefish::DomainSeparator;
    use spongefish_pow::blake3::Blake3PoW;

    use common::{
        crypto::{
            fields::{Field64, Field64_2},
            merkle_tree::{
                blake3::{Blake3Compress, Blake3LeafHash, Blake3MerkleTreeParams},
                parameters::default_config,
            },
        },
        ntt::RSDefault,
        parameters::{
            DeduplicationStrategy, FoldingFactor, MerkleProofStrategy, MultivariateParameters,
            ProtocolParameters, SoundnessType,
        },
        poly_utils::{
            coeffs::CoefficientList, evals::EvaluationsList, multilinear::MultilinearPoint,
        },
        utils::test_serde,
        whir::{
            committer::{CommitmentReader, CommitmentWriter},
            domainsep::WhirDomainSeparator,
            parameters::WhirConfig,
            statement::{Statement, Weights},
        },
    };

    use prover::Prover;
    use verifier::Verifier;

    /// Merkle tree configuration type for commitment layers.
    type MerkleConfig = Blake3MerkleTreeParams<F>;

    /// PoW strategy used for grinding challenges in Fiat-Shamir transcript.
    type PowStrategy = Blake3PoW;

    /// Field type used in the tests.
    type F = Field64;

    /// Extension field type used in the tests.
    type EF = Field64_2;

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
    ) {
        // Number of coefficients in the multilinear polynomial (2^num_variables)
        let num_coeffs = 1 << num_variables;

        // Randomness source
        let mut rng = ark_std::test_rng();
        // Generate Merkle parameters: hash function and compression function
        let (leaf_hash_params, two_to_one_params) =
            default_config::<EF, Blake3LeafHash<EF>, Blake3Compress>(&mut rng);

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
            batch_size: 1,
            deduplication_strategy: DeduplicationStrategy::Enabled,
            merkle_proof_strategy: MerkleProofStrategy::Compressed,
        };

        let reed_solomon = Arc::new(RSDefault);
        let basefield_reed_solomon = reed_solomon.clone();
        // Build global configuration from multivariate + protocol parameters
        let params = WhirConfig::new(reed_solomon, basefield_reed_solomon, mv_params, whir_params);

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
            let eval = polynomial.evaluate_at_extension(point);
            let weights = Weights::evaluation(point.clone());
            statement.add_constraint(weights, eval);
        }

        // Construct a coefficient vector for linear sumcheck constraint
        let input = CoefficientList::new(
            (0..1 << num_variables)
                .map(<EF as Field>::BasePrimeField::from)
                .collect(),
        );

        // Define weights for linear combination
        let linear_claim_weight = Weights::linear(input.into());

        // Convert polynomial to extension field representation
        let poly = EvaluationsList::from(polynomial.clone().to_extension());

        // Compute the weighted sum of the polynomial (for sumcheck)
        let sum = linear_claim_weight.weighted_sum(&poly);

        // Add linear constraint to the statement
        statement.add_constraint(linear_claim_weight, sum);

        // Base of the geometric progression [1, 2, 4, 8, 0, ...]
        let geometric_base = F::from(2u64);

        // Number of non-zero terms in the progression.
        let geometric_n = num_coeffs - 1;

        // Create geometric progression weights: [1, a, a^2, a^3, ..., a^(n-1), 0, 0, ..., 0]
        let mut geometric_weights = vec![F::ONE];
        let mut current_power = geometric_base;
        for _i in 1..geometric_n {
            geometric_weights.push(current_power);
            current_power *= geometric_base;
        }

        // Fill remaining corners with zeros
        geometric_weights.resize(num_coeffs, F::from(0));

        // Create EvaluationsList from the geometric progression
        let geometric_weight_list = EvaluationsList::new(geometric_weights);

        // Create geometric weight function
        let geometric_claim_weight =
            Weights::geometric(geometric_base, geometric_n, geometric_weight_list);

        // Convert polynomial to base field evaluation form for weighted_sum
        let poly_base = EvaluationsList::from(polynomial.clone());

        // Compute the weighted sum for geometric constraint
        let geometric_sum = geometric_claim_weight.weighted_sum(&poly_base);

        // Add geometric constraint to statement
        statement.add_constraint(geometric_claim_weight, geometric_sum);

        // Define the Fiat-Shamir domain separator for committing and proving
        let domainsep = DomainSeparator::new("🌪️")
            .commit_statement(&params)
            .add_whir_proof(&params);

        // Initialize the Merlin transcript from the domain separator
        let mut prover_state = domainsep.to_prover_state();

        // Create a commitment to the polynomial and generate auxiliary witness data
        let committer = CommitmentWriter::new(params.clone());
        let witness = committer.commit(&mut prover_state, &polynomial).unwrap();

        // Instantiate the prover with the given parameters
        let prover = Prover::new(params.clone());

        // Generate a STARK proof for the given statement and witness
        prover
            .prove(&mut prover_state, statement.clone(), witness)
            .unwrap();

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
            .verify(&mut verifier_state, parsed_commitment, statement)
            .unwrap();
    }

    #[test]
    fn test_whir_proof_lifecycle() {
        let folding_factors = [1, 2, 3, 4];
        let soundness_type = [
            SoundnessType::ConjectureList,
            SoundnessType::ProvableList,
            SoundnessType::UniqueDecoding,
        ];
        let num_points = [0, 1, 2];
        let pow_bits = [0, 5, 10];

        for folding_factor in folding_factors {
            let num_variables = folding_factor..=3 * folding_factor;
            for num_variable in num_variables {
                for num_points in num_points {
                    for soundness_type in soundness_type {
                        for pow_bits in pow_bits {
                            make_whir_things(
                                num_variable,
                                FoldingFactor::Constant(folding_factor),
                                num_points,
                                soundness_type,
                                pow_bits,
                            );
                        }
                    }
                }
            }
        }
    }

    /// Test batch proving with multiple independent polynomials and statements.
    ///
    /// Creates N separate polynomials, commits to each independently, and uses RLC to batch-prove
    /// them together. This verifies the full lifecycle: commitment, batch proving, and verification.
    fn make_whir_batch_things(
        num_variables: usize,
        folding_factor: FoldingFactor,
        num_points_per_poly: usize,
        num_polynomials: usize,
        soundness_type: SoundnessType,
        pow_bits: usize,
    ) {
        let num_coeffs = 1 << num_variables;
        let mut rng = ark_std::test_rng();

        let (leaf_hash_params, two_to_one_params) =
            default_config::<EF, Blake3LeafHash<EF>, Blake3Compress>(&mut rng);

        let mv_params = MultivariateParameters::new(num_variables);
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
            batch_size: 1,
            deduplication_strategy: DeduplicationStrategy::Enabled,
            merkle_proof_strategy: MerkleProofStrategy::Compressed,
        };

        let reed_solomon = Arc::new(RSDefault);
        let basefield_reed_solomon = reed_solomon.clone();
        let params = WhirConfig::new(reed_solomon, basefield_reed_solomon, mv_params, whir_params);

        // Create N different polynomials
        let polynomials: Vec<_> = (0..num_polynomials)
            .map(|i| {
                // Different polynomials: first is all 1s, second is all 2s, etc.
                CoefficientList::new(vec![F::from((i + 1) as u64); num_coeffs])
            })
            .collect();

        // Create N statements, one for each polynomial
        let mut statements = Vec::new();
        for poly in &polynomials {
            let mut statement = Statement::new(num_variables);

            // Add random point constraints
            for _ in 0..num_points_per_poly {
                let point = MultilinearPoint::rand(&mut rng, num_variables);
                let eval = poly.evaluate_at_extension(&point);
                statement.add_constraint(Weights::evaluation(point), eval);
            }

            // Add linear constraint
            let input = CoefficientList::new(
                (0..1 << num_variables)
                    .map(<EF as Field>::BasePrimeField::from)
                    .collect(),
            );
            let linear_claim_weight = Weights::linear(input.into());
            let poly_evals = EvaluationsList::from(poly.clone().to_extension());
            let sum = linear_claim_weight.weighted_sum(&poly_evals);
            statement.add_constraint(linear_claim_weight, sum);

            statements.push(statement);
        }

        // Set up domain separator for batch proving
        // Each polynomial needs its own commitment phase
        let mut domainsep = DomainSeparator::new("🌪️");
        for _ in 0..num_polynomials {
            domainsep = domainsep.commit_statement(&params);
        }

        // Calculate total constraints for the N×M evaluation matrix
        let total_constraints = num_polynomials * params.committment_ood_samples
            + statements
                .iter()
                .map(|s| s.constraints.len())
                .sum::<usize>();
        domainsep = domainsep.add_whir_batch_proof(&params, num_polynomials, total_constraints);
        let mut prover_state = domainsep.to_prover_state();

        // Commit to each polynomial and generate witnesses
        let committer = CommitmentWriter::new(params.clone());
        let mut witnesses = Vec::new();

        for poly in &polynomials {
            let witness = committer.commit(&mut prover_state, poly).unwrap();
            witnesses.push(witness);
        }

        // Batch prove all polynomials together
        let prover = Prover::new(params.clone());
        let result = prover.prove_batch(&mut prover_state, &statements, &witnesses);

        assert!(result.is_ok(), "Batch proving failed: {:?}", result.err());

        // Set up verifier and parse commitments
        let commitment_reader = CommitmentReader::new(&params);
        let verifier = Verifier::new(&params);

        // Reconstruct verifier's transcript view
        let mut verifier_state = domainsep.to_verifier_state(prover_state.narg_string());

        // Parse all N commitments from the transcript
        let mut parsed_commitments = Vec::new();
        for _ in 0..num_polynomials {
            let parsed_commitment = commitment_reader
                .parse_commitment(&mut verifier_state)
                .unwrap();
            parsed_commitments.push(parsed_commitment);
        }

        // Verify the batched proof
        verifier
            .verify_batch(&mut verifier_state, &parsed_commitments, &statements)
            .unwrap();
    }

    #[test]
    fn test_whir_batch() {
        // Test with different configurations
        let folding_factors = [2, 3, 4];
        let num_polynomials = [2, 3, 4];
        let num_points = [0, 1, 2];

        for folding_factor in folding_factors {
            for num_polys in num_polynomials {
                for num_points_per_poly in num_points {
                    make_whir_batch_things(
                        folding_factor * 2, // num_variables
                        FoldingFactor::Constant(folding_factor),
                        num_points_per_poly,
                        num_polys,
                        SoundnessType::ConjectureList,
                        0, // pow_bits
                    );
                }
            }
        }
    }

    #[test]
    fn test_whir_batch_single_polynomial() {
        // Edge case: batch proving with just one polynomial should also work
        make_whir_batch_things(
            6, // num_variables
            FoldingFactor::Constant(2),
            2, // num_points_per_poly
            1, // num_polynomials (single!)
            SoundnessType::ConjectureList,
            0,
        );
    }

    /// Test that batch verification rejects proofs with mismatched polynomials.
    ///
    /// This security test verifies that the cross-term commitment prevents the prover from
    /// using a different polynomial than what was committed. The prover commits to poly2 but
    /// attempts to use poly_wrong for evaluation, which should cause verification to fail.
    #[test]
    fn test_whir_batch_rejects_invalid_constraint() {
        // Setup parameters
        let num_variables = 4;
        let folding_factor = FoldingFactor::Constant(2);
        let num_polynomials = 2;
        let num_coeffs = 1 << num_variables;
        let mut rng = ark_std::test_rng();

        let (leaf_hash_params, two_to_one_params) =
            default_config::<EF, Blake3LeafHash<EF>, Blake3Compress>(&mut rng);

        let mv_params = MultivariateParameters::new(num_variables);
        let whir_params = ProtocolParameters::<MerkleConfig, PowStrategy> {
            initial_statement: true,
            security_level: 32,
            pow_bits: 0,
            folding_factor,
            leaf_hash_params,
            two_to_one_params,
            soundness_type: SoundnessType::ConjectureList,
            _pow_parameters: Default::default(),
            starting_log_inv_rate: 1,
            batch_size: 1,
            deduplication_strategy: DeduplicationStrategy::Enabled,
            merkle_proof_strategy: MerkleProofStrategy::Compressed,
        };

        let reed_solomon = Arc::new(RSDefault);
        let basefield_reed_solomon = reed_solomon.clone();
        let params = WhirConfig::new(reed_solomon, basefield_reed_solomon, mv_params, whir_params);

        // Create test polynomials
        let poly1 = CoefficientList::new(vec![F::ONE; num_coeffs]);
        let poly2 = CoefficientList::new(vec![F::from(2u64); num_coeffs]);
        let poly_wrong = CoefficientList::new(vec![F::from(999u64); num_coeffs]);

        // Create valid statements for poly1 and poly2
        let mut statement1 = Statement::new(num_variables);
        let point1 = MultilinearPoint::rand(&mut rng, num_variables);
        let eval1 = poly1.evaluate_at_extension(&point1);
        statement1.add_constraint(Weights::evaluation(point1), eval1);

        let mut statement2 = Statement::new(num_variables);
        let point2 = MultilinearPoint::rand(&mut rng, num_variables);
        let eval2 = poly2.evaluate_at_extension(&point2);
        statement2.add_constraint(Weights::evaluation(point2), eval2);

        let statements = vec![statement1, statement2];

        // Commit to the correct polynomials
        let mut domainsep = DomainSeparator::new("🌪️");
        for _ in 0..num_polynomials {
            domainsep = domainsep.commit_statement(&params);
        }
        let total_constraints = num_polynomials * params.committment_ood_samples
            + statements
                .iter()
                .map(|s| s.constraints.len())
                .sum::<usize>();
        domainsep = domainsep.add_whir_batch_proof(&params, num_polynomials, total_constraints);
        let mut prover_state = domainsep.to_prover_state();

        let committer = CommitmentWriter::new(params.clone());
        let witness1 = committer.commit(&mut prover_state, &poly1).unwrap();
        let witness2_committed = committer.commit(&mut prover_state, &poly2).unwrap();

        // ATTACK: Create a fake witness using poly_wrong instead of poly2
        // The commitment is valid for poly2, but we'll use poly_wrong for evaluation
        let mut witness2_fake = witness2_committed;
        witness2_fake.polynomial = poly_wrong;

        let witnesses = vec![witness1, witness2_fake];

        // Generate proof with the mismatched polynomial
        // The prover will compute cross-terms using poly_wrong, not poly2
        let prover = Prover::new(params.clone());
        let result = prover.prove_batch(&mut prover_state, &statements, &witnesses);

        // Proof generation succeeds (prover doesn't verify polynomial-commitment consistency)
        assert!(result.is_ok(), "Prover generated proof");

        // Verification should fail because the cross-terms don't match the commitment
        let commitment_reader = CommitmentReader::new(&params);
        let verifier = Verifier::new(&params);
        let mut verifier_state = domainsep.to_verifier_state(prover_state.narg_string());

        let mut parsed_commitments = Vec::new();
        for _ in 0..num_polynomials {
            let parsed_commitment = commitment_reader
                .parse_commitment(&mut verifier_state)
                .unwrap();
            parsed_commitments.push(parsed_commitment);
        }

        let verify_result =
            verifier.verify_batch(&mut verifier_state, &parsed_commitments, &statements);
        assert!(
            verify_result.is_err(),
            "Verifier should reject mismatched polynomial"
        );
    }

    /// Test batch proving with batch_size > 1 (multiple polynomials per commitment).
    ///
    /// This tests the case where each commitment contains multiple stacked polynomials
    /// (e.g., masked witness + random blinding for ZK), and we batch-prove multiple
    /// such commitments together.
    ///
    /// This was a regression test for a bug where the RLC combination of stacked
    /// leaf answers was incorrect when batch_size > 1.
    fn make_whir_batch_with_batch_size(
        num_variables: usize,
        folding_factor: FoldingFactor,
        num_points_per_poly: usize,
        num_witnesses: usize,
        batch_size: usize,
        soundness_type: SoundnessType,
        pow_bits: usize,
    ) {
        let num_coeffs = 1 << num_variables;
        let mut rng = ark_std::test_rng();

        let (leaf_hash_params, two_to_one_params) =
            default_config::<EF, Blake3LeafHash<EF>, Blake3Compress>(&mut rng);

        let mv_params = MultivariateParameters::new(num_variables);
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
            batch_size, // KEY: batch_size > 1
            deduplication_strategy: DeduplicationStrategy::Enabled,
            merkle_proof_strategy: MerkleProofStrategy::Compressed,
        };

        let reed_solomon = Arc::new(RSDefault);
        let basefield_reed_solomon = reed_solomon.clone();
        let params = WhirConfig::new(reed_solomon, basefield_reed_solomon, mv_params, whir_params);

        // Create polynomials for each witness
        // Each witness will contain batch_size polynomials committed together
        let mut all_polynomials: Vec<Vec<CoefficientList<F>>> = Vec::new();
        for w in 0..num_witnesses {
            let witness_polys: Vec<_> = (0..batch_size)
                .map(|b| {
                    // Different polynomials: witness 0 batch 0 = all 1s, witness 0 batch 1 = all 2s, etc.
                    CoefficientList::new(vec![
                        F::from(((w * batch_size + b) + 1) as u64);
                        num_coeffs
                    ])
                })
                .collect();
            all_polynomials.push(witness_polys);
        }

        // Create statements - one per witness, based on the first polynomial in each batch
        // (the internally-batched polynomial is what the statement is about)
        let mut statements = Vec::new();
        for witness_polys in &all_polynomials {
            let poly = &witness_polys[0]; // Use first poly for statement (will be batched internally)
            let mut statement = Statement::new(num_variables);

            for _ in 0..num_points_per_poly {
                let point = MultilinearPoint::rand(&mut rng, num_variables);
                let eval = poly.evaluate_at_extension(&point);
                statement.add_constraint(Weights::evaluation(point), eval);
            }

            // Add a linear constraint
            let input = CoefficientList::new(
                (0..1 << num_variables)
                    .map(<EF as Field>::BasePrimeField::from)
                    .collect(),
            );
            let linear_claim_weight = Weights::linear(input.into());
            let poly_evals = EvaluationsList::from(poly.clone().to_extension());
            let sum = linear_claim_weight.weighted_sum(&poly_evals);
            statement.add_constraint(linear_claim_weight, sum);

            statements.push(statement);
        }

        // Set up domain separator
        let mut domainsep = DomainSeparator::new("🌪️");
        for _ in 0..num_witnesses {
            domainsep = domainsep.commit_statement(&params);
        }

        let total_constraints = num_witnesses * params.committment_ood_samples
            + statements
                .iter()
                .map(|s| s.constraints.len())
                .sum::<usize>();
        domainsep = domainsep.add_whir_batch_proof(&params, num_witnesses, total_constraints);
        let mut prover_state = domainsep.to_prover_state();

        // Commit using commit_batch (stacks batch_size polynomials per witness)
        let committer = CommitmentWriter::new(params.clone());
        let mut witnesses = Vec::new();

        for witness_polys in &all_polynomials {
            let poly_refs: Vec<_> = witness_polys.iter().collect();
            let witness = committer
                .commit_batch(&mut prover_state, &poly_refs)
                .unwrap();
            witnesses.push(witness);
        }

        // Batch prove all witnesses together
        let prover = Prover::new(params.clone());
        let result = prover.prove_batch(&mut prover_state, &statements, &witnesses);
        assert!(
            result.is_ok(),
            "Batch proving with batch_size={} failed: {:?}",
            batch_size,
            result.err()
        );

        // Verify
        let commitment_reader = CommitmentReader::new(&params);
        let verifier = Verifier::new(&params);
        let mut verifier_state = domainsep.to_verifier_state(prover_state.narg_string());

        let mut parsed_commitments = Vec::new();
        for _ in 0..num_witnesses {
            let parsed_commitment = commitment_reader
                .parse_commitment(&mut verifier_state)
                .unwrap();
            parsed_commitments.push(parsed_commitment);
        }

        let verify_result =
            verifier.verify_batch(&mut verifier_state, &parsed_commitments, &statements);
        assert!(
            verify_result.is_ok(),
            "Batch verification with batch_size={} failed: {:?}",
            batch_size,
            verify_result.err()
        );
    }

    #[test]
    fn test_whir_batch_with_batch_size_2() {
        // This is the key regression test for the batch_size > 1 bug
        let batch_sizes = [2, 3];
        let num_witnesses = [2, 3];
        let folding_factors = [2, 3];

        for batch_size in batch_sizes {
            for num_witness in num_witnesses {
                for folding_factor in folding_factors {
                    make_whir_batch_with_batch_size(
                        folding_factor * 2, // num_variables
                        FoldingFactor::Constant(folding_factor),
                        1, // num_points_per_poly
                        num_witness,
                        batch_size,
                        SoundnessType::ConjectureList,
                        0, // pow_bits
                    );
                }
            }
        }
    }
}
