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
    use std::iter;

    use ark_std::UniformRand;

    use super::super::Config;
    use crate::{
        algebra::{
            fields::Field64,
            polynomials::{CoefficientList, MultilinearPoint},
            Weights,
        },
        hash,
        parameters::{FoldingFactor, MultivariateParameters, ProtocolParameters, SoundnessType},
        transcript::{codecs::Empty, DomainSeparator, ProverState, VerifierState},
    };

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
        eprintln!("\n---------------------");
        eprintln!("Test parameters: ");
        eprintln!("  num_polynomials: {batch_size}");
        eprintln!("  num_variables  : {num_variables}");
        eprintln!("  folding_factor : {:?}", &folding_factor);
        eprintln!("  num_points     : {num_points:?}");
        eprintln!("  soundness_type : {soundness_type:?}");
        eprintln!("  pow_bits       : {pow_bits}");

        // Number of coefficients in the multilinear polynomial (2^num_variables)
        let num_coeffs = 1 << num_variables;

        // Randomness source
        let mut rng = ark_std::test_rng();
        // Configure multivariate polynomial parameters
        let mv_params = MultivariateParameters::new(num_variables);

        // Configure the WHIR protocol parameters
        let whir_params = ProtocolParameters {
            initial_statement: true,
            security_level: 32,
            pow_bits,
            folding_factor,
            soundness_type,
            starting_log_inv_rate: 1,
            batch_size,
            hash_id: hash::SHA2,
        };

        // Build global configuration from multivariate + protocol parameters
        let params = Config::new(mv_params, &whir_params);

        let mut poly_list = Vec::<CoefficientList<F>>::with_capacity(batch_size);

        (0..batch_size).for_each(|_| poly_list.push(random_poly(num_coeffs)));

        // Construct a coefficient vector for linear sumcheck constraint
        let weight_poly = CoefficientList::new((0..1 << num_variables).map(F::from).collect());

        // Generate `num_points` random points in the multilinear domain
        let points: Vec<_> = (0..num_points)
            .map(|_| MultilinearPoint::rand(&mut rng, num_variables))
            .collect();

        // Define the Fiat-Shamir IOPattern for committing and proving
        let ds = DomainSeparator::protocol(&params)
            .session(&format!("Test at {}:{}", file!(), line!()))
            .instance(&Empty);

        // Initialize the Merlin transcript from the domain separator
        let mut prover_state = ProverState::new_std(&ds);

        // Create a commitment to the polynomial and generate auxiliary witness data
        let poly_refs = poly_list.iter().collect::<Vec<_>>();
        let batched_witness = params.commit(&mut prover_state, &poly_refs);

        // Create a weights matrix and evaluations for each polynomial
        let weights = points
            .iter()
            .map(|point| Weights::evaluation(point.clone()))
            .chain(iter::once(Weights::linear(weight_poly.into())))
            .collect::<Vec<_>>();
        let weights_refs = weights.iter().collect::<Vec<_>>();
        let values = weights
            .iter()
            .flat_map(|weights| poly_list.iter().map(|poly| weights.evaluate(poly)))
            .collect::<Vec<_>>();

        // Generate a STARK proof for the given statement and witness
        params.prove(
            &mut prover_state,
            &poly_refs,
            &[&batched_witness],
            &weights_refs,
            &values,
        );

        // Reconstruct verifier's view of the transcript using the IOPattern and prover's data
        let proof = prover_state.proof();
        let mut verifier_state = VerifierState::new_std(&ds, &proof);

        let commitment = params.receive_commitment(&mut verifier_state).unwrap();

        // Verify that the generated proof satisfies the statement
        assert!(params
            .verify(&mut verifier_state, &[&commitment], &weights_refs, &values)
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
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}
