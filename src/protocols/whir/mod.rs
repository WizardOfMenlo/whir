#![allow(type_alias_bounds)] // We need the bound to reference F::BasePrimeField.

mod config;
mod prover;
mod verifier;

use std::fmt::Debug;

use ark_ff::{FftField, Field};
use ark_std::rand::{CryptoRng, RngCore};
use serde::{Deserialize, Serialize};
#[cfg(feature = "tracing")]
use tracing::instrument;

use crate::{
    hash::Hash,
    protocols::{irs_commit, proof_of_work, sumcheck},
    transcript::{
        Codec, DuplexSpongeInterface, ProverMessage, ProverState, VerificationResult, VerifierState,
    },
};

#[derive(Clone, PartialEq, Eq, Serialize, Deserialize, Debug)]
#[serde(bound = "F: FftField")]
pub struct Config<F>
where
    F: FftField,
{
    pub initial_committer: irs_commit::BasefieldConfig<F>,
    pub initial_sumcheck: sumcheck::Config<F>,
    pub initial_skip_pow: proof_of_work::Config,
    pub round_configs: Vec<RoundConfig<F>>,
    pub final_sumcheck: sumcheck::Config<F>,
    pub final_pow: proof_of_work::Config,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "F: FftField")]
pub struct RoundConfig<F>
where
    F: FftField,
{
    pub irs_committer: irs_commit::Config<F>,
    pub sumcheck: sumcheck::Config<F>,
    pub pow: proof_of_work::Config,
}

pub type Witness<F: FftField> = irs_commit::Witness<F::BasePrimeField, F>;
pub type Commitment<F: Field> = irs_commit::Commitment<F>;

impl<F: FftField> Config<F> {
    /// Commit to one or more vectors.
    #[cfg_attr(feature = "tracing", instrument(skip_all, fields(size = vectors.first().unwrap().len())))]
    pub fn commit<H, R>(
        &self,
        prover_state: &mut ProverState<H, R>,
        vectors: &[&[F::BasePrimeField]],
    ) -> Witness<F>
    where
        H: DuplexSpongeInterface,
        R: RngCore + CryptoRng,
        F: Codec<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        self.initial_committer.commit(prover_state, vectors)
    }

    /// Receive a commitment to vectors.
    pub fn receive_commitment<H>(
        &self,
        verifier_state: &mut VerifierState<H>,
    ) -> VerificationResult<Commitment<F>>
    where
        H: DuplexSpongeInterface,
        F: Codec<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        self.initial_committer.receive_commitment(verifier_state)
    }

    /// Disable proof-of-work for test.
    #[cfg(test)]
    pub(crate) fn disable_pow(&mut self) {
        use std::u64;
        self.initial_sumcheck.round_pow.threshold = u64::MAX;
        for round in &mut self.round_configs {
            round.sumcheck.round_pow.threshold = u64::MAX;
            round.pow.threshold = u64::MAX;
        }
        self.final_sumcheck.round_pow.threshold = u64::MAX;
        self.final_pow.threshold = u64::MAX;
    }
}

#[cfg(test)]
mod tests {
    use std::u64;

    use ark_ff::{Field, UniformRand};

    use super::*;
    use crate::{
        algebra::{
            embedding::Basefield,
            fields::{Field64, Field64_3},
            linear_form::{Covector, Evaluate, LinearForm, MultilinearExtension},
            MultilinearPoint,
        },
        hash,
        parameters::ProtocolParameters,
        transcript::{codecs::Empty, DomainSeparator, ProverState, VerifierState},
        utils::test_serde,
    };

    /// Field type used in the tests.
    type F = Field64;

    /// Extension field type used in the tests.
    type EF = Field64_3;

    /// Run a complete WHIR proof lifecycle: commit, prove, and verify.
    ///
    /// This function:
    /// - builds a multilinear polynomial with a specified number of variables,
    /// - constructs a statement with constraints based on evaluations and linear relations,
    /// - commits to the polynomial using a Merkle-based commitment scheme,
    /// - generates a proof using the WHIR prover,
    /// - verifies the proof using the WHIR verifier.
    fn make_whir_things(
        num_variables: usize,
        initial_folding_factor: usize,
        folding_factor: usize,
        num_points: usize,
        unique_decoding: bool,
        pow_bits: usize,
    ) {
        // Number of coefficients in the multilinear polynomial (2^num_variables)
        let num_coeffs = 1 << num_variables;

        // Randomness source
        let mut rng = ark_std::test_rng();

        // Configure the WHIR protocol parameters
        let whir_params = ProtocolParameters {
            security_level: 32,
            pow_bits,
            initial_folding_factor,
            folding_factor,
            unique_decoding,
            starting_log_inv_rate: 1,
            batch_size: 1,
            hash_id: hash::SHA2,
        };

        // Build global configuration from protocol parameters
        let mut params = Config::new(1 << num_variables, &whir_params);
        params.disable_pow();
        eprintln!("{params}");

        // Test that the config is serializable
        test_serde(&params);

        // Our test vector is all ones in the basefield.
        let vector = vec![F::ONE; num_coeffs];

        // Generate `num_points` random points in the multilinear domain
        let points: Vec<_> = (0..num_points)
            .map(|_| MultilinearPoint::rand(&mut rng, num_variables))
            .collect();

        // Initialize a statement with no constraints yet
        let mut linear_forms: Vec<Box<dyn LinearForm<EF>>> = Vec::new();
        let mut evaluations = Vec::new();

        // For each random point, evaluate the mle of the vector and create a constraint
        for point in &points {
            let linear_form = MultilinearExtension {
                point: point.0.clone(),
            };
            evaluations.push(linear_form.evaluate(params.embedding(), &vector));
            linear_forms.push(Box::new(linear_form));
        }

        // Construct a covector for linear sumcheck constraint
        let covector = Covector {
            deferred: false,
            vector: (0..1 << num_variables).map(EF::from).collect(),
        };

        // Compute the weighted sum of the vector
        let sum = covector.evaluate(params.embedding(), &vector);

        // Add linear constraint to the statement
        linear_forms.push(Box::new(covector));
        evaluations.push(sum);

        // Define the Fiat-Shamir domain separator for committing and proving
        let ds = DomainSeparator::protocol(&params)
            .session(&format!("Test at {}:{}", file!(), line!()))
            .instance(&Empty);

        // Initialize the Merlin transcript from the domain separator
        let mut prover_state = ProverState::new_std(&ds);

        // Create a commitment to the polynomial and generate auxiliary witness data
        let witness = params.commit(&mut prover_state, &[&vector]);

        // Generate a STARK proof for the given statement and witness
        let linear_form_refs = linear_forms
            .iter()
            .map(|l| l.as_ref() as &dyn LinearForm<EF>)
            .collect::<Vec<_>>();
        params.prove(
            &mut prover_state,
            &[&vector],
            &[&witness],
            &linear_form_refs,
            &evaluations,
        );

        // Reconstruct verifier's view of the transcript using the DomainSeparator and prover's data
        let proof = prover_state.proof();
        let mut verifier_state = VerifierState::new_std(&ds, &proof);

        // Parse the commitment
        let commitment = params.receive_commitment(&mut verifier_state).unwrap();

        // Verify that the generated proof satisfies the statement
        params
            .verify(
                &mut verifier_state,
                &[&commitment],
                &linear_form_refs,
                &evaluations,
            )
            .unwrap();
    }

    #[test]
    fn test_whir_1() {
        for folding_factor in [1, 2, 3, 4] {
            let num_variables = folding_factor..=3 * folding_factor;
            for num_variable in num_variables {
                for num_points in [0, 1, 2] {
                    for unique_decoding in [true, false] {
                        for pow_bits in [0, 5, 10] {
                            eprintln!();
                            dbg!(
                                folding_factor,
                                num_variable,
                                num_points,
                                unique_decoding,
                                pow_bits
                            );

                            make_whir_things(
                                num_variable,
                                folding_factor,
                                folding_factor,
                                num_points,
                                unique_decoding,
                                pow_bits,
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_fail() {
        make_whir_things(3, 2, 2, 0, false, 0);
    }

    #[test]
    fn test_whir_mixed_folding_factors() {
        let folding_factors = [1, 2, 3, 4];
        let num_points = [0, 1, 2];

        for initial_folding_factor in folding_factors {
            for folding_factor in folding_factors {
                if initial_folding_factor == folding_factor {
                    continue;
                }
                let n = std::cmp::max(initial_folding_factor, folding_factor);
                let num_variables = n..=3 * n;
                for num_variable in num_variables {
                    for num_points in num_points {
                        eprintln!();
                        dbg!(
                            initial_folding_factor,
                            folding_factor,
                            num_variable,
                            num_points,
                        );

                        make_whir_things(
                            num_variable,
                            initial_folding_factor,
                            folding_factor,
                            num_points,
                            false,
                            5,
                        );
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
        initial_folding_factor: usize,
        folding_factor: usize,
        num_points_per_poly: usize,
        num_vectors: usize,
        unique_decoding: bool,
        pow_bits: usize,
    ) {
        let num_coeffs = 1 << num_variables;
        let mut rng = ark_std::test_rng();

        let whir_params = ProtocolParameters {
            security_level: 32,
            pow_bits,
            initial_folding_factor,
            folding_factor,
            unique_decoding,
            starting_log_inv_rate: 1,
            batch_size: 1,
            hash_id: hash::SHA2,
        };

        let mut params = Config::new(1 << num_variables, &whir_params);
        params.disable_pow();
        eprintln!("{params}");

        // Create N different vectors
        let vectors: Vec<_> = (0..num_vectors)
            .map(|i| {
                // Different vectors: first is all 1s, second is all 2s, etc.
                vec![F::from((i + 1) as u64); num_coeffs]
            })
            .collect();
        let vec_refs = vectors.iter().map(|v| v.as_slice()).collect::<Vec<_>>();

        // Create weights to constraint the polynomials with.
        // Add random point constraints
        let mut linear_forms: Vec<Box<dyn Evaluate<Basefield<EF>>>> = Vec::new();
        for _ in 0..num_points_per_poly {
            linear_forms.push(Box::new(MultilinearExtension {
                point: MultilinearPoint::rand(&mut rng, num_variables).0,
            }));
        }
        // Add linear constraint
        linear_forms.push(Box::new(Covector {
            deferred: false,
            vector: ((0..1 << num_variables).map(EF::from).collect()),
        }));

        // Evaluate all polys on all weights to get constraints
        let evaluations = linear_forms
            .iter()
            .flat_map(|linear_form| {
                vec_refs
                    .iter()
                    .map(|vec| linear_form.evaluate(params.embedding(), vec))
            })
            .collect::<Vec<_>>();

        // Set up domain separator for batch proving
        // Each polynomial needs its own commitment phase
        let ds = DomainSeparator::protocol(&params)
            .session(&format!("Test at {}:{}", file!(), line!()))
            .instance(&Empty);
        let mut prover_state = ProverState::new_std(&ds);

        // Commit to each polynomial and generate witnesses
        let mut witnesses = Vec::new();
        for &vec in &vec_refs {
            let witness = params.commit(&mut prover_state, &[vec]);
            witnesses.push(witness);
        }
        let witness_refs = witnesses.iter().collect::<Vec<_>>();

        // Batch prove all polynomials together
        let linear_form_refs = linear_forms
            .iter()
            .map(|l| l.as_ref() as &dyn LinearForm<EF>)
            .collect::<Vec<_>>();
        let (_point, _evals) = params.prove(
            &mut prover_state,
            &vec_refs,
            &witness_refs,
            &linear_form_refs,
            &evaluations,
        );

        // Reconstruct verifier's transcript view
        let proof = prover_state.proof();
        let mut verifier_state = VerifierState::new_std(&ds, &proof);

        // Parse all N commitments from the transcript
        let mut commitments = Vec::new();
        for _ in 0..num_vectors {
            let commitment = params.receive_commitment(&mut verifier_state).unwrap();
            commitments.push(commitment);
        }
        let commitment_refs = commitments.iter().collect::<Vec<_>>();

        // Verify the batched proof
        params
            .verify(
                &mut verifier_state,
                &commitment_refs,
                &linear_form_refs,
                &evaluations,
            )
            .unwrap();
    }

    #[test]
    fn test_whir_batch_1() {
        // Test with different configurations
        let folding_factors = [1, 2, 3, 4];
        let num_polynomials = [2, 3, 4];
        let num_points = [0, 1, 2];

        for initial_folding_factor in folding_factors {
            for folding_factor in folding_factors {
                let n = std::cmp::max(initial_folding_factor, folding_factor);
                // TODO: Batching with small number of variables..
                for num_variables in (initial_folding_factor + folding_factor)..=3 * n {
                    for num_polys in num_polynomials {
                        for num_points_per_poly in num_points {
                            eprintln!();
                            dbg!(
                                initial_folding_factor,
                                folding_factor,
                                num_variables,
                                num_polys,
                                num_points_per_poly,
                            );
                            make_whir_batch_things(
                                num_variables,
                                initial_folding_factor,
                                folding_factor,
                                num_points_per_poly,
                                num_polys,
                                false,
                                0, // pow_bits
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_whir_batch_single_polynomial() {
        // Edge case: batch proving with just one polynomial should also work
        make_whir_batch_things(
            6, // num_variables
            2, // initial_folding_factor
            2, // folding_factor
            2, // num_points_per_poly
            1, // num_polynomials (single!)
            false, 0,
        );
    }

    /// Test that batch verification rejects proofs with mismatched polynomials.
    ///
    /// This security test verifies that the cross-term commitment prevents the prover from
    /// using a different polynomial than what was committed. The prover commits to poly2 but
    /// attempts to use poly_wrong for evaluation, which should cause verification to fail.
    #[test]
    #[cfg_attr(feature = "verifier_panics", should_panic)]
    #[cfg_attr(
        debug_assertions,
        ignore = "debug_assert in prover panics on intentionally invalid input"
    )]
    fn test_whir_batch_rejects_invalid_constraint() {
        // Setup parameters
        let num_variables = 4;
        let initial_folding_factor = 2;
        let folding_factor = 2;
        let num_polynomials = 2;
        let num_coeffs = 1 << num_variables;
        let mut rng = ark_std::test_rng();

        let whir_params = ProtocolParameters {
            security_level: 32,
            pow_bits: 0,
            initial_folding_factor,
            folding_factor,
            unique_decoding: false,
            starting_log_inv_rate: 1,
            batch_size: 1,
            hash_id: hash::SHA2,
        };

        let mut params = Config::<EF>::new(1 << num_variables, &whir_params);
        params.disable_pow();

        let embedding = Basefield::<EF>::new();

        // Create test vectors
        let vec1 = vec![F::ONE; num_coeffs];
        let vec2 = vec![F::from(2u64); num_coeffs];
        let vec_wrong = vec![F::from(999u64); num_coeffs];

        // Create test weights
        let linear_forms: [Box<dyn Evaluate<Basefield<EF>>>; 2] = [
            Box::new(MultilinearExtension {
                point: MultilinearPoint::rand(&mut rng, num_variables).0,
            }),
            Box::new(MultilinearExtension {
                point: MultilinearPoint::rand(&mut rng, num_variables).0,
            }),
        ];
        // Create valid evaluations for (poly1, polywrong)
        let evaluations = linear_forms
            .iter()
            .flat_map(|weights| [&vec1, &vec_wrong].map(|v| weights.evaluate(&embedding, v)))
            .collect::<Vec<_>>();

        // Commit to the correct polynomials
        let ds = DomainSeparator::protocol(&params)
            .session(&format!("Test at {}:{}", file!(), line!()))
            .instance(&Empty);
        let mut prover_state = ProverState::new_std(&ds);

        let witness1 = params.commit(&mut prover_state, &[&vec1]);
        let witness2 = params.commit(&mut prover_state, &[&vec2]);

        // Generate proof with mismatched polynomials
        // The prover will compute cross-terms using poly_wrong, not poly2
        let linear_form_refs = linear_forms
            .iter()
            .map(|l| l.as_ref() as &dyn LinearForm<EF>)
            .collect::<Vec<_>>();
        let (_evalpoint, _values) = params.prove(
            &mut prover_state,
            &[&vec1, &vec_wrong],
            &[&witness1, &witness2],
            &linear_form_refs,
            &evaluations,
        );

        // Verification should fail because the cross-terms don't match the commitment
        let proof = prover_state.proof();
        let mut verifier_state = VerifierState::new_std(&ds, &proof);

        let mut commitments = Vec::new();
        for _ in 0..num_polynomials {
            let parsed_commitment = params.receive_commitment(&mut verifier_state).unwrap();
            commitments.push(parsed_commitment);
        }

        let verify_result = params.verify(
            &mut verifier_state,
            &[&commitments[0], &commitments[1]],
            &linear_form_refs,
            &evaluations,
        );
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
        initial_folding_factor: usize,
        folding_factor: usize,
        num_points_per_poly: usize,
        num_witnesses: usize,
        batch_size: usize,
        unique_decoding: bool,
        pow_bits: usize,
    ) {
        let num_coeffs = 1 << num_variables;
        let mut rng = ark_std::test_rng();

        let whir_params = ProtocolParameters {
            security_level: 32,
            pow_bits,
            initial_folding_factor,
            folding_factor,
            unique_decoding,
            starting_log_inv_rate: 1,
            batch_size, // KEY: batch_size > 1
            hash_id: hash::SHA2,
        };

        let mut params = Config::<EF>::new(1 << num_variables, &whir_params);
        params.disable_pow();

        // Create weights for constraints
        let embedding = Basefield::new();

        // Create polynomials for each witness
        // Each witness will contain batch_size polynomials committed together
        let all_vectors: Vec<Vec<F>> = (0..num_witnesses * batch_size)
            .map(|i| vec![F::from((i + 1) as u64); num_coeffs])
            .collect::<Vec<_>>();
        let vec_refs = all_vectors.iter().map(|p| p.as_slice()).collect::<Vec<_>>();

        let mut linear_forms: Vec<Box<dyn Evaluate<Basefield<EF>>>> = Vec::new();
        for _ in 0..num_points_per_poly {
            linear_forms.push(Box::new(MultilinearExtension {
                point: MultilinearPoint::rand(&mut rng, num_variables).0,
            }));
        }
        // Add a linear constraint
        linear_forms.push(Box::new(Covector {
            deferred: false,
            vector: (0..1 << num_variables).map(EF::from).collect(),
        }));

        // Create evaluations for each constraint and polynomial
        let evaluations = linear_forms
            .iter()
            .flat_map(|linear_form| {
                vec_refs
                    .iter()
                    .map(|vec| linear_form.evaluate(&embedding, vec))
            })
            .collect::<Vec<_>>();

        // Set up domain separator
        let ds = DomainSeparator::protocol(&params)
            .session(&format!("Test at {}:{}", file!(), line!()))
            .instance(&Empty);
        let mut prover_state = ProverState::new_std(&ds);

        // Commit using commit_batch (stacks batch_size polynomials per witness)
        let mut witnesses = Vec::new();
        for witness_polys in vec_refs.chunks(batch_size) {
            let witness = params.commit(&mut prover_state, witness_polys);
            witnesses.push(witness);
        }
        let witness_refs = witnesses.iter().collect::<Vec<_>>();

        // Batch prove all witnesses together
        let linear_form_refs = linear_forms
            .iter()
            .map(|l| l.as_ref() as &dyn LinearForm<EF>)
            .collect::<Vec<_>>();
        let (_point, _evals) = params.prove(
            &mut prover_state,
            &vec_refs,
            &witness_refs,
            &linear_form_refs,
            &evaluations,
        );

        // Verify
        let proof = prover_state.proof();
        let mut verifier_state = VerifierState::new_std(&ds, &proof);

        let mut commitments = Vec::new();
        for _ in 0..num_witnesses {
            let commitment = params.receive_commitment(&mut verifier_state).unwrap();
            commitments.push(commitment);
        }
        let commitment_refs = commitments.iter().collect::<Vec<_>>();

        let verify_result = params.verify(
            &mut verifier_state,
            &commitment_refs,
            &linear_form_refs,
            &evaluations,
        );
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
                        folding_factor,
                        folding_factor,
                        1, // num_points_per_poly
                        num_witness,
                        batch_size,
                        false,
                        0, // pow_bits
                    );
                }
            }
        }
    }

    fn random_vector(num_coefficients: usize) -> Vec<F> {
        let mut store = Vec::<F>::with_capacity(num_coefficients);
        let mut rng = ark_std::rand::thread_rng();
        (0..num_coefficients).for_each(|_| store.push(F::rand(&mut rng)));
        store
    }

    /// Run a complete WHIR proof lifecycle: commit, prove, and verify.
    fn make_batched_whir_things(
        batch_size: usize,
        num_variables: usize,
        initial_folding_factor: usize,
        folding_factor: usize,
        num_points: usize,
        unique_decoding: bool,
        pow_bits: usize,
    ) {
        eprintln!("\n---------------------");
        eprintln!("Test parameters: ");
        eprintln!("  num_vectors     : {batch_size}");
        eprintln!("  num_variables   : {num_variables}");
        eprintln!("  initial_folding : {initial_folding_factor}");
        eprintln!("  folding_factor  : {folding_factor}");
        eprintln!("  num_points      : {num_points:?}");
        eprintln!("  unique_decoding : {unique_decoding:?}");
        eprintln!("  pow_bits        : {pow_bits}");

        // Number of coefficients in the multilinear polynomial (2^num_variables)
        let num_coeffs = 1 << num_variables;

        // Randomness source
        let mut rng = ark_std::test_rng();

        // Configure the WHIR protocol parameters
        let whir_params = ProtocolParameters {
            security_level: 32,
            pow_bits,
            initial_folding_factor,
            folding_factor,
            unique_decoding,
            starting_log_inv_rate: 1,
            batch_size,
            hash_id: hash::SHA2,
        };

        // Build global configuration from multivariate + protocol parameters
        let mut params = Config::new(1 << num_variables, &whir_params);
        params.disable_pow();

        let vectors: Vec<Vec<F>> = (0..batch_size).map(|_| random_vector(num_coeffs)).collect();
        let vec_refs = vectors.iter().map(|v| v.as_slice()).collect::<Vec<_>>();

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
        let batched_witness = params.commit(&mut prover_state, &vec_refs);

        // Create a weights matrix and evaluations for each polynomial
        let mut linear_forms: Vec<Box<dyn Evaluate<Basefield<F>>>> = Vec::new();
        for point in points {
            linear_forms.push(Box::new(MultilinearExtension {
                point: point.0.clone(),
            }));
        }
        linear_forms.push(Box::new(Covector {
            deferred: false,
            vector: (0..1 << num_variables).map(F::from).collect(),
        }));
        let values = linear_forms
            .iter()
            .flat_map(|linear_form| {
                vec_refs
                    .iter()
                    .map(|vec| linear_form.evaluate(params.embedding(), vec))
            })
            .collect::<Vec<_>>();

        // Generate a proof for the given statement and witness
        let weights_dyn_refs = linear_forms
            .iter()
            .map(|w| w.as_ref() as &dyn LinearForm<F>)
            .collect::<Vec<_>>();
        params.prove(
            &mut prover_state,
            &vec_refs,
            &[&batched_witness],
            &weights_dyn_refs,
            &values,
        );

        // Reconstruct verifier's view of the transcript using the IOPattern and prover's data
        let proof = prover_state.proof();
        let mut verifier_state = VerifierState::new_std(&ds, &proof);

        let commitment = params.receive_commitment(&mut verifier_state).unwrap();

        // Verify that the generated proof satisfies the statement
        assert!(params
            .verify(
                &mut verifier_state,
                &[&commitment],
                &weights_dyn_refs,
                &values
            )
            .is_ok());
    }

    #[test]
    fn test_batched_whir() {
        let folding_factors = [1, 4];
        let unique_decoding_options = [false, true];
        let num_points = [0, 2];
        let pow_bits = [0, 10];

        for folding_factor in folding_factors {
            let num_variables = (2 * folding_factor)..=3 * folding_factor;
            for num_variable in num_variables {
                for num_points in num_points {
                    for unique_decoding in unique_decoding_options {
                        for pow_bits in pow_bits {
                            for batch_size in 1..=4 {
                                make_batched_whir_things(
                                    batch_size,
                                    num_variable,
                                    folding_factor,
                                    folding_factor,
                                    num_points,
                                    unique_decoding,
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
