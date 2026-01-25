use ark_ff::FftField;
use spongefish::{Codec, Decoding, DuplexSpongeInterface, VerificationError, VerificationResult};

use super::{
    committer::reader::ParsedCommitment,
    parameters::{RoundConfig, WhirConfig},
    statement::{Constraint, Statement, Weights},
};
use crate::{
    algebra::poly_utils::{coeffs::CoefficientList, multilinear::MultilinearPoint},
    hash::Hash,
    protocols::{matrix_commit, sumcheck},
    transcript::{codecs::U64, ProverMessage, VerifierMessage, VerifierState},
    type_info::Type,
    utils::expand_randomness,
    whir::utils::get_challenge_stir_queries,
};

pub struct Verifier<'a, F: FftField> {
    params: &'a WhirConfig<F>,
}

impl<'a, F: FftField> Verifier<'a, F> {
    pub const fn new(params: &'a WhirConfig<F>) -> Self {
        Self { params }
    }

    /// Verify a WHIR proof.
    ///
    /// Returns the constraint evaluation point and the values of the deferred constraints.
    /// It is the callers responsibility to verify the deferred constraints.
    #[allow(clippy::too_many_lines)]
    pub fn verify<H>(
        &self,
        verifier_state: &mut VerifierState<'_, H>,
        parsed_commitment: &ParsedCommitment<F>,
        statement: &Statement<F>,
    ) -> VerificationResult<(MultilinearPoint<F>, Vec<F>)>
    where
        H: DuplexSpongeInterface,
        F: Codec<[H::U]>,
        u8: Decoding<[H::U]>,
        [u8; 32]: Decoding<[H::U]>,
        U64: Codec<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        // During the rounds we collect constraints, combination randomness, folding randomness
        // and we update the claimed sum of constraint evaluation.
        let mut round_constraints = Vec::new();
        let mut round_folding_randomness = Vec::new();
        let mut claimed_sum = F::ZERO;
        let mut prev_committer = &self.params.initial_matrix_committer;
        let mut prev_commitment = parsed_commitment.clone();

        // Optional initial sumcheck round
        if self.params.initial_statement {
            // Combine OODS and statement constraints to claimed_sum
            let constraints: Vec<_> = prev_commitment
                .oods_constraints()
                .into_iter()
                .chain(statement.constraints.iter().cloned())
                .collect();

            let combination_randomness =
                self.combine_constraints(verifier_state, &mut claimed_sum, &constraints)?;
            round_constraints.push((combination_randomness, constraints));

            // Initial sumcheck
            let config = sumcheck::Config {
                field: Type::new(),
                initial_size: 1 << self.params.folding_factor.at_round(0), // Not used
                rounds: vec![
                    sumcheck::RoundConfig {
                        pow: self.params.starting_folding_pow
                    };
                    self.params.folding_factor.at_round(0)
                ],
            };
            let folding_randomness = config.verify(verifier_state.inner_mut(), &mut claimed_sum)?;
            round_folding_randomness.push(folding_randomness);
        } else {
            assert_eq!(prev_commitment.ood_points.len(), 0);
            assert!(statement.constraints.is_empty());
            round_constraints.push((vec![], vec![]));

            let mut folding_randomness = vec![F::ZERO; self.params.folding_factor.at_round(0)];
            for randomness in &mut folding_randomness {
                *randomness = verifier_state.verifier_message();
            }
            round_folding_randomness.push(MultilinearPoint(folding_randomness));

            // PoW
            self.params
                .starting_folding_pow
                .verify(verifier_state.inner_mut())?;
        }

        for round_index in 0..self.params.n_rounds() {
            // Fetch round parameters from config
            let round_params = &self.params.round_parameters[round_index];

            // Receive commitment to the folded polynomial (likely encoded at higher expansion)
            let new_commitment = ParsedCommitment::receive(
                verifier_state,
                &round_params.matrix_committer,
                round_params.num_variables,
                round_params.ood_samples,
                1,
            )?;

            // Verify in-domain challenges on the previous commitment.
            let stir_constraints = self.verify_stir_challenges(
                round_index,
                verifier_state,
                round_params,
                prev_committer,
                &prev_commitment,
                round_folding_randomness.last().unwrap(),
            )?;

            // Add out-of-domain and in-domain constraints to claimed_sum
            let constraints: Vec<Constraint<F>> = new_commitment
                .oods_constraints()
                .into_iter()
                .chain(stir_constraints.into_iter())
                .collect();
            let combination_randomness =
                self.combine_constraints(verifier_state, &mut claimed_sum, &constraints)?;
            round_constraints.push((combination_randomness.clone(), constraints));

            let config = sumcheck::Config {
                field: Type::new(),
                initial_size: 1 << self.params.folding_factor.at_round(round_index + 1), // Not used
                rounds: vec![
                    sumcheck::RoundConfig {
                        pow: round_params.folding_pow
                    };
                    self.params.folding_factor.at_round(round_index + 1)
                ],
            };
            let folding_randomness = config.verify(verifier_state.inner_mut(), &mut claimed_sum)?;
            round_folding_randomness.push(folding_randomness);

            // Update round parameters
            prev_committer = &round_params.matrix_committer;
            prev_commitment = new_commitment;
        }

        // In the final round we receive the full polynomial instead of a commitment.
        let mut final_coefficients = vec![F::ZERO; 1 << self.params.final_sumcheck_rounds];
        for coeff in &mut final_coefficients {
            *coeff = verifier_state.prover_message()?;
        }
        let final_coefficients = CoefficientList::new(final_coefficients);

        // Verify in-domain challenges on the previous commitment.
        let stir_constraints = self.verify_stir_challenges(
            self.params.n_rounds(),
            verifier_state,
            &self.params.final_round_config(),
            prev_committer,
            &prev_commitment,
            round_folding_randomness.last().unwrap(),
        )?;

        // Verify stir constraints directly on final polynomial
        if !stir_constraints
            .iter()
            .all(|c| c.verify(&final_coefficients))
        {
            return Err(VerificationError);
        }

        let config = sumcheck::Config {
            field: Type::new(),
            initial_size: 1 << self.params.final_sumcheck_rounds, // Not used
            rounds: vec![
                sumcheck::RoundConfig {
                    pow: self.params.final_folding_pow
                };
                self.params.final_sumcheck_rounds
            ],
        };
        let final_sumcheck_randomness =
            config.verify(verifier_state.inner_mut(), &mut claimed_sum)?;
        round_folding_randomness.push(final_sumcheck_randomness.clone());

        // Compute folding randomness across all rounds.
        let folding_randomness = MultilinearPoint(
            round_folding_randomness
                .into_iter()
                .rev()
                .flat_map(|poly| poly.0.into_iter())
                .collect(),
        );

        // Compute evaluation of weights in folding randomness
        // Some weight computations can be deferred and will be returned for the caller
        // to verify.
        let deferred: Vec<F> = verifier_state.prover_hint_ark()?;
        let evaluation_of_weights =
            self.eval_constraints_poly(&round_constraints, &deferred, folding_randomness.clone());

        // Check the final sumcheck evaluation
        let final_value = final_coefficients.evaluate(&final_sumcheck_randomness);
        if claimed_sum != evaluation_of_weights * final_value {
            return Err(VerificationError);
        }

        Ok((folding_randomness, deferred))
    }

    /// Verify a batched WHIR proof for multiple commitments.
    ///
    /// This verifies a batch proof generated by `prove_batch`. The verifier reads the N×M
    /// constraint evaluation matrix from the transcript, samples the batching randomness γ,
    /// and reconstructs the combined constraints using RLC. Round 0 verifies openings in all
    /// N original commitment trees, while subsequent rounds verify the single batched polynomial.
    ///
    /// Returns the constraint evaluation point and values of deferred constraints.
    #[allow(clippy::too_many_lines)]
    pub fn verify_batch<H>(
        &self,
        verifier_state: &mut VerifierState<'_, H>,
        parsed_commitments: &[ParsedCommitment<F>],
        statements: &[Statement<F>],
    ) -> VerificationResult<(MultilinearPoint<F>, Vec<F>)>
    where
        H: DuplexSpongeInterface,
        F: Codec<[H::U]>,
        u8: Decoding<[H::U]>,
        [u8; 32]: Decoding<[H::U]>,
        U64: Codec<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        assert!(!parsed_commitments.is_empty());
        assert_eq!(parsed_commitments.len(), statements.len());

        // Step 1: Read the N×M constraint evaluation matrix from the transcript

        // Collect all constraint weights to determine matrix dimensions
        let mut all_constraints_info: Vec<(Weights<F>, bool)> = Vec::new();

        // OOD constraints from each commitment
        for commitment in parsed_commitments {
            for point in &commitment.ood_points {
                let ml_point = MultilinearPoint::expand_from_univariate(
                    *point,
                    self.params.mv_parameters.num_variables,
                );
                all_constraints_info.push((Weights::evaluation(ml_point), false));
            }
        }

        // Statement constraints
        for statement in statements {
            for constraint in &statement.constraints {
                all_constraints_info
                    .push((constraint.weights.clone(), constraint.defer_evaluation));
            }
        }

        // Read the N×M evaluation matrix from transcript
        let num_polynomials = parsed_commitments.len();
        let num_constraints = all_constraints_info.len();
        let mut constraint_evals_matrix = Vec::with_capacity(num_polynomials);

        for _ in 0..num_polynomials {
            let mut poly_evals = vec![F::ZERO; num_constraints];
            for eval in &mut poly_evals {
                *eval = verifier_state.prover_message()?;
            }
            constraint_evals_matrix.push(poly_evals);
        }

        // Step 2: Sample batching randomness γ (cryptographically bound to matrix)
        let batching_randomness: F = verifier_state.verifier_message();

        let mut round_constraints = Vec::new();
        let mut round_folding_randomness = Vec::new();
        let mut claimed_sum = F::ZERO;

        // Step 3: Reconstruct combined constraints using RLC of the evaluation matrix
        // For each constraint j: combined_eval[j] = Σᵢ γⁱ·eval[i][j]
        if self.params.initial_statement {
            let mut all_constraints = Vec::new();

            for (constraint_idx, (weights, defer_evaluation)) in
                all_constraints_info.into_iter().enumerate()
            {
                let mut combined_eval = F::ZERO;
                let mut pow = F::ONE;
                for poly_evals in &constraint_evals_matrix {
                    combined_eval += pow * poly_evals[constraint_idx];
                    pow *= batching_randomness;
                }

                all_constraints.push(Constraint {
                    weights,
                    sum: combined_eval,
                    defer_evaluation,
                });
            }

            let combination_randomness =
                self.combine_constraints(verifier_state, &mut claimed_sum, &all_constraints)?;
            round_constraints.push((combination_randomness, all_constraints));

            // Initial sumcheck on the combined constraints
            let config = sumcheck::Config {
                field: Type::new(),
                initial_size: 1 << self.params.folding_factor.at_round(0), // Not used
                rounds: vec![
                    sumcheck::RoundConfig {
                        pow: self.params.starting_folding_pow
                    };
                    self.params.folding_factor.at_round(0)
                ],
            };
            let folding_randomness = config.verify(verifier_state.inner_mut(), &mut claimed_sum)?;
            round_folding_randomness.push(folding_randomness);
        } else {
            for commitment in parsed_commitments {
                assert_eq!(commitment.ood_points.len(), 0);
            }
            for statement in statements {
                assert!(statement.constraints.is_empty());
            }
            round_constraints.push((vec![], vec![]));

            let mut folding_randomness = vec![F::ZERO; self.params.folding_factor.at_round(0)];
            for randomness in &mut folding_randomness {
                *randomness = verifier_state.verifier_message();
            }
            round_folding_randomness.push(MultilinearPoint(folding_randomness));

            self.params
                .starting_folding_pow
                .verify(verifier_state.inner_mut())?;
        }

        // Round 0: Batch-specific verification
        //
        // Unlike regular verification, Round 0 for batch verification:
        // 1. Receives commitment to the batched folded polynomial (for Round 1+)
        // 2. Verifies STIR queries against all N original commitment trees
        // 3. RLC-combines the N query answers to check consistency with batched polynomial
        //
        // This ensures the batched polynomial correctly combines the N original polynomials
        // while maintaining verifiability against the original commitments.
        let round_params = &self.params.round_parameters[0];

        // Receive commitment to the batched folded polynomial
        let mut prev_committer = &round_params.matrix_committer;
        let mut prev_commitment = ParsedCommitment::receive(
            verifier_state,
            &round_params.matrix_committer,
            round_params.num_variables,
            round_params.ood_samples,
            1,
        )?;

        // Verify STIR challenges on N original witness trees
        round_params.pow.verify(verifier_state.inner_mut())?;

        let stir_challenges_indexes = get_challenge_stir_queries(
            verifier_state,
            round_params.domain_size,
            round_params.folding_factor,
            round_params.num_queries,
        );

        // Verify Merkle openings in all N original commitment trees
        let mut all_answers: Vec<Vec<F>> = Vec::with_capacity(parsed_commitments.len());
        for commitment in parsed_commitments {
            let answers: Vec<F> = verifier_state.prover_hint_ark()?;
            self.params.initial_matrix_committer.verify(
                verifier_state,
                commitment.matrix_commitment,
                &stir_challenges_indexes,
                &answers,
            )?;
            all_answers.push(answers);
        }

        // RLC-combine the N query answers: combined[j] = Σᵢ γⁱ·answers[i][j]
        let fold_size = 1 << round_params.folding_factor;
        let leaf_size = fold_size * self.params.batch_size;
        let rlc_answers: Vec<Vec<F>> = (0..stir_challenges_indexes.len())
            .map(|query_idx| {
                let mut combined = vec![F::ZERO; fold_size];
                let mut pow = F::ONE;
                for (commitment, witness_answers) in parsed_commitments.iter().zip(&all_answers) {
                    let stacked_answer =
                        &witness_answers[query_idx * leaf_size..(query_idx + 1) * leaf_size];

                    // First, internally reduce stacked leaf using commitment's batching_randomness
                    let mut internal_pow = F::ONE;
                    for poly_idx in 0..self.params.batch_size {
                        let start = poly_idx * fold_size;
                        for j in 0..fold_size {
                            combined[j] += pow * internal_pow * stacked_answer[start + j];
                        }
                        internal_pow *= commitment.batching_randomness;
                    }

                    pow *= batching_randomness;
                }
                combined
            })
            .collect();

        // Compute STIR constraints using the RLC'd answers
        let folds: Vec<F> = rlc_answers
            .into_iter()
            .map(|answers| CoefficientList::new(answers).evaluate(&round_folding_randomness[0]))
            .collect();

        let stir_constraints: Vec<Constraint<F>> = stir_challenges_indexes
            .iter()
            .map(|&index| round_params.exp_domain_gen.pow([index as u64]))
            .zip(&folds)
            .map(|(point, &value)| Constraint {
                weights: Weights::univariate(point, round_params.num_variables),
                sum: value,
                defer_evaluation: false,
            })
            .collect();

        // Add batched commitment's OOD constraints and STIR constraints
        let constraints: Vec<Constraint<F>> = prev_commitment
            .oods_constraints()
            .into_iter()
            .chain(stir_constraints)
            .collect();

        let combination_randomness =
            self.combine_constraints(verifier_state, &mut claimed_sum, &constraints)?;
        round_constraints.push((combination_randomness, constraints));

        let config = sumcheck::Config {
            field: Type::new(),
            initial_size: 1 << self.params.folding_factor.at_round(1), // Not used
            rounds: vec![
                sumcheck::RoundConfig {
                    pow: round_params.folding_pow
                };
                self.params.folding_factor.at_round(1)
            ],
        };
        let folding_randomness = config.verify(verifier_state.inner_mut(), &mut claimed_sum)?;
        round_folding_randomness.push(folding_randomness);

        // Rounds 1+: Standard WHIR verification on the single batched polynomial
        // From here on, the protocol is identical to regular verification
        for round_index in 1..self.params.n_rounds() {
            let round_params = &self.params.round_parameters[round_index];

            let new_commitment = ParsedCommitment::receive(
                verifier_state,
                &round_params.matrix_committer,
                round_params.num_variables,
                round_params.ood_samples,
                1,
            )?;

            let stir_constraints = self.verify_stir_challenges(
                round_index,
                verifier_state,
                round_params,
                prev_committer,
                &prev_commitment,
                round_folding_randomness.last().unwrap(),
            )?;

            let constraints: Vec<Constraint<F>> = new_commitment
                .oods_constraints()
                .into_iter()
                .chain(stir_constraints.into_iter())
                .collect();
            let combination_randomness =
                self.combine_constraints(verifier_state, &mut claimed_sum, &constraints)?;
            round_constraints.push((combination_randomness.clone(), constraints));

            let config = sumcheck::Config {
                field: Type::new(),
                initial_size: 1 << self.params.folding_factor.at_round(round_index + 1), // Not used
                rounds: vec![
                    sumcheck::RoundConfig {
                        pow: round_params.folding_pow
                    };
                    self.params.folding_factor.at_round(round_index + 1)
                ],
            };
            let folding_randomness = config.verify(verifier_state.inner_mut(), &mut claimed_sum)?;
            round_folding_randomness.push(folding_randomness);

            prev_committer = &round_params.matrix_committer;
            prev_commitment = new_commitment;
        }

        // Final round (same as regular verify)
        let mut final_coefficients = vec![F::ZERO; 1 << self.params.final_sumcheck_rounds];
        for coeff in &mut final_coefficients {
            *coeff = verifier_state.prover_message()?;
        }
        let final_coefficients = CoefficientList::new(final_coefficients);

        let stir_constraints = self.verify_stir_challenges(
            self.params.n_rounds(),
            verifier_state,
            &self.params.final_round_config(),
            prev_committer,
            &prev_commitment,
            round_folding_randomness.last().unwrap(),
        )?;

        if !stir_constraints
            .iter()
            .all(|c| c.verify(&final_coefficients))
        {
            return Err(VerificationError);
        }

        let config = sumcheck::Config {
            field: Type::new(),
            initial_size: 1 << self.params.final_sumcheck_rounds, // Not used
            rounds: vec![
                sumcheck::RoundConfig {
                    pow: self.params.final_folding_pow
                };
                self.params.final_sumcheck_rounds
            ],
        };
        let final_sumcheck_randomness =
            config.verify(verifier_state.inner_mut(), &mut claimed_sum)?;
        round_folding_randomness.push(final_sumcheck_randomness.clone());

        // Compute folding randomness across all rounds
        let folding_randomness = MultilinearPoint(
            round_folding_randomness
                .into_iter()
                .rev()
                .flat_map(|poly| poly.0.into_iter())
                .collect(),
        );

        // Compute evaluation of weights in folding randomness
        let deferred: Vec<F> = verifier_state.prover_hint_ark()?;
        let evaluation_of_weights =
            self.eval_constraints_poly(&round_constraints, &deferred, folding_randomness.clone());

        // Check the final sumcheck evaluation
        let final_value = final_coefficients.evaluate(&final_sumcheck_randomness);
        if claimed_sum != evaluation_of_weights * final_value {
            return Err(VerificationError);
        }

        Ok((folding_randomness, deferred))
    }

    /// Create a random linear combination of constraints and add it to the claim.
    /// Returns the randomness used.
    pub fn combine_constraints<H>(
        &self,
        verifier_state: &mut VerifierState<'_, H>,
        claimed_sum: &mut F,
        constraints: &[Constraint<F>],
    ) -> VerificationResult<Vec<F>>
    where
        H: DuplexSpongeInterface,
        F: Codec<[H::U]>,
        u8: Decoding<[H::U]>,
        [u8; 32]: Decoding<[H::U]>,
        U64: Codec<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        let combination_randomness_gen = verifier_state.verifier_message();
        let combination_randomness =
            expand_randomness(combination_randomness_gen, constraints.len());
        *claimed_sum += constraints
            .iter()
            .zip(&combination_randomness)
            .map(|(c, rand)| c.sum * rand)
            .sum::<F>();

        Ok(combination_randomness)
    }

    /// Verify a STIR challenges against a commitment and return the constraints.
    pub fn verify_stir_challenges<H>(
        &self,
        round_index: usize,
        verifier_state: &mut VerifierState<'_, H>,
        params: &RoundConfig<F>,
        committer: &matrix_commit::Config<F>,
        commitment: &ParsedCommitment<F>,
        folding_randomness: &MultilinearPoint<F>,
    ) -> VerificationResult<Vec<Constraint<F>>>
    where
        H: DuplexSpongeInterface,
        F: Codec<[H::U]>,
        u8: Decoding<[H::U]>,
        [u8; 32]: Decoding<[H::U]>,
        U64: Codec<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        params.pow.verify(verifier_state.inner_mut())?;

        let stir_challenges_indexes = get_challenge_stir_queries(
            verifier_state,
            params.domain_size,
            params.folding_factor,
            params.num_queries,
        );

        // Always open against the single batched commitment
        let mut answers: Vec<F> = verifier_state.prover_hint_ark()?;
        committer.verify(
            verifier_state,
            commitment.matrix_commitment,
            &stir_challenges_indexes,
            &answers,
        )?;

        // If this is the first round and batching > 1, RLC per leaf to fold_size
        if round_index == 0 && self.params.batch_size > 1 {
            let fold_size = 1 << params.folding_factor;
            answers = crate::whir::utils::rlc_batched_leaves(
                &answers,
                fold_size,
                self.params.batch_size,
                commitment.batching_randomness,
            );
        }

        // Compute STIR Constraints
        let folds: Vec<F> = answers
            .chunks_exact(1 << params.folding_factor)
            .map(|answers| CoefficientList::new(answers.to_vec()).evaluate(folding_randomness))
            .collect();

        let stir_constraints = stir_challenges_indexes
            .iter()
            .map(|&index| params.exp_domain_gen.pow([index as u64]))
            .zip(&folds)
            .map(|(point, &value)| Constraint {
                weights: Weights::univariate(point, params.num_variables),
                sum: value,
                defer_evaluation: false,
            })
            .collect();

        Ok(stir_constraints)
    }

    /// Evaluate the random linear combination of constraints in `point`.
    fn eval_constraints_poly(
        &self,
        constraints: &[(Vec<F>, Vec<Constraint<F>>)],
        deferred: &[F],
        mut point: MultilinearPoint<F>,
    ) -> F {
        let mut num_variables = self.params.mv_parameters.num_variables;
        let mut deferred = deferred.iter().copied();
        let mut value = F::ZERO;

        for (round, (randomness, constraints)) in constraints.iter().enumerate() {
            assert_eq!(randomness.len(), constraints.len());
            if round > 0 {
                num_variables -= self.params.folding_factor.at_round(round - 1);
                point = MultilinearPoint(point.0[..num_variables].to_vec());
            }
            value += constraints
                .iter()
                .zip(randomness)
                .map(|(constraint, &randomness)| {
                    let value = if constraint.defer_evaluation {
                        deferred.next().unwrap()
                    } else {
                        constraint.weights.compute(&point)
                    };
                    value * randomness
                })
                .sum::<F>();
        }
        value
    }
}
