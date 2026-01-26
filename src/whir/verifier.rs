use ark_crypto_primitives::merkle_tree::Config;
use ark_ff::FftField;
use spongefish::{Codec, Decoding, DuplexSpongeInterface, VerificationError, VerificationResult};

use super::{
    committer::reader::ParsedCommitment,
    parameters::{RoundConfig, WhirConfig},
    statement::{Constraint, Statement, Weights},
};
use crate::{
    poly_utils::{coeffs::CoefficientList, multilinear::MultilinearPoint},
    sumcheck,
    transcript::{codecs::U64, FieldConfig, ProverMessage, VerifierMessage, VerifierState},
    utils::expand_randomness,
    whir::{merkle, prover::RootPath, utils::get_challenge_stir_queries},
};

pub struct Verifier<'a, F, MerkleConfig>
where
    F: FftField,
    MerkleConfig: Config,
{
    params: &'a WhirConfig<F, MerkleConfig>,
    merkle_state: merkle::VerifierMerkleState<'a, MerkleConfig>,
}

impl<'a, F, MerkleConfig> Verifier<'a, F, MerkleConfig>
where
    F: FftField,
    MerkleConfig: Config<Leaf = [F]>,
{
    pub const fn new(params: &'a WhirConfig<F, MerkleConfig>) -> Self {
        Self {
            params,
            merkle_state: merkle::VerifierMerkleState::new(
                params.merkle_proof_strategy,
                &params.leaf_hash_params,
                &params.two_to_one_params,
            ),
        }
    }

    /// Verify a WHIR proof.
    ///
    /// Returns the constraint evaluation point and the values of the deferred constraints.
    /// It is the callers responsibility to verify the deferred constraints.
    #[allow(clippy::too_many_lines)]
    pub fn verify<H>(
        &self,
        verifier_state: &mut VerifierState<'_, H>,
        parsed_commitment: &ParsedCommitment<F, MerkleConfig::InnerDigest>,
        statement: &Statement<F>,
    ) -> VerificationResult<(MultilinearPoint<F>, Vec<F>)>
    where
        H: DuplexSpongeInterface,
        F: Codec<[H::U]>,
        u8: Decoding<[H::U]>,
        [u8; 32]: Decoding<[H::U]>,
        U64: Codec<[H::U]>,
        MerkleConfig::InnerDigest: ProverMessage<[H::U]>,
    {
        // During the rounds we collect constraints, combination randomness, folding randomness
        // and we update the claimed sum of constraint evaluation.
        let mut round_constraints = Vec::new();
        let mut round_folding_randomness = Vec::new();
        let mut claimed_sum = F::ZERO;
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
                field: FieldConfig::new(),
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
            let new_commitment =
                ParsedCommitment::<F, MerkleConfig::InnerDigest>::parse::<H, MerkleConfig>(
                    verifier_state.inner_mut(),
                    round_params.num_variables,
                    round_params.ood_samples,
                )?;

            // Verify in-domain challenges on the previous commitment.
            let stir_constraints = self.verify_stir_challenges(
                round_index,
                verifier_state,
                round_params,
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
                field: FieldConfig::new(),
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
            field: FieldConfig::new(),
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
        parsed_commitments: &[ParsedCommitment<F, MerkleConfig::InnerDigest>],
        statements: &[Statement<F>],
    ) -> VerificationResult<(MultilinearPoint<F>, Vec<F>)>
    where
        H: DuplexSpongeInterface,
        F: Codec<[H::U]>,
        u8: Decoding<[H::U]>,
        [u8; 32]: Decoding<[H::U]>,
        U64: Codec<[H::U]>,
        MerkleConfig::InnerDigest: ProverMessage<[H::U]>,
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
                field: FieldConfig::new(),
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
        let mut prev_commitment =
            ParsedCommitment::<F, MerkleConfig::InnerDigest>::parse::<H, MerkleConfig>(
                verifier_state.inner_mut(),
                round_params.num_variables,
                round_params.ood_samples,
            )?;

        // Verify STIR challenges on N original witness trees
        round_params.pow.verify(verifier_state.inner_mut())?;

        let stir_challenges_indexes = get_challenge_stir_queries(
            verifier_state,
            round_params.domain_size,
            round_params.folding_factor,
            round_params.num_queries,
            &self.params.deduplication_strategy,
        );

        // Verify Merkle openings in all N original commitment trees
        let mut all_answers = Vec::with_capacity(parsed_commitments.len());
        for commitment in parsed_commitments {
            let answers = self.verify_merkle_proof(
                verifier_state,
                &commitment.root,
                &stir_challenges_indexes,
            )?;
            all_answers.push(answers);
        }

        // RLC-combine the N query answers: combined[j] = Σᵢ γⁱ·answers[i][j]
        let fold_size = 1 << round_params.folding_factor;
        let rlc_answers: Vec<Vec<F>> = (0..stir_challenges_indexes.len())
            .map(|query_idx| {
                let mut combined = vec![F::ZERO; fold_size];
                let mut pow = F::ONE;
                for (commitment, witness_answers) in parsed_commitments.iter().zip(&all_answers) {
                    let stacked_answer = &witness_answers[query_idx];

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
            field: FieldConfig::new(),
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

            let new_commitment =
                ParsedCommitment::<F, MerkleConfig::InnerDigest>::parse::<H, MerkleConfig>(
                    verifier_state.inner_mut(),
                    round_params.num_variables,
                    round_params.ood_samples,
                )?;

            let stir_constraints = self.verify_stir_challenges(
                round_index,
                verifier_state,
                round_params,
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
                field: FieldConfig::new(),
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
            field: FieldConfig::new(),
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

    pub fn verify_batch_prefold<H>(
        &self,
        verifier_state: &mut VerifierState<'_, H>,
        parsed_commitments: &[ParsedCommitment<F, MerkleConfig::InnerDigest>],
        statements: &[Statement<F>],
    ) -> VerificationResult<(MultilinearPoint<F>, Vec<F>)>
    where
        H: DuplexSpongeInterface,
        F: Codec<[H::U]>,
        u8: Decoding<[H::U]>,
        [u8; 32]: Decoding<[H::U]>,
        U64: Codec<[H::U]>,
        MerkleConfig::InnerDigest: ProverMessage<[H::U]>,
    {
        // Validation
        assert_eq!(parsed_commitments.len(), 2);
        assert_eq!(statements.len(), 2);

        let num_vars_0 = parsed_commitments[0].num_variables;
        let num_vars_1 = parsed_commitments[1].num_variables;

        // If the two witnesses have the same arity, we can just verify the batch as standard
        if num_vars_0 == num_vars_1 {
            return Ok(self.verify_batch(verifier_state, parsed_commitments, statements)?);
        }

        // Determine which witness is larger (has more variables); prefold that one.
        let (small_idx, big_idx, num_vars_small, num_vars_big) = if num_vars_0 < num_vars_1 {
            (0usize, 1usize, num_vars_0, num_vars_1)
        } else {
            (1usize, 0usize, num_vars_1, num_vars_0)
        };

        assert_eq!(
            num_vars_big,
            num_vars_small + 1,
            "PreFold requires arity difference of exactly 1"
        );

        let statement_g = &statements[big_idx];

        // --- Phase 0: Run sumcheck on g's constraints to derive prefold randomness ---
        //
        // If g has constraints, verify the sumcheck on them. The sumcheck produces randomness
        // (r_0, ..., r_n) where n = num_vars_big - 1. We use r_n (the last coordinate,
        // corresponding to the high-indexed variable) as the prefold randomness α.
        //
        // The sumcheck claim g(r_0, ..., r_n) = v becomes g'(r_0, ..., r_{n-1}) = v
        // after folding at α = r_n. This is added as a constraint on g'.
        //
        // If g has no constraints, we just sample α directly.
        let (prefold_randomness, g_sumcheck_claim, g_sumcheck_point) =
            if !statement_g.constraints.is_empty() {
                // Verify sumcheck on g's constraints
                let combination_randomness_gen: F = verifier_state.verifier_message();

                // Compute the initial claimed sum from g's constraints
                let mut claimed_sum = F::ZERO;
                let mut pow = F::ONE;
                for constraint in &statement_g.constraints {
                    claimed_sum += pow * constraint.sum;
                    pow *= combination_randomness_gen;
                }

                let sumcheck_config = sumcheck::Config {
                    field: FieldConfig::new(),
                    initial_size: 1 << num_vars_big, // Not used
                    rounds: vec![
                        sumcheck::RoundConfig {
                            pow: self.params.starting_folding_pow
                        };
                        num_vars_big // One round per variable of g
                    ],
                };

                // Run sumcheck verification, getting randomness (r_0, ..., r_n)
                let sumcheck_randomness =
                    sumcheck_config.verify(verifier_state.inner_mut(), &mut claimed_sum)?;

                // The last coordinate r_n becomes the prefold randomness α
                // (fold() folds the high-indexed/last variable)
                let alpha = *sumcheck_randomness.0.last().unwrap();
                let prefold_randomness = MultilinearPoint(vec![alpha]);

                // The remaining coordinates (r_0, ..., r_{n-1}) form the evaluation point for g'
                let g_prime_point =
                    MultilinearPoint(sumcheck_randomness.0[..num_vars_small].to_vec());

                // After sumcheck, claimed_sum should be W(r) * g(r)
                // We store the expected g'(r_0, ..., r_{n-1}) value, which will be verified later
                // through the combined constraint on h = f + γ*g'

                // Compute the weight evaluation at the sumcheck point
                // For Evaluation constraints, W(r) = eq(z, r)
                let mut weight_eval = F::ZERO;
                let mut pow = F::ONE;
                for constraint in &statement_g.constraints {
                    let w_eval = match &constraint.weights {
                        Weights::Evaluation { point } => point.eq_poly_outside(&sumcheck_randomness),
                        Weights::Linear { weight } | Weights::Geometric { weight, .. } => {
                            weight.evaluate(&sumcheck_randomness)
                        }
                    };
                    weight_eval += pow * w_eval;
                    pow *= combination_randomness_gen;
                }

                // claimed_sum = W(r) * g(r) => g(r) = claimed_sum / W(r)
                // g(r) = g'(r_0, ..., r_{n-1}) after folding
                let g_prime_value = claimed_sum * weight_eval.inverse().unwrap();

                (
                    prefold_randomness,
                    Some(g_prime_value),
                    Some(g_prime_point),
                )
            } else {
                // No constraints on g, just sample α directly
                let alpha: F = verifier_state.verifier_message();

                if !self.params.starting_folding_pow.difficulty.is_zero() {
                    self.params
                        .starting_folding_pow
                        .verify(verifier_state.inner_mut())?;
                }

                (MultilinearPoint(vec![alpha]), None, None)
            };

        // --- Phase 1: Parse commitment to g' (committed under the SMALL config) ---
        let g_folded_commitment =
            ParsedCommitment::<F, MerkleConfig::InnerDigest>::parse::<H, MerkleConfig>(
                verifier_state.inner_mut(),
                num_vars_small,
                self.params.committment_ood_samples,
            )?;

        if !self.params.starting_folding_pow.difficulty.is_zero() {
            self.params
                .starting_folding_pow
                .verify(verifier_state.inner_mut())?;
        }

        // Verify consistency via STIR queries on ORIGINAL g
        let g_domain_size = self
            .params
            .starting_domain
            .size()
            .checked_shl((num_vars_big - num_vars_small) as u32)
            .expect("domain size overflow in prefold");
        let g_folding_factor = num_vars_big - num_vars_small;
        let stir_indexes = get_challenge_stir_queries(
            verifier_state,
            g_domain_size,
            g_folding_factor,
            self.params.round_parameters[0].num_queries,
            &self.params.deduplication_strategy,
        );

        let original_g_answers: Vec<Vec<F>> = verifier_state.prover_hint_ark()?;
        self.merkle_state.read_and_verify_proof(
            verifier_state,
            &stir_indexes,
            &parsed_commitments[big_idx].root,
            original_g_answers.iter().map(|v| v.as_slice()),
        )?;

        let g_folded_stir_evals: Vec<F> = verifier_state.prover_hint_ark()?;
        for (answer_chunk, &expected_folded) in original_g_answers.iter().zip(&g_folded_stir_evals)
        {
            // `answer_chunk` is a stacked leaf when internal batching is enabled.
            // Reduce it using g's batching randomness, then fold the 2-point chunk at α.
            let fold_size = 1 << g_folding_factor;
            let mut reduced = vec![F::ZERO; fold_size];
            let mut internal_pow = F::ONE;
            for poly_idx in 0..self.params.batch_size {
                let start = poly_idx * fold_size;
                for j in 0..fold_size {
                    reduced[j] += internal_pow * answer_chunk[start + j];
                }
                internal_pow *= parsed_commitments[big_idx].batching_randomness;
            }

            let actual_folded = CoefficientList::new(reduced).evaluate(&prefold_randomness);
            if actual_folded != expected_folded {
                return Err(VerificationError);
            }
        }

        // --- Phase 2: Read evaluation matrix for (f, g') BEFORE sampling γ ---
        //
        // Matrix columns are:
        //  - OOD points from f commitment (arity n)
        //  - OOD points from g' commitment (arity n)
        //  - Statement constraints on f (arity n)
        //  - Sumcheck-derived constraint on g' (if g had constraints)
        //
        // Note: g's original constraints (arity n+1) were handled by sumcheck in Phase 0.
        // The sumcheck claim becomes a single evaluation constraint on g' at arity n.
        let mut all_constraints_info: Vec<(Weights<F>, bool)> = Vec::new();

        // OOD constraints from f (arity n)
        for point in &parsed_commitments[small_idx].ood_points {
            let ml_point = MultilinearPoint::expand_from_univariate(*point, num_vars_small);
            all_constraints_info.push((Weights::evaluation(ml_point), false));
        }

        // OOD constraints from g' (arity n)
        for point in &g_folded_commitment.ood_points {
            let ml_point = MultilinearPoint::expand_from_univariate(*point, num_vars_small);
            all_constraints_info.push((Weights::evaluation(ml_point), false));
        }

        // Statement constraints on f (the smaller polynomial)
        for constraint in &statements[small_idx].constraints {
            all_constraints_info
                .push((constraint.weights.clone(), constraint.defer_evaluation));
        }

        // Sumcheck-derived constraint on g' (from g's original constraints)
        if let Some(ref point) = g_sumcheck_point {
            all_constraints_info.push((Weights::evaluation(point.clone()), false));
        }

        let num_constraints = all_constraints_info.len();
        let mut constraint_evals_matrix = Vec::with_capacity(2);
        for _ in 0..2 {
            let mut row = vec![F::ZERO; num_constraints];
            for eval in &mut row {
                *eval = verifier_state.prover_message()?;
            }
            constraint_evals_matrix.push(row);
        }

        // --- Phase 3: Sample batching randomness γ ---
        let batching_randomness: F = verifier_state.verifier_message();

        // --- Phase 4: Build combined constraints at common arity (n) using the matrix ---
        let mut all_constraints = Vec::new();
        for (constraint_idx, (weights, defer_evaluation)) in
            all_constraints_info.into_iter().enumerate()
        {
            let combined_eval = constraint_evals_matrix[0][constraint_idx]
                + batching_randomness * constraint_evals_matrix[1][constraint_idx];
            all_constraints.push(Constraint {
                weights,
                sum: combined_eval,
                defer_evaluation,
            });
        }

        // Verify the sumcheck-derived constraint on g'
        // The combined sum for the g' constraint should be:
        //   f(point) + γ * g'(point) = matrix[0][idx] + γ * matrix[1][idx]
        // where matrix[1][idx] = g'(point) should equal the claimed sumcheck value
        if let Some(g_prime_value) = g_sumcheck_claim {
            // The last constraint is the sumcheck-derived one
            let idx = num_constraints - 1;
            let reported_g_prime_eval = constraint_evals_matrix[1][idx];
            if reported_g_prime_eval != g_prime_value {
                return Err(VerificationError);
            }
        }

        let mut round_constraints = Vec::new();
        let mut round_folding_randomness = Vec::new();
        let mut claimed_sum = F::ZERO;

        // Initial sumcheck on the combined constraints (identical to standard batch)
        let combination_randomness =
            self.combine_constraints(verifier_state, &mut claimed_sum, &all_constraints)?;
        round_constraints.push((combination_randomness, all_constraints));
        let folding_randomness = sumcheck::Config {
            field: FieldConfig::new(),
            initial_size: 1 << self.params.folding_factor.at_round(0), // Not used
            rounds: vec![
                sumcheck::RoundConfig {
                    pow: self.params.starting_folding_pow
                };
                self.params.folding_factor.at_round(0)
            ],
        }
        .verify(verifier_state.inner_mut(), &mut claimed_sum)?;
        round_folding_randomness.push(folding_randomness);

        // --- Round 0: batch-specific verification against BOTH original commitment trees (f and g') ---
        let round_params = &self.params.round_parameters[0];

        // Receive commitment to the batched folded polynomial
        let mut prev_commitment =
            ParsedCommitment::<F, MerkleConfig::InnerDigest>::parse::<H, MerkleConfig>(
                verifier_state.inner_mut(),
                round_params.num_variables,
                round_params.ood_samples,
            )?;

        round_params.pow.verify(verifier_state.inner_mut())?;

        let stir_challenges_indexes = get_challenge_stir_queries(
            verifier_state,
            round_params.domain_size,
            round_params.folding_factor,
            round_params.num_queries,
            &self.params.deduplication_strategy,
        );

        // Verify Merkle openings in f and g' commitment trees
        let answers_f = self.verify_merkle_proof(
            verifier_state,
            &parsed_commitments[small_idx].root,
            &stir_challenges_indexes,
        )?;
        let answers_g = self.verify_merkle_proof(
            verifier_state,
            &g_folded_commitment.root,
            &stir_challenges_indexes,
        )?;

        let fold_size = 1 << round_params.folding_factor;
        let rlc_answers: Vec<Vec<F>> = (0..stir_challenges_indexes.len())
            .map(|query_idx| {
                let mut combined = vec![F::ZERO; fold_size];

                // f contribution (internally reduce stacked leaf using η_f if batch_size>1)
                let mut internal_pow = F::ONE;
                for poly_idx in 0..self.params.batch_size {
                    let start = poly_idx * fold_size;
                    for j in 0..fold_size {
                        combined[j] += internal_pow * answers_f[query_idx][start + j];
                    }
                    internal_pow *= parsed_commitments[small_idx].batching_randomness;
                }

                // g' contribution:
                // - prover commits to g' with a stacked-leaf layout of size fold_size * batch_size,
                //   where only the first chunk is non-zero.
                // - g_folded_commitment.batching_randomness is 0, so the same reduction logic works.
                let mut internal_pow = F::ONE;
                for poly_idx in 0..self.params.batch_size {
                    let start = poly_idx * fold_size;
                    for j in 0..fold_size {
                        combined[j] +=
                            batching_randomness * internal_pow * answers_g[query_idx][start + j];
                    }
                    internal_pow *= g_folded_commitment.batching_randomness;
                }
                combined
            })
            .collect();

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

        let constraints: Vec<Constraint<F>> = prev_commitment
            .oods_constraints()
            .into_iter()
            .chain(stir_constraints)
            .collect();

        let combination_randomness =
            self.combine_constraints(verifier_state, &mut claimed_sum, &constraints)?;
        round_constraints.push((combination_randomness, constraints));

        let folding_randomness = sumcheck::Config {
            field: FieldConfig::new(),
            initial_size: 1 << self.params.folding_factor.at_round(1), // Not used
            rounds: vec![
                sumcheck::RoundConfig {
                    pow: round_params.folding_pow
                };
                self.params.folding_factor.at_round(1)
            ],
        }
        .verify(verifier_state.inner_mut(), &mut claimed_sum)?;
        round_folding_randomness.push(folding_randomness);

        // Rounds 1+: Standard WHIR verification on the single batched polynomial
        for round_index in 1..self.params.n_rounds() {
            let round_params = &self.params.round_parameters[round_index];

            let new_commitment =
                ParsedCommitment::<F, MerkleConfig::InnerDigest>::parse::<H, MerkleConfig>(
                    verifier_state.inner_mut(),
                    round_params.num_variables,
                    round_params.ood_samples,
                )?;

            let stir_constraints = self.verify_stir_challenges(
                round_index,
                verifier_state,
                round_params,
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
            round_constraints.push((combination_randomness, constraints));

            let folding_randomness = sumcheck::Config {
                field: FieldConfig::new(),
                initial_size: 1 << self.params.folding_factor.at_round(round_index + 1), // Not used
                rounds: vec![
                    sumcheck::RoundConfig {
                        pow: round_params.folding_pow
                    };
                    self.params.folding_factor.at_round(round_index + 1)
                ],
            }
            .verify(verifier_state.inner_mut(), &mut claimed_sum)?;
            round_folding_randomness.push(folding_randomness);

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
            &prev_commitment,
            round_folding_randomness.last().unwrap(),
        )?;

        if !stir_constraints
            .iter()
            .all(|c| c.verify(&final_coefficients))
        {
            return Err(VerificationError);
        }

        let final_sumcheck_randomness = sumcheck::Config {
            field: FieldConfig::new(),
            initial_size: 1 << self.params.final_sumcheck_rounds, // Not used
            rounds: vec![
                sumcheck::RoundConfig {
                    pow: self.params.final_folding_pow
                };
                self.params.final_sumcheck_rounds
            ],
        }
        .verify(verifier_state.inner_mut(), &mut claimed_sum)?;
        round_folding_randomness.push(final_sumcheck_randomness.clone());

        // Compute folding randomness across all rounds
        let folding_randomness = MultilinearPoint(
            round_folding_randomness
                .into_iter()
                .rev()
                .flat_map(|poly| poly.0.into_iter())
                .collect(),
        );

        // Compute evaluation of weights in folding randomness (deferred checks)
        let deferred: Vec<F> = verifier_state.prover_hint_ark()?;
        let evaluation_of_weights =
            self.eval_constraints_poly(&round_constraints, &deferred, folding_randomness.clone());

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
        MerkleConfig::InnerDigest: ProverMessage<[H::U]>,
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
        commitment: &ParsedCommitment<F, MerkleConfig::InnerDigest>,
        folding_randomness: &MultilinearPoint<F>,
    ) -> VerificationResult<Vec<Constraint<F>>>
    where
        H: DuplexSpongeInterface,
        F: Codec<[H::U]>,
        u8: Decoding<[H::U]>,
        [u8; 32]: Decoding<[H::U]>,
        U64: Codec<[H::U]>,
        MerkleConfig::InnerDigest: ProverMessage<[H::U]>,
    {
        params.pow.verify(verifier_state.inner_mut())?;

        let stir_challenges_indexes = get_challenge_stir_queries(
            verifier_state,
            params.domain_size,
            params.folding_factor,
            params.num_queries,
            &self.params.deduplication_strategy,
        );

        // Always open against the single batched commitment
        let mut answers =
            self.verify_merkle_proof(verifier_state, &commitment.root, &stir_challenges_indexes)?;

        // If this is the first round and batching > 1, RLC per leaf to fold_size
        if round_index == 0 && self.params.batch_size > 1 {
            let fold_size = 1 << params.folding_factor;
            answers = crate::whir::utils::rlc_batched_leaves(
                answers,
                fold_size,
                self.params.batch_size,
                commitment.batching_randomness,
            );
        }

        // Compute STIR Constraints
        let folds: Vec<F> = answers
            .into_iter()
            .map(|answers| CoefficientList::new(answers).evaluate(folding_randomness))
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

    pub fn verify_first_round<H>(
        &self,
        verifier_state: &mut VerifierState<'_, H>,
        root: &[MerkleConfig::InnerDigest],
        indices: &[usize],
    ) -> VerificationResult<Vec<Vec<F>>>
    where
        H: DuplexSpongeInterface,
        F: Codec<[H::U]>,
        u8: Decoding<[H::U]>,
        [u8; 32]: Decoding<[H::U]>,
        U64: Codec<[H::U]>,
        MerkleConfig::InnerDigest: ProverMessage<[H::U]>,
    {
        let answers: Vec<Vec<F>> = verifier_state.prover_hint_ark()?;

        let first_round_merkle_proof: Vec<RootPath<F, MerkleConfig>> =
            verifier_state.prover_hint_ark()?;

        if root.len() != first_round_merkle_proof.len() {
            return Err(VerificationError);
        }

        let correct =
            first_round_merkle_proof
                .iter()
                .zip(root.iter())
                .all(|((path, answers), root_hash)| {
                    path.verify(
                        &self.params.leaf_hash_params,
                        &self.params.two_to_one_params,
                        root_hash,
                        answers.iter().map(|a| a.as_ref()),
                    )
                    .unwrap()
                        && path.leaf_indexes == *indices
                });

        if !correct {
            return Err(VerificationError);
        }

        Ok(answers)
    }

    /// Verify a merkle multi-opening proof for the provided indices.
    pub fn verify_merkle_proof<H>(
        &self,
        verifier_state: &mut VerifierState<'_, H>,
        root: &MerkleConfig::InnerDigest,
        indices: &[usize],
    ) -> VerificationResult<Vec<Vec<F>>>
    where
        H: DuplexSpongeInterface,
        F: Codec<[H::U]>,
        u8: Decoding<[H::U]>,
        [u8; 32]: Decoding<[H::U]>,
        U64: Codec<[H::U]>,
        MerkleConfig::InnerDigest: ProverMessage<[H::U]>,
    {
        // Receive claimed leafs
        let answers: Vec<Vec<F>> = verifier_state.prover_hint_ark()?;

        self.merkle_state.read_and_verify_proof(
            verifier_state,
            indices,
            root,
            answers.iter().map(|a| a.as_slice()),
        )?;

        Ok(answers)
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
