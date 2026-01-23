use std::marker::PhantomData;

use ark_crypto_primitives::merkle_tree::Config;
use ark_ff::FftField;
use spongefish::{
    codecs::arkworks_algebra::{FieldToUnitDeserialize, UnitToField},
    ProofError, ProofResult, UnitToBytes,
};
use spongefish_pow::{self, PoWChallenge};

use super::{
    committer::reader::ParsedCommitment,
    parameters::{RoundConfig, WhirConfig},
    statement::{Constraint, Statement, Weights},
    utils::HintDeserialize,
};
use crate::{
    poly_utils::{coeffs::CoefficientList, multilinear::MultilinearPoint},
    sumcheck::SumcheckPolynomial,
    utils::expand_randomness,
    whir::{
        merkle,
        prover::{BatchingMode, RootPath},
        utils::{get_challenge_stir_queries, DigestToUnitDeserialize},
    },
};

pub struct Verifier<'a, F, MerkleConfig, PowStrategy, VerifierState>
where
    F: FftField,
    MerkleConfig: Config,
    VerifierState: UnitToBytes
        + UnitToField<F>
        + FieldToUnitDeserialize<F>
        + PoWChallenge
        + DigestToUnitDeserialize<MerkleConfig>
        + HintDeserialize,
{
    params: &'a WhirConfig<F, MerkleConfig, PowStrategy>,
    merkle_state: merkle::VerifierMerkleState<'a, MerkleConfig>,
    _state: PhantomData<VerifierState>,
}

impl<'a, F, MerkleConfig, PowStrategy, VerifierState>
    Verifier<'a, F, MerkleConfig, PowStrategy, VerifierState>
where
    F: FftField,
    MerkleConfig: Config<Leaf = [F]>,
    PowStrategy: spongefish_pow::PowStrategy,
    VerifierState: UnitToBytes
        + UnitToField<F>
        + FieldToUnitDeserialize<F>
        + PoWChallenge
        + DigestToUnitDeserialize<MerkleConfig>
        + HintDeserialize,
{
    pub const fn new(params: &'a WhirConfig<F, MerkleConfig, PowStrategy>) -> Self {
        Self {
            params,
            merkle_state: merkle::VerifierMerkleState::new(
                params.merkle_proof_strategy,
                &params.leaf_hash_params,
                &params.two_to_one_params,
            ),
            _state: PhantomData,
        }
    }

    /// Verify a WHIR proof.
    ///
    /// Returns the constraint evaluation point and the values of the deferred constraints.
    /// It is the callers responsibility to verify the deferred constraints.
    #[allow(clippy::too_many_lines)]
    pub fn verify(
        &self,
        verifier_state: &mut VerifierState,
        parsed_commitment: &ParsedCommitment<F, MerkleConfig::InnerDigest>,
        statement: &Statement<F>,
    ) -> ProofResult<(MultilinearPoint<F>, Vec<F>)> {
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
            let folding_randomness = self.verify_sumcheck_rounds(
                verifier_state,
                &mut claimed_sum,
                self.params.folding_factor.at_round(0),
                self.params.starting_folding_pow_bits,
            )?;
            round_folding_randomness.push(folding_randomness);
        } else {
            assert_eq!(prev_commitment.ood_points.len(), 0);
            assert!(statement.constraints.is_empty());
            round_constraints.push((vec![], vec![]));

            let mut folding_randomness = vec![F::ZERO; self.params.folding_factor.at_round(0)];
            verifier_state.fill_challenge_scalars(&mut folding_randomness)?;
            round_folding_randomness.push(MultilinearPoint(folding_randomness));

            // PoW
            self.verify_proof_of_work(verifier_state, self.params.starting_folding_pow_bits)?;
        }

        for round_index in 0..self.params.n_rounds() {
            // Fetch round parameters from config
            let round_params = &self.params.round_parameters[round_index];

            // Receive commitment to the folded polynomial (likely encoded at higher expansion)
            let new_commitment = ParsedCommitment::<F, MerkleConfig::InnerDigest>::parse(
                verifier_state,
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

            let folding_randomness = self.verify_sumcheck_rounds(
                verifier_state,
                &mut claimed_sum,
                self.params.folding_factor.at_round(round_index + 1),
                round_params.folding_pow_bits,
            )?;
            round_folding_randomness.push(folding_randomness);

            // Update round parameters
            prev_commitment = new_commitment;
        }

        // In the final round we receive the full polynomial instead of a commitment.
        let mut final_coefficients = vec![F::ZERO; 1 << self.params.final_sumcheck_rounds];
        verifier_state.fill_next_scalars(&mut final_coefficients)?;
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
            return Err(ProofError::InvalidProof);
        }

        let final_sumcheck_randomness = self.verify_sumcheck_rounds(
            verifier_state,
            &mut claimed_sum,
            self.params.final_sumcheck_rounds,
            self.params.final_folding_pow_bits,
        )?;
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
        let deferred: Vec<F> = verifier_state.hint()?;
        let evaluation_of_weights =
            self.eval_constraints_poly(&round_constraints, &deferred, folding_randomness.clone());

        // Check the final sumcheck evaluation
        let final_value = final_coefficients.evaluate(&final_sumcheck_randomness);
        if claimed_sum != evaluation_of_weights * final_value {
            return Err(ProofError::InvalidProof);
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
    pub fn verify_batch(
        &self,
        verifier_state: &mut VerifierState,
        parsed_commitments: &[ParsedCommitment<F, MerkleConfig::InnerDigest>],
        statements: &[Statement<F>],
        mode: BatchingMode,
    ) -> ProofResult<(MultilinearPoint<F>, Vec<F>)> {
        match mode {
            BatchingMode::Standard => {
                self.verify_batch_standard(verifier_state, parsed_commitments, statements)
            }
            BatchingMode::PreFoldSecond => {
                self.verify_batch_prefold(verifier_state, parsed_commitments, statements)
            }
        }
    }

    /// Standard batch verification: all witnesses have same arity
    #[allow(clippy::too_many_lines)]
    fn verify_batch_standard(
        &self,
        verifier_state: &mut VerifierState,
        parsed_commitments: &[ParsedCommitment<F, MerkleConfig::InnerDigest>],
        statements: &[Statement<F>],
    ) -> ProofResult<(MultilinearPoint<F>, Vec<F>)> {
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
            verifier_state.fill_next_scalars(&mut poly_evals)?;
            constraint_evals_matrix.push(poly_evals);
        }

        // Step 2: Sample batching randomness γ (cryptographically bound to matrix)
        let [batching_randomness] = verifier_state.challenge_scalars()?;

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
            let folding_randomness = self.verify_sumcheck_rounds(
                verifier_state,
                &mut claimed_sum,
                self.params.folding_factor.at_round(0),
                self.params.starting_folding_pow_bits,
            )?;
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
            verifier_state.fill_challenge_scalars(&mut folding_randomness)?;
            round_folding_randomness.push(MultilinearPoint(folding_randomness));

            self.verify_proof_of_work(verifier_state, self.params.starting_folding_pow_bits)?;
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
        let mut prev_commitment = ParsedCommitment::<F, MerkleConfig::InnerDigest>::parse(
            verifier_state,
            round_params.num_variables,
            round_params.ood_samples,
        )?;

        // Verify STIR challenges on N original witness trees
        self.verify_proof_of_work(verifier_state, round_params.pow_bits)?;

        let stir_challenges_indexes = get_challenge_stir_queries(
            round_params.domain_size,
            round_params.folding_factor,
            round_params.num_queries,
            verifier_state,
            &self.params.deduplication_strategy,
        )?;

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

        let folding_randomness = self.verify_sumcheck_rounds(
            verifier_state,
            &mut claimed_sum,
            self.params.folding_factor.at_round(1),
            round_params.folding_pow_bits,
        )?;
        round_folding_randomness.push(folding_randomness);

        // Rounds 1+: Standard WHIR verification on the single batched polynomial
        // From here on, the protocol is identical to regular verification
        for round_index in 1..self.params.n_rounds() {
            let round_params = &self.params.round_parameters[round_index];

            let new_commitment = ParsedCommitment::<F, MerkleConfig::InnerDigest>::parse(
                verifier_state,
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

            let folding_randomness = self.verify_sumcheck_rounds(
                verifier_state,
                &mut claimed_sum,
                self.params.folding_factor.at_round(round_index + 1),
                round_params.folding_pow_bits,
            )?;
            round_folding_randomness.push(folding_randomness);

            prev_commitment = new_commitment;
        }

        // Final round (same as regular verify)
        let mut final_coefficients = vec![F::ZERO; 1 << self.params.final_sumcheck_rounds];
        verifier_state.fill_next_scalars(&mut final_coefficients)?;
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
            return Err(ProofError::InvalidProof);
        }

        let final_sumcheck_randomness = self.verify_sumcheck_rounds(
            verifier_state,
            &mut claimed_sum,
            self.params.final_sumcheck_rounds,
            self.params.final_folding_pow_bits,
        )?;
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
        let deferred: Vec<F> = verifier_state.hint()?;
        let evaluation_of_weights =
            self.eval_constraints_poly(&round_constraints, &deferred, folding_randomness.clone());

        // Check the final sumcheck evaluation
        let final_value = final_coefficients.evaluate(&final_sumcheck_randomness);
        if claimed_sum != evaluation_of_weights * final_value {
            return Err(ProofError::InvalidProof);
        }

        Ok((folding_randomness, deferred))
    }

    /// PreFold batch verification: exactly 2 witnesses, one has arity+1.
    /// Verifies pre-folding of the larger polynomial once to match arities before batching.
    #[allow(clippy::too_many_lines)]
    fn verify_batch_prefold(
        &self,
        verifier_state: &mut VerifierState,
        parsed_commitments: &[ParsedCommitment<F, MerkleConfig::InnerDigest>],
        statements: &[Statement<F>],
    ) -> ProofResult<(MultilinearPoint<F>, Vec<F>)> {
        // Validation
        assert_eq!(parsed_commitments.len(), 2);
        assert_eq!(statements.len(), 2);

        let num_vars_0 = parsed_commitments[0].num_variables;
        let num_vars_1 = parsed_commitments[1].num_variables;

        // If the two witnesses have the same arity, we can just verify the batch as standard
        if num_vars_0 == num_vars_1 {
            return Ok(self.verify_batch_standard(
                verifier_state,
                parsed_commitments,
                statements,
            )?);
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

        // --- Phase 1: Prefold verification (α, commit to g', prove consistency vs original g) ---
        let [alpha] = verifier_state.challenge_scalars()?;
        let prefold_randomness = MultilinearPoint(vec![alpha]);

        if self.params.starting_folding_pow_bits > 0. {
            verifier_state.challenge_pow::<PowStrategy>(self.params.starting_folding_pow_bits)?;
        }

        // Parse commitment to g' (committed under the SMALL config)
        let g_folded_commitment = ParsedCommitment::<F, MerkleConfig::InnerDigest>::parse(
            verifier_state,
            num_vars_small,
            self.params.committment_ood_samples,
        )?;

        if self.params.round_parameters[0].pow_bits > 0. {
            verifier_state
                .challenge_pow::<PowStrategy>(self.params.round_parameters[0].pow_bits)?;
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
            g_domain_size,
            g_folding_factor,
            self.params.round_parameters[0].num_queries,
            verifier_state,
            &self.params.deduplication_strategy,
        )?;

        let original_g_answers: Vec<Vec<F>> = verifier_state.hint()?;
        self.merkle_state
            .read_and_verify_proof::<VerifierState, _>(
                verifier_state,
                &stir_indexes,
                &parsed_commitments[big_idx].root,
                original_g_answers.iter().map(|v| v.as_slice()),
            )?;

        let g_folded_stir_evals: Vec<F> = verifier_state.hint()?;
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
                return Err(ProofError::InvalidProof);
            }
        }

        // --- Phase 2: Read evaluation matrix for (f, g') BEFORE sampling γ ---
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

        // Statement constraints (some may be arity n+1 and will be ignored later)
        for statement in statements {
            for constraint in &statement.constraints {
                all_constraints_info
                    .push((constraint.weights.clone(), constraint.defer_evaluation));
            }
        }

        let num_constraints = all_constraints_info.len();
        let mut constraint_evals_matrix = Vec::with_capacity(2);
        for _ in 0..2 {
            let mut row = vec![F::ZERO; num_constraints];
            verifier_state.fill_next_scalars(&mut row)?;
            constraint_evals_matrix.push(row);
        }

        // --- Phase 3: Sample batching randomness γ ---
        let [batching_randomness] = verifier_state.challenge_scalars()?;

        // --- Phase 4: Build combined constraints at common arity (n) using the matrix ---
        let mut all_constraints = Vec::new();
        for (constraint_idx, (weights, defer_evaluation)) in
            all_constraints_info.into_iter().enumerate()
        {
            if weights.num_variables() == num_vars_small {
                let combined_eval = constraint_evals_matrix[0][constraint_idx]
                    + batching_randomness * constraint_evals_matrix[1][constraint_idx];
                all_constraints.push(Constraint {
                    weights,
                    sum: combined_eval,
                    defer_evaluation,
                });
            }
        }

        let mut round_constraints = Vec::new();
        let mut round_folding_randomness = Vec::new();
        let mut claimed_sum = F::ZERO;

        // Initial sumcheck on the combined constraints (identical to standard batch)
        let combination_randomness =
            self.combine_constraints(verifier_state, &mut claimed_sum, &all_constraints)?;
        round_constraints.push((combination_randomness, all_constraints));
        let folding_randomness = self.verify_sumcheck_rounds(
            verifier_state,
            &mut claimed_sum,
            self.params.folding_factor.at_round(0),
            self.params.starting_folding_pow_bits,
        )?;
        round_folding_randomness.push(folding_randomness);

        // --- Round 0: batch-specific verification against BOTH original commitment trees (f and g') ---
        let round_params = &self.params.round_parameters[0];

        // Receive commitment to the batched folded polynomial
        let mut prev_commitment = ParsedCommitment::<F, MerkleConfig::InnerDigest>::parse(
            verifier_state,
            round_params.num_variables,
            round_params.ood_samples,
        )?;

        self.verify_proof_of_work(verifier_state, round_params.pow_bits)?;

        let stir_challenges_indexes = get_challenge_stir_queries(
            round_params.domain_size,
            round_params.folding_factor,
            round_params.num_queries,
            verifier_state,
            &self.params.deduplication_strategy,
        )?;

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
                        combined[j] += batching_randomness * internal_pow * answers_g[query_idx][start + j];
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

        let folding_randomness = self.verify_sumcheck_rounds(
            verifier_state,
            &mut claimed_sum,
            self.params.folding_factor.at_round(1),
            round_params.folding_pow_bits,
        )?;
        round_folding_randomness.push(folding_randomness);

        // Rounds 1+: Standard WHIR verification on the single batched polynomial
        for round_index in 1..self.params.n_rounds() {
            let round_params = &self.params.round_parameters[round_index];

            let new_commitment = ParsedCommitment::<F, MerkleConfig::InnerDigest>::parse(
                verifier_state,
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

            let folding_randomness = self.verify_sumcheck_rounds(
                verifier_state,
                &mut claimed_sum,
                self.params.folding_factor.at_round(round_index + 1),
                round_params.folding_pow_bits,
            )?;
            round_folding_randomness.push(folding_randomness);

            prev_commitment = new_commitment;
        }

        // Final round (same as regular verify)
        let mut final_coefficients = vec![F::ZERO; 1 << self.params.final_sumcheck_rounds];
        verifier_state.fill_next_scalars(&mut final_coefficients)?;
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
            return Err(ProofError::InvalidProof);
        }

        let final_sumcheck_randomness = self.verify_sumcheck_rounds(
            verifier_state,
            &mut claimed_sum,
            self.params.final_sumcheck_rounds,
            self.params.final_folding_pow_bits,
        )?;
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
        let deferred: Vec<F> = verifier_state.hint()?;
        let evaluation_of_weights =
            self.eval_constraints_poly(&round_constraints, &deferred, folding_randomness.clone());

        let final_value = final_coefficients.evaluate(&final_sumcheck_randomness);
        if claimed_sum != evaluation_of_weights * final_value {
            return Err(ProofError::InvalidProof);
        }

        Ok((folding_randomness, deferred))
    }

    /// Create a random linear combination of constraints and add it to the claim.
    /// Returns the randomness used.
    pub fn combine_constraints(
        &self,
        verifier_state: &mut VerifierState,
        claimed_sum: &mut F,
        constraints: &[Constraint<F>],
    ) -> ProofResult<Vec<F>> {
        let [combination_randomness_gen] = verifier_state.challenge_scalars()?;
        let combination_randomness =
            expand_randomness(combination_randomness_gen, constraints.len());
        *claimed_sum += constraints
            .iter()
            .zip(&combination_randomness)
            .map(|(c, rand)| c.sum * rand)
            .sum::<F>();

        Ok(combination_randomness)
    }

    /// Verify rounds of sumcheck updating the claimed_sum and returning the folding randomness.
    pub fn verify_sumcheck_rounds(
        &self,
        verifier_state: &mut VerifierState,
        claimed_sum: &mut F,
        rounds: usize,
        proof_of_work: f64,
    ) -> ProofResult<MultilinearPoint<F>> {
        let mut randomness = Vec::with_capacity(rounds);
        println!(
            "[SUMCHECK] Starting verification, rounds={}, initial claimed_sum={:?}",
            rounds, claimed_sum
        );
        if rounds == 0 {
            println!("[SUMCHECK] No rounds to verify, returning empty randomness");
            return Ok(MultilinearPoint(randomness));
        }
        for round_idx in 0..rounds {
            // Receive this round's sumcheck polynomial
            let sumcheck_poly_evals: [_; 3] = verifier_state.next_scalars()?;
            let sumcheck_poly = SumcheckPolynomial::new(sumcheck_poly_evals.to_vec(), 1);
            let poly_sum = sumcheck_poly.sum_over_boolean_hypercube();

            // Verify claimed sum is consistent with polynomial
            if poly_sum != *claimed_sum {
                println!(
                    "[SUMCHECK] FAILED at round {}: poly_sum={:?}, claimed_sum={:?}",
                    round_idx, poly_sum, *claimed_sum
                );
                return Err(ProofError::InvalidProof);
            }
            println!("[SUMCHECK] Round {} passed", round_idx);

            // Proof of work per round
            self.verify_proof_of_work(verifier_state, proof_of_work)?;

            // Receive folding randomness
            let [folding_randomness_single] = verifier_state.challenge_scalars()?;
            randomness.push(folding_randomness_single);

            // Update claimed sum using folding randomness
            *claimed_sum = sumcheck_poly.evaluate_at_point(&folding_randomness_single.into());
        }

        randomness.reverse();
        Ok(MultilinearPoint(randomness))
    }

    /// Verify a STIR challenges against a commitment and return the constraints.
    pub fn verify_stir_challenges(
        &self,
        round_index: usize,
        verifier_state: &mut VerifierState,
        params: &RoundConfig<F>,
        commitment: &ParsedCommitment<F, MerkleConfig::InnerDigest>,
        folding_randomness: &MultilinearPoint<F>,
    ) -> ProofResult<Vec<Constraint<F>>> {
        self.verify_proof_of_work(verifier_state, params.pow_bits)?;

        let stir_challenges_indexes = get_challenge_stir_queries(
            params.domain_size,
            params.folding_factor,
            params.num_queries,
            verifier_state,
            &self.params.deduplication_strategy,
        )?;

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

    pub fn verify_first_round(
        &self,
        verifier_state: &mut VerifierState,
        root: &[MerkleConfig::InnerDigest],
        indices: &[usize],
    ) -> ProofResult<Vec<Vec<F>>> {
        let answers: Vec<Vec<F>> = verifier_state.hint()?;

        let first_round_merkle_proof: Vec<RootPath<F, MerkleConfig>> = verifier_state.hint()?;

        if root.len() != first_round_merkle_proof.len() {
            return Err(ProofError::InvalidProof);
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
            return Err(ProofError::InvalidProof);
        }

        Ok(answers)
    }

    /// Verify a merkle multi-opening proof for the provided indices.
    pub fn verify_merkle_proof(
        &self,
        verifier_state: &mut VerifierState,
        root: &MerkleConfig::InnerDigest,
        indices: &[usize],
    ) -> ProofResult<Vec<Vec<F>>> {
        // Receive claimed leafs
        let answers: Vec<Vec<F>> = verifier_state.hint()?;

        self.merkle_state
            .read_and_verify_proof::<VerifierState, _>(
                verifier_state,
                indices,
                root,
                answers.iter().map(|a| a.as_slice()),
            )?;

        Ok(answers)
    }

    /// Verify a proof of work challenge.
    /// Does nothing when `bits == 0.`.
    pub fn verify_proof_of_work(
        &self,
        verifier_state: &mut VerifierState,
        bits: f64,
    ) -> ProofResult<()> {
        if bits > 0. {
            verifier_state.challenge_pow::<PowStrategy>(bits)?;
        }
        Ok(())
    }

    /// Evaluate the random linear combination of constraints in `point`.
    fn eval_constraints_poly(
        &self,
        constraints: &[(Vec<F>, Vec<Constraint<F>>)],
        deferred: &[F],
        mut point: MultilinearPoint<F>,
    ) -> F {
        let mut num_variables = self.params.mv_parameters.num_variables;
        let mut deferred_iter = deferred.iter().copied();
        let mut value = F::ZERO;

        println!(
            "[EVAL] Starting eval_constraints_poly with {} deferred values",
            deferred.len()
        );
        for (round, (randomness, constraints)) in constraints.iter().enumerate() {
            assert_eq!(randomness.len(), constraints.len());
            if round > 0 {
                num_variables -= self.params.folding_factor.at_round(round - 1);
                point = MultilinearPoint(point.0[..num_variables].to_vec());
            }
            println!(
                "[EVAL] Round {}: {} constraints, num_variables={}",
                round,
                constraints.len(),
                num_variables
            );
            let mut deferred_count = 0;
            let mut computed_count = 0;
            let round_value: F = constraints
                .iter()
                .enumerate()
                .zip(randomness)
                .map(|((idx, constraint), &rand)| {
                    let val = if constraint.defer_evaluation {
                        let deferred_val = deferred_iter.next().unwrap();
                        println!("[EVAL] Round {} constraint {}: DEFERRED, val={:?}, rand={:?}, contrib={:?}", 
                            round, idx, deferred_val, rand, deferred_val * rand);
                        deferred_count += 1;
                        deferred_val
                    } else {
                        let computed_val = constraint.weights.compute(&point);
                        println!("[EVAL] Round {} constraint {}: COMPUTED, val={:?}, rand={:?}, contrib={:?}", 
                            round, idx, computed_val, rand, computed_val * rand);
                        computed_count += 1;
                        computed_val
                    };
                    val * rand
                })
                .sum();
            println!(
                "[EVAL] Round {}: {} deferred, {} computed, round_value={:?}",
                round, deferred_count, computed_count, round_value
            );
            value += round_value;
        }
        println!("[EVAL] Final value={:?}", value);
        value
    }
}
