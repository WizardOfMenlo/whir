use ark_crypto_primitives::merkle_tree::{Config, MerkleTree, MultiPath};
use ark_ff::FftField;
use ark_poly::EvaluationDomain;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use spongefish::{
    codecs::arkworks_algebra::{FieldToUnitSerialize, UnitToField},
    ProofResult, UnitToBytes,
};
use spongefish_pow::{self, PoWChallenge};
#[cfg(feature = "tracing")]
use tracing::{instrument, span, Level};

use super::{
    committer::Witness,
    parameters::WhirConfig,
    statement::{Statement, Weights},
    utils::HintSerialize,
};
use crate::{
    domain::Domain,
    poly_utils::{coeffs::CoefficientList, multilinear::MultilinearPoint},
    sumcheck::SumcheckSingle,
    utils::expand_randomness,
    whir::{
        merkle,
        parameters::RoundConfig,
        utils::{
            get_challenge_stir_queries, rlc_batched_leaves, sample_ood_points,
            DigestToUnitSerialize,
        },
    },
};

pub type RootPath<F, MC> = (MultiPath<MC>, Vec<Vec<F>>);

/// Batching mode for prove_batch
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BatchingMode {
    /// Standard batching: all witnesses have same arity
    Standard,

    /// Pre-fold second witness: exactly 2 witnesses, second has arity+1
    PreFoldSecond,
}

pub struct Prover<F, MerkleConfig, PowStrategy>
where
    F: FftField,
    MerkleConfig: Config,
{
    config: WhirConfig<F, MerkleConfig, PowStrategy>,
    merkle_state: merkle::ProverMerkleState,
}

impl<F, MerkleConfig, PowStrategy> Prover<F, MerkleConfig, PowStrategy>
where
    F: FftField,
    MerkleConfig: Config<Leaf = [F]>,
    PowStrategy: spongefish_pow::PowStrategy,
{
    pub const fn new(config: WhirConfig<F, MerkleConfig, PowStrategy>) -> Self {
        let merkle_state = merkle::ProverMerkleState::new(config.merkle_proof_strategy);
        Self {
            config,
            merkle_state,
        }
    }

    pub const fn config(&self) -> &WhirConfig<F, MerkleConfig, PowStrategy> {
        &self.config
    }

    pub(crate) fn validate_parameters(&self) -> bool {
        self.config.mv_parameters.num_variables
            == self
                .config
                .folding_factor
                .total_number(self.config.n_rounds())
                + self.config.final_sumcheck_rounds
    }

    pub(crate) fn validate_statement(&self, statement: &Statement<F>) -> bool {
        statement.num_variables() == self.config.mv_parameters.num_variables
            && (self.config.initial_statement || statement.constraints.is_empty())
    }

    pub(crate) fn validate_witness(&self, witness: &Witness<F, MerkleConfig>) -> bool {
        assert_eq!(witness.ood_points.len(), witness.ood_answers.len());
        if !self.config.initial_statement {
            assert!(witness.ood_points.is_empty());
        }
        witness.polynomial.num_variables() == self.config.mv_parameters.num_variables
    }

    /// Proves that the commitment satisfies constraints in `statement`.
    ///
    /// When called without any constraints it only perfoms a low-degree test.
    /// Returns the constraint evaluation point and values of deferred constraints.
    #[cfg_attr(feature = "tracing", instrument(skip_all))]
    pub fn prove<ProverState>(
        &self,
        prover_state: &mut ProverState,
        mut statement: Statement<F>,
        witness: Witness<F, MerkleConfig>,
    ) -> ProofResult<(MultilinearPoint<F>, Vec<F>)>
    where
        ProverState: UnitToField<F>
            + FieldToUnitSerialize<F>
            + UnitToBytes
            + PoWChallenge
            + DigestToUnitSerialize<MerkleConfig>
            + HintSerialize,
    {
        assert!(self.validate_parameters());
        assert!(self.validate_statement(&statement));
        assert!(self.validate_witness(&witness));

        // Convert witness ood_points into constraints
        let new_constraints = witness
            .ood_points
            .into_iter()
            .zip(witness.ood_answers)
            .map(|(point, evaluation)| {
                let weights = Weights::evaluation(MultilinearPoint::expand_from_univariate(
                    point,
                    self.config.mv_parameters.num_variables,
                ));
                (weights, evaluation)
            })
            .collect();

        statement.add_constraints_in_front(new_constraints);
        let mut sumcheck_prover = None;
        let folding_randomness = if self.config.initial_statement {
            // If there is initial statement, then we run the sum-check for
            // this initial statement.
            let [combination_randomness_gen] = prover_state.challenge_scalars()?;

            // Create the sumcheck prover
            let mut sumcheck = SumcheckSingle::new(
                witness.polynomial.clone(),
                &statement,
                combination_randomness_gen,
            );

            let folding_randomness = sumcheck.compute_sumcheck_polynomials::<PowStrategy, _>(
                prover_state,
                self.config.folding_factor.at_round(0),
                self.config.starting_folding_pow_bits,
            )?;

            sumcheck_prover = Some(sumcheck);
            folding_randomness
        } else {
            // If there is no initial statement, there is no need to run the
            // initial rounds of the sum-check, and the verifier directly sends
            // the initial folding randomnesses.
            let mut folding_randomness = vec![F::ZERO; self.config.folding_factor.at_round(0)];
            prover_state.fill_challenge_scalars(&mut folding_randomness)?;

            if self.config.starting_folding_pow_bits > 0. {
                prover_state.challenge_pow::<PowStrategy>(self.config.starting_folding_pow_bits)?;
            }
            MultilinearPoint(folding_randomness)
        };
        let mut randomness_vec = Vec::with_capacity(self.config.mv_parameters.num_variables);
        randomness_vec.extend(folding_randomness.0.iter().rev().copied());
        randomness_vec.resize(self.config.mv_parameters.num_variables, F::ZERO);

        let mut round_state = RoundState {
            domain: self.config.starting_domain.clone(),
            round: 0,
            sumcheck_prover,
            folding_randomness,
            coefficients: witness.polynomial,
            prev_merkle: witness.merkle_tree,
            prev_merkle_answers: witness.merkle_leaves,
            randomness_vec,
            statement,
            batching_randomness: witness.batching_randomness,
        };

        // Run WHIR rounds
        for _round in 0..=self.config.n_rounds() {
            self.round(prover_state, &mut round_state)?;
        }

        // Hints for deferred constraints
        let constraint_eval =
            MultilinearPoint(round_state.randomness_vec.iter().copied().rev().collect());
        let deferred = round_state
            .statement
            .constraints
            .iter()
            .filter(|constraint| constraint.defer_evaluation)
            .map(|constraint| constraint.weights.compute(&constraint_eval))
            .collect();

        prover_state.hint::<Vec<F>>(&deferred)?;

        Ok((constraint_eval, deferred))
    }

    /// Proves that multiple commitments satisfy their respective constraints by batching them.
    ///
    /// This combines multiple independent witnesses and statements into a single proof using
    /// random linear combination (RLC). The prover commits to a complete cross-term evaluation
    /// matrix before sampling the batching randomness γ, preventing adaptive attacks.
    ///
    /// After Round 0, all polynomials are combined into a single batched polynomial and the
    /// protocol proceeds as standard WHIR.
    ///
    /// Returns the constraint evaluation point and values of deferred constraints.
    #[allow(clippy::too_many_lines)]
    #[cfg_attr(feature = "tracing", instrument(skip_all))]
    pub fn prove_batch<ProverState>(
        &self,
        prover_state: &mut ProverState,
        statements: &[Statement<F>],
        witnesses: &[Witness<F, MerkleConfig>],
        mode: BatchingMode,
    ) -> ProofResult<(MultilinearPoint<F>, Vec<F>)>
    where
        ProverState: UnitToField<F>
            + FieldToUnitSerialize<F>
            + UnitToBytes
            + PoWChallenge
            + DigestToUnitSerialize<MerkleConfig>
            + HintSerialize,
    {
        match mode {
            BatchingMode::Standard => {
                self.prove_batch_standard(prover_state, statements, witnesses)
            }
            BatchingMode::PreFoldSecond => {
                self.prove_batch_prefold(prover_state, statements, witnesses)
            }
        }
    }

    /// Standard batch proving: all witnesses have same arity
    #[allow(clippy::too_many_lines)]
    #[cfg_attr(feature = "tracing", instrument(skip_all))]
    fn prove_batch_standard<ProverState>(
        &self,
        prover_state: &mut ProverState,
        statements: &[Statement<F>],
        witnesses: &[Witness<F, MerkleConfig>],
    ) -> ProofResult<(MultilinearPoint<F>, Vec<F>)>
    where
        ProverState: UnitToField<F>
            + FieldToUnitSerialize<F>
            + UnitToBytes
            + PoWChallenge
            + DigestToUnitSerialize<MerkleConfig>
            + HintSerialize,
    {
        // Validation
        assert!(self.validate_parameters());
        assert!(
            !witnesses.is_empty(),
            "Cannot batch prove with no witnesses"
        );
        assert_eq!(
            witnesses.len(),
            statements.len(),
            "Number of witnesses must match number of statements"
        );

        let num_vars = witnesses[0].polynomial.num_variables();
        assert_eq!(
            num_vars, self.config.mv_parameters.num_variables,
            "Witness polynomial variables must match config"
        );

        for (idx, (statement, witness)) in statements.iter().zip(witnesses).enumerate() {
            assert!(
                self.validate_statement(statement),
                "Statement {idx} is invalid"
            );
            assert!(self.validate_witness(witness), "Witness {idx} is invalid");
            assert_eq!(
                witness.polynomial.num_variables(),
                num_vars,
                "All witness polynomials must have the same number of variables"
            );
            assert_eq!(
                statement.num_variables(),
                num_vars,
                "All statements must have the same number of variables"
            );
        }

        // Step 1: Commit to the full N×M constraint evaluation matrix BEFORE sampling γ.
        //
        // Security: The prover must commit to ALL cross-term evaluations (P_i(w_j) for all i,j)
        // before learning the batching randomness γ. This prevents the prover from adaptively
        // choosing P_i(w_j) values after seeing γ, which would allow breaking soundness by
        // constructing polynomials that satisfy P_batched = Σ γ^i·P_i at constraint points
        // but differ elsewhere.

        // Collect all constraint weights from OOD samples and statements
        let mut all_constraint_weights = Vec::new();

        // OOD constraints from each witness
        for witness in witnesses {
            for point in &witness.ood_points {
                let ml_point = MultilinearPoint::expand_from_univariate(*point, num_vars);
                all_constraint_weights.push(Weights::evaluation(ml_point));
            }
        }

        // Statement constraints
        for statement in statements {
            for constraint in &statement.constraints {
                all_constraint_weights.push(constraint.weights.clone());
            }
        }

        // Evaluate EVERY polynomial at EVERY constraint point
        // This creates an N×M matrix where N = #polynomials, M = #constraints
        let mut constraint_evals_matrix = Vec::with_capacity(witnesses.len());
        for witness in witnesses {
            let mut poly_evals = Vec::with_capacity(all_constraint_weights.len());
            for weights in &all_constraint_weights {
                let eval = weights.evaluate(&witness.polynomial);
                poly_evals.push(eval);
            }
            constraint_evals_matrix.push(poly_evals);
        }

        // Commit the evaluation matrix to the transcript
        let all_evals_flat: Vec<F> = constraint_evals_matrix
            .iter()
            .flat_map(|row| row.iter().copied())
            .collect();
        prover_state.add_scalars(&all_evals_flat)?;

        // Step 2: Sample batching randomness γ (cryptographically bound to committed matrix)
        let [batching_randomness] = prover_state.challenge_scalars()?;

        // Step 3: Materialize the batched polynomial P_batched = P₀ + γ·P₁ + γ²·P₂ + ...
        let mut batched_coeffs = witnesses[0].polynomial.coeffs().to_vec();
        let mut pow = batching_randomness;
        for witness in witnesses.iter().skip(1) {
            for (dst, src) in batched_coeffs.iter_mut().zip(witness.polynomial.coeffs()) {
                *dst += pow * src;
            }
            pow *= batching_randomness;
        }
        let batched_poly = CoefficientList::new(batched_coeffs);

        // Step 4: Build combined statement using RLC of the committed evaluation matrix
        // For each constraint j: combined_eval[j] = Σᵢ γⁱ·eval[i][j]
        let mut combined_statement = Statement::new(num_vars);

        for (constraint_idx, weights) in all_constraint_weights.into_iter().enumerate() {
            let mut combined_eval = F::ZERO;
            let mut pow = F::ONE;
            for poly_evals in &constraint_evals_matrix {
                combined_eval += pow * poly_evals[constraint_idx];
                pow *= batching_randomness;
            }
            combined_statement.add_constraint(weights, combined_eval);
        }

        // Run initial sumcheck on batched polynomial with combined statement
        let mut sumcheck_prover = None;
        let folding_randomness = if self.config.initial_statement {
            let [combination_randomness_gen] = prover_state.challenge_scalars()?;
            let mut sumcheck = SumcheckSingle::new(
                batched_poly.clone(),
                &combined_statement,
                combination_randomness_gen,
            );
            let folding_randomness = sumcheck.compute_sumcheck_polynomials::<PowStrategy, _>(
                prover_state,
                self.config.folding_factor.at_round(0),
                self.config.starting_folding_pow_bits,
            )?;
            sumcheck_prover = Some(sumcheck);
            folding_randomness
        } else {
            let mut folding_randomness = vec![F::ZERO; self.config.folding_factor.at_round(0)];
            prover_state.fill_challenge_scalars(&mut folding_randomness)?;
            if self.config.starting_folding_pow_bits > 0. {
                prover_state.challenge_pow::<PowStrategy>(self.config.starting_folding_pow_bits)?;
            }
            MultilinearPoint(folding_randomness)
        };

        let mut randomness_vec = Vec::with_capacity(self.config.mv_parameters.num_variables);
        randomness_vec.extend(folding_randomness.0.iter().rev().copied());
        randomness_vec.resize(self.config.mv_parameters.num_variables, F::ZERO);

        // Round 0: Batch-specific handling
        //
        // Unlike regular proving, Round 0 for batch proving:
        // 1. Folds and commits the batched polynomial (for Round 1+)
        // 2. Queries all N original witness Merkle trees (not the batched tree)
        // 3. RLC-combines the N query answers before feeding them to sumcheck
        //
        // This allows the verifier to check consistency with the N original commitments
        // while the rest of the protocol operates on the single batched polynomial.
        let round_params = &self.config.round_parameters[0];
        let num_variables_after_fold = num_vars - self.config.folding_factor.at_round(0);
        let batched_folded_poly = batched_poly.fold(&folding_randomness);

        // Build Merkle tree for the batched folded polynomial
        let new_domain = self.config.starting_domain.scale(2);
        let expansion = new_domain.size() / batched_folded_poly.num_coeffs();
        let folding_factor_next = self.config.folding_factor.at_round(1);
        let batched_evals = self.config.reed_solomon.interleaved_encode(
            batched_folded_poly.coeffs(),
            expansion,
            folding_factor_next,
        );

        #[cfg(not(feature = "parallel"))]
        let leafs_iter = batched_evals.chunks_exact(1 << folding_factor_next);
        #[cfg(feature = "parallel")]
        let leafs_iter = batched_evals.par_chunks_exact(1 << folding_factor_next);

        let batched_merkle_tree = MerkleTree::new(
            &self.config.leaf_hash_params,
            &self.config.two_to_one_params,
            leafs_iter,
        )
        .unwrap();
        prover_state.add_digest(batched_merkle_tree.root())?;

        // Handle OOD (Out-Of-Domain) samples
        let (ood_points, ood_answers) = sample_ood_points(
            prover_state,
            round_params.ood_samples,
            num_variables_after_fold,
            |point| batched_folded_poly.evaluate(point),
        )?;

        // PoW
        if round_params.pow_bits > 0. {
            prover_state.challenge_pow::<PowStrategy>(round_params.pow_bits)?;
        }

        // STIR Queries: Open all N original witness trees at challenge positions
        let stir_indexes = get_challenge_stir_queries(
            self.config.starting_domain.size(),
            self.config.folding_factor.at_round(0),
            round_params.num_queries,
            prover_state,
            &self.config.deduplication_strategy,
        )?;

        let fold_size = 1 << self.config.folding_factor.at_round(0);
        let leaf_size = fold_size * self.config.batch_size;
        let mut all_witness_answers = Vec::with_capacity(witnesses.len());

        // For each witness, provide Merkle proof opening at the challenge indices
        for witness in witnesses {
            let answers: Vec<_> = stir_indexes
                .iter()
                .map(|&i| witness.merkle_leaves[i * leaf_size..(i + 1) * leaf_size].to_vec())
                .collect();
            prover_state.hint::<Vec<Vec<F>>>(&answers)?;
            self.merkle_state.write_proof_hint(
                &witness.merkle_tree,
                &stir_indexes,
                prover_state,
            )?;
            all_witness_answers.push(answers);
        }

        // RLC-combine the N query answers: combined[j] = Σᵢ γⁱ·witness_answers[i][j]
        // This produces the expected answers for the batched polynomial
        let rlc_answers: Vec<Vec<F>> = (0..stir_indexes.len())
            .map(|query_idx| {
                let mut combined = vec![F::ZERO; fold_size];
                let mut pow = F::ONE;
                for (witness_idx, witness) in witnesses.iter().enumerate() {
                    let answer = &all_witness_answers[witness_idx][query_idx];

                    // First, internally reduce stacked leaf using witness's batching_randomness
                    let mut internal_pow = F::ONE;
                    for poly_idx in 0..self.config.batch_size {
                        let start = poly_idx * fold_size;
                        for j in 0..fold_size {
                            combined[j] += pow * internal_pow * answer[start + j];
                        }
                        internal_pow *= witness.batching_randomness;
                    }

                    pow *= batching_randomness;
                }
                combined
            })
            .collect();

        // Compute STIR challenges and evaluations
        let domain_scaled_gen = self
            .config
            .starting_domain
            .backing_domain
            .element(1 << self.config.folding_factor.at_round(0));
        let stir_challenges: Vec<MultilinearPoint<F>> = ood_points
            .iter()
            .copied()
            .chain(
                stir_indexes
                    .iter()
                    .map(|&i| domain_scaled_gen.pow([i as u64])),
            )
            .map(|univariate| {
                MultilinearPoint::expand_from_univariate(univariate, num_variables_after_fold)
            })
            .collect();

        let mut stir_evaluations = ood_answers;
        stir_evaluations.extend(
            rlc_answers
                .iter()
                .map(|answer| CoefficientList::new(answer.clone()).evaluate(&folding_randomness)),
        );

        // Randomness for combination
        let [combination_randomness_gen] = prover_state.challenge_scalars()?;
        let combination_randomness =
            expand_randomness(combination_randomness_gen, stir_challenges.len());

        // Update sumcheck with STIR constraints
        let mut sumcheck_prover = sumcheck_prover.map_or_else(
            || {
                let mut statement = Statement::new(batched_folded_poly.num_variables());
                for (point, eval) in stir_challenges.iter().zip(stir_evaluations.iter()) {
                    statement.add_constraint(Weights::evaluation(point.clone()), *eval);
                }
                SumcheckSingle::new(
                    batched_folded_poly.clone(),
                    &statement,
                    combination_randomness[1],
                )
            },
            |mut sumcheck| {
                sumcheck.add_new_equality(
                    &stir_challenges,
                    &stir_evaluations,
                    &combination_randomness,
                );
                sumcheck
            },
        );

        let folding_randomness = sumcheck_prover.compute_sumcheck_polynomials::<PowStrategy, _>(
            prover_state,
            folding_factor_next,
            round_params.folding_pow_bits,
        )?;

        let start_idx = self.config.folding_factor.at_round(0);
        let dst_randomness = &mut randomness_vec[start_idx..][..folding_randomness.0.len()];
        for (dst, src) in dst_randomness
            .iter_mut()
            .zip(folding_randomness.0.iter().rev())
        {
            *dst = *src;
        }

        // Transition to Round 1: From here on, the protocol operates on the single batched
        // polynomial. All subsequent rounds use standard WHIR proving with one Merkle tree.
        let mut round_state = RoundState {
            domain: new_domain,
            round: 1,
            sumcheck_prover: Some(sumcheck_prover),
            folding_randomness,
            coefficients: batched_folded_poly,
            prev_merkle: batched_merkle_tree,
            prev_merkle_answers: batched_evals,
            randomness_vec,
            statement: combined_statement,
            batching_randomness,
        };

        // Execute standard WHIR rounds 1 through n on the batched polynomial
        for _round in 1..=self.config.n_rounds() {
            self.round(prover_state, &mut round_state)?;
        }

        // Hints for deferred constraints
        let constraint_eval =
            MultilinearPoint(round_state.randomness_vec.iter().copied().rev().collect());
        let deferred = round_state
            .statement
            .constraints
            .iter()
            .filter(|constraint| constraint.defer_evaluation)
            .map(|constraint| constraint.weights.compute(&constraint_eval))
            .collect();

        prover_state.hint::<Vec<F>>(&deferred)?;

        Ok((constraint_eval, deferred))
    }

    /// PreFold batch proving: exactly 2 witnesses, one has arity+1.
    /// Pre-folds the larger polynomial once to match arities before batching.
    #[allow(clippy::too_many_lines)]
    #[cfg_attr(feature = "tracing", instrument(skip_all))]
    fn prove_batch_prefold<ProverState>(
        &self,
        prover_state: &mut ProverState,
        statements: &[Statement<F>],
        witnesses: &[Witness<F, MerkleConfig>],
    ) -> ProofResult<(MultilinearPoint<F>, Vec<F>)>
    where
        ProverState: UnitToField<F>
            + FieldToUnitSerialize<F>
            + UnitToBytes
            + PoWChallenge
            + DigestToUnitSerialize<MerkleConfig>
            + HintSerialize,
    {
        // Validation (simplified PreFold assumes exactly 2 witnesses, arity diff 1)
        assert!(self.validate_parameters());
        assert_eq!(
            witnesses.len(),
            2,
            "PreFoldSecond mode requires exactly 2 witnesses"
        );
        assert_eq!(
            statements.len(),
            2,
            "PreFoldSecond mode requires exactly 2 statements"
        );

        let num_vars_0 = witnesses[0].polynomial.num_variables();
        let num_vars_1 = witnesses[1].polynomial.num_variables();

        // If called with equal arity, just fall back to standard batching.
        if num_vars_0 == num_vars_1 {
            return self.prove_batch_standard(prover_state, statements, witnesses);
        }

        // Determine which witness is larger (has more variables); prefold that one.
        let (small_idx, big_idx, num_vars_small, num_vars_big) = if num_vars_0 < num_vars_1 {
            (0usize, 1usize, num_vars_0, num_vars_1)
        } else {
            (1usize, 0usize, num_vars_1, num_vars_0)
        };

        assert_eq!(
            num_vars_small, self.config.mv_parameters.num_variables,
            "PreFoldSecond expects prover config to match the smaller arity"
        );
        assert_eq!(
            num_vars_big,
            num_vars_small + 1,
            "PreFold requires arity difference of exactly 1"
        );

        let witness_f = &witnesses[small_idx];
        let witness_g = &witnesses[big_idx];

        // --- Phase 1: Prefold g -> g' (commit to g' with the SMALL config) ---
        let [alpha] = prover_state.challenge_scalars()?;
        let prefold_randomness = MultilinearPoint(vec![alpha]);

        if self.config.starting_folding_pow_bits > 0. {
            prover_state.challenge_pow::<PowStrategy>(self.config.starting_folding_pow_bits)?;
        }

        let g_folded = witness_g.polynomial.fold(&prefold_randomness);
        assert_eq!(g_folded.num_variables(), num_vars_small);

        // Commit to g' under the SMALL config.
        //
        // We intentionally implement the commitment inline here to avoid requiring
        // `WhirConfig: Clone` or additional transcript trait bounds.
        let base_domain = self
            .config
            .starting_domain
            .base_domain
            .expect("starting_domain.base_domain must be set");
        let expansion = base_domain.size() / g_folded.num_coeffs();
        let commit_fold_size = 1 << self.config.folding_factor.at_round(0);

        let g_folded_evals = self.config.reed_solomon.interleaved_encode(
            g_folded.coeffs(),
            expansion,
            self.config.folding_factor.at_round(0),
        );

        // If internal batching is enabled (batch_size > 1), we commit to g' using a
        // *stacked-leaf* Merkle tree shape so that the later "Round 0" batching code can
        // reuse the standard internal-leaf reduction logic.
        //
        // Concretely, each leaf is laid out as:
        //   [ g'(chunk), 0(chunk), 0(chunk), ... ]
        // with length = commit_fold_size * batch_size, and we set witness_g_folded.batching_randomness = 0.
        //
        // This keeps the folded witness as a single polynomial (g'), while keeping the leaf layout
        // compatible with the batch proof "internal batching" reduction code.
        let (g_folded_merkle, g_folded_merkle_leaves) = if self.config.batch_size == 1 {
            #[cfg(not(feature = "parallel"))]
            let leafs_iter = g_folded_evals.chunks_exact(commit_fold_size);
            #[cfg(feature = "parallel")]
            let leafs_iter = g_folded_evals.par_chunks_exact(commit_fold_size);

            let merkle = MerkleTree::<MerkleConfig>::new(
                &self.config.leaf_hash_params,
                &self.config.two_to_one_params,
                leafs_iter,
            )
            .unwrap();
            (merkle, g_folded_evals)
        } else {
            let stacked_leaf_size = commit_fold_size * self.config.batch_size;
            let num_leaves = g_folded_evals.len() / commit_fold_size;
            let mut stacked_leaves = vec![F::ZERO; num_leaves * stacked_leaf_size];

            for leaf_idx in 0..num_leaves {
                let src_start = leaf_idx * commit_fold_size;
                let dst_start = leaf_idx * stacked_leaf_size;
                stacked_leaves[dst_start..dst_start + commit_fold_size]
                    .copy_from_slice(&g_folded_evals[src_start..src_start + commit_fold_size]);
            }

            #[cfg(not(feature = "parallel"))]
            let leafs_iter = stacked_leaves.chunks_exact(stacked_leaf_size);
            #[cfg(feature = "parallel")]
            let leafs_iter = stacked_leaves.par_chunks_exact(stacked_leaf_size);

            let merkle = MerkleTree::<MerkleConfig>::new(
                &self.config.leaf_hash_params,
                &self.config.two_to_one_params,
                leafs_iter,
            )
            .unwrap();
            (merkle, stacked_leaves)
        };

        prover_state.add_digest(g_folded_merkle.root())?;

        let (g_folded_ood_points, g_folded_ood_answers) = sample_ood_points(
            prover_state,
            self.config.committment_ood_samples,
            num_vars_small,
            |point| g_folded.evaluate(point),
        )?;

        let witness_g_folded = Witness {
            polynomial: g_folded,
            merkle_tree: g_folded_merkle,
            merkle_leaves: g_folded_merkle_leaves,
            ood_points: g_folded_ood_points,
            ood_answers: g_folded_ood_answers,
            batching_randomness: F::ZERO,
        };

        // PoW before STIR queries on original g (consistency check)
        if self.config.round_parameters[0].pow_bits > 0. {
            prover_state.challenge_pow::<PowStrategy>(self.config.round_parameters[0].pow_bits)?;
        }

        // STIR queries on ORIGINAL g (prove consistency: g'(leaf) = g(α, leaf))
        // PreFold folds exactly 1 variable => fold_size=2 in the original g commitment tree.
        let g_domain_size = self
            .config
            .starting_domain
            .size()
            .checked_shl((num_vars_big - num_vars_small) as u32)
            .expect("domain size overflow in prefold");
        let g_folding_factor = num_vars_big - num_vars_small;
        let stir_indexes = get_challenge_stir_queries(
            g_domain_size,
            g_folding_factor,
            self.config.round_parameters[0].num_queries,
            prover_state,
            &self.config.deduplication_strategy,
        )?;

        let fold_size = 1 << g_folding_factor;
        let leaf_size = fold_size * self.config.batch_size;
        let original_g_answers: Vec<Vec<F>> = stir_indexes
            .iter()
            .map(|&i| witness_g.merkle_leaves[i * leaf_size..(i + 1) * leaf_size].to_vec())
            .collect();

        prover_state.hint::<Vec<Vec<F>>>(&original_g_answers)?;
        self.merkle_state
            .write_proof_hint(&witness_g.merkle_tree, &stir_indexes, prover_state)?;

        // Provide g' evaluations at the same queried leaf positions.
        // We compute them directly from the opened chunk by:
        //   1) internally reducing the stacked leaf using g's batching_randomness (if batch_size>1),
        //   2) folding the 2-point chunk at α (since PreFold folds exactly one variable).
        // The verifier performs the same computation.
        let g_folded_stir_evals: Vec<F> = original_g_answers
            .iter()
            .map(|chunk| {
                let mut reduced = vec![F::ZERO; fold_size];
                let mut internal_pow = F::ONE;
                for poly_idx in 0..self.config.batch_size {
                    let start = poly_idx * fold_size;
                    for j in 0..fold_size {
                        reduced[j] += internal_pow * chunk[start + j];
                    }
                    internal_pow *= witness_g.batching_randomness;
                }
                CoefficientList::new(reduced).evaluate(&prefold_randomness)
            })
            .collect();
        prover_state.hint::<Vec<F>>(&g_folded_stir_evals)?;

        // --- Phase 2: Commit evaluation matrix for (f, g') BEFORE sampling γ ---
        //
        // Matrix columns are:
        //  - OOD points from f commitment (arity n)
        //  - OOD points from g' commitment (arity n)
        //  - All statement constraints (some may be arity n+1; those will be ignored later)
        let mut all_constraint_weights = Vec::new();

        for point in &witness_f.ood_points {
            let ml_point = MultilinearPoint::expand_from_univariate(*point, num_vars_small);
            all_constraint_weights.push(Weights::evaluation(ml_point));
        }
        for point in &witness_g_folded.ood_points {
            let ml_point = MultilinearPoint::expand_from_univariate(*point, num_vars_small);
            all_constraint_weights.push(Weights::evaluation(ml_point));
        }
        for statement in statements {
            for constraint in &statement.constraints {
                all_constraint_weights.push(constraint.weights.clone());
            }
        }

        let mut constraint_evals_matrix = Vec::with_capacity(2);
        for poly in [&witness_f.polynomial, &witness_g_folded.polynomial] {
            let mut row = Vec::with_capacity(all_constraint_weights.len());
            for weights in &all_constraint_weights {
                if weights.num_variables() == poly.num_variables() {
                    row.push(weights.evaluate(poly));
                } else {
                    // Constraint arity doesn't match this polynomial; treat as non-applicable.
                    row.push(F::ZERO);
                }
            }
            constraint_evals_matrix.push(row);
        }

        let all_evals_flat: Vec<F> = constraint_evals_matrix
            .iter()
            .flat_map(|row| row.iter().copied())
            .collect();
        prover_state.add_scalars(&all_evals_flat)?;

        // --- Phase 3: Sample batching randomness γ ---
        let [batching_randomness] = prover_state.challenge_scalars()?;

        // --- Phase 4: Materialize batched polynomial h = f + γ·g' ---
        let mut batched_coeffs = witness_f.polynomial.coeffs().to_vec();
        for (dst, src) in batched_coeffs
            .iter_mut()
            .zip(witness_g_folded.polynomial.coeffs())
        {
            *dst += batching_randomness * src;
        }
        let batched_poly = CoefficientList::new(batched_coeffs);

        // --- Phase 5: Build combined statement at the common arity (n) ---
        let mut combined_statement = Statement::new(num_vars_small);
        for (constraint_idx, weights) in all_constraint_weights.iter().enumerate() {
            if weights.num_variables() == num_vars_small {
                let combined_eval = constraint_evals_matrix[0][constraint_idx]
                    + batching_randomness * constraint_evals_matrix[1][constraint_idx];
                combined_statement.add_constraint(weights.clone(), combined_eval);
            }
        }

        // --- Phase 6: Run batch proving identical to standard batch from here on ---
        let mut sumcheck_prover = None;
        let folding_randomness = if self.config.initial_statement {
            let [combination_randomness_gen] = prover_state.challenge_scalars()?;
            let mut sumcheck = SumcheckSingle::new(
                batched_poly.clone(),
                &combined_statement,
                combination_randomness_gen,
            );
            let folding_randomness = sumcheck.compute_sumcheck_polynomials::<PowStrategy, _>(
                prover_state,
                self.config.folding_factor.at_round(0),
                self.config.starting_folding_pow_bits,
            )?;
            sumcheck_prover = Some(sumcheck);
            folding_randomness
        } else {
            let mut folding_randomness = vec![F::ZERO; self.config.folding_factor.at_round(0)];
            prover_state.fill_challenge_scalars(&mut folding_randomness)?;
            if self.config.starting_folding_pow_bits > 0. {
                prover_state.challenge_pow::<PowStrategy>(self.config.starting_folding_pow_bits)?;
            }
            MultilinearPoint(folding_randomness)
        };

        let mut randomness_vec = Vec::with_capacity(self.config.mv_parameters.num_variables);
        randomness_vec.extend(folding_randomness.0.iter().rev().copied());
        randomness_vec.resize(self.config.mv_parameters.num_variables, F::ZERO);

        // Round 0: batch-specific binding to BOTH original commitment trees (f and g')
        let round_params = &self.config.round_parameters[0];
        let num_variables_after_fold = num_vars_0 - self.config.folding_factor.at_round(0);
        let batched_folded_poly = batched_poly.fold(&folding_randomness);

        // Build Merkle tree for the batched folded polynomial (used from Round 1+)
        let new_domain = self.config.starting_domain.scale(2);
        let expansion = new_domain.size() / batched_folded_poly.num_coeffs();
        let folding_factor_next = self.config.folding_factor.at_round(1);
        let batched_evals = self.config.reed_solomon.interleaved_encode(
            batched_folded_poly.coeffs(),
            expansion,
            folding_factor_next,
        );

        #[cfg(not(feature = "parallel"))]
        let leafs_iter = batched_evals.chunks_exact(1 << folding_factor_next);
        #[cfg(feature = "parallel")]
        let leafs_iter = batched_evals.par_chunks_exact(1 << folding_factor_next);

        let batched_merkle_tree = MerkleTree::new(
            &self.config.leaf_hash_params,
            &self.config.two_to_one_params,
            leafs_iter,
        )
        .unwrap();
        prover_state.add_digest(batched_merkle_tree.root())?;

        let (ood_points, ood_answers) = sample_ood_points(
            prover_state,
            round_params.ood_samples,
            num_variables_after_fold,
            |point| batched_folded_poly.evaluate(point),
        )?;

        if round_params.pow_bits > 0. {
            prover_state.challenge_pow::<PowStrategy>(round_params.pow_bits)?;
        }

        let stir_indexes = get_challenge_stir_queries(
            self.config.starting_domain.size(),
            self.config.folding_factor.at_round(0),
            round_params.num_queries,
            prover_state,
            &self.config.deduplication_strategy,
        )?;

        let fold_size = 1 << self.config.folding_factor.at_round(0);
        let leaf_size = fold_size * self.config.batch_size;
        let mut all_witness_answers = Vec::with_capacity(2);

        // Provide openings for f and for g' (both committed under the SMALL config)
        for witness in [witness_f, &witness_g_folded] {
            let answers: Vec<Vec<F>> = stir_indexes
                .iter()
                .map(|&i| witness.merkle_leaves[i * leaf_size..(i + 1) * leaf_size].to_vec())
                .collect();
            prover_state.hint::<Vec<Vec<F>>>(&answers)?;
            self.merkle_state.write_proof_hint(
                &witness.merkle_tree,
                &stir_indexes,
                prover_state,
            )?;
            all_witness_answers.push(answers);
        }

        // RLC-combine the 2 query answers: combined[j] = answers_f[j] + γ·answers_g'[j]
        let rlc_answers: Vec<Vec<F>> = (0..stir_indexes.len())
            .map(|query_idx| {
                let mut combined = vec![F::ZERO; fold_size];
                let mut pow = F::ONE;
                for (witness_idx, witness) in [witness_f, &witness_g_folded].into_iter().enumerate()
                {
                    let answer = &all_witness_answers[witness_idx][query_idx];

                    // Internal batching (disabled in this mode by batch_size==1, but kept for symmetry)
                    let mut internal_pow = F::ONE;
                    for poly_idx in 0..self.config.batch_size {
                        let start = poly_idx * fold_size;
                        for j in 0..fold_size {
                            combined[j] += pow * internal_pow * answer[start + j];
                        }
                        internal_pow *= witness.batching_randomness;
                    }

                    pow *= batching_randomness;
                }
                combined
            })
            .collect();

        let domain_scaled_gen = self
            .config
            .starting_domain
            .backing_domain
            .element(1 << self.config.folding_factor.at_round(0));
        let stir_challenges: Vec<MultilinearPoint<F>> = ood_points
            .iter()
            .copied()
            .chain(
                stir_indexes
                    .iter()
                    .map(|&i| domain_scaled_gen.pow([i as u64])),
            )
            .map(|univariate| {
                MultilinearPoint::expand_from_univariate(univariate, num_variables_after_fold)
            })
            .collect();

        let mut stir_evaluations = ood_answers;
        stir_evaluations.extend(
            rlc_answers
                .iter()
                .map(|answer| CoefficientList::new(answer.clone()).evaluate(&folding_randomness)),
        );

        let [combination_randomness_gen] = prover_state.challenge_scalars()?;
        let combination_randomness =
            expand_randomness(combination_randomness_gen, stir_challenges.len());

        let mut sumcheck_prover = sumcheck_prover.map_or_else(
            || {
                let mut statement = Statement::new(batched_folded_poly.num_variables());
                for (point, eval) in stir_challenges.iter().zip(stir_evaluations.iter()) {
                    statement.add_constraint(Weights::evaluation(point.clone()), *eval);
                }
                SumcheckSingle::new(
                    batched_folded_poly.clone(),
                    &statement,
                    combination_randomness[1],
                )
            },
            |mut sumcheck| {
                sumcheck.add_new_equality(
                    &stir_challenges,
                    &stir_evaluations,
                    &combination_randomness,
                );
                sumcheck
            },
        );

        let folding_randomness = sumcheck_prover.compute_sumcheck_polynomials::<PowStrategy, _>(
            prover_state,
            folding_factor_next,
            round_params.folding_pow_bits,
        )?;

        let start_idx = self.config.folding_factor.at_round(0);
        let dst_randomness = &mut randomness_vec[start_idx..][..folding_randomness.0.len()];
        for (dst, src) in dst_randomness
            .iter_mut()
            .zip(folding_randomness.0.iter().rev())
        {
            *dst = *src;
        }

        // Transition to Round 1+
        let mut round_state = RoundState {
            domain: new_domain,
            round: 1,
            sumcheck_prover: Some(sumcheck_prover),
            folding_randomness,
            coefficients: batched_folded_poly,
            prev_merkle: batched_merkle_tree,
            prev_merkle_answers: batched_evals,
            randomness_vec,
            statement: combined_statement,
            batching_randomness,
        };

        for _round in 1..=self.config.n_rounds() {
            self.round(prover_state, &mut round_state)?;
        }

        // Hints for deferred constraints
        let constraint_eval =
            MultilinearPoint(round_state.randomness_vec.iter().copied().rev().collect());
        let deferred: Vec<F> = round_state
            .statement
            .constraints
            .iter()
            .filter(|constraint| constraint.defer_evaluation)
            .map(|constraint| constraint.weights.compute(&constraint_eval))
            .collect();
        prover_state.hint::<Vec<F>>(&deferred)?;

        Ok((constraint_eval, deferred))
    }

    #[allow(clippy::too_many_lines)]
    #[cfg_attr(feature = "tracing", instrument(skip_all, fields(size = round_state.coefficients.num_coeffs())))]
    fn round<ProverState>(
        &self,
        prover_state: &mut ProverState,
        round_state: &mut RoundState<F, MerkleConfig>,
    ) -> ProofResult<()>
    where
        ProverState: UnitToField<F>
            + UnitToBytes
            + FieldToUnitSerialize<F>
            + PoWChallenge
            + DigestToUnitSerialize<MerkleConfig>
            + HintSerialize,
    {
        // Fold the coefficients
        let folded_coefficients = round_state
            .coefficients
            .fold(&round_state.folding_randomness);

        let num_variables = self.config.mv_parameters.num_variables
            - self.config.folding_factor.total_number(round_state.round);
        // num_variables should match the folded_coefficients here.
        assert_eq!(num_variables, folded_coefficients.num_variables());

        // Base case
        if round_state.round == self.config.n_rounds() {
            return self.final_round(prover_state, round_state, &folded_coefficients);
        }

        let round_params = &self.config.round_parameters[round_state.round];

        // Compute the folding factors for later use
        let folding_factor = self.config.folding_factor.at_round(round_state.round);
        let folding_factor_next = self.config.folding_factor.at_round(round_state.round + 1);

        // Fold the coefficients, and compute fft of polynomial (and commit)
        let new_domain = round_state.domain.scale(2);
        let expansion = new_domain.size() / folded_coefficients.num_coeffs();
        let evals = self.config.reed_solomon.interleaved_encode(
            folded_coefficients.coeffs(),
            expansion,
            folding_factor_next,
        );

        #[cfg(not(feature = "parallel"))]
        let leafs_iter = evals.chunks_exact(1 << folding_factor_next);
        #[cfg(feature = "parallel")]
        let leafs_iter = evals.par_chunks_exact(1 << folding_factor_next);
        let merkle_tree = {
            #[cfg(feature = "tracing")]
            let _span = span!(Level::INFO, "MerkleTree::new", size = leafs_iter.len()).entered();
            MerkleTree::new(
                &self.config.leaf_hash_params,
                &self.config.two_to_one_params,
                leafs_iter,
            )
            .unwrap()
        };

        let root = merkle_tree.root();
        prover_state.add_digest(root)?;

        // Handle OOD (Out-Of-Domain) samples
        let (ood_points, ood_answers) = sample_ood_points(
            prover_state,
            round_params.ood_samples,
            num_variables,
            |point| folded_coefficients.evaluate(point),
        )?;

        // PoW
        if round_params.pow_bits > 0. {
            #[cfg(feature = "tracing")]
            let _span = span!(
                Level::INFO,
                "challenge_pow",
                pow_bits = round_params.pow_bits
            )
            .entered();
            prover_state.challenge_pow::<PowStrategy>(round_params.pow_bits)?;
        }

        // STIR Queries
        let (stir_challenges, stir_challenges_indexes) = self.compute_stir_queries(
            prover_state,
            round_state,
            num_variables,
            round_params,
            ood_points,
        )?;

        let fold_size = 1 << folding_factor;
        let leaf_size = if round_state.round == 0 && self.config.batch_size > 1 {
            fold_size * self.config.batch_size
        } else {
            fold_size
        };
        let mut answers: Vec<_> = stir_challenges_indexes
            .iter()
            .map(|i| round_state.prev_merkle_answers[i * leaf_size..(i + 1) * leaf_size].to_vec())
            .collect();

        prover_state.hint::<Vec<Vec<F>>>(&answers)?;
        self.merkle_state.write_proof_hint(
            &round_state.prev_merkle,
            &stir_challenges_indexes,
            prover_state,
        )?;

        if round_state.round == 0 && self.config.batch_size > 1 {
            answers = rlc_batched_leaves(
                answers,
                fold_size,
                self.config.batch_size,
                round_state.batching_randomness,
            );
        }

        let mut stir_evaluations = ood_answers;
        stir_evaluations.extend(answers.iter().map(|answers| {
            CoefficientList::new(answers.clone()).evaluate(&round_state.folding_randomness)
        }));

        // Randomness for combination
        let [combination_randomness_gen] = prover_state.challenge_scalars()?;
        let combination_randomness =
            expand_randomness(combination_randomness_gen, stir_challenges.len());

        #[allow(clippy::map_unwrap_or)]
        let mut sumcheck_prover = round_state
            .sumcheck_prover
            .take()
            .map(|mut sumcheck_prover| {
                sumcheck_prover.add_new_equality(
                    &stir_challenges,
                    &stir_evaluations,
                    &combination_randomness,
                );
                sumcheck_prover
            })
            .unwrap_or_else(|| {
                let mut statement = Statement::new(folded_coefficients.num_variables());

                for (point, eval) in stir_challenges.into_iter().zip(stir_evaluations) {
                    let weights = Weights::evaluation(point);
                    statement.add_constraint(weights, eval);
                }
                SumcheckSingle::new(
                    folded_coefficients.clone(),
                    &statement,
                    combination_randomness[1],
                )
            });

        let folding_randomness = sumcheck_prover.compute_sumcheck_polynomials::<PowStrategy, _>(
            prover_state,
            folding_factor_next,
            round_params.folding_pow_bits,
        )?;

        let start_idx = self.config.folding_factor.total_number(round_state.round);
        let dst_randomness =
            &mut round_state.randomness_vec[start_idx..][..folding_randomness.0.len()];

        for (dst, src) in dst_randomness
            .iter_mut()
            .zip(folding_randomness.0.iter().rev())
        {
            *dst = *src;
        }

        // Update round state
        round_state.round += 1;
        round_state.domain = new_domain;
        round_state.sumcheck_prover = Some(sumcheck_prover);
        round_state.folding_randomness = folding_randomness;
        round_state.coefficients = folded_coefficients;
        round_state.prev_merkle = merkle_tree;
        round_state.prev_merkle_answers = evals;

        Ok(())
    }

    #[cfg_attr(feature = "tracing", instrument(skip_all, fields(size = folded_coefficients.num_coeffs())))]
    fn final_round<ProverState>(
        &self,
        prover_state: &mut ProverState,
        round_state: &mut RoundState<F, MerkleConfig>,
        folded_coefficients: &CoefficientList<F>,
    ) -> ProofResult<()>
    where
        ProverState:
            UnitToField<F> + UnitToBytes + FieldToUnitSerialize<F> + PoWChallenge + HintSerialize,
    {
        // Directly send coefficients of the polynomial to the verifier.
        println!(
            "[PROVER] Final round: sending {} coefficients: {:?}",
            folded_coefficients.coeffs().len(),
            folded_coefficients.coeffs()
        );
        println!(
            "[PROVER] Final round: num_variables={}",
            folded_coefficients.num_variables()
        );
        println!(
            "[PROVER] Final round: round_state.statement has {} constraints",
            round_state.statement.constraints.len()
        );
        prover_state.add_scalars(folded_coefficients.coeffs())?;

        // Precompute the folding factors for later use
        let folding_factor = self.config.folding_factor.at_round(round_state.round);

        // PoW
        if self.config.final_pow_bits > 0. {
            #[cfg(feature = "tracing")]
            let _span = span!(
                Level::INFO,
                "challenge_pow",
                pow_bits = self.config.final_pow_bits
            )
            .entered();
            prover_state.challenge_pow::<PowStrategy>(self.config.final_pow_bits)?;
        }

        // Final verifier queries and answers. The indices are over the
        // *folded* domain.
        let final_challenge_indexes = get_challenge_stir_queries(
            // The size of the *original* domain before folding
            round_state.domain.size(),
            // The folding factor we used to fold the previous polynomial
            folding_factor,
            self.config.final_queries,
            prover_state,
            &self.config.deduplication_strategy,
        )?;

        // Every query requires opening these many in the previous Merkle tree
        let fold_size = 1 << folding_factor;
        let answers = final_challenge_indexes
            .iter()
            .map(|i| round_state.prev_merkle_answers[i * fold_size..(i + 1) * fold_size].to_vec())
            .collect::<Vec<_>>();

        prover_state.hint::<Vec<Vec<F>>>(&answers)?;
        self.merkle_state.write_proof_hint(
            &round_state.prev_merkle,
            &final_challenge_indexes,
            prover_state,
        )?;

        // Final sumcheck
        if self.config.final_sumcheck_rounds > 0 {
            let final_folding_randomness = round_state
                .sumcheck_prover
                .clone()
                .unwrap_or_else(|| {
                    SumcheckSingle::new(folded_coefficients.clone(), &round_state.statement, F::ONE)
                })
                .compute_sumcheck_polynomials::<PowStrategy, _>(
                    prover_state,
                    self.config.final_sumcheck_rounds,
                    self.config.final_folding_pow_bits,
                )?;
            let start_idx = self.config.folding_factor.total_number(round_state.round);
            let rand_dst = &mut round_state.randomness_vec
                [start_idx..start_idx + final_folding_randomness.0.len()];

            for (dst, src) in rand_dst
                .iter_mut()
                .zip(final_folding_randomness.0.iter().rev())
            {
                *dst = *src;
            }
        }

        Ok(())
    }

    fn compute_stir_queries<ProverState>(
        &self,
        prover_state: &mut ProverState,
        round_state: &RoundState<F, MerkleConfig>,
        num_variables: usize,
        round_params: &RoundConfig<F>,
        ood_points: Vec<F>,
    ) -> ProofResult<(Vec<MultilinearPoint<F>>, Vec<usize>)>
    where
        ProverState: UnitToBytes,
    {
        let stir_challenges_indexes = get_challenge_stir_queries(
            round_state.domain.size(),
            self.config.folding_factor.at_round(round_state.round),
            round_params.num_queries,
            prover_state,
            &self.config.deduplication_strategy,
        )?;

        // Compute the generator of the folded domain, in the extension field
        let domain_scaled_gen = round_state
            .domain
            .backing_domain
            .element(1 << self.config.folding_factor.at_round(round_state.round));
        let stir_challenges = ood_points
            .into_iter()
            .chain(
                stir_challenges_indexes
                    .iter()
                    .map(|i| domain_scaled_gen.pow([*i as u64])),
            )
            .map(|univariate| MultilinearPoint::expand_from_univariate(univariate, num_variables))
            .collect();

        Ok((stir_challenges, stir_challenges_indexes))
    }
}

/// Represents the prover state during a single round of the WHIR protocol.
///
/// Each WHIR round folds the polynomial, commits to the new evaluations,
/// responds to verifier queries, and updates internal randomness for the next step.
/// This struct tracks all data needed to perform that round, and passes it forward
/// across recursive iterations.
pub(crate) struct RoundState<F, MerkleConfig>
where
    F: FftField,
    MerkleConfig: Config,
{
    /// Index of the current WHIR round (0-based).
    ///
    /// Increases after each folding iteration.
    pub(crate) round: usize,

    /// Domain over which the current polynomial is evaluated.
    ///
    /// Grows with each round due to NTT expansion.
    pub(crate) domain: Domain<F>,

    /// Optional sumcheck prover used to enforce constraints.
    ///
    /// Present in rounds with non-empty constraint systems.
    pub(crate) sumcheck_prover: Option<SumcheckSingle<F>>,

    /// Folding randomness sampled by the verifier.
    ///
    /// Used to reduce the number of variables in the polynomial.
    pub(crate) folding_randomness: MultilinearPoint<F>,

    /// Current polynomial in coefficient form.
    ///
    /// Folded and evaluated to produce new commitments and Merkle trees.
    pub(crate) coefficients: CoefficientList<F>,

    /// Merkle tree commitment to the polynomial evaluations from the previous round.
    ///
    /// Used to prove query openings from the folded function.
    pub(crate) prev_merkle: MerkleTree<MerkleConfig>,

    /// Flat list of evaluations corresponding to `prev_merkle` leaves.
    ///
    /// Each folded function is evaluated on a domain and split into leaves.
    pub(crate) prev_merkle_answers: Vec<F>,

    /// Accumulator for all folding randomness across rounds.
    ///
    /// Ordered with the most recent round’s randomness at the front.
    pub(crate) randomness_vec: Vec<F>,

    /// Constraint system being enforced in this round.
    ///
    /// May be updated during recursion as queries are folded and batched.
    pub(crate) statement: Statement<F>,

    pub(crate) batching_randomness: F,
}
