# PreFold Approach - Simplified Implementation Plan

**Constraints**:
- **Exactly 2 witnesses** (not N witnesses)
- **Maximum arity difference: 1 variable** (not arbitrary)
- **Pre-fold exactly once** (not multiple rounds)

**Date**: 2026-01-19  
**Status**: Simplified for specific use case

---

## Table of Contents

1. [Simplified Overview](#simplified-overview)
2. [Simplified Architecture](#simplified-architecture)
3. [File Changes](#file-changes)
4. [Implementation Steps](#implementation-steps)
5. [Testing](#testing)

---

## Simplified Overview

### Problem
You have two witnesses:
- `f: {0,1}^n → F` (smaller)
- `g: {0,1}^{n+1} → F` (larger, exactly 1 variable more)

### Solution
1. Commit to both `f` and `g`
2. Sample `α` from transcript
3. Fold `g` once: `g'(y) = g(α, y)` where `g': {0,1}^n → F`
4. Commit to `g'`
5. Sample `γ` from transcript
6. Batch: `h = f + γ·g'`
7. Run standard WHIR on `h`

**No MultiWitnessConfig needed!** Just modify existing batch proving to handle this specific case.

---

## Simplified Architecture

### New Enum: BatchingMode

Instead of complex `MultiWitnessConfig`, just add a mode to `prove_batch`:

```rust
pub enum BatchingMode {
    /// Standard batching: all witnesses same arity
    Standard,
    
    /// Pre-fold second witness by 1 variable
    /// Assumes: exactly 2 witnesses, second has arity+1
    PreFoldSecond,
}
```

### Modified Function Signature

```rust
impl Prover {
    pub fn prove_batch<ProverState>(
        &self,
        prover_state: &mut ProverState,
        statements: &[Statement<F>],
        witnesses: &[Witness<F, MerkleConfig>],
        mode: BatchingMode,  // NEW PARAMETER
    ) -> ProofResult<(MultilinearPoint<F>, Vec<F>)>
    {
        match mode {
            BatchingMode::Standard => {
                // Existing batch proving logic
                self.prove_batch_standard(prover_state, statements, witnesses)
            }
            BatchingMode::PreFoldSecond => {
                // New pre-fold logic
                assert_eq!(witnesses.len(), 2);
                assert_eq!(
                    witnesses[1].polynomial.num_variables(),
                    witnesses[0].polynomial.num_variables() + 1
                );
                self.prove_batch_prefold(prover_state, statements, witnesses)
            }
        }
    }
}
```

---

## File Changes

### 1. `src/whir/prover.rs` - Main Changes

#### Add BatchingMode Enum

```rust
/// Batching mode for prove_batch
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BatchingMode {
    /// Standard batching: all witnesses have same arity
    Standard,
    
    /// Pre-fold second witness: exactly 2 witnesses, second has arity+1
    PreFoldSecond,
}
```

**Location**: Add after the `Prover` struct definition (around line 45)

---

#### Modify `prove_batch` Signature

**Current** (line 207):
```rust
pub fn prove_batch<ProverState>(
    &self,
    prover_state: &mut ProverState,
    statements: &[Statement<F>],
    witnesses: &[Witness<F, MerkleConfig>],
) -> ProofResult<(MultilinearPoint<F>, Vec<F>)>
```

**New**:
```rust
pub fn prove_batch<ProverState>(
    &self,
    prover_state: &mut ProverState,
    statements: &[Statement<F>],
    witnesses: &[Witness<F, MerkleConfig>],
    mode: BatchingMode,  // NEW
) -> ProofResult<(MultilinearPoint<F>, Vec<F>)>
where
    ProverState: /* same trait bounds */
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
```

---

#### Add `prove_batch_standard` (Refactored Existing Logic)

**Action**: Rename current `prove_batch` body to `prove_batch_standard`

```rust
fn prove_batch_standard<ProverState>(
    &self,
    prover_state: &mut ProverState,
    statements: &[Statement<F>],
    witnesses: &[Witness<F, MerkleConfig>],
) -> ProofResult<(MultilinearPoint<F>, Vec<F>)>
where
    ProverState: /* same trait bounds */
{
    // All existing prove_batch logic goes here unchanged
    // (lines 207-564 in current code)
}
```

---

#### Add `prove_batch_prefold` (New Logic)

```rust
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
    // Validation
    assert_eq!(witnesses.len(), 2, "PreFoldSecond mode requires exactly 2 witnesses");
    let num_vars_0 = witnesses[0].polynomial.num_variables();
    let num_vars_1 = witnesses[1].polynomial.num_variables();
    assert_eq!(
        num_vars_1,
        num_vars_0 + 1,
        "Second witness must have exactly 1 more variable"
    );
    
    // Phase 1: Commit to evaluation matrix BEFORE pre-folding
    // (Same as standard batch proving, but for 2 witnesses with different arities)
    
    let mut all_constraint_weights = Vec::new();
    
    // OOD constraints from both witnesses
    for witness in witnesses {
        for point in &witness.ood_points {
            let ml_point = MultilinearPoint::expand_from_univariate(
                *point,
                witness.polynomial.num_variables(),  // Use actual arity
            );
            all_constraint_weights.push(Weights::evaluation(ml_point));
        }
    }
    
    // Statement constraints
    for statement in statements {
        for constraint in &statement.constraints {
            all_constraint_weights.push(constraint.weights.clone());
        }
    }
    
    // Build N×M matrix (2×M in this case)
    let mut constraint_evals_matrix = Vec::with_capacity(2);
    for witness in witnesses {
        let mut poly_evals = Vec::with_capacity(all_constraint_weights.len());
        for weights in &all_constraint_weights {
            // Evaluate witness at constraint point
            // If arities don't match, evaluation is undefined (constraint doesn't apply)
            if weights.num_variables() == witness.polynomial.num_variables() {
                let eval = weights.evaluate(&witness.polynomial);
                poly_evals.push(eval);
            } else {
                // Constraint doesn't apply to this witness (different arity)
                poly_evals.push(F::ZERO);  // Placeholder
            }
        }
        constraint_evals_matrix.push(poly_evals);
    }
    
    // Commit matrix to transcript
    let all_evals_flat: Vec<F> = constraint_evals_matrix
        .iter()
        .flat_map(|row| row.iter().copied())
        .collect();
    prover_state.add_scalars(&all_evals_flat)?;
    
    // Phase 2: Pre-fold second witness
    
    // Sample pre-folding randomness α
    let [alpha] = prover_state.challenge_scalars()?;
    let folding_randomness = MultilinearPoint(vec![alpha]);
    
    // PoW
    if self.config.starting_folding_pow_bits > 0. {
        prover_state.challenge_pow::<PowStrategy>(self.config.starting_folding_pow_bits)?;
    }
    
    // Fold g
    let g_folded = witnesses[1].polynomial.fold(&folding_randomness);
    assert_eq!(g_folded.num_variables(), num_vars_0);
    
    // RS encode folded polynomial
    let new_domain = self.config.starting_domain.scale(2);
    let expansion = new_domain.size() / g_folded.num_coeffs();
    let folding_factor_next = self.config.folding_factor.at_round(1);
    
    let g_folded_evals = self.config.reed_solomon.interleaved_encode(
        g_folded.coeffs(),
        expansion,
        folding_factor_next,
    );
    
    // Build Merkle tree for g'
    #[cfg(not(feature = "parallel"))]
    let leafs_iter = g_folded_evals.chunks_exact(1 << folding_factor_next);
    #[cfg(feature = "parallel")]
    let leafs_iter = g_folded_evals.par_chunks_exact(1 << folding_factor_next);
    
    let g_folded_merkle = MerkleTree::new(
        &self.config.leaf_hash_params,
        &self.config.two_to_one_params,
        leafs_iter,
    ).unwrap();
    
    prover_state.add_digest(g_folded_merkle.root())?;
    
    // OOD samples for g'
    let (g_folded_ood_points, g_folded_ood_answers) = sample_ood_points(
        prover_state,
        self.config.committment_ood_samples,
        num_vars_0,
        |point| g_folded.evaluate(point),
    )?;
    
    // PoW
    if self.config.round_parameters[0].pow_bits > 0. {
        prover_state.challenge_pow::<PowStrategy>(self.config.round_parameters[0].pow_bits)?;
    }
    
    // STIR queries on ORIGINAL g (not folded)
    // This proves consistency: g'(z) = g(α, z)
    let stir_indexes = get_challenge_stir_queries(
        self.config.starting_domain.size(),
        self.config.folding_factor.at_round(0),
        self.config.round_parameters[0].num_queries,
        prover_state,
        &self.config.deduplication_strategy,
    )?;
    
    let fold_size = 1 << self.config.folding_factor.at_round(0);
    let original_g_answers: Vec<_> = stir_indexes
        .iter()
        .map(|&i| witnesses[1].merkle_leaves[i * fold_size..(i + 1) * fold_size].to_vec())
        .collect();
    
    prover_state.hint::<Vec<Vec<F>>>(&original_g_answers)?;
    self.merkle_state.write_proof_hint(
        &witnesses[1].merkle_tree,
        &stir_indexes,
        prover_state,
    )?;
    
    // Also provide corresponding evaluations from g'
    let domain_scaled_gen = self
        .config
        .starting_domain
        .backing_domain
        .element(1 << self.config.folding_factor.at_round(0));
    
    let g_folded_stir_evals: Vec<F> = stir_indexes
        .iter()
        .map(|&i| {
            let univariate_point = domain_scaled_gen.pow([i as u64]);
            let ml_point = MultilinearPoint::expand_from_univariate(univariate_point, num_vars_0);
            g_folded.evaluate(&ml_point)
        })
        .collect();
    
    prover_state.add_scalars(&g_folded_stir_evals)?;
    
    // Phase 3: Sample batching randomness γ
    let [batching_randomness] = prover_state.challenge_scalars()?;
    
    // Phase 4: Materialize batched polynomial
    // h = f + γ·g'
    let mut batched_coeffs = witnesses[0].polynomial.coeffs().to_vec();
    for (dst, src) in batched_coeffs.iter_mut().zip(g_folded.coeffs()) {
        *dst += batching_randomness * src;
    }
    let batched_poly = CoefficientList::new(batched_coeffs);
    
    // Phase 5: Build combined statement
    // Only use constraints that apply to common arity (num_vars_0)
    let mut combined_statement = Statement::new(num_vars_0);
    
    // Add OOD constraints from f (witness 0)
    for (point, answer) in witnesses[0].ood_points.iter().zip(&witnesses[0].ood_answers) {
        let ml_point = MultilinearPoint::expand_from_univariate(*point, num_vars_0);
        combined_statement.add_constraint(Weights::evaluation(ml_point), *answer);
    }
    
    // Add OOD constraints from g' (folded)
    for (point, answer) in g_folded_ood_points.iter().zip(&g_folded_ood_answers) {
        let ml_point = MultilinearPoint::expand_from_univariate(*point, num_vars_0);
        let combined_eval = *answer * batching_randomness;
        combined_statement.add_constraint(Weights::evaluation(ml_point), combined_eval);
    }
    
    // Add statement constraints (only those with arity = num_vars_0)
    for (stmt_idx, statement) in statements.iter().enumerate() {
        for constraint in &statement.constraints {
            if constraint.weights.num_variables() == num_vars_0 {
                // This constraint applies to witness stmt_idx
                let mut combined_eval = F::ZERO;
                if stmt_idx == 0 {
                    combined_eval += constraint.sum;
                } else {
                    combined_eval += batching_randomness * constraint.sum;
                }
                combined_statement.add_constraint(
                    constraint.weights.clone(),
                    combined_eval,
                );
            }
        }
    }
    
    // Phase 6: Run standard WHIR on batched polynomial
    // Use existing logic from here on
    
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
    
    let mut randomness_vec = Vec::with_capacity(num_vars_0);
    randomness_vec.extend(folding_randomness.0.iter().rev().copied());
    randomness_vec.resize(num_vars_0, F::ZERO);
    
    let mut round_state = RoundState {
        domain: self.config.starting_domain.clone(),
        round: 0,
        sumcheck_prover: Some(sumcheck),
        folding_randomness,
        coefficients: batched_poly,
        prev_merkle: witnesses[0].merkle_tree.clone(),  // Use f's tree initially
        prev_merkle_answers: witnesses[0].merkle_leaves.clone(),
        randomness_vec,
        statement: combined_statement,
        batching_randomness,
    };
    
    // Run standard WHIR rounds
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
```

**Location**: Add after `prove_batch` (around line 565)

---

### 2. `src/whir/verifier.rs` - Main Changes

#### Modify `verify_batch` Signature

**Current** (line 217):
```rust
pub fn verify_batch(
    &self,
    verifier_state: &mut VerifierState,
    parsed_commitments: &[ParsedCommitment<F, MerkleConfig::InnerDigest>],
    statements: &[Statement<F>],
) -> ProofResult<(MultilinearPoint<F>, Vec<F>)>
```

**New**:
```rust
pub fn verify_batch(
    &self,
    verifier_state: &mut VerifierState,
    parsed_commitments: &[ParsedCommitment<F, MerkleConfig::InnerDigest>],
    statements: &[Statement<F>],
    mode: BatchingMode,  // NEW
) -> ProofResult<(MultilinearPoint<F>, Vec<F>)>
{
    match mode {
        BatchingMode::Standard => {
            self.verify_batch_standard(verifier_state, parsed_commitments, statements)
        }
        BatchingMode::PreFoldSecond => {
            self.verify_batch_prefold(verifier_state, parsed_commitments, statements)
        }
    }
}
```

---

#### Add `verify_batch_prefold`

```rust
fn verify_batch_prefold(
    &self,
    verifier_state: &mut VerifierState,
    parsed_commitments: &[ParsedCommitment<F, MerkleConfig::InnerDigest>],
    statements: &[Statement<F>],
) -> ProofResult<(MultilinearPoint<F>, Vec<F>)>
where
    VerifierState: /* same trait bounds */
{
    // Validation
    assert_eq!(parsed_commitments.len(), 2);
    assert_eq!(statements.len(), 2);
    
    let num_vars_0 = parsed_commitments[0].num_variables;
    let num_vars_1 = parsed_commitments[1].num_variables;
    assert_eq!(num_vars_1, num_vars_0 + 1);
    
    // Phase 1: Read evaluation matrix
    let mut all_constraints_info = Vec::new();
    
    // OOD constraints from both commitments
    for commitment in parsed_commitments {
        for point in &commitment.ood_points {
            let ml_point = MultilinearPoint::expand_from_univariate(
                *point,
                commitment.num_variables,
            );
            all_constraints_info.push((Weights::evaluation(ml_point), commitment.num_variables));
        }
    }
    
    // Statement constraints
    for statement in statements {
        for constraint in &statement.constraints {
            all_constraints_info.push((
                constraint.weights.clone(),
                statement.num_variables(),
            ));
        }
    }
    
    // Read 2×M evaluation matrix
    let mut constraint_evals_matrix = Vec::with_capacity(2);
    for _ in 0..2 {
        let mut row = vec![F::ZERO; all_constraints_info.len()];
        verifier_state.fill_next_scalars(&mut row)?;
        constraint_evals_matrix.push(row);
    }
    
    // Phase 2: Verify pre-folding
    
    // Sample pre-folding randomness α (same as prover)
    let [alpha] = verifier_state.challenge_scalars()?;
    let folding_randomness = MultilinearPoint(vec![alpha]);
    
    // PoW
    if self.params.starting_folding_pow_bits > 0. {
        verifier_state.challenge_pow::<PowStrategy>(self.params.starting_folding_pow_bits)?;
    }
    
    // Parse g' commitment
    let g_folded_commitment = ParsedCommitment::parse(
        verifier_state,
        num_vars_0,
        self.params.committment_ood_samples,
    )?;
    
    // PoW
    if self.params.round_parameters[0].pow_bits > 0. {
        verifier_state.challenge_pow::<PowStrategy>(self.params.round_parameters[0].pow_bits)?;
    }
    
    // Verify consistency: g'(z) = g(α, z) via STIR queries
    let stir_indexes = get_challenge_stir_queries(
        self.params.starting_domain.size(),
        self.params.folding_factor.at_round(0),
        self.params.round_parameters[0].num_queries,
        verifier_state,
        &self.params.deduplication_strategy,
    )?;
    
    // Read original g answers
    let original_g_answers: Vec<Vec<F>> = verifier_state.hint()?;
    
    // Verify Merkle proofs for original g
    self.merkle_state.read_and_verify_proof::<VerifierState, _>(
        verifier_state,
        &stir_indexes,
        &parsed_commitments[1].root,
        original_g_answers.iter().map(|v| v.as_slice()),
    )?;
    
    // Read g' evaluations at STIR points
    let mut g_folded_stir_evals = vec![F::ZERO; stir_indexes.len()];
    verifier_state.fill_next_scalars(&mut g_folded_stir_evals)?;
    
    // Check consistency: For each STIR query, g'(z) should equal
    // the evaluation of the unfolded chunk at α
    let fold_size = 1 << self.params.folding_factor.at_round(0);
    for (answer_chunk, &expected_folded) in original_g_answers.iter().zip(&g_folded_stir_evals) {
        // Evaluate the chunk polynomial at α
        let chunk_poly = CoefficientList::new(answer_chunk.clone());
        let actual_folded = chunk_poly.evaluate(&folding_randomness);
        
        if actual_folded != expected_folded {
            return Err(ProofError::InvalidProof);
        }
    }
    
    // Phase 3: Sample batching randomness γ
    let [batching_randomness] = verifier_state.challenge_scalars()?;
    
    // Phase 4: Build combined constraints using RLC
    let mut combined_constraints = Vec::new();
    
    // Combine evaluations from matrix
    for (constraint_idx, (weights, arity)) in all_constraints_info.into_iter().enumerate() {
        if arity == num_vars_0 {
            // This constraint applies at common arity
            let mut combined_eval = F::ZERO;
            let mut pow = F::ONE;
            for poly_evals in &constraint_evals_matrix {
                combined_eval += pow * poly_evals[constraint_idx];
                pow *= batching_randomness;
            }
            combined_constraints.push(Constraint {
                weights,
                sum: combined_eval,
                defer_evaluation: false,
            });
        }
    }
    
    // Phase 5: Run standard WHIR verification on batched polynomial
    
    let [combination_randomness_gen] = verifier_state.challenge_scalars()?;
    let combination_randomness = expand_randomness(
        combination_randomness_gen,
        combined_constraints.len(),
    );
    
    let mut claimed_sum = F::ZERO;
    for (constraint, &rand) in combined_constraints.iter().zip(&combination_randomness) {
        claimed_sum += constraint.sum * rand;
    }
    
    let folding_randomness = self.verify_sumcheck_rounds(
        verifier_state,
        &mut claimed_sum,
        self.params.folding_factor.at_round(0),
        self.params.starting_folding_pow_bits,
    )?;
    
    let mut round_constraints = vec![(combination_randomness, combined_constraints)];
    let mut round_folding_randomness = vec![folding_randomness];
    
    // Continue with standard WHIR rounds...
    // (Similar to existing verify_batch logic)
    
    // ... (rest of verification)
    
    Ok((folding_randomness, deferred))
}
```

**Location**: Add after `verify_batch` (around line 517)

---

### 3. `src/whir/domainsep.rs` - Simplified Changes

No need for complex multi-arity support. Just add:

```rust
impl WhirDomainSeparator for DomainSeparator {
    /// Add pre-fold phase for exactly 1 variable
    fn add_prefold_single_variable(self) -> Self {
        self.add_bytes(b"whir_prefold_single_var")
            .add_bytes(b"prefold_folding_randomness")
            .add_bytes(b"prefold_folded_commitment")
    }
}
```

**Usage**:
```rust
// In your application code
let domainsep = DomainSeparator::new("🌪️")
    .commit_statement(&config)  // For f (n vars)
    .commit_statement(&config)  // For g (n+1 vars)
    .add_prefold_single_variable()  // Pre-fold g
    .add_whir_batch_proof(&config, 2, total_constraints);
```

**Location**: Add to impl block (around line 150)

---

## Implementation Steps

### Phase 1: Add BatchingMode (1-2 hours)

1. Add `BatchingMode` enum to `src/whir/prover.rs`
2. Update `prove_batch` signature
3. Refactor existing logic into `prove_batch_standard`
4. Update all test calls to use `BatchingMode::Standard`

**Test**: Existing tests should still pass

---

### Phase 2: Implement `prove_batch_prefold` (1-2 days)

1. Copy the code from above
2. Adjust for your specific field types
3. Test with simple 2-witness example

**Test**:
```rust
#[test]
fn test_prefold_prove() {
    let poly_10 = CoefficientList::new(vec![F::ONE; 1 << 10]);
    let poly_11 = CoefficientList::new(vec![F::from(2); 1 << 11]);
    
    // ... create witnesses and statements
    
    let result = prover.prove_batch(
        &mut prover_state,
        &[statement_10, statement_11],
        &[witness_10, witness_11],
        BatchingMode::PreFoldSecond,
    );
    
    assert!(result.is_ok());
}
```

---

### Phase 3: Implement `verify_batch_prefold` (1-2 days)

1. Copy the code from above
2. Match the prover's transcript exactly
3. Test end-to-end

**Test**:
```rust
#[test]
fn test_prefold_end_to_end() {
    // ... (prove as above)
    
    let result = verifier.verify_batch(
        &mut verifier_state,
        &[commitment_10, commitment_11],
        &[statement_10, statement_11],
        BatchingMode::PreFoldSecond,
    );
    
    assert!(result.is_ok());
}
```

---

### Phase 4: Update Domain Separator (few hours)

1. Add `add_prefold_single_variable` method
2. Update your application code to use it
3. Verify transcript consistency

---

### Phase 5: Integration & Testing (1 day)

1. Test with your actual provekit integration
2. Test edge cases
3. Test malicious prover attempts

---

## Testing

### Unit Tests

```rust
#[test]
fn test_batching_mode_standard() {
    // Test existing functionality still works
    let witnesses = vec![witness_10a, witness_10b];  // Both 10 vars
    
    let result = prover.prove_batch(
        &mut prover_state,
        statements,
        &witnesses,
        BatchingMode::Standard,
    );
    
    assert!(result.is_ok());
}

#[test]
fn test_batching_mode_prefold() {
    let witnesses = vec![witness_10, witness_11];  // 10 and 11 vars
    
    let result = prover.prove_batch(
        &mut prover_state,
        statements,
        &witnesses,
        BatchingMode::PreFoldSecond,
    );
    
    assert!(result.is_ok());
}

#[test]
#[should_panic(expected = "exactly 2 witnesses")]
fn test_prefold_wrong_count() {
    let witnesses = vec![witness_10];  // Only 1 witness
    
    prover.prove_batch(
        &mut prover_state,
        statements,
        &witnesses,
        BatchingMode::PreFoldSecond,
    );
}

#[test]
#[should_panic(expected = "exactly 1 more variable")]
fn test_prefold_wrong_arity_diff() {
    let witnesses = vec![witness_10, witness_12];  // Difference of 2
    
    prover.prove_batch(
        &mut prover_state,
        statements,
        &witnesses,
        BatchingMode::PreFoldSecond,
    );
}
```

### Integration Tests

```rust
#[test]
fn test_prefold_soundness() {
    // Generate valid proof
    let (eval_point, deferred) = prover.prove_batch(
        &mut prover_state,
        &[statement_10, statement_11],
        &[witness_10, witness_11],
        BatchingMode::PreFoldSecond,
    ).unwrap();
    
    // Verify should succeed
    let result = verifier.verify_batch(
        &mut verifier_state,
        &[commitment_10, commitment_11],
        &[statement_10, statement_11],
        BatchingMode::PreFoldSecond,
    );
    
    assert!(result.is_ok());
    
    // Verify evaluations are correct
    let (verify_eval_point, verify_deferred) = result.unwrap();
    assert_eq!(eval_point, verify_eval_point);
    assert_eq!(deferred, verify_deferred);
}

#[test]
fn test_prefold_malicious_prover() {
    // Prover commits to g
    let witness_11 = create_witness(&poly_11);
    
    // Sample alpha
    let [alpha] = prover_state.challenge_scalars()?;
    
    // Prover folds g correctly
    let g_folded = poly_11.fold(&MultilinearPoint(vec![alpha]));
    
    // But prover commits to DIFFERENT polynomial
    let g_malicious = g_folded.clone();
    g_malicious.coeffs_mut()[0] += F::ONE;  // Tamper
    
    let malicious_witness = create_witness(&g_malicious);
    
    // Try to batch
    let result = prover.prove_batch(
        &mut prover_state,
        &[statement_10, statement_11],
        &[witness_10, malicious_witness],
        BatchingMode::PreFoldSecond,
    );
    
    // Proof generation succeeds (prover doesn't check)
    assert!(result.is_ok());
    
    // But verification should fail
    let verify_result = verifier.verify_batch(
        &mut verifier_state,
        &[commitment_10, commitment_11_malicious],
        &[statement_10, statement_11],
        BatchingMode::PreFoldSecond,
    );
    
    assert!(verify_result.is_err());
}
```

---

## Complexity Comparison

### Original Complex Plan
- New types: 4 (MultiWitnessConfig, PreFoldInfo, PreFoldedWitness, PreFoldState)
- Modified functions: 10+
- Lines of code: ~2000+
- Testing scenarios: 20+
- Implementation time: 6 weeks

### Simplified Plan
- New types: 1 (BatchingMode enum)
- Modified functions: 4 (prove_batch, verify_batch, 2 new helper methods)
- Lines of code: ~500
- Testing scenarios: 8
- Implementation time: **1 week**

---

## Key Simplifications

1. **No MultiWitnessConfig**: Just add a `mode` parameter
2. **No complex looping**: Exactly 2 witnesses, exactly 1 pre-fold
3. **No arbitrary arity handling**: Hardcoded difference of 1
4. **Simpler domain separator**: Just one method addition
5. **Easier testing**: Fewer edge cases

---

## Migration Path

### Backward Compatibility

Old code:
```rust
prover.prove_batch(&mut state, statements, witnesses)?;
```

New code:
```rust
prover.prove_batch(&mut state, statements, witnesses, BatchingMode::Standard)?;
```

**Or** keep old signature and add new one:
```rust
// Keep existing (deprecated)
pub fn prove_batch(...) -> Result { 
    self.prove_batch_with_mode(..., BatchingMode::Standard)
}

// Add new
pub fn prove_batch_with_mode(..., mode: BatchingMode) -> Result {
    // Implementation
}
```

---

## Next Steps

1. **Review this simplified plan** with your team
2. **Start with Phase 1** (add BatchingMode enum)
3. **Test incrementally** after each phase
4. **Integration** with provekit after verification works

---

## Summary

Your constraints make this **much simpler**:
- ✅ No complex multi-witness configuration
- ✅ No variable-length pre-folding loops
- ✅ Straightforward implementation
- ✅ **1 week instead of 6 weeks**

The core idea remains the same, but the implementation is now focused and manageable!

---

**Ready to start? Begin with Phase 1!** 🚀

