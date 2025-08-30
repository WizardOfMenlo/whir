use std::marker::PhantomData;

use ark_crypto_primitives::merkle_tree::{Config, MultiPath};
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
        prover::RootPath,
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
        for _ in 0..rounds {
            // Receive this round's sumcheck polynomial
            let sumcheck_poly_evals: [_; 3] = verifier_state.next_scalars()?;
            let sumcheck_poly = SumcheckPolynomial::new(sumcheck_poly_evals.to_vec(), 1);

            // Verify claimed sum is consistent with polynomial
            if sumcheck_poly.sum_over_boolean_hypercube() != *claimed_sum {
                return Err(ProofError::InvalidProof);
            }

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

        // Receive merkle proof for leaf indices
        let merkle_proof: MultiPath<MerkleConfig> = verifier_state.hint()?;
        if merkle_proof.leaf_indexes != indices {
            return Err(ProofError::InvalidProof);
        }

        // Verify merkle proof
        let correct = merkle_proof
            .verify(
                &self.params.leaf_hash_params,
                &self.params.two_to_one_params,
                root,
                answers.iter().map(|a| a.as_ref()),
            )
            .map_err(|_e| ProofError::InvalidProof)?;
        if !correct {
            return Err(ProofError::InvalidProof);
        }

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
