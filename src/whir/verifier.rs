use std::{iter, marker::PhantomData};

use ark_crypto_primitives::merkle_tree::{Config, MultiPath};
use ark_ff::FftField;
use ark_poly::EvaluationDomain;
use spongefish::{
    codecs::arkworks_algebra::{FieldToUnitDeserialize, UnitToField},
    ProofError, ProofResult, UnitToBytes,
};
use spongefish_pow::{self, PoWChallenge};

use super::{
    committer::reader::ParsedCommitment,
    parameters::{RoundConfig, WhirConfig},
    parsed_proof::ParsedRound,
    statement::{Constraint, Statement, Weights},
    utils::HintDeserialize,
};
use crate::{
    poly_utils::{coeffs::CoefficientList, multilinear::MultilinearPoint},
    sumcheck::SumcheckPolynomial,
    utils::expand_randomness,
    whir::utils::{get_challenge_stir_queries, DigestToUnitDeserialize},
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
        let mut constraints_at_round = Vec::new();

        // Optional initial sumcheck round
        let mut claimed_sum = F::ZERO;
        let mut folding_randomness;
        let initial_combination_randomness;
        if self.params.initial_statement {
            // Collect initial constraints
            let constraints: Vec<_> = parsed_commitment
                .oods_constraints()
                .into_iter()
                .chain(statement.constraints.iter().cloned())
                .collect();

            // Derive combination randomness and first sumcheck polynomial
            let [combination_randomness_gen] = verifier_state.challenge_scalars()?;
            initial_combination_randomness =
                expand_randomness(combination_randomness_gen, constraints.len());

            // Compute initial sum
            claimed_sum = constraints
                .iter()
                .zip(&initial_combination_randomness)
                .map(|(c, rand)| c.sum * rand)
                .sum();

            // Initial sumcheck
            folding_randomness = self.verify_sumcheck_rounds(
                verifier_state,
                &mut claimed_sum,
                self.params.folding_factor.at_round(0),
                self.params.starting_folding_pow_bits,
            )?;

            // Store constraints for final evaluation
            constraints_at_round.push(constraints);
        } else {
            assert_eq!(parsed_commitment.ood_points.len(), 0);
            assert!(statement.constraints.is_empty());

            initial_combination_randomness = vec![F::ONE];

            let mut folding_randomness_vec = vec![F::ZERO; self.params.folding_factor.at_round(0)];
            verifier_state.fill_challenge_scalars(&mut folding_randomness_vec)?;
            folding_randomness = MultilinearPoint(folding_randomness_vec);

            // PoW
            self.verify_proof_of_work(verifier_state, self.params.starting_folding_pow_bits)?;

            constraints_at_round.push(vec![]);
        }

        let mut prev_commitment = parsed_commitment.clone();
        let mut num_variables =
            self.params.mv_parameters.num_variables - self.params.folding_factor.at_round(0);
        let mut domain_gen = self.params.starting_domain.backing_domain.group_gen();
        let mut exp_domain_gen = domain_gen.pow([1 << self.params.folding_factor.at_round(0)]);
        let mut domain_gen_inv = self.params.starting_domain.backing_domain.group_gen_inv();
        let mut domain_size = self.params.starting_domain.size();
        let mut rounds = vec![];

        for round_index in 0..self.params.n_rounds() {
            let round_params = &self.params.round_parameters[round_index];
            let folding_factor = self.params.folding_factor.at_round(round_index);

            let new_commitment = ParsedCommitment::<F, MerkleConfig::InnerDigest>::parse(
                verifier_state,
                num_variables,
                round_params.ood_samples,
            )?;

            let stir_challenges_indexes = get_challenge_stir_queries(
                domain_size,
                folding_factor,
                round_params.num_queries,
                verifier_state,
            )?;

            let answers = self.verify_merkle_proof(
                verifier_state,
                &prev_commitment.root,
                &stir_challenges_indexes,
            )?;

            self.verify_proof_of_work(verifier_state, round_params.pow_bits)?;

            // Compute STIR Constraints
            let folds = self.params.fold_optimisation.stir_evaluations_verifier(
                self.params,
                round_index,
                domain_gen_inv,
                &stir_challenges_indexes,
                &folding_randomness,
                &answers,
            );
            let stir_constraints = stir_challenges_indexes
                .iter()
                .map(|&index| exp_domain_gen.pow([index as u64]))
                .zip(&folds)
                .map(|(point, &value)| Constraint {
                    weights: Weights::univariate(point, num_variables),
                    sum: value,
                    deferred: false,
                });

            // Add OODS and STIR constraints to claimed_sum
            let constraints: Vec<Constraint<F>> = new_commitment
                .oods_constraints()
                .into_iter()
                .chain(stir_constraints)
                .collect();
            let [combination_randomness_gen] = verifier_state.challenge_scalars()?;
            let combination_randomness = expand_randomness(
                combination_randomness_gen,
                stir_challenges_indexes.len() + round_params.ood_samples,
            );
            claimed_sum += constraints
                .iter()
                .zip(&combination_randomness)
                .map(|(c, rand)| c.sum * rand)
                .sum::<F>();

            let new_folding_randomness = self.verify_sumcheck_rounds(
                verifier_state,
                &mut claimed_sum,
                self.params.folding_factor.at_round(round_index + 1),
                round_params.folding_pow_bits,
            )?;

            rounds.push(ParsedRound {
                folding_randomness,
                ood_points: new_commitment.ood_points.clone(),
                stir_challenges_points: stir_challenges_indexes
                    .iter()
                    .map(|index| exp_domain_gen.pow([*index as u64]))
                    .collect(),
                combination_randomness,
            });
            constraints_at_round.push(constraints);

            folding_randomness = new_folding_randomness;

            prev_commitment = new_commitment;
            num_variables -= folding_factor;
            domain_gen = domain_gen * domain_gen;
            exp_domain_gen =
                domain_gen.pow([1 << self.params.folding_factor.at_round(round_index + 1)]);
            domain_gen_inv = domain_gen_inv * domain_gen_inv;
            domain_size /= 2;
        }

        let mut final_coefficients = vec![F::ZERO; 1 << self.params.final_sumcheck_rounds];
        verifier_state.fill_next_scalars(&mut final_coefficients)?;
        let final_coefficients = CoefficientList::new(final_coefficients);

        // Final queries verify
        let final_randomness_indexes = get_challenge_stir_queries(
            domain_size,
            self.params.folding_factor.at_round(self.params.n_rounds()),
            self.params.final_queries,
            verifier_state,
        )?;
        let final_randomness_points: Vec<F> = final_randomness_indexes
            .iter()
            .map(|index| exp_domain_gen.pow([*index as u64]))
            .collect();

        let final_randomness_answers = self.verify_merkle_proof(
            verifier_state,
            &prev_commitment.root,
            &final_randomness_indexes,
        )?;

        self.verify_proof_of_work(verifier_state, self.params.final_pow_bits)?;

        // Check the foldings computed from the proof match the evaluations of the polynomial
        let final_folds = self.params.fold_optimisation.stir_evaluations_verifier(
            self.params,
            self.params.n_rounds(),
            domain_gen_inv,
            &final_randomness_indexes,
            &folding_randomness,
            &final_randomness_answers,
        );

        let final_evaluations = final_coefficients.evaluate_at_univariate(&final_randomness_points);
        if !final_folds
            .iter()
            .zip(final_evaluations)
            .all(|(&fold, eval)| fold == eval)
        {
            return Err(ProofError::InvalidProof);
        }

        let final_sumcheck_randomness = self.verify_sumcheck_rounds(
            verifier_state,
            &mut claimed_sum,
            self.params.final_sumcheck_rounds,
            self.params.final_folding_pow_bits,
        )?;

        let deferred: Vec<F> = verifier_state.hint()?;

        // Compute folding randomness across all rounds.
        let folding_randomness = MultilinearPoint(
            iter::once(&final_sumcheck_randomness.0)
                .chain(iter::once(&folding_randomness.0))
                .chain(rounds.iter().rev().map(|r| &r.folding_randomness.0))
                .flatten()
                .copied()
                .collect(),
        );

        // Final v · w Check
        let prev_sumcheck_poly_eval = claimed_sum;

        // Check the final sumcheck evaluation
        let evaluation_of_v_poly = self.compute_w_poly(
            parsed_commitment,
            statement,
            folding_randomness.clone(),
            &initial_combination_randomness,
            &rounds,
            &deferred,
        );
        let final_value = final_coefficients.evaluate(&final_sumcheck_randomness);

        if prev_sumcheck_poly_eval != evaluation_of_v_poly * final_value {
            return Err(ProofError::InvalidProof);
        }

        Ok((folding_randomness, deferred))
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

            // Receive folding randomness
            let [folding_randomness_single] = verifier_state.challenge_scalars()?;
            randomness.push(folding_randomness_single);

            // Update claimed sum using folding randomness
            *claimed_sum = sumcheck_poly.evaluate_at_point(&folding_randomness_single.into());

            // Proof of work per round
            self.verify_proof_of_work(verifier_state, proof_of_work)?;
        }

        randomness.reverse();
        Ok(MultilinearPoint(randomness))
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

    fn compute_w_poly(
        &self,
        parsed_commitment: &ParsedCommitment<F, MerkleConfig::InnerDigest>,
        statement: &Statement<F>,
        mut folding_randomness: MultilinearPoint<F>,
        initial_combination_randomness: &[F],
        rounds: &[ParsedRound<F>],
        deferred: &[F],
    ) -> F {
        let mut num_variables = self.params.mv_parameters.num_variables;

        let constraints: Vec<_> = parsed_commitment
            .oods_constraints()
            .into_iter()
            .chain(statement.constraints.iter().cloned())
            .collect();

        let mut deferred = deferred.iter().copied();
        let mut value: F = constraints
            .iter()
            .zip(initial_combination_randomness)
            .map(|(constraint, randomness)| {
                if constraint.deferred {
                    deferred.next().unwrap()
                } else {
                    *randomness * constraint.weights.compute(&folding_randomness)
                }
            })
            .sum();

        for (round, round_proof) in rounds.iter().enumerate() {
            num_variables -= self.params.folding_factor.at_round(round);
            folding_randomness = MultilinearPoint(folding_randomness.0[..num_variables].to_vec());

            let stir_challenges = round_proof
                .ood_points
                .iter()
                .chain(&round_proof.stir_challenges_points)
                .map(|&univariate| {
                    MultilinearPoint::expand_from_univariate(univariate, num_variables)
                    // TODO:
                    // Maybe refactor outside
                });

            let sum_of_claims: F = stir_challenges
                .zip(&round_proof.combination_randomness)
                .map(|(pt, rand)| pt.eq_poly_outside(&folding_randomness) * rand)
                .sum();

            value += sum_of_claims;
        }

        value
    }
}
