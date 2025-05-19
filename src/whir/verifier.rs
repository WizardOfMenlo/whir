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

// TODO: Merge these into RoundConfig
pub struct StirChallengParams<F> {
    round_index: usize,
    domain_size: usize,
    num_variables: usize,
    folding_factor: usize,
    num_queries: usize,
    pow_bits: f64,
    domain_gen: F,
    domain_gen_inv: F,
    exp_domain_gen: F,
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
            // Combine OODS and statement constraints to claimed_sum
            let constraints: Vec<_> = parsed_commitment
                .oods_constraints()
                .into_iter()
                .chain(statement.constraints.iter().cloned())
                .collect();
            initial_combination_randomness =
                self.combine_constraints(verifier_state, &mut claimed_sum, &constraints)?;
            constraints_at_round.push(constraints);

            // Initial sumcheck
            folding_randomness = self.verify_sumcheck_rounds(
                verifier_state,
                &mut claimed_sum,
                self.params.folding_factor.at_round(0),
                self.params.starting_folding_pow_bits,
            )?;
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

        // Proof agnostic round parameters
        // TODO: Move to RoundConfig
        let mut params = {
            let domain_gen = self.params.starting_domain.backing_domain.group_gen();
            StirChallengParams {
                round_index: 0,
                domain_size: self.params.starting_domain.size(),
                num_variables: self.params.mv_parameters.num_variables
                    - self.params.folding_factor.at_round(0),
                num_queries: 0,
                folding_factor: 0,
                pow_bits: 0.,
                domain_gen,
                domain_gen_inv: self.params.starting_domain.backing_domain.group_gen_inv(),
                exp_domain_gen: domain_gen.pow([1 << self.params.folding_factor.at_round(0)]),
            }
        };

        let mut prev_commitment = parsed_commitment.clone();
        let mut rounds = vec![];
        for round_index in 0..self.params.n_rounds() {
            // Fetch round parameters from config
            let round_params = &self.params.round_parameters[round_index];
            params.round_index = round_index;
            params.folding_factor = self.params.folding_factor.at_round(round_index);
            params.num_queries = round_params.num_queries;
            params.pow_bits = round_params.pow_bits;

            // Receive commitment to the folded polynomial (likely encoded at higher expansion)
            let new_commitment = ParsedCommitment::<F, MerkleConfig::InnerDigest>::parse(
                verifier_state,
                params.num_variables,
                round_params.ood_samples,
            )?;

            // Verify in-domain challenges on the previous commitment.
            let (stir_constraints, stir_challenges_points) = self.verify_stir_challenges(
                verifier_state,
                &params,
                &prev_commitment,
                &folding_randomness,
            )?;

            // Add out-of-domain and in-domain constraints to claimed_sum
            let constraints: Vec<Constraint<F>> = new_commitment
                .oods_constraints()
                .into_iter()
                .chain(stir_constraints.into_iter())
                .collect();
            let combination_randomness =
                self.combine_constraints(verifier_state, &mut claimed_sum, &constraints)?;
            constraints_at_round.push(constraints);

            let new_folding_randomness = self.verify_sumcheck_rounds(
                verifier_state,
                &mut claimed_sum,
                self.params.folding_factor.at_round(round_index + 1),
                round_params.folding_pow_bits,
            )?;

            rounds.push(ParsedRound {
                folding_randomness,
                ood_points: new_commitment.ood_points.clone(),
                stir_challenges_points,
                combination_randomness,
            });

            folding_randomness = new_folding_randomness;

            prev_commitment = new_commitment;
            params.num_variables -= params.folding_factor;
            params.domain_gen = params.domain_gen.square();
            params.exp_domain_gen = params
                .domain_gen
                .pow([1 << self.params.folding_factor.at_round(round_index + 1)]);
            params.domain_gen_inv = params.domain_gen_inv.square();
            params.domain_size /= 2;
        }

        // Final round parameters.
        params.round_index = self.params.n_rounds();
        params.num_queries = self.params.final_queries;
        params.folding_factor = self.params.folding_factor.at_round(self.params.n_rounds());
        params.pow_bits = self.params.final_pow_bits;

        // In the final round we receive the full polynomial instead of a commitment.
        let mut final_coefficients = vec![F::ZERO; 1 << self.params.final_sumcheck_rounds];
        verifier_state.fill_next_scalars(&mut final_coefficients)?;
        let final_coefficients = CoefficientList::new(final_coefficients);

        // Verify in-domain challenges on the previous commitment.
        let (stir_constraints, _) = self.verify_stir_challenges(
            verifier_state,
            &params,
            &prev_commitment,
            &folding_randomness,
        )?;

        // Verify stir constraints direclty on final polynomial
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
        let deferred: Vec<F> = verifier_state.hint()?;

        let evaluation_of_v_poly = self.compute_w_poly(
            &constraints_at_round,
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

    /// Verify a STIR challenges against a commitment and return the constraints.
    pub fn verify_stir_challenges(
        &self,
        verifier_state: &mut VerifierState,
        params: &StirChallengParams<F>,
        commitment: &ParsedCommitment<F, MerkleConfig::InnerDigest>,
        folding_randomness: &MultilinearPoint<F>,
    ) -> ProofResult<(Vec<Constraint<F>>, Vec<F>)> {
        let stir_challenges_indexes = get_challenge_stir_queries(
            params.domain_size,
            params.folding_factor,
            params.num_queries,
            verifier_state,
        )?;

        let answers =
            self.verify_merkle_proof(verifier_state, &commitment.root, &stir_challenges_indexes)?;

        self.verify_proof_of_work(verifier_state, params.pow_bits)?;

        // Compute STIR Constraints
        let folds = self.params.fold_optimisation.stir_evaluations_verifier(
            self.params,
            params.round_index,
            params.domain_gen_inv,
            &stir_challenges_indexes,
            folding_randomness,
            &answers,
        );
        let stir_constraints = stir_challenges_indexes
            .iter()
            .map(|&index| params.exp_domain_gen.pow([index as u64]))
            .zip(&folds)
            .map(|(point, &value)| Constraint {
                weights: Weights::univariate(point, params.num_variables),
                sum: value,
                deferred: false,
            })
            .collect();

        // Compute stir points
        // TODO: This is redundant with constraints
        let stir_challenges_points = stir_challenges_indexes
            .iter()
            .map(|index| params.exp_domain_gen.pow([*index as u64]))
            .collect();

        Ok((stir_constraints, stir_challenges_points))
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
        constraints: &[Vec<Constraint<F>>],
        mut folding_randomness: MultilinearPoint<F>,
        initial_combination_randomness: &[F],
        rounds: &[ParsedRound<F>],
        deferred: &[F],
    ) -> F {
        let mut num_variables = self.params.mv_parameters.num_variables;

        let mut deferred = deferred.iter().copied();
        let mut value: F = constraints[0]
            .iter()
            .zip(initial_combination_randomness)
            .map(|(constraint, &randomness)| {
                let value = if constraint.deferred {
                    deferred.next().unwrap()
                } else {
                    constraint.weights.compute(&folding_randomness)
                };
                value * randomness
            })
            .sum();

        for (round, round_proof) in rounds.iter().enumerate() {
            num_variables -= self.params.folding_factor.at_round(round);
            folding_randomness = MultilinearPoint(folding_randomness.0[..num_variables].to_vec());

            value += constraints[round + 1]
                .iter()
                .zip(&round_proof.combination_randomness)
                .map(|(constraint, &randomness)| {
                    let value = if constraint.deferred {
                        deferred.next().unwrap()
                    } else {
                        constraint.weights.compute(&folding_randomness)
                    };
                    value * randomness
                })
                .sum::<F>();
        }

        value
    }
}
