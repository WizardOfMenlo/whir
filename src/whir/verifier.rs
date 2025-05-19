use std::iter;

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
    parameters::WhirConfig,
    parsed_proof::{ParsedProof, ParsedRound},
    statement::{Constraint, Statement, Weights},
    utils::HintDeserialize,
};
use crate::{
    poly_utils::{coeffs::CoefficientList, multilinear::MultilinearPoint},
    sumcheck::SumcheckPolynomial,
    utils::expand_randomness,
    whir::utils::{get_challenge_stir_queries, DigestToUnitDeserialize},
};

pub struct Verifier<'a, F, MerkleConfig, PowStrategy>
where
    F: FftField,
    MerkleConfig: Config,
{
    params: &'a WhirConfig<F, MerkleConfig, PowStrategy>,
}

impl<'a, F, MerkleConfig, PowStrategy> Verifier<'a, F, MerkleConfig, PowStrategy>
where
    F: FftField,
    MerkleConfig: Config<Leaf = [F]>,
    PowStrategy: spongefish_pow::PowStrategy,
{
    pub const fn new(params: &'a WhirConfig<F, MerkleConfig, PowStrategy>) -> Self {
        Self { params }
    }

    /// Verify a WHIR proof.
    ///
    /// Returns the constraint evaluation point and the values of the deferred constraints.
    /// It is the callers responsibility to verify the deferred constraints.
    #[allow(clippy::too_many_lines)]
    pub fn verify<VerifierState>(
        &self,
        verifier_state: &mut VerifierState,
        parsed_commitment: &ParsedCommitment<F, MerkleConfig::InnerDigest>,
        statement: &Statement<F>,
    ) -> ProofResult<(MultilinearPoint<F>, Vec<F>)>
    where
        VerifierState: UnitToBytes
            + UnitToField<F>
            + FieldToUnitDeserialize<F>
            + PoWChallenge
            + DigestToUnitDeserialize<MerkleConfig>
            + HintDeserialize,
    {
        // We first do a pass in which we rederive all the FS challenges
        // Then we will check the algebraic part (so to optimise inversions)
        let (mut claimed_sum, parsed) =
            self.parse_proof(verifier_state, parsed_commitment, statement)?;

        // Check the foldings computed from the proof match the evaluations of the polynomial
        let final_folds = self
            .params
            .fold_optimisation
            .stir_final_evaluations_verifier(&parsed, self.params);

        let final_evaluations = parsed
            .final_coefficients
            .evaluate_at_univariate(&parsed.final_randomness_points);
        if !final_folds
            .iter()
            .zip(final_evaluations)
            .all(|(&fold, eval)| fold == eval)
        {
            return Err(ProofError::InvalidProof);
        }

        // Check the final sumchecks
        for (sumcheck_poly, new_randomness) in &parsed.final_sumcheck_rounds {
            if sumcheck_poly.sum_over_boolean_hypercube() != claimed_sum {
                return Err(ProofError::InvalidProof);
            }
            claimed_sum = sumcheck_poly.evaluate_at_point(&(*new_randomness).into());
        }

        // Final v · w Check
        let prev_sumcheck_poly_eval = claimed_sum;

        // Check the final sumcheck evaluation
        let evaluation_of_v_poly =
            self.compute_w_poly(parsed_commitment, statement, &parsed, &parsed.deferred);
        let final_value = parsed
            .final_coefficients
            .evaluate(&parsed.final_sumcheck_randomness);

        if prev_sumcheck_poly_eval != evaluation_of_v_poly * final_value {
            return Err(ProofError::InvalidProof);
        }

        Ok((parsed.final_sumcheck_randomness, parsed.deferred))
    }

    #[allow(clippy::too_many_lines)]
    fn parse_proof<VerifierState>(
        &self,
        verifier_state: &mut VerifierState,
        parsed_commitment: &ParsedCommitment<F, MerkleConfig::InnerDigest>,
        statement: &Statement<F>,
    ) -> ProofResult<(F, ParsedProof<F>)>
    where
        VerifierState: UnitToBytes
            + UnitToField<F>
            + FieldToUnitDeserialize<F>
            + PoWChallenge
            + DigestToUnitDeserialize<MerkleConfig>
            + HintDeserialize,
    {
        // Initial combination and sumcheck rounds
        let evaluations: Vec<_> = statement.constraints.iter().map(|c| c.sum).collect();

        // Optional initial sumcheck round
        let mut claimed_sum = F::ZERO;
        let mut folding_randomness;
        let initial_combination_randomness;
        if self.params.initial_statement {
            // Derive combination randomness and first sumcheck polynomial
            let [combination_randomness_gen] = verifier_state.challenge_scalars()?;
            initial_combination_randomness = expand_randomness(
                combination_randomness_gen,
                parsed_commitment.ood_points.len() + statement.constraints.len(),
            );

            // Compute initial sum
            claimed_sum = parsed_commitment
                .ood_answers
                .iter()
                .copied()
                .chain(evaluations)
                .zip(&initial_combination_randomness)
                .map(|(ans, rand)| ans * rand)
                .sum();

            // Initial sumcheck
            let mut randomness = Vec::new();
            for _ in 0..self.params.folding_factor.at_round(0) {
                let sumcheck_poly_evals: [_; 3] = verifier_state.next_scalars()?;
                let sumcheck_poly = SumcheckPolynomial::new(sumcheck_poly_evals.to_vec(), 1);

                if sumcheck_poly.sum_over_boolean_hypercube() != claimed_sum {
                    return Err(ProofError::InvalidProof);
                }

                let [folding_randomness_single] = verifier_state.challenge_scalars()?;
                randomness.push(folding_randomness_single);

                claimed_sum = sumcheck_poly.evaluate_at_point(&folding_randomness_single.into());

                if self.params.starting_folding_pow_bits > 0. {
                    verifier_state
                        .challenge_pow::<PowStrategy>(self.params.starting_folding_pow_bits)?;
                }
            }

            randomness.reverse();
            folding_randomness = MultilinearPoint(randomness);
        } else {
            assert_eq!(parsed_commitment.ood_points.len(), 0);
            assert!(statement.constraints.is_empty());

            initial_combination_randomness = vec![F::ONE];

            let mut folding_randomness_vec = vec![F::ZERO; self.params.folding_factor.at_round(0)];
            verifier_state.fill_challenge_scalars(&mut folding_randomness_vec)?;
            folding_randomness = MultilinearPoint(folding_randomness_vec);

            // PoW
            if self.params.starting_folding_pow_bits > 0. {
                verifier_state
                    .challenge_pow::<PowStrategy>(self.params.starting_folding_pow_bits)?;
            }
        }

        let mut prev_root = parsed_commitment.root.clone();
        let mut domain_gen = self.params.starting_domain.backing_domain.group_gen();
        let mut exp_domain_gen = domain_gen.pow([1 << self.params.folding_factor.at_round(0)]);
        let mut domain_gen_inv = self.params.starting_domain.backing_domain.group_gen_inv();
        let mut domain_size = self.params.starting_domain.size();
        let mut rounds = vec![];

        for round_index in 0..self.params.n_rounds() {
            let round_params = &self.params.round_parameters[round_index];

            let new_root = verifier_state.read_digest()?;

            let mut ood_points = vec![F::ZERO; round_params.ood_samples];
            let mut ood_answers = vec![F::ZERO; round_params.ood_samples];
            if round_params.ood_samples > 0 {
                verifier_state.fill_challenge_scalars(&mut ood_points)?;
                verifier_state.fill_next_scalars(&mut ood_answers)?;
            }

            let stir_challenges_indexes = get_challenge_stir_queries(
                domain_size,
                self.params.folding_factor.at_round(round_index),
                round_params.num_queries,
                verifier_state,
            )?;

            let stir_challenges_points = stir_challenges_indexes
                .iter()
                .map(|index| exp_domain_gen.pow([*index as u64]))
                .collect();

            let answers: Vec<Vec<F>> = verifier_state.hint()?;
            let merkle_proof: MultiPath<MerkleConfig> = verifier_state.hint()?;

            if !merkle_proof
                .verify(
                    &self.params.leaf_hash_params,
                    &self.params.two_to_one_params,
                    &prev_root,
                    answers.iter().map(|a| a.as_ref()),
                )
                .unwrap()
                || merkle_proof.leaf_indexes != stir_challenges_indexes
            {
                return Err(ProofError::InvalidProof);
            }

            if round_params.pow_bits > 0. {
                verifier_state.challenge_pow::<PowStrategy>(round_params.pow_bits)?;
            }

            let [combination_randomness_gen] = verifier_state.challenge_scalars()?;
            let combination_randomness = expand_randomness(
                combination_randomness_gen,
                stir_challenges_indexes.len() + round_params.ood_samples,
            );

            // Add OODS and STIR evaluations to claimed sum
            let folds = self.params.fold_optimisation.stir_evaluations_verifier(
                self.params,
                round_index,
                domain_gen_inv,
                &stir_challenges_indexes,
                &folding_randomness,
                &answers,
            );
            let values = ood_answers.iter().copied().chain(folds.iter().copied());
            claimed_sum += values
                .zip(&combination_randomness)
                .map(|(val, rand)| val * rand)
                .sum::<F>();

            let mut sumcheck_rounds =
                Vec::with_capacity(self.params.folding_factor.at_round(round_index + 1));

            for _ in 0..self.params.folding_factor.at_round(round_index + 1) {
                // Receive sumcheck polynomial
                let sumcheck_poly_evals: [_; 3] = verifier_state.next_scalars()?;
                let sumcheck_poly = SumcheckPolynomial::new(sumcheck_poly_evals.to_vec(), 1);
                if sumcheck_poly.sum_over_boolean_hypercube() != claimed_sum {
                    return Err(ProofError::InvalidProof);
                }

                let [folding_randomness_single] = verifier_state.challenge_scalars()?;
                sumcheck_rounds.push((sumcheck_poly.clone(), folding_randomness_single));
                claimed_sum = sumcheck_poly.evaluate_at_point(&folding_randomness_single.into());

                if round_params.folding_pow_bits > 0. {
                    verifier_state.challenge_pow::<PowStrategy>(round_params.folding_pow_bits)?;
                }
            }

            let new_folding_randomness =
                MultilinearPoint(sumcheck_rounds.iter().map(|&(_, r)| r).rev().collect());

            let round = ParsedRound {
                folding_randomness,
                ood_points,
                ood_answers,
                stir_challenges_indexes,
                stir_challenges_points,
                stir_challenges_answers: answers,
                combination_randomness,
                sumcheck_rounds,
                domain_gen_inv,
            };
            rounds.push(round);

            folding_randomness = new_folding_randomness;

            prev_root = new_root;
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
        let final_randomness_points = final_randomness_indexes
            .iter()
            .map(|index| exp_domain_gen.pow([*index as u64]))
            .collect();

        let final_randomness_answers: Vec<Vec<F>> = verifier_state.hint()?;
        let final_merkle_proof: MultiPath<MerkleConfig> = verifier_state.hint()?;

        if !final_merkle_proof
            .verify(
                &self.params.leaf_hash_params,
                &self.params.two_to_one_params,
                &prev_root,
                final_randomness_answers.iter().map(|a| a.as_ref()),
            )
            .unwrap()
            || final_merkle_proof.leaf_indexes != final_randomness_indexes
        {
            return Err(ProofError::InvalidProof);
        }

        if self.params.final_pow_bits > 0. {
            verifier_state.challenge_pow::<PowStrategy>(self.params.final_pow_bits)?;
        }

        let mut final_sumcheck_rounds = Vec::with_capacity(self.params.final_sumcheck_rounds);
        for _ in 0..self.params.final_sumcheck_rounds {
            let sumcheck_poly_evals: [_; 3] = verifier_state.next_scalars()?;
            let sumcheck_poly = SumcheckPolynomial::new(sumcheck_poly_evals.to_vec(), 1);
            let [folding_randomness_single] = verifier_state.challenge_scalars()?;
            final_sumcheck_rounds.push((sumcheck_poly, folding_randomness_single));

            if self.params.final_folding_pow_bits > 0. {
                verifier_state.challenge_pow::<PowStrategy>(self.params.final_folding_pow_bits)?;
            }
        }
        let final_sumcheck_randomness = MultilinearPoint(
            final_sumcheck_rounds
                .iter()
                .map(|&(_, r)| r)
                .rev()
                .collect(),
        );
        let deferred: Vec<F> = verifier_state.hint()?;

        Ok((
            claimed_sum,
            ParsedProof {
                initial_combination_randomness,
                rounds,
                final_domain_gen_inv: domain_gen_inv,
                final_folding_randomness: folding_randomness,
                final_randomness_indexes,
                final_randomness_points,
                final_randomness_answers,
                final_sumcheck_rounds,
                final_sumcheck_randomness,
                final_coefficients,
                deferred,
            },
        ))
    }

    fn compute_w_poly(
        &self,
        parsed_commitment: &ParsedCommitment<F, MerkleConfig::InnerDigest>,
        statement: &Statement<F>,
        proof: &ParsedProof<F>,
        deferred: &[F],
    ) -> F {
        let mut num_variables = self.params.mv_parameters.num_variables;

        let mut folding_randomness = MultilinearPoint(
            iter::once(&proof.final_sumcheck_randomness.0)
                .chain(iter::once(&proof.final_folding_randomness.0))
                .chain(proof.rounds.iter().rev().map(|r| &r.folding_randomness.0))
                .flatten()
                .copied()
                .collect(),
        );

        let constraints: Vec<_> = parsed_commitment
            .ood_points
            .iter()
            .zip(&parsed_commitment.ood_answers)
            .map(|(&point, &eval)| {
                let weights = Weights::evaluation(MultilinearPoint::expand_from_univariate(
                    point,
                    num_variables,
                ));
                Constraint {
                    weights,
                    sum: eval,
                    deferred: false,
                }
            })
            .chain(statement.constraints.iter().cloned())
            .collect();

        let mut deferred = deferred.iter().copied();
        let mut value: F = constraints
            .iter()
            .zip(&proof.initial_combination_randomness)
            .map(|(constraint, randomness)| {
                if constraint.deferred {
                    deferred.next().unwrap()
                } else {
                    *randomness * constraint.weights.compute(&folding_randomness)
                }
            })
            .sum();

        for (round, round_proof) in proof.rounds.iter().enumerate() {
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
