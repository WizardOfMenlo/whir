use std::iter;

use ark_crypto_primitives::merkle_tree::Config;
use ark_ff::FftField;
use ark_poly::EvaluationDomain;
use spongefish::{
    codecs::arkworks_algebra::{FieldToUnitDeserialize, UnitToField},
    ProofError, ProofResult, UnitToBytes,
};
use spongefish_pow::{self, PoWChallenge};

use super::{
    challenges::{ChallengeField, ChallengeIndices},
    committer::Commitment,
    parameters::WhirConfig,
    parsed_proof::{ParsedProof, ParsedRound},
    statement::{StatementVerifier, VerifierWeights},
    WhirProof,
};
use crate::{
    poly_utils::{coeffs::CoefficientList, multilinear::MultilinearPoint},
    sumcheck::SumcheckPolynomial,
    whir::utils::DigestToUnitDeserialize,
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

    #[allow(clippy::too_many_lines)]
    pub(crate) fn parse_proof<VerifierState>(
        &self,
        verifier_state: &mut VerifierState,
        commitments: &[&Commitment<F, MerkleConfig::InnerDigest>],
        whir_proof: &WhirProof<MerkleConfig, F>,
        statement_num_constraints: usize,
    ) -> ProofResult<ParsedProof<F>>
    where
        VerifierState: UnitToBytes
            + UnitToField<F>
            + FieldToUnitDeserialize<F>
            + PoWChallenge
            + DigestToUnitDeserialize<MerkleConfig>,
    {
        assert_eq!(
            self.params.mv_parameters.num_variables,
            self.params
                .folding_factor
                .total_number(self.params.n_rounds())
                + self.params.final_sumcheck_rounds
        );
        for commitment in commitments {
            // TODO: Num polynomials as part of config.
            assert_eq!(
                commitment.num_variables(),
                self.params.mv_parameters.num_variables
            );
            assert_eq!(
                commitment.ood_points.len(),
                self.params.committment_ood_samples
            );
            assert_eq!(
                commitment.ood_answers.len(),
                self.params.committment_ood_samples
            );
        }
        let num_polynomials = commitments.iter().map(|c| c.num_polynomials()).sum();
        let num_constraints =
            commitments.iter().map(|c| c.num_constraints()).sum() + statement_num_constraints;
        if !self.params.initial_statement {
            assert_eq!(num_constraints, 0);
        }

        // Parse polynomial and constraint combination randomness
        let initial_polynomial_randomness = verifier_state.challenge_geometric_vec(num_polynomials);
        let initial_constraint_randomness = verifier_state.challenge_geometric_vec(num_constraints);

        // Optional initial sumcheck round
        let mut sumcheck_rounds = Vec::new();
        let mut folding_randomness;
        if self.params.initial_statement {
            for _ in 0..self.params.folding_factor.at_round(0) {
                let sumcheck_poly_evals: [F; 3] = verifier_state.next_scalars()?;
                let sumcheck_poly = SumcheckPolynomial::new(sumcheck_poly_evals.to_vec(), 1);
                let [folding_randomness_single] = verifier_state.challenge_scalars()?;
                sumcheck_rounds.push((sumcheck_poly, folding_randomness_single));

                if self.params.starting_folding_pow_bits > 0. {
                    verifier_state
                        .challenge_pow::<PowStrategy>(self.params.starting_folding_pow_bits)?;
                }
            }
            folding_randomness =
                MultilinearPoint(sumcheck_rounds.iter().map(|&(_, r)| r).rev().collect());
        } else {
            folding_randomness = MultilinearPoint(
                verifier_state.challenge_vec(self.params.folding_factor.at_round(0))?,
            );

            // PoW
            if self.params.starting_folding_pow_bits > 0. {
                verifier_state
                    .challenge_pow::<PowStrategy>(self.params.starting_folding_pow_bits)?;
            }
        }

        // Not needed in the first round
        let mut prev_roots = commitments.iter().map(|c| c.root).collect::<Vec<_>>();
        let mut domain_gen = self.params.starting_domain.backing_domain.group_gen();
        let mut exp_domain_gen = domain_gen.pow([1 << self.params.folding_factor.at_round(0)]);
        let mut domain_gen_inv = self.params.starting_domain.backing_domain.group_gen_inv();
        let mut domain_size = self.params.starting_domain.size();
        let mut rounds = vec![];

        for r in 0..self.params.n_rounds() {
            let round_params = &self.params.round_parameters[r];

            let new_root = verifier_state.read_digest()?;

            let mut ood_points = verifier_state.challenge_vec(round_params.ood_samples);
            // TODO: oods answers for each polynomial.
            let mut ood_answers = vec![F::ZERO; round_params.ood_samples];

            let stir_challenges_indexes = verifier_state.challenge_indices(
                domain_size >> self.params.folding_factor.at_round(r),
                round_params.num_queries,
            )?;

            let stir_challenges_points = stir_challenges_indexes
                .iter()
                .map(|index| exp_domain_gen.pow([*index as u64]))
                .collect();

            let merkle_proofs = whir_proof.merkle_proofs[r].clone();
            let mut answers = Vec::new();
            for ((proof, leaves), root) in merkle_proofs.iter().zip(prev_roots.iter()) {
                assert_eq!(proof.leaf_indexes, stir_challenges_indexes);
                // TODO: This is verifying, not parsing, move to appropriate place.
                proof
                    .verify(
                        &self.params.leaf_hash_params,
                        &self.params.two_to_one_params,
                        root,
                        leaves.iter().map(|a| a.as_ref()),
                    )
                    .map_err(|_| ProofError::InvalidProof)?;
                // TODO: Would be nicer if we separate the polynomials.
                answers.push(leaves.clone());
            }

            if round_params.pow_bits > 0. {
                verifier_state.challenge_pow::<PowStrategy>(round_params.pow_bits)?;
            }

            // Randomness for combining polynomials and constraints
            let num_polynomials = 1;
            let num_constraints = stir_challenges_indexes.len() + round_params.ood_samples;
            let polynomial_randomness = verifier_state.challenge_geometric_vec(num_polynomials)?;
            let constraint_randomness = verifier_state.challenge_geometric_vec(num_constraints)?;

            let mut sumcheck_rounds =
                Vec::with_capacity(self.params.folding_factor.at_round(r + 1));
            for _ in 0..self.params.folding_factor.at_round(r + 1) {
                let sumcheck_poly_evals: [_; 3] = verifier_state.next_scalars()?;
                let sumcheck_poly = SumcheckPolynomial::new(sumcheck_poly_evals.to_vec(), 1);
                let [folding_randomness_single] = verifier_state.challenge_scalars()?;
                sumcheck_rounds.push((sumcheck_poly, folding_randomness_single));

                if round_params.folding_pow_bits > 0. {
                    verifier_state.challenge_pow::<PowStrategy>(round_params.folding_pow_bits)?;
                }
            }

            let new_folding_randomness =
                MultilinearPoint(sumcheck_rounds.iter().map(|&(_, r)| r).rev().collect());

            rounds.push(ParsedRound {
                folding_randomness,
                ood_points,
                ood_answers,
                stir_challenges_indexes,
                stir_challenges_points,
                stir_challenges_answers: answers,
                constraint_randomness,
                polynomial_randomness,
                sumcheck_rounds,
                domain_gen_inv,
            });

            folding_randomness = new_folding_randomness;

            prev_roots = vec![new_root];
            domain_gen = domain_gen * domain_gen;
            exp_domain_gen = domain_gen.pow([1 << self.params.folding_factor.at_round(r + 1)]);
            domain_gen_inv = domain_gen_inv * domain_gen_inv;
            domain_size /= 2;
        }

        let final_coefficients = CoefficientList::new(
            verifier_state.challenge_vec(1 << self.params.final_sumcheck_rounds)?,
        );

        // Final queries verify
        let final_randomness_indexes = verifier_state.challenge_indices(
            domain_size >> self.params.folding_factor.at_round(self.params.n_rounds()),
            self.params.final_queries,
        )?;
        let final_randomness_points = final_randomness_indexes
            .iter()
            .map(|index| exp_domain_gen.pow([*index as u64]))
            .collect();

        let (final_merkle_proof, final_randomness_answers) =
            whir_proof.round_data(self.params.n_rounds(), commitments[0]);

        if self.params.n_rounds() == 0 {
            // There's only a single round, so we need to validate against the
            // root batched node
            if !whir_proof.validate_first_round(
                commitments,
                &final_randomness_indexes,
                &self.params.leaf_hash_params,
                &self.params.two_to_one_params,
            ) {
                return Err(ProofError::InvalidProof);
            }
        } else {
            if !final_merkle_proof
                .verify(
                    &self.params.leaf_hash_params,
                    &self.params.two_to_one_params,
                    &prev_roots,
                    final_randomness_answers.iter().map(|a| a.as_ref()),
                )
                .unwrap()
                || final_merkle_proof.leaf_indexes != final_randomness_indexes
            {
                return Err(ProofError::InvalidProof);
            }
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

        Ok(ParsedProof {
            initial_polynomial_randomness,
            initial_constraint_randomness,
            initial_sumcheck_rounds: sumcheck_rounds,
            rounds,
            final_domain_gen_inv: domain_gen_inv,
            final_folding_randomness: folding_randomness,
            final_randomness_indexes,
            final_randomness_points,
            final_randomness_answers: final_randomness_answers.clone(),
            final_sumcheck_rounds,
            final_sumcheck_randomness,
            final_coefficients,
            statement_values_at_random_point: whir_proof.statement_values_at_random_point.clone(),
        })
    }

    fn compute_w_poly(
        &self,
        ood_points: &[F],
        ood_response: &[F],
        statement: &StatementVerifier<F>,
        proof: &ParsedProof<F>,
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

        let mut new_constraints: Vec<_> = ood_points
            .iter()
            .zip(ood_response.iter())
            .map(|(&point, &eval)| {
                let weights = VerifierWeights::evaluation(
                    MultilinearPoint::expand_from_univariate(point, num_variables),
                );
                (weights, eval)
            })
            .collect();

        let mut proof_values_iter = proof.statement_values_at_random_point.iter();
        for (weights, expected_result) in &statement.constraints {
            match weights {
                VerifierWeights::Evaluation { point } => {
                    new_constraints
                        .push((VerifierWeights::evaluation(point.clone()), *expected_result));
                }
                VerifierWeights::Linear { .. } => {
                    let term = proof_values_iter
                        .next()
                        .expect("Not enough proof statement values for linear constraints");
                    new_constraints.push((
                        VerifierWeights::linear(num_variables, Some(*term)),
                        *expected_result,
                    ));
                }
            }
        }

        let mut value: F = new_constraints
            .iter()
            .zip(&proof.initial_constraint_randomness)
            .map(|((weight, _), randomness)| *randomness * weight.compute(&folding_randomness))
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
                .zip(&round_proof.constraint_randomness)
                .map(|(pt, rand)| pt.eq_poly_outside(&folding_randomness) * rand)
                .sum();

            value += sum_of_claims;
        }

        value
    }

    #[allow(clippy::too_many_lines)]
    pub(crate) fn verify_parsed(
        &self,
        statements: &[&StatementVerifier<F>],
        commitments: &[&Commitment<F, MerkleConfig::InnerDigest>],
        parsed_proof: &ParsedProof<F>,
    ) -> ProofResult<()> {
        let evaluations: Vec<_> = statement.constraints.iter().map(|c| c.1).collect();

        let computed_folds = self
            .params
            .fold_optimisation
            .stir_evaluations_verifier(&parsed_proof, &self.params);

        let mut prev_sumcheck = None;

        let (ood_points, ood_answers) = parsed_commitment.ood_points;

        // Initial sumcheck verification
        if let Some((poly, randomness)) = parsed_proof.initial_sumcheck_rounds.first().cloned() {
            if poly.sum_over_boolean_hypercube()
                != ood_answers
                    .iter()
                    .copied()
                    .chain(evaluations)
                    .zip(&parsed_proof.initial_constraint_randomness)
                    .map(|(ans, rand)| ans * rand)
                    .sum()
            {
                println!("Initial sumcheck failed");
                return Err(ProofError::InvalidProof);
            }

            let mut current = (poly, randomness);

            // Check the rest of the rounds
            for (next_poly, next_rand) in &parsed_proof.initial_sumcheck_rounds[1..] {
                if next_poly.sum_over_boolean_hypercube()
                    != current.0.evaluate_at_point(&current.1.into())
                {
                    return Err(ProofError::InvalidProof);
                }
                current = (next_poly.clone(), *next_rand);
            }

            prev_sumcheck = Some(current);
        }

        // Sumcheck rounds
        for (round, folds) in parsed_proof.rounds.iter().zip(&computed_folds) {
            let (sumcheck_poly, new_randomness) = &round.sumcheck_rounds[0];

            let values = round
                .ood_answers
                .iter()
                .copied()
                .chain(folds.iter().copied());

            let prev_eval = prev_sumcheck
                .as_ref()
                .map_or(F::ZERO, |(p, r)| p.evaluate_at_point(&(*r).into()));

            let claimed_sum = prev_eval
                + values
                    .zip(&round.constraint_randomness)
                    .map(|(val, rand)| val * rand)
                    .sum::<F>();

            if sumcheck_poly.sum_over_boolean_hypercube() != claimed_sum {
                return Err(ProofError::InvalidProof);
            }

            prev_sumcheck = Some((sumcheck_poly.clone(), *new_randomness));

            // Check the rest of the round
            for (sumcheck_poly, new_randomness) in &round.sumcheck_rounds[1..] {
                let (prev_poly, randomness) = prev_sumcheck.unwrap();
                if sumcheck_poly.sum_over_boolean_hypercube()
                    != prev_poly.evaluate_at_point(&randomness.into())
                {
                    return Err(ProofError::InvalidProof);
                }
                prev_sumcheck = Some((sumcheck_poly.clone(), *new_randomness));
            }
        }

        // Check the foldings computed from the proof match the evaluations of
        // the polynomial
        let final_folds = &computed_folds.last().expect("final folds missing");
        let final_evaluations = parsed_proof
            .final_coefficients
            .evaluate_at_univariate(&parsed_proof.final_randomness_points);
        if !final_folds
            .iter()
            .zip(final_evaluations)
            .all(|(&fold, eval)| fold == eval)
        {
            return Err(ProofError::InvalidProof);
        }

        // Check the final sumchecks
        if self.params.final_sumcheck_rounds > 0 {
            let claimed_sum = prev_sumcheck
                .as_ref()
                .map_or(F::ZERO, |(p, r)| p.evaluate_at_point(&(*r).into()));

            let (sumcheck_poly, new_randomness) = &parsed_proof.final_sumcheck_rounds[0];

            if sumcheck_poly.sum_over_boolean_hypercube() != claimed_sum {
                return Err(ProofError::InvalidProof);
            }

            prev_sumcheck = Some((sumcheck_poly.clone(), *new_randomness));

            // Check the rest of the round
            for (sumcheck_poly, new_randomness) in &parsed_proof.final_sumcheck_rounds[1..] {
                let (prev_poly, randomness) = prev_sumcheck.unwrap();
                if sumcheck_poly.sum_over_boolean_hypercube()
                    != prev_poly.evaluate_at_point(&randomness.into())
                {
                    return Err(ProofError::InvalidProof);
                }
                prev_sumcheck = Some((sumcheck_poly.clone(), *new_randomness));
            }
        }

        // Final v · w Check
        let prev_sumcheck_poly_eval =
            prev_sumcheck.map_or(F::ZERO, |(poly, rand)| poly.evaluate_at_point(&rand.into()));

        // Check the final sumcheck evaluation
        let evaluation_of_v_poly =
            self.compute_w_poly(ood_points, &ood_answers, statement, parsed_proof);
        let final_value = parsed_proof
            .final_coefficients
            .evaluate(&parsed_proof.final_sumcheck_randomness);

        if prev_sumcheck_poly_eval != evaluation_of_v_poly * final_value {
            return Err(ProofError::InvalidProof);
        }

        Ok(())
    }

    pub fn verify<VerifierState>(
        &self,
        verifier_state: &mut VerifierState,
        commitments: &[&Commitment<F, MerkleConfig::InnerDigest>],
        statements: &[&StatementVerifier<F>],
        whir_proof: &WhirProof<MerkleConfig, F>,
    ) -> ProofResult<()>
    where
        VerifierState: UnitToBytes
            + UnitToField<F>
            + FieldToUnitDeserialize<F>
            + PoWChallenge
            + DigestToUnitDeserialize<MerkleConfig>,
    {
        let parsed_proof = self.parse_proof(
            verifier_state,
            commitments,
            whir_proof,
            statements.iter().map(|s| s.num_constraints()).sum(),
        )?;

        self.verify_parsed(statements, commitments, &parsed_proof)
    }
}
