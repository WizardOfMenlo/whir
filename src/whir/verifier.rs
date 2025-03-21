use std::iter;

use ark_crypto_primitives::merkle_tree::Config;
use ark_ff::FftField;
use ark_poly::EvaluationDomain;
use nimue::{
    plugins::ark::{FieldChallenges, FieldReader},
    ByteChallenges, ProofError, ProofResult,
};
use nimue_pow::{self, PoWChallenge};

use super::{
    parameters::WhirConfig,
    parsed_proof::{ParsedProof, ParsedRound},
    statement::{StatementVerifier, VerifierWeights},
    WhirProof,
};
use crate::whir::fs_utils::{get_challenge_stir_queries, DigestReader};
use crate::{
    parameters::FoldType,
    poly_utils::{coeffs::CoefficientList, fold::compute_fold, multilinear::MultilinearPoint},
    sumcheck::SumcheckPolynomial,
    utils::expand_randomness,
};

pub struct Verifier<F, MerkleConfig, PowStrategy>
where
    F: FftField,
    MerkleConfig: Config,
{
    params: WhirConfig<F, MerkleConfig, PowStrategy>,
    two_inv: F,
}

#[derive(Clone)]
struct ParsedCommitment<F, D> {
    root: D,
    ood_points: Vec<F>,
    ood_answers: Vec<F>,
}

impl<F, MerkleConfig, PowStrategy> Verifier<F, MerkleConfig, PowStrategy>
where
    F: FftField,
    MerkleConfig: Config<Leaf = [F]>,
    PowStrategy: nimue_pow::PowStrategy,
{
    pub fn new(params: WhirConfig<F, MerkleConfig, PowStrategy>) -> Self {
        Self {
            params,
            two_inv: F::from(2).inverse().unwrap(), // The only inverse in the entire code :)
        }
    }

    fn parse_commitment<Arthur>(
        &self,
        arthur: &mut Arthur,
    ) -> ProofResult<ParsedCommitment<F, MerkleConfig::InnerDigest>>
    where
        Arthur: FieldReader<F> + FieldChallenges<F> + DigestReader<MerkleConfig>,
    {
        let root = arthur.read_digest()?;

        let mut ood_points = vec![F::ZERO; self.params.committment_ood_samples];
        let mut ood_answers = vec![F::ZERO; self.params.committment_ood_samples];
        if self.params.committment_ood_samples > 0 {
            arthur.fill_challenge_scalars(&mut ood_points)?;
            arthur.fill_next_scalars(&mut ood_answers)?;
        }

        Ok(ParsedCommitment {
            root,
            ood_points,
            ood_answers,
        })
    }

    #[allow(clippy::too_many_lines)]
    fn parse_proof<Arthur>(
        &self,
        arthur: &mut Arthur,
        parsed_commitment: &ParsedCommitment<F, MerkleConfig::InnerDigest>,
        statement_points_len: usize, // Will be needed later
        whir_proof: &WhirProof<MerkleConfig, F>,
    ) -> ProofResult<ParsedProof<F>>
    where
        Arthur: FieldReader<F> + PoWChallenge + ByteChallenges + DigestReader<MerkleConfig>,
    {
        let mut sumcheck_rounds = Vec::new();
        let mut folding_randomness: MultilinearPoint<F>;
        let initial_combination_randomness;
        if self.params.initial_statement {
            // Derive combination randomness and first sumcheck polynomial
            let [combination_randomness_gen]: [F; 1] = arthur.challenge_scalars()?;
            initial_combination_randomness = expand_randomness(
                combination_randomness_gen,
                parsed_commitment.ood_points.len() + statement_points_len,
            );

            // Initial sumcheck
            sumcheck_rounds.reserve_exact(self.params.folding_factor.at_round(0));
            for _ in 0..self.params.folding_factor.at_round(0) {
                let sumcheck_poly_evals: [F; 3] = arthur.next_scalars()?;
                let sumcheck_poly = SumcheckPolynomial::new(sumcheck_poly_evals.to_vec(), 1);
                let [folding_randomness_single] = arthur.challenge_scalars()?;
                sumcheck_rounds.push((sumcheck_poly, folding_randomness_single));

                if self.params.starting_folding_pow_bits > 0. {
                    arthur.challenge_pow::<PowStrategy>(self.params.starting_folding_pow_bits)?;
                }
            }

            folding_randomness =
                MultilinearPoint(sumcheck_rounds.iter().map(|&(_, r)| r).rev().collect());
        } else {
            assert_eq!(parsed_commitment.ood_points.len(), 0);
            assert_eq!(statement_points_len, 0);

            initial_combination_randomness = vec![F::ONE];

            let mut folding_randomness_vec = vec![F::ZERO; self.params.folding_factor.at_round(0)];
            arthur.fill_challenge_scalars(&mut folding_randomness_vec)?;
            folding_randomness = MultilinearPoint(folding_randomness_vec);

            // PoW
            if self.params.starting_folding_pow_bits > 0. {
                arthur.challenge_pow::<PowStrategy>(self.params.starting_folding_pow_bits)?;
            }
        }

        let mut prev_root = parsed_commitment.root.clone();
        let mut domain_gen = self.params.starting_domain.backing_domain.group_gen();
        let mut exp_domain_gen = domain_gen.pow([1 << self.params.folding_factor.at_round(0)]);
        let mut domain_gen_inv = self.params.starting_domain.backing_domain.group_gen_inv();
        let mut domain_size = self.params.starting_domain.size();
        let mut rounds = vec![];

        for r in 0..self.params.n_rounds() {
            let (merkle_proof, answers) = &whir_proof.merkle_paths[r];
            let round_params = &self.params.round_parameters[r];

            let new_root = arthur.read_digest()?;

            let mut ood_points = vec![F::ZERO; round_params.ood_samples];
            let mut ood_answers = vec![F::ZERO; round_params.ood_samples];
            if round_params.ood_samples > 0 {
                arthur.fill_challenge_scalars(&mut ood_points)?;
                arthur.fill_next_scalars(&mut ood_answers)?;
            }

            let stir_challenges_indexes = get_challenge_stir_queries(
                domain_size,
                self.params.folding_factor.at_round(r),
                round_params.num_queries,
                arthur,
            )?;

            let stir_challenges_points = stir_challenges_indexes
                .iter()
                .map(|index| exp_domain_gen.pow([*index as u64]))
                .collect();

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
                arthur.challenge_pow::<PowStrategy>(round_params.pow_bits)?;
            }

            let [combination_randomness_gen] = arthur.challenge_scalars()?;
            let combination_randomness = expand_randomness(
                combination_randomness_gen,
                stir_challenges_indexes.len() + round_params.ood_samples,
            );

            let mut sumcheck_rounds =
                Vec::with_capacity(self.params.folding_factor.at_round(r + 1));
            for _ in 0..self.params.folding_factor.at_round(r + 1) {
                let sumcheck_poly_evals: [F; 3] = arthur.next_scalars()?;
                let sumcheck_poly = SumcheckPolynomial::new(sumcheck_poly_evals.to_vec(), 1);
                let [folding_randomness_single] = arthur.challenge_scalars()?;
                sumcheck_rounds.push((sumcheck_poly, folding_randomness_single));

                if round_params.folding_pow_bits > 0. {
                    arthur.challenge_pow::<PowStrategy>(round_params.folding_pow_bits)?;
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
                stir_challenges_answers: answers.clone(),
                combination_randomness,
                sumcheck_rounds,
                domain_gen_inv,
            });

            folding_randomness = new_folding_randomness;

            prev_root = new_root.clone();
            domain_gen = domain_gen * domain_gen;
            exp_domain_gen = domain_gen.pow([1 << self.params.folding_factor.at_round(r + 1)]);
            domain_gen_inv = domain_gen_inv * domain_gen_inv;
            domain_size /= 2;
        }

        let mut final_coefficients = vec![F::ZERO; 1 << self.params.final_sumcheck_rounds];
        arthur.fill_next_scalars(&mut final_coefficients)?;
        let final_coefficients = CoefficientList::new(final_coefficients);

        // Final queries verify
        let final_randomness_indexes = get_challenge_stir_queries(
            domain_size,
            self.params.folding_factor.at_round(self.params.n_rounds()),
            self.params.final_queries,
            arthur,
        )?;
        let final_randomness_points = final_randomness_indexes
            .iter()
            .map(|index| exp_domain_gen.pow([*index as u64]))
            .collect();

        let (final_merkle_proof, final_randomness_answers) =
            &whir_proof.merkle_paths[whir_proof.merkle_paths.len() - 1];
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
            arthur.challenge_pow::<PowStrategy>(self.params.final_pow_bits)?;
        }

        let mut final_sumcheck_rounds = Vec::with_capacity(self.params.final_sumcheck_rounds);
        for _ in 0..self.params.final_sumcheck_rounds {
            let sumcheck_poly_evals: [F; 3] = arthur.next_scalars()?;
            let sumcheck_poly = SumcheckPolynomial::new(sumcheck_poly_evals.to_vec(), 1);
            let [folding_randomness_single] = arthur.challenge_scalars()?;
            final_sumcheck_rounds.push((sumcheck_poly, folding_randomness_single));

            if self.params.final_folding_pow_bits > 0. {
                arthur.challenge_pow::<PowStrategy>(self.params.final_folding_pow_bits)?;
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
            initial_combination_randomness,
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
        parsed_commitment: &ParsedCommitment<F, MerkleConfig::InnerDigest>,
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

        let mut new_constraints: Vec<(VerifierWeights<F>, F)> = Vec::new();
        for (point, evaluation) in parsed_commitment
            .ood_points
            .clone()
            .into_iter()
            .zip(parsed_commitment.ood_answers.clone())
        {
            let weights = VerifierWeights::evaluation(MultilinearPoint::expand_from_univariate(
                point,
                num_variables,
            ));
            new_constraints.push((weights, evaluation));
        }
        let mut proof_values_iter = proof.statement_values_at_random_point.iter();
        for constraint in &statement.constraints {
            match &constraint.0 {
                VerifierWeights::Evaluation { point } => {
                    new_constraints
                        .push((VerifierWeights::evaluation(point.clone()), constraint.1));
                }
                VerifierWeights::Linear { .. } => {
                    let term = proof_values_iter
                        .next()
                        .expect("Not enough proof statement values for linear constraints");
                    new_constraints.push((
                        VerifierWeights::linear(num_variables, Some(*term)),
                        constraint.1,
                    ));
                }
            }
        }
        let mut value: F = new_constraints
            .iter()
            .zip(&proof.initial_combination_randomness)
            .map(|((weight, _), randomness)| *randomness * weight.compute(&folding_randomness))
            .sum();

        for (round, round_proof) in proof.rounds.iter().enumerate() {
            num_variables -= self.params.folding_factor.at_round(round);
            folding_randomness = MultilinearPoint(folding_randomness.0[..num_variables].to_vec());

            let ood_points = &round_proof.ood_points;
            let stir_challenges_points = &round_proof.stir_challenges_points;
            let stir_challenges = ood_points
                .iter()
                .chain(stir_challenges_points)
                .copied()
                .map(|univariate| {
                    MultilinearPoint::expand_from_univariate(univariate, num_variables)
                    // TODO:
                    // Maybe refactor outside
                });

            let sum_of_claims: F = stir_challenges
                .into_iter()
                .map(|point| point.eq_poly_outside(&folding_randomness))
                .zip(&round_proof.combination_randomness)
                .map(|(point, rand)| point * rand)
                .sum();

            value += sum_of_claims;
        }

        value
    }

    fn compute_folds(&self, parsed: &ParsedProof<F>) -> Vec<Vec<F>> {
        match self.params.fold_optimisation {
            FoldType::Naive => self.compute_folds_full(parsed),
            FoldType::ProverHelps => parsed.compute_folds_helped(),
        }
    }

    fn compute_folds_full(&self, parsed: &ParsedProof<F>) -> Vec<Vec<F>> {
        let mut domain_size = self.params.starting_domain.backing_domain.size();

        let mut result = Vec::new();

        for (round_index, round) in parsed.rounds.iter().enumerate() {
            let coset_domain_size = 1 << self.params.folding_factor.at_round(round_index);
            // This is such that coset_generator^coset_domain_size = F::ONE
            //let _coset_generator = domain_gen.pow(&[(domain_size / coset_domain_size) as u64]);
            let coset_generator_inv = round
                .domain_gen_inv
                .pow([(domain_size / coset_domain_size) as u64]);

            let evaluations: Vec<_> = round
                .stir_challenges_indexes
                .iter()
                .zip(&round.stir_challenges_answers)
                .map(|(index, answers)| {
                    // The coset is w^index * <w_coset_generator>
                    //let _coset_offset = domain_gen.pow(&[*index as u64]);
                    let coset_offset_inv = round.domain_gen_inv.pow([*index as u64]);

                    compute_fold(
                        answers,
                        &round.folding_randomness.0,
                        coset_offset_inv,
                        coset_generator_inv,
                        self.two_inv,
                        self.params.folding_factor.at_round(round_index),
                    )
                })
                .collect();
            result.push(evaluations);
            domain_size /= 2;
        }

        let coset_domain_size = 1 << self.params.folding_factor.at_round(parsed.rounds.len());
        let domain_gen_inv = parsed.final_domain_gen_inv;

        // Final round
        let coset_generator_inv = domain_gen_inv.pow([(domain_size / coset_domain_size) as u64]);
        let evaluations: Vec<_> = parsed
            .final_randomness_indexes
            .iter()
            .zip(&parsed.final_randomness_answers)
            .map(|(index, answers)| {
                // The coset is w^index * <w_coset_generator>
                //let _coset_offset = domain_gen.pow(&[*index as u64]);
                let coset_offset_inv = domain_gen_inv.pow([*index as u64]);

                compute_fold(
                    answers,
                    &parsed.final_folding_randomness.0,
                    coset_offset_inv,
                    coset_generator_inv,
                    self.two_inv,
                    self.params.folding_factor.at_round(parsed.rounds.len()),
                )
            })
            .collect();
        result.push(evaluations);

        result
    }

    #[allow(clippy::too_many_lines)]
    pub fn verify<Arthur>(
        &self,
        arthur: &mut Arthur,
        statement: &StatementVerifier<F>,
        whir_proof: &WhirProof<MerkleConfig, F>,
    ) -> ProofResult<()>
    where
        Arthur: FieldReader<F> + ByteChallenges + PoWChallenge + DigestReader<MerkleConfig>,
    {
        // We first do a pass in which we rederive all the FS challenges
        // Then we will check the algebraic part (so to optimise inversions)
        let parsed_commitment = self.parse_commitment(arthur)?;
        let evaluations: Vec<_> = statement.constraints.iter().map(|c| c.1).collect();
        let parsed = self.parse_proof(
            arthur,
            &parsed_commitment,
            statement.constraints.len(),
            whir_proof,
        )?;

        let computed_folds = self.compute_folds(&parsed);

        let mut prev_sumcheck = None;

        // Initial sumcheck verification
        if let Some(round) = parsed.initial_sumcheck_rounds.first() {
            // Check the first polynomial
            let (mut prev_poly, mut randomness) = round.clone();
            if prev_poly.sum_over_boolean_hypercube()
                != parsed_commitment
                    .ood_answers
                    .iter()
                    .copied()
                    .chain(evaluations)
                    .zip(&parsed.initial_combination_randomness)
                    .map(|(ans, rand)| ans * rand)
                    .sum()
            {
                println!("Initial sumcheck failed");
                return Err(ProofError::InvalidProof);
            }

            // Check the rest of the rounds
            for (sumcheck_poly, new_randomness) in &parsed.initial_sumcheck_rounds[1..] {
                if sumcheck_poly.sum_over_boolean_hypercube()
                    != prev_poly.evaluate_at_point(&randomness.into())
                {
                    return Err(ProofError::InvalidProof);
                }
                prev_poly = sumcheck_poly.clone();
                randomness = *new_randomness;
            }

            prev_sumcheck = Some((prev_poly, randomness));
        }

        // Sumcheck rounds
        for (round, folds) in parsed.rounds.iter().zip(&computed_folds) {
            let (sumcheck_poly, new_randomness) = &round.sumcheck_rounds[0];

            let values = round.ood_answers.iter().copied().chain(folds.clone());

            let prev_eval = prev_sumcheck
                .as_ref()
                .map_or(F::ZERO, |(p, r)| p.evaluate_at_point(&(*r).into()));

            let claimed_sum = prev_eval
                + values
                    .zip(&round.combination_randomness)
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

        // Check the foldings computed from the proof match the evaluations of the polynomial
        let final_folds = &computed_folds[computed_folds.len() - 1];
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
        if self.params.final_sumcheck_rounds > 0 {
            let claimed_sum = prev_sumcheck
                .as_ref()
                .map_or(F::ZERO, |(p, r)| p.evaluate_at_point(&(*r).into()));

            let (sumcheck_poly, new_randomness) = &parsed.final_sumcheck_rounds[0];

            if sumcheck_poly.sum_over_boolean_hypercube() != claimed_sum {
                return Err(ProofError::InvalidProof);
            }

            prev_sumcheck = Some((sumcheck_poly.clone(), *new_randomness));

            // Check the rest of the round
            for (sumcheck_poly, new_randomness) in &parsed.final_sumcheck_rounds[1..] {
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
        let evaluation_of_v_poly = self.compute_w_poly(&parsed_commitment, statement, &parsed);
        let final_value = parsed
            .final_coefficients
            .evaluate(&parsed.final_sumcheck_randomness);

        if prev_sumcheck_poly_eval != evaluation_of_v_poly * final_value {
            return Err(ProofError::InvalidProof);
        }

        Ok(())
    }
}
