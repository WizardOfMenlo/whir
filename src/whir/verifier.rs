use std::iter;

use ark_crypto_primitives::merkle_tree::{Config, LeafParam, TwoToOneParam};
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
    statement::{StatementVerifier, VerifierWeights},
    RoundZeroProofValidator, WhirCommitmentData, WhirProof, WhirProofRoundData,
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

impl<F, MerkleConfig> RoundZeroProofValidator<ParsedCommitment<F, MerkleConfig::InnerDigest>>
    for WhirProof<MerkleConfig, F>
where
    MerkleConfig: Config<Leaf = [F]>,
    F: Sized + Clone + ark_serialize::CanonicalSerialize + ark_serialize::CanonicalDeserialize,
{
    type MerkleConfig = MerkleConfig;
    fn validate_first_round(
        &self,
        commitment: &ParsedCommitment<F, MerkleConfig::InnerDigest>,
        stir_challenges_indexes: &[usize],
        leaf_hash: &LeafParam<MerkleConfig>,
        inner_node_hash: &TwoToOneParam<MerkleConfig>,
    ) -> bool {
        let (merkle_proof, answers) = &self.merkle_paths[0];

        merkle_proof
            .verify(
                &leaf_hash,
                &inner_node_hash,
                &commitment.root,
                answers.iter().map(|a| a.as_ref()),
            )
            .unwrap()
            && merkle_proof.leaf_indexes == *stir_challenges_indexes
    }
}

impl<F, M: Config> WhirCommitmentData<F, M> for ParsedCommitment<F, M::InnerDigest> {
    fn committed_root(&self) -> &<M as Config>::InnerDigest {
        &self.root
    }

    fn ood_data(&self) -> (&[F], &[F]) {
        (&self.ood_points, &self.ood_answers)
    }

    fn batching_randomness(&self) -> Option<F> {
        None
    }
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
    pub(crate) fn parse_proof<VerifierState, CommitmentType, WhirProof>(
        &self,
        verifier_state: &mut VerifierState,
        parsed_commitment: &CommitmentType,
        whir_proof: &WhirProof,
        statement_points_len: usize,
    ) -> ProofResult<ParsedProof<F>>
    where
        CommitmentType: WhirCommitmentData<F, MerkleConfig>,
        WhirProof: RoundZeroProofValidator<CommitmentType, MerkleConfig = MerkleConfig>
            + WhirProofRoundData<F, MerkleConfig>,
        VerifierState: UnitToBytes
            + UnitToField<F>
            + FieldToUnitDeserialize<F>
            + PoWChallenge
            + DigestToUnitDeserialize<MerkleConfig>,
    {
        let (committed_ood_points, _) = parsed_commitment.ood_data();
        let mut sumcheck_rounds = Vec::new();
        let mut folding_randomness;
        let initial_combination_randomness;
        if self.params.initial_statement {
            // Derive combination randomness and first sumcheck polynomial
            let [combination_randomness_gen] = verifier_state.challenge_scalars()?;
            initial_combination_randomness = expand_randomness(
                combination_randomness_gen,
                committed_ood_points.len() + statement_points_len,
            );

            // Initial sumcheck
            sumcheck_rounds.reserve_exact(self.params.folding_factor.at_round(0));
            for _ in 0..self.params.folding_factor.at_round(0) {
                let sumcheck_poly_evals: [_; 3] = verifier_state.next_scalars()?;
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
            assert_eq!(committed_ood_points.len(), 0);
            assert_eq!(statement_points_len, 0);

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

        let mut prev_root = parsed_commitment.committed_root().clone();
        let mut domain_gen = self.params.starting_domain.backing_domain.group_gen();
        let mut exp_domain_gen = domain_gen.pow([1 << self.params.folding_factor.at_round(0)]);
        let mut domain_gen_inv = self.params.starting_domain.backing_domain.group_gen_inv();
        let mut domain_size = self.params.starting_domain.size();
        let mut rounds = vec![];

        for r in 0..self.params.n_rounds() {
            let round_params = &self.params.round_parameters[r];

            let new_root = verifier_state.read_digest()?;

            let mut ood_points = vec![F::ZERO; round_params.ood_samples];
            let mut ood_answers = vec![F::ZERO; round_params.ood_samples];
            if round_params.ood_samples > 0 {
                verifier_state.fill_challenge_scalars(&mut ood_points)?;
                verifier_state.fill_next_scalars(&mut ood_answers)?;
            }

            let stir_challenges_indexes = get_challenge_stir_queries(
                domain_size,
                self.params.folding_factor.at_round(r),
                round_params.num_queries,
                verifier_state,
            )?;

            let stir_challenges_points = stir_challenges_indexes
                .iter()
                .map(|index| exp_domain_gen.pow([*index as u64]))
                .collect();

            let (merkle_proof, answers) =
                whir_proof.round_data(r, parsed_commitment.batching_randomness());

            if r > 0 {
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
            } else if !whir_proof.validate_first_round(
                parsed_commitment,
                &stir_challenges_indexes,
                &self.params.leaf_hash_params,
                &self.params.two_to_one_params,
            ) {
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
                stir_challenges_answers: answers.clone(),
                combination_randomness,
                sumcheck_rounds,
                domain_gen_inv,
            });

            folding_randomness = new_folding_randomness;

            prev_root = new_root;
            domain_gen = domain_gen * domain_gen;
            exp_domain_gen = domain_gen.pow([1 << self.params.folding_factor.at_round(r + 1)]);
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

        let (final_merkle_proof, final_randomness_answers) = whir_proof.round_data(
            self.params.n_rounds(),
            parsed_commitment.batching_randomness(),
        );

        if self.params.n_rounds() == 0 {
            // There's only a single round, so we need to validate against the
            // root batched node
            if !whir_proof.validate_first_round(
                parsed_commitment,
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
                    &prev_root,
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
            statement_values_at_random_point: whir_proof.statement_values().to_vec(),
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
            .zip(&proof.initial_combination_randomness)
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
                .zip(&round_proof.combination_randomness)
                .map(|(pt, rand)| pt.eq_poly_outside(&folding_randomness) * rand)
                .sum();

            value += sum_of_claims;
        }

        value
    }

    #[allow(clippy::too_many_lines)]
    pub(crate) fn verify_parsed<CommitmentType>(
        &self,
        statement: &StatementVerifier<F>,
        parsed_commitment: &CommitmentType,
        parsed_proof: &ParsedProof<F>,
    ) -> ProofResult<()>
    where
        CommitmentType: WhirCommitmentData<F, MerkleConfig>,
    {
        let evaluations: Vec<_> = statement.constraints.iter().map(|c| c.1).collect();

        let computed_folds = self
            .params
            .fold_optimisation
            .stir_evaluations_verifier(&parsed_proof, &self.params);

        let mut prev_sumcheck = None;

        let (ood_points, ood_answers) = parsed_commitment.ood_data();
        // Initial sumcheck verification
        if let Some((poly, randomness)) = parsed_proof.initial_sumcheck_rounds.first().cloned() {
            if poly.sum_over_boolean_hypercube()
                != ood_answers
                    .iter()
                    .copied()
                    .chain(evaluations)
                    .zip(&parsed_proof.initial_combination_randomness)
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
            self.compute_w_poly(ood_points, ood_answers, statement, parsed_proof);
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
        parsed_commitment: &ParsedCommitment<F, MerkleConfig::InnerDigest>,
        statement: &StatementVerifier<F>,
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
            parsed_commitment,
            whir_proof,
            statement.constraints.len(),
        )?;

        self.verify_parsed(statement, parsed_commitment, &parsed_proof)
    }
}
