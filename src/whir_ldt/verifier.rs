use std::iter;

use ark_crypto_primitives::merkle_tree::Config;
use ark_ff::{FftField, PrimeField};
use ark_poly::EvaluationDomain;
use nimue::{
    plugins::ark::{FieldChallenges, FieldReader},
    plugins::pow::PoWChallenge,
    Arthur, ByteChallenges, ByteReader, ProofError, ProofResult,
};
use rand::{Rng, SeedableRng};

use crate::{
    poly_utils::{eq_poly_outside, fold::compute_fold, MultilinearPoint},
    sumcheck::proof::SumcheckPolynomial,
    utils::{self, expand_randomness},
};

use super::{parameters::WhirConfig, WhirProof};

pub struct Verifier<F, MerkleConfig>
where
    F: FftField,
    MerkleConfig: Config,
{
    params: WhirConfig<F, MerkleConfig>,
    two_inv: F,
}

#[derive(Clone)]
struct ParsedCommitment<D> {
    root: D,
}

#[derive(Clone)]
struct ParsedProof<F> {
    rounds: Vec<ParsedRound<F>>,
    final_domain_gen_inv: F,
    final_randomness_indexes: Vec<usize>,
    final_randomness_answers: Vec<Vec<F>>,
    final_folding_randomness: MultilinearPoint<F>,
    final_coefficient: F,
}

#[derive(Debug, Clone)]
struct ParsedRound<F> {
    folding_randomness: MultilinearPoint<F>,
    ood_points: Vec<F>,
    ood_answers: Vec<F>,
    stir_challenges_indexes: Vec<usize>,
    stir_challenges_answers: Vec<Vec<F>>,
    combination_randomness: Vec<F>,
    sumcheck_rounds: Vec<(SumcheckPolynomial<F>, F)>,
    domain_gen: F,
    domain_gen_inv: F,
}

impl<F, MerkleConfig> Verifier<F, MerkleConfig>
where
    F: FftField + PrimeField,
    MerkleConfig: Config<Leaf = Vec<F>>,
    MerkleConfig::InnerDigest: AsRef<[u8]> + From<[u8; 32]>,
{
    pub fn new(params: WhirConfig<F, MerkleConfig>) -> Self {
        Verifier {
            params,
            two_inv: F::from(2).inverse().unwrap(), // The only inverse in the entire code :)
        }
    }

    fn parse_commitment(
        &self,
        arthur: &mut Arthur,
    ) -> ProofResult<ParsedCommitment<MerkleConfig::InnerDigest>> {
        let root: [u8; 32] = arthur.next_bytes()?;

        Ok(ParsedCommitment { root: root.into() })
    }

    fn parse_proof(
        &self,
        arthur: &mut Arthur,
        parsed_commitment: &ParsedCommitment<MerkleConfig::InnerDigest>,
        whir_proof: &WhirProof<MerkleConfig>,
    ) -> ProofResult<ParsedProof<F>> {
        // Derive initial combination randomness
        let mut folding_randomness = vec![F::ZERO; self.params.folding_factor];
        arthur.fill_challenge_scalars(&mut folding_randomness)?;
        let mut folding_randomness = MultilinearPoint(folding_randomness);

        let mut prev_root = parsed_commitment.root.clone();
        let mut domain_gen = self.params.starting_domain.backing_domain.group_gen();
        let mut domain_gen_inv = self.params.starting_domain.backing_domain.group_gen_inv();
        let mut domain_size = self.params.starting_domain.size();
        let mut rounds = vec![];

        for r in 0..self.params.n_rounds() {
            let (merkle_proof, answers) = &whir_proof.0[r];
            let round_params = &self.params.round_parameters[r];

            let new_root: [u8; 32] = arthur.next_bytes()?;

            let mut ood_points = vec![F::ZERO; round_params.ood_samples];
            let mut ood_answers = vec![F::ZERO; round_params.ood_samples];
            if round_params.ood_samples > 0 {
                arthur.fill_challenge_scalars(&mut ood_points)?;
                arthur.fill_next_scalars(&mut ood_answers)?;
            }

            let mut stir_queries_seed = [0u8; 32];
            arthur.fill_challenge_bytes(&mut stir_queries_seed)?;
            let mut stir_gen = rand_chacha::ChaCha20Rng::from_seed(stir_queries_seed);
            let folded_domain_size = domain_size / (1 << self.params.folding_factor);
            let stir_challenges_indexes = utils::dedup(
                (0..round_params.num_queries).map(|_| stir_gen.gen_range(0..folded_domain_size)),
            );

            if !merkle_proof
                .verify(
                    &self.params.leaf_hash_params,
                    &self.params.two_to_one_params,
                    &prev_root,
                    answers,
                )
                .unwrap()
                || merkle_proof.leaf_indexes != stir_challenges_indexes
            {
                return Err(ProofError::InvalidProof);
            }

            if round_params.pow_bits > 0 {
                arthur.challenge_pow(round_params.pow_bits)?;
            }

            let [combination_randomness_gen] = arthur.challenge_scalars()?;
            let combination_randomness = expand_randomness(
                combination_randomness_gen,
                stir_challenges_indexes.len() + 1,
            );

            let mut sumcheck_rounds = Vec::with_capacity(self.params.folding_factor);
            for _ in 0..self.params.folding_factor {
                let sumcheck_poly_evals: [F; 3] = arthur.next_scalars()?;
                let sumcheck_poly = SumcheckPolynomial::new(sumcheck_poly_evals.to_vec(), 1);
                let [folding_randomness_single] = arthur.challenge_scalars()?;
                sumcheck_rounds.push((sumcheck_poly, folding_randomness_single));
            }

            let new_folding_randomness =
                MultilinearPoint(sumcheck_rounds.iter().map(|&(_, r)| r).rev().collect());

            rounds.push(ParsedRound {
                folding_randomness,
                ood_points,
                ood_answers,
                stir_challenges_indexes,
                stir_challenges_answers: answers.to_vec(),
                combination_randomness,
                sumcheck_rounds,
                domain_gen,
                domain_gen_inv,
            });

            folding_randomness = new_folding_randomness;

            prev_root = new_root.into();
            domain_gen = domain_gen * domain_gen;
            domain_gen_inv = domain_gen_inv * domain_gen_inv;
            domain_size /= 2;
        }

        let [final_coefficient]: [F; 1] = arthur.next_scalars()?;

        // Final queries verify
        let mut queries_seed = [0u8; 32];
        arthur.fill_challenge_bytes(&mut queries_seed)?;
        let mut final_gen = rand_chacha::ChaCha20Rng::from_seed(queries_seed);
        let folded_domain_size = domain_size / (1 << self.params.folding_factor);
        let final_randomness_indexes = utils::dedup(
            (0..self.params.final_queries).map(|_| final_gen.gen_range(0..folded_domain_size)),
        );

        let (final_merkle_proof, final_randomness_answers) = &whir_proof.0[whir_proof.0.len() - 1];
        if !final_merkle_proof
            .verify(
                &self.params.leaf_hash_params,
                &self.params.two_to_one_params,
                &prev_root,
                final_randomness_answers,
            )
            .unwrap()
            || final_merkle_proof.leaf_indexes != final_randomness_indexes
        {
            return Err(ProofError::InvalidProof);
        }

        if self.params.final_pow_bits > 0 {
            arthur.challenge_pow(self.params.final_pow_bits)?;
        }

        Ok(ParsedProof {
            rounds,
            final_domain_gen_inv: domain_gen_inv,
            final_folding_randomness: folding_randomness,
            final_randomness_indexes,
            final_randomness_answers: final_randomness_answers.to_vec(),
            final_coefficient,
        })
    }

    fn compute_v_poly(&self, proof: &ParsedProof<F>) -> F {
        let mut num_variables = self.params.mv_parameters.num_variables;

        let mut folding_randomness = MultilinearPoint(
            iter::once(&proof.final_folding_randomness.0)
                .chain(proof.rounds.iter().rev().map(|r| &r.folding_randomness.0))
                .flatten()
                .copied()
                .collect(),
        );

        let mut value = F::ZERO;

        for round_proof in &proof.rounds {
            num_variables -= self.params.folding_factor;
            folding_randomness = MultilinearPoint(folding_randomness.0[..num_variables].to_vec());

            let exp_domain_gen = round_proof
                .domain_gen
                .pow([1 << self.params.folding_factor]);
            let ood_points = &round_proof.ood_points;
            let stir_challenges_indexes = &round_proof.stir_challenges_indexes;
            let stir_challenges: Vec<_> = ood_points
                .iter()
                .cloned()
                .chain(
                    stir_challenges_indexes
                        .iter()
                        .map(|i| exp_domain_gen.pow([*i as u64])),
                )
                .map(|univariate| {
                    MultilinearPoint::expand_from_univariate(univariate, num_variables)
                })
                .collect();

            let sum_of_claims: F = stir_challenges
                .into_iter()
                .map(|point| eq_poly_outside(&point, &folding_randomness))
                .zip(&round_proof.combination_randomness)
                .map(|(point, rand)| point * rand)
                .sum();

            value = value + sum_of_claims;
        }

        value
    }

    fn compute_folds(&self, parsed: &ParsedProof<F>) -> Vec<Vec<F>> {
        let mut domain_size = self.params.starting_domain.backing_domain.size();
        let coset_domain_size = 1 << self.params.folding_factor;

        let mut result = Vec::new();

        for round in &parsed.rounds {
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
                        self.params.folding_factor,
                    )
                })
                .collect();
            result.push(evaluations);
            domain_size /= 2;
        }

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
                    self.params.folding_factor,
                )
            })
            .collect();
        result.push(evaluations);

        result
    }

    pub fn verify(
        &self,
        arthur: &mut Arthur,
        whir_proof: &WhirProof<MerkleConfig>,
    ) -> ProofResult<()> {
        // We first do a pass in which we rederive all the FS challenges
        // Then we will check the algebraic part (so to optimise inversions)
        let parsed_commitment = self.parse_commitment(arthur)?;
        let parsed = self.parse_proof(arthur, &parsed_commitment, whir_proof)?;

        let computed_folds = self.compute_folds(&parsed);

        let mut prev: Option<(SumcheckPolynomial<F>, F)> = None;
        for (round, folds) in parsed.rounds.iter().zip(&computed_folds) {
            let (sumcheck_poly, new_randomness) = &round.sumcheck_rounds[0].clone();

            let values = round.ood_answers.iter().copied().chain(folds.clone());

            let prev_eval = if let Some((prev_poly, randomness)) = prev {
                prev_poly.evaluate_at_point(&randomness.into())
            } else {
                F::ZERO
            };
            let claimed_sum = prev_eval
                + values
                    .zip(&round.combination_randomness)
                    .map(|(val, rand)| val * rand)
                    .sum::<F>();

            if sumcheck_poly.sum_over_hypercube() != claimed_sum {
                return Err(ProofError::InvalidProof);
            }

            prev = Some((sumcheck_poly.clone(), *new_randomness));

            // Check the rest of the round
            for (sumcheck_poly, new_randomness) in &round.sumcheck_rounds[1..] {
                let (prev_poly, randomness) = prev.unwrap();
                if sumcheck_poly.sum_over_hypercube()
                    != prev_poly.evaluate_at_point(&randomness.into())
                {
                    return Err(ProofError::InvalidProof);
                }
                prev = Some((sumcheck_poly.clone(), *new_randomness));
            }
        }

        // Check the foldings
        let final_folds = &computed_folds[computed_folds.len() - 1];
        if !final_folds.iter().all(|&x| x == parsed.final_coefficient) {
            return Err(ProofError::InvalidProof);
        }

        // Check the final sumcheck
        let evaluation_of_v_poly = self.compute_v_poly(&parsed);

        let prev_sumcheck_poly_eval = if let Some((prev_poly, randomness)) = prev {
            prev_poly.evaluate_at_point(&randomness.into())
        } else {
            F::ZERO
        };

        if prev_sumcheck_poly_eval != evaluation_of_v_poly * parsed.final_coefficient {
            return Err(ProofError::InvalidProof);
        }

        Ok(())
    }
}
