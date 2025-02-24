use std::iter;

use ark_crypto_primitives::merkle_tree::Config;
use ark_ff::{FftField, Field};
use ark_poly::{univariate::DensePolynomial, DenseUVPolynomial, EvaluationDomain, Polynomial};
use nimue::{
    plugins::ark::{FieldChallenges, FieldReader},
    Arthur, ByteChallenges, ByteReader, ProofError, ProofResult,
};
use nimue_pow::{self, PoWChallenge};
use rand::{Rng, SeedableRng};

use crate::{
    poly_utils::{fold::compute_fold_univariate, univariate::naive_interpolation},
    utils,
};

use super::{parameters::StirConfig, StirProof};

pub struct Verifier<F, MerkleConfig, PowStrategy>
where
    F: FftField,
    MerkleConfig: Config,
{
    two_inv: F,
    params: StirConfig<F, MerkleConfig, PowStrategy>,
}

#[derive(Clone)]
struct ParsedCommitment<D> {
    root: D,
}

#[derive(Clone)]
struct ParsedProof<F: Field> {
    rounds: Vec<ParsedRound<F>>,
    final_domain_gen: F,
    final_offset: F,
    final_randomness_indexes: Vec<usize>,
    final_randomness_points: Vec<F>,
    final_randomness_cosets: Vec<Vec<F>>,
    final_randomness_answers: Vec<Vec<F>>,
    final_folding_randomness: F,
    final_coefficients: DensePolynomial<F>,
}

#[derive(Debug, Clone)]
struct ParsedRound<F> {
    folding_randomness: F,
    ood_points: Vec<F>,
    ood_answers: Vec<F>,
    stir_challenges_indexes: Vec<usize>,
    stir_challenges_points: Vec<F>,
    stir_challenges_cosets: Vec<Vec<F>>,
    stir_challenges_answers: Vec<Vec<F>>,
    combination_randomness: F,
    domain_gen: F,
    domain_gen_inv: F,
    domain_offset: F,
    folded_domain_size: usize,
}

impl<F, MerkleConfig, PowStrategy> Verifier<F, MerkleConfig, PowStrategy>
where
    F: FftField,
    MerkleConfig: Config<Leaf = [F]>,
    MerkleConfig::InnerDigest: AsRef<[u8]> + From<[u8; 32]>,
    PowStrategy: nimue_pow::PowStrategy,
{
    pub fn new(params: StirConfig<F, MerkleConfig, PowStrategy>) -> Self {
        Verifier {
            params,
            two_inv: F::from(2).inverse().unwrap(),
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
        stir_proof: &StirProof<MerkleConfig, F>,
    ) -> ProofResult<ParsedProof<F>> {
        // Derive initial combination randomness
        let [mut folding_randomness] = arthur.challenge_scalars()?;

        // PoW
        if self.params.starting_folding_pow_bits > 0. {
            arthur.challenge_pow::<PowStrategy>(self.params.starting_folding_pow_bits)?;
        }

        let fold_size = 1 << self.params.folding_factor;
        let mut prev_root = parsed_commitment.root.clone();
        let mut domain_gen = self.params.starting_domain.backing_domain.group_gen();
        let mut exp_domain_gen = domain_gen.pow([1 << self.params.folding_factor]);
        let root_of_unity = self.params.starting_domain.backing_domain.group_gen();
        let mut domain_offset = F::ONE;
        let mut exp_domain_offset = F::ONE;
        let mut domain_gen_inv = self.params.starting_domain.backing_domain.group_gen_inv();
        let mut domain_size = self.params.starting_domain.size();
        let mut rounds = vec![];

        for r in 0..self.params.n_rounds() {
            let (merkle_proof, answers) = &stir_proof.merkle_proofs[r];
            let round_params = &self.params.round_parameters[r];

            let new_root: [u8; 32] = arthur.next_bytes()?;

            // PHASE 2:
            let mut ood_points = vec![F::ZERO; round_params.ood_samples];
            let mut ood_answers = vec![F::ZERO; round_params.ood_samples];
            if round_params.ood_samples > 0 {
                arthur.fill_challenge_scalars(&mut ood_points)?;
                arthur.fill_next_scalars(&mut ood_answers)?;
            }

            // PHASE 3:
            let mut stir_queries_seed = [0u8; 32];
            arthur.fill_challenge_bytes(&mut stir_queries_seed)?;
            let mut stir_gen = rand_chacha::ChaCha20Rng::from_seed(stir_queries_seed);
            let folded_domain_size = domain_size / (1 << self.params.folding_factor);
            let stir_challenges_indexes = utils::dedup(
                (0..round_params.num_queries).map(|_| stir_gen.gen_range(0..folded_domain_size)),
            );
            let stir_challenges_points = stir_challenges_indexes
                .iter()
                .map(|index| exp_domain_offset * exp_domain_gen.pow([*index as u64]))
                .collect();

            let stir_challenges_cosets: Vec<Vec<F>> = stir_challenges_indexes
                .iter()
                .map(|&i| {
                    // Note: the domain has not been restructured the same way that the evaluations have.
                    (i..i + fold_size * folded_domain_size)
                        .step_by(folded_domain_size)
                        .map(|j| domain_offset * domain_gen.pow([j as u64]))
                        .collect()
                })
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

            let [combination_randomness] = arthur.challenge_scalars()?;
            let [new_folding_randomness] = arthur.challenge_scalars()?;

            rounds.push(ParsedRound {
                folding_randomness,
                ood_points,
                ood_answers,
                stir_challenges_indexes,
                stir_challenges_points,
                stir_challenges_cosets,
                stir_challenges_answers: answers.to_vec(),
                combination_randomness,
                domain_gen,
                domain_gen_inv,
                domain_offset,
                folded_domain_size,
            });

            folding_randomness = new_folding_randomness;

            prev_root = new_root.into();
            exp_domain_gen = exp_domain_gen * exp_domain_gen;
            domain_offset = domain_offset * domain_offset * root_of_unity;
            exp_domain_offset = domain_offset.pow([1 << self.params.folding_factor]);
            domain_gen = domain_gen * domain_gen;
            domain_gen_inv = domain_gen_inv * domain_gen_inv;
            domain_size /= 2;
        }

        let mut final_coefficients = vec![F::ZERO; 1 << self.params.final_log_degree];
        arthur.fill_next_scalars(&mut final_coefficients)?;
        let final_coefficients = DensePolynomial::from_coefficients_vec(final_coefficients);

        // Final queries verify
        let mut queries_seed = [0u8; 32];
        arthur.fill_challenge_bytes(&mut queries_seed)?;
        let mut final_gen = rand_chacha::ChaCha20Rng::from_seed(queries_seed);
        let folded_domain_size = domain_size / (1 << self.params.folding_factor);
        let final_randomness_indexes = utils::dedup(
            (0..self.params.final_queries).map(|_| final_gen.gen_range(0..folded_domain_size)),
        );
        let final_randomness_points = final_randomness_indexes
            .iter()
            .map(|index| exp_domain_offset * exp_domain_gen.pow([*index as u64]))
            .collect();

        let final_randomness_cosets: Vec<Vec<F>> = final_randomness_indexes
            .iter()
            .map(|&i| {
                // Note: the domain has not been restructured the same way that the evaluations have.
                (i..i + fold_size * folded_domain_size)
                    .step_by(folded_domain_size)
                    .map(|j| domain_offset * domain_gen.pow([j as u64]))
                    .collect()
            })
            .collect();

        let (final_merkle_proof, final_randomness_answers) =
            &stir_proof.merkle_proofs[stir_proof.merkle_proofs.len() - 1];
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

        Ok(ParsedProof {
            rounds,
            final_domain_gen: domain_gen,
            final_offset: domain_offset,
            final_randomness_indexes,
            final_randomness_points,
            final_randomness_cosets,
            final_randomness_answers: final_randomness_answers.to_vec(),
            final_folding_randomness: folding_randomness,
            final_coefficients,
        })
    }

    pub fn verify(
        &self,
        arthur: &mut Arthur,
        stir_proof: &StirProof<MerkleConfig, F>,
    ) -> ProofResult<()> {
        // PHASE 1:
        // We first do a pass in which we rederive all the FS challenges and verify Merkle paths.
        // We take care of all the crytpo here.
        let parsed_commitment = self.parse_commitment(arthur)?;
        let parsed = self.parse_proof(arthur, &parsed_commitment, stir_proof)?;

        // PHASE 2:
        // We do a pass which at each point defines and update a virtual function
        let committed_functions_cosets: Vec<Vec<Vec<F>>> = parsed
            .rounds
            .iter()
            .map(|r| &r.stir_challenges_cosets)
            .chain(iter::once(&parsed.final_randomness_cosets))
            .cloned()
            .collect();

        let committed_functions_answers: Vec<Vec<Vec<F>>> = parsed
            .rounds
            .iter()
            .map(|r| &r.stir_challenges_answers)
            .chain(iter::once(&parsed.final_randomness_answers))
            .cloned()
            .collect();

        let domain_gens: Vec<_> = parsed
            .rounds
            .iter()
            .map(|r| r.domain_gen)
            .chain(iter::once(parsed.final_domain_gen))
            .collect();

        let domain_offsets: Vec<_> = parsed
            .rounds
            .iter()
            .map(|r| r.domain_offset)
            .chain(iter::once(parsed.final_offset))
            .collect();

        // We first compute the evaluations of the first function
        let initial_folding_randomness = if parsed.rounds.is_empty() {
            parsed.final_folding_randomness
        } else {
            parsed.rounds[0].folding_randomness
        };

        // This array is the evaluations of the current function (already includes the virtual evaluations!)
        let mut evaluations: Vec<_> = committed_functions_cosets[0]
            .iter()
            .zip(committed_functions_answers[0].iter())
            .map(|(points, evals)| {
                // Here we need to interpolate
                naive_interpolation(points.iter().copied().zip(evals.iter().copied()))
                    .evaluate(&initial_folding_randomness)
            })
            .collect();

        let mut domain_size = self.params.starting_domain.backing_domain.size();
        let coset_domain_size = 1 << self.params.folding_factor;

        for (r_num, (r, (domain_gen, domain_offset))) in parsed
            .rounds
            .iter()
            .zip(
                domain_gens
                    .iter()
                    .skip(1)
                    .zip(domain_offsets.iter().skip(1)),
            )
            .enumerate()
        {
            // The next virtual function is defined by the following
            // TODO: This actually is just a single value that we need
            let combination_randomness = r.combination_randomness;
            let quotient_set: Vec<_> = r
                .ood_points
                .iter()
                .chain(&r.stir_challenges_points)
                .copied()
                .collect();
            let quotient_answers: Vec<_> =
                r.ood_answers.iter().chain(&evaluations).copied().collect();
            let num_terms = quotient_set.len();

            let ans_polynomial = naive_interpolation(
                quotient_set
                    .iter()
                    .zip(quotient_answers)
                    .map(|(x, y)| (*x, y)),
            );

            // The points that we are querying the new function at
            let evaluation_indexes = if r_num == parsed.rounds.len() - 1 {
                &parsed.final_randomness_indexes
            } else {
                &parsed.rounds[r_num + 1].stir_challenges_indexes
            }
            .clone();

            let coset_generator_inv = r
                .domain_gen_inv
                .pow([(domain_size / coset_domain_size) as u64]);

            // The evaluations of the previous committed function ->
            // need to be reshaped into evaluations of the virtual function f'
            let committed_answers = &committed_functions_answers[r_num + 1];
            let mut new_evaluations = Vec::with_capacity(committed_answers.len());

            for (index, answer) in evaluation_indexes.into_iter().zip(committed_answers) {
                // Coset eval is the evaluations of the virtual function on the coset
                let mut coset_evals = Vec::with_capacity(1 << self.params.folding_factor);
                let coset_offset_inv = r.domain_gen_inv.pow([index as u64]); // What does this do?
                #[allow(clippy::needless_range_loop)]
                for j in 0..1 << self.params.folding_factor {
                    // TODO: Optimize
                    let evaluation_point = *domain_offset
                        * domain_gen.pow([(index + j * r.folded_domain_size) as u64]);

                    let numerator = answer[j] - ans_polynomial.evaluate(&evaluation_point);

                    // Just an eval of the vanishing polynomial
                    let denom: F = quotient_set
                        .iter()
                        .map(|point| evaluation_point - point)
                        .product();
                    let denom_inv = denom.inverse().unwrap();

                    // Scaling factor:
                    // If xr = 1 this is just the num_terms
                    // If xr != 1 this is (1 - (xr)^num_terms)/(1 - xr)
                    let common_factor = evaluation_point * combination_randomness;
                    let common_factor_inverse = (F::ONE - common_factor).inverse().unwrap();

                    let scale_factor = if common_factor != F::ONE {
                        (F::ONE - common_factor.pow([(num_terms + 1) as u64]))
                            * common_factor_inverse
                    } else {
                        F::from((num_terms + 1) as u64)
                    };

                    coset_evals.push(scale_factor * numerator * denom_inv);
                }

                let eval = compute_fold_univariate(
                    &coset_evals,
                    r.folding_randomness,
                    coset_offset_inv,
                    coset_generator_inv,
                    self.two_inv,
                    self.params.folding_factor,
                );

                new_evaluations.push(eval);
            }

            evaluations = new_evaluations;
            domain_size /= 2;
        }

        // Check the foldings computed from the proof match the evaluations of the polynomial
        let final_folds = &evaluations;
        let final_evaluations = parsed
            .final_randomness_points
            .iter()
            .map(|point| parsed.final_coefficients.evaluate(point));
        let final_evaluations: Vec<_> = final_evaluations.collect();
        if !final_folds
            .iter()
            .zip(&final_evaluations)
            .all(|(&fold, &eval)| fold == eval)
        {
            for (a, b) in final_folds.iter().zip(final_evaluations) {
                println!("{}\t{}", a, b);
            }
            dbg!("Error path 3");
            return Err(ProofError::InvalidProof);
        }

        Ok(())
    }
}
