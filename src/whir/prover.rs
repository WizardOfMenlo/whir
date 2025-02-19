use super::{committer::Witness, parameters::WhirConfig, statement::{EvaluationWeights, Statement}, WhirProof};
use crate::{
    domain::Domain,
    ntt::expand_from_coeff,
    parameters::FoldType,
    poly_utils::{
        coeffs::CoefficientList,
        fold::{compute_fold, restructure_evaluations},
        MultilinearPoint,
    },
    sumcheck::SumcheckSingle,
    utils::{self, expand_randomness},
};
use ark_crypto_primitives::merkle_tree::{Config, MerkleTree, MultiPath};
use ark_ff::FftField;
use ark_poly::EvaluationDomain;
use nimue::{
    plugins::ark::{FieldChallenges, FieldWriter},
    ByteChallenges, ByteWriter, ProofResult,
};
use nimue_pow::{self, PoWChallenge};

use crate::whir::fs_utils::{get_challenge_stir_queries, DigestWriter};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

pub struct Prover<F, MerkleConfig, PowStrategy>(pub WhirConfig<F, MerkleConfig, PowStrategy>)
where
    F: FftField,
    MerkleConfig: Config;

impl<F, MerkleConfig, PowStrategy> Prover<F, MerkleConfig, PowStrategy>
where
    F: FftField,
    MerkleConfig: Config<Leaf = [F]>,
    PowStrategy: nimue_pow::PowStrategy,
{
    fn validate_parameters(&self) -> bool {
        self.0.mv_parameters.num_variables
            == (self.0.n_rounds() + 1) * self.0.folding_factor + self.0.final_sumcheck_rounds
    }

    fn validate_statement(&self, statement: &Statement<F>) -> bool {
        if !statement.num_variables() == self.0.mv_parameters.num_variables
        {
            return false;
        }
        if !self.0.initial_statement && !statement.constraints.is_empty() {
            return false;
        }
        true
    }

    fn validate_witness(&self, witness: &Witness<F, MerkleConfig>) -> bool {
        assert_eq!(witness.ood_points.len(), witness.ood_answers.len());
        if !self.0.initial_statement {
            assert!(witness.ood_points.is_empty());
        }
        witness.polynomial.num_variables() == self.0.mv_parameters.num_variables
    }

    pub fn prove<Merlin>(
        &self,
        merlin: &mut Merlin,
        statement_new: &mut Statement<F>,
        witness: Witness<F, MerkleConfig>,
        verifier_statement: &mut Statement<F>,
    ) -> ProofResult<WhirProof<MerkleConfig, F>>
    where
        Merlin: FieldChallenges<F>
            + FieldWriter<F>
            + ByteChallenges
            + ByteWriter
            + PoWChallenge
            + DigestWriter<MerkleConfig>,
    {
        assert!(self.validate_parameters());
        assert!(self.validate_statement(&statement_new));
        assert!(self.validate_witness(&witness));

       let statement = statement_new.clone();
        for (point, evaluation) in witness.ood_points.into_iter().zip(witness.ood_answers) {
            let weights: Box<EvaluationWeights<F>> = Box::new(EvaluationWeights::new(MultilinearPoint::expand_from_univariate(point, self.0.mv_parameters.num_variables)));
            statement_new.add_constraint_in_front(weights.clone(), evaluation);
        }

        
        let mut sumcheck_prover = None;
        let folding_randomness = if self.0.initial_statement {
            let [combination_randomness_gen] = merlin.challenge_scalars()?;
            sumcheck_prover = {
                let mut sumcheck = SumcheckSingle::new(witness.polynomial.clone());
                sumcheck.add_weighted_sum(
                    statement_new,
                    combination_randomness_gen
                );

                Some(sumcheck)
            };

            sumcheck_prover
                .as_mut()
                .unwrap()
                .compute_sumcheck_polynomials::<PowStrategy, Merlin>(
                    merlin,
                    self.0.folding_factor,
                    self.0.starting_folding_pow_bits,
                )?
        } else {
            let mut folding_randomness = vec![F::ZERO; self.0.folding_factor];
            merlin.fill_challenge_scalars(&mut folding_randomness)?;

            if self.0.starting_folding_pow_bits > 0. {
                merlin.challenge_pow::<PowStrategy>(self.0.starting_folding_pow_bits)?;
            }
            MultilinearPoint(folding_randomness)
        };
        let mut randomness_vec = vec![F::ZERO; self.0.mv_parameters.num_variables];
        randomness_vec[..folding_randomness.0.len()].copy_from_slice(&folding_randomness.0);

        let round_state = RoundState {
            domain: self.0.starting_domain.clone(),
            round: 0,
            sumcheck_prover,
            folding_randomness,
            coefficients: witness.polynomial,
            prev_merkle: witness.merkle_tree,
            prev_merkle_answers: witness.merkle_leaves,
            merkle_proofs: vec![],
        };

        self.round(merlin, round_state, &statement, verifier_statement, &mut randomness_vec)
    }

    fn round<Merlin>(
        &self,
        merlin: &mut Merlin,
        mut round_state: RoundState<F, MerkleConfig>,
        prover_statement: &Statement<F>,
        verifier_statement: &mut Statement<F>,
        randomness_vec: &mut Vec<F>
    ) -> ProofResult<WhirProof<MerkleConfig, F>>
    where
        Merlin: FieldChallenges<F>
            + ByteChallenges
            + FieldWriter<F>
            + ByteWriter
            + PoWChallenge
            + DigestWriter<MerkleConfig>,
    {
        // Fold the coefficients
        let folded_coefficients = round_state
            .coefficients
            .fold(&round_state.folding_randomness);

        let num_variables =
            self.0.mv_parameters.num_variables - (round_state.round + 1) * self.0.folding_factor;

        // Base case
        if round_state.round == self.0.n_rounds() {
            // Coefficients of the polynomial
            merlin.add_scalars(folded_coefficients.coeffs())?;

            // Final verifier queries and answers
            let final_challenge_indexes = get_challenge_stir_queries(
                round_state.domain.size(),
                self.0.folding_factor,
                self.0.final_queries,
                merlin,
            )?;

            let merkle_proof = round_state
                .prev_merkle
                .generate_multi_proof(final_challenge_indexes.clone())
                .unwrap();
            let fold_size = 1 << self.0.folding_factor;
            let answers = final_challenge_indexes
                .into_iter()
                .map(|i| {
                    round_state.prev_merkle_answers[i * fold_size..(i + 1) * fold_size].to_vec()
                })
                .collect();
            round_state.merkle_proofs.push((merkle_proof, answers));

            // PoW
            if self.0.final_pow_bits > 0. {
                merlin.challenge_pow::<PowStrategy>(self.0.final_pow_bits)?;
            }

            // Final sumcheck
            if self.0.final_sumcheck_rounds > 0 {
                round_state
                    .sumcheck_prover
                    .unwrap_or_else(|| SumcheckSingle::new(folded_coefficients.clone()))
                    .compute_sumcheck_polynomials::<PowStrategy, Merlin>(
                        merlin,
                        self.0.final_sumcheck_rounds,
                        self.0.final_folding_pow_bits,
                    )?;
            }

            let mut randomness_vec_rev = randomness_vec.clone();
            randomness_vec_rev.reverse();

            for (weights, value) in &prover_statement.constraints {
                match weights.get_point_if_evaluation() {
                    Some(point) => {
                        verifier_statement.add_constraint(Box::new(EvaluationWeights::new(point.clone())), value.clone());
                    }
                    None => {
                        let affine_claim_verifier = weights.get_statement_for_verifier(&MultilinearPoint(randomness_vec_rev.clone()));
                        if let Some(affine_claim_verifier) = affine_claim_verifier {
                            verifier_statement.add_constraint(Box::new(affine_claim_verifier), value.clone());
                        }
                    }
                }
            }
            return Ok(WhirProof(round_state.merkle_proofs));
        }

        let round_params = &self.0.round_parameters[round_state.round];

        // Fold the coefficients, and compute fft of polynomial (and commit)
        let new_domain = round_state.domain.scale(2);
        let expansion = new_domain.size() / folded_coefficients.num_coeffs();
        let evals = expand_from_coeff(folded_coefficients.coeffs(), expansion);
        // TODO: `stack_evaluations` and `restructure_evaluations` are really in-place algorithms.
        // They also partially overlap and undo one another. We should merge them.
        let folded_evals = utils::stack_evaluations(evals, self.0.folding_factor);
        let folded_evals = restructure_evaluations(
            folded_evals,
            self.0.fold_optimisation,
            new_domain.backing_domain.group_gen(),
            new_domain.backing_domain.group_gen_inv(),
            self.0.folding_factor,
        );

        #[cfg(not(feature = "parallel"))]
        let leafs_iter = folded_evals.chunks_exact(1 << self.0.folding_factor);
        #[cfg(feature = "parallel")]
        let leafs_iter = folded_evals.par_chunks_exact(1 << self.0.folding_factor);
        let merkle_tree = MerkleTree::<MerkleConfig>::new(
            &self.0.leaf_hash_params,
            &self.0.two_to_one_params,
            leafs_iter,
        )
        .unwrap();

        let root = merkle_tree.root();
        merlin.add_digest(root)?;

        // OOD Samples
        let mut ood_points = vec![F::ZERO; round_params.ood_samples];
        let mut ood_answers = Vec::with_capacity(round_params.ood_samples);
        if round_params.ood_samples > 0 {
            merlin.fill_challenge_scalars(&mut ood_points)?;
            ood_answers.extend(ood_points.iter().map(|ood_point| {
                folded_coefficients.evaluate(&MultilinearPoint::expand_from_univariate(
                    *ood_point,
                    num_variables,
                ))
            }));
            merlin.add_scalars(&ood_answers)?;
        }

        // STIR queries
        let stir_challenges_indexes = get_challenge_stir_queries(
            round_state.domain.size(),
            self.0.folding_factor,
            round_params.num_queries,
            merlin,
        )?;
        let domain_scaled_gen = round_state
            .domain
            .backing_domain
            .element(1 << self.0.folding_factor);
        let stir_challenges: Vec<_> = ood_points
            .into_iter()
            .chain(
                stir_challenges_indexes
                    .iter()
                    .map(|i| domain_scaled_gen.pow([*i as u64])),
            )
            .map(|univariate| MultilinearPoint::expand_from_univariate(univariate, num_variables))
            .collect();

        let merkle_proof = round_state
            .prev_merkle
            .generate_multi_proof(stir_challenges_indexes.clone())
            .unwrap();
        let fold_size = 1 << self.0.folding_factor;
        let answers: Vec<_> = stir_challenges_indexes
            .iter()
            .map(|i| round_state.prev_merkle_answers[i * fold_size..(i + 1) * fold_size].to_vec())
            .collect();
        // Evaluate answers in the folding randomness.
        let mut stir_evaluations = ood_answers.clone();
        match self.0.fold_optimisation {
            FoldType::Naive => {
                // See `Verifier::compute_folds_full`
                let domain_size = round_state.domain.backing_domain.size();
                let domain_gen = round_state.domain.backing_domain.element(1);
                let domain_gen_inv = domain_gen.inverse().unwrap();
                let coset_domain_size = 1 << self.0.folding_factor;
                let coset_generator_inv =
                    domain_gen_inv.pow([(domain_size / coset_domain_size) as u64]);
                stir_evaluations.extend(stir_challenges_indexes.iter().zip(&answers).map(
                    |(index, answers)| {
                        // The coset is w^index * <w_coset_generator>
                        //let _coset_offset = domain_gen.pow(&[*index as u64]);
                        let coset_offset_inv = domain_gen_inv.pow([*index as u64]);

                        compute_fold(
                            answers,
                            &round_state.folding_randomness.0,
                            coset_offset_inv,
                            coset_generator_inv,
                            F::from(2).inverse().unwrap(),
                            self.0.folding_factor,
                        )
                    },
                ))
            }
            FoldType::ProverHelps => stir_evaluations.extend(answers.iter().map(|answers| {
                CoefficientList::new(answers.to_vec()).evaluate(&round_state.folding_randomness)
            })),
        }
        round_state.merkle_proofs.push((merkle_proof, answers));

        // PoW
        if round_params.pow_bits > 0. {
            merlin.challenge_pow::<PowStrategy>(round_params.pow_bits)?;
        }

        // Randomness for combination
        let [combination_randomness_gen] = merlin.challenge_scalars()?;
        let combination_randomness =
            expand_randomness(combination_randomness_gen, stir_challenges.len());

        let mut sumcheck_prover = round_state
            .sumcheck_prover
            .take()
            .map(|mut sumcheck_prover| {
                sumcheck_prover.add_new_equality(
                    &stir_challenges,
                    &stir_evaluations,
                    &combination_randomness,
                );
                sumcheck_prover
            })
            .unwrap_or_else(|| {
                let mut sumcheck = SumcheckSingle::new(folded_coefficients.clone());
                sumcheck.add_new_equality(
                    &stir_challenges,
                    &stir_evaluations,
                    &combination_randomness,
                );
                sumcheck
            });

        let folding_randomness = sumcheck_prover
            .compute_sumcheck_polynomials::<PowStrategy, Merlin>(
                merlin,
                self.0.folding_factor,
                round_params.folding_pow_bits,
            )?;

        let start_idx = (round_state.round + 1) * self.0.folding_factor;
        randomness_vec[start_idx..start_idx + folding_randomness.0.len()]
            .copy_from_slice(&folding_randomness.0);
 
        let round_state = RoundState {
            round: round_state.round + 1,
            domain: new_domain,
            sumcheck_prover: Some(sumcheck_prover),
            folding_randomness,
            coefficients: folded_coefficients, // TODO: Is this redundant with `sumcheck_prover.coeff` ?
            prev_merkle: merkle_tree,
            prev_merkle_answers: folded_evals,
            merkle_proofs: round_state.merkle_proofs,
        };

        self.round(merlin, round_state, prover_statement, verifier_statement, randomness_vec)
    }
}

struct RoundState<F, MerkleConfig>
where
    F: FftField,
    MerkleConfig: Config,
{
    round: usize,
    domain: Domain<F>,
    sumcheck_prover: Option<SumcheckSingle<F>>,
    folding_randomness: MultilinearPoint<F>,
    coefficients: CoefficientList<F>,
    prev_merkle: MerkleTree<MerkleConfig>,
    prev_merkle_answers: Vec<F>,
    merkle_proofs: Vec<(MultiPath<MerkleConfig>, Vec<Vec<F>>)>,
}
