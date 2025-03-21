use super::{
    committer::Witness,
    parameters::WhirConfig,
    statement::{Statement, Weights},
    WhirProof,
};
use crate::{
    domain::Domain,
    ntt::expand_from_coeff,
    poly_utils::{
        coeffs::CoefficientList, fold::transform_evaluations, multilinear::MultilinearPoint,
    },
    sumcheck::SumcheckSingle,
    utils::expand_randomness,
    whir::utils::sample_ood_points,
};
use ark_crypto_primitives::merkle_tree::{Config, MerkleTree, MultiPath};
use ark_ff::FftField;
use ark_poly::EvaluationDomain;
use nimue::{
    plugins::ark::{FieldChallenges, FieldWriter},
    ByteChallenges, ProofResult,
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
            == self.0.folding_factor.total_number(self.0.n_rounds()) + self.0.final_sumcheck_rounds
    }

    fn validate_statement(&self, statement: &Statement<F>) -> bool {
        if !statement.num_variables() == self.0.mv_parameters.num_variables {
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
        mut statement: Statement<F>,
        witness: Witness<F, MerkleConfig>,
    ) -> ProofResult<WhirProof<MerkleConfig, F>>
    where
        Merlin: FieldWriter<F> + ByteChallenges + PoWChallenge + DigestWriter<MerkleConfig>,
    {
        assert!(
            self.validate_parameters()
                && self.validate_statement(&statement)
                && self.validate_witness(&witness)
        );

        // Convert witness ood_points into constraints
        let new_constraints = witness
            .ood_points
            .into_iter()
            .zip(witness.ood_answers)
            .map(|(point, evaluation)| {
                let weights = Weights::evaluation(MultilinearPoint::expand_from_univariate(
                    point,
                    self.0.mv_parameters.num_variables,
                ));
                (weights, evaluation)
            })
            .collect();

        statement.add_constraints_in_front(new_constraints);
        let mut sumcheck_prover = None;
        let folding_randomness = if self.0.initial_statement {
            // If there is initial statement, then we run the sum-check for
            // this initial statement.
            let [combination_randomness_gen] = merlin.challenge_scalars()?;
            sumcheck_prover = {
                let sumcheck = SumcheckSingle::new(
                    witness.polynomial.clone(),
                    &statement,
                    combination_randomness_gen,
                );
                Some(sumcheck)
            };

            sumcheck_prover
                .as_mut()
                .unwrap()
                .compute_sumcheck_polynomials::<PowStrategy, Merlin>(
                    merlin,
                    self.0.folding_factor.at_round(0),
                    self.0.starting_folding_pow_bits,
                )?
        } else {
            // If there is no initial statement, there is no need to run the
            // initial rounds of the sum-check, and the verifier directly sends
            // the initial folding randomnesses.
            let mut folding_randomness = vec![F::ZERO; self.0.folding_factor.at_round(0)];
            merlin.fill_challenge_scalars(&mut folding_randomness)?;

            if self.0.starting_folding_pow_bits > 0. {
                merlin.challenge_pow::<PowStrategy>(self.0.starting_folding_pow_bits)?;
            }
            MultilinearPoint(folding_randomness)
        };
        let mut randomness_vec = vec![F::ZERO; self.0.mv_parameters.num_variables];
        let mut arr = folding_randomness.clone().0;
        arr.reverse();
        randomness_vec[..folding_randomness.0.len()].copy_from_slice(&arr);

        let round_state = RoundState {
            domain: self.0.starting_domain.clone(),
            round: 0,
            sumcheck_prover,
            folding_randomness,
            coefficients: witness.polynomial,
            prev_merkle: witness.merkle_tree,
            prev_merkle_answers: witness.merkle_leaves,
            merkle_proofs: vec![],
            randomness_vec,
            statement,
        };

        self.round(merlin, round_state)
    }

    #[allow(clippy::too_many_lines)]
    fn round<Merlin>(
        &self,
        merlin: &mut Merlin,
        mut round_state: RoundState<F, MerkleConfig>,
    ) -> ProofResult<WhirProof<MerkleConfig, F>>
    where
        Merlin: ByteChallenges + FieldWriter<F> + PoWChallenge + DigestWriter<MerkleConfig>,
    {
        // Fold the coefficients
        let folded_coefficients = round_state
            .coefficients
            .fold(&round_state.folding_randomness);

        let num_variables = self.0.mv_parameters.num_variables
            - self.0.folding_factor.total_number(round_state.round);
        // num_variables should match the folded_coefficients here.
        assert_eq!(num_variables, folded_coefficients.num_variables());

        // Base case
        if round_state.round == self.0.n_rounds() {
            return self.final_round(merlin, round_state, &folded_coefficients);
        }

        let round_params = &self.0.round_parameters[round_state.round];

        // Fold the coefficients, and compute fft of polynomial (and commit)
        let new_domain = round_state.domain.scale(2);
        let expansion = new_domain.size() / folded_coefficients.num_coeffs();
        let mut evals = expand_from_coeff(folded_coefficients.coeffs(), expansion);
        transform_evaluations(
            &mut evals,
            self.0.fold_optimisation,
            new_domain.backing_domain.group_gen(),
            new_domain.backing_domain.group_gen_inv(),
            self.0.folding_factor.at_round(round_state.round + 1),
        );

        #[cfg(not(feature = "parallel"))]
        let leafs_iter =
            evals.chunks_exact(1 << self.0.folding_factor.at_round(round_state.round + 1));
        #[cfg(feature = "parallel")]
        let leafs_iter =
            evals.par_chunks_exact(1 << self.0.folding_factor.at_round(round_state.round + 1));
        let merkle_tree = MerkleTree::new(
            &self.0.leaf_hash_params,
            &self.0.two_to_one_params,
            leafs_iter,
        )
        .unwrap();

        let root = merkle_tree.root();
        merlin.add_digest(root)?;

        // Handle OOD (Out-Of-Domain) samples
        let (ood_points, ood_answers) =
            sample_ood_points(merlin, round_params.ood_samples, num_variables, |point| {
                folded_coefficients.evaluate(point)
            })?;

        // STIR queries
        let stir_challenges_indexes = get_challenge_stir_queries(
            round_state.domain.size(), // Current domain size *before* folding
            self.0.folding_factor.at_round(round_state.round), // Current fold factor
            round_params.num_queries,
            merlin,
        )?;
        // Compute the generator of the folded domain, in the extension field
        let domain_scaled_gen = round_state
            .domain
            .backing_domain
            .element(1 << self.0.folding_factor.at_round(round_state.round));
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
        let fold_size = 1 << self.0.folding_factor.at_round(round_state.round);
        let answers: Vec<_> = stir_challenges_indexes
            .iter()
            .map(|i| round_state.prev_merkle_answers[i * fold_size..(i + 1) * fold_size].to_vec())
            .collect();
        // Evaluate answers in the folding randomness.
        let mut stir_evaluations = ood_answers;
        self.0.fold_optimisation.stir_evaluations_prover(
            &round_state,
            &stir_challenges_indexes,
            &answers,
            self.0.folding_factor,
            &mut stir_evaluations,
        );
        round_state.merkle_proofs.push((merkle_proof, answers));

        // PoW
        if round_params.pow_bits > 0. {
            merlin.challenge_pow::<PowStrategy>(round_params.pow_bits)?;
        }

        // Randomness for combination
        let [combination_randomness_gen] = merlin.challenge_scalars()?;
        let combination_randomness =
            expand_randomness(combination_randomness_gen, stir_challenges.len());

        #[allow(clippy::map_unwrap_or)]
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
                let mut statement: Statement<F> =
                    Statement::<F>::new(folded_coefficients.num_variables());

                for (point, eval) in stir_challenges.into_iter().zip(stir_evaluations) {
                    let weights = Weights::evaluation(point.clone());
                    statement.add_constraint(weights, eval);
                }
                SumcheckSingle::new(
                    folded_coefficients.clone(),
                    &statement,
                    combination_randomness[1],
                )
            });

        let folding_randomness = sumcheck_prover
            .compute_sumcheck_polynomials::<PowStrategy, Merlin>(
                merlin,
                self.0.folding_factor.at_round(round_state.round + 1),
                round_params.folding_pow_bits,
            )?;

        let start_idx = self.0.folding_factor.total_number(round_state.round);
        let mut arr = folding_randomness.clone().0;
        arr.reverse();

        round_state.randomness_vec[start_idx..start_idx + folding_randomness.0.len()]
            .copy_from_slice(&arr);

        let round_state = RoundState {
            round: round_state.round + 1,
            domain: new_domain,
            sumcheck_prover: Some(sumcheck_prover),
            folding_randomness,
            coefficients: folded_coefficients, // TODO: Is this redundant with `sumcheck_prover.coeff` ?
            prev_merkle: merkle_tree,
            prev_merkle_answers: evals,
            merkle_proofs: round_state.merkle_proofs,
            randomness_vec: round_state.randomness_vec.clone(),
            statement: round_state.statement,
        };

        self.round(merlin, round_state)
    }

    fn final_round<Merlin>(
        &self,
        merlin: &mut Merlin,
        mut round_state: RoundState<F, MerkleConfig>,
        folded_coefficients: &CoefficientList<F>,
    ) -> ProofResult<WhirProof<MerkleConfig, F>>
    where
        Merlin: ByteChallenges + FieldWriter<F> + PoWChallenge + DigestWriter<MerkleConfig>,
    {
        // Directly send coefficients of the polynomial to the verifier.
        merlin.add_scalars(folded_coefficients.coeffs())?;

        // Final verifier queries and answers. The indices are over the
        // *folded* domain.
        let final_challenge_indexes = get_challenge_stir_queries(
            // The size of the *original* domain before folding
            round_state.domain.size(),
            // The folding factor we used to fold the previous polynomial
            self.0.folding_factor.at_round(round_state.round),
            self.0.final_queries,
            merlin,
        )?;

        let merkle_proof = round_state
            .prev_merkle
            .generate_multi_proof(final_challenge_indexes.clone())
            .unwrap();
        // Every query requires opening these many in the previous Merkle tree
        let fold_size = 1 << self.0.folding_factor.at_round(round_state.round);
        let answers = final_challenge_indexes
            .into_iter()
            .map(|i| round_state.prev_merkle_answers[i * fold_size..(i + 1) * fold_size].to_vec())
            .collect();
        round_state.merkle_proofs.push((merkle_proof, answers));

        // PoW
        if self.0.final_pow_bits > 0. {
            merlin.challenge_pow::<PowStrategy>(self.0.final_pow_bits)?;
        }

        // Final sumcheck
        if self.0.final_sumcheck_rounds > 0 {
            let final_folding_randomness = round_state
                .sumcheck_prover
                .unwrap_or_else(|| {
                    SumcheckSingle::new(folded_coefficients.clone(), &round_state.statement, F::ONE)
                })
                .compute_sumcheck_polynomials::<PowStrategy, Merlin>(
                    merlin,
                    self.0.final_sumcheck_rounds,
                    self.0.final_folding_pow_bits,
                )?;
            let start_idx = self.0.folding_factor.total_number(round_state.round);
            let mut arr = final_folding_randomness.clone().0;
            arr.reverse();
            round_state.randomness_vec[start_idx..start_idx + final_folding_randomness.0.len()]
                .copy_from_slice(&arr);
        }

        let mut randomness_vec_rev = round_state.randomness_vec.clone();
        randomness_vec_rev.reverse();

        let statement_values_at_random_point = round_state
            .statement
            .constraints
            .iter()
            .filter_map(|(weights, _)| {
                if let Weights::Linear { weight } = weights {
                    Some(weight.eval_extension(&MultilinearPoint(randomness_vec_rev.clone())))
                } else {
                    None
                }
            })
            .collect();

        Ok(WhirProof {
            merkle_paths: round_state.merkle_proofs,
            statement_values_at_random_point,
        })
    }
}

pub(crate) struct RoundState<F, MerkleConfig>
where
    F: FftField,
    MerkleConfig: Config,
{
    pub(crate) round: usize,
    pub(crate) domain: Domain<F>,
    pub(crate) sumcheck_prover: Option<SumcheckSingle<F>>,
    pub(crate) folding_randomness: MultilinearPoint<F>,
    pub(crate) coefficients: CoefficientList<F>,
    pub(crate) prev_merkle: MerkleTree<MerkleConfig>,
    pub(crate) prev_merkle_answers: Vec<F>,
    pub(crate) merkle_proofs: Vec<(MultiPath<MerkleConfig>, Vec<Vec<F>>)>,
    pub(crate) randomness_vec: Vec<F>,
    pub(crate) statement: Statement<F>,
}
