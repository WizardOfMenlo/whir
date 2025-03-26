use ark_crypto_primitives::merkle_tree::{Config, MerkleTree, MultiPath};
use ark_ff::FftField;
use ark_poly::EvaluationDomain;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use spongefish::{
    codecs::arkworks_algebra::{FieldToUnitSerialize, UnitToField},
    ProofResult, UnitToBytes,
};
use spongefish_pow::{self, PoWChallenge};

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
    whir::{
        parameters::RoundConfig,
        utils::{get_challenge_stir_queries, sample_ood_points, DigestToUnitSerialize},
    },
};

pub struct Prover<F, MerkleConfig, PowStrategy>(pub WhirConfig<F, MerkleConfig, PowStrategy>)
where
    F: FftField,
    MerkleConfig: Config;

impl<F, MerkleConfig, PowStrategy> Prover<F, MerkleConfig, PowStrategy>
where
    F: FftField,
    MerkleConfig: Config<Leaf = [F]>,
    PowStrategy: spongefish_pow::PowStrategy,
{
    fn validate_parameters(&self) -> bool {
        self.0.mv_parameters.num_variables
            == self.0.folding_factor.total_number(self.0.n_rounds()) + self.0.final_sumcheck_rounds
    }

    fn validate_statement(&self, statement: &Statement<F>) -> bool {
        statement.num_variables() == self.0.mv_parameters.num_variables
            && (self.0.initial_statement || statement.constraints.is_empty())
    }

    fn validate_witness(&self, witness: &Witness<F, MerkleConfig>) -> bool {
        assert_eq!(witness.ood_points.len(), witness.ood_answers.len());
        if !self.0.initial_statement {
            assert!(witness.ood_points.is_empty());
        }
        witness.polynomial.num_variables() == self.0.mv_parameters.num_variables
    }

    pub fn prove<ProverState>(
        &self,
        prover_state: &mut ProverState,
        mut statement: Statement<F>,
        witness: Witness<F, MerkleConfig>,
    ) -> ProofResult<WhirProof<MerkleConfig, F>>
    where
        ProverState: UnitToField<F>
            + FieldToUnitSerialize<F>
            + UnitToBytes
            + PoWChallenge
            + DigestToUnitSerialize<MerkleConfig>,
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
            let [combination_randomness_gen] = prover_state.challenge_scalars()?;

            // Create the sumcheck prover
            let mut sumcheck = SumcheckSingle::new(
                witness.polynomial.clone(),
                &statement,
                combination_randomness_gen,
            );

            let folding_randomness = sumcheck.compute_sumcheck_polynomials::<PowStrategy, _>(
                prover_state,
                self.0.folding_factor.at_round(0),
                self.0.starting_folding_pow_bits,
            )?;

            sumcheck_prover = Some(sumcheck);
            folding_randomness
        } else {
            // If there is no initial statement, there is no need to run the
            // initial rounds of the sum-check, and the verifier directly sends
            // the initial folding randomnesses.
            let mut folding_randomness = vec![F::ZERO; self.0.folding_factor.at_round(0)];
            prover_state.fill_challenge_scalars(&mut folding_randomness)?;

            if self.0.starting_folding_pow_bits > 0. {
                prover_state.challenge_pow::<PowStrategy>(self.0.starting_folding_pow_bits)?;
            }
            MultilinearPoint(folding_randomness)
        };
        let mut randomness_vec = Vec::with_capacity(self.0.mv_parameters.num_variables);
        randomness_vec.extend(folding_randomness.0.iter().rev().copied());
        randomness_vec.resize(self.0.mv_parameters.num_variables, F::ZERO);

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

        self.round(prover_state, round_state)
    }

    #[allow(clippy::too_many_lines)]
    fn round<ProverState>(
        &self,
        prover_state: &mut ProverState,
        mut round_state: RoundState<F, MerkleConfig>,
    ) -> ProofResult<WhirProof<MerkleConfig, F>>
    where
        ProverState: UnitToField<F>
            + UnitToBytes
            + FieldToUnitSerialize<F>
            + PoWChallenge
            + DigestToUnitSerialize<MerkleConfig>,
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
            return self.final_round(prover_state, round_state, &folded_coefficients);
        }

        let round_params = &self.0.round_parameters[round_state.round];

        // Compute the folding factors for later use
        let folding_factor = self.0.folding_factor.at_round(round_state.round);
        let folding_factor_next = self.0.folding_factor.at_round(round_state.round + 1);

        // Fold the coefficients, and compute fft of polynomial (and commit)
        let new_domain = round_state.domain.scale(2);
        let expansion = new_domain.size() / folded_coefficients.num_coeffs();
        let mut evals = expand_from_coeff(folded_coefficients.coeffs(), expansion);
        transform_evaluations(
            &mut evals,
            self.0.fold_optimisation,
            new_domain.backing_domain.group_gen(),
            new_domain.backing_domain.group_gen_inv(),
            folding_factor_next,
        );

        #[cfg(not(feature = "parallel"))]
        let leafs_iter = evals.chunks_exact(1 << folding_factor_next);
        #[cfg(feature = "parallel")]
        let leafs_iter = evals.par_chunks_exact(1 << folding_factor_next);
        let merkle_tree = MerkleTree::new(
            &self.0.leaf_hash_params,
            &self.0.two_to_one_params,
            leafs_iter,
        )
        .unwrap();

        let root = merkle_tree.root();
        prover_state.add_digest(root)?;

        // Handle OOD (Out-Of-Domain) samples
        let (ood_points, ood_answers) = sample_ood_points(
            prover_state,
            round_params.ood_samples,
            num_variables,
            |point| folded_coefficients.evaluate(point),
        )?;

        // STIR Queries
        let (stir_challenges, stir_challenges_indexes) = self.compute_stir_queries(
            prover_state,
            &round_state,
            num_variables,
            round_params,
            ood_points,
        )?;

        let merkle_proof = round_state
            .prev_merkle
            .generate_multi_proof(stir_challenges_indexes.clone())
            .unwrap();
        let fold_size = 1 << folding_factor;
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
            prover_state.challenge_pow::<PowStrategy>(round_params.pow_bits)?;
        }

        // Randomness for combination
        let [combination_randomness_gen] = prover_state.challenge_scalars()?;
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
                let mut statement = Statement::new(folded_coefficients.num_variables());

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

        let folding_randomness = sumcheck_prover.compute_sumcheck_polynomials::<PowStrategy, _>(
            prover_state,
            folding_factor_next,
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

        self.round(prover_state, round_state)
    }

    fn final_round<ProverState>(
        &self,
        prover_state: &mut ProverState,
        mut round_state: RoundState<F, MerkleConfig>,
        folded_coefficients: &CoefficientList<F>,
    ) -> ProofResult<WhirProof<MerkleConfig, F>>
    where
        ProverState: UnitToField<F> + UnitToBytes + FieldToUnitSerialize<F> + PoWChallenge,
    {
        // Directly send coefficients of the polynomial to the verifier.
        prover_state.add_scalars(folded_coefficients.coeffs())?;

        // Precompute the folding factors for later use
        let folding_factor = self.0.folding_factor.at_round(round_state.round);

        // Final verifier queries and answers. The indices are over the
        // *folded* domain.
        let final_challenge_indexes = get_challenge_stir_queries(
            // The size of the *original* domain before folding
            round_state.domain.size(),
            // The folding factor we used to fold the previous polynomial
            folding_factor,
            self.0.final_queries,
            prover_state,
        )?;

        let merkle_proof = round_state
            .prev_merkle
            .generate_multi_proof(final_challenge_indexes.clone())
            .unwrap();
        // Every query requires opening these many in the previous Merkle tree
        let fold_size = 1 << folding_factor;
        let answers = final_challenge_indexes
            .into_iter()
            .map(|i| round_state.prev_merkle_answers[i * fold_size..(i + 1) * fold_size].to_vec())
            .collect();
        round_state.merkle_proofs.push((merkle_proof, answers));

        // PoW
        if self.0.final_pow_bits > 0. {
            prover_state.challenge_pow::<PowStrategy>(self.0.final_pow_bits)?;
        }

        // Final sumcheck
        if self.0.final_sumcheck_rounds > 0 {
            let final_folding_randomness = round_state
                .sumcheck_prover
                .unwrap_or_else(|| {
                    SumcheckSingle::new(folded_coefficients.clone(), &round_state.statement, F::ONE)
                })
                .compute_sumcheck_polynomials::<PowStrategy, _>(
                    prover_state,
                    self.0.final_sumcheck_rounds,
                    self.0.final_folding_pow_bits,
                )?;
            let start_idx = self.0.folding_factor.total_number(round_state.round);
            let rand_dst = &mut round_state.randomness_vec
                [start_idx..start_idx + final_folding_randomness.0.len()];

            for (dst, src) in rand_dst
                .iter_mut()
                .zip(final_folding_randomness.0.iter().rev())
            {
                *dst = *src;
            }
        }

        let mut randomness_vec_rev = round_state.randomness_vec;
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

    fn compute_stir_queries<ProverState>(
        &self,
        prover_state: &mut ProverState,
        round_state: &RoundState<F, MerkleConfig>,
        num_variables: usize,
        round_params: &RoundConfig,
        ood_points: Vec<F>,
    ) -> ProofResult<(Vec<MultilinearPoint<F>>, Vec<usize>)>
    where
        ProverState: UnitToBytes,
    {
        let stir_challenges_indexes = get_challenge_stir_queries(
            round_state.domain.size(),
            self.0.folding_factor.at_round(round_state.round),
            round_params.num_queries,
            prover_state,
        )?;

        // Compute the generator of the folded domain, in the extension field
        let domain_scaled_gen = round_state
            .domain
            .backing_domain
            .element(1 << self.0.folding_factor.at_round(round_state.round));
        let stir_challenges = ood_points
            .into_iter()
            .chain(
                stir_challenges_indexes
                    .iter()
                    .map(|i| domain_scaled_gen.pow([*i as u64])),
            )
            .map(|univariate| MultilinearPoint::expand_from_univariate(univariate, num_variables))
            .collect();

        Ok((stir_challenges, stir_challenges_indexes))
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
