use super::{
    committer::Witness,
    parameters::WhirConfig,
    statement::{Statement, Weights},
    WhirProof,
};
use crate::{
    domain::Domain,
    ntt::expand_from_coeff,
    parameters::FoldType,
    poly_utils::{
        coeffs::CoefficientList,
        fold::{compute_fold, restructure_evaluations},
        multilinear::MultilinearPoint,
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
        Merlin: FieldChallenges<F>
            + FieldWriter<F>
            + ByteChallenges
            + ByteWriter
            + PoWChallenge
            + DigestWriter<MerkleConfig>,
    {
        assert!(self.validate_parameters());
        assert!(self.validate_statement(&statement));
        assert!(self.validate_witness(&witness));

        let mut new_constraints = Vec::new();

        for (point, evaluation) in witness.ood_points.into_iter().zip(witness.ood_answers) {
            let weights: Weights<F> = Weights::evaluation(
                MultilinearPoint::expand_from_univariate(point, self.0.mv_parameters.num_variables),
            );
            new_constraints.push((weights, evaluation));
        }

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
            randomness_vec: randomness_vec.clone(),
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

        let num_variables = self.0.mv_parameters.num_variables
            - self.0.folding_factor.total_number(round_state.round);
        // num_variables should match the folded_coefficients here.
        assert_eq!(num_variables, folded_coefficients.num_variables());

        // Base case
        if round_state.round == self.0.n_rounds() {
            // Directly send coefficients of the polynomial to the verifier.
            merlin.add_scalars(folded_coefficients.coeffs())?;

            // Final verifier queries and answers. The indices are over the
            // *folded* domain.
            let final_challenge_indexes = get_challenge_stir_queries(
                round_state.domain.size(), // The size of the *original* domain before folding
                self.0.folding_factor.at_round(round_state.round), // The folding factor we used to fold the previous polynomial
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
                let final_folding_randomness = round_state
                    .sumcheck_prover
                    .unwrap_or_else(|| {
                        SumcheckSingle::new(
                            folded_coefficients.clone(),
                            &round_state.statement,
                            F::from(1),
                        )
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

            let mut statement_values_at_random_point = vec![];
            for (weights, _) in &round_state.statement.constraints {
                if let Weights::Linear { weight } = weights {
                    statement_values_at_random_point
                        .push(weight.eval_extension(&MultilinearPoint(randomness_vec_rev.clone())));
                }
            }
            return Ok(WhirProof {
                merkle_paths: round_state.merkle_proofs,
                statement_values_at_random_point,
            });
        }

        let round_params = &self.0.round_parameters[round_state.round];

        // Fold the coefficients, and compute fft of polynomial (and commit)
        let new_domain = round_state.domain.scale(2);
        let expansion = new_domain.size() / folded_coefficients.num_coeffs();
        let evals = expand_from_coeff(folded_coefficients.coeffs(), expansion);
        // Group the evaluations into leaves by the *next* round folding factor
        // TODO: `stack_evaluations` and `restructure_evaluations` are really in-place algorithms.
        // They also partially overlap and undo one another. We should merge them.
        let folded_evals = utils::stack_evaluations(
            evals,
            self.0.folding_factor.at_round(round_state.round + 1), // Next round fold factor
        );
        let folded_evals = restructure_evaluations(
            folded_evals,
            self.0.fold_optimisation,
            new_domain.backing_domain.group_gen(),
            new_domain.backing_domain.group_gen_inv(),
            self.0.folding_factor.at_round(round_state.round + 1),
        );

        #[cfg(not(feature = "parallel"))]
        let leafs_iter = folded_evals.chunks_exact(
            1 << self
                .0
                .folding_factor
                .get_folding_factor_of_round(round_state.round + 1),
        );
        #[cfg(feature = "parallel")]
        let leafs_iter = folded_evals
            .par_chunks_exact(1 << self.0.folding_factor.at_round(round_state.round + 1));
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
        let mut stir_evaluations = ood_answers.clone();
        match self.0.fold_optimisation {
            FoldType::Naive => {
                // See `Verifier::compute_folds_full`
                let domain_size = round_state.domain.backing_domain.size();
                let domain_gen = round_state.domain.backing_domain.element(1);
                let domain_gen_inv = domain_gen.inverse().unwrap();
                let coset_domain_size = 1 << self.0.folding_factor.at_round(round_state.round);
                // The domain (before folding) is split into cosets of size
                // `coset_domain_size` (which is just `fold_size`). Each coset
                // is generated by powers of `coset_generator` (which is just the
                // `fold_size`-root of unity) multiplied by a different
                // `coset_offset`.
                // For example, if `fold_size = 16`, and the domain size is N, then
                // the domain is (1, w, w^2, ..., w^(N-1)), the domain generator
                // is w, and the coset generator is w^(N/16).
                // The first coset is (1, w^(N/16), w^(2N/16), ..., w^(15N/16))
                // which is also a subgroup <w^(N/16)> itself (the coset_offset is 1).
                // The second coset would be w * <w^(N/16)>, the third coset would be
                // w^2 * <w^(N/16)>, and so on. Until w^(N/16-1) * <w^(N/16)>.
                let coset_generator_inv =
                    domain_gen_inv.pow([(domain_size / coset_domain_size) as u64]);
                stir_evaluations.extend(stir_challenges_indexes.iter().zip(&answers).map(
                    |(index, answers)| {
                        // The coset is w^index * <w_coset_generator>
                        //let _coset_offset = domain_gen.pow(&[*index as u64]);
                        let coset_offset_inv = domain_gen_inv.pow([*index as u64]);

                        // In the Naive mode, the oracle consists directly of the
                        // evaluations of f over the domain. We leverage an
                        // algorithm to compute the evaluations of the folded f
                        // at the corresponding point in folded domain (which is
                        // coset_offset^fold_size).
                        compute_fold(
                            answers,
                            &round_state.folding_randomness.0,
                            coset_offset_inv,
                            coset_generator_inv,
                            F::from(2).inverse().unwrap(),
                            self.0.folding_factor.at_round(round_state.round),
                        )
                    },
                ));
            }
            FoldType::ProverHelps => stir_evaluations.extend(answers.iter().map(|answers| {
                // In the ProverHelps mode, the oracle values have been linearly
                // transformed such that they are exactly the coefficients of the
                // multilinear polynomial whose evaluation at the folding randomness
                // is just the folding of f evaluated at the folded point.
                CoefficientList::new(answers.clone()).evaluate(&round_state.folding_randomness)
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
            prev_merkle_answers: folded_evals,
            merkle_proofs: round_state.merkle_proofs,
            randomness_vec: round_state.randomness_vec.clone(),
            statement: round_state.statement,
        };

        self.round(merlin, round_state)
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
    randomness_vec: Vec<F>,
    statement: Statement<F>,
}
