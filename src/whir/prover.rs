use super::{committer::Witness, parameters::WhirConfig, Statement, WhirProof};
use crate::{
    domain::Domain,
    fs_utils::EVMFs,
    ntt::expand_from_coeff,
    parameters::FoldType,
    poly_utils::{
        coeffs::CoefficientList,
        fold::{compute_fold, restructure_evaluations},
        MultilinearPoint,
    },
    sumcheck::prover_not_skipping::SumcheckProverNotSkipping,
    utils::{self, expand_randomness},
};
use ark_crypto_primitives::merkle_tree::{Config, MerkleTree, MultiPath};
use ark_ff::FftField;
use ark_ff::Field;
use ark_poly::EvaluationDomain;
use ark_serialize::SerializationError;
use nimue::{
    plugins::{
        ark::{FieldChallenges, FieldWriter},
        pow::{self, PoWChallenge},
    },
    ByteChallenges, ByteWriter, Merlin, ProofResult,
};
use num_bigint::BigUint;
use rand::{Rng, SeedableRng};

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
    MerkleConfig::InnerDigest: AsRef<[u8]>,
    PowStrategy: pow::PowStrategy,
{
    fn validate_parameters(&self) -> bool {
        self.0.mv_parameters.num_variables
            == (self.0.n_rounds() + 1) * self.0.folding_factor + self.0.final_sumcheck_rounds
    }

    fn validate_statement(&self, statement: &Statement<F>) -> bool {
        statement
            .points
            .iter()
            .all(|point| point.0.len() == self.0.mv_parameters.num_variables)
    }

    fn validate_witness(&self, witness: &Witness<F, MerkleConfig>) -> bool {
        witness.polynomial.num_variables() == self.0.mv_parameters.num_variables
    }

    pub fn evm_prove(
        &self,
        evmfs: &mut EVMFs<F>,
        statement: Statement<F>,
        witness: Witness<F, MerkleConfig>,
    ) -> ProofResult<WhirProof<MerkleConfig, F>>
    where
        Merlin: FieldChallenges<F> + ByteWriter,
    {
        assert!(self.validate_parameters());
        assert!(self.validate_statement(&statement));
        assert!(self.validate_witness(&witness));
        let [combination_randomness_gen] = [evmfs.squeeze_scalars(1)[0]];
        // let [combination_randomness_gen] = merlin.challenge_scalars()?;

        let initial_claims: Vec<_> = witness
            .ood_points
            .into_iter()
            .map(|ood_point| {
                MultilinearPoint::expand_from_univariate(
                    ood_point,
                    self.0.mv_parameters.num_variables,
                )
            })
            .chain(statement.points)
            .collect();
        let combination_randomness =
            expand_randomness(combination_randomness_gen, initial_claims.len());
        let initial_answers: Vec<_> = witness
            .ood_answers
            .into_iter()
            .chain(statement.evaluations)
            .collect();

        let mut sumcheck_prover = SumcheckProverNotSkipping::new(
            witness.polynomial.clone(),
            &initial_claims,
            &combination_randomness,
            &initial_answers,
        );

        let folding_randomness = sumcheck_prover.evm_compute_sumcheck_polynomials::<PowStrategy>(
            evmfs,
            self.0.folding_factor,
            self.0.starting_folding_pow_bits,
        )?;
        //let folding_randomness = sumcheck_prover.compute_sumcheck_polynomials::<PowStrategy>(
        //    merlin,
        //    self.0.folding_factor,
        //    self.0.starting_folding_pow_bits,
        //)?;

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

        self.evm_round(evmfs, round_state)
    }
    pub fn prove(
        &self,
        merlin: &mut Merlin,
        statement: Statement<F>,
        witness: Witness<F, MerkleConfig>,
    ) -> ProofResult<WhirProof<MerkleConfig, F>>
    where
        Merlin: FieldChallenges<F> + ByteWriter,
    {
        assert!(self.validate_parameters());
        assert!(self.validate_statement(&statement));
        assert!(self.validate_witness(&witness));

        let [combination_randomness_gen] = merlin.challenge_scalars()?;
        let initial_claims: Vec<_> = witness
            .ood_points
            .into_iter()
            .map(|ood_point| {
                MultilinearPoint::expand_from_univariate(
                    ood_point,
                    self.0.mv_parameters.num_variables,
                )
            })
            .chain(statement.points)
            .collect();
        let combination_randomness =
            expand_randomness(combination_randomness_gen, initial_claims.len());
        let initial_answers: Vec<_> = witness
            .ood_answers
            .into_iter()
            .chain(statement.evaluations)
            .collect();

        let mut sumcheck_prover = SumcheckProverNotSkipping::new(
            witness.polynomial.clone(),
            &initial_claims,
            &combination_randomness,
            &initial_answers,
        );

        let folding_randomness = sumcheck_prover.compute_sumcheck_polynomials::<PowStrategy>(
            merlin,
            self.0.folding_factor,
            self.0.starting_folding_pow_bits,
        )?;

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

        self.round(merlin, round_state)
    }

    fn evm_round(
        &self,
        evmfs: &mut EVMFs<F>,
        mut round_state: RoundState<F, MerkleConfig>,
    ) -> ProofResult<WhirProof<MerkleConfig, F>> {
        // Fold the coefficients
        let folded_coefficients = round_state
            .coefficients
            .fold(&round_state.folding_randomness);

        let num_variables =
            self.0.mv_parameters.num_variables - (round_state.round + 1) * self.0.folding_factor;

        // Base case
        if round_state.round == self.0.n_rounds() {
            // Coefficients of the polynomial
            evmfs.absorb_scalars(folded_coefficients.coeffs())?;
            // merlin.add_scalars(folded_coefficients.coeffs())?;

            // Final verifier queries and answers
            // let queries_seed = evmfs.squeeze_bytes(32)[0];

            // let mut final_gen_2 = evmfs.absorb_bytes(&queries_seed);
            let final_gen = evmfs.squeeze_scalars(self.0.final_queries);
            let max_target = BigUint::from(round_state.domain.folded_size(self.0.folding_factor));
            let final_challenge_indexes = utils::dedup(
                final_gen
                    .into_iter()
                    .map(|idx| to_range(idx, &max_target))
                    .collect::<Vec<usize>>(),
            );
            let merkle_proof = round_state
                .prev_merkle
                .generate_multi_proof(final_challenge_indexes.clone())
                .unwrap();

            // let mut queries_seed = [0u8; 32];
            // merlin.fill_challenge_bytes(&mut queries_seed)?;
            //let mut final_gen = rand_chacha::ChaCha20Rng::from_seed(queries_seed);
            //let final_challenge_indexes = utils::dedup((0..self.0.final_queries).map(|_| {
            //    final_gen.gen_range(0..round_state.domain.folded_size(self.0.folding_factor))
            //}));

            //let merkle_proof = round_state
            //    .prev_merkle
            //    .generate_multi_proof(final_challenge_indexes.clone())
            //    .unwrap();
            let fold_size = 1 << self.0.folding_factor;
            let answers = final_challenge_indexes
                .into_iter()
                .map(|i| {
                    round_state.prev_merkle_answers[i * fold_size..(i + 1) * fold_size].to_vec()
                })
                .collect();
            round_state.merkle_proofs.push((merkle_proof, answers));

            if self.0.final_pow_bits > 0. {
                evmfs.challenge_pow::<PowStrategy>(self.0.final_pow_bits)?;
            }

            // Final sumcheck
            round_state
                .sumcheck_prover
                .evm_compute_sumcheck_polynomials::<PowStrategy>(
                    evmfs,
                    self.0.final_sumcheck_rounds,
                    self.0.final_folding_pow_bits,
                )?;

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
        evmfs.absorb_bytes(root.as_ref());
        // merlin.add_bytes(root.as_ref())?;

        // OOD Samples
        let mut ood_points = Vec::with_capacity(round_params.ood_samples);
        // let mut ood_points = vec![F::ZERO; round_params.ood_samples];
        let mut ood_answers = Vec::with_capacity(round_params.ood_samples);
        if round_params.ood_samples > 0 {
            ood_points = evmfs.squeeze_scalars(round_params.ood_samples);
            // evmfs.squeeze_scalars()
            // merlin.fill_challenge_scalars(&mut ood_points)?;
            ood_answers.extend(ood_points.iter().map(|ood_point| {
                folded_coefficients.evaluate(&MultilinearPoint::expand_from_univariate(
                    *ood_point,
                    num_variables,
                ))
            }));
            evmfs.absorb_scalars(&ood_answers)?;
            // merlin.add_scalars(&ood_answers)?;
        }

        // STIR queries
        // let mut stir_queries_seed = evmfs.squeeze_bytes(32)[0];
        let stir_gen = evmfs.squeeze_scalars(round_params.num_queries);
        let max_target = BigUint::from(round_state.domain.folded_size(self.0.folding_factor));
        let stir_challenges_indexes = utils::dedup(
            stir_gen
                .into_iter()
                .map(|idx| to_range(idx, &max_target))
                .collect::<Vec<usize>>(),
        );

        // let mut stir_queries_seed = [0u8; 32];
        // merlin.fill_challenge_bytes(&mut stir_queries_seed)?;
        // let mut stir_gen = rand_chacha::ChaCha20Rng::from_seed(stir_queries_seed);
        // let stir_challenges_indexes =
        //     utils::dedup((0..round_params.num_queries).map(|_| {
        //         stir_gen.gen_range(0..round_state.domain.folded_size(self.0.folding_factor))
        //     }));

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

        if round_params.pow_bits > 0. {
            evmfs.challenge_pow::<PowStrategy>(round_params.pow_bits)?;
        }

        // Randomness for combination
        let [combination_randomness_gen] = [evmfs.squeeze_scalars(1)[0]];

        // let [combination_randomness_gen] = merlin.challenge_scalars()?;
        let combination_randomness =
            expand_randomness(combination_randomness_gen, stir_challenges.len());

        round_state.sumcheck_prover.add_new_equality(
            &stir_challenges,
            &combination_randomness,
            &stir_evaluations,
        );

        let folding_randomness = round_state
            .sumcheck_prover
            .evm_compute_sumcheck_polynomials::<PowStrategy>(
                evmfs,
                self.0.folding_factor,
                round_params.folding_pow_bits,
            )?;

        let round_state = RoundState {
            round: round_state.round + 1,
            domain: new_domain,
            sumcheck_prover: round_state.sumcheck_prover,
            folding_randomness,
            coefficients: folded_coefficients, // TODO: Is this redundant with `sumcheck_prover.coeff` ?
            prev_merkle: merkle_tree,
            prev_merkle_answers: folded_evals,
            merkle_proofs: round_state.merkle_proofs,
        };

        self.evm_round(evmfs, round_state)
    }

    fn round(
        &self,
        merlin: &mut Merlin,
        mut round_state: RoundState<F, MerkleConfig>,
    ) -> ProofResult<WhirProof<MerkleConfig, F>> {
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
            let mut queries_seed = [0u8; 32];
            merlin.fill_challenge_bytes(&mut queries_seed)?;
            let mut final_gen = rand_chacha::ChaCha20Rng::from_seed(queries_seed);
            let final_challenge_indexes = utils::dedup((0..self.0.final_queries).map(|_| {
                final_gen.gen_range(0..round_state.domain.folded_size(self.0.folding_factor))
            }));

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
            round_state
                .sumcheck_prover
                .compute_sumcheck_polynomials::<PowStrategy>(
                    merlin,
                    self.0.final_sumcheck_rounds,
                    self.0.final_folding_pow_bits,
                )?;

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
        merlin.add_bytes(root.as_ref())?;

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
        let mut stir_queries_seed = [0u8; 32];
        merlin.fill_challenge_bytes(&mut stir_queries_seed)?;
        let mut stir_gen = rand_chacha::ChaCha20Rng::from_seed(stir_queries_seed);
        let stir_challenges_indexes =
            utils::dedup((0..round_params.num_queries).map(|_| {
                stir_gen.gen_range(0..round_state.domain.folded_size(self.0.folding_factor))
            }));
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

        round_state.sumcheck_prover.add_new_equality(
            &stir_challenges,
            &combination_randomness,
            &stir_evaluations,
        );

        let folding_randomness = round_state
            .sumcheck_prover
            .compute_sumcheck_polynomials::<PowStrategy>(
                merlin,
                self.0.folding_factor,
                round_params.folding_pow_bits,
            )?;

        let round_state = RoundState {
            round: round_state.round + 1,
            domain: new_domain,
            sumcheck_prover: round_state.sumcheck_prover,
            folding_randomness,
            coefficients: folded_coefficients, // TODO: Is this redundant with `sumcheck_prover.coeff` ?
            prev_merkle: merkle_tree,
            prev_merkle_answers: folded_evals,
            merkle_proofs: round_state.merkle_proofs,
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
    sumcheck_prover: SumcheckProverNotSkipping<F>,
    folding_randomness: MultilinearPoint<F>,
    coefficients: CoefficientList<F>,
    prev_merkle: MerkleTree<MerkleConfig>,
    prev_merkle_answers: Vec<F>,
    merkle_proofs: Vec<(MultiPath<MerkleConfig>, Vec<Vec<F>>)>,
}

fn scalar_to_bytes(scalar: impl Field) -> Result<[u8; 32], SerializationError> {
    let mut bytes = [0_u8; 32];
    scalar.serialize_uncompressed(&mut bytes[..])?;
    Ok(bytes)
}

// WARNING: adhoc code that should not be trusted
pub fn to_range(element: impl Field, max_target: &BigUint) -> usize {
    let element = BigUint::from_bytes_le(&scalar_to_bytes(element).unwrap());
    let modulo = element % max_target;
    usize::from_str_radix(&modulo.to_string(), 10).unwrap()
}

#[cfg(test)]
pub mod tests {
    use crate::crypto::fields::FieldBn256;
    use crate::fs_utils::EVMFs;
    use crate::utils;
    use num_bigint::BigUint;

    use super::to_range;
    type F = FieldBn256;

    #[test]
    fn test_to_range_and_sort() {
        // used in the context of sorting out indexes when doing merkle tree queries
        let mut evmfs = EVMFs::<F>::new();
        evmfs.absorb_bytes(&[4_u8, 2_u8]);
        let squeezed = evmfs.squeeze_scalars(5);
        let max_range = BigUint::from(15_u8);
        let ranged_squeezed = squeezed
            .into_iter()
            .map(|x| to_range(x, &max_range))
            .collect::<Vec<usize>>();
        let ordered_deduped_squeezed = utils::dedup(ranged_squeezed);
        assert_eq!(ordered_deduped_squeezed, vec![0, 3, 11, 12]);
    }
}
