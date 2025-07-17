use ark_crypto_primitives::merkle_tree::{Config, LeafParam, MerkleTree, MultiPath, TwoToOneParam};
use ark_ff::{FftField, Field};
use ark_poly::EvaluationDomain;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use spongefish::{
    codecs::arkworks_algebra::{FieldToUnitSerialize, UnitToField},
    ProofResult, UnitToBytes,
};
use spongefish_pow::{self, PoWChallenge};
#[cfg(feature = "tracing")]
use tracing::{instrument, span, Level};

use super::{
    committer::{BatchingData, Witness},
    parameters::WhirConfig,
    statement::{Statement, Weights},
    utils::HintSerialize,
};
use crate::{
    domain::Domain,
    ntt::interleaved_rs_encode,
    poly_utils::{coeffs::CoefficientList, multilinear::MultilinearPoint},
    sumcheck::SumcheckSingle,
    utils::expand_randomness,
    whir::{
        committer::reader::ParsedCommitment,
        parameters::RoundConfig,
        utils::{
            fma_stir_queries, get_challenge_stir_queries, sample_ood_points, validate_stir_queries,
            DigestToUnitSerialize,
        },
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
    pub(crate) fn validate_parameters(&self) -> bool {
        self.0.mv_parameters.num_variables
            == self.0.folding_factor.total_number(self.0.n_rounds()) + self.0.final_sumcheck_rounds
    }

    pub(crate) fn validate_statement(&self, statement: &Statement<F>) -> bool {
        statement.num_variables() == self.0.mv_parameters.num_variables
            && (self.0.initial_statement || statement.constraints.is_empty())
    }

    pub(crate) fn validate_witness(&self, witness: &Witness<F, MerkleConfig>) -> bool {
        assert_eq!(witness.ood_points.len(), witness.ood_answers.len());
        if !self.0.initial_statement {
            assert!(witness.ood_points.is_empty());
        }
        witness.polynomial.num_variables() == self.0.mv_parameters.num_variables
    }

    // Creates the merkle root paths if there are multiple oracles. Even in case of a single oracle, the first path is stored in the top level path and the rest are stored in other rounds
    #[cfg_attr(feature = "tracing", instrument(skip_all))]
    fn create_root_paths(
        &self,
        mut all_round_paths: Vec<(MultiPath<MerkleConfig>, Vec<Vec<F>>)>,
        batching_data: &[BatchingData<F, MerkleConfig>],
        _batching_randomness: &F, // Only used in debug builds
    ) -> ProofResult<(
        Vec<(MultiPath<MerkleConfig>, Vec<Vec<F>>)>, // first round paths
        Vec<(MultiPath<MerkleConfig>, Vec<Vec<F>>)>, // rest of the paths
    )> {
        assert!(
            all_round_paths.len() > 0,
            "There must be at least one round of merkle path"
        );

        let fold_sz = 1 << self.0.folding_factor.at_round(0);
        let rest = all_round_paths.split_off(1); // Splits vector into two parts efficiently

        if batching_data.is_empty() {
            return Ok((all_round_paths, rest));
        }

        let (batched_mt, batched_stir) = all_round_paths.into_iter().next().unwrap();

        let mut first_round_paths: Vec<(MultiPath<MerkleConfig>, Vec<Vec<F>>)> =
            Vec::with_capacity(batching_data.len());

        let stir_indexes = batched_mt.leaf_indexes.clone();

        for BatchingData {
            merkle_tree,
            merkle_leaves,
            ..
        } in batching_data
        {
            let paths = merkle_tree
                .generate_multi_proof(stir_indexes.clone())
                .map_err(|_| spongefish::ProofError::InvalidProof)?;

            let answers: Vec<_> = stir_indexes
                .iter()
                .map(|i| merkle_leaves[i * fold_sz..(i + 1) * fold_sz].to_vec())
                .collect();

            first_round_paths.push((paths, answers));
        }

        debug_assert!(
            validate_stir_queries(_batching_randomness, &batched_stir, &first_round_paths,),
            "The computed value of STIR queries must match the value in the leaf nodes"
        );

        Ok((first_round_paths, rest))
    }

    /// Proves that the commitment satisfies constraints in `statement`.
    ///
    /// When called without any constraints it only perfoms a low-degree test.
    /// Returns the constraint evaluation point and values of deferred constraints.
    #[cfg_attr(feature = "tracing", instrument(skip_all))]
    pub fn prove<ProverState>(
        &self,
        prover_state: &mut ProverState,
        mut statement: Statement<F>,
        witness: Witness<F, MerkleConfig>,
    ) -> ProofResult<(MultilinearPoint<F>, Vec<F>)>
    where
        ProverState: UnitToField<F>
            + FieldToUnitSerialize<F>
            + UnitToBytes
            + PoWChallenge
            + DigestToUnitSerialize<MerkleConfig>
            + HintSerialize,
    {
        assert!(self.validate_parameters());
        assert!(self.validate_statement(&statement));
        assert!(self.validate_witness(&witness));

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

        let mut round_state = RoundState {
            domain: self.0.starting_domain.clone(),
            round: 0,
            sumcheck_prover,
            folding_randomness,
            coefficients: witness.polynomial,
            prev_merkle: witness.merkle_tree,
            prev_merkle_answers: witness.merkle_leaves,
            randomness_vec,
            statement,
        };

        // Run WHIR rounds
        for _round in 0..=self.0.n_rounds() {
            self.round(prover_state, &mut round_state)?;
        }

        // Hints for deferred constraints
        let constraint_eval =
            MultilinearPoint(round_state.randomness_vec.iter().copied().rev().collect());
        let deferred = round_state
            .statement
            .constraints
            .iter()
            .filter(|constraint| constraint.defer_evaluation)
            .map(|constraint| constraint.weights.compute(&constraint_eval))
            .collect();

        let (round0_merkle_paths, rest_paths) = self.create_root_paths(
            round_state.merkle_proofs,
            &witness.batching_data,
            &witness.batching_randomness,
        )?;

        prover_state.hint::<Vec<F>>(&deferred)?;

        // Ok(WhirProof {
        //     round0_merkle_paths,
        //     merkle_paths: rest_paths,
        //     statement_values_at_random_point,
        // })
        Ok((constraint_eval, deferred))
    }

    #[allow(clippy::too_many_lines)]
    #[cfg_attr(feature = "tracing", instrument(skip_all, fields(size = round_state.coefficients.num_coeffs())))]
    fn round<ProverState>(
        &self,
        prover_state: &mut ProverState,
        round_state: &mut RoundState<F, MerkleConfig>,
    ) -> ProofResult<()>
    where
        ProverState: UnitToField<F>
            + UnitToBytes
            + FieldToUnitSerialize<F>
            + PoWChallenge
            + DigestToUnitSerialize<MerkleConfig>
            + HintSerialize,
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
        let evals =
            interleaved_rs_encode(folded_coefficients.coeffs(), expansion, folding_factor_next);

        #[cfg(not(feature = "parallel"))]
        let leafs_iter = evals.chunks_exact(1 << folding_factor_next);
        #[cfg(feature = "parallel")]
        let leafs_iter = evals.par_chunks_exact(1 << folding_factor_next);
        let merkle_tree = {
            #[cfg(feature = "tracing")]
            let _span = span!(Level::INFO, "MerkleTree::new", size = leafs_iter.len()).entered();
            MerkleTree::new(
                &self.0.leaf_hash_params,
                &self.0.two_to_one_params,
                leafs_iter,
            )
            .unwrap()
        };

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
            round_state,
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

        prover_state.hint::<Vec<Vec<F>>>(&answers)?;
        prover_state.hint::<MultiPath<MerkleConfig>>(&merkle_proof)?;

        // Evaluate answers in the folding randomness.
        let mut stir_evaluations = ood_answers;
        stir_evaluations.extend(answers.iter().map(|answers| {
            CoefficientList::new(answers.clone()).evaluate(&round_state.folding_randomness)
        }));

        // PoW
        if round_params.pow_bits > 0. {
            #[cfg(feature = "tracing")]
            let _span = span!(
                Level::INFO,
                "challenge_pow",
                pow_bits = round_params.pow_bits
            )
            .entered();
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
                    let weights = Weights::evaluation(point);
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
        let dst_randomness =
            &mut round_state.randomness_vec[start_idx..][..folding_randomness.0.len()];

        for (dst, src) in dst_randomness
            .iter_mut()
            .zip(folding_randomness.0.iter().rev())
        {
            *dst = *src;
        }

        // Update round state
        round_state.round += 1;
        round_state.domain = new_domain;
        round_state.sumcheck_prover = Some(sumcheck_prover);
        round_state.folding_randomness = folding_randomness;
        round_state.coefficients = folded_coefficients;
        round_state.prev_merkle = merkle_tree;
        round_state.prev_merkle_answers = evals;

        Ok(())
    }

    #[cfg_attr(feature = "tracing", instrument(skip_all, fields(size = folded_coefficients.num_coeffs())))]
    fn final_round<ProverState>(
        &self,
        prover_state: &mut ProverState,
        round_state: &mut RoundState<F, MerkleConfig>,
        folded_coefficients: &CoefficientList<F>,
    ) -> ProofResult<()>
    where
        ProverState:
            UnitToField<F> + UnitToBytes + FieldToUnitSerialize<F> + PoWChallenge + HintSerialize,
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
            .collect::<Vec<_>>();

        prover_state.hint::<Vec<Vec<F>>>(&answers)?;
        prover_state.hint::<MultiPath<MerkleConfig>>(&merkle_proof)?;

        // PoW
        if self.0.final_pow_bits > 0. {
            #[cfg(feature = "tracing")]
            let _span = span!(
                Level::INFO,
                "challenge_pow",
                pow_bits = self.0.final_pow_bits
            )
            .entered();
            prover_state.challenge_pow::<PowStrategy>(self.0.final_pow_bits)?;
        }

        // Final sumcheck
        if self.0.final_sumcheck_rounds > 0 {
            let final_folding_randomness = round_state
                .sumcheck_prover
                .clone()
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

        Ok(())
    }

    fn compute_stir_queries<ProverState>(
        &self,
        prover_state: &mut ProverState,
        round_state: &RoundState<F, MerkleConfig>,
        num_variables: usize,
        round_params: &RoundConfig<F>,
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

impl<F, MerkleConfig> WhirProof<MerkleConfig, F>
where
    MerkleConfig: Config<Leaf = [F]>,
    F: Sized + Clone + CanonicalSerialize + CanonicalDeserialize + Field,
{
    pub(crate) fn round_data(
        &self,
        whir_round: usize,
        batching_randomness: F,
    ) -> (MultiPath<MerkleConfig>, Vec<Vec<F>>) {
        assert!(whir_round <= self.merkle_paths.len());
        assert!(
            self.round0_merkle_paths.len() > 0,
            "There must be at least on round of data"
        );

        if whir_round == 0 {
            let mut multiplier = batching_randomness;
            let (merkle_paths, mut result) = self.round0_merkle_paths.first().unwrap().clone();

            for (_, leaf) in &self.round0_merkle_paths[1..] {
                fma_stir_queries(multiplier, leaf.as_slice(), result.as_mut_slice());
                multiplier *= batching_randomness;
            }

            (merkle_paths, result)
        } else {
            self.merkle_paths[whir_round - 1].clone()
        }
    }
}

impl<F, MerkleConfig> WhirProof<MerkleConfig, F>
where
    MerkleConfig: Config<Leaf = [F]>,
    F: Sized + Clone + ark_serialize::CanonicalSerialize + ark_serialize::CanonicalDeserialize,
{
    pub(crate) fn validate_first_round(
        &self,
        commitment: &ParsedCommitment<F, MerkleConfig::InnerDigest>,
        stir_challenges_indexes: &[usize],
        leaf_hash: &LeafParam<MerkleConfig>,
        inner_node_hash: &TwoToOneParam<MerkleConfig>,
    ) -> bool {
        commitment.root.len() == self.round0_merkle_paths.len()
            && self
                .round0_merkle_paths
                .iter()
                .zip(commitment.root.iter())
                .all(|((path, answers), root_hash)| {
                    path.verify(
                        leaf_hash,
                        inner_node_hash,
                        root_hash,
                        answers.iter().map(|a| a.as_ref()),
                    )
                    .unwrap()
                        && path.leaf_indexes == *stir_challenges_indexes
                })
    }
}

/// Represents the prover state during a single round of the WHIR protocol.
///
/// Each WHIR round folds the polynomial, commits to the new evaluations,
/// responds to verifier queries, and updates internal randomness for the next step.
/// This struct tracks all data needed to perform that round, and passes it forward
/// across recursive iterations.
pub(crate) struct RoundState<F, MerkleConfig>
where
    F: FftField,
    MerkleConfig: Config,
{
    /// Index of the current WHIR round (0-based).
    ///
    /// Increases after each folding iteration.
    pub(crate) round: usize,

    /// Domain over which the current polynomial is evaluated.
    ///
    /// Grows with each round due to NTT expansion.
    pub(crate) domain: Domain<F>,

    /// Optional sumcheck prover used to enforce constraints.
    ///
    /// Present in rounds with non-empty constraint systems.
    pub(crate) sumcheck_prover: Option<SumcheckSingle<F>>,

    /// Folding randomness sampled by the verifier.
    ///
    /// Used to reduce the number of variables in the polynomial.
    pub(crate) folding_randomness: MultilinearPoint<F>,

    /// Current polynomial in coefficient form.
    ///
    /// Folded and evaluated to produce new commitments and Merkle trees.
    pub(crate) coefficients: CoefficientList<F>,

    /// Merkle tree commitment to the polynomial evaluations from the previous round.
    ///
    /// Used to prove query openings from the folded function.
    pub(crate) prev_merkle: MerkleTree<MerkleConfig>,

    /// Flat list of evaluations corresponding to `prev_merkle` leaves.
    ///
    /// Each folded function is evaluated on a domain and split into leaves.
    pub(crate) prev_merkle_answers: Vec<F>,

    /// Accumulator for all folding randomness across rounds.
    ///
    /// Ordered with the most recent round’s randomness at the front.
    pub(crate) randomness_vec: Vec<F>,

    /// Constraint system being enforced in this round.
    ///
    /// May be updated during recursion as queries are folded and batched.
    pub(crate) statement: Statement<F>,
}
