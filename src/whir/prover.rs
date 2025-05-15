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
    committer::Witness,
    parameters::WhirConfig,
    statement::{Statement, Weights},
    WhirProof,
};
use crate::{
    domain::Domain,
    ntt::expand_from_coeff,
    poly_utils::{
        coeffs::CoefficientList, evals::EvaluationsList, fold::transform_evaluations,
        multilinear::MultilinearPoint,
    },
    sumcheck::SumcheckSingle,
    utils::expand_randomness,
    whir::{
        committer::reader::ParsedCommitment,
        parameters::RoundConfig,
        stir_evaluations,
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
    pub fn prove<ProverState>(
        &self,
        prover_state: &mut ProverState,
        statement: Statement<F>,
        witness: &Witness<F, MerkleConfig>,
    ) -> ProofResult<WhirProof<MerkleConfig, F>>
    where
        ProverState: UnitToField<F>
            + FieldToUnitSerialize<F>
            + UnitToBytes
            + PoWChallenge
            + DigestToUnitSerialize<MerkleConfig>,
    {
        self.prove_many(prover_state, &[witness], vec![statement])
    }

    #[cfg_attr(feature = "tracing", instrument(skip_all))]
    pub fn prove_many<ProverState>(
        &self,
        prover_state: &mut ProverState,
        witnesses: &[&Witness<F, MerkleConfig>],
        mut statements: Vec<Statement<F>>,
    ) -> ProofResult<WhirProof<MerkleConfig, F>>
    where
        ProverState: UnitToField<F>
            + FieldToUnitSerialize<F>
            + UnitToBytes
            + PoWChallenge
            + DigestToUnitSerialize<MerkleConfig>,
    {
        assert_eq!(
            self.0.mv_parameters.num_variables,
            self.0.folding_factor.total_number(self.0.n_rounds()) + self.0.final_sumcheck_rounds
        );
        for witness in witnesses {
            assert_eq!(witness.ood_points.len(), self.0.committment_ood_samples);
            assert_eq!(witness.ood_answers.len(), self.0.committment_ood_samples);
            for poly in &witness.polynomials {
                assert_eq!(poly.num_variables(), self.0.mv_parameters.num_variables);
            }
        }
        let polynomials = witnesses
            .iter()
            .flat_map(|w| w.polynomials.iter())
            .collect::<Vec<_>>();
        assert_eq!(statements.len(), polynomials.len());
        for statement in &statements {
            assert_eq!(
                statement.num_variables(),
                self.0.mv_parameters.num_variables
            );
        }

        // Add any OODS constraints to the statements.
        for (statement, oods_constraints) in statements.iter_mut().zip(
            witnesses
                .iter()
                .flat_map(|w| w.oods_constraints().into_iter()),
        ) {
            statement.extend(oods_constraints)
        }
        let initial_statement = statements.iter().any(|s| !s.is_empty());
        assert_eq!(initial_statement, self.0.initial_statement);

        // Random linear combination of the batch polynomials
        // TODO: Take powers of a single scalar?
        let batch_randomness = {
            let mut randomness = vec![F::ONE; polynomials.len()];
            if randomness.len() > 1 {
                prover_state.fill_challenge_scalars(&mut randomness)?;
            }
            randomness
        };
        let coefficients = {
            let mut coeffs = vec![F::ZERO; self.0.mv_parameters.num_coefficients()];
            for (poly, &r) in polynomials.iter().zip(batch_randomness.iter()) {
                for (c, &p) in coeffs.iter_mut().zip(poly.coeffs()) {
                    *c += r * p;
                }
            }
            CoefficientList::new(coeffs)
        };

        // Random linear combination of the constraints
        // TODO: Take powers of a single scalar?
        let num_constraints = statements.iter().map(|s| s.constraints.len()).sum();
        let statement_randomness = {
            let mut randomness = vec![F::ONE; num_constraints];
            if randomness.len() > 1 {
                prover_state.fill_challenge_scalars(&mut randomness)?;
            }
            randomness
        };
        let weighted_sum = if num_constraints > 0 {
            let mut evals =
                EvaluationsList::new(vec![F::ZERO; self.0.mv_parameters.num_coefficients()]);
            let mut sum = F::ZERO;
            let mut s_randomness = statement_randomness.iter().copied();
            for (s, &poly_randomness) in statements.iter().zip(batch_randomness.iter()) {
                for c in s.constraints.iter() {
                    let r = poly_randomness * s_randomness.next().unwrap();
                    c.weights.accumulate(&mut evals, r);
                    sum += r * c.sum;
                }
            }
            Some((evals, sum))
        } else {
            None
        };

        // Do the first sumcheck round (or directly fold if no initial statement)
        let mut sumcheck_prover = None;
        let folding_randomness = if let Some((weights, sum)) = weighted_sum {
            // Create the sumcheck prover
            let mut sumcheck = SumcheckSingle::new(coefficients.clone(), weights, sum);

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

        // Collect sumcheck random variable assignments
        let mut randomness_vec = Vec::with_capacity(self.0.mv_parameters.num_variables);
        randomness_vec.extend(folding_randomness.0.iter().rev().copied());
        randomness_vec.resize(self.0.mv_parameters.num_variables, F::ZERO);

        // Construct the round state
        let mut round_state = RoundState {
            domain: self.0.starting_domain.clone(),
            round: 0,
            sumcheck_prover,
            folding_randomness,
            batch_randomness,
            coefficients,
            prev_merkles: witnesses
                .iter()
                .map(|w| (w.merkle_tree.clone(), w.merkle_leaves.clone()))
                .collect(),
            merkle_proofs: vec![],
            randomness_vec,
        };

        // Run WHIR rounds
        for _round in 0..=self.0.n_rounds() {
            self.round(prover_state, &mut round_state)?;
        }

        // Extract WhirProof
        Ok(WhirProof {
            merkle_proofs: round_state.merkle_proofs,
        })
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
        // TODO: This duplicates `CommitmentWriter::commit`
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

        // STIR Queries (also includes OODS)
        let (stir_challenges, stir_challenges_indexes) = self.compute_stir_queries(
            prover_state,
            round_state,
            num_variables,
            round_params,
            ood_points,
        )?;

        // Construct merkle proofs for each tree, and compute leaves for the oracle. The oracle
        // is an random linear combination of the witness polynomials.
        let mut merkle_proofs = Vec::new();
        let coset_size = 1_usize << self.0.folding_factor.at_round(round_state.round);
        let mut combined_answers = vec![vec![F::ZERO; coset_size]; stir_challenges_indexes.len()];
        let mut poly_offset = 0;
        for (tree, leafs) in round_state.prev_merkles.iter() {
            let leaf_size = leafs.len() >> (tree.height() - 1);
            debug_assert_eq!(leaf_size % coset_size, 0);
            let num_polys = leaf_size / coset_size;
            let randomness = &round_state.batch_randomness[poly_offset..poly_offset + num_polys];
            let mut leaves = vec![];
            for (answer, index) in combined_answers
                .iter_mut()
                .zip(stir_challenges_indexes.iter())
            {
                let start = index * leaf_size;
                let end = (index + 1) * leaf_size;
                let cosets = &leafs[start..end];
                leaves.push(cosets.to_vec());
                for (&r, coset) in randomness.iter().zip(cosets.chunks_exact(coset_size)) {
                    for (a, &c) in answer.iter_mut().zip(coset.iter()) {
                        *a += r * c;
                    }
                }
            }
            poly_offset += num_polys;
            merkle_proofs.push((
                tree.generate_multi_proof(stir_challenges_indexes.clone())
                    .expect("Error creating merkle proof"),
                leaves,
            ));
        }
        round_state.merkle_proofs.push(merkle_proofs);
        //
        // Evaluate answers in the folding randomness.
        let mut stir_evaluations = ood_answers;
        self.0.fold_optimisation.stir_evaluations_prover(
            round_state,
            &stir_challenges_indexes,
            &combined_answers,
            self.0.folding_factor,
            &mut stir_evaluations,
        );

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
                let (weights, sum) = statement.combine(combination_randomness[1]);
                SumcheckSingle::new(folded_coefficients.clone(), weights, sum)
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
        round_state.batch_randomness = vec![F::ONE];
        round_state.prev_merkles = vec![(merkle_tree, evals)];

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

        assert_eq!(
            round_state.prev_merkles.len(),
            1,
            "Final round with multiple trees unimplemented."
        );
        let merkle_proof = round_state.prev_merkles[0]
            .0
            .generate_multi_proof(final_challenge_indexes.clone())
            .unwrap();
        // Every query requires opening these many in the previous Merkle tree
        let fold_size = 1 << folding_factor;
        let answers = final_challenge_indexes
            .into_iter()
            .map(|i| round_state.prev_merkles[0].1[i * fold_size..(i + 1) * fold_size].to_vec())
            .collect();
        round_state
            .merkle_proofs
            .push(vec![(merkle_proof, answers)]);

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
                    let (weights, sum) = todo!(); // round_state.statement.combine(F::ONE);
                    SumcheckSingle::new(folded_coefficients.clone(), weights, sum)
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
    pub(crate) batch_randomness: Vec<F>,
    pub(crate) coefficients: CoefficientList<F>,
    pub(crate) prev_merkles: Vec<(MerkleTree<MerkleConfig>, Vec<F>)>,
    pub(crate) merkle_proofs: Vec<Vec<(MultiPath<MerkleConfig>, Vec<Vec<F>>)>>,
    pub(crate) randomness_vec: Vec<F>,
}
