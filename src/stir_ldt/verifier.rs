use std::iter;

use ark_crypto_primitives::merkle_tree::{Config, MultiPath};
use ark_ff::{FftField, Field};
use ark_poly::{univariate::DensePolynomial, DenseUVPolynomial, EvaluationDomain, Polynomial};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use spongefish::{
    codecs::arkworks_algebra::{FieldToUnitDeserialize, UnitToField},
    BytesToUnitDeserialize, ProofError, ProofResult, UnitToBytes, VerifierState,
};
use spongefish_pow::{self, PoWChallenge};

use super::{
    committer::reader::ParsedCommitment,
    parameters::{RoundConfig, StirConfig},
    StirProof,
};
use crate::{
    parameters::FoldType,
    poly_utils::{fold::compute_fold_univariate, univariate::naive_interpolation},
    utils,
};

pub struct Verifier<'a, F, MerkleConfig, PowStrategy>
where
    F: FftField,
    MerkleConfig: Config,
{
    two_inv: F,
    params: &'a StirConfig<F, MerkleConfig, PowStrategy>,
}

#[derive(Clone)]
struct ParsedProof<F: Field> {
    rounds: Vec<ParsedRound<F>>,
    final_domain_gen: F,
    final_domain_gen_inv: F,
    final_domain_offset: F,
    final_domain_offset_inv: F,
    final_r_shift_indexes: Vec<usize>,
    final_r_shift_points: Vec<F>,
    final_r_shift_virtual_evals: Vec<Vec<F>>,
    final_r_fold: F,
    p_poly: DensePolynomial<F>,
}

#[derive(Debug, Clone)]
struct ParsedRound<F> {
    r_fold: F,
    ood_points: Vec<F>,
    ood_evals: Vec<F>,
    r_shift_indexes: Vec<usize>,
    r_shift_points: Vec<F>,
    r_shift_virtual_evals: Vec<Vec<F>>,
    r_comb: F,
    domain_gen: F,
    domain_gen_inv: F,
    domain_offset: F,
    domain_offset_inv: F,
    folded_domain_size: usize,
}

impl<'a, F, MerkleConfig, PowStrategy> Verifier<'a, F, MerkleConfig, PowStrategy>
where
    F: FftField,
    MerkleConfig: Config<Leaf = [F]>,
    MerkleConfig::InnerDigest: AsRef<[u8]> + From<[u8; 32]>,
    PowStrategy: spongefish_pow::PowStrategy,
{
    pub fn new(params: &'a StirConfig<F, MerkleConfig, PowStrategy>) -> Self {
        Verifier {
            two_inv: F::from(2).inverse().unwrap(),
            params,
        }
    }

    fn parse_round(
        &self,
        round_config: &RoundConfig,
        merkle_proof: &MultiPath<MerkleConfig>,
        virtual_evals: &[Vec<F>],
        verifier_state: &mut VerifierState,
        ctx: &mut ParseProofContext<F, MerkleConfig::InnerDigest>,
    ) -> ProofResult<()> {
        // TODO: does this have to happen before the other arthur calls?
        let next_root: [u8; 32] = verifier_state.next_bytes()?;

        // PHASE 2:
        let mut ood_points = vec![F::ZERO; round_config.ood_samples];
        let mut ood_evals = vec![F::ZERO; round_config.ood_samples];
        if round_config.ood_samples > 0 {
            verifier_state.fill_challenge_scalars(&mut ood_points)?;
            verifier_state.fill_next_scalars(&mut ood_evals)?;
        }

        // PHASE 3:
        let mut shift_queries_seed = [0u8; 32];
        verifier_state.fill_challenge_bytes(&mut shift_queries_seed)?;
        let mut stir_gen = ChaCha20Rng::from_seed(shift_queries_seed);
        let folded_domain_size = ctx.domain_size / (1 << self.params.folding_factor);
        let r_shift_indexes = utils::dedup(
            (0..round_config.num_queries).map(|_| stir_gen.gen_range(0..folded_domain_size)),
        );

        // Gen the points in L^k at which we're querying.
        // TODO: Check if this is really necessary
        let r_shift_points = r_shift_indexes
            .iter()
            .map(|index| ctx.domain_offset_exp * ctx.domain_gen_exp.pow([*index as u64]))
            .collect();

        if !merkle_proof
            .verify(
                &self.params.leaf_hash_params,
                &self.params.two_to_one_params,
                &ctx.prev_root,
                virtual_evals.iter().map(|a| a.as_ref()),
            )
            .unwrap()
        {
            return Err(ProofError::InvalidProof);
        }

        if round_config.pow_bits > 0. {
            verifier_state.challenge_pow::<PowStrategy>(round_config.pow_bits)?;
        }

        let [r_comb] = verifier_state.challenge_scalars()?;

        ctx.rounds.push(ParsedRound {
            r_fold: ctx.r_fold,
            ood_points,
            ood_evals,
            r_shift_indexes,
            r_shift_points,
            r_shift_virtual_evals: virtual_evals.to_vec(), // f in first round, g later.
            r_comb,
            domain_gen: ctx.domain_gen, // generator of the coset L at a given round.
            domain_gen_inv: ctx.domain_gen_inv, // inverse of `domain_gen` of the coset L at a given round.
            domain_offset: ctx.domain_offset,
            domain_offset_inv: ctx.domain_offset_inv,
            folded_domain_size,
        });

        // Update context
        let [new_r_fold] = verifier_state.challenge_scalars()?;
        ctx.r_fold = new_r_fold;

        ctx.prev_root = next_root.into();

        ctx.domain_gen *= ctx.domain_gen;
        ctx.domain_gen_inv *= ctx.domain_gen_inv;
        ctx.domain_gen_exp *= ctx.domain_gen_exp;

        ctx.domain_offset = ctx.domain_offset * ctx.domain_offset * ctx.root_of_unity;
        ctx.domain_offset_inv =
            ctx.domain_offset_inv * ctx.domain_offset_inv * ctx.root_of_unity_inv;
        ctx.domain_offset_exp =
            ctx.domain_offset_exp * ctx.domain_offset_exp * ctx.root_of_unity_exp;

        ctx.domain_size /= 2;

        Ok(())
    }

    fn parse_final_round(
        &self,
        final_merkle_proof: &MultiPath<MerkleConfig>,
        final_virtual_evals: &[Vec<F>],
        verifier_state: &mut VerifierState,
        ctx: ParseProofContext<F, MerkleConfig::InnerDigest>,
    ) -> ProofResult<ParsedProof<F>> {
        let mut final_coefficients = vec![F::ZERO; 1 << self.params.final_log_degree];
        verifier_state.fill_next_scalars(&mut final_coefficients)?;
        let p_poly = DensePolynomial::from_coefficients_vec(final_coefficients);

        // Final queries verify
        let mut final_shift_queries_seed = [0u8; 32];
        verifier_state.fill_challenge_bytes(&mut final_shift_queries_seed)?;
        let mut final_gen = ChaCha20Rng::from_seed(final_shift_queries_seed);
        let folded_domain_size = ctx.domain_size / (1 << self.params.folding_factor);
        let final_r_shift_indexes = utils::dedup(
            (0..self.params.final_queries).map(|_| final_gen.gen_range(0..folded_domain_size)),
        );
        let final_r_shift_points = final_r_shift_indexes
            .iter()
            .map(|index| ctx.domain_offset_exp * ctx.domain_gen_exp.pow([*index as u64]))
            .collect();

        if !final_merkle_proof
            .verify(
                &self.params.leaf_hash_params,
                &self.params.two_to_one_params,
                &ctx.prev_root,
                final_virtual_evals.iter().map(|a| a.as_ref()),
            )
            .unwrap()
            || final_merkle_proof.leaf_indexes != final_r_shift_indexes
        {
            return Err(ProofError::InvalidProof);
        }

        if self.params.final_pow_bits > 0. {
            verifier_state.challenge_pow::<PowStrategy>(self.params.final_pow_bits)?;
        }

        Ok(ParsedProof {
            rounds: ctx.rounds,
            final_domain_gen: ctx.domain_gen,
            final_domain_gen_inv: ctx.domain_gen_inv,
            final_domain_offset: ctx.domain_offset,
            final_domain_offset_inv: ctx.domain_offset_inv,
            final_r_shift_indexes,
            final_r_shift_points,
            final_r_shift_virtual_evals: final_virtual_evals.to_vec(),
            final_r_fold: ctx.r_fold,
            p_poly,
        })
    }

    fn parse_proof(
        &self,
        verifier_state: &mut VerifierState,
        parsed_commitment: &ParsedCommitment<MerkleConfig::InnerDigest>,
        stir_proof: &StirProof<F, MerkleConfig>,
    ) -> ProofResult<ParsedProof<F>> {
        assert_eq!(
            stir_proof.merkle_proofs.len(),
            self.params.round_parameters.len() + 1
        );

        // Derive initial combination randomness
        let [r_fold] = verifier_state.challenge_scalars()?;

        // PoW
        if self.params.starting_folding_pow_bits > 0. {
            verifier_state.challenge_pow::<PowStrategy>(self.params.starting_folding_pow_bits)?;
        }

        let root_of_unity = self.params.starting_domain.backing_domain.group_gen();
        let domain_offset = self.params.starting_domain.backing_domain.coset_offset();
        let mut ctx = ParseProofContext {
            r_fold,
            domain_size: self.params.starting_domain.size(),
            root_of_unity,
            root_of_unity_inv: root_of_unity.inverse().unwrap(),
            root_of_unity_exp: root_of_unity.pow([1 << self.params.folding_factor]),
            domain_gen: root_of_unity,
            domain_gen_inv: self.params.starting_domain.backing_domain.group_gen_inv(),
            domain_gen_exp: root_of_unity.pow([1 << self.params.folding_factor]),
            domain_offset,
            domain_offset_inv: self
                .params
                .starting_domain
                .backing_domain
                .coset_offset_inv(),
            domain_offset_exp: domain_offset.pow([1 << self.params.folding_factor]),
            prev_root: parsed_commitment.root.clone(),
            rounds: vec![],
        };

        // In the first round, the proof is directly the function f.
        // In subsequent rounds, it is the function g, which is the fold of the
        // previous function f.
        for (round_config, (merkle_proof, virtual_evals)) in self
            .params
            .round_parameters
            .iter()
            .zip(&stir_proof.merkle_proofs)
        {
            self.parse_round(
                round_config,
                merkle_proof,
                virtual_evals,
                verifier_state,
                &mut ctx,
            )?;
        }

        let (final_merkle_proof, final_virtual_evals) = stir_proof.merkle_proofs.last().unwrap();
        self.parse_final_round(final_merkle_proof, final_virtual_evals, verifier_state, ctx)
    }

    #[allow(clippy::too_many_lines)]
    pub fn verify(
        &self,
        verifier_state: &mut VerifierState,
        parsed_commitment: &ParsedCommitment<MerkleConfig::InnerDigest>,
        stir_proof: &StirProof<F, MerkleConfig>,
    ) -> ProofResult<()> {
        // PHASE 1:
        // We first do a pass in which we rederive all the FS challenges and verify Merkle paths.
        // We take care of all the crytpo here.
        // let parsed_commitment = parse_commitment(verifier_state)?;
        let parsed = self.parse_proof(verifier_state, parsed_commitment, stir_proof)?;

        // PHASE 2:
        // We do a pass which at each point defines and update a virtual function
        let (
            r_folds,
            all_r_shift_indexes,
            all_r_shift_virtual_evals,
            domain_gens,
            domain_gen_invs,
            domain_offsets,
            domain_offset_invs,
        ) = {
            let parsed: &ParsedProof<F> = &parsed;
            let r_folds: Vec<F> = parsed
                .rounds
                .iter()
                .map(|r| &r.r_fold)
                .chain(iter::once(&parsed.final_r_fold))
                .copied()
                .collect();

            let all_r_shift_indexes: Vec<Vec<usize>> = parsed
                .rounds
                .iter()
                .map(|r| &r.r_shift_indexes)
                .chain(iter::once(&parsed.final_r_shift_indexes))
                .cloned()
                .collect();

            let all_r_shift_virtual_evals: Vec<Vec<Vec<F>>> = parsed
                .rounds
                .iter()
                .map(|r| &r.r_shift_virtual_evals)
                .chain(iter::once(&parsed.final_r_shift_virtual_evals))
                .cloned()
                .collect();

            let domain_gens: Vec<_> = parsed
                .rounds
                .iter()
                .map(|r| r.domain_gen)
                .chain(iter::once(parsed.final_domain_gen))
                .collect();

            let domain_gen_invs: Vec<_> = parsed
                .rounds
                .iter()
                .map(|r| r.domain_gen_inv)
                .chain(iter::once(parsed.final_domain_gen_inv))
                .collect();

            let domain_offsets: Vec<_> = parsed
                .rounds
                .iter()
                .map(|r| r.domain_offset)
                .chain(iter::once(parsed.final_domain_offset))
                .collect();

            let domain_offset_invs: Vec<_> = parsed
                .rounds
                .iter()
                .map(|r| r.domain_offset_inv)
                .chain(iter::once(parsed.final_domain_offset_inv))
                .collect();
            (
                r_folds,
                all_r_shift_indexes,
                all_r_shift_virtual_evals,
                domain_gens,
                domain_gen_invs,
                domain_offsets,
                domain_offset_invs,
            )
        };

        // Suppose we have a function f evaluated on L = off * <gen>. To evaluate the k-wise fold of
        // f at a point i in L^k, we need points k in L to evaluate the virtual oracle. These points
        // are evaluatons of f at the following points:
        //  { off * gen^(i + j * |L|/k) | j = 0, 1, ..., k-1 }.
        // These points are a coset defined by an offset and generator which are
        // `off * gen^i` and `gen^(|L|/k)` respectively.

        let coset_domain_size: usize = 1 << self.params.folding_factor;
        let mut coset_gen_inv =
            domain_gen_invs[0].pow([
                (self.params.starting_domain.backing_domain.size() / coset_domain_size) as u64
            ]);

        let mut r_shift_evals = self.get_r_shift_evals(
            &r_folds,
            &all_r_shift_indexes,
            &all_r_shift_virtual_evals,
            &domain_gen_invs,
            &domain_offset_invs,
            coset_gen_inv,
        );

        for (r_index, round) in parsed.rounds.iter().enumerate() {
            // The next virtual function is defined by the following
            // TODO: This actually is just a single value that we need
            let r_num = r_index + 1;
            let r_comb = round.r_comb;

            let quotient_set: Vec<_> = round
                .ood_points
                .iter()
                .chain(&round.r_shift_points)
                .copied()
                .collect();

            let quotient_answers: Vec<_> = round
                .ood_evals
                .iter()
                .chain(&r_shift_evals)
                .copied()
                .collect();

            let num_terms = quotient_set.len(); // The highest power in the vanishing polynomial.

            let ans_polynomial = naive_interpolation(
                quotient_set
                    .iter()
                    .zip(quotient_answers.clone())
                    .map(|(x, y)| (*x, y)),
            );

            // The points that we are querying the new function at
            let evaluation_indexes = &all_r_shift_indexes[r_num];
            // Note: The folded domain size of the next round is half that of `this` round.
            coset_gen_inv = domain_gen_invs[r_num].pow([(round.folded_domain_size / 2) as u64]);

            // The evaluations of the previous committed function need to be reshaped into
            // evaluations of the virtual function f'
            let g_virtual_evals = &all_r_shift_virtual_evals[r_num];
            let mut f_prime_virtual_evals = Vec::with_capacity(g_virtual_evals.len());

            for (index, answer) in evaluation_indexes.iter().zip(g_virtual_evals) {
                // Coset eval is the evaluations of the virtual function on the coset
                let coset_offset_inv =
                    domain_offset_invs[r_num] * domain_gen_invs[r_num].pow([*index as u64]); // What does this do?
                let mut coset_evals = Vec::with_capacity(1 << self.params.folding_factor);
                #[allow(clippy::needless_range_loop)]
                for j in 0..1 << self.params.folding_factor {
                    // TODO: Optimize
                    // The evaluation poincs are the i + j * folded_domain_size elements of the
                    // domain. for a domain whose represented by offset * <gen>,, this translates
                    // to a domain
                    let x = domain_offsets[r_num]
                        * domain_gens[r_num]
                            .pow([(index + j * (round.folded_domain_size / 2)) as u64]);

                    let numerator = answer[j] - ans_polynomial.evaluate(&x);

                    // Just an eval of the vanishing polynomial
                    let denominator: F = quotient_set.iter().map(|point| x - point).product();
                    let denom_inv = denominator.inverse().unwrap();

                    // Scaling factor:
                    // If xr = 1 this is just the num_terms + 1
                    // If xr != 1 this is (1 - (xr)^{num_terms + 1})/(1 - xr)
                    let common_factor = x * r_comb;

                    let scale_factor = if common_factor == F::ONE {
                        F::from((num_terms + 1) as u64)
                    } else {
                        let common_factor_inverse = (F::ONE - common_factor).inverse().unwrap();
                        (F::ONE - common_factor.pow([(num_terms + 1) as u64]))
                            * common_factor_inverse
                    };

                    coset_evals.push(scale_factor * numerator * denom_inv);
                }

                let f_prime_eval = compute_fold_univariate(
                    &coset_evals,
                    r_folds[r_num],
                    coset_offset_inv,
                    coset_gen_inv,
                    self.two_inv,
                    self.params.folding_factor,
                );

                f_prime_virtual_evals.push(f_prime_eval);
            }

            r_shift_evals = f_prime_virtual_evals;
        }

        // Check the foldings computed from the proof match the evaluations of the polynomial
        let final_folds = &r_shift_evals;
        let final_evaluations = parsed
            .final_r_shift_points
            .iter()
            .map(|point| parsed.p_poly.evaluate(point));
        let final_evaluations: Vec<_> = final_evaluations.collect();
        if !final_folds
            .iter()
            .zip(&final_evaluations)
            .all(|(&fold, &eval)| fold == eval)
        {
            return Err(ProofError::InvalidProof);
        }

        Ok(())
    }

    fn get_r_shift_evals(
        &self,
        r_folds: &[F],
        all_r_shift_indexes: &[Vec<usize>],
        all_r_shift_virtual_evals: &[Vec<Vec<F>>],
        domain_gen_invs: &[F],
        domain_offset_invs: &[F],
        coset_gen_inv: F,
    ) -> Vec<F>
    where
        F: FftField,
    {
        match self.params.fold_optimization {
            FoldType::Naive => all_r_shift_indexes[0]
                .iter()
                .zip(all_r_shift_virtual_evals[0].iter())
                .map(|(index, coset_eval)| {
                    let coset_offset_inv =
                        domain_offset_invs[0] * domain_gen_invs[0].pow([*index as u64]);
                    compute_fold_univariate(
                        coset_eval,
                        r_folds[0],
                        coset_offset_inv,
                        coset_gen_inv,
                        self.two_inv,
                        self.params.folding_factor,
                    )
                })
                .collect(),
            FoldType::ProverHelps => all_r_shift_virtual_evals[0]
                .iter()
                .map(|coeffs| DensePolynomial::from_coefficients_vec(coeffs.clone()))
                .map(|poly| poly.evaluate(&r_folds[0]))
                .collect(),
        }
        // all_r_shift_indexes[0]
        //     .iter()
        //     .zip(all_r_shift_virtual_evals[0].iter())
        //     .map(|(index, coset_eval)| {
        //         let coset_offset_inv =
        //             domain_offset_invs[0] * domain_gen_invs[0].pow([*index as u64]);
        //         compute_fold_univariate(
        //             coset_eval,
        //             r_folds[0],
        //             coset_offset_inv,
        //             coset_gen_inv,
        //             self.two_inv,
        //             self.params.folding_factor,
        //         )
        //     })
        //     .collect()
    }
}

struct ParseProofContext<F, D> {
    r_fold: F,
    domain_size: usize,
    root_of_unity: F,
    root_of_unity_inv: F,
    root_of_unity_exp: F,
    domain_gen: F,
    domain_gen_inv: F,
    domain_gen_exp: F,
    domain_offset: F,
    domain_offset_inv: F,
    domain_offset_exp: F,
    prev_root: D,
    rounds: Vec<ParsedRound<F>>,
}
