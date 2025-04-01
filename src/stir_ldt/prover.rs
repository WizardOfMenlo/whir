use super::{
    committer::Witness,
    parameters::{RoundConfig, StirConfig},
    StirProof,
};
use crate::{
    domain::Domain,
    poly_utils::{self},
    utils::{self, expand_randomness},
};
use ark_crypto_primitives::merkle_tree::{Config, MerkleTree, MultiPath};
use ark_ff::FftField;
use ark_poly::{univariate::DensePolynomial, DenseUVPolynomial, EvaluationDomain, Polynomial};
use nimue::{
    plugins::ark::{FieldChallenges, FieldWriter},
    ByteChallenges, ByteWriter, Merlin, ProofResult,
};
use nimue_pow::{self, PoWChallenge};
use rand::{Rng, SeedableRng};

use rand_chacha::ChaCha20Rng;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

pub struct Prover<F, MerkleConfig, PowStrategy>(StirConfig<F, MerkleConfig, PowStrategy>)
where
    F: FftField,
    MerkleConfig: Config;

impl<F, MerkleConfig, PowStrategy> Prover<F, MerkleConfig, PowStrategy>
where
    F: FftField,
    MerkleConfig: Config<Leaf = [F]>,
    MerkleConfig::InnerDigest: AsRef<[u8]>,
    PowStrategy: nimue_pow::PowStrategy,
{
    pub fn new(config: StirConfig<F, MerkleConfig, PowStrategy>) -> Self {
        Self::validate_config(&config);
        Self(config)
    }

    pub fn prove(
        &self,
        merlin: &mut Merlin,
        witness: &Witness<F, MerkleConfig>,
    ) -> ProofResult<StirProof<F, MerkleConfig>>
    where
        Merlin: FieldChallenges<F> + ByteWriter,
    {
        self.validate_witness(witness);

        let [r_fold] = merlin.challenge_scalars()?;

        // PoW: we need to compensate in order to achieve the target number of bits of security.
        if self.0.starting_folding_pow_bits > 0. {
            merlin.challenge_pow::<PowStrategy>(self.0.starting_folding_pow_bits)?;
        }

        let mut ctx = RoundContext {
            f_domain: self.0.starting_domain.clone(),
            r_fold,
            f_poly: witness.polynomial.clone(),
            merkle: witness.merkle_tree.clone(),
            evals: witness.merkle_leaves.clone(),
            merkle_proofs: vec![],
        };

        for round_config in &self.0.round_parameters {
            self.normal_round(merlin, round_config, &mut ctx)?;
        }

        self.final_round(merlin, ctx)
    }

    fn fold(coeffs: &[F], r_fold: F, folding_factor: usize) -> DensePolynomial<F> {
        #[cfg(not(feature = "parallel"))]
        let coeffs = coeffs
            .chunks_exact(1 << folding_factor)
            .map(|coeffs| DensePolynomial::from_coefficients_slice(coeffs).evaluate(&r_fold))
            .collect();

        #[cfg(feature = "parallel")]
        let coeffs = coeffs
            .par_chunks_exact(1 << folding_factor)
            .map(|coeffs| DensePolynomial::from_coefficients_slice(coeffs).evaluate(&r_fold))
            .collect();

        DensePolynomial::from_coefficients_vec(coeffs)
    }

    fn validate_config(config: &StirConfig<F, MerkleConfig, PowStrategy>) {
        // Check that for each round the repetition parameters are appropriate.
        // This is the inequality from Construction 5.2, bullet point 6.
        // let mut degree = 1 << self.0.uv_parameters.log_degree;
        // for round_param in self.0.round_parameters.iter() {
        //     degree /= 1 << self.0.folding_factor;
        //     if round_param.num_queries + round_param.ood_samples <= degree {
        //         return false;
        //     }
        // }

        // Check that the degrees add up
        assert_eq!(
            config.uv_parameters.log_degree,
            (config.round_parameters.len() + 1) * config.folding_factor + config.final_log_degree,
        )
    }

    fn validate_witness(&self, witness: &Witness<F, MerkleConfig>) {
        assert_eq!(
            (witness.polynomial.degree() + 1),
            1 << self.0.uv_parameters.log_degree,
        )
    }

    fn normal_round(
        &self,
        merlin: &mut Merlin,
        round_config: &RoundConfig,
        ctx: &mut RoundContext<F, MerkleConfig>,
    ) -> ProofResult<()> {
        // Fold the coefficients
        let g_poly = Self::fold(&ctx.f_poly, ctx.r_fold, self.0.folding_factor);

        // PHASE 1 (folding):
        let (g_domain, g_evals_folded, g_merkle) = self.folding_phase(&g_poly, ctx);
        let g_root = g_merkle.root();
        // Commit to (aka Send) the polynomial.
        merlin.add_bytes(g_root.as_ref())?;

        // PHASE 2 (OOD sampling):
        // These are the ri_out's from the paper.
        let mut ood_points = vec![F::ZERO; round_config.ood_samples];
        // These are the beta's from the paper.
        if round_config.ood_samples > 0 {
            merlin.fill_challenge_scalars(&mut ood_points)?;
            merlin.add_scalars(
                &ood_points
                    .iter()
                    .map(|ood_point| g_poly.evaluate(ood_point))
                    .collect::<Vec<_>>(),
            )?;
        }

        // PHASE 3 (STIR queries):
        let r_shift_indexes = self.stir_phase(merlin, round_config.num_queries, ctx)?;
        let l_k = ctx
            .f_domain
            .scale(1 << self.0.folding_factor)
            .backing_domain;
        let quotient_set: Vec<_> = ood_points
            .iter()
            .copied()
            .chain(r_shift_indexes.iter().map(|&i| l_k.element(i)))
            .collect();

        // PoW
        if round_config.pow_bits > 0. {
            merlin.challenge_pow::<PowStrategy>(round_config.pow_bits)?;
        }

        // The quotient polynomial is then computed
        let q_poly = poly_utils::univariate::poly_quotient(&g_poly, &quotient_set);

        // Randomness for combination
        let [r_comb] = merlin.challenge_scalars()?;
        let comb_rand_coeffs = expand_randomness(r_comb, quotient_set.len() + 1);

        // This is the polynomial 1 + r * x + r^2 * x^2 + ... + r^n * x^n where n = |quotient_set|
        let scaling_polynomial = DensePolynomial::from_coefficients_vec(comb_rand_coeffs);

        let f_prime_poly = &q_poly * scaling_polynomial;

        let [new_folding_randomness] = merlin.challenge_scalars()?;

        // Update RoundContext
        ctx.f_domain = g_domain;
        ctx.r_fold = new_folding_randomness;
        ctx.f_poly = f_prime_poly;
        ctx.merkle = g_merkle;
        ctx.evals = g_evals_folded;

        Ok(())
    }

    fn final_round(
        &self,
        merlin: &mut Merlin,
        mut ctx: RoundContext<F, MerkleConfig>,
    ) -> ProofResult<StirProof<F, MerkleConfig>> {
        // Fold the coefficients
        let g_poly = Self::fold(&ctx.f_poly, ctx.r_fold, self.0.folding_factor);

        // Coefficients of the final polynomial p
        let mut p_poly = g_poly.coeffs; // One last fold of the f function. If log_initial_degree % folding_factor = 0, then this is constant.
        p_poly.resize(1 << self.0.final_log_degree, F::ZERO);
        // Send the coefficients directly
        merlin.add_scalars(&p_poly)?;

        self.stir_phase(merlin, self.0.final_queries, &mut ctx)?;

        // PoW
        if self.0.final_pow_bits > 0. {
            merlin.challenge_pow::<PowStrategy>(self.0.final_pow_bits)?;
        }

        Ok(StirProof {
            merkle_proofs: ctx.merkle_proofs,
        })
    }

    fn folding_phase(
        &self,
        g_poly: &DensePolynomial<F>,
        ctx: &RoundContext<F, MerkleConfig>,
    ) -> (Domain<F>, Vec<F>, MerkleTree<MerkleConfig>) {
        // (1.) Fold the coefficients (2.) compute fft of polynomial (3.) commit
        let g_domain = ctx.f_domain.scale_with_offset(2);
        // TODO: This is not doing the efficient evaulations. In order to make it faster we need to
        // implement the shifting in the ntt engine.
        let g_evals = g_poly
            .evaluate_over_domain_by_ref(g_domain.backing_domain)
            .evals;

        // TODO: `stack_evaluations` and `restructure_evaluations` are really in-place algorithms.
        // They also partially overlap and undo one another. We should merge them.
        let g_evals_folded = utils::stack_evaluations(g_evals, self.0.folding_factor);

        // At this point folded evals is a matrix of size (new_domain.size()) X (1 << folding_factor)
        // This allows for the evaluation of the virutal function using an interpolation on the rows.
        //
        // The leaves of the merkle tree are the k points that are the
        // roots of unity of the index in the previous domain. This
        // allows for the evaluation of the virtual function.
        #[cfg(not(feature = "parallel"))]
        let leaf_iterator = g_evals_folded.chunks_exact(1 << self.0.folding_factor);

        #[cfg(feature = "parallel")]
        let leaf_iterator = g_evals_folded.par_chunks_exact(1 << self.0.folding_factor);

        let g_merkle = MerkleTree::<MerkleConfig>::new(
            &self.0.leaf_hash_params,
            &self.0.two_to_one_params,
            leaf_iterator,
        )
        .unwrap();

        (g_domain, g_evals_folded, g_merkle)
    }

    fn stir_phase(
        &self,
        merlin: &mut Merlin,
        num_queries: usize,
        ctx: &mut RoundContext<F, MerkleConfig>,
    ) -> ProofResult<Vec<usize>> {
        let stir_gen = &mut ChaCha20Rng::from_seed(merlin.challenge_bytes()?);
        let size_of_folded_domain = ctx.f_domain.folded_size(self.0.folding_factor);
        // Obtain t random integers between 0 and size of the folded domain.
        // These are the r_shifts from the paper.

        // TODO: this could return fewer than `round_parameters.num_queries` elements
        let r_shift_indexes =
            utils::dedup((0..num_queries).map(|_| stir_gen.gen_range(0..size_of_folded_domain)));

        let virtual_evals = self.indexes_to_coset_evaluations(
            r_shift_indexes.clone(),
            1 << self.0.folding_factor,
            &ctx.evals,
        );

        // Merkle proof for the previous evaluations.
        let merkle_proof = ctx
            .merkle
            .generate_multi_proof(r_shift_indexes.clone())
            .unwrap();

        ctx.merkle_proofs
            .push((merkle_proof, virtual_evals.clone()));

        Ok(r_shift_indexes)
    }

    fn indexes_to_coset_evaluations(
        &self,
        stir_challenges_indexes: Vec<usize>,
        fold_size: usize,
        evals: &[F],
    ) -> Vec<Vec<F>>
    where
        F: FftField,
        MerkleConfig: Config<Leaf = [F]>,
        PowStrategy: nimue_pow::PowStrategy,
    {
        assert!(evals.len() % fold_size == 0);
        let stir_challenges_virtual_evals: Vec<Vec<F>> = stir_challenges_indexes
            .iter()
            .map(|i| evals[i * fold_size..(i + 1) * fold_size].to_vec())
            .collect();
        stir_challenges_virtual_evals
    }
}

struct RoundContext<F: FftField, MerkleConfig: Config> {
    f_domain: Domain<F>,
    r_fold: F,
    f_poly: DensePolynomial<F>,
    // NOTE: merkle and eval refer to f in the first round
    // and to g_{i-1} in every following round i
    merkle: MerkleTree<MerkleConfig>,
    evals: Vec<F>,
    merkle_proofs: Vec<(MultiPath<MerkleConfig>, Vec<Vec<F>>)>,
}
