use super::{committer::Witness, parameters::StirConfig, StirProof};
use crate::{
    domain::Domain,
    parameters::FoldType,
    poly_utils::{self, fold::restructure_evaluations},
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

#[cfg(feature = "parallel")]
use rayon::prelude::*;

pub struct Prover<F, MerkleConfig, PowStrategy>(pub StirConfig<F, MerkleConfig, PowStrategy>)
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
    fn validate_parameters(&self) -> bool {
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
        self.0.uv_parameters.log_degree
            == (self.0.n_rounds() + 1) * self.0.folding_factor + self.0.final_log_degree
    }

    fn validate_witness(&self, witness: &Witness<F, MerkleConfig>) -> bool {
        (witness.polynomial.degree() + 1) == 1 << self.0.uv_parameters.log_degree
    }

    pub fn prove(
        &self,
        merlin: &mut Merlin,
        witness: &Witness<F, MerkleConfig>,
    ) -> ProofResult<StirProof<MerkleConfig, F>>
    where
        Merlin: FieldChallenges<F> + ByteWriter,
    {
        assert!(self.validate_parameters());
        assert!(self.validate_witness(&witness));

        let [r_fold] = merlin.challenge_scalars()?;

        // PoW: we need to compensate in order to achieve the target number of bits of security.
        if self.0.starting_folding_pow_bits > 0. {
            merlin.challenge_pow::<PowStrategy>(self.0.starting_folding_pow_bits)?;
        }

        let round_state = RoundState {
            f_domain: self.0.starting_domain.clone(),
            round: 0, // The last completed round.
            r_fold,
            merkle_and_eval: MerkleAndEval::F {
                merkle: witness.merkle_tree.clone(),
                evals: witness.merkle_leaves.clone(),
            },
            f_poly: witness.polynomial.clone(),
            merkle_proofs: Vec::new(),
        };

        self.round(merlin, round_state)
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
    fn round(
        &self,
        merlin: &mut Merlin,
        mut round_state: RoundState<F, MerkleConfig>,
    ) -> ProofResult<StirProof<MerkleConfig, F>> {
        // Fold the coefficients
        let r_num = round_state.round + 1;
        let g_poly = Self::fold(
            &round_state.f_poly,
            round_state.r_fold,
            self.0.folding_factor,
        );

        // Base case
        if round_state.round == self.0.n_rounds() {
            // Coefficients of the final polynomial p
            let mut p_poly = g_poly.coeffs; // One last fold of the f function. If log_initial_degree % folding_factor = 0, then this is constant.
            p_poly.resize(1 << self.0.final_log_degree, F::ZERO);
            // Send the coefficients directly
            merlin.add_scalars(&p_poly)?;

            // Final verifier queries and answers
            let mut final_shift_queries_seed = [0u8; 32];
            merlin.fill_challenge_bytes(&mut final_shift_queries_seed)?;

            let mut final_gen = rand_chacha::ChaCha20Rng::from_seed(final_shift_queries_seed);
            let final_r_shift_indexes = utils::dedup((0..self.0.final_queries).map(|_| {
                final_gen.gen_range(0..round_state.f_domain.folded_size(self.0.folding_factor))
            }));

            let fold_size = 1 << self.0.folding_factor;

            let (merkle, evals) = match round_state.merkle_and_eval {
                MerkleAndEval::F { merkle, evals } => (merkle, evals),
                MerkleAndEval::G { merkle, evals } => (merkle, evals),
            };

            let virtual_evals =
                self.indexes_to_coset_evaluations(final_r_shift_indexes.clone(), fold_size, &evals);
            let merkle_proof = merkle
                .generate_multi_proof(final_r_shift_indexes.clone())
                .unwrap();
            round_state
                .merkle_proofs
                .push((merkle_proof, virtual_evals));

            // PoW
            if self.0.final_pow_bits > 0. {
                merlin.challenge_pow::<PowStrategy>(self.0.final_pow_bits)?;
            }

            return Ok(StirProof {
                merkle_proofs: round_state.merkle_proofs,
            });
        }

        let round_parameters = &self.0.round_parameters[round_state.round];

        // PHASE 1:
        // (1.) Fold the coefficients (2.) compute fft of polynomial (3.) commit
        let g_domain = round_state.f_domain.scale_with_offset(2);
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
        // TODO: for stir we do only Naive, so this will need to be adapted.
        let g_evals_folded = restructure_evaluations(
            g_evals_folded,
            FoldType::Naive,
            g_domain.backing_domain.group_gen(),
            g_domain.backing_domain.group_gen_inv(),
            self.0.folding_factor,
        );

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

        let g_root = g_merkle.root();
        // Commit to (aka Send) the polynomial.
        merlin.add_bytes(g_root.as_ref())?;

        // PHASE 2:
        // OOD Sampling
        // These are the ri_out's from the paper.
        let mut ood_points = vec![F::ZERO; round_parameters.ood_samples];
        // These are the beta's from the paper.
        let mut ood_evals = Vec::with_capacity(round_parameters.ood_samples);
        if round_parameters.ood_samples > 0 {
            merlin.fill_challenge_scalars(&mut ood_points)?;
            ood_evals.extend(
                ood_points
                    .iter()
                    .map(|ood_point| g_poly.evaluate(ood_point)),
            );
            merlin.add_scalars(&ood_evals)?;
        }

        // PHASE 3:
        // STIR queries
        let mut shift_queries_seed = [0u8; 32];
        merlin.fill_challenge_bytes(&mut shift_queries_seed)?;
        let mut stir_gen = rand_chacha::ChaCha20Rng::from_seed(shift_queries_seed);

        let size_of_folded_domain = round_state.f_domain.folded_size(self.0.folding_factor);
        // Obtain t random integers between 0 and size of the folded domain.
        // These are the r_shifts from the paper.
        let r_shift_indexes = utils::dedup(
            (0..round_parameters.num_queries).map(|_| stir_gen.gen_range(0..size_of_folded_domain)),
        );

        let fold_size = 1 << self.0.folding_factor;
        let l_k = round_state.f_domain.scale(fold_size).backing_domain;

        let r_shift_points: Vec<F> = r_shift_indexes.iter().map(|&i| l_k.element(i)).collect();

        let (merkle, evals) = match round_state.merkle_and_eval {
            MerkleAndEval::F { merkle, evals } => (merkle, evals),
            MerkleAndEval::G { merkle, evals } => (merkle, evals),
        };

        let r_shift_virtual_evals =
            self.indexes_to_coset_evaluations(r_shift_indexes.clone(), fold_size, &evals);

        // Merkle proof for the previous evaluations.
        let shift_challenges_proof = merkle
            .generate_multi_proof(r_shift_indexes.clone())
            .unwrap();

        round_state
            .merkle_proofs
            .push((shift_challenges_proof, r_shift_virtual_evals));

        let quotient_set: Vec<F> = ood_points
            .clone()
            .into_iter()
            .chain(r_shift_points.clone().into_iter())
            .collect();

        // PoW
        if round_parameters.pow_bits > 0. {
            merlin.challenge_pow::<PowStrategy>(round_parameters.pow_bits)?;
        }

        // The quotient polynomial is then computed
        let q_poly = poly_utils::univariate::poly_quotient(&g_poly, &quotient_set);

        // Randomness for combination
        let [r_comb]: [F; 1] = merlin.challenge_scalars()?;
        let comb_rand_coeffs = expand_randomness(r_comb, quotient_set.len() + 1);

        // This is the polynomial 1 + r * x + r^2 * x^2 + ... + r^n * x^n where n = |quotient_set|
        let scaling_polynomial = DensePolynomial::from_coefficients_vec(comb_rand_coeffs);

        let f_prime_poly = &q_poly * scaling_polynomial;

        let [folding_randomness] = merlin.challenge_scalars()?;

        let round_state = RoundState {
            round: round_state.round + 1,
            f_domain: g_domain,
            r_fold: folding_randomness,
            f_poly: f_prime_poly,
            merkle_and_eval: MerkleAndEval::G {
                merkle: g_merkle,
                evals: g_evals_folded,
            },
            merkle_proofs: round_state.merkle_proofs,
        };

        self.round(merlin, round_state)
    }

    fn indexes_to_coset_evaluations(
        &self,
        stir_challenges_indexes: Vec<usize>,
        fold_size: usize,
        evals: &Vec<F>,
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

struct RoundState<F, MerkleConfig>
where
    F: FftField,
    MerkleConfig: Config,
{
    round: usize,
    f_domain: Domain<F>,
    r_fold: F,
    f_poly: DensePolynomial<F>,
    merkle_and_eval: MerkleAndEval<F, MerkleConfig>,
    merkle_proofs: Vec<(MultiPath<MerkleConfig>, Vec<Vec<F>>)>,
}

enum MerkleAndEval<F, MerkleConfig>
where
    F: FftField,
    MerkleConfig: Config,
{
    F {
        merkle: MerkleTree<MerkleConfig>,
        evals: Vec<F>,
    },
    G {
        merkle: MerkleTree<MerkleConfig>,
        evals: Vec<F>,
    },
}
