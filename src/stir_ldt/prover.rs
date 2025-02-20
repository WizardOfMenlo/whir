use super::{committer::Witness, parameters::StirConfig, StirProof};
use crate::{
    domain::Domain,
    ntt::expand_from_coeff,
    parameters::FoldType,
    poly_utils::{self, fold::restructure_evaluations, univariate::naive_interpolation},
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
        self.0.round_parameters.iter().fold(
            1 << self.0.uv_parameters.log_degree,
            |degree, round_parameters| {
                assert!(
                    round_parameters.num_queries + round_parameters.ood_samples
                        <= degree / (1 << self.0.folding_factor)
                );
                degree / (1 << self.0.folding_factor)
            },
        );

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
        witness: Witness<F, MerkleConfig>,
    ) -> ProofResult<StirProof<MerkleConfig, F>>
    where
        Merlin: FieldChallenges<F> + ByteWriter,
    {
        assert!(self.validate_parameters());
        assert!(self.validate_witness(&witness));

        let [folding_randomness] = merlin.challenge_scalars()?;

        // PoW: we need to compensate in order to achieve the target number of bits of security.
        if self.0.starting_folding_pow_bits > 0. {
            merlin.challenge_pow::<PowStrategy>(self.0.starting_folding_pow_bits)?;
        }

        let round_state = RoundState {
            domain: self.0.starting_domain.clone(),
            round: 0,
            folding_randomness,
            coefficients: witness.polynomial,
            prev_merkle: witness.merkle_tree,
            prev_merkle_answers: witness.merkle_leaves,
            merkle_proofs: vec![],
        };

        self.round(merlin, round_state)
    }

    fn fold(coeffs: &[F], folding_randomness: F, folding_factor: usize) -> DensePolynomial<F> {
        #[cfg(not(feature = "parallel"))]
        let coeffs = coeffs
            .chunks_exact(1 << folding_factor)
            .map(|coeffs| {
                DensePolynomial::from_coefficients_slice(coeffs).evaluate(&folding_randomness)
            })
            .collect();
        #[cfg(feature = "parallel")]
        let coeffs = coeffs
            .par_chunks_exact(1 << folding_factor)
            .map(|coeffs| {
                DensePolynomial::from_coefficients_slice(coeffs).evaluate(&folding_randomness)
            })
            .collect();

        DensePolynomial::from_coefficients_vec(coeffs)
    }
    fn round(
        &self,
        merlin: &mut Merlin,
        mut round_state: RoundState<F, MerkleConfig>,
    ) -> ProofResult<StirProof<MerkleConfig, F>> {
        // Fold the coefficients
        let g_poly = Self::fold(
            &round_state.coefficients,
            round_state.folding_randomness,
            self.0.folding_factor,
        );

        // Base case
        if round_state.round == self.0.n_rounds() {
            // Coefficients of the polynomial
            let mut coeffs = g_poly.coeffs;
            coeffs.resize(1 << self.0.final_log_degree, F::ZERO);
            merlin.add_scalars(&coeffs)?;

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

            return Ok(StirProof {
                merkle_proofs: round_state.merkle_proofs,
            });
        }

        let round_parameters = &self.0.round_parameters[round_state.round];

        // PHASE 1:
        // (1.) Fold the coefficients (2.) compute fft of polynomial (3.) commit
        let g_domain = round_state.domain.scale_with_offset(2);
        // TODO: This is not doing the efficient evaulations. In order to make it faster we need to
        // implement the shifting in the ntt engine.
        let g_evaluations = g_poly
            .evaluate_over_domain_by_ref(g_domain.backing_domain)
            .evals;
        // TODO: `stack_evaluations` and `restructure_evaluations` are really in-place algorithms.
        // They also partially overlap and undo one another. We should merge them.
        let g_folded_evaluations = utils::stack_evaluations(g_evaluations, self.0.folding_factor);
        // At this point folded evals is a matrix of size (new_domain.size()) X (1 << folding_factor)
        // This allows for the evaluation of the virutal function using an interpolation on the rows.
        // TODO: for stir we do only Naive, so this will need to be adapted.
        let g_folded_evaluations = restructure_evaluations(
            g_folded_evaluations,
            FoldType::Naive,
            g_domain.backing_domain.group_gen(),
            g_domain.backing_domain.group_gen_inv(),
            self.0.folding_factor,
        );

        // The leaves of the merkle tree are the k points that are the
        // roots of unity of the index in the previous domain. This
        // allows for the evaluation of the virtual function.
        #[cfg(not(feature = "parallel"))]
        let leaf_iterator = folded_evals.chunks_exact(1 << self.0.folding_factor);

        #[cfg(feature = "parallel")]
        let leaf_iterator = g_folded_evaluations.par_chunks_exact(1 << self.0.folding_factor);

        let g_merkle = MerkleTree::<MerkleConfig>::new(
            &self.0.leaf_hash_params,
            &self.0.two_to_one_params,
            leaf_iterator,
        )
        .unwrap();

        let g_root = g_merkle.root();
        merlin.add_bytes(g_root.as_ref())?;

        // PHASE 2:
        // OOD Sampling
        // These are the ri_out's from the paper.
        let mut ood_sample_points = vec![F::ZERO; round_parameters.ood_samples];
        // These are the beta's from the paper.
        let mut betas = Vec::with_capacity(round_parameters.ood_samples);
        if round_parameters.ood_samples > 0 {
            merlin.fill_challenge_scalars(&mut ood_sample_points)?;
            betas.extend(
                ood_sample_points
                    .iter()
                    .map(|ood_point| g_poly.evaluate(ood_point)),
            );
            merlin.add_scalars(&betas)?;
        }

        // PHASE 3:
        // STIR queries
        let mut stir_queries_seed = [0u8; 32];
        merlin.fill_challenge_bytes(&mut stir_queries_seed)?;
        let mut stir_gen = rand_chacha::ChaCha20Rng::from_seed(stir_queries_seed);

        let size_of_folded_domain = round_state.domain.folded_size(self.0.folding_factor);
        // Obtain t random integers between 0 and size of the folded domain.
        // These are the r_shifts from the paper.
        let stir_challenges_indexes = utils::dedup(
            (0..round_parameters.num_queries).map(|_| stir_gen.gen_range(0..size_of_folded_domain)),
        );

        let fold_size = 1 << self.0.folding_factor;
        let l_k = round_state.domain.scale(fold_size).backing_domain;

        let stir_challenges_points: Vec<F> = stir_challenges_indexes
            .iter()
            .map(|&i| l_k.element(i))
            .collect();

        // These are the virtual oracle domain points: the challenge field elements prior to evaluation.
        let stir_challenges_cosets: Vec<Vec<F>> = stir_challenges_indexes
            .iter()
            .map(|&i| {
                // Note: the domain has not been restructured the same way that the evaluations have.
                (i..i + fold_size * size_of_folded_domain)
                    .step_by(size_of_folded_domain)
                    .map(|j| round_state.domain.backing_domain.element(j))
                    .collect()
            })
            .collect();

        // These are the virtual oracle evaluation points: the polynomial evaluated at domain point.
        let stir_challenges_virtual_evals: Vec<Vec<F>> = stir_challenges_indexes
            .iter()
            .map(|i| round_state.prev_merkle_answers[i * fold_size..(i + 1) * fold_size].to_vec())
            .collect();

        // Example assertion. Should be refined, but it's really just a sanity check. Eventually
        // these should go.
        assert!(stir_challenges_cosets
            .iter()
            .zip(stir_challenges_virtual_evals.iter())
            .all(|(x, y)| x
                .iter()
                .zip(y.iter())
                .all(|(a, b)| round_state.coefficients.evaluate(a).eq(b))));
        assert!(stir_challenges_cosets.iter().all(|v| v.len() == fold_size));
        assert!(stir_challenges_virtual_evals
            .iter()
            .all(|v| v.len() == fold_size));

        // Merkle proof for the previous evaluations.
        let stir_challenges_proof = round_state
            .prev_merkle
            .generate_multi_proof(stir_challenges_indexes.clone())
            .unwrap();

        round_state
            .merkle_proofs
            .push((stir_challenges_proof, stir_challenges_virtual_evals));

        let quotient_set: Vec<F> = ood_sample_points
            .clone()
            .into_iter()
            .chain(stir_challenges_cosets.clone().into_iter())
            .collect();

        // PoW
        if round_parameters.pow_bits > 0. {
            merlin.challenge_pow::<PowStrategy>(round_parameters.pow_bits)?;
        }

        // Randomness for combination
        let [comb_rand]: [F; 1] = merlin.challenge_scalars()?;
        let comb_rand_coeffs = std::iter::successors(Some(F::ONE), |&prev| Some(comb_rand * prev))
            .take(quotient_set.len())
            .collect();

        // The quotient polynomial is then computed
        let quotient_polynomial = poly_utils::univariate::poly_quotient(&g_poly, &quotient_set);

        // This is the polynomial 1 + r * x + r^2 * x^2 + ... + r^n * x^n where n = |quotient_set|
        let scaling_polynomial = DensePolynomial::from_coefficients_vec(comb_rand_coeffs);

        let new_coefficients = &quotient_polynomial * scaling_polynomial;

        let [folding_randomness] = merlin.challenge_scalars()?;

        let round_state = RoundState {
            round: round_state.round + 1,
            domain: g_domain,
            folding_randomness,
            coefficients: new_coefficients,
            prev_merkle: g_merkle,
            prev_merkle_answers: g_folded_evaluations,
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
    folding_randomness: F,
    coefficients: DensePolynomial<F>,
    prev_merkle: MerkleTree<MerkleConfig>,
    prev_merkle_answers: Vec<F>,
    merkle_proofs: Vec<(MultiPath<MerkleConfig>, Vec<Vec<F>>)>,
}
