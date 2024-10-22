use std::iter;

use super::{committer::Witness, parameters::StirConfig, StirProof};
use crate::{
    domain::Domain,
    ntt::expand_from_coeff,
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

        // PoW
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
        let folded_coefficients = Self::fold(
            &round_state.coefficients,
            round_state.folding_randomness,
            self.0.folding_factor,
        );

        // Base case
        if round_state.round == self.0.n_rounds() {
            // Coefficients of the polynomial
            let mut coeffs = folded_coefficients.coeffs;
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

        let round_params = &self.0.round_parameters[round_state.round];

        // Fold the coefficients, and compute fft of polynomial (and commit)
        let new_domain = round_state.domain.scale(2);
        let expansion = new_domain.size() / (folded_coefficients.degree() + 1);
        let evals = expand_from_coeff(folded_coefficients.coeffs(), expansion);
        // TODO: `stack_evaluations` and `restructure_evaluations` are really in-place algorithms.
        // They also partially overlap and undo one another. We should merge them.
        let folded_evals = utils::stack_evaluations(evals, self.0.folding_factor);
        let folded_evals = restructure_evaluations(
            folded_evals,
            FoldType::Naive,
            new_domain.backing_domain.group_gen(),
            new_domain.backing_domain.group_gen_inv(),
            self.0.folding_factor,
        );

        #[cfg(not(feature = "parallel"))]
        let leaf_iter = folded_evals.chunks_exact(1 << self.0.folding_factor);

        #[cfg(feature = "parallel")]
        let leaf_iter = folded_evals.par_chunks_exact(1 << self.0.folding_factor);

        let merkle_tree = MerkleTree::<MerkleConfig>::new(
            &self.0.leaf_hash_params,
            &self.0.two_to_one_params,
            leaf_iter,
        )
        .unwrap();

        let root = merkle_tree.root();
        merlin.add_bytes(root.as_ref())?;

        // OOD Samples
        let mut ood_points = vec![F::ZERO; round_params.ood_samples];
        let mut ood_answers = Vec::with_capacity(round_params.ood_samples);
        if round_params.ood_samples > 0 {
            merlin.fill_challenge_scalars(&mut ood_points)?;
            ood_answers.extend(
                ood_points
                    .iter()
                    .map(|ood_point| folded_coefficients.evaluate(ood_point)),
            );
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
        stir_evaluations.extend(answers.iter().map(|answers| {
            DensePolynomial::from_coefficients_slice(answers)
                .evaluate(&round_state.folding_randomness)
        }));

        round_state.merkle_proofs.push((merkle_proof, answers));

        // PoW
        if round_params.pow_bits > 0. {
            merlin.challenge_pow::<PowStrategy>(round_params.pow_bits)?;
        }

        // Randomness for combination
        let [combination_randomness_gen]: [F; 1] = merlin.challenge_scalars()?;
        let combination_randomness =
            expand_randomness(combination_randomness_gen, stir_challenges.len());

        // The quotient polynomial is then computed
        let quotient_polynomial =
            poly_utils::univariate::poly_quotient(&folded_coefficients, &stir_challenges);

        // This is the polynomial 1 + r * x + r^2 * x^2 + ... + r^n * x^n where n = |quotient_set|
        let scaling_polynomial = DensePolynomial::from_coefficients_vec(combination_randomness);

        let new_coefficients = &quotient_polynomial * scaling_polynomial;

        let [folding_randomness] = merlin.challenge_scalars()?;

        let round_state = RoundState {
            round: round_state.round + 1,
            domain: new_domain,
            folding_randomness,
            coefficients: new_coefficients,
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
    folding_randomness: F,
    coefficients: DensePolynomial<F>,
    prev_merkle: MerkleTree<MerkleConfig>,
    prev_merkle_answers: Vec<F>,
    merkle_proofs: Vec<(MultiPath<MerkleConfig>, Vec<Vec<F>>)>,
}
