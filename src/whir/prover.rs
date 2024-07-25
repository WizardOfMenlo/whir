use ark_crypto_primitives::merkle_tree::{Config, MerkleTree, MultiPath};
use ark_ff::{FftField, PrimeField};
use ark_poly::{univariate::DensePolynomial, EvaluationDomain};
use nimue::{
    plugins::{
        ark::{FieldChallenges, FieldWriter},
        pow::PoWChallenge,
    },
    ByteChallenges, ByteWriter, Merlin, ProofResult,
};
use rand::{Rng, SeedableRng};

use crate::{
    domain::Domain,
    poly_utils::{coeffs::CoefficientList, MultilinearPoint},
    sumcheck::prover_not_skipping::SumcheckProverNotSkipping,
    utils::{self, expand_randomness},
};

use super::{committer::Witness, parameters::WhirConfig, Statement, WhirProof};

pub struct Prover<F, MerkleConfig>(pub WhirConfig<F, MerkleConfig>)
where
    F: FftField,
    MerkleConfig: Config;

impl<F, MerkleConfig> Prover<F, MerkleConfig>
where
    F: FftField + PrimeField,
    MerkleConfig: Config<Leaf = Vec<F>>,
    MerkleConfig::InnerDigest: AsRef<[u8]>,
{
    fn validate_parameters(&self) -> bool {
        self.0.mv_parameters.num_variables == (self.0.n_rounds() + 1) * self.0.folding_factor
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

    pub fn prove(
        &self,
        merlin: &mut Merlin,
        statement: Statement<F>,
        witness: Witness<F, MerkleConfig>,
    ) -> ProofResult<WhirProof<MerkleConfig>>
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

        let mut sumcheck_prover = SumcheckProverNotSkipping::new(
            witness.polynomial.clone(),
            &initial_claims,
            &combination_randomness,
        );

        let folding_randomness =
            sumcheck_prover.compute_sumcheck_polynomials(merlin, self.0.folding_factor)?;

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

    fn round(
        &self,
        merlin: &mut Merlin,
        mut round_state: RoundState<F, MerkleConfig>,
    ) -> ProofResult<WhirProof<MerkleConfig>> {
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
            let answers = final_challenge_indexes
                .into_iter()
                .map(|i| &round_state.prev_merkle_answers[i])
                .cloned()
                .collect();
            round_state.merkle_proofs.push((merkle_proof, answers));

            // PoW
            if self.0.final_pow_bits > 0 {
                merlin.challenge_pow(self.0.final_pow_bits)?;
            }

            return Ok(WhirProof(round_state.merkle_proofs));
        }

        let round_params = &self.0.round_parameters[round_state.round];

        // Fold the coefficients, and compute fft of polynomial (and commit)
        let new_domain = round_state.domain.scale(2);
        let univariate: DensePolynomial<_> = folded_coefficients.clone().into();
        let evals = univariate
            .evaluate_over_domain_by_ref(new_domain.backing_domain)
            .evals;

        let folded_evals = utils::stack_evaluations(evals, self.0.folding_factor);

        let merkle_tree = MerkleTree::<MerkleConfig>::new(
            &self.0.leaf_hash_params,
            &self.0.two_to_one_params,
            &folded_evals,
        )
        .unwrap();

        let root = merkle_tree.root();
        merlin.add_bytes(root.as_ref())?;

        // OOD Samples
        let mut ood_points = vec![F::ZERO; round_params.ood_samples];
        if round_params.ood_samples > 0 {
            merlin.fill_challenge_scalars(&mut ood_points)?;
            let ood_answers: Vec<_> = ood_points
                .iter()
                .map(|ood_point| {
                    folded_coefficients.evaluate(&MultilinearPoint::expand_from_univariate(
                        *ood_point,
                        num_variables,
                    ))
                })
                .collect();
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
        let answers = stir_challenges_indexes
            .into_iter()
            .map(|i| &round_state.prev_merkle_answers[i])
            .cloned()
            .collect();
        round_state.merkle_proofs.push((merkle_proof, answers));

        // PoW
        if round_params.pow_bits > 0 {
            merlin.challenge_pow(round_params.pow_bits)?;
        }

        // Randomness for combination
        let [combination_randomness_gen] = merlin.challenge_scalars()?;
        let combination_randomness =
            expand_randomness(combination_randomness_gen, stir_challenges.len());

        round_state
            .sumcheck_prover
            .add_new_equality(&stir_challenges, &combination_randomness);

        let folding_randomness = round_state
            .sumcheck_prover
            .compute_sumcheck_polynomials(merlin, self.0.folding_factor)?;

        let round_state = RoundState {
            round: round_state.round + 1,
            domain: new_domain,
            sumcheck_prover: round_state.sumcheck_prover,
            folding_randomness,
            coefficients: folded_coefficients,
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
    prev_merkle_answers: Vec<Vec<F>>,
    merkle_proofs: Vec<(MultiPath<MerkleConfig>, Vec<Vec<F>>)>,
}
