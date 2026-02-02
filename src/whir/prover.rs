use std::slice;

use ark_ff::FftField;
use ark_poly::EvaluationDomain;
use ark_std::rand::{CryptoRng, RngCore};
#[cfg(feature = "tracing")]
use tracing::instrument;

use super::{
    committer::Witness,
    parameters::WhirConfig,
    statement::{Statement, Weights},
};
use crate::{
    algebra::{
        domain::Domain,
        embedding::{self, Embedding},
        mixed_dot,
        poly_utils::{coeffs::CoefficientList, multilinear::MultilinearPoint},
        tensor_product,
    },
    hash::Hash,
    protocols::{
        geometric_challenge::geometric_challenge,
        irs_commit, matrix_commit,
        sumcheck::{self, SumcheckSingle},
    },
    transcript::{
        codecs::U64, Codec, Decoding, DuplexSpongeInterface, ProverMessage, ProverState,
        VerifierMessage,
    },
    type_info::Type,
    utils::{expand_randomness, zip_strict},
    whir::{
        parameters::RoundConfig,
        utils::{get_challenge_stir_queries, sample_ood_points},
    },
};

pub struct Prover<F: FftField> {
    config: WhirConfig<F>,
}

impl<F> Prover<F>
where
    F: FftField,
{
    pub const fn new(config: WhirConfig<F>) -> Self {
        Self { config }
    }

    pub const fn config(&self) -> &WhirConfig<F> {
        &self.config
    }

    pub(crate) fn validate_parameters(&self) -> bool {
        self.config.mv_parameters.num_variables
            == self
                .config
                .folding_factor
                .total_number(self.config.n_rounds())
                + self.config.final_sumcheck_rounds
    }

    pub(crate) const fn validate_statement(&self, statement: &Statement<F>) -> bool {
        statement.num_variables() == self.config.mv_parameters.num_variables
            && (self.config.initial_statement || statement.constraints.is_empty())
    }

    pub(crate) fn validate_witness(&self, witness: &Witness<F>) -> bool {
        if !self.config.initial_statement {
            assert!(witness.witness.out_of_domain.points.is_empty());
        }
        assert_eq!(
            witness.polynomials.len(),
            self.config.initial_committer.num_polynomials
        );
        assert!(witness
            .polynomials
            .iter()
            .all(|p| p.num_coeffs() == self.config.initial_committer.polynomial_size));
        true
    }

    /// Proves that the commitment satisfies constraints in `statement`.
    ///
    /// When called without any constraints it only perfoms a low-degree test.
    /// Returns the constraint evaluation point and values of deferred constraints.
    ///
    /// Q: In a batch commitment, which polynomial does the statement apply to?
    #[cfg_attr(feature = "tracing", instrument(skip_all))]
    pub fn prove<H, R>(
        &self,
        prover_state: &mut ProverState<H, R>,
        mut statement: Statement<F>,
        witness: Witness<F>,
    ) -> (MultilinearPoint<F>, Vec<F>)
    where
        H: DuplexSpongeInterface,
        R: RngCore + CryptoRng,
        F: Codec<[H::U]>,
        u8: Decoding<[H::U]>,
        [u8; 32]: Decoding<[H::U]>,
        U64: Codec<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        self.prove_batch(prover_state, &[statement], &[witness])
    }

    /// Proves that multiple commitments satisfy their respective constraints by batching them.
    ///
    /// This combines multiple independent witnesses and statements into a single proof using
    /// random linear combination (RLC). The prover commits to a complete cross-term evaluation
    /// matrix before sampling the batching randomness γ, preventing adaptive attacks.
    ///
    /// After Round 0, all polynomials are combined into a single batched polynomial and the
    /// protocol proceeds as standard WHIR.
    ///
    /// Returns the constraint evaluation point and values of deferred constraints.
    #[allow(clippy::too_many_lines)]
    #[cfg_attr(feature = "tracing", instrument(skip_all))]
    pub fn prove_batch<H, R>(
        &self,
        prover_state: &mut ProverState<H, R>,
        statements: &[Statement<F>],
        witnesses: &[Witness<F>],
    ) -> (MultilinearPoint<F>, Vec<F>)
    where
        H: DuplexSpongeInterface,
        R: RngCore + CryptoRng,
        F: Codec<[H::U]>,
        [u8; 32]: Decoding<[H::U]>,
        U64: Codec<[H::U]>,
        u8: Decoding<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        assert!(self.validate_parameters());
        let initial_committer = &self.config.initial_committer;
        let embedding = initial_committer.embedding();
        let num_variables = initial_committer.polynomial_size.trailing_zeros() as usize;

        // Collect all polynomials from all witnesses and validate
        let polynomials: Vec<&CoefficientList<F::BasePrimeField>> = witnesses
            .iter()
            .flat_map(|w| w.polynomials.iter())
            .collect();
        if polynomials.is_empty() {
            // TODO: Implement something sensible.
            unimplemented!("Empty case not implemented");
        }

        // Validate statements
        assert_eq!(
            statements.len(),
            polynomials.len(),
            "Number of polynomials must match number of statements"
        );
        for (polynomial, statement) in polynomials.iter().zip(statements) {
            assert_eq!(statement.num_variables(), num_variables);

            // TODO: Add a `mixed_verify` function that takes an embedding into account.
            // Also, this test is optional as the proof will fail when the statements are false.
            let polynomial = polynomial.lift(embedding);
            assert!(statement.verify(&polynomial));
        }

        // Step 1: Commit to the full N×M constraint evaluation matrix BEFORE sampling γ.
        //
        // Security: The prover must commit to ALL cross-term evaluations (P_i(w_j) for all i,j)
        // before learning the batching randomness γ. This prevents the prover from adaptively
        // choosing P_i(w_j) values after seeing γ, which would allow breaking soundness by
        // constructing polynomials that satisfy P_batched = Σ γ^i·P_i at constraint points
        // but differ elsewhere.

        // Collect all constraint weights from OOD samples and statements
        let mut all_constraint_weights = Vec::new();
        all_constraint_weights.extend(witnesses.iter().flat_map(|w| {
            w.witness
                .out_of_domain()
                .weights(&embedding::Identity::<F>::new(), num_variables)
        }));
        all_constraint_weights.extend(
            statements
                .iter()
                .flat_map(|s| s.constraints.iter().map(|c| c.weights.clone())),
        );
        let num_constraints = all_constraint_weights.len();

        // Complete evaluations of EVERY polynomial at EVERY constraint point
        // This creates an N×M matrix where N = #polynomials, M = #constraints
        // We commit this matrix to the script.
        let mut constraint_evals_matrix: Vec<Option<F>> =
            vec![None; polynomials.len() * num_constraints];
        // OODs Points
        let mut constraint_offset = 0;
        let mut polynomial_offset = 0;
        for witness in witnesses {
            for row in witness.witness.out_of_domain().rows() {
                let mut index = polynomial_offset + constraint_offset;
                for value in row {
                    assert!(constraint_evals_matrix[index].is_none());
                    constraint_evals_matrix[index] = Some(*value);
                    index += num_constraints;
                }
                constraint_offset += 1;
            }
            polynomial_offset += witness.witness.out_of_domain().num_columns() * num_constraints;
        }
        // Statement
        let mut index = witnesses
            .iter()
            .map(|w| w.witness.out_of_domain().num_points())
            .sum::<usize>();
        for statement in statements.iter() {
            for constraint in &statement.constraints {
                assert!(constraint_evals_matrix[index].is_none());
                constraint_evals_matrix[index] = Some(constraint.sum);
                index += 1; // To next column
            }
            index += num_constraints; // To next row, same column
        }
        // Completion
        for (polynomial, row) in zip_strict(
            &polynomials,
            constraint_evals_matrix.chunks_exact_mut(all_constraint_weights.len()),
        ) {
            // TODO: Avoid lifting.
            let polynomial = polynomial.lift(embedding);
            for (weights, cell) in zip_strict(&all_constraint_weights, row) {
                if let Some(value) = cell {
                    // TODO: Expensive check, do only during test
                    assert_eq!(weights.evaluate(&polynomial), *value);
                } else {
                    let eval = weights.evaluate(&polynomial);
                    prover_state.prover_message(&eval);
                    *cell = Some(eval);
                }
            }
        }
        let constraint_evals_matrix: Vec<F> = constraint_evals_matrix
            .into_iter()
            .map(|e| e.unwrap())
            .collect();

        // Step 2: Sample batching randomness γ (cryptographically bound to committed matrix)
        let batching_weights: Vec<F> = geometric_challenge(prover_state, polynomials.len());

        // Step 3: Materialize the batched polynomial P_batched = P₀ + γ·P₁ + γ²·P₂ + ...
        let mut batched_coeffs = vec![F::ZERO; initial_committer.polynomial_size];
        for (weight, polynomial) in batching_weights.iter().zip(polynomials) {
            for (acc, src) in batched_coeffs.iter_mut().zip(polynomial.coeffs()) {
                *acc += embedding.mixed_mul(*weight, *src);
            }
        }
        let batched_poly = CoefficientList::new(batched_coeffs);

        // Step 4: Build combined statement using RLC of the committed evaluation matrix
        // For each constraint j: combined_eval[j] = Σᵢ γⁱ·eval[i][j]
        let mut combined_statement = Statement::new(num_variables);

        for (constraint_idx, weights) in all_constraint_weights.into_iter().enumerate() {
            let mut combined_eval = F::ZERO;
            for (weight, poly_evals) in batching_weights
                .iter()
                .zip(constraint_evals_matrix.chunks_exact(num_constraints))
            {
                combined_eval += *weight * poly_evals[constraint_idx];
            }
            combined_statement.add_constraint(weights, combined_eval);
        }

        // Run initial sumcheck on batched polynomial with combined statement
        let mut sumcheck_prover = None;
        let folding_randomness = if self.config.initial_statement {
            let combination_randomness_gen = prover_state.verifier_message();
            let sumcheck_config = sumcheck::Config {
                // TODO: Make part of parameters
                field: Type::<F>::new(),
                initial_size: batched_poly.num_coeffs(),
                rounds: vec![
                    sumcheck::RoundConfig {
                        pow: self.config.starting_folding_pow
                    };
                    self.config.folding_factor.at_round(0)
                ],
            };
            let mut sumcheck = SumcheckSingle::new(
                batched_poly.clone(),
                &combined_statement,
                combination_randomness_gen,
            );
            let folding_randomness = sumcheck_config.prove(prover_state, &mut sumcheck);
            sumcheck_prover = Some(sumcheck);
            folding_randomness
        } else {
            let mut folding_randomness = vec![F::ZERO; self.config.folding_factor.at_round(0)];
            for randomness in &mut folding_randomness {
                *randomness = prover_state.verifier_message();
            }
            self.config.starting_folding_pow.prove(prover_state);
            MultilinearPoint(folding_randomness)
        };

        let mut randomness_vec = Vec::with_capacity(self.config.mv_parameters.num_variables);
        randomness_vec.extend(folding_randomness.0.iter().rev().copied());
        randomness_vec.resize(self.config.mv_parameters.num_variables, F::ZERO);

        let irs_witnesses_owned = witnesses
            .iter()
            .map(|w| w.witness.clone())
            .collect::<Vec<_>>();
        let mut round_state = RoundState {
            domain: self.config.starting_domain.clone(),
            round: 0,
            sumcheck_prover,
            folding_randomness,
            coefficients: batched_poly,
            prev_commitment: RoundWitness::Initial {
                witnesses: irs_witnesses_owned.as_slice(),
                batching_weights,
            },
            randomness_vec,
            statement: combined_statement,
        };

        // Execute standard WHIR rounds on the batched polynomial
        for _round in 0..=self.config.n_rounds() {
            self.round(prover_state, &mut round_state);
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

        prover_state.prover_hint_ark(&deferred);

        (constraint_eval, deferred)
    }

    #[allow(clippy::too_many_lines)]
    #[cfg_attr(feature = "tracing", instrument(skip_all, fields(size = round_state.coefficients.num_coeffs())))]
    fn round<H, R>(&self, prover_state: &mut ProverState<H, R>, round_state: &mut RoundState<F>)
    where
        H: DuplexSpongeInterface,
        R: RngCore + CryptoRng,
        F: Codec<[H::U]>,
        u8: Decoding<[H::U]>,
        [u8; 32]: Decoding<[H::U]>,
        U64: Codec<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        // Fold the coefficients

        let folded_coefficients = round_state
            .coefficients
            .fold(&round_state.folding_randomness);

        let num_variables = self.config.mv_parameters.num_variables
            - self.config.folding_factor.total_number(round_state.round);
        // num_variables should match the folded_coefficients here.
        assert_eq!(num_variables, folded_coefficients.num_variables());

        // Base case
        if round_state.round == self.config.n_rounds() {
            return self.final_round(prover_state, round_state, &folded_coefficients);
        }

        let round_params = &self.config.round_configs[round_state.round];

        // Compute the folding factors for later use
        let folding_factor = self.config.folding_factor.at_round(round_state.round);
        let folding_factor_next = self.config.folding_factor.at_round(round_state.round + 1);

        // Fold the coefficients, and compute fft of polynomial (and commit)
        let new_domain = round_state.domain.scale(2); // TODO: Why 2?
        let expansion = new_domain.size() / folded_coefficients.num_coeffs();
        let evals = self.config.reed_solomon.interleaved_encode(
            folded_coefficients.coeffs(),
            expansion,
            folding_factor_next,
        );

        // Commit to the matrix of evaluations
        let matrix_witness = round_params.matrix_committer.commit(prover_state, &evals);

        // Handle OOD (Out-Of-Domain) samples
        let (ood_points, ood_answers) = sample_ood_points(
            prover_state,
            round_params.ood_samples,
            num_variables,
            |point| folded_coefficients.evaluate(point),
        );

        // PoW
        round_params.pow.prove(prover_state);

        // Open the previous round's commitment, producing in-domain evaluations.
        // We prepend these with the OOD points and answers to produce a new
        // set of constraints for the next round.
        let (stir_challenges, stir_evaluations) = match &round_state.prev_commitment {
            RoundWitness::Initial {
                witnesses,
                batching_weights,
            } => {
                let as_refs = witnesses.iter().map(|w| w).collect::<Vec<_>>();
                let config = &self.config.initial_committer;
                let in_domain = config.open(prover_state, &as_refs);

                // Convert oods_points
                let mut points = ood_points
                    .iter()
                    .copied()
                    .map(|a| MultilinearPoint::expand_from_univariate(a, num_variables))
                    .collect::<Vec<_>>();
                let mut evals = ood_answers;

                // Convert RS evaluation point to multivariate over extension
                let weights = tensor_product(
                    &batching_weights,
                    &round_state.folding_randomness.coeff_weights(true),
                );
                points.extend(in_domain.points(config.embedding(), num_variables));
                evals.extend(in_domain.values(config.embedding(), &weights));

                (points, evals)
            }

            RoundWitness::Round {
                prev_matrix,
                prev_matrix_committer,
                prev_matrix_witness,
            } => {
                // STIR Queries
                let (stir_challenges, stir_challenges_indexes) = self.compute_stir_queries(
                    prover_state,
                    round_state,
                    num_variables,
                    round_params,
                    ood_points,
                );

                let fold_size = 1 << folding_factor;
                let leaf_size = fold_size;
                let answers: Vec<F> = stir_challenges_indexes
                    .iter()
                    .flat_map(|i| {
                        prev_matrix[i * leaf_size..(i + 1) * leaf_size]
                            .iter()
                            .copied()
                    })
                    .collect();

                assert_eq!(
                    answers.len(),
                    prev_matrix_committer.num_cols * stir_challenges_indexes.len()
                );
                prover_state.prover_hint_ark(&answers);
                prev_matrix_committer.open(
                    prover_state,
                    prev_matrix_witness,
                    &stir_challenges_indexes,
                );

                let mut stir_evaluations = ood_answers;
                stir_evaluations.extend(answers.chunks(fold_size).map(|answers| {
                    CoefficientList::new(answers.to_vec()).evaluate(&round_state.folding_randomness)
                }));
                (stir_challenges, stir_evaluations)
            }
        };

        // Randomness for combination
        let combination_randomness_gen = prover_state.verifier_message();
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

        let sumcheck_config = sumcheck::Config {
            field: Type::<F>::new(),
            initial_size: 1 << sumcheck_prover.num_variables(),
            rounds: vec![
                sumcheck::RoundConfig {
                    pow: round_params.folding_pow,
                };
                folding_factor_next
            ],
        };
        let folding_randomness = sumcheck_config.prove(prover_state, &mut sumcheck_prover);

        let start_idx = self.config.folding_factor.total_number(round_state.round);
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
        round_state.prev_commitment = RoundWitness::Round {
            prev_matrix: evals,
            prev_matrix_committer: round_params.matrix_committer.clone(),
            prev_matrix_witness: matrix_witness,
        };
    }

    #[cfg_attr(feature = "tracing", instrument(skip_all, fields(size = folded_coefficients.num_coeffs())))]
    fn final_round<H, R>(
        &self,
        prover_state: &mut ProverState<H, R>,
        round_state: &mut RoundState<F>,
        folded_coefficients: &CoefficientList<F>,
    ) where
        H: DuplexSpongeInterface,
        R: RngCore + CryptoRng,
        F: Codec<[H::U]>,
        u8: Decoding<[H::U]>,
        [u8; 32]: Decoding<[H::U]>,
        U64: Codec<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        // Directly send coefficients of the polynomial to the verifier.
        for coeff in folded_coefficients.coeffs() {
            prover_state.prover_message(coeff);
        }

        // Precompute the folding factors for later use
        let folding_factor = self.config.folding_factor.at_round(round_state.round);

        // PoW
        self.config.final_pow.prove(prover_state);

        match &round_state.prev_commitment {
            RoundWitness::Initial { witnesses, .. } => {
                for witness in witnesses.iter() {
                    let in_domain = self.config.initial_committer.open(prover_state, &[witness]);

                    // The verifier will directly test these on the final polynomial.
                    // It has the final polynomial in full, so it needs no furhter help
                    // from us.
                    drop(in_domain);
                }
            }
            RoundWitness::Round {
                prev_matrix,
                prev_matrix_committer,
                prev_matrix_witness,
            } => {
                // Final verifier queries and answers. The indices are over the
                // *folded* domain.
                let final_challenge_indexes = get_challenge_stir_queries(
                    prover_state,
                    // The size of the *original* domain before folding
                    round_state.domain.size(),
                    // The folding factor we used to fold the previous polynomial
                    folding_factor,
                    self.config.final_queries,
                );

                // Every query requires opening these many in the previous Merkle tree
                let fold_size = 1 << folding_factor;
                let answers = final_challenge_indexes
                    .iter()
                    .flat_map(|i| {
                        prev_matrix[i * fold_size..(i + 1) * fold_size]
                            .iter()
                            .copied()
                    })
                    .collect::<Vec<F>>();

                assert_eq!(
                    answers.len(),
                    prev_matrix_committer.num_cols * final_challenge_indexes.len()
                );
                prover_state.prover_hint_ark(&answers);
                prev_matrix_committer.open(
                    prover_state,
                    prev_matrix_witness,
                    &final_challenge_indexes,
                );
            }
        }

        // Final sumcheck
        let mut final_folding_sumcheck = round_state.sumcheck_prover.clone().unwrap_or_else(|| {
            SumcheckSingle::new(folded_coefficients.clone(), &round_state.statement, F::ONE)
        });

        let sumcheck_config = sumcheck::Config {
            field: Type::<F>::new(),
            initial_size: 1 << final_folding_sumcheck.num_variables(),
            rounds: vec![
                sumcheck::RoundConfig {
                    pow: self.config.final_folding_pow
                };
                self.config.final_sumcheck_rounds
            ],
        };

        let final_folding_randomness =
            sumcheck_config.prove(prover_state, &mut final_folding_sumcheck);

        let start_idx = self.config.folding_factor.total_number(round_state.round);
        let rand_dst = &mut round_state.randomness_vec
            [start_idx..start_idx + final_folding_randomness.0.len()];

        for (dst, src) in rand_dst
            .iter_mut()
            .zip(final_folding_randomness.0.iter().rev())
        {
            *dst = *src;
        }
    }

    fn compute_stir_queries<H, R>(
        &self,
        prover_state: &mut ProverState<H, R>,
        round_state: &RoundState<F>,
        num_variables: usize,
        round_params: &RoundConfig<F>,
        ood_points: Vec<F>,
    ) -> (Vec<MultilinearPoint<F>>, Vec<usize>)
    where
        H: DuplexSpongeInterface,
        R: RngCore + CryptoRng,
        u8: Decoding<[H::U]>,
    {
        let stir_challenges_indexes = get_challenge_stir_queries(
            prover_state,
            round_state.domain.size(),
            self.config.folding_factor.at_round(round_state.round),
            round_params.num_queries,
        );

        // Compute the generator of the folded domain, in the extension field
        let domain_scaled_gen = round_state
            .domain
            .backing_domain
            .element(1 << self.config.folding_factor.at_round(round_state.round));
        let stir_challenges = ood_points
            .into_iter()
            .chain(
                stir_challenges_indexes
                    .iter()
                    .map(|i| domain_scaled_gen.pow([*i as u64])),
            )
            .map(|univariate| MultilinearPoint::expand_from_univariate(univariate, num_variables))
            .collect();

        (stir_challenges, stir_challenges_indexes)
    }
}

pub(crate) enum RoundWitness<'a, F: FftField> {
    Initial {
        witnesses: &'a [irs_commit::Witness<F::BasePrimeField, F>],
        batching_weights: Vec<F>,
    },
    Round {
        prev_matrix: Vec<F>,
        prev_matrix_committer: matrix_commit::Config<F>,
        prev_matrix_witness: matrix_commit::Witness,
    },
}

/// Represents the prover state during a single round of the WHIR protocol.
///
/// Each WHIR round folds the polynomial, commits to the new evaluations,
/// responds to verifier queries, and updates internal randomness for the next step.
/// This struct tracks all data needed to perform that round, and passes it forward
/// across recursive iterations.
pub(crate) struct RoundState<'a, F>
where
    F: FftField,
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

    /// Matrix commitment to the polynomial evaluations from the previous round.
    ///
    /// Used to prove query openings from the folded function.
    pub(crate) prev_commitment: RoundWitness<'a, F>,

    /// Flat list of evaluations corresponding to `prev_merkle` leaves.
    ///
    /// Each folded function is evaluated on a domain and split into leaves.

    /// Accumulator for all folding randomness across rounds.
    ///
    /// Ordered with the most recent round’s randomness at the front.
    pub(crate) randomness_vec: Vec<F>,

    /// Constraint system being enforced in this round.
    ///
    /// May be updated during recursion as queries are folded and batched.
    pub(crate) statement: Statement<F>,
}
