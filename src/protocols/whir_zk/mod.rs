mod committer;
mod prover;
pub(crate) mod utils;
mod verifier;

pub use utils::BlindingEvaluations;

use ark_ff::{AdditiveGroup, FftField, Field};
use ark_std::{
    rand::{CryptoRng, RngCore},
    UniformRand,
};
use serde::Serialize;

use crate::{
    algebra::{fields::FieldWithSize, polynomials::CoefficientList},
    parameters::{FoldingFactor, MultivariateParameters, ProtocolParameters, SoundnessType},
    protocols::{irs_commit, whir},
};

// ── ZK WHIR Config ───────────────────────────────────────────────────

/// Configuration for the ZK WHIR protocol.
///
/// Bundles the main WHIR config (for the witness polynomial) and the
/// blinding polynomial WHIR config (for blinding polynomials committed
/// in a separate Merkle tree).
///
/// ZK parameters (`num_witness_variables`, `num_blinding_variables`) are
/// derivable from the two configs and exposed as methods.
#[derive(Clone, Serialize)]
#[serde(bound = "F: FftField")]
pub struct Config<F: FftField> {
    /// WHIR config for the blinded witness polynomial commitment.
    pub blinded_commitment: whir::Config<F>,
    /// WHIR config for blinding polynomials (ℓ+1 variables, batch_size = N×(μ+1)).
    pub blinding_commitment: whir::Config<F>,
}

/// Commitment produced by the ZK WHIR commit phase.
///
/// Contains one commitment per blinded witness polynomial (f̂ᵢ = fᵢ + mskᵢ)
/// and a single batch commitment for all blinding polynomials.
pub struct Commitment<F: FftField> {
    pub(super) f_hat: Vec<whir::Commitment<F>>,
    pub(super) blinding: whir::Commitment<F>,
}

impl<F: FftField> Commitment<F> {
    /// References to the blinded polynomial commitments [[f̂₁]], ..., [[f̂ₙ]].
    pub fn f_hat_commitments(&self) -> Vec<&whir::Commitment<F>> {
        self.f_hat.iter().collect()
    }

    /// Reference to the blinding polynomial commitment.
    pub fn blinding_commitment(&self) -> &whir::Commitment<F> {
        &self.blinding
    }
}

impl<F: FftField + FieldWithSize> Config<F> {
    /// Build a ZK WHIR config with automatically-derived blinding parameters.
    ///
    /// Constructs the blinded commitment config from `main_mv_params` and
    /// `main_whir_params`, then derives the blinding polynomial config
    /// automatically. The blinding config inherits `security_level`,
    /// `soundness_type`, `starting_log_inv_rate`, and `hash_id` from the
    /// main parameters, sets `pow_bits` to 0, and computes `num_variables`
    /// and `batch_size` from the derived ZK parameters.
    pub fn new(
        main_mv_params: MultivariateParameters<F>,
        main_whir_params: &ProtocolParameters,
        blinding_folding_factor: FoldingFactor,
        num_polynomials: usize,
    ) -> Self {
        let blinded_commitment = whir::Config::new(main_mv_params, main_whir_params);
        let num_blinding_variables = Self::compute_num_blinding_variables(&blinded_commitment);
        let num_witness_variables = blinded_commitment.initial_num_variables();
        let blinding_mv_params = MultivariateParameters::new(num_blinding_variables + 1);
        let blinding_whir_params = ProtocolParameters {
            initial_statement: true,
            security_level: main_whir_params.security_level,
            pow_bits: 0,
            folding_factor: blinding_folding_factor,
            soundness_type: SoundnessType::ConjectureList,
            starting_log_inv_rate: main_whir_params.starting_log_inv_rate,
            batch_size: num_polynomials * (num_witness_variables + 1),
            hash_id: main_whir_params.hash_id,
        };
        let blinding_commitment = whir::Config::new(blinding_mv_params, &blinding_whir_params);
        Self {
            blinded_commitment,
            blinding_commitment,
        }
    }

    /// Compute the number of blinding variables (ℓ) from the blinded commitment config.
    ///
    /// ℓ is chosen such that 2^ℓ > conservative query upper bound.
    fn compute_num_blinding_variables(blinded: &whir::Config<F>) -> usize {
        let num_witness_variables = blinded.initial_num_variables();
        let folding_factor_size = 1 << blinded.initial_sumcheck.num_rounds;
        let initial_query_count = blinded
            .round_configs
            .first()
            .map_or(blinded.initial_committer.in_domain_samples, |r| {
                r.irs_committer.in_domain_samples
            });

        let query_upper_bound =
            2 * folding_factor_size * initial_query_count + 4 * num_witness_variables + 10;
        let num_blinding_variables = (query_upper_bound as f64).log2().ceil() as usize;
        assert!(
            num_blinding_variables < num_witness_variables,
            "ZK requires ℓ < μ (ℓ={num_blinding_variables}, μ={num_witness_variables}). \
             Increase num_variables or lower security_level/queries. \
             (q_ub={query_upper_bound}, k={folding_factor_size}, q1={initial_query_count})"
        );
        num_blinding_variables
    }
}

impl<F: FftField> Config<F> {
    /// Number of variables in the witness polynomial (μ in the paper).
    pub fn num_witness_variables(&self) -> usize {
        self.blinded_commitment.initial_num_variables()
    }

    /// Number of variables for blinding polynomials (ℓ in the paper).
    pub fn num_blinding_variables(&self) -> usize {
        self.blinding_commitment.initial_num_variables() - 1
    }

    /// Interleaving depth of the initial IRS commitment (= 2^folding_factor).
    pub(super) fn interleaving_depth(&self) -> usize {
        self.blinded_commitment.initial_committer.interleaving_depth
    }

    /// Generator ω of the full NTT domain (size = num_rows × interleaving_depth).
    pub(super) fn omega_full(&self) -> F::BasePrimeField {
        let num_rows = self.blinded_commitment.initial_committer.num_rows();
        let full_domain_size = num_rows * self.interleaving_depth();
        crate::algebra::ntt::generator(full_domain_size)
            .expect("Full IRS domain should have primitive root")
    }

    /// Sub-domain generator (ω_sub = ω^interleaving_depth).
    fn omega_sub(&self) -> F::BasePrimeField {
        self.blinded_commitment.initial_committer.generator()
    }

    /// ζ = ω^num_rows — the interleaving_depth-th root of unity.
    pub(super) fn zeta(&self) -> F::BasePrimeField {
        let num_rows = self.blinded_commitment.initial_committer.num_rows();
        self.omega_full().pow([num_rows as u64])
    }

    /// Precomputed sub-domain powers [1, ω_sub, ω_sub², ..., ω_sub^(num_rows-1)].
    pub(super) fn omega_powers(&self) -> Vec<F::BasePrimeField> {
        let num_rows = self.blinded_commitment.initial_committer.num_rows();
        crate::algebra::geometric_sequence(self.omega_sub(), num_rows)
    }

    /// Find the index of `alpha_base` in the sub-domain powers.
    #[inline]
    pub(super) fn query_index(
        &self,
        alpha_base: F::BasePrimeField,
        omega_powers: &[F::BasePrimeField],
    ) -> usize {
        omega_powers
            .iter()
            .position(|&p| p == alpha_base)
            .expect("Query point must be in IRS domain")
    }

    /// Compute the k coset gamma points for a query at `alpha_base`,
    /// lifted to the extension field via the blinded commitment embedding.
    pub(super) fn coset_gammas(
        &self,
        alpha_base: F::BasePrimeField,
        omega_powers: &[F::BasePrimeField],
    ) -> Vec<F> {
        use crate::algebra::embedding::Embedding;
        let embedding = self.blinded_commitment.embedding();
        let idx = self.query_index(alpha_base, omega_powers);
        let omega_full = self.omega_full();
        let zeta = self.zeta();
        let coset_offset = omega_full.pow([idx as u64]);
        let interleaving_depth = self.interleaving_depth();
        (0..interleaving_depth)
            .map(|coset_elem_idx| {
                let gamma_base = coset_offset * zeta.pow([coset_elem_idx as u64]);
                embedding.map(gamma_base)
            })
            .collect()
    }

    /// Compute all gamma points for a set of query points (flat list).
    pub(super) fn all_gammas(&self, query_points: &[F::BasePrimeField]) -> Vec<F> {
        let omega_powers = self.omega_powers();
        query_points
            .iter()
            .flat_map(|&alpha| self.coset_gammas(alpha, &omega_powers))
            .collect()
    }
}

/// Witness: contains commitment witnesses for all ZK components.
///
/// Produced by [`Config::commit`] and consumed by [`Config::prove`].
#[derive(Clone)]
pub struct Witness<F: FftField> {
    /// Witnesses for [[f̂₁]] = [[f₁ + msk₁]], ..., [[fₙ]] = [[fₙ + mskₙ]] in main WHIR
    pub(super) f_hat_witnesses: Vec<irs_commit::Witness<F::BasePrimeField, F>>,

    /// Single batch witness for all blinding polynomials [[M, ĝ₁, ..., ĝμ]]
    /// committed via blinding_poly_commitment config with batch_size = μ+1
    pub(super) blinding_witness: irs_commit::Witness<F::BasePrimeField, F>,

    /// Blinding data for each polynomial
    pub(super) blinding_polynomials: Vec<BlindingPolynomials<F>>,

    /// Base-field M polynomials (for blinding WHIR prove)
    pub(super) m_polys_base: Vec<CoefficientList<F::BasePrimeField>>,

    /// Base-field embedded ĝⱼ polynomials (for blinding WHIR prove)
    pub(super) g_hats_embedded_bases: Vec<Vec<CoefficientList<F::BasePrimeField>>>,
}

// Blinding Polynomials
/// Random blinding polynomials sampled before the witness polynomial.
///
/// For each committed polynomial, these provide the ZK blinding:
/// msk (masking), g₀ (initial blinding), M (combined), and ĝ₁..ĝμ (per-round blinding).
///
/// All coefficients live in the base prime field. This is required because these
/// polynomials are committed via the base-field IRS commitment scheme.
#[derive(Clone)]
pub struct BlindingPolynomials<F: FftField> {
    pub(super) msk: CoefficientList<F::BasePrimeField>,
    pub(super) g0_hat: CoefficientList<F::BasePrimeField>,
    pub(super) m_poly: CoefficientList<F::BasePrimeField>,
    pub(super) g_hats: Vec<CoefficientList<F::BasePrimeField>>,
}

impl<F: FftField> BlindingPolynomials<F> {
    pub(super) fn sample<R: RngCore + CryptoRng>(
        rng: &mut R,
        num_blinding_variables: usize,
        num_witness_variables: usize,
    ) -> Self {
        let blinding_poly_size = 1 << num_blinding_variables;
        let m_poly_size = 1 << (num_blinding_variables + 1);

        let msk_coeffs: Vec<F::BasePrimeField> = (0..blinding_poly_size)
            .map(|_| F::BasePrimeField::rand(rng))
            .collect();
        let msk = CoefficientList::new(msk_coeffs.clone());

        let g0_coeffs: Vec<F::BasePrimeField> = (0..blinding_poly_size)
            .map(|_| F::BasePrimeField::rand(rng))
            .collect();
        let g0 = CoefficientList::new(g0_coeffs.clone());

        let mut m_coeffs = vec![F::BasePrimeField::ZERO; m_poly_size];
        for (coeff_idx, (&g0_coeff, &msk_coeff)) in
            g0_coeffs.iter().zip(msk_coeffs.iter()).enumerate()
        {
            m_coeffs[2 * coeff_idx] = g0_coeff;
            m_coeffs[2 * coeff_idx + 1] = msk_coeff;
        }
        let m_poly = CoefficientList::new(m_coeffs);

        let g_hats = (0..num_witness_variables)
            .map(|_| {
                let coeffs = (0..blinding_poly_size)
                    .map(|_| F::BasePrimeField::rand(rng))
                    .collect();
                CoefficientList::new(coeffs)
            })
            .collect();

        Self {
            msk,
            g0_hat: g0,
            m_poly,
            g_hats,
        }
    }

    /// Batch-evaluate all blinding polynomials at multiple extension-field gamma
    /// points using mixed-field Horner evaluation.
    ///
    /// For each gamma, evaluates msk, g₀, and all ĝⱼ (base-field coefficients
    /// evaluated at an extension-field point) in a single pass per gamma point.
    ///
    /// Returns a Vec of `BlindingEvaluations` (one per gamma point), in the same
    /// order as the input gammas.
    pub(super) fn batch_evaluate(
        &self,
        gammas: &[F],
        masking_challenge: F,
    ) -> Vec<utils::BlindingEvaluations<F>> {
        use crate::algebra::{embedding::Basefield, mixed_univariate_evaluate};

        let embedding = Basefield::<F>::new();

        // Evaluate all blinding polynomials at a single gamma point.
        let eval_at = |&gamma: &F| -> utils::BlindingEvaluations<F> {
            let msk_val = mixed_univariate_evaluate(&embedding, self.msk.coeffs(), gamma);
            let g0_val = mixed_univariate_evaluate(&embedding, self.g0_hat.coeffs(), gamma);
            let m_eval = g0_val - masking_challenge * msk_val;
            let g_hat_evals = self
                .g_hats
                .iter()
                .map(|g_hat| mixed_univariate_evaluate(&embedding, g_hat.coeffs(), gamma))
                .collect();
            utils::BlindingEvaluations {
                gamma,
                m_eval,
                g_hat_evals,
            }
        };

        // Parallelize across gamma points (typically q×k, often hundreds).
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            gammas.par_iter().map(eval_at).collect()
        }
        #[cfg(not(feature = "parallel"))]
        {
            gammas.iter().map(eval_at).collect()
        }
    }
}

#[cfg(test)]
mod tests {

    use ark_std::rand::{rngs::StdRng, SeedableRng};

    use super::*;
    use crate::{
        algebra::{
            fields::{Field64, Field64_2},
            polynomials::{CoefficientList, MultilinearPoint},
            Weights,
        },
        hash,
        parameters::{FoldingFactor, MultivariateParameters, ProtocolParameters, SoundnessType},
        transcript::{codecs::Empty, DomainSeparator, ProverState, VerifierState},
    };

    /// Field type used in the tests.
    type F = Field64;

    /// Extension field type used in the tests.
    type EF = Field64_2;

    /// Run a complete zkWHIR proof lifecycle: sample preprocessing, commit, prove, verify.
    ///
    /// This function:
    /// - builds N multilinear polynomials with the specified number of variables,
    /// - constructs a `whir_zk::Config` bundling main + blinding WHIR configs,
    /// - commits (producing f̂ᵢ = fᵢ + mskᵢ and blinding commitments),
    /// - generates a ZK proof, then verifies it.
    fn make_whir_zk_things(
        num_variables: usize,
        folding_factor: FoldingFactor,
        num_points: usize,
        num_polynomials: usize,
        soundness_type: SoundnessType,
        pow_bits: usize,
    ) {
        assert!(num_polynomials >= 1, "need at least 1 polynomial");
        let num_coeffs = 1 << num_variables;
        let mut rng = StdRng::seed_from_u64(12345);

        // ── Main WHIR parameters ──
        let mv_params = MultivariateParameters::new(num_variables);
        let whir_params = ProtocolParameters {
            initial_statement: true,
            security_level: 32,
            pow_bits,
            folding_factor,
            soundness_type,
            starting_log_inv_rate: 1,
            batch_size: 1,
            hash_id: hash::SHA2,
        };

        // ── Build ZK config with auto-derived blinding parameters ──
        let zk_config = Config::new(
            mv_params,
            &whir_params,
            FoldingFactor::Constant(1),
            num_polynomials,
        );
        eprintln!(
            "ZK params: num_blinding_variables={}, num_witness_variables={}, num_polynomials={}",
            zk_config.num_blinding_variables(),
            zk_config.num_witness_variables(),
            num_polynomials
        );
        eprintln!("{}", zk_config.blinded_commitment);

        // ── Sample N random polynomials ──
        let mut polynomials: Vec<CoefficientList<F>> = Vec::with_capacity(num_polynomials);
        for poly_idx in 0..num_polynomials {
            // Each polynomial has distinct coefficients
            let coeffs: Vec<F> = (0..num_coeffs)
                .map(|coeff_idx| F::from((poly_idx * num_coeffs + coeff_idx + 1) as u64))
                .collect();
            polynomials.push(CoefficientList::new(coeffs));
        }

        // ── Create evaluation constraints ──
        // evaluations layout: row-major [w₀_p₀, w₀_p₁, ..., w₁_p₀, ...]
        let mut weights = Vec::new();
        let mut evaluations = Vec::new();
        for _ in 0..num_points {
            let point = MultilinearPoint::rand(&mut rng, num_variables);
            weights.push(Weights::evaluation(point.clone()));
            for poly in &polynomials {
                evaluations
                    .push(poly.mixed_evaluate(zk_config.blinded_commitment.embedding(), &point));
            }
        }
        let weight_refs: Vec<&Weights<EF>> = weights.iter().collect();
        let polynomial_refs: Vec<&CoefficientList<F>> = polynomials.iter().collect();

        // ── Set up Fiat-Shamir transcript ──
        let ds = DomainSeparator::protocol(&zk_config)
            .session(&format!("ZK Test at {}:{}", file!(), line!()))
            .instance(&Empty);
        let mut prover_state = ProverState::new_std(&ds);

        // ── ZK commitment: commit to f̂ᵢ = fᵢ + mskᵢ, plus blinding polynomials ──
        let zk_witness = zk_config.commit(&mut prover_state, &polynomial_refs);

        // ── ZK proof: prove knowledge of {fᵢ} via blinded virtual oracle ──
        let (_point, _evals) = zk_config.prove(
            &mut prover_state,
            &polynomial_refs,
            &zk_witness,
            &weight_refs,
            &evaluations,
        );
        let proof = prover_state.proof();
        let mut verifier_state = VerifierState::new_std(&ds, &proof);

        // ── Receive commitments from transcript (mirrors commit order) ──
        let commitment = zk_config
            .receive_commitments(&mut verifier_state, num_polynomials)
            .unwrap();

        // ── Verify ZK proof ──
        let verify_result =
            zk_config.verify(&mut verifier_state, &commitment, &weight_refs, &evaluations);
        assert!(
            verify_result.is_ok(),
            "ZK verification failed (N={num_polynomials}): {:?}",
            verify_result.err()
        );
    }

    #[test]
    fn test_whir_zk_basic() {
        // ZK requires ℓ < μ. With security_level=32, the query count drives ℓ ≈ 8–10,
        // so num_variables (= μ) must be large enough.
        let configs: &[(usize, usize)] = &[
            // (num_variables, folding_factor)
            (10, 2),
            (12, 2),
            (12, 3),
            (12, 4),
        ];
        let num_points = [0, 1, 2];

        for &(num_variable, folding_factor) in configs {
            for num_points in num_points {
                eprintln!();
                dbg!(num_variable, folding_factor, num_points);

                make_whir_zk_things(
                    num_variable,
                    FoldingFactor::Constant(folding_factor),
                    num_points,
                    1, // single polynomial
                    SoundnessType::ConjectureList,
                    0,
                );
            }
        }
    }

    #[test]
    fn test_whir_zk_with_pow() {
        // Test ZK with proof-of-work enabled
        make_whir_zk_things(
            12,
            FoldingFactor::Constant(2),
            2,
            1,
            SoundnessType::ConjectureList,
            5,
        );
    }

    #[test]
    fn test_whir_zk_soundness_types() {
        // Test ZK across different soundness types
        let soundness_types = [
            SoundnessType::ConjectureList,
            SoundnessType::ProvableList,
            SoundnessType::UniqueDecoding,
        ];

        for soundness_type in soundness_types {
            eprintln!();
            dbg!(soundness_type);
            make_whir_zk_things(12, FoldingFactor::Constant(2), 1, 1, soundness_type, 0);
        }
    }

    #[test]
    fn test_whir_zk_mixed_folding() {
        // Test ZK with mixed folding factors
        make_whir_zk_things(
            12,
            FoldingFactor::ConstantFromSecondRound(3, 3),
            1,
            1,
            SoundnessType::ConjectureList,
            0,
        );
    }

    #[test]
    fn test_whir_zk_multi_polynomial() {
        // Test ZK with multiple polynomials (batched proving/verification)
        let configs: &[(usize, usize, usize, usize)] = &[
            // (num_variables, folding_factor, num_points, num_polynomials)
            (10, 2, 1, 2), // 2 polynomials, 1 constraint
            (10, 2, 2, 2), // 2 polynomials, 2 constraints
            (12, 2, 1, 3), // 3 polynomials, 1 constraint
            (12, 3, 2, 2), // 2 polynomials, 2 constraints, larger folding
        ];

        for &(num_variables, folding_factor, num_points, num_polynomials) in configs {
            eprintln!();
            dbg!(num_variables, folding_factor, num_points, num_polynomials);

            make_whir_zk_things(
                num_variables,
                FoldingFactor::Constant(folding_factor),
                num_points,
                num_polynomials,
                SoundnessType::ConjectureList,
                0,
            );
        }
    }
}
