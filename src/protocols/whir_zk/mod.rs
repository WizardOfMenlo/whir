mod committer;
mod prover;
mod utils;
mod verifier;

use ark_ff::{FftField, PrimeField};
use serde::{Deserialize, Serialize};

pub use self::committer::{Commitment, Witness};
use crate::{
    algebra::embedding::{Embedding, Identity},
    parameters::{FoldingFactor, MultivariateParameters, ProtocolParameters, SoundnessType},
    protocols::whir,
};

/// Policy inputs for `ell` computation.
///
/// Leakage upper bound used by the current sizing rule:
/// `q_ub = k1*q_delta_1 + k2*q_delta_2 + (d+1)*mu + t1 + t2`.
#[derive(Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Debug)]
pub struct BlindingSizePolicy {
    /// Query count for witness-side WHIR first-round leakage term.
    pub q_delta_1: usize,
    /// Query count for blinding-side WHIR first-round leakage term.
    pub q_delta_2: usize,
    /// Last-round clear leakage budget for witness-side WHIR.
    pub t1: usize,
    /// Last-round clear leakage budget for blinding-side WHIR.
    pub t2: usize,
    /// Degree `d` of each sumcheck round polynomial in the leakage term `(d+1)*mu`.
    pub sumcheck_round_degree: usize,
}

/// ZK WHIR configuration: witness-side WHIR + blinding-side WHIR.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize, Debug)]
#[serde(bound = "")]
pub struct Config<F: FftField + PrimeField> {
    pub blinded_commitment: whir::Config<F, Identity<F>>,
    pub blinding_commitment: whir::Config<F, Identity<F>>,
}

impl<F: FftField + PrimeField> Config<F> {
    fn default_blinding_size_policy(main_whir_params: &ProtocolParameters) -> BlindingSizePolicy {
        let protocol_security_level_main = main_whir_params
            .security_level
            .saturating_sub(main_whir_params.pow_bits);
        let q_delta_1 = whir::Config::<F>::queries(
            main_whir_params.soundness_type,
            protocol_security_level_main,
            main_whir_params.starting_log_inv_rate,
        );
        let q_delta_2 = whir::Config::<F>::queries(
            SoundnessType::ConjectureList,
            main_whir_params.security_level,
            main_whir_params.starting_log_inv_rate,
        );

        // Default send-in-clear thresholds match query complexities.
        BlindingSizePolicy {
            q_delta_1,
            q_delta_2,
            t1: q_delta_1,
            t2: q_delta_2,
            // For PCS, round polynomials are cubic (d=3), giving 4*mu.
            sumcheck_round_degree: 3,
        }
    }

    /// Build a zkWHIR config from the given WHIR parameters.
    pub fn new(
        main_mv_params: MultivariateParameters<F>,
        main_whir_params: &ProtocolParameters,
        blinding_folding_factor: FoldingFactor,
        num_polynomials: usize,
    ) -> Self {
        let size_policy = Self::default_blinding_size_policy(main_whir_params);
        Self::new_with_blinding_size_policy(
            main_mv_params,
            main_whir_params,
            blinding_folding_factor,
            num_polynomials,
            size_policy,
        )
    }

    /// Build a zkWHIR config with an explicit blinding size policy.
    ///
    /// * `blinding_folding_factor` — folding factor for the blinding-side WHIR.
    /// * `num_polynomials` — number of polynomials that will be committed.
    /// * `size_policy` — controls the blinding polynomial size `ell`.
    pub fn new_with_blinding_size_policy(
        main_mv_params: MultivariateParameters<F>,
        main_whir_params: &ProtocolParameters,
        blinding_folding_factor: FoldingFactor,
        num_polynomials: usize,
        size_policy: BlindingSizePolicy,
    ) -> Self {
        let blinded_commitment =
            whir::Config::new(main_mv_params, main_whir_params).into_identity();
        let num_witness_variables = blinded_commitment.initial_num_variables();
        let blinding_first_round_interleaving_depth = 1usize << blinding_folding_factor.at_round(0);
        let num_blinding_variables = Self::compute_num_blinding_variables(
            &blinded_commitment,
            blinding_first_round_interleaving_depth,
            size_policy,
        );

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
        let blinding_commitment =
            whir::Config::new(blinding_mv_params, &blinding_whir_params).into_identity();

        Self {
            blinded_commitment,
            blinding_commitment,
        }
    }

    fn compute_num_blinding_variables(
        blinded: &whir::Config<F, Identity<F>>,
        blinding_first_round_interleaving_depth: usize,
        size_policy: BlindingSizePolicy,
    ) -> usize {
        // Doc formula:
        // q_ub = k1*q(delta1) + k2*q(delta2) + (d+1)*mu + T1 + T2,
        // choose smallest ell with 2^ell > q_ub.
        let num_witness_variables = blinded.initial_num_variables();
        assert!(
            size_policy.t1 >= size_policy.q_delta_1,
            "invalid blinding size policy: T1 must satisfy T1 >= q(delta1)"
        );
        assert!(
            size_policy.t2 >= size_policy.q_delta_2,
            "invalid blinding size policy: T2 must satisfy T2 >= q(delta2)"
        );
        let k1 = 1usize << blinded.initial_sumcheck.num_rounds;
        let k2 = blinding_first_round_interleaving_depth;
        let sumcheck_coeff_leakage = size_policy
            .sumcheck_round_degree
            .saturating_add(1)
            .saturating_mul(num_witness_variables);
        let query_upper_bound = k1
            .saturating_mul(size_policy.q_delta_1)
            .saturating_add(k2.saturating_mul(size_policy.q_delta_2))
            .saturating_add(size_policy.t1)
            .saturating_add(size_policy.t2)
            .saturating_add(sumcheck_coeff_leakage);

        let num_blinding_variables = (usize::BITS - query_upper_bound.leading_zeros()) as usize;
        assert!(
            num_blinding_variables < num_witness_variables,
            "blinding variables ({num_blinding_variables}) must be fewer than \
             witness variables ({num_witness_variables})"
        );
        debug_assert!(
            (1usize << num_blinding_variables) > query_upper_bound,
            "2^ell ({}) must exceed query upper bound ({query_upper_bound})",
            1usize << num_blinding_variables
        );
        num_blinding_variables
    }

    /// Number of variables in the witness polynomial (`μ`).
    pub fn num_witness_variables(&self) -> usize {
        self.blinded_commitment.initial_num_variables()
    }

    /// Number of blinding variables (`ℓ`).
    pub fn num_blinding_variables(&self) -> usize {
        self.blinding_commitment.initial_num_variables() - 1
    }

    /// Interleaving depth of the initial IRS commitment (= 2^folding_factor).
    pub(crate) const fn interleaving_depth(&self) -> usize {
        self.blinded_commitment.initial_committer.interleaving_depth
    }

    /// Generator ω of the full NTT domain (size = num_rows × interleaving_depth).
    pub(crate) fn omega_full(&self) -> F {
        let num_rows = self.blinded_commitment.initial_committer.num_rows();
        let full_domain_size = num_rows * self.interleaving_depth();
        crate::algebra::ntt::generator(full_domain_size)
            .expect("full IRS domain should have primitive root")
    }

    /// Sub-domain generator (ω_sub = ω^interleaving_depth).
    fn omega_sub(&self) -> F {
        self.blinded_commitment.initial_committer.generator()
    }

    /// ζ = ω^num_rows — the interleaving_depth-th root of unity.
    pub(crate) fn zeta(&self) -> F {
        let num_rows = self.blinded_commitment.initial_committer.num_rows();
        self.omega_full().pow([num_rows as u64])
    }

    /// Precomputed sub-domain powers [1, ω_sub, ω_sub², ..., ω_sub^(num_rows-1)].
    pub(crate) fn omega_powers(&self) -> Vec<F> {
        let num_rows = self.blinded_commitment.initial_committer.num_rows();
        crate::algebra::geometric_sequence(self.omega_sub(), num_rows)
    }

    /// Find the index of `alpha_base` in the sub-domain powers.
    pub(crate) fn query_index(alpha_base: F, omega_powers: &[F]) -> usize {
        omega_powers
            .iter()
            .position(|&p| p == alpha_base)
            .expect("query point must be in IRS domain")
    }

    /// Compute all gamma points for a set of query points (flat list).
    pub(crate) fn all_gammas(&self, query_points: &[F]) -> Vec<F> {
        let omega_powers = self.omega_powers();
        let interleaving_depth = self.interleaving_depth();
        let omega_full = self.omega_full();
        let zeta_powers = crate::algebra::geometric_sequence(self.zeta(), interleaving_depth);
        let embedding = self.blinded_commitment.embedding();

        let mut gammas = Vec::with_capacity(query_points.len() * interleaving_depth);
        for &alpha in query_points {
            let idx = Self::query_index(alpha, &omega_powers);
            let coset_offset = omega_full.pow([idx as u64]);
            for &zp in &zeta_powers {
                gammas.push(embedding.map(coset_offset * zp));
            }
        }
        gammas
    }
}

#[cfg(test)]
mod tests {
    use ark_ff::Field;

    use super::*;
    use crate::{
        algebra::{
            fields::Field64,
            linear_form::{Evaluate, LinearForm, MultilinearExtension},
            MultilinearPoint,
        },
        hash,
        parameters::{FoldingFactor, MultivariateParameters, ProtocolParameters, SoundnessType},
        transcript::{codecs::Empty, DomainSeparator, ProverState, VerifierState},
    };

    type F = Field64;

    fn make_test_blinding_size_policy() -> BlindingSizePolicy {
        BlindingSizePolicy {
            q_delta_1: 4,
            q_delta_2: 4,
            t1: 4,
            t2: 4,
            sumcheck_round_degree: 3,
        }
    }

    const TEST_NUM_VARIABLES: usize = 8;
    const TEST_NUM_COEFFS: usize = 1 << TEST_NUM_VARIABLES;

    fn make_test_config(num_polynomials: usize) -> Config<F> {
        let mv_params = MultivariateParameters::new(TEST_NUM_VARIABLES);
        let whir_params = ProtocolParameters {
            initial_statement: true,
            security_level: 16,
            pow_bits: 0,
            folding_factor: FoldingFactor::Constant(2),
            soundness_type: SoundnessType::ConjectureList,
            starting_log_inv_rate: 1,
            batch_size: 1,
            hash_id: hash::SHA2,
        };
        Config::<F>::new_with_blinding_size_policy(
            mv_params,
            &whir_params,
            FoldingFactor::Constant(2),
            num_polynomials,
            make_test_blinding_size_policy(),
        )
    }

    fn linear_form_refs(forms: &[Box<dyn LinearForm<F>>]) -> Vec<&dyn LinearForm<F>> {
        forms
            .iter()
            .map(|w| w.as_ref() as &dyn LinearForm<F>)
            .collect()
    }

    fn make_two_poly_vectors(mul: u64, add: u64) -> (Vec<F>, Vec<F>) {
        let v0 = vec![F::ONE; TEST_NUM_COEFFS];
        let v1 = (0..TEST_NUM_COEFFS)
            .map(|i| F::from((i as u64).wrapping_mul(mul).wrapping_add(add)))
            .collect();
        (v0, v1)
    }

    fn compute_evaluations(
        params: &Config<F>,
        forms: &[&MultilinearExtension<F>],
        vectors: &[&[F]],
    ) -> Vec<F> {
        let embedding = params.blinded_commitment.embedding();
        let mut evals = Vec::with_capacity(forms.len() * vectors.len());
        for &form in forms {
            for &vector in vectors {
                evals.push(form.evaluate(embedding, vector));
            }
        }
        evals
    }

    fn prove_and_verify(
        params: &Config<F>,
        vectors: &[&[F]],
        linear_form_refs: &[&dyn LinearForm<F>],
        evaluations: &[F],
        session_tag: &str,
    ) {
        let tag = session_tag.to_owned();
        let ds = DomainSeparator::protocol(params)
            .session(&tag)
            .instance(&Empty);
        let mut prover_state = ProverState::new_std(&ds);
        let witness = params.commit(&mut prover_state, vectors);
        params.prove(
            &mut prover_state,
            vectors,
            witness,
            linear_form_refs,
            evaluations,
        );
        let proof = prover_state.proof();
        let mut verifier_state = VerifierState::new_std(&ds, &proof);
        let commitment = params
            .receive_commitments(&mut verifier_state, vectors.len())
            .expect("receive commitments");
        params
            .verify(
                &mut verifier_state,
                linear_form_refs,
                evaluations,
                &commitment,
            )
            .expect("verify zk wrapper");
    }

    #[test]
    fn test_whir_zk_stage1_single_poly() {
        let mut rng = ark_std::test_rng();
        let params = make_test_config(1);

        let vector = vec![F::ONE; TEST_NUM_COEFFS];
        let point = MultilinearPoint::rand(&mut rng, TEST_NUM_VARIABLES);
        let form = MultilinearExtension { point: point.0 };
        let evaluation = form.evaluate(params.blinded_commitment.embedding(), &vector);
        let forms: Vec<Box<dyn LinearForm<F>>> = vec![Box::new(form)];
        let refs = linear_form_refs(&forms);

        prove_and_verify(
            &params,
            &[&vector],
            &refs,
            &[evaluation],
            &format!("zk-stage1 {}:{}", file!(), line!()),
        );
    }

    #[test]
    fn test_whir_zk_stage1_multi_poly_multi_query() {
        let mut rng = ark_std::test_rng();
        let params = make_test_config(2);

        let (v0, v1) = make_two_poly_vectors(17, 3);
        let vectors = [&v0[..], &v1[..]];

        let p0 = MultilinearPoint::rand(&mut rng, TEST_NUM_VARIABLES);
        let p1 = MultilinearPoint::rand(&mut rng, TEST_NUM_VARIABLES);
        let f0 = MultilinearExtension { point: p0.0 };
        let f1 = MultilinearExtension { point: p1.0 };
        let evaluations = compute_evaluations(&params, &[&f0, &f1], &vectors);

        let forms: Vec<Box<dyn LinearForm<F>>> = vec![Box::new(f0), Box::new(f1)];
        let refs = linear_form_refs(&forms);

        prove_and_verify(
            &params,
            &vectors,
            &refs,
            &evaluations,
            &format!("zk-stage1-multi {}:{}", file!(), line!()),
        );
    }

    #[test]
    /// Verification must reject when the public evaluations are tampered with.
    /// Both `Err` and a panic (from debug transcript checks) count as rejection.
    fn test_whir_zk_stage1_rejects_wrong_evaluations() {
        let mut rng = ark_std::test_rng();
        let params = make_test_config(2);

        let (v0, v1) = make_two_poly_vectors(29, 7);
        let vectors = [&v0[..], &v1[..]];

        let p0 = MultilinearPoint::rand(&mut rng, TEST_NUM_VARIABLES);
        let p1 = MultilinearPoint::rand(&mut rng, TEST_NUM_VARIABLES);
        let f0 = MultilinearExtension { point: p0.0 };
        let f1 = MultilinearExtension { point: p1.0 };
        let evaluations = compute_evaluations(&params, &[&f0, &f1], &vectors);

        let forms: Vec<Box<dyn LinearForm<F>>> = vec![Box::new(f0), Box::new(f1)];
        let refs = linear_form_refs(&forms);

        let ds = DomainSeparator::protocol(&params)
            .session(&format!("zk-stage1-negative {}:{}", file!(), line!()))
            .instance(&Empty);
        let mut prover_state = ProverState::new_std(&ds);
        let witness = params.commit(&mut prover_state, &vectors);
        params.prove(&mut prover_state, &vectors, witness, &refs, &evaluations);

        let proof = prover_state.proof();
        let mut wrong_evaluations = evaluations.clone();
        wrong_evaluations[0] += F::ONE;

        let verify_outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut verifier_state = VerifierState::new_std(&ds, &proof);
            let commitment = params
                .receive_commitments(&mut verifier_state, 2)
                .expect("receive commitments");
            params.verify(&mut verifier_state, &refs, &wrong_evaluations, &commitment)
        }));
        if let Ok(result) = verify_outcome {
            assert!(
                result.is_err(),
                "verification should reject wrong public evaluations"
            );
        }
    }

    #[test]
    fn test_whir_zk_stage1_rejects_tampered_proof() {
        let mut rng = ark_std::test_rng();
        let params = make_test_config(2);

        let (v0, v1) = make_two_poly_vectors(13, 11);
        let vectors = [&v0[..], &v1[..]];

        let p0 = MultilinearPoint::rand(&mut rng, TEST_NUM_VARIABLES);
        let p1 = MultilinearPoint::rand(&mut rng, TEST_NUM_VARIABLES);
        let f0 = MultilinearExtension { point: p0.0 };
        let f1 = MultilinearExtension { point: p1.0 };
        let evaluations = compute_evaluations(&params, &[&f0, &f1], &vectors);

        let forms: Vec<Box<dyn LinearForm<F>>> = vec![Box::new(f0), Box::new(f1)];
        let refs = linear_form_refs(&forms);

        let ds = DomainSeparator::protocol(&params)
            .session(&format!("zk-stage1-tamper {}:{}", file!(), line!()))
            .instance(&Empty);
        let mut prover_state = ProverState::new_std(&ds);
        let witness = params.commit(&mut prover_state, &vectors);
        params.prove(&mut prover_state, &vectors, witness, &refs, &evaluations);

        let mut proof = prover_state.proof();
        if let Some(last) = proof.narg_string.last_mut() {
            *last ^= 1;
        } else {
            panic!("expected non-empty proof transcript");
        }

        let verify_outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut verifier_state = VerifierState::new_std(&ds, &proof);
            let commitment = params
                .receive_commitments(&mut verifier_state, 2)
                .expect("receive commitments");
            params.verify(&mut verifier_state, &refs, &evaluations, &commitment)
        }));
        if let Ok(result) = verify_outcome {
            assert!(
                result.is_err(),
                "verification should reject tampered proof bytes"
            );
        } else {
            // In debug transcript mode, tampering may trigger an interaction-pattern panic
            // before returning `Err`, which is still a valid rejection.
        }
    }

    /// Soundness exploit: malicious prover generates proof for WRONG evaluation.
    /// A sound PCS must reject; if verify() returns Ok, g_eval freedom lets
    /// the prover forge arbitrary evaluation claims.
    #[test]
    fn test_whir_zk_malicious_prover_wrong_evaluation() {
        let mut rng = ark_std::test_rng();
        let params = make_test_config(1);

        let vector = vec![F::ONE; TEST_NUM_COEFFS];
        let point = MultilinearPoint::rand(&mut rng, TEST_NUM_VARIABLES);
        let form = MultilinearExtension { point: point.0 };
        let correct_evaluation = form.evaluate(params.blinded_commitment.embedding(), &vector);
        let wrong_evaluation = correct_evaluation + F::from(42u64);

        let forms: Vec<Box<dyn LinearForm<F>>> = vec![Box::new(form)];
        let refs = linear_form_refs(&forms);

        let ds = DomainSeparator::protocol(&params)
            .session(&format!("zk-malicious {}:{}", file!(), line!()))
            .instance(&Empty);

        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut prover_state = ProverState::new_std(&ds);
            let witness = params.commit(&mut prover_state, &[&vector]);
            params.prove(
                &mut prover_state,
                &[&vector],
                witness,
                &refs,
                &[wrong_evaluation],
            );

            let proof = prover_state.proof();
            let mut verifier_state = VerifierState::new_std(&ds, &proof);
            let commitment = params
                .receive_commitments(&mut verifier_state, 1)
                .expect("receive commitments");
            params.verify(&mut verifier_state, &refs, &[wrong_evaluation], &commitment)
        }));

        if let Ok(result) = outcome {
            assert!(
                result.is_err(),
                "SOUNDNESS BUG: verifier accepted wrong evaluation from malicious prover \
             (correct={correct_evaluation:?}, claimed={wrong_evaluation:?})"
            );
        }
    }
}
