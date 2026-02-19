use std::borrow::Cow;

use ark_ff::FftField;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::{Config, Witness};
use crate::{
    algebra::linear_form::{Covector, Evaluate, LinearForm},
    hash::Hash,
    protocols::whir_zk::utils::{
        beq_covector_at_gamma, recombine_doc_claim_from_components, BlindingEvaluations,
        BlindingPolynomials,
    },
    transcript::{ProverMessage, VerifierMessage},
};

/// Precompute per-gamma evaluation data for the round-0 blinding consistency check.
///
/// For each gamma, a single `beq_covector` is built and shared across all polynomials
/// (avoiding recomputing the eq-weights tensor product once per polynomial), and its
/// weight vector is kept for inline accumulation of the batched blinding linear form.
///
/// Parallel feature: gammas are evaluated concurrently; the sequential write pass
/// (transcript + accumulation) happens in the caller.
fn precompute_gamma_data<F: FftField>(
    h_gammas: &[F],
    blinding_polynomials: &[BlindingPolynomials<F>],
    masking_challenge: F,
    num_blinding_variables: usize,
) -> Vec<(Vec<F>, Vec<BlindingEvaluations<F>>)> {
    // Embed each polynomial's g_hats once; the result is reused for every gamma.
    let embedded_g_hats_per_poly: Vec<_> = blinding_polynomials
        .iter()
        .map(|p| p.embedded_g_hats())
        .collect();

    let eval_one_gamma = |&gamma: &F| {
        let beq_cov = beq_covector_at_gamma(gamma, masking_challenge, num_blinding_variables);
        let evals = blinding_polynomials
            .iter()
            .zip(embedded_g_hats_per_poly.iter())
            .map(|(bp, eg)| bp.evaluate_with_covector(&beq_cov, eg, gamma))
            .collect();
        (beq_cov.vector, evals)
    };

    #[cfg(feature = "parallel")]
    return h_gammas.par_iter().map(eval_one_gamma).collect();

    #[cfg(not(feature = "parallel"))]
    return h_gammas.iter().map(eval_one_gamma).collect();
}

impl<F: FftField> Config<F> {
    pub fn prove<H, R>(
        &self,
        prover_state: &mut crate::transcript::ProverState<H, R>,
        polynomials: &[&[F::BasePrimeField]],
        witness: &Witness<F>,
        weights: &[&dyn crate::algebra::linear_form::LinearForm<F>],
        evaluations: &[F],
    ) -> (crate::algebra::MultilinearPoint<F>, Vec<F>)
    where
        H: crate::transcript::DuplexSpongeInterface<U = u8>,
        R: ark_std::rand::RngCore + ark_std::rand::CryptoRng,
        F: crate::transcript::Codec<[H::U]>,
        [u8; 32]: crate::transcript::Decoding<[H::U]>,
        crate::transcript::codecs::U64: crate::transcript::Codec<[H::U]>,
        u8: crate::transcript::Decoding<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        assert_eq!(
            self.blinded_commitment.initial_committer.num_vectors, 1,
            "zkWHIR currently expects one vector per commitment"
        );
        assert_eq!(
            polynomials.len(),
            witness.f_hat_witnesses.len(),
            "witness/polynomial length mismatch"
        );
        assert_eq!(
            polynomials.len(),
            witness.f_hat_vectors.len(),
            "masked vector/polynomial length mismatch"
        );
        assert_eq!(
            witness.blinding_vectors.len(),
            self.blinding_commitment.initial_committer.num_vectors,
            "blinding vectors/witness mismatch"
        );
        assert_eq!(
            evaluations.len(),
            weights.len() * polynomials.len(),
            "evaluation matrix must be row-major weights x polynomials"
        );

        // Transcript order for the zkWHIR round-0 stitching checks:
        // 1) sample beta,
        // 2) sample non-zero rho,
        // 3) send g-evaluations used in rho*f + g,
        // 4) sample tau1 and tau2,
        // 5) send raw openings [m(gamma,-rho), g_1(gamma), ..., g_mu(gamma)] for gamma in Gamma,
        // 6) send tau1/tau2-combined claims and tau2-batched h(gamma) claims.
        let blinding_challenge: F = prover_state.verifier_message();
        let masking_challenge: F = prover_state.verifier_message();
        assert_ne!(
            masking_challenge,
            F::ZERO,
            "zkWHIR requires non-zero masking challenge rho"
        );

        let mut modified_evaluations = Vec::with_capacity(evaluations.len());
        let embedding = self.blinded_commitment.embedding();
        for (weight_idx, weight) in weights.iter().enumerate() {
            for (poly_idx, f_hat_vector) in witness.f_hat_vectors.iter().enumerate() {
                let idx = weight_idx * polynomials.len() + poly_idx;
                let f_hat_eval =
                    Covector::from(*weight).evaluate(embedding, f_hat_vector.as_slice());
                // Enforce L = rho * f + g with L instantiated by the committed masked witness f_hat.
                let g_eval = f_hat_eval - masking_challenge * evaluations[idx];
                prover_state.prover_message(&g_eval);
                debug_assert_eq!(masking_challenge * evaluations[idx] + g_eval, f_hat_eval);
                modified_evaluations.push(f_hat_eval);
            }
        }
        let num_witness_vars = self.num_witness_variables();
        let witness_refs: Vec<_> = witness.f_hat_witnesses.iter().collect();
        let initial_in_domain = self
            .blinded_commitment
            .initial_committer
            .open(prover_state, &witness_refs);

        // Doc-faithful Gamma surface: expand each base query alpha_i into coset points
        // alpha_i * Omega_k used by the first folding round.
        let h_gammas = self.all_gammas(&initial_in_domain.points);

        // Sample batching challenges before revealing any raw opening payload.
        let tau1: F = prover_state.verifier_message();
        let tau2: F = prover_state.verifier_message();

        let num_polynomials = polynomials.len();
        let num_blinding_variables = self.num_blinding_variables();

        // Parallel-compute per-gamma evaluation data (one beq_covector shared per gamma).
        // Returns (beq_weights_vec, per_poly_evals) for each gamma, ready for sequential output.
        let all_gamma_data = precompute_gamma_data(
            &h_gammas,
            &witness.blinding_polynomials,
            masking_challenge,
            num_blinding_variables,
        );

        // Stream transcript payload and accumulate claims + batched beq linear form.
        // m_eval = M(gamma, -rho), g_hat_eval_j = g_hat_j(gamma, -rho).
        let mut m_claims = vec![F::ZERO; num_polynomials];
        let mut g_hat_claims_per_poly = vec![vec![F::ZERO; num_witness_vars]; num_polynomials];
        let mut batched_h_claims = vec![F::ZERO; num_polynomials];
        let mut beq_weight_accum = vec![F::ZERO; 1 << (num_blinding_variables + 1)];
        let mut tau2_power = F::ONE;
        for (beq_weights, per_poly_evals) in &all_gamma_data {
            for (poly_idx, eval) in per_poly_evals.iter().enumerate() {
                prover_state.prover_message(&eval.m_eval);
                for g_hat_eval in &eval.g_hat_evals {
                    prover_state.prover_message(g_hat_eval);
                }
                m_claims[poly_idx] += tau2_power * eval.m_eval;
                for (claim, &g_hat_eval) in g_hat_claims_per_poly[poly_idx]
                    .iter_mut()
                    .zip(eval.g_hat_evals.iter())
                {
                    *claim += tau2_power * g_hat_eval;
                }
                batched_h_claims[poly_idx] += tau2_power * eval.compute_h_value(blinding_challenge);
            }
            for (acc, &w) in beq_weight_accum.iter_mut().zip(beq_weights.iter()) {
                *acc += tau2_power * w;
            }
            tau2_power *= tau2;
        }

        let mut combined_doc_claims = Vec::with_capacity(num_polynomials);
        let mut batched_blinding_subproof_claims =
            Vec::with_capacity(num_polynomials * (1 + num_witness_vars));
        for poly_idx in 0..num_polynomials {
            let m_claim = m_claims[poly_idx];
            let g_hat_claims = &g_hat_claims_per_poly[poly_idx];
            batched_blinding_subproof_claims.push(m_claim);
            batched_blinding_subproof_claims.extend_from_slice(g_hat_claims);
            combined_doc_claims.push(recombine_doc_claim_from_components(
                m_claim,
                g_hat_claims,
                tau1,
            ));
        }
        let beq_weights = Covector::new(beq_weight_accum);
        for claim in &combined_doc_claims {
            prover_state.prover_message(claim);
        }
        for h_claim in &batched_h_claims {
            prover_state.prover_message(h_claim);
        }

        let vectors = witness
            .f_hat_vectors
            .iter()
            .map(Vec::as_slice)
            .collect::<Vec<_>>();
        let result = self.blinded_commitment.prove(
            prover_state,
            vectors
                .iter()
                .map(|v| Cow::Borrowed(*v))
                .collect::<Vec<_>>(),
            witness_refs.iter().map(|w| Cow::Borrowed(*w)).collect(),
            weights
                .iter()
                .map(|w| {
                    // Materialize the linear form into a Covector, but preserve the
                    // original `deferred` flag so the verifier follows the same path.
                    // Changed after the memory opt PR #225
                    let mut cov = Covector::from(*w);
                    cov.deferred = w.deferred();
                    Box::new(cov) as Box<dyn LinearForm<F>>
                })
                .collect(),
            Cow::Borrowed(modified_evaluations.as_slice()),
        );

        let blinding_vectors = witness
            .blinding_vectors
            .iter()
            .map(Vec::as_slice)
            .collect::<Vec<_>>();
        let blinding_witnesses = vec![&witness.blinding_witness];
        let _ = self.blinding_commitment.prove(
            prover_state,
            blinding_vectors.iter().map(|v| Cow::Borrowed(*v)).collect(),
            blinding_witnesses
                .iter()
                .map(|w| Cow::Borrowed(*w))
                .collect(),
            vec![Box::new(beq_weights) as Box<dyn LinearForm<F>>],
            Cow::Borrowed(batched_blinding_subproof_claims.as_slice()),
        );

        result
    }
}
