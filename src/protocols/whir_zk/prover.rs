use ark_ff::FftField;

use super::{Config, Witness};
use crate::{
    algebra::linear_form::{Covector, Evaluate, LinearForm},
    hash::Hash,
    protocols::whir_zk::utils::{
        batch_with_challenge, compute_per_polynomial_claims, construct_batched_eq_weights,
        recombine_doc_claim_from_components,
    },
    transcript::{ProverMessage, VerifierMessage},
};

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

        let mut g_evaluations = Vec::with_capacity(evaluations.len());
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
                g_evaluations.push(g_eval);
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

        // Raw round-0 claims at each gamma:
        // m_eval = M(gamma, -rho), g_hat_eval_j = g_hat_j(gamma, -rho).
        let raw_blinding_evals_per_poly = witness
            .blinding_polynomials
            .iter()
            .map(|blinding_polynomial| {
                h_gammas
                    .iter()
                    .copied()
                    .map(|gamma| blinding_polynomial.evaluate_at(gamma, masking_challenge))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        for gamma_idx in 0..h_gammas.len() {
            for poly_idx in 0..polynomials.len() {
                let eval = &raw_blinding_evals_per_poly[poly_idx][gamma_idx];
                prover_state.prover_message(&eval.m_eval);
                for g_hat_eval in &eval.g_hat_evals {
                    prover_state.prover_message(g_hat_eval);
                }
            }
        }
        let mut combined_doc_claims = Vec::with_capacity(polynomials.len());
        let mut batched_h_claims = Vec::with_capacity(polynomials.len());
        let mut batched_blinding_subproof_claims =
            Vec::with_capacity(polynomials.len() * (1 + num_witness_vars));
        for blinding_evals in &raw_blinding_evals_per_poly {
            // Keep per-vector claims for the internal WHIR subproof input.
            let (m_claim, g_hat_claims) = compute_per_polynomial_claims(blinding_evals, tau2);
            batched_blinding_subproof_claims.push(m_claim);
            batched_blinding_subproof_claims.extend_from_slice(&g_hat_claims);

            // Public doc-faithful claim path:
            // S = M + Σ_j tau1^j * g_j, then outer tau2 batching over Gamma with factor 2.
            combined_doc_claims.push(recombine_doc_claim_from_components(
                m_claim,
                &g_hat_claims,
                tau1,
            ));

            // tau2 batching for h(gamma) values.
            let h_claim = batch_with_challenge(
                blinding_evals
                    .iter()
                    .map(|eval| eval.compute_h_value(blinding_challenge)),
                tau2,
            );
            batched_h_claims.push(h_claim);
        }
        let beq_weights = construct_batched_eq_weights(
            &raw_blinding_evals_per_poly[0],
            masking_challenge,
            tau2,
            self.num_blinding_variables(),
        );
        for claim in &combined_doc_claims {
            prover_state.prover_message(claim);
        }
        for h_claim in &batched_h_claims {
            prover_state.prover_message(h_claim);
        }
        for (idx, g_eval) in g_evaluations.iter().copied().enumerate() {
            debug_assert_eq!(
                masking_challenge * evaluations[idx] + g_eval,
                modified_evaluations[idx]
            );
        }

        let vectors = witness
            .f_hat_vectors
            .iter()
            .map(Vec::as_slice)
            .collect::<Vec<_>>();
        let result = self.blinded_commitment.prove(
            prover_state,
            &vectors,
            &witness_refs,
            weights,
            &modified_evaluations,
        );

        let blinding_forms: Vec<&dyn LinearForm<F>> = vec![&beq_weights];

        let blinding_vectors = witness
            .blinding_vectors
            .iter()
            .map(Vec::as_slice)
            .collect::<Vec<_>>();
        let blinding_witnesses = vec![&witness.blinding_witness];
        let _ = self.blinding_commitment.prove(
            prover_state,
            &blinding_vectors,
            &blinding_witnesses,
            &blinding_forms,
            &batched_blinding_subproof_claims,
        );

        result
    }
}
