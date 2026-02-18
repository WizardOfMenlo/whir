use ark_ff::FftField;

use super::Config;
use crate::{
    algebra::linear_form::LinearForm,
    hash::Hash,
    protocols::whir_zk::utils::{
        batch_with_challenge, compute_per_polynomial_claims, construct_batched_eq_weights,
        recombine_doc_claim_from_components, BlindingEvaluations,
    },
    transcript::{ProverMessage, VerificationError, VerifierMessage},
};

impl<F: FftField> Config<F> {
    pub fn verify<H>(
        &self,
        verifier_state: &mut crate::transcript::VerifierState<'_, H>,
        weights: &[&dyn crate::algebra::linear_form::LinearForm<F>],
        evaluations: &[F],
        commitment: &super::Commitment<F>,
    ) -> crate::transcript::VerificationResult<()>
    where
        H: crate::transcript::DuplexSpongeInterface,
        F: crate::transcript::Codec<[H::U]>,
        u8: crate::transcript::Decoding<[H::U]>,
        [u8; 32]: crate::transcript::Decoding<[H::U]>,
        crate::transcript::codecs::U64: crate::transcript::Codec<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        assert_eq!(
            self.blinded_commitment.initial_committer.num_vectors, 1,
            "zkWHIR currently expects one vector per commitment"
        );
        let num_polynomials = commitment.f_hat.len();
        if evaluations.len() != weights.len() * num_polynomials {
            return Err(VerificationError);
        }

        // Transcript order mirrors prover for round-0 consistency checks:
        // beta, non-zero rho, g-evals, tau1, tau2, raw blinding evals,
        // combined doc claims, batched h claims.
        let blinding_challenge: F = verifier_state.verifier_message();
        let masking_challenge: F = verifier_state.verifier_message();
        if masking_challenge == F::ZERO {
            return Err(VerificationError);
        }
        let g_evals: Vec<F> =
            verifier_state.prover_messages_vec(weights.len() * num_polynomials)?;
        let commitments = commitment.f_hat.iter().collect::<Vec<_>>();
        let initial_in_domain = self
            .blinded_commitment
            .initial_committer
            .verify(verifier_state, &commitments)?;

        // Doc-faithful Gamma surface: expand each base query alpha_i into coset points
        // alpha_i * Omega_k used by the first folding round.
        let h_gammas = self.all_gammas(&initial_in_domain.points);
        let num_witness_vars = self.num_witness_variables();
        let tau1: F = verifier_state.verifier_message();
        let tau2: F = verifier_state.verifier_message();
        let raw_eval_count = h_gammas.len() * num_polynomials * (1 + num_witness_vars);
        let raw_blinding_evals_flat: Vec<F> = verifier_state.prover_messages_vec(raw_eval_count)?;
        let mut raw_blinding_evals_per_poly =
            vec![Vec::with_capacity(h_gammas.len()); num_polynomials];
        let mut cursor = 0usize;
        for gamma in h_gammas.iter().copied() {
            for poly_evals in &mut raw_blinding_evals_per_poly {
                // Parse `(m_eval, g_hat_1_eval, ..., g_hat_mu_eval)` for each (gamma, polynomial).
                let m_eval = raw_blinding_evals_flat[cursor];
                cursor += 1;
                let g_hat_evals =
                    raw_blinding_evals_flat[cursor..cursor + num_witness_vars].to_vec();
                cursor += num_witness_vars;
                poly_evals.push(BlindingEvaluations {
                    gamma,
                    m_eval,
                    g_hat_evals,
                });
            }
        }
        if cursor != raw_blinding_evals_flat.len() {
            return Err(VerificationError);
        }
        let combined_doc_claims: Vec<F> = verifier_state.prover_messages_vec(num_polynomials)?;
        let batched_h_claims: Vec<F> = verifier_state.prover_messages_vec(num_polynomials)?;
        let modified_evaluations: Vec<F> = evaluations
            .iter()
            .zip(g_evals.iter())
            .map(|(&eval, &g_eval)| masking_challenge * eval + g_eval)
            .collect();

        self.blinded_commitment
            .verify(verifier_state, &commitments, weights, &modified_evaluations)
            .map(|_| ())?;

        let expected_batched_h_claims = (0..num_polynomials)
            .map(|poly_idx| {
                batch_with_challenge(
                    raw_blinding_evals_per_poly[poly_idx]
                        .iter()
                        .map(|eval| eval.compute_h_value(blinding_challenge)),
                    tau2,
                )
            })
            .collect::<Vec<_>>();
        for eval_idx in 0..evaluations.len() {
            let recomposed = masking_challenge * evaluations[eval_idx] + g_evals[eval_idx];
            if recomposed != modified_evaluations[eval_idx] {
                return Err(VerificationError);
            }
        }
        if batched_h_claims != expected_batched_h_claims {
            return Err(VerificationError);
        }
        let mut expected_combined_doc_claims = Vec::with_capacity(num_polynomials);
        let mut expected_batched_blinding_subproof_claims =
            Vec::with_capacity(num_polynomials * (1 + num_witness_vars));
        for blinding_evals in &raw_blinding_evals_per_poly {
            let (m_claim, g_hat_claims) = compute_per_polynomial_claims(blinding_evals, tau2);
            expected_combined_doc_claims.push(recombine_doc_claim_from_components(
                m_claim,
                &g_hat_claims,
                tau1,
            ));
            expected_batched_blinding_subproof_claims.push(m_claim);
            for g_hat_claim in g_hat_claims {
                expected_batched_blinding_subproof_claims.push(g_hat_claim);
            }
        }
        if combined_doc_claims != expected_combined_doc_claims {
            return Err(VerificationError);
        }

        // Verify the blinding WHIR subproof against the same batched beq form used to
        // derive raw round-0 claims at (gamma, -rho).
        let beq_weights = construct_batched_eq_weights(
            &raw_blinding_evals_per_poly[0],
            masking_challenge,
            tau2,
            self.num_blinding_variables(),
        );
        let blinding_commitments = vec![&commitment.blinding];
        let blinding_forms: Vec<&dyn LinearForm<F>> = vec![&beq_weights];
        self.blinding_commitment.verify(
            verifier_state,
            &blinding_commitments,
            &blinding_forms,
            &expected_batched_blinding_subproof_claims,
        )?;

        Ok(())
    }
}
