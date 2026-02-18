use ark_ff::FftField;

use super::Config;
use crate::{
    algebra::linear_form::LinearForm,
    hash::Hash,
    protocols::whir_zk::utils::{
        construct_batched_eq_weights_from_gammas, recombine_doc_claim_from_components,
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
        let mut m_claims = vec![F::ZERO; num_polynomials];
        let mut g_hat_claims_per_poly = vec![vec![F::ZERO; num_witness_vars]; num_polynomials];
        let mut expected_batched_h_claims = vec![F::ZERO; num_polynomials];
        let mut tau2_power = F::ONE;
        for &gamma in &h_gammas {
            for poly_idx in 0..num_polynomials {
                // Parse `(m_eval, g_hat_1_eval, ..., g_hat_mu_eval)` for each (gamma, polynomial).
                let m_eval: F = verifier_state.prover_message()?;
                let mut h_value = m_eval;
                let mut blinding_power = blinding_challenge;
                let mut gamma_power = gamma;
                m_claims[poly_idx] += tau2_power * m_eval;
                for g_hat_idx in 0..num_witness_vars {
                    let g_hat_eval: F = verifier_state.prover_message()?;
                    g_hat_claims_per_poly[poly_idx][g_hat_idx] += tau2_power * g_hat_eval;
                    h_value += blinding_power * gamma_power * g_hat_eval;
                    blinding_power *= blinding_challenge;
                    gamma_power = gamma_power.square();
                }
                expected_batched_h_claims[poly_idx] += tau2_power * h_value;
            }
            tau2_power *= tau2;
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

        if batched_h_claims != expected_batched_h_claims {
            return Err(VerificationError);
        }
        let mut expected_combined_doc_claims = Vec::with_capacity(num_polynomials);
        let mut expected_batched_blinding_subproof_claims =
            Vec::with_capacity(num_polynomials * (1 + num_witness_vars));
        for poly_idx in 0..num_polynomials {
            let m_claim = m_claims[poly_idx];
            let g_hat_claims = &g_hat_claims_per_poly[poly_idx];
            expected_combined_doc_claims.push(recombine_doc_claim_from_components(
                m_claim,
                g_hat_claims,
                tau1,
            ));
            expected_batched_blinding_subproof_claims.push(m_claim);
            for &g_hat_claim in g_hat_claims {
                expected_batched_blinding_subproof_claims.push(g_hat_claim);
            }
        }
        if combined_doc_claims != expected_combined_doc_claims {
            return Err(VerificationError);
        }

        // Verify the blinding WHIR subproof against the same batched beq form used to
        // derive raw round-0 claims at (gamma, -rho).
        let beq_weights = construct_batched_eq_weights_from_gammas(
            &h_gammas,
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
