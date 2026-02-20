use ark_ff::FftField;

use super::Config;
use crate::{
    algebra::linear_form::{Covector, LinearForm},
    hash::Hash,
    protocols::whir_zk::utils::{
        build_combined_and_subproof_claims, construct_batched_eq_weights_from_gammas,
        fold_weight_to_mask_size,
    },
    transcript::{
        codecs::U64, Codec, Decoding, DuplexSpongeInterface, ProverMessage, VerificationResult,
        VerifierMessage, VerifierState,
    },
    verify,
};

impl<F: FftField> Config<F> {
    #[allow(clippy::too_many_lines)]
    pub fn verify<H>(
        &self,
        verifier_state: &mut VerifierState<'_, H>,
        weights: &[&dyn LinearForm<F>],
        evaluations: &[F],
        commitment: &super::Commitment<F>,
    ) -> VerificationResult<()>
    where
        H: DuplexSpongeInterface,
        F: Codec<[H::U]>,
        u8: Decoding<[H::U]>,
        [u8; 32]: Decoding<[H::U]>,
        U64: Codec<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        assert_eq!(
            self.blinded_commitment.initial_committer.num_vectors, 1,
            "zkWHIR currently expects one vector per commitment"
        );
        let num_polynomials = commitment.f_hat.len();
        verify!(evaluations.len() == weights.len() * num_polynomials);

        // Transcript order mirrors prover for evaluation binding:
        // beta, w_folded_evals, non-zero rho, Opening#1, tau1, tau2,
        // raw blinding evals, combined claims, batched h claims,
        // inner blinded WHIR, blinding WHIR with w_folded weights.
        let blinding_challenge: F = verifier_state.verifier_message();

        let num_witness_vars = self.num_witness_variables();
        let num_blinding_variables = self.num_blinding_variables();
        let num_witness_vars_plus_1 = num_witness_vars + 1;
        let num_w_folded_evals = weights.len() * num_polynomials * num_witness_vars_plus_1;
        let w_folded_blinding_evals: Vec<F> =
            verifier_state.prover_messages_vec(num_w_folded_evals)?;

        let m_evals: Vec<F> = w_folded_blinding_evals
            .chunks_exact(num_witness_vars_plus_1)
            .map(|block| block[0])
            .collect();

        let masking_challenge: F = verifier_state.verifier_message();
        verify!(masking_challenge != F::ZERO);
        let commitments = commitment.f_hat.iter().collect::<Vec<_>>();
        let initial_in_domain = self
            .blinded_commitment
            .initial_committer
            .verify(verifier_state, &commitments)?;

        // Doc-faithful Gamma surface: expand each base query alpha_i into coset points
        // alpha_i * Omega_k used by the first folding round.
        let h_gammas = self.all_gammas(&initial_in_domain.points);
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
        let combined_claims: Vec<F> = verifier_state.prover_messages_vec(num_polynomials)?;
        let batched_h_claims: Vec<F> = verifier_state.prover_messages_vec(num_polynomials)?;
        let modified_evaluations: Vec<F> = evaluations
            .iter()
            .zip(m_evals.iter())
            .map(|(&eval, &m_eval)| eval + m_eval)
            .collect();

        self.blinded_commitment
            .verify(verifier_state, &commitments, weights, &modified_evaluations)
            .map(|_| ())?;

        verify!(batched_h_claims == expected_batched_h_claims);
        let g_hat_slices: Vec<&[F]> = g_hat_claims_per_poly.iter().map(Vec::as_slice).collect();
        let (expected_combined_claims, expected_batched_blinding_subproof_claims) =
            build_combined_and_subproof_claims(&m_claims, &g_hat_slices, tau1);
        verify!(combined_claims == expected_combined_claims);

        let beq_weights = construct_batched_eq_weights_from_gammas(
            &h_gammas,
            masking_challenge,
            tau2,
            num_blinding_variables,
        );
        let w_folded_weights: Vec<Covector<F>> = weights
            .iter()
            .map(|&w| fold_weight_to_mask_size(w, num_witness_vars, num_blinding_variables))
            .collect();
        let mut blinding_forms: Vec<&dyn LinearForm<F>> =
            Vec::with_capacity(1 + w_folded_weights.len());
        blinding_forms.push(&beq_weights);
        for wf in &w_folded_weights {
            blinding_forms.push(wf);
        }
        let mut all_expected_blinding_claims =
            Vec::with_capacity(blinding_forms.len() * num_polynomials * num_witness_vars_plus_1);
        all_expected_blinding_claims.extend_from_slice(&expected_batched_blinding_subproof_claims);
        all_expected_blinding_claims.extend_from_slice(&w_folded_blinding_evals);
        let blinding_commitments = vec![&commitment.blinding];
        self.blinding_commitment.verify(
            verifier_state,
            &blinding_commitments,
            &blinding_forms,
            &all_expected_blinding_claims,
        )?;

        Ok(())
    }
}
