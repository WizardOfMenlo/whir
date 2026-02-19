use std::{any::Any, borrow::Cow};

use ark_ff::FftField;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::{Config, Witness};
use crate::{
    algebra::linear_form::{Covector, Evaluate, LinearForm, MultilinearExtension},
    hash::Hash,
    protocols::whir_zk::utils::{beq_covector_at_gamma, recombine_doc_claim_from_components},
    transcript::{ProverMessage, VerifierMessage},
};

impl<F: FftField> Config<F> {
    #[allow(clippy::too_many_lines)]
    pub fn prove<H, R>(
        &self,
        prover_state: &mut crate::transcript::ProverState<H, R>,
        polynomials: &[&[F::BasePrimeField]],
        witness: Witness<F>,
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

        // Compute f_hat evaluations per weight per polynomial.
        let embedding = self.blinded_commitment.embedding();
        let mut modified_evaluations = Vec::with_capacity(evaluations.len());
        for (weight_idx, &weight) in weights.iter().enumerate() {
            let maybe_mle = (weight as &dyn Any).downcast_ref::<MultilinearExtension<F>>();
            // Temporary fallback Covector materialised only when the weight is not an MLE.
            let fallback_cov: Option<Covector<F>> = if maybe_mle.is_none() {
                Some(Covector::from(weight))
            } else {
                None
            };
            for (poly_idx, f_hat_vector) in witness.f_hat_vectors.iter().enumerate() {
                let idx = weight_idx * polynomials.len() + poly_idx;
                let f_hat_eval = if let Some(mle) = maybe_mle {
                    mle.evaluate(embedding, f_hat_vector.as_slice())
                } else {
                    fallback_cov
                        .as_ref()
                        .unwrap()
                        .evaluate(embedding, f_hat_vector.as_slice())
                };
                // Enforce L = rho * f + g with L instantiated by the committed masked witness.
                let g_eval = f_hat_eval - masking_challenge * evaluations[idx];
                prover_state.prover_message(&g_eval);
                debug_assert_eq!(masking_challenge * evaluations[idx] + g_eval, f_hat_eval);
                modified_evaluations.push(f_hat_eval);
            }
            // fallback_cov is dropped here (if it was Some), keeping the live set small.
        }

        let initial_in_domain = {
            let witness_refs: Vec<_> = witness.f_hat_witnesses.iter().collect();
            self.blinded_commitment
                .initial_committer
                .open(prover_state, &witness_refs)
        };

        // Doc-faithful Gamma surface: expand each base query alpha_i into coset points
        // alpha_i * Omega_k used by the first folding round.
        let h_gammas = self.all_gammas(&initial_in_domain.points);

        // Sample batching challenges before revealing any raw opening payload.
        let tau1: F = prover_state.verifier_message();
        let tau2: F = prover_state.verifier_message();

        let num_polynomials = polynomials.len();
        let num_witness_vars = self.num_witness_variables();
        let num_blinding_variables = self.num_blinding_variables();
        let num_witness_vars_plus_1 = num_witness_vars + 1;

        let mut m_claims = vec![F::ZERO; num_polynomials];
        // Flat layout: g_hat_claims[poly_idx * num_witness_vars + j]
        let mut g_hat_claims = vec![F::ZERO; num_polynomials * num_witness_vars];
        let mut batched_h_claims = vec![F::ZERO; num_polynomials];
        let mut beq_weight_accum = vec![F::ZERO; 1 << (num_blinding_variables + 1)];

        #[cfg(not(feature = "parallel"))]
        {
            let mut tau2_power = F::ONE;
            for &gamma in &h_gammas {
                let beq_cov =
                    beq_covector_at_gamma(gamma, masking_challenge, num_blinding_variables);
                for (poly_idx, bp) in witness.blinding_polynomials.iter().enumerate() {
                    let eg = &witness.blinding_vectors[poly_idx * num_witness_vars_plus_1 + 1
                        ..(poly_idx + 1) * num_witness_vars_plus_1];
                    let eval = bp.evaluate_with_covector(&beq_cov, eg, gamma);
                    prover_state.prover_message(&eval.m_eval);
                    for g_hat_eval in &eval.g_hat_evals {
                        prover_state.prover_message(g_hat_eval);
                    }
                    m_claims[poly_idx] += tau2_power * eval.m_eval;
                    for (j, &g) in eval.g_hat_evals.iter().enumerate() {
                        g_hat_claims[poly_idx * num_witness_vars + j] += tau2_power * g;
                    }
                    batched_h_claims[poly_idx] +=
                        tau2_power * eval.compute_h_value(blinding_challenge);
                }
                for (acc, &w) in beq_weight_accum.iter_mut().zip(beq_cov.vector.iter()) {
                    *acc += tau2_power * w;
                }
                tau2_power *= tau2;
            }
        }

        // Parallel: compute per-gamma evaluations concurrently, apply sequentially for
        // transcript ordering.  Uses blinding_vectors slices to avoid extra allocations.
        #[cfg(feature = "parallel")]
        {
            let all_gamma_data: Vec<(Vec<F>, Vec<(F, Vec<F>, F)>)> = h_gammas
                .par_iter()
                .map(|&gamma| {
                    let beq_cov =
                        beq_covector_at_gamma(gamma, masking_challenge, num_blinding_variables);
                    let poly_data: Vec<(F, Vec<F>, F)> = witness
                        .blinding_polynomials
                        .iter()
                        .enumerate()
                        .map(|(i, bp)| {
                            let eg = &witness.blinding_vectors[i * num_witness_vars_plus_1 + 1
                                ..(i + 1) * num_witness_vars_plus_1];
                            let eval = bp.evaluate_with_covector(&beq_cov, eg, gamma);
                            let h_val = eval.compute_h_value(blinding_challenge);
                            (eval.m_eval, eval.g_hat_evals, h_val)
                        })
                        .collect();
                    (beq_cov.vector, poly_data)
                })
                .collect();

            let mut tau2_power = F::ONE;
            for (beq_weights, poly_data) in &all_gamma_data {
                for (poly_idx, (m_eval, g_hat_evals, h_val)) in poly_data.iter().enumerate() {
                    prover_state.prover_message(m_eval);
                    for g_hat_eval in g_hat_evals {
                        prover_state.prover_message(g_hat_eval);
                    }
                    m_claims[poly_idx] += tau2_power * *m_eval;
                    for (j, &g) in g_hat_evals.iter().enumerate() {
                        g_hat_claims[poly_idx * num_witness_vars + j] += tau2_power * g;
                    }
                    batched_h_claims[poly_idx] += tau2_power * *h_val;
                }
                for (acc, &w) in beq_weight_accum.iter_mut().zip(beq_weights.iter()) {
                    *acc += tau2_power * w;
                }
                tau2_power *= tau2;
            }
        }

        let mut combined_doc_claims = Vec::with_capacity(num_polynomials);
        let mut batched_blinding_subproof_claims =
            Vec::with_capacity(num_polynomials * num_witness_vars_plus_1);
        for poly_idx in 0..num_polynomials {
            let m_claim = m_claims[poly_idx];
            let g_hat_slice =
                &g_hat_claims[poly_idx * num_witness_vars..(poly_idx + 1) * num_witness_vars];
            batched_blinding_subproof_claims.push(m_claim);
            batched_blinding_subproof_claims.extend_from_slice(g_hat_slice);
            combined_doc_claims.push(recombine_doc_claim_from_components(
                m_claim,
                g_hat_slice,
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

        // Build lightweight linear forms for the inner prove calls.
        let linear_forms_for_prove: Vec<Box<dyn LinearForm<F>>> = weights
            .iter()
            .map(|&w| -> Box<dyn LinearForm<F>> {
                if let Some(mle) = (w as &dyn Any).downcast_ref::<MultilinearExtension<F>>() {
                    Box::new(MultilinearExtension {
                        point: mle.point.clone(),
                    })
                } else {
                    let mut cov = Covector::from(w);
                    cov.deferred = w.deferred();
                    Box::new(cov)
                }
            })
            .collect();

        let Witness {
            f_hat_witnesses,
            f_hat_vectors,
            blinding_polynomials: _,
            blinding_vectors,
            blinding_witness,
        } = witness;

        let result = self.blinded_commitment.prove(
            prover_state,
            f_hat_vectors.into_iter().map(Cow::Owned).collect(),
            f_hat_witnesses.into_iter().map(Cow::Owned).collect(),
            linear_forms_for_prove,
            Cow::Borrowed(modified_evaluations.as_slice()),
        );

        let _ = self.blinding_commitment.prove(
            prover_state,
            blinding_vectors.into_iter().map(Cow::Owned).collect(),
            vec![Cow::Owned(blinding_witness)],
            vec![Box::new(beq_weights) as Box<dyn LinearForm<F>>],
            Cow::Borrowed(batched_blinding_subproof_claims.as_slice()),
        );

        result
    }
}
