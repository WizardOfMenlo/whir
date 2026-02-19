use std::{any::Any, borrow::Cow};

use ark_ff::FftField;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
#[cfg(feature = "tracing")]
use tracing::instrument;

use super::{Config, Witness};
use crate::{
    algebra::{
        embedding::Basefield,
        linear_form::{Covector, Evaluate, LinearForm, MultilinearExtension},
        mixed_dot_seq,
    },
    hash::Hash,
    protocols::whir_zk::utils::{fill_eq_weights_at_gamma, recombine_doc_claim_from_components},
    transcript::{ProverMessage, VerifierMessage},
};

impl<F: FftField> Config<F> {
    #[allow(clippy::too_many_lines)]
    #[cfg_attr(feature = "tracing", instrument(skip_all))]
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

        // Destructure witness early so blinding_polynomials can be dropped after the
        // gamma evaluation block, freeing memory before the inner prove calls.
        let Witness {
            f_hat_witnesses,
            f_hat_vectors,
            blinding_polynomials,
            blinding_vectors,
            blinding_witness,
        } = witness;

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
            for (poly_idx, f_hat_vector) in f_hat_vectors.iter().enumerate() {
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
            #[cfg(feature = "tracing")]
            let _span = tracing::info_span!("open_f_hat").entered();
            let witness_refs: Vec<_> = f_hat_witnesses.iter().collect();
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

        let weight_size = 1usize << (num_blinding_variables + 1);

        // Flat layout per gamma: [m_eval, g_hat_0, ..., g_hat_{mu-1}, h_val] × num_polynomials.
        let stride_per_poly = 1 + num_witness_vars + 1;
        let stride_per_gamma = num_polynomials * stride_per_poly;
        let num_gammas = h_gammas.len();
        let mut eval_results = vec![F::ZERO; num_gammas * stride_per_gamma];

        #[cfg(feature = "parallel")]
        let beq_weight_accum = {
            let batch = {
                let cores = rayon::current_num_threads();
                ((num_gammas + cores - 1) / cores).max(1)
            };
            let batch_stride = batch * stride_per_gamma;
            eval_results
                .par_chunks_mut(batch_stride)
                .enumerate()
                .fold(
                    || (vec![F::ZERO; weight_size], vec![F::ZERO; weight_size]),
                    |(mut accum, mut eq_buf), (chunk_idx, chunk)| {
                        let base_gi = chunk_idx * batch;
                        let chunk_gammas = chunk.len() / stride_per_gamma;
                        let embedding = Basefield::<F>::new();
                        let mut tau2_pow = {
                            let mut p = F::ONE;
                            for _ in 0..base_gi {
                                p *= tau2;
                            }
                            p
                        };
                        for local in 0..chunk_gammas {
                            let gamma = h_gammas[base_gi + local];
                            let slot = &mut chunk
                                [local * stride_per_gamma..(local + 1) * stride_per_gamma];
                            fill_eq_weights_at_gamma(
                                &mut eq_buf,
                                gamma,
                                masking_challenge,
                                num_blinding_variables,
                            );
                            for (a, &v) in accum.iter_mut().zip(eq_buf[..weight_size].iter()) {
                                *a += tau2_pow * v;
                            }
                            for (poly_idx, bp) in blinding_polynomials.iter().enumerate() {
                                let eg = &blinding_vectors[poly_idx * num_witness_vars_plus_1 + 1
                                    ..(poly_idx + 1) * num_witness_vars_plus_1];
                                let off = poly_idx * stride_per_poly;
                                slot[off] =
                                    mixed_dot_seq(&embedding, &eq_buf[..weight_size], &bp.m_poly);
                                for (j, g_hat_embedded) in eg.iter().enumerate() {
                                    slot[off + 1 + j] = mixed_dot_seq(
                                        &embedding,
                                        &eq_buf[..weight_size],
                                        g_hat_embedded,
                                    );
                                }
                                let mut h = slot[off];
                                let mut bp_pow = blinding_challenge;
                                let mut gp = gamma;
                                for j in 0..eg.len() {
                                    h += bp_pow * gp * slot[off + 1 + j];
                                    bp_pow *= blinding_challenge;
                                    gp = gp.square();
                                }
                                slot[off + 1 + num_witness_vars] = h;
                            }
                            tau2_pow *= tau2;
                        }
                        (accum, eq_buf)
                    },
                )
                .map(|(accum, _)| accum)
                .reduce(
                    || vec![F::ZERO; weight_size],
                    |mut a, b| {
                        for (x, &y) in a.iter_mut().zip(b.iter()) {
                            *x += y;
                        }
                        a
                    },
                )
        };

        #[cfg(not(feature = "parallel"))]
        let beq_weight_accum = {
            let embedding = Basefield::<F>::new();
            let mut eq_buf = vec![F::ZERO; weight_size];
            let mut accum = vec![F::ZERO; weight_size];
            let mut tau2_pow = F::ONE;
            for (gi, &gamma) in h_gammas.iter().enumerate() {
                fill_eq_weights_at_gamma(
                    &mut eq_buf,
                    gamma,
                    masking_challenge,
                    num_blinding_variables,
                );
                for (a, &v) in accum.iter_mut().zip(eq_buf[..weight_size].iter()) {
                    *a += tau2_pow * v;
                }
                for (poly_idx, bp) in blinding_polynomials.iter().enumerate() {
                    let eg = &blinding_vectors[poly_idx * num_witness_vars_plus_1 + 1
                        ..(poly_idx + 1) * num_witness_vars_plus_1];
                    let off = gi * stride_per_gamma + poly_idx * stride_per_poly;
                    eval_results[off] =
                        mixed_dot_seq(&embedding, &eq_buf[..weight_size], &bp.m_poly);
                    for (j, g_hat_embedded) in eg.iter().enumerate() {
                        eval_results[off + 1 + j] =
                            mixed_dot_seq(&embedding, &eq_buf[..weight_size], g_hat_embedded);
                    }
                    let mut h = eval_results[off];
                    let mut bp_pow = blinding_challenge;
                    let mut gp = gamma;
                    for j in 0..eg.len() {
                        h += bp_pow * gp * eval_results[off + 1 + j];
                        bp_pow *= blinding_challenge;
                        gp = gp.square();
                    }
                    eval_results[off + 1 + num_witness_vars] = h;
                }
                tau2_pow *= tau2;
            }
            accum
        };

        // Sequential transcript writes + claim accumulation from the naturally-ordered flat array.
        {
            let mut tau2_pow = F::ONE;
            for gi in 0..num_gammas {
                let base = gi * stride_per_gamma;
                for poly_idx in 0..num_polynomials {
                    let off = base + poly_idx * stride_per_poly;
                    let m_eval = eval_results[off];
                    let g_hat_evals = &eval_results[off + 1..off + 1 + num_witness_vars];
                    let h_val = eval_results[off + 1 + num_witness_vars];

                    prover_state.prover_message(&m_eval);
                    for g in g_hat_evals {
                        prover_state.prover_message(g);
                    }

                    m_claims[poly_idx] += tau2_pow * m_eval;
                    for (j, &g) in g_hat_evals.iter().enumerate() {
                        g_hat_claims[poly_idx * num_witness_vars + j] += tau2_pow * g;
                    }
                    batched_h_claims[poly_idx] += tau2_pow * h_val;
                }
                tau2_pow *= tau2;
            }
        }

        drop(eval_results);

        drop(blinding_polynomials);

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

        let result = {
            #[cfg(feature = "tracing")]
            let _span = tracing::info_span!("inner_blinded_prove").entered();
            self.blinded_commitment.prove(
                prover_state,
                f_hat_vectors.into_iter().map(Cow::Owned).collect(),
                f_hat_witnesses.into_iter().map(Cow::Owned).collect(),
                weights,
                Cow::Borrowed(modified_evaluations.as_slice()),
            )
        };

        {
            #[cfg(feature = "tracing")]
            let _span = tracing::info_span!("inner_blinding_prove").entered();
            let beq_weight_ref: &dyn LinearForm<F> = &beq_weights;
            let _ = self.blinding_commitment.prove(
                prover_state,
                blinding_vectors.into_iter().map(Cow::Owned).collect(),
                vec![Cow::Owned(blinding_witness)],
                &[beq_weight_ref],
                Cow::Borrowed(batched_blinding_subproof_claims.as_slice()),
            );
        }

        result
    }
}
