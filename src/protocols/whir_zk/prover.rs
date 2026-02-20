use std::borrow::Cow;

use ark_ff::FftField;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
#[cfg(feature = "tracing")]
use tracing::instrument;

use super::{Config, Witness};
use crate::{
    algebra::{
        embedding::Basefield,
        linear_form::{Covector, Evaluate, LinearForm},
        mixed_dot_seq,
    },
    hash::Hash,
    protocols::whir_zk::utils::{
        build_combined_and_subproof_claims, fill_eq_weights_at_gamma, fold_weight_to_mask_size,
        BlindingPolynomials,
    },
    transcript::{ProverMessage, VerifierMessage},
};

/// Evaluate all blinding polynomials at a single gamma point, writing per-polynomial
/// results `[m_eval, g_hat_0, ..., g_hat_{mu-1}, h_val]` into `output`.
#[inline]
#[allow(clippy::too_many_arguments)]
fn evaluate_polys_at_gamma<F: FftField>(
    output: &mut [F],
    eq_weights: &[F],
    embedding: Basefield<F>,
    blinding_polynomials: &[BlindingPolynomials<F>],
    blinding_vectors: &[Vec<F::BasePrimeField>],
    blinding_challenge: F,
    gamma: F,
    num_witness_vars: usize,
    num_witness_vars_plus_1: usize,
    stride_per_poly: usize,
) {
    for (poly_idx, bp) in blinding_polynomials.iter().enumerate() {
        let embedded_g_hats = &blinding_vectors
            [poly_idx * num_witness_vars_plus_1 + 1..(poly_idx + 1) * num_witness_vars_plus_1];
        let off = poly_idx * stride_per_poly;
        output[off] = mixed_dot_seq(&embedding, eq_weights, &bp.m_poly);
        for (j, g_hat) in embedded_g_hats.iter().enumerate() {
            output[off + 1 + j] = mixed_dot_seq(&embedding, eq_weights, g_hat);
        }
        let mut h = output[off];
        let mut blinding_pow = blinding_challenge;
        let mut gamma_pow = gamma;
        for j in 0..embedded_g_hats.len() {
            h += blinding_pow * gamma_pow * output[off + 1 + j];
            blinding_pow *= blinding_challenge;
            gamma_pow = gamma_pow.square();
        }
        output[off + 1 + num_witness_vars] = h;
    }
}

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

        // Transcript order for zkWHIR evaluation binding:
        // 1) sample beta,
        // 2) send w_folded evaluations (binding M_eval before rho),
        // 3) sample non-zero rho,
        // 4) compute modified_eval = F + M_eval,
        // 5) Opening#1 of f_hat,
        // 6) sample tau1, tau2,
        // 7) send raw blinding openings + combined/batched claims,
        // 8) inner blinded WHIR,
        // 9) blinding WHIR with additional w_folded weights.
        let blinding_challenge: F = prover_state.verifier_message();

        let embedding = self.blinding_commitment.embedding();
        let num_polynomials = polynomials.len();
        let num_witness_vars = self.num_witness_variables();
        let num_blinding_variables = self.num_blinding_variables();
        let num_witness_vars_plus_1 = num_witness_vars + 1;

        // Compute w_folded evaluations of all blinding vectors BEFORE rho for binding.
        // For each (weight, polynomial, blinding_vector), w_folded_blinding_evals stores
        // <w_folded, blinding_vector>.  The v=0 entry per polynomial is
        // M_eval = <w_folded, m_poly>, the masking contribution to f_hat_eval.
        let mut w_folded_weights: Vec<Covector<F>> = Vec::with_capacity(weights.len());
        let mut m_evals: Vec<F> = Vec::with_capacity(evaluations.len());
        let mut w_folded_blinding_evals: Vec<F> =
            Vec::with_capacity(weights.len() * num_polynomials * num_witness_vars_plus_1);

        for &weight in weights {
            let w_folded =
                fold_weight_to_mask_size(weight, num_witness_vars, num_blinding_variables);
            for poly_idx in 0..num_polynomials {
                let base = poly_idx * num_witness_vars_plus_1;
                for v in 0..num_witness_vars_plus_1 {
                    let eval: F = w_folded.evaluate(embedding, &blinding_vectors[base + v]);
                    w_folded_blinding_evals.push(eval);
                    if v == 0 {
                        m_evals.push(eval);
                    }
                }
            }
            w_folded_weights.push(w_folded);
        }
        prover_state.prover_message_fields(&w_folded_blinding_evals);

        // Sample non-zero rho AFTER masking evaluations are committed to the transcript.
        let masking_challenge: F = prover_state.verifier_message();
        assert_ne!(
            masking_challenge,
            F::ZERO,
            "zkWHIR requires non-zero masking challenge rho"
        );

        // modified_eval = F + M_eval; for an honest prover this equals <weight, f_hat>.
        let modified_evaluations: Vec<F> = evaluations
            .iter()
            .zip(m_evals.iter())
            .map(|(&eval, &m)| eval + m)
            .collect();

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
                num_gammas.div_ceil(cores).max(1)
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
                        let mut tau2_pow = tau2.pow([base_gi as u64]);
                        for local in 0..chunk_gammas {
                            let gamma = h_gammas[base_gi + local];
                            fill_eq_weights_at_gamma(
                                &mut eq_buf,
                                gamma,
                                masking_challenge,
                                num_blinding_variables,
                            );
                            for (a, &v) in accum.iter_mut().zip(eq_buf.iter()) {
                                *a += tau2_pow * v;
                            }
                            let slot = &mut chunk
                                [local * stride_per_gamma..(local + 1) * stride_per_gamma];
                            evaluate_polys_at_gamma(
                                slot,
                                &eq_buf,
                                embedding,
                                &blinding_polynomials,
                                &blinding_vectors,
                                blinding_challenge,
                                gamma,
                                num_witness_vars,
                                num_witness_vars_plus_1,
                                stride_per_poly,
                            );
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
                for (a, &v) in accum.iter_mut().zip(eq_buf.iter()) {
                    *a += tau2_pow * v;
                }
                evaluate_polys_at_gamma(
                    &mut eval_results[gi * stride_per_gamma..(gi + 1) * stride_per_gamma],
                    &eq_buf,
                    embedding,
                    &blinding_polynomials,
                    &blinding_vectors,
                    blinding_challenge,
                    gamma,
                    num_witness_vars,
                    num_witness_vars_plus_1,
                    stride_per_poly,
                );
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

                    prover_state.prover_message_field(&m_eval);
                    prover_state.prover_message_fields(g_hat_evals);

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

        let g_hat_slices: Vec<&[F]> = (0..num_polynomials)
            .map(|i| &g_hat_claims[i * num_witness_vars..(i + 1) * num_witness_vars])
            .collect();
        let (combined_claims, batched_blinding_subproof_claims) =
            build_combined_and_subproof_claims(&m_claims, &g_hat_slices, tau1);
        let beq_weights = Covector::new(beq_weight_accum);
        prover_state.prover_message_fields(&combined_claims);
        prover_state.prover_message_fields(&batched_h_claims);

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
            let mut blinding_forms: Vec<&dyn LinearForm<F>> =
                Vec::with_capacity(1 + w_folded_weights.len());
            blinding_forms.push(&beq_weights);
            for wf in &w_folded_weights {
                blinding_forms.push(wf);
            }
            let mut all_blinding_claims = Vec::with_capacity(
                blinding_forms.len() * num_polynomials * num_witness_vars_plus_1,
            );
            all_blinding_claims.extend_from_slice(&batched_blinding_subproof_claims);
            all_blinding_claims.extend_from_slice(&w_folded_blinding_evals);
            let _ = self.blinding_commitment.prove(
                prover_state,
                blinding_vectors.into_iter().map(Cow::Owned).collect(),
                vec![Cow::Owned(blinding_witness)],
                &blinding_forms,
                Cow::Owned(all_blinding_claims),
            );
        }

        result
    }
}
