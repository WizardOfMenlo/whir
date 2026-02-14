#![allow(type_alias_bounds)] // We need the bound to reference F::BasePrimeField.

use ark_ff::FftField;
use ark_std::rand::{CryptoRng, RngCore};
#[cfg(feature = "tracing")]
use tracing::instrument;

use super::utils::{interleave_blinding_poly_refs, BlindingPolynomials};
use super::{Config, Witness};
use crate::utils::zip_strict;
use crate::{
    algebra::{add_base_with_projection, polynomials::CoefficientList, project_all_to_base},
    hash::Hash,
    protocols::whir::Commitment,
    transcript::{
        Codec, DuplexSpongeInterface, ProverMessage, ProverState, VerificationResult, VerifierState,
    },
};

impl<F: FftField> Config<F> {
    #[allow(clippy::too_many_lines)]
    #[cfg_attr(feature = "tracing", instrument(skip_all, fields(num_polynomials = polynomials.len())))]
    pub fn commit<H, R>(
        &self,
        prover_state: &mut ProverState<H, R>,
        polynomials: &[&CoefficientList<F::BasePrimeField>],
    ) -> Witness<F>
    where
        H: DuplexSpongeInterface,
        R: RngCore + CryptoRng,
        F: Codec<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        #[cfg(feature = "alloc-track")]
        let mut __snap = crate::alloc_snap!();

        let num_blinding_vars = self.num_blinding_variables();
        let num_witness_vars = self.num_witness_variables();

        // Sample blinding polynomials internally from the prover's RNG.
        let blinding_polynomials: Vec<BlindingPolynomials<F>> = (0..polynomials.len())
            .map(|_| {
                BlindingPolynomials::sample(prover_state.rng(), num_blinding_vars, num_witness_vars)
            })
            .collect();

        // Commit to the polynomials
        // 1. Compute f̂ = f + msk directly in base field.
        //    Both f_coeffs and msk are base field elements (msk is sampled from BasePrimeField
        //    then lifted to F), so the addition can be done in BasePrimeField directly,
        //    avoiding a needless round-trip through the extension field.
        let mut f_hat_witnesses = Vec::new();
        for (polynomial, blinding_polynomial) in
            zip_strict(polynomials.iter(), blinding_polynomials.iter())
        {
            // f̂ = f + msk in base field (msk projected from extension, zero-padded).
            let f_hat_coeffs = add_base_with_projection::<F>(
                polynomial.coeffs(),
                blinding_polynomial.msk.coeffs(),
            );
            let f_hat = CoefficientList::new(f_hat_coeffs);
            let f_hat_witness = self.main.commit(prover_state, &[&f_hat]);
            f_hat_witnesses.push(f_hat_witness);
        }

        #[cfg(feature = "alloc-track")]
        crate::alloc_report!("commit::f_hat_commit", __snap);

        // 3. Prepare all blinding polynomials in base field for batch commitment
        //    Order: [M, ĝ₁_embedded, ..., ĝμ_embedded]
        //    For each polynomial, we commit to the M polynomial and the ĝ polynomials
        let num_blinding_commitment_vars = self.blinding_poly_commitment.initial_num_variables();
        let mut g_hats_embedded_bases = Vec::new();
        let mut m_polys_base = Vec::new();
        for blinding_polynomial in &blinding_polynomials {
            // Convert M polynomial to base field
            let m_base_field_polynomial =
                CoefficientList::new(project_all_to_base(blinding_polynomial.m_poly.coeffs()));

            // Embed each ĝⱼ from ℓ to (ℓ+1) variables, then convert to base field
            let embed_g_hat = |g_hat: &CoefficientList<F>| -> CoefficientList<F::BasePrimeField> {
                let embedded = g_hat.embed_into_variables(num_blinding_commitment_vars);
                CoefficientList::new(project_all_to_base(embedded.coeffs()))
            };
            #[cfg(feature = "parallel")]
            let g_hats_embedded_base: Vec<CoefficientList<F::BasePrimeField>> = {
                use rayon::prelude::*;
                blinding_polynomial
                    .g_hats
                    .par_iter()
                    .map(embed_g_hat)
                    .collect()
            };
            #[cfg(not(feature = "parallel"))]
            let g_hats_embedded_base: Vec<CoefficientList<F::BasePrimeField>> =
                blinding_polynomial.g_hats.iter().map(embed_g_hat).collect();
            m_polys_base.push(m_base_field_polynomial);
            g_hats_embedded_bases.push(g_hats_embedded_base);
        }

        #[cfg(feature = "alloc-track")]
        crate::alloc_report!("commit::prepare_blinding_polys", __snap);

        // 4. Batch-commit all μ+1 blinding polynomials in ONE IRS commit
        //    (blinding config has batch_size = μ+1, so one Merkle tree for all)
        //    Layout: [M₁, ĝ₁₁, ..., ĝ₁μ, M₂, ĝ₂₁, ..., ĝ₂μ, ..., Mₙ, ĝₙ₁, ..., ĝₙμ]
        let blinding_poly_refs =
            interleave_blinding_poly_refs::<F>(&m_polys_base, &g_hats_embedded_bases);
        let blinding_witness = self
            .blinding_poly_commitment
            .commit(prover_state, &blinding_poly_refs);

        #[cfg(feature = "alloc-track")]
        crate::alloc_report!("commit::blinding_batch_commit", __snap);

        Witness {
            f_hat_witnesses,
            blinding_witness,
            blinding_polynomials,
            m_polys_base,
            g_hats_embedded_bases,
        }
    }

    /// Receive commitments from the transcript: one f̂ commitment per polynomial,
    /// plus a single batch commitment for all blinding polynomials.
    ///
    /// Returns `(f_hat_commitments, blinding_commitment)`.
    pub fn receive_commitments<H>(
        &self,
        verifier_state: &mut VerifierState<'_, H>,
        num_polynomials: usize,
    ) -> VerificationResult<(Vec<Commitment<F>>, Commitment<F>)>
    where
        H: DuplexSpongeInterface,
        F: Codec<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        let f_hat_commitments = (0..num_polynomials)
            .map(|_| self.main.receive_commitment(verifier_state))
            .collect::<Result<Vec<_>, _>>()?;
        let blinding_commitment = self
            .blinding_poly_commitment
            .receive_commitment(verifier_state)?;
        Ok((f_hat_commitments, blinding_commitment))
    }
}
