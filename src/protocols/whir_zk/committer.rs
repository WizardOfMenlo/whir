#![allow(type_alias_bounds)] // We need the bound to reference F::BasePrimeField.

use ark_ff::FftField;
use ark_std::rand::{CryptoRng, RngCore};
#[cfg(feature = "tracing")]
use tracing::instrument;

use super::utils::interleave_blinding_poly_refs;
use super::{BlindingPolynomials, Config, Witness};
use crate::utils::zip_strict;
use crate::{
    algebra::polynomials::CoefficientList,
    hash::Hash,
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
        //    Both f and msk are base-field polynomials.
        let mut f_hat_witnesses = Vec::new();
        for (polynomial, blinding_polynomial) in
            zip_strict(polynomials.iter(), blinding_polynomials.iter())
        {
            let f_coeffs = polynomial.coeffs();
            let msk_coeffs = blinding_polynomial.msk.coeffs();
            let mut f_hat_coeffs = f_coeffs.to_vec();
            for (dst, &src) in f_hat_coeffs.iter_mut().zip(msk_coeffs) {
                *dst += src;
            }
            let f_hat = CoefficientList::new(f_hat_coeffs);
            let f_hat_witness = self.blinded_commitment.commit(prover_state, &[&f_hat]);
            f_hat_witnesses.push(f_hat_witness);
        }

        // 3. Prepare all blinding polynomials for batch commitment.
        //    m_poly and g_hats are already in base field; just embed g_hats from
        //    ℓ to (ℓ+1) variables.
        let num_blinding_commitment_vars = self.blinding_commitment.initial_num_variables();
        let mut g_hats_embedded_bases = Vec::new();
        let mut m_polys_base = Vec::new();
        for blinding_polynomial in &blinding_polynomials {
            let embed_g_hat =
                |g_hat: &CoefficientList<F::BasePrimeField>| -> CoefficientList<F::BasePrimeField> {
                    g_hat.embed_into_variables(num_blinding_commitment_vars)
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
            m_polys_base.push(blinding_polynomial.m_poly.clone());
            g_hats_embedded_bases.push(g_hats_embedded_base);
        }

        // 4. Batch-commit all μ+1 blinding polynomials in ONE IRS commit
        //    (blinding config has batch_size = μ+1, so one Merkle tree for all)
        //    Layout: [M₁, ĝ₁₁, ..., ĝ₁μ, M₂, ĝ₂₁, ..., ĝ₂μ, ..., Mₙ, ĝₙ₁, ..., ĝₙμ]
        let blinding_poly_refs =
            interleave_blinding_poly_refs::<F>(&m_polys_base, &g_hats_embedded_bases);
        let blinding_witness = self
            .blinding_commitment
            .commit(prover_state, &blinding_poly_refs);

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
    pub fn receive_commitments<H>(
        &self,
        verifier_state: &mut VerifierState<'_, H>,
        num_polynomials: usize,
    ) -> VerificationResult<super::Commitment<F>>
    where
        H: DuplexSpongeInterface,
        F: Codec<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        let f_hat = (0..num_polynomials)
            .map(|_| self.blinded_commitment.receive_commitment(verifier_state))
            .collect::<Result<Vec<_>, _>>()?;
        let blinding = self
            .blinding_commitment
            .receive_commitment(verifier_state)?;
        Ok(super::Commitment { f_hat, blinding })
    }
}
