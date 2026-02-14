#![allow(type_alias_bounds)] // We need the bound to reference F::BasePrimeField.

use ark_ff::FftField;
use ark_std::rand::{CryptoRng, RngCore};
#[cfg(feature = "tracing")]
use tracing::instrument;

use super::config::Config;
use super::utils::{interleave_helper_poly_refs, HelperPolynomials, ZkWitness};
use crate::utils::zip_strict;
use crate::{
    algebra::{add_base_with_projection, polynomials::CoefficientList, project_all_to_base},
    hash::Hash,
    transcript::{Codec, DuplexSpongeInterface, ProverMessage, ProverState},
};

impl<F: FftField> Config<F> {
    #[allow(clippy::too_many_lines)]
    #[cfg_attr(feature = "tracing", instrument(skip_all, fields(num_polynomials = polynomials.len())))]
    pub fn commit<H, R>(
        &self,
        prover_state: &mut ProverState<H, R>,
        polynomials: &[&CoefficientList<F::BasePrimeField>],
    ) -> ZkWitness<F>
    where
        H: DuplexSpongeInterface,
        R: RngCore + CryptoRng,
        F: Codec<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        #[cfg(feature = "alloc-track")]
        let mut __snap = crate::alloc_snap!();

        // Sample helper polynomials internally from the prover's RNG.
        let helper_polynomials: Vec<HelperPolynomials<F>> = (0..polynomials.len())
            .map(|_| HelperPolynomials::sample(prover_state.rng(), self.zk_params.clone()))
            .collect();

        // Commit to the polynomials
        // 1. Compute f̂ = f + msk directly in base field.
        //    Both f_coeffs and msk are base field elements (msk is sampled from BasePrimeField
        //    then lifted to F), so the addition can be done in BasePrimeField directly,
        //    avoiding a needless round-trip through the extension field.
        let mut f_hat_witnesses = Vec::new();
        for (polynomial, helper_polynomial) in
            zip_strict(polynomials.iter(), helper_polynomials.iter())
        {
            // f̂ = f + msk in base field (msk projected from extension, zero-padded).
            let f_hat_coeffs =
                add_base_with_projection::<F>(polynomial.coeffs(), helper_polynomial.msk.coeffs());
            let f_hat = CoefficientList::new(f_hat_coeffs);
            let f_hat_witness = self.main.commit(prover_state, &[&f_hat]);
            f_hat_witnesses.push(f_hat_witness);
        }

        #[cfg(feature = "alloc-track")]
        crate::alloc_report!("commit::f_hat_commit", __snap);

        // 3. Prepare all helper polynomials in base field for batch commitment
        //    Order: [M, ĝ₁_embedded, ..., ĝμ_embedded]
        //    For each polynomial, we commit to the M polynomial and the ĝ polynomials
        let mut g_hats_embedded_bases = Vec::new();
        let mut m_polys_base = Vec::new();
        for helper_polynomial in &helper_polynomials {
            // Convert M polynomial to base field
            let m_base_field_polynomial =
                CoefficientList::new(project_all_to_base(helper_polynomial.m_poly.coeffs()));

            // Embed each ĝⱼ from ℓ to (ℓ+1) variables, then convert to base field
            let embed_g_hat = |g_hat: &CoefficientList<F>| -> CoefficientList<F::BasePrimeField> {
                let embedded =
                    g_hat.embed_into_variables(helper_polynomial.params.num_helper_variables + 1);
                CoefficientList::new(project_all_to_base(embedded.coeffs()))
            };
            #[cfg(feature = "parallel")]
            let g_hats_embedded_base: Vec<CoefficientList<F::BasePrimeField>> = {
                use rayon::prelude::*;
                helper_polynomial.g_hats.par_iter().map(embed_g_hat).collect()
            };
            #[cfg(not(feature = "parallel"))]
            let g_hats_embedded_base: Vec<CoefficientList<F::BasePrimeField>> =
                helper_polynomial.g_hats.iter().map(embed_g_hat).collect();
            m_polys_base.push(m_base_field_polynomial);
            g_hats_embedded_bases.push(g_hats_embedded_base);
        }

        #[cfg(feature = "alloc-track")]
        crate::alloc_report!("commit::prepare_helper_polys", __snap);

        // 4. Batch-commit all μ+1 helper polynomials in ONE IRS commit
        //    (helper config has batch_size = μ+1, so one Merkle tree for all)
        //    Layout: [M₁, ĝ₁₁, ..., ĝ₁μ, M₂, ĝ₂₁, ..., ĝ₂μ, ..., Mₙ, ĝₙ₁, ..., ĝₙμ]
        let helper_poly_refs =
            interleave_helper_poly_refs::<F>(&m_polys_base, &g_hats_embedded_bases);
        let helper_witness = self.helper.commit(prover_state, &helper_poly_refs);

        #[cfg(feature = "alloc-track")]
        crate::alloc_report!("commit::helper_batch_commit", __snap);

        ZkWitness {
            f_hat_witnesses,
            helper_witness,
            helper_polynomials,
            m_polys_base,
            g_hats_embedded_bases,
        }
    }
}
