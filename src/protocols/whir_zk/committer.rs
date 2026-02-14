#![allow(type_alias_bounds)] // We need the bound to reference F::BasePrimeField.

use ark_ff::FftField;
use ark_std::rand::{CryptoRng, RngCore};
#[cfg(feature = "tracing")]
use tracing::instrument;

use super::utils::{
    interleave_helper_poly_refs, prepare_helper_polynomials, ZkPreprocessingPolynomials, ZkWitness,
};
use crate::{
    algebra::{add_base_with_projection, polynomials::CoefficientList},
    hash::Hash,
    protocols::whir::Config,
    transcript::{Codec, DuplexSpongeInterface, ProverMessage, ProverState},
    utils::zip_strict,
};

impl<F: FftField> Config<F> {
    #[allow(clippy::too_many_lines)]
    #[cfg_attr(feature = "tracing", instrument(skip_all, fields(num_polynomials = polynomials.len())))]
    pub fn commit_zk<H, R>(
        &self,
        prover_state: &mut ProverState<H, R>,
        polynomials: &[&CoefficientList<F::BasePrimeField>],
        helper_config: &Config<F>,
        preprocessings: &[&ZkPreprocessingPolynomials<F>],
    ) -> ZkWitness<F>
    where
        H: DuplexSpongeInterface,
        R: RngCore + CryptoRng,
        F: Codec<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        #[cfg(feature = "alloc-track")]
        let mut __snap = crate::alloc_snap!();

        // Commit to the polynomials
        // 1. Compute f̂ = f + msk directly in base field.
        //    Both f_coeffs and msk are base field elements (msk is sampled from BasePrimeField
        //    then lifted to F), so the addition can be done in BasePrimeField directly,
        //    avoiding a needless round-trip through the extension field.
        let mut f_hat_witnesses = Vec::new();
        for (polynomial, preprocessing) in zip_strict(polynomials, preprocessings) {
            // f̂ = f + msk in base field (msk projected from extension, zero-padded).
            let f_hat_coeffs =
                add_base_with_projection::<F>(polynomial.coeffs(), preprocessing.msk.coeffs());
            let f_hat = CoefficientList::new(f_hat_coeffs);
            let f_hat_witness = self.commit(prover_state, &[&f_hat]);
            f_hat_witnesses.push(f_hat_witness);
        }

        #[cfg(feature = "alloc-track")]
        crate::alloc_report!("commit_zk::f_hat_commit", __snap);

        // 3. Prepare all helper polynomials in base field for batch commitment
        let (m_polys_base, g_hats_embedded_bases) = prepare_helper_polynomials(preprocessings);

        #[cfg(feature = "alloc-track")]
        crate::alloc_report!("commit_zk::prepare_helper_polys", __snap);

        // 4. Batch-commit all μ+1 helper polynomials in ONE IRS commit
        //    (helper_config has batch_size = μ+1, so one Merkle tree for all)
        //    Layout: [M₁, ĝ₁₁, ..., ĝ₁μ, M₂, ĝ₂₁, ..., ĝ₂μ, ..., Mₙ, ĝₙ₁, ..., ĝₙμ]
        let helper_poly_refs =
            interleave_helper_poly_refs::<F>(&m_polys_base, &g_hats_embedded_bases);
        let helper_witness = helper_config.commit(prover_state, &helper_poly_refs);

        #[cfg(feature = "alloc-track")]
        crate::alloc_report!("commit_zk::helper_batch_commit", __snap);

        ZkWitness {
            f_hat_witnesses,
            helper_witness,
            preprocessings: preprocessings.iter().copied().cloned().collect(),
            m_polys_base,
            g_hats_embedded_bases,
        }
    }
}
