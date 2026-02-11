#![allow(type_alias_bounds)] // We need the bound to reference F::BasePrimeField.

use std::sync::Arc;

use ark_ff::{FftField, Field};
use ark_std::rand::{CryptoRng, RngCore};
#[cfg(feature = "tracing")]
use tracing::instrument;

use super::Config;
use crate::{
    algebra::{embedding::Embedding, polynomials::CoefficientList},
    hash::Hash,
    protocols::{
        irs_commit,
        whir::zk::{ZkPreprocessingPolynomials, ZkWitness},
    },
    transcript::{
        Codec, DuplexSpongeInterface, ProverMessage, ProverState, VerificationResult, VerifierState,
    },
    utils::zip_strict,
};

pub type Witness<F: FftField> = irs_commit::Witness<F::BasePrimeField, F>;
pub type Commitment<F: Field> = irs_commit::Commitment<F>;

impl<F: FftField> Config<F> {
    /// Commit to one or more polynomials in coefficient form.
    #[cfg_attr(feature = "tracing", instrument(skip_all, fields(size = polynomials.first().unwrap().num_coeffs())))]
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
        let poly_refs = polynomials
            .iter()
            .map(|poly| poly.coeffs())
            .collect::<Vec<_>>();
        self.initial_committer
            .commit(prover_state, poly_refs.as_slice())
    }

    pub fn receive_commitment<H>(
        &self,
        verifier_state: &mut VerifierState<H>,
    ) -> VerificationResult<Commitment<F>>
    where
        H: DuplexSpongeInterface,
        F: Codec<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        self.initial_committer.receive_commitment(verifier_state)
    }

    #[allow(clippy::too_many_lines)]
    pub fn commit_zk<H, R>(
        &self,
        prover_state: &mut ProverState<H, R>,
        polynomials: &[&CoefficientList<F::BasePrimeField>],
        helper_config: &Config<F>,
        preprocessings: Vec<Arc<ZkPreprocessingPolynomials<F>>>,
    ) -> ZkWitness<F>
    where
        H: DuplexSpongeInterface,
        R: RngCore + CryptoRng,
        F: Codec<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        // Commit to the polynomials
        // 1. Compute f̂ = f + msk directly in base field (fused, avoids intermediate Vecs)
        let embedding = self.embedding();
        let mut f_hat_witnesses = Vec::new();
        for (polynomial, preprocessing) in zip_strict(polynomials, &preprocessings) {
            let f_coeffs = polynomial.coeffs();
            let msk_coeffs = preprocessing.msk.coeffs();
            let msk_len = msk_coeffs.len();

            // Fused: compute embed(f) + msk and extract base field in one pass.
            // Avoids allocating extend_msk() (S elements) and intermediate f_hat_coeffs (S elements).
            #[cfg(feature = "parallel")]
            let f_hat_base_field_coeffs: Vec<F::BasePrimeField> = {
                use rayon::prelude::*;
                (0..f_coeffs.len())
                    .into_par_iter()
                    .map(|i| {
                        let msk_c = if i < msk_len { msk_coeffs[i] } else { F::ZERO };
                        (embedding.map(f_coeffs[i]) + msk_c)
                            .to_base_prime_field_elements()
                            .next()
                            .expect("coefficient should be in base field")
                    })
                    .collect()
            };
            #[cfg(not(feature = "parallel"))]
            let f_hat_base_field_coeffs: Vec<F::BasePrimeField> = (0..f_coeffs.len())
                .map(|i| {
                    let msk_c = if i < msk_len { msk_coeffs[i] } else { F::ZERO };
                    (embedding.map(f_coeffs[i]) + msk_c)
                        .to_base_prime_field_elements()
                        .next()
                        .expect("coefficient should be in base field")
                })
                .collect();
            let f_hat = CoefficientList::new(f_hat_base_field_coeffs);
            let f_hat_witness = self.commit(prover_state, &[&f_hat]);
            f_hat_witnesses.push(f_hat_witness);
        }

        // 3. Prepare all helper polynomials in base field for batch commitment
        //    Order: [M, ĝ₁_embedded, ..., ĝμ_embedded]
        //    For each polynomial, we commit to the M polynomial and the ĝ polynomials
        let mut g_hats_embedded_bases = Vec::new();
        let mut m_polys_base = Vec::new();
        for (_, preprocessing) in zip_strict(polynomials, &preprocessings) {
            // Calculate the M polynomial
            #[cfg(feature = "parallel")]
            let m_base_field_coeffs: Vec<F::BasePrimeField> = {
                use rayon::prelude::*;
                preprocessing
                    .m_poly
                    .coeffs()
                    .par_iter()
                    .map(|c| {
                        c.to_base_prime_field_elements()
                            .next()
                            .expect("coefficient should be in base field")
                    })
                    .collect()
            };
            #[cfg(not(feature = "parallel"))]
            let m_base_field_coeffs: Vec<F::BasePrimeField> = preprocessing
                .m_poly
                .coeffs()
                .iter()
                .map(|&c| {
                    c.to_base_prime_field_elements()
                        .next()
                        .expect("coefficient should be in base field")
                })
                .collect();
            let m_base_field_polynomial = CoefficientList::new(m_base_field_coeffs);

            // Parallelize the embedding + base field conversion of all ĝⱼ polynomials
            #[cfg(feature = "parallel")]
            let g_hats_embedded_base: Vec<CoefficientList<F::BasePrimeField>> = {
                use rayon::prelude::*;
                preprocessing
                    .g_hats
                    .par_iter()
                    .map(|g_hat| {
                        let embedded = Self::embed_to_larger(g_hat, preprocessing.params.ell + 1);
                        let coeffs: Vec<F::BasePrimeField> = embedded
                            .coeffs()
                            .iter()
                            .map(|c| {
                                c.to_base_prime_field_elements()
                                    .next()
                                    .expect("coefficient should be in base field")
                            })
                            .collect();
                        CoefficientList::new(coeffs)
                    })
                    .collect()
            };
            #[cfg(not(feature = "parallel"))]
            let g_hats_embedded_base: Vec<CoefficientList<F::BasePrimeField>> = preprocessing
                .g_hats
                .iter()
                .map(|g_hat| {
                    let embedded = Self::embed_to_larger(g_hat, preprocessing.params.ell + 1);
                    let coeffs: Vec<F::BasePrimeField> = embedded
                        .coeffs()
                        .iter()
                        .map(|&c| {
                            c.to_base_prime_field_elements()
                                .next()
                                .expect("coefficient should be in base field")
                        })
                        .collect();
                    CoefficientList::new(coeffs)
                })
                .collect();
            m_polys_base.push(m_base_field_polynomial);
            g_hats_embedded_bases.push(g_hats_embedded_base);
        }

        // 4. Batch-commit all μ+1 helper polynomials in ONE IRS commit
        //    (helper_config has batch_size = μ+1, so one Merkle tree for all)
        //    Layout: [M₁, ĝ₁₁, ..., ĝ₁μ, M₂, ĝ₂₁, ..., ĝ₂μ, ..., Mₙ, ĝₙ₁, ..., ĝₙμ]
        //    Collect references directly — avoids cloning every polynomial.
        let helper_poly_refs: Vec<&CoefficientList<F::BasePrimeField>> = m_polys_base
            .iter()
            .zip(g_hats_embedded_bases.iter())
            .flat_map(|(m_poly, g_hats)| std::iter::once(m_poly).chain(g_hats.iter()))
            .collect();

        let helper_witness = helper_config.commit(prover_state, &helper_poly_refs);

        ZkWitness {
            f_hat_witnesses,
            helper_witness,
            preprocessings,
            m_polys_base,
            g_hats_embedded_bases,
        }
    }

    /// Embed ℓ-variate polynomial into n-variate (n > ℓ)
    /// by treating extra variables as having zero contribution
    fn embed_to_larger(poly: &CoefficientList<F>, n: usize) -> CoefficientList<F> {
        let ell = poly.num_variables();
        assert!(n >= ell);

        let factor = 1 << (n - ell);
        let new_size = 1 << n;
        let mut coeffs = vec![F::ZERO; new_size];

        for (i, &c) in poly.coeffs().iter().enumerate() {
            // Coefficient at index i in ℓ-variate
            // maps to index i * factor in n-variate
            coeffs[i * factor] = c;
        }

        CoefficientList::new(coeffs)
    }
}
