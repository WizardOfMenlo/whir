#![allow(type_alias_bounds)]

use ark_ff::FftField;
use ark_std::rand::{rngs::StdRng, SeedableRng};

use super::{utils::BlindingPolynomials, Config};
use crate::{hash::Hash, transcript::ProverMessage};

/// zkWHIR commitment object.
pub struct Commitment<F: FftField> {
    pub f_hat: Vec<crate::protocols::whir::Commitment<F>>,
    pub blinding: crate::protocols::whir::Commitment<F>,
}

/// zkWHIR witness object.
#[derive(Clone)]
pub struct Witness<F: FftField> {
    pub f_hat_witnesses:
        Vec<crate::protocols::irs_commit::Witness<F::BasePrimeField, F>>,
    pub f_hat_vectors: Vec<Vec<F::BasePrimeField>>,
    pub r_vector_indices: Vec<usize>,
    pub blinding_polynomials: Vec<BlindingPolynomials<F>>,
    pub blinding_vectors: Vec<Vec<F::BasePrimeField>>,
    pub blinding_witness: crate::protocols::irs_commit::Witness<F::BasePrimeField, F>,
}

impl<F: FftField> Config<F> {
    pub fn commit<H, R>(
        &self,
        prover_state: &mut crate::transcript::ProverState<H, R>,
        polynomials: &[&[F::BasePrimeField]],
    ) -> Witness<F>
    where
        H: crate::transcript::DuplexSpongeInterface,
        R: ark_std::rand::RngCore + ark_std::rand::CryptoRng,
        F: crate::transcript::Codec<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        assert_eq!(
            self.blinded_commitment.initial_committer.num_vectors,
            1,
            "zkWHIR currently expects one vector per commitment"
        );
        assert_eq!(
            self.blinded_commitment.initial_size(),
            self.blinding_commitment.initial_size(),
            "blinded and blinding commitment vector sizes must match"
        );

        let mut f_hat_witnesses = Vec::with_capacity(polynomials.len());
        let mut f_hat_vectors = Vec::with_capacity(polynomials.len());
        let mut blinding_polynomials = Vec::with_capacity(polynomials.len());
        // Sample fresh blinding coefficients without adding transcript interactions.
        let mut blinding_rng = StdRng::from_entropy();
        let num_blinding_vars = self.num_blinding_variables();
        let num_witness_vars = self.num_witness_variables();
        for (poly_idx, &poly) in polynomials.iter().enumerate() {
            let blinding = BlindingPolynomials::sample(
                &mut blinding_rng,
                poly_idx,
                num_blinding_vars,
                num_witness_vars,
            );
            let mask_vec = blinding.m_poly.clone();
            let f_hat_vec = poly
                .iter()
                .zip(mask_vec.iter())
                .map(|(&coeff, &mask)| coeff + mask)
                .collect::<Vec<_>>();
            let witness = self.blinded_commitment.commit(prover_state, &[f_hat_vec.as_slice()]);
            f_hat_witnesses.push(witness);
            f_hat_vectors.push(f_hat_vec);
            blinding_polynomials.push(blinding);
        }

        let blinding_num_vectors = self.blinding_commitment.initial_committer.num_vectors;
        assert_eq!(
            blinding_num_vectors,
            polynomials.len() * (num_witness_vars + 1),
            "blinding commitment layout mismatch: expected n*(mu+1) vectors"
        );
        let mut r_vector_indices = Vec::with_capacity(polynomials.len());
        let mut blinding_vectors = Vec::with_capacity(blinding_num_vectors);
        for poly_idx in 0..polynomials.len() {
            let base = poly_idx * (num_witness_vars + 1);
            r_vector_indices.push(base);
            let layout = blinding_polynomials[poly_idx].layout_vectors();
            debug_assert_eq!(layout.len(), num_witness_vars + 1);
            blinding_vectors.extend(layout);
        }
        let blinding_vector_refs = blinding_vectors
            .iter()
            .map(Vec::as_slice)
            .collect::<Vec<_>>();
        let blinding_witness = self
            .blinding_commitment
            .commit(prover_state, &blinding_vector_refs);

        Witness {
            f_hat_witnesses,
            f_hat_vectors,
            r_vector_indices,
            blinding_polynomials,
            blinding_vectors,
            blinding_witness,
        }
    }

    pub fn receive_commitments<H>(
        &self,
        verifier_state: &mut crate::transcript::VerifierState<'_, H>,
        num_polynomials: usize,
    ) -> crate::transcript::VerificationResult<Commitment<F>>
    where
        H: crate::transcript::DuplexSpongeInterface,
        F: crate::transcript::Codec<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        let f_hat = (0..num_polynomials)
            .map(|_| self.blinded_commitment.receive_commitment(verifier_state))
            .collect::<Result<Vec<_>, _>>()?;
        let blinding = self.blinding_commitment.receive_commitment(verifier_state)?;
        Ok(Commitment { f_hat, blinding })
    }

}
