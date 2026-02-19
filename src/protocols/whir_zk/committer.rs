use ark_ff::FftField;
use ark_std::rand::{rngs::StdRng, SeedableRng};

use super::{utils::BlindingPolynomials, Config};
use crate::{
    hash::Hash,
    protocols::{irs_commit, whir},
    transcript::ProverMessage,
};

/// zkWHIR commitment object.
pub struct Commitment<F: FftField> {
    pub f_hat: Vec<whir::Commitment<F>>,
    pub blinding: whir::Commitment<F>,
}

/// zkWHIR witness object.
#[derive(Clone)]
pub struct Witness<F: FftField> {
    pub f_hat_witnesses: Vec<irs_commit::Witness<F::BasePrimeField, F>>,
    pub f_hat_vectors: Vec<Vec<F::BasePrimeField>>,
    pub blinding_polynomials: Vec<BlindingPolynomials<F>>,
    pub blinding_vectors: Vec<Vec<F::BasePrimeField>>,
    pub blinding_witness: irs_commit::Witness<F::BasePrimeField, F>,
}

impl<F: FftField> Config<F> {
    /// Lift an `ell+1` mask vector to the witness-side `mu` domain size.
    ///
    /// The current implementation uses periodic repetition to match the larger
    /// witness commitment size while keeping the blinding-side commitment in
    /// the smaller `ell+1` space.
    fn lift_mask_to_witness_size(
        mask_vector: &[F::BasePrimeField],
        witness_size: usize,
    ) -> Vec<F::BasePrimeField> {
        assert!(
            !mask_vector.is_empty(),
            "blinding mask vector must be non-empty"
        );
        assert!(
            witness_size >= mask_vector.len(),
            "witness vector smaller than blinding mask vector"
        );
        assert_eq!(
            witness_size % mask_vector.len(),
            0,
            "witness size must be multiple of blinding mask size"
        );
        let repeats = witness_size / mask_vector.len();
        let mut lifted = Vec::with_capacity(witness_size);
        for _ in 0..repeats {
            lifted.extend_from_slice(mask_vector);
        }
        lifted
    }

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
            self.blinded_commitment.initial_committer.num_vectors, 1,
            "zkWHIR currently expects one vector per commitment"
        );

        let mut f_hat_witnesses = Vec::with_capacity(polynomials.len());
        let mut f_hat_vectors = Vec::with_capacity(polynomials.len());
        let mut blinding_polynomials = Vec::with_capacity(polynomials.len());
        // Sample fresh blinding coefficients without adding transcript interactions.
        let mut blinding_rng = StdRng::from_entropy();
        let num_blinding_vars = self.num_blinding_variables();
        let num_witness_vars = self.num_witness_variables();
        for &poly in polynomials {
            let blinding =
                BlindingPolynomials::sample(&mut blinding_rng, num_blinding_vars, num_witness_vars);
            let mask_vec = Self::lift_mask_to_witness_size(
                &blinding.m_poly,
                self.blinded_commitment.initial_size(),
            );
            let f_hat_vec = poly
                .iter()
                .zip(mask_vec.iter())
                .map(|(&coeff, &mask)| coeff + mask)
                .collect::<Vec<_>>();
            let witness = self
                .blinded_commitment
                .commit(prover_state, &[f_hat_vec.as_slice()]);
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
        let mut blinding_vectors = Vec::with_capacity(blinding_num_vectors);
        for poly in blinding_polynomials.iter().take(polynomials.len()) {
            let layout = poly.layout_vectors();
            debug_assert_eq!(layout.len(), num_witness_vars + 1);
            blinding_vectors.extend(layout);
        }
        // for poly_idx in 0..polynomials.len() {
        //     let layout = blinding_polynomials[poly_idx].layout_vectors();
        //     debug_assert_eq!(layout.len(), num_witness_vars + 1);
        //     blinding_vectors.extend(layout);
        // }
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
        let blinding = self
            .blinding_commitment
            .receive_commitment(verifier_state)?;
        Ok(Commitment { f_hat, blinding })
    }
}
