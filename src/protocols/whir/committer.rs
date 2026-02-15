#![allow(type_alias_bounds)] // We need the bound to reference F::BasePrimeField.

use ark_ff::FftField;
use ark_std::rand::{CryptoRng, RngCore};
#[cfg(feature = "tracing")]
use tracing::instrument;

use super::{Commitment, Config, Witness};
use crate::{
    algebra::polynomials::CoefficientList,
    hash::Hash,
    transcript::{
        Codec, DuplexSpongeInterface, ProverMessage, ProverState, VerificationResult, VerifierState,
    },
};

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
}
