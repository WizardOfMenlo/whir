#![allow(type_alias_bounds)] // We need the bound to reference F::BasePrimeField.

use ark_ff::{FftField, Field};
use ark_std::rand::{CryptoRng, RngCore};
#[cfg(feature = "tracing")]
use tracing::instrument;

use super::Config;
use crate::{
    algebra::polynomials::CoefficientList,
    hash::Hash,
    protocols::irs_commit,
    transcript::{
        Codec, DuplexSpongeInterface, ProverMessage, ProverState, VerificationResult, VerifierState,
    },
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
}
