#![allow(type_alias_bounds)] // We need the bound to reference M::Source.

use ark_ff::{FftField, Field};
use ark_std::rand::{CryptoRng, RngCore};
#[cfg(feature = "tracing")]
use tracing::instrument;

use super::Config;
use crate::{
    algebra::embedding::Embedding,
    hash::Hash,
    protocols::irs_commit,
    transcript::{
        Codec, DuplexSpongeInterface, ProverMessage, ProverState, VerificationResult, VerifierState,
    },
};

pub type Witness<F: FftField, M: Embedding<Target = F>> = irs_commit::Witness<M::Source, F>;
pub type Commitment<F: Field> = irs_commit::Commitment<F>;

impl<F, M> Config<F, M>
where
    F: FftField,
    M: Embedding<Target = F>,
    M::Source: FftField,
{
    /// Commit to one or more vectors.
    #[cfg_attr(feature = "tracing", instrument(skip_all, fields(size = vectors.first().unwrap().len())))]
    pub fn commit<H, R>(
        &self,
        prover_state: &mut ProverState<H, R>,
        vectors: &[&[M::Source]],
    ) -> Witness<F, M>
    where
        H: DuplexSpongeInterface,
        R: RngCore + CryptoRng,
        F: Codec<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        self.initial_committer.commit(prover_state, vectors)
    }

    /// Receive a commitment to vectors.
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
