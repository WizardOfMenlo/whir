//! Quadratic sumcheck protocol.

use std::fmt;

use ark_ff::Field;
use ark_std::rand::{CryptoRng, RngCore};
use serde::{Deserialize, Serialize};
#[cfg(feature = "tracing")]
use tracing::instrument;

use crate::{
    algebra::{
        dot,
        poly_utils::{evals::EvaluationsList, multilinear::MultilinearPoint},
        sumcheck::{compute_sumcheck_polynomial, fold},
    },
    ensure,
    protocols::proof_of_work,
    transcript::{
        codecs::U64, Codec, Decoding, DuplexSpongeInterface, ProverState, VerificationResult,
        VerifierMessage, VerifierState,
    },
    type_info::Type,
    verify,
};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct Config<F>
where
    F: Field,
{
    pub field: Type<F>,
    pub initial_size: usize,
    pub rounds: Vec<RoundConfig>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RoundConfig {
    pub pow: proof_of_work::Config,
}

impl<F: Field> Config<F> {
    pub fn validate(&self) -> Result<(), &'static str> {
        ensure!(
            self.initial_size.is_power_of_two(),
            "Initial size must be power of two."
        );
        ensure!(
            self.initial_size.ilog2() as usize >= self.rounds.len(),
            "Initial size must be >= 2^{rounds}."
        );
        Ok(())
    }

    pub const fn final_size(&self) -> usize {
        self.initial_size >> self.num_rounds()
    }

    pub const fn num_rounds(&self) -> usize {
        self.rounds.len()
    }

    /// Runs the quadratic sumcheck protocol as configured.
    ///
    /// It reduces a claim of the form `dot(a, b) == sum` to an exponentially
    /// smaller claim `dot(a', b') == sum'` where `a' = fold(a, randomness)`
    /// and similarly for `b`.
    ///
    /// This function:
    /// - Samples random values to progressively reduce the polynomial.
    /// - Applies proof-of-work grinding if required.
    /// - Returns the sampled folding randomness values used in each reduction step.
    #[cfg_attr(feature = "tracing", instrument(skip(self, prover_state)))]
    pub fn prove<H, R>(
        &self,
        prover_state: &mut ProverState<H, R>,
        a: &mut EvaluationsList<F>,
        b: &mut EvaluationsList<F>,
        sum: &mut F,
    ) -> MultilinearPoint<F>
    where
        H: DuplexSpongeInterface,
        R: CryptoRng + RngCore,
        F: Codec<[H::U]>,
        [u8; 32]: Decoding<[H::U]>,
        U64: Codec<[H::U]>,
    {
        self.validate().expect("Invalid configuration");
        assert_eq!(a.num_evals(), self.initial_size);
        assert_eq!(b.num_evals(), self.initial_size);
        debug_assert_eq!(dot(a.evals(), b.evals()), *sum);

        let mut res = Vec::with_capacity(self.num_rounds());
        for round in &self.rounds {
            // Send sumcheck polynomial c0 and c2
            let (c0, c2) = compute_sumcheck_polynomial(a.evals(), b.evals());
            let c1 = *sum - c0.double() - c2;
            prover_state.prover_message(&c0);
            prover_state.prover_message(&c2);

            // Do Proof of Work (if any)
            round.pow.prove(prover_state);

            // Receive the random evaluation point
            let folding_randomness = prover_state.verifier_message::<F>();
            res.push(folding_randomness);

            // Fold the inputs
            *a = EvaluationsList::new(fold(folding_randomness, a.evals()));
            *b = EvaluationsList::new(fold(folding_randomness, b.evals()));
            *sum = (c2 * folding_randomness + c1) * folding_randomness + c0;
        }

        res.reverse();
        MultilinearPoint(res)
    }

    #[cfg_attr(feature = "tracing", instrument(skip(verifier_state)))]
    pub fn verify<H>(
        &self,
        verifier_state: &mut VerifierState<H>,
        sum: &mut F,
    ) -> VerificationResult<MultilinearPoint<F>>
    where
        H: DuplexSpongeInterface,
        F: Codec<[H::U]>,
        [u8; 32]: Decoding<[H::U]>,
        U64: Codec<[H::U]>,
    {
        verify!(self.validate().is_ok());

        let mut res = Vec::with_capacity(self.num_rounds());
        for round in &self.rounds {
            // Receive sumcheck polynomial c0 and c2
            let c0: F = verifier_state.prover_message()?;
            let c2: F = verifier_state.prover_message()?;
            let c1 = *sum - c0.double() - c2;

            // Check proof of work (if any)
            round.pow.verify(verifier_state)?;

            // Receive the random evaluation point
            let folding_randomness = verifier_state.verifier_message::<F>();
            res.push(folding_randomness);

            // Update the sum
            *sum = (c2 * folding_randomness + c1) * folding_randomness + c0;
        }

        res.reverse();
        Ok(MultilinearPoint(res))
    }
}

impl<F: Field> fmt::Display for Config<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "size {} rounds {} pow",
            self.initial_size,
            self.num_rounds()
        )?;
        for round in &self.rounds {
            write!(f, " {:.2}", round.pow.difficulty())?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {

    // TODO: Proptest based tests checking invariants and post conditions.
}
