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
        sumcheck::{compute_sumcheck_polynomial, fold},
        MultilinearPoint,
    },
    protocols::proof_of_work,
    transcript::{
        codecs::U64, Codec, Decoding, DuplexSpongeInterface, ProverState, VerificationResult,
        VerifierMessage, VerifierState,
    },
    type_info::Type,
};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct Config<F>
where
    F: Field,
{
    pub field: Type<F>,
    pub initial_size: usize,
    pub round_pow: proof_of_work::Config,
    pub num_rounds: usize,
}

impl<F: Field> Config<F> {
    pub fn final_size(&self) -> usize {
        assert!(
            self.num_rounds == 0 || self.initial_size.next_power_of_two() >= 1 << self.num_rounds
        );
        if self.initial_size == 0 || self.num_rounds == 0 {
            self.initial_size
        } else {
            self.initial_size.next_power_of_two() >> self.num_rounds
        }
    }

    /// Runs the quadratic sumcheck protocol as configured.
    ///
    /// It reduces a claim of the form `dot(a, b) == sum` to an exponentially
    /// smaller claim `dot(a', b') == sum'` where `a'` is `a` folded in place
    /// and similarly for `b`.
    ///
    /// This function:
    /// - Samples random values to progressively reduce the polynomial.
    /// - Applies proof-of-work grinding if required.
    /// - Returns the sampled folding randomness values used in each reduction step.
    #[cfg_attr(feature = "tracing", instrument(skip_all))]
    pub fn prove<H, R>(
        &self,
        prover_state: &mut ProverState<H, R>,
        a: &mut Vec<F>,
        b: &mut Vec<F>,
        sum: &mut F,
    ) -> MultilinearPoint<F>
    where
        H: DuplexSpongeInterface,
        R: CryptoRng + RngCore,
        F: Codec<[H::U]>,
        [u8; 32]: Decoding<[H::U]>,
        U64: Codec<[H::U]>,
    {
        assert!(
            self.num_rounds == 0 || self.initial_size.next_power_of_two() >= 1 << self.num_rounds
        );
        assert_eq!(a.len(), self.initial_size);
        assert_eq!(b.len(), self.initial_size);
        debug_assert_eq!(dot(a, b), *sum);

        let mut res = Vec::with_capacity(self.num_rounds);
        for _ in 0..self.num_rounds {
            debug_assert!(a.len() > 1);
            // Send sumcheck polynomial c0 and c2
            let (c0, c2) = compute_sumcheck_polynomial(a, b);
            let c1 = *sum - c0.double() - c2;
            prover_state.prover_message(&c0);
            prover_state.prover_message(&c2);

            // Do Proof of Work (if any)
            self.round_pow.prove(prover_state);

            // Receive the random evaluation point
            let folding_randomness = prover_state.verifier_message::<F>();
            res.push(folding_randomness);

            // Fold the inputs
            fold(a, folding_randomness);
            fold(b, folding_randomness);
            *sum = (c2 * folding_randomness + c1) * folding_randomness + c0;
        }

        MultilinearPoint(res)
    }

    #[cfg_attr(feature = "tracing", instrument(skip_all))]
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
        assert!(
            self.num_rounds == 0 || self.initial_size.next_power_of_two() >= 1 << self.num_rounds
        );
        let mut res = Vec::with_capacity(self.num_rounds);
        for _ in 0..self.num_rounds {
            // Receive sumcheck polynomial c0 and c2
            let c0: F = verifier_state.prover_message()?;
            let c2: F = verifier_state.prover_message()?;
            let c1 = *sum - c0.double() - c2;

            // Check proof of work (if any)
            self.round_pow.verify(verifier_state)?;

            // Receive the random evaluation point
            let folding_randomness = verifier_state.verifier_message::<F>();
            res.push(folding_randomness);

            // Update the sum
            *sum = (c2 * folding_randomness + c1) * folding_randomness + c0;
        }

        Ok(MultilinearPoint(res))
    }
}

impl<F: Field> fmt::Display for Config<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "size {} rounds {} pow {:.2}",
            self.initial_size,
            self.num_rounds,
            self.round_pow.difficulty()
        )
    }
}

#[cfg(test)]
mod tests {

    // TODO: Proptest based tests checking invariants and post conditions.
    use ark_std::rand::{
        distributions::{Distribution, Standard},
        rngs::StdRng,
        SeedableRng,
    };
    use proptest::{proptest, strategy::Strategy};
    #[cfg(feature = "tracing")]
    use tracing::instrument;

    use super::*;
    use crate::{
        algebra::{fields, multilinear_extend, random_vector},
        transcript::DomainSeparator,
    };

    impl<F: Field> Config<F> {
        pub fn arbitrary() -> impl Strategy<Value = Self> {
            (0_usize..(1 << 12), 0_usize..12).prop_map(|(initial_size, num_rounds)| {
                let num_rounds =
                    num_rounds.min(initial_size.next_power_of_two().trailing_zeros() as usize);
                Self {
                    field: Type::new(),
                    initial_size,
                    num_rounds,
                    round_pow: proof_of_work::Config::none(),
                }
            })
        }
    }

    #[cfg_attr(feature = "tracing", instrument)]
    fn test_config<F>(seed: u64, config: &Config<F>)
    where
        F: Field + Codec,
        Standard: Distribution<F>,
    {
        // Pseudo-random Instance
        let instance = U64(seed);
        let ds = DomainSeparator::protocol(config)
            .session(&format!("Test at {}:{}", file!(), line!()))
            .instance(&instance);
        let mut rng = StdRng::seed_from_u64(seed);
        let initial_vector = random_vector(&mut rng, config.initial_size);
        let initial_covector = random_vector(&mut rng, config.initial_size);
        let initial_sum = dot(&initial_vector, &initial_covector);

        // Prover
        let mut vector = initial_vector.clone();
        let mut covector = initial_covector.clone();
        let mut sum = initial_sum;
        let mut prover_state = ProverState::new_std(&ds);
        let point = config.prove(&mut prover_state, &mut vector, &mut covector, &mut sum);
        assert_eq!(vector.len(), config.final_size());
        assert_eq!(covector.len(), config.final_size());
        assert_eq!(dot(&vector, &covector), sum);
        if config.final_size() == 1 {
            assert_eq!(multilinear_extend(&initial_vector, &point.0), vector[0]);
            assert_eq!(multilinear_extend(&initial_covector, &point.0), covector[0]);
        } else {
            // TODO: Check correct folding.
        }
        let proof = prover_state.proof();

        // Verifier
        let mut verifier_sum = initial_sum;
        let mut verifier_state = VerifierState::new_std(&ds, &proof);
        let verifier_point = config
            .verify(&mut verifier_state, &mut verifier_sum)
            .unwrap();
        assert_eq!(verifier_point, point);
        assert_eq!(verifier_sum, sum);
        verifier_state.check_eof().unwrap();
    }

    fn test_sumcheck<F>()
    where
        F: Field + Codec,
        Standard: Distribution<F>,
    {
        crate::tests::init();
        proptest!(|(seed: u64, config in Config::arbitrary())| {
            test_config(seed, &config);
        });
    }

    #[test]
    fn test_field64_1() {
        test_sumcheck::<fields::Field64>();
    }

    #[test]
    #[ignore = "Somewhat expensive and redundant"]
    fn test_field64_2() {
        test_sumcheck::<fields::Field64_2>();
    }

    #[test]
    #[ignore = "Somewhat expensive and redundant"]
    fn test_field64_3() {
        test_sumcheck::<fields::Field64_3>();
    }

    #[test]
    #[ignore = "Somewhat expensive and redundant"]
    fn test_field128() {
        test_sumcheck::<fields::Field128>();
    }

    #[test]
    #[ignore = "Somewhat expensive and redundant"]
    fn test_field192() {
        test_sumcheck::<fields::Field192>();
    }

    #[test]
    #[ignore = "Somewhat expensive and redundant"]
    fn test_field256() {
        test_sumcheck::<fields::Field256>();
    }
}
