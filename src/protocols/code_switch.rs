//! Code-switching IOR: R_{C, C_zk, sl} → R_{C', C_zk, sl'}
//!
//! Reduces a proximity claim about oracle f (source code C) to a proximity
//! claim about oracle g (target code C'). Supports optional ZK via mask oracle.

use std::fmt;

use ark_ff::Field;
use ark_std::rand::{distributions::Standard, prelude::Distribution, CryptoRng, RngCore};
use serde::{Deserialize, Serialize};
#[cfg(feature = "tracing")]
use tracing::instrument;

use crate::{
    algebra::{
        dot,
        embedding::{Embedding, Identity},
        geometric_accumulate, lift, mixed_dot, scalar_mul, univariate_evaluate,
    },
    hash::Hash,
    protocols::{
        geometric_challenge::geometric_challenge,
        irs_commit::{Commitment as IrsCommitment, Config as IrsConfig, Witness as IrsWitness},
    },
    transcript::{
        Codec, Decoding, DuplexSpongeInterface, ProverMessage, ProverState, VerificationResult,
        VerifierMessage, VerifierState,
    },
};

/// Code-switching IOR config with optional ZK.
#[derive(Clone, PartialEq, Eq, Debug, Hash, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct Config<M: Embedding> {
    pub source: IrsConfig<M>,
    pub target: IrsConfig<Identity<M::Target>>,
    pub message_mask_length: usize, // l_zk
    pub out_domain_samples: usize,
}

/// Prover output from the code-switch.
#[must_use]
#[derive(Clone, Debug)]
pub struct Witness<F: Field> {
    pub message: Vec<F>,
    pub target_witness: IrsWitness<F>,
}

/// Verifier output from the code-switch.
pub type Commitment<F> = IrsCommitment<F>;

/// Mask input for the code-switch prover.
// TODO : This may be removed after parameter selection PR
pub enum MaskInput<'a, F> {
    Disabled,
    Enabled(&'a [F]),
}

impl<M: Embedding> Config<M> {
    /// Create a code-switch config.
    ///
    /// The orchestrator is responsible for:
    /// - Setting `target_config.mask_length` for ZK mode before passing it in.
    /// - Computing `out_domain_samples` from the security budget.
    /// - Setting `message_mask_length` = mask oracle message length (0 for non-ZK).
    pub fn new(
        source_config: IrsConfig<M>,
        target_config: IrsConfig<Identity<M::Target>>,
        out_domain_samples: usize,
        message_mask_length: usize,
    ) -> Self
    where
        M: Default,
    {
        assert_eq!(
            source_config.num_vectors, 1,
            "code-switch requires a single source vector"
        );
        // TODO : add support for interleaving depth l
        // Current commitment.open() gives l values per position and
        // we need to remove the ood_points opening and account for
        // it in code switch.
        // Current accumulation will break if interleaving depth > 1
        // Leaving this for now as after irs_commit is refactored we
        // can add the proper implementation here in code switch.
        assert_eq!(
            source_config.interleaving_depth, 1,
            "Currently code switch supports interleaving depth = 1"
        );
        assert_eq!(
            source_config.interleaving_depth, target_config.interleaving_depth,
            "source and target interleaving_depth must match"
        );
        assert_eq!(
            target_config.vector_size,
            source_config.message_length() * target_config.interleaving_depth,
            "target vector_size must equal source message_length × target interleaving_depth"
        );
        // Theorem 9.6: ℓ_zk ≥ r (mask oracle must cover source randomness).
        if message_mask_length > 0 {
            assert!(
                message_mask_length >= source_config.mask_length,
                "message_mask_length ({message_mask_length}) must be >= source randomness length ({})",
                source_config.mask_length,
            );
        }
        assert!(
            source_config.mask_length == 0 || message_mask_length > 0,
            "source with message_mask_length > 0 requires ZK mode (message_mask_length > 0)"
        );

        Self {
            source: source_config,
            target: target_config,
            message_mask_length,
            out_domain_samples,
        }
    }

    /// Length of the covector for this code-switch.
    pub fn covector_length(&self) -> usize {
        self.source.message_length() + self.message_mask_length.max(self.source.mask_length)
    }

    /// Prove the code-switch.
    #[cfg_attr(feature = "tracing", instrument(skip_all))]
    pub fn prove<H, R>(
        &self,
        prover_state: &mut ProverState<H, R>,
        message: Vec<M::Target>,
        witness: &IrsWitness<M::Source, M::Target>,
        covector: &mut [M::Target],
        mask_input: &MaskInput<'_, M::Target>,
    ) -> Witness<M::Target>
    where
        H: DuplexSpongeInterface,
        R: RngCore + CryptoRng,
        Standard: Distribution<M::Target>,
        M::Target: Codec<[H::U]>,
        u8: Decoding<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        assert_eq!(message.len(), self.source.message_length());
        assert_eq!(covector.len(), self.covector_length());
        let mask_msg: Option<&[M::Target]> = match &mask_input {
            MaskInput::Disabled => {
                assert_eq!(
                    self.message_mask_length, 0,
                    "MaskInput::Disabled requires message_mask_length == 0"
                );
                None
            }
            MaskInput::Enabled(mask) => {
                assert_eq!(
                    mask.len(),
                    self.message_mask_length,
                    "mask_msg length must equal message_mask_length"
                );
                Some(mask)
            }
        };

        // Step 1: g := Enc_{C'}(f, r') — Construction 9.7 Step 1, p.55
        let target_witness = self.target.commit(prover_state, &[&message]);

        // Step 2-3: OOD challenge + answers — Construction 9.7 Steps 2-3, p.55
        // y := ze_ood(ρ) · [f; r; s] = f(α) + α^ℓ · (r,s)(α)
        let ood_points: Vec<M::Target> = prover_state.verifier_message_vec(self.out_domain_samples);
        let msg_len = message.len();
        for &point in &ood_points {
            let f_eval = univariate_evaluate(&message, point);
            if let Some(mask) = mask_msg {
                let mask_eval = univariate_evaluate(mask, point);
                let shift = point.pow([msg_len as u64]);
                prover_state.prover_message(&(f_eval + shift * mask_eval));
            } else {
                prover_state.prover_message(&f_eval);
            }
        }

        // Step 4: in-domain queries — Construction 9.7 Step 4, p.55
        let source_evaluations = self.source.open(prover_state, &[witness]);

        // Step 5: batching — Construction 9.7 Step 5, p.55
        let num_ood = self.out_domain_samples;
        let num_in_domain = source_evaluations.matrix.len();
        let batching_coeffs =
            geometric_challenge::<_, M::Target>(prover_state, 1 + num_ood + num_in_domain);
        let (&original_sl_coeff, constraint_rlc_coeffs) = batching_coeffs.split_first().unwrap();
        let (ood_rlc_coeffs, in_domain_rlc_coeffs) = constraint_rlc_coeffs.split_at(num_ood);

        // Covector update — sl' from Completeness proof (p.55-56)
        let eval_points = lift(self.source.embedding(), &source_evaluations.points);
        scalar_mul(covector, original_sl_coeff);
        if self.message_mask_length == 0 {
            // Non-ZK: single accumulate over all points
            let all_points: Vec<_> = ood_points.iter().chain(&eval_points).copied().collect();
            let pows: Vec<_> = ood_rlc_coeffs
                .iter()
                .chain(in_domain_rlc_coeffs)
                .copied()
                .collect();
            geometric_accumulate(covector, pows, &all_points);
        } else {
            // ZK: OOD contributes to full [f; r; s], in-domain only to [f; r]
            geometric_accumulate(covector, ood_rlc_coeffs.to_vec(), &ood_points);
            geometric_accumulate(
                &mut covector[..self.source.masked_message_length()],
                in_domain_rlc_coeffs.to_vec(),
                &eval_points,
            );
        }

        Witness {
            message,
            target_witness,
        }
    }

    /// Verify the code-switch.
    ///
    /// Returns the target commitment. In ZK mode, the caller **must**
    /// additionally run `mask_proximity::verify` on the mask commitment
    /// to ensure the mask oracle `(r, s)` is close to a `C_zk` codeword.
    /// Without this check, soundness is not guaranteed.
    #[cfg_attr(feature = "tracing", instrument(skip_all))]
    pub fn verify<H>(
        &self,
        verifier_state: &mut VerifierState<H>,
        sum: &mut M::Target,
        commitment: &Commitment<M::Target>,
    ) -> VerificationResult<Commitment<M::Target>>
    where
        H: DuplexSpongeInterface,
        Standard: Distribution<M::Target>,
        M::Target: Codec<[H::U]>,
        u8: Decoding<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        // Step 1: target commitment — Construction 9.7 Step 1, p.55
        // Mask oracle is committed in the shared mask tree by the orchestrator.
        let target_commitment = self.target.receive_commitment(verifier_state)?;

        // Step 2-3: OOD — Construction 9.7 Steps 2-3, p.55
        // In ZK mode, ood_answers = f(α) + α^ℓ · (r,s)(α) where (r,s) is
        // the mask oracle message committed in the shared tree.
        let _ood_points: Vec<M::Target> =
            verifier_state.verifier_message_vec(self.out_domain_samples);
        let ood_answers: Vec<M::Target> =
            verifier_state.prover_messages_vec(self.out_domain_samples)?;

        // Step 4: source opening — Construction 9.7 Step 4, p.55
        let source_evaluations = self.source.verify(verifier_state, &[commitment])?;

        // Step 5: batching + μ' — Construction 9.7 Decision phase, p.55
        let num_ood = self.out_domain_samples;
        let num_in_domain = source_evaluations.matrix.len();
        let coeffs = geometric_challenge(verifier_state, 1 + num_ood + num_in_domain);
        let (&original_sl_coeff, all_rlc_coeffs) = coeffs.split_first().unwrap();
        let (ood_rlc_coeffs, in_domain_rlc_coeffs) = all_rlc_coeffs.split_at(num_ood);

        *sum = original_sl_coeff * *sum
            + dot(ood_rlc_coeffs, &ood_answers)
            + mixed_dot(
                self.source.embedding(),
                in_domain_rlc_coeffs,
                &source_evaluations.matrix,
            );

        Ok(target_commitment)
    }
}

impl<M: Embedding> fmt::Display for Config<M> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CodeSwitch(source={}, target={}, ood={}, zk={})",
            self.source,
            self.target,
            self.out_domain_samples,
            self.message_mask_length != 0,
        )
    }
}

#[cfg(test)]
mod tests {
    use ark_std::rand::{
        distributions::Standard, prelude::Distribution, rngs::StdRng, Rng, SeedableRng,
    };
    use proptest::{
        bool, prelude::Strategy, prop_assume, proptest, sample::select, strategy::Just,
    };

    use super::*;
    use crate::{
        algebra::{embedding::Identity, fields, ntt, random_vector},
        transcript::{codecs::U64, DomainSeparator},
    };

    impl<M: Embedding> Config<M> {
        pub fn arbitrary(embedding: M) -> impl Strategy<Value = Self>
        where
            M: Default + 'static,
        {
            let valid_sizes = (1..=256)
                .filter(|&n| ntt::next_order::<M::Source>(n) == Some(n))
                .collect::<Vec<_>>();

            // fresh_s_len: extra padding beyond r for ZK privacy (0..=3)
            (
                select(valid_sizes),
                0_usize..=3,
                bool::ANY,
                0_usize..=5,
                0_usize..=3,
            )
                .prop_flat_map(move |(size, src_mask_len, zk, ood, fresh_s_len)| {
                    let source_mask = if zk { src_mask_len.max(1) } else { 0 };
                    let source = IrsConfig::arbitrary(embedding.clone(), 1, size, source_mask, 1);
                    (source, Just(zk), Just(ood), Just(fresh_s_len))
                })
                .prop_flat_map(|(source, zk, ood, fresh_s_len)| {
                    let msg_len = source.message_length();
                    let target_mask = usize::from(zk);
                    let target = IrsConfig::<Identity<M::Target>>::arbitrary(
                        Identity::new(),
                        1,
                        msg_len,
                        target_mask,
                        1,
                    );
                    let r = source.mask_length * source.num_messages();
                    // message_mask_length = r + fresh_s_len (paper: ℓ_zk ≥ r, Section 9.2)
                    let message_mask_length = if zk { r + fresh_s_len } else { 0 };
                    (Just(source), target, Just(ood), Just(message_mask_length))
                })
                .prop_map(move |(source, target, ood, message_mask_length)| {
                    Self::new(source, target, ood, message_mask_length)
                })
        }
    }

    /// Simulate what the orchestrator does: build (r || fresh_s) from the
    /// source IRS witness. Returns empty vec in non-ZK mode.
    fn build_mask_msg<F: Field>(
        config: &Config<Identity<F>>,
        source_witness: &IrsWitness<F>,
        rng: &mut impl RngCore,
    ) -> Vec<F>
    where
        Standard: Distribution<F>,
    {
        if config.message_mask_length == 0 {
            return Vec::new();
        }
        let mut mask = lift(config.source.embedding(), &source_witness.masks);
        mask.extend(random_vector::<F>(
            rng,
            config.message_mask_length - mask.len(),
        ));
        mask
    }

    fn mask_input<F>(mask_msg: &[F]) -> MaskInput<'_, F> {
        if mask_msg.is_empty() {
            MaskInput::Disabled
        } else {
            MaskInput::Enabled(mask_msg)
        }
    }

    fn test_config<F: Field + Codec<[u8]>>(seed: u64, config: &Config<Identity<F>>)
    where
        Standard: Distribution<F>,
        Hash: ProverMessage<[u8]>,
    {
        let mut rng = StdRng::seed_from_u64(seed);
        let message: Vec<F> = random_vector(&mut rng, config.source.message_length());
        let initial_sum: F = rng.gen();

        let mut covector: Vec<F> = random_vector(&mut rng, config.source.message_length());
        covector.resize(config.covector_length(), F::ZERO);

        let instance = U64(seed);
        let ds = DomainSeparator::protocol(config)
            .session(&format!("Test at {}:{}", file!(), line!()))
            .instance(&instance);
        let mut prover_state = ProverState::new_std(&ds);
        let source_witness = config.source.commit(&mut prover_state, &[&message]);
        let mask_msg = build_mask_msg(config, &source_witness, &mut rng);

        let witness = config.prove(
            &mut prover_state,
            message.clone(),
            &source_witness,
            &mut covector,
            &mask_input(&mask_msg),
        );
        let proof = prover_state.proof();

        let mut verifier_state = VerifierState::new_std(&ds, &proof);
        let source_commitment = config
            .source
            .receive_commitment(&mut verifier_state)
            .unwrap();
        let mut verifier_sum = initial_sum;
        let _ = config
            .verify(&mut verifier_state, &mut verifier_sum, &source_commitment)
            .unwrap();
        verifier_state.check_eof().unwrap();
        assert_eq!(witness.message, message);
    }

    fn test_ior_identity_config<F: Field + Codec<[u8]>>(seed: u64, config: &Config<Identity<F>>)
    where
        Standard: Distribution<F>,
        Hash: ProverMessage<[u8]>,
    {
        let mut rng = StdRng::seed_from_u64(seed);
        let message: Vec<F> = random_vector(&mut rng, config.source.message_length());

        let mut covector: Vec<F> = random_vector(&mut rng, config.source.message_length());
        covector.resize(config.covector_length(), F::ZERO);

        let instance = U64(seed);
        let ds = DomainSeparator::protocol(config)
            .session(&format!("Test at {}:{}", file!(), line!()))
            .instance(&instance);
        let mut prover_state = ProverState::new_std(&ds);
        let source_witness = config.source.commit(&mut prover_state, &[&message]);
        let mask_msg = build_mask_msg(config, &source_witness, &mut rng);

        // h = [f; mask_msg] or just f
        let h: Vec<F> = if mask_msg.is_empty() {
            message.clone()
        } else {
            message.iter().chain(mask_msg.iter()).copied().collect()
        };
        let initial_mu = dot(&h, &covector);

        let _witness = config.prove(
            &mut prover_state,
            message,
            &source_witness,
            &mut covector,
            &mask_input(&mask_msg),
        );
        let proof = prover_state.proof();

        let mut verifier_state = VerifierState::new_std(&ds, &proof);
        let source_commitment = config
            .source
            .receive_commitment(&mut verifier_state)
            .unwrap();
        let mut verifier_sum = initial_mu;
        let _ = config
            .verify(&mut verifier_state, &mut verifier_sum, &source_commitment)
            .unwrap();
        verifier_state.check_eof().unwrap();

        assert_eq!(dot(&h, &covector), verifier_sum);
    }

    fn test_tampered_ood_config<F: Field + Codec<[u8]>>(seed: u64, config: &Config<Identity<F>>)
    where
        Standard: Distribution<F>,
        Hash: ProverMessage<[u8]>,
    {
        let instance = U64(seed);
        let ds = DomainSeparator::protocol(config)
            .session(&format!("Test at {}:{}", file!(), line!()))
            .instance(&instance);
        let mut rng = StdRng::seed_from_u64(seed);
        let message: Vec<F> = random_vector(&mut rng, config.source.message_length());

        let mut covector: Vec<F> = random_vector(&mut rng, config.source.message_length());
        covector.resize(config.covector_length(), F::ZERO);
        let initial_mu = dot(&message, &covector);

        // Commit honest f, but prove with f' ≠ f
        let mut prover_state = ProverState::new_std(&ds);
        let source_witness = config.source.commit(&mut prover_state, &[&message]);
        let mut tampered = message.clone();
        tampered[0] += F::ONE;
        let _witness = config.prove(
            &mut prover_state,
            tampered,
            &source_witness,
            &mut covector,
            &MaskInput::Disabled,
        );
        let proof = prover_state.proof();

        let mut verifier_state = VerifierState::new_std(&ds, &proof);
        let source_commitment = config
            .source
            .receive_commitment(&mut verifier_state)
            .unwrap();
        let mut verifier_sum = initial_mu;
        let _ = config
            .verify(&mut verifier_state, &mut verifier_sum, &source_commitment)
            .unwrap();
        verifier_state.check_eof().unwrap();

        // Sum diverges — downstream sumcheck would reject
        assert_ne!(dot(&message, &covector), verifier_sum);
    }

    fn test<F: Field + Codec<[u8]> + 'static>()
    where
        Standard: Distribution<F>,
        Hash: ProverMessage<[u8]>,
    {
        crate::tests::init();
        let configs = Config::arbitrary(Identity::<F>::new());
        proptest!(|(seed: u64, config in configs)| {
            test_config(seed, &config);
        });
    }

    #[test]
    fn test_field64() {
        test::<fields::Field64>();
    }

    #[test]
    #[ignore = "Somewhat expensive and redundant"]
    fn test_field64_2() {
        test::<fields::Field64_2>();
    }

    #[test]
    #[ignore = "Somewhat expensive and redundant"]
    fn test_field64_3() {
        test::<fields::Field64_3>();
    }

    #[test]
    #[ignore = "Somewhat expensive and redundant"]
    fn test_field128() {
        test::<fields::Field128>();
    }

    #[test]
    #[ignore = "Somewhat expensive and redundant"]
    fn test_field192() {
        test::<fields::Field192>();
    }

    #[test]
    #[ignore = "Somewhat expensive and redundant"]
    fn test_field256() {
        test::<fields::Field256>();
    }

    #[test]
    fn test_ior_identity() {
        crate::tests::init();
        let configs = Config::arbitrary(Identity::<fields::Field64>::new());
        proptest!(|(seed: u64, config in configs)| {
            prop_assume!(config.source.in_domain_samples > 0);
            test_ior_identity_config(seed, &config);
        });
    }

    #[test]
    fn test_tampered_ood() {
        crate::tests::init();
        let configs = Config::arbitrary(Identity::<fields::Field64>::new());
        proptest!(|(seed: u64, config in configs)| {
            prop_assume!(
                config.message_mask_length == 0
                    && config.source.mask_length == 0
                    && config.out_domain_samples > 0
            );
            test_tampered_ood_config(seed, &config);
        });
    }
}
