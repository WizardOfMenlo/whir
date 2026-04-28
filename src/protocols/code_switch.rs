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
        fields::FieldWithSize,
        geometric_accumulate, lift, mixed_dot, random_vector, scalar_mul, univariate_evaluate,
    },
    engines::EngineId,
    hash::Hash,
    protocols::{
        geometric_challenge::geometric_challenge,
        irs_commit::{
            num_ood_samples, Commitment as IrsCommitment, Config as IrsConfig,
            Witness as IrsWitness,
        },
    },
    transcript::{
        Codec, Decoding, DuplexSpongeInterface, ProverMessage, ProverState, VerificationResult,
        VerifierMessage, VerifierState,
    },
    type_info::Typed,
};

/// Code-switching IOR config with optional ZK.
#[derive(Clone, PartialEq, Eq, Debug, Hash, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct Config<M: Embedding> {
    pub embedding: Typed<M>,
    pub source: IrsConfig<M>,
    pub target: IrsConfig<Identity<M::Target>>,
    pub out_domain_samples: usize,
    pub mask_commit: Option<IrsConfig<Identity<M::Target>>>,
}

/// Next stage's query budgets for ZK encoding (Prop 3.19).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash, Serialize, Deserialize)]
pub struct ZkQueryBudget {
    /// Queries the next stage makes to the target oracle g.
    pub target: usize,
    /// Queries the next stage makes to the mask oracle s.
    pub mask: usize,
}

/// ZK mask oracle: the message and its IRS witness.
#[derive(Clone, Debug)]
pub struct MaskOracle<F: Field> {
    pub message: Vec<F>,
    pub witness: IrsWitness<F>,
}

/// Prover output from the code-switch.
#[derive(Clone, Debug)]
pub struct Witness<F: Field> {
    pub message: Vec<F>,
    pub target_witness: IrsWitness<F>,
    pub mask: Option<MaskOracle<F>>,
}

/// Verifier output from the code-switch.
#[derive(Clone, Debug)]
pub struct Commitment<F: Field> {
    pub target: IrsCommitment<F>,
    pub mask: Option<IrsCommitment<F>>,
}

impl<M: Embedding> Config<M> {
    /// `zk`: `None` for non-ZK, `Some(budget)` for ZK with next stage's query budgets.
    pub fn new(
        security_target: f64,
        unique_decoding: bool,
        hash_id: EngineId,
        source_config: IrsConfig<M>,
        target_log_inv_rate: usize,
        target_interleaving_depth: usize,
        zk: Option<ZkQueryBudget>,
    ) -> Self
    where
        M: Default,
        M::Target: Default,
    {
        assert!(target_log_inv_rate > 0);
        assert!(target_interleaving_depth > 0);

        let source_message_length = source_config.message_length();
        let target_rate = 0.5_f64.powf(target_log_inv_rate as f64);
        let target_vector_size = source_message_length * target_interleaving_depth;

        let mut target_config = IrsConfig::<Identity<M::Target>>::new(
            security_target,
            unique_decoding,
            hash_id,
            1,
            target_vector_size,
            target_interleaving_depth,
            target_rate,
        );

        let mask_commit = zk.map(|budget| {
            assert!(budget.target > 0, "ZK requires nonzero target query budget");
            assert!(budget.mask > 0, "ZK requires nonzero mask query budget");
            // TODO : find a better way to set mask length for ZK code-switch
            // leaving right now for code switching. This will change in parameter
            // selection
            let target_slack =
                target_config.codeword_length - target_config.message_length();
            assert!(
                budget.target <= target_slack,
                "budget.target ({}) exceeds target codeword slack ({codeword_len} - {msg_len} = {target_slack})",
                budget.target,
                codeword_len = target_config.codeword_length,
                msg_len = target_config.message_length(),
            );
            target_config.mask_length = budget.target;
            target_config.recompute_security_parameters(security_target, unique_decoding);

            let source_randomness_len = source_config.mask_length * source_config.num_messages();
            assert!(
                source_randomness_len > 0,
                "ZK code-switch requires source_config.mask_length > 0"
            );
            // TODO : move the mask config out for shared mask tree per iteration
            // inputs to the new function will also change
            let mut mask_config = IrsConfig::<Identity<M::Target>>::new(
                security_target,
                unique_decoding,
                hash_id,
                1,
                source_randomness_len,
                1,
                source_config.rate(),
            );
            let mask_slack =
                mask_config.codeword_length - mask_config.message_length();
            assert!(
                budget.mask <= mask_slack,
                "budget.mask ({}) exceeds mask codeword slack ({codeword_len} - {msg_len} = {mask_slack})",
                budget.mask,
                codeword_len = mask_config.codeword_length,
                msg_len = mask_config.message_length(),
            );
            mask_config.mask_length = budget.mask;
            mask_config.recompute_security_parameters(security_target, unique_decoding);
            mask_config
        });

        assert!(
            source_config.mask_length == 0 || mask_commit.is_some(),
            "source with ZK randomness requires ZK code-switch (mask_commit)"
        );

        let (list_size, degree) = mask_commit.as_ref().map_or_else(
            || (target_config.list_size(), source_message_length),
            |mask_cfg| {
                (
                    target_config.list_size() * mask_cfg.list_size(),
                    source_message_length + mask_cfg.message_length(),
                )
            },
        );
        let out_domain_samples = num_ood_samples(
            unique_decoding,
            security_target,
            M::Target::field_size_bits(),
            list_size,
            degree,
        );

        Self {
            embedding: Typed::<M>::default(),
            source: source_config,
            target: target_config,
            out_domain_samples,
            mask_commit,
        }
    }

    /// Length of the covector for this code-switch.
    pub fn covector_length(&self) -> usize {
        self.source.masked_message_length()
    }

    /// Prove the code-switch
    #[cfg_attr(feature = "tracing", instrument(skip_all))]
    pub fn prove<H, R>(
        &self,
        prover_state: &mut ProverState<H, R>,
        message: Vec<M::Target>,
        masks: &[M::Source],
        witness: &IrsWitness<M::Source, M::Target>,
        covector: &mut [M::Target],
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
        assert_eq!(
            self.source.num_messages(),
            1,
            "code-switch only supports num_messages() == 1 (single polynomial)"
        );
        assert_eq!(
            masks.len(),
            self.source.mask_length * self.source.num_messages()
        );
        assert!(
            self.mask_commit.is_some() == (self.target.mask_length > 0),
            "mask config and target mask_length must agree"
        );

        // Step 1a: g := Enc_{C'}(f, r') — Construction 9.7 Step 1, p.55
        let target_witness = self.target.commit(prover_state, &[&message]);

        // Step 1b: s := Enc_{C_zk}((r || padding), r'') — Construction 9.7 Step 1, p.55
        #[allow(clippy::option_if_let_else)]
        let mask = if let Some(mask_config) = &self.mask_commit {
            let mask_msg_len = mask_config.message_length();
            let r_embedded = lift(self.source.embedding(), masks);
            let embedded_randomness_len = r_embedded.len();
            let mut mask_msg = Vec::with_capacity(mask_msg_len);
            mask_msg.extend_from_slice(&r_embedded);
            let random_padding: Vec<M::Target> =
                random_vector(prover_state.rng(), mask_msg_len - embedded_randomness_len);
            mask_msg.extend_from_slice(&random_padding);
            let witness = mask_config.commit(prover_state, &[&mask_msg]);
            Some(MaskOracle {
                message: mask_msg,
                witness,
            })
        } else {
            None
        };

        // Step 2-3: OOD challenge + answers — Construction 9.7 Steps 2-3, p.55
        // TODO : check the private zero evader for code switch protocol.
        let ood_points: Vec<M::Target> = prover_state.verifier_message_vec(self.out_domain_samples);
        let msg_len = message.len();
        for &point in &ood_points {
            let f_eval = univariate_evaluate(&message, point);
            if let Some(ref mask_oracle) = mask {
                let mask_msg_eval = univariate_evaluate(&mask_oracle.message, point);
                let shift = point.pow([msg_len as u64]);
                prover_state.prover_message(&(f_eval + shift * mask_msg_eval));
            } else {
                prover_state.prover_message(&f_eval);
            }
        }

        // Step 4: in-domain queries — Construction 9.7 Step 4, p.55
        let source_evaluations = self.source.open(prover_state, &[witness]);

        // Step 4.1: batching — Construction 9.7 Step 4, p.55
        let num_ood = self.out_domain_samples;
        let num_in_domain = source_evaluations.matrix.len();
        let batching_coeffs =
            geometric_challenge::<_, M::Target>(prover_state, 1 + num_ood + num_in_domain);
        let (&original_sl_coeff, constraint_rlc_coeffs) = batching_coeffs.split_first().unwrap();
        let (ood_rlc_coeffs, in_domain_rlc_coeffs) = constraint_rlc_coeffs.split_at(num_ood);

        // Covector update — sl' from Completeness proof (p.55-56)
        let eval_points = lift(self.source.embedding(), &source_evaluations.points);
        let all_points: Vec<_> = ood_points.iter().chain(&eval_points).copied().collect();
        let pows: Vec<_> = ood_rlc_coeffs
            .iter()
            .chain(in_domain_rlc_coeffs)
            .copied()
            .collect();
        scalar_mul(covector, original_sl_coeff);
        geometric_accumulate(covector, pows, &all_points);

        Witness {
            message,
            target_witness,
            mask,
        }
    }

    /// Verify the code-switch
    #[cfg_attr(feature = "tracing", instrument(skip_all))]
    pub fn verify<H>(
        &self,
        verifier_state: &mut VerifierState<H>,
        sum: &mut M::Target,
        commitment: &IrsCommitment<M::Target>,
    ) -> VerificationResult<Commitment<M::Target>>
    where
        H: DuplexSpongeInterface,
        Standard: Distribution<M::Target>,
        M::Target: Codec<[H::U]>,
        u8: Decoding<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        assert_eq!(
            self.source.num_messages(),
            1,
            "code-switch only supports num_messages() == 1 (single polynomial)"
        );
        assert!(
            self.mask_commit.is_some() == (self.target.mask_length > 0),
            "mask config and target mask_length must agree"
        );

        // Step 1: commitments — Construction 9.7 Step 1, p.55
        let target_commitment = self.target.receive_commitment(verifier_state)?;
        let mask_commitment = self
            .mask_commit
            .as_ref()
            .map(|mask_cfg| mask_cfg.receive_commitment(verifier_state))
            .transpose()?;

        // Step 2-3: OOD — Construction 9.7 Steps 2-3, p.55
        let _ood_points: Vec<M::Target> =
            verifier_state.verifier_message_vec(self.out_domain_samples);
        let ood_answers: Vec<M::Target> =
            verifier_state.prover_messages_vec(self.out_domain_samples)?;

        // Step 4: source opening — Construction 9.7 Step 4, p.55
        let source_evaluations = self.source.verify(verifier_state, &[commitment])?;

        // Step 4.1: batching + μ' — Construction 9.7 Decision phase, p.55
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

        Ok(Commitment {
            target: target_commitment,
            mask: mask_commitment,
        })
    }
}

impl<M: Embedding> fmt::Display for Config<M> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Config(source={}, target={}, ood_samples={}",
            self.source, self.target, self.out_domain_samples,
        )?;
        if let Some(mask) = &self.mask_commit {
            write!(f, ", mask={mask}")?;
        }
        write!(f, ")")
    }
}

#[cfg(test)]
mod tests {
    use ark_std::rand::{
        distributions::Standard, prelude::Distribution, rngs::StdRng, Rng, SeedableRng,
    };
    use proptest::{bool, prelude::Strategy, proptest, sample::select, strategy::Just};

    use super::*;
    use crate::{
        algebra::{embedding::Identity, fields, ntt, random_vector},
        transcript::{codecs::U64, DomainSeparator},
    };

    impl<M: Embedding> Config<M> {
        pub fn arbitrary(embedding: M) -> impl Strategy<Value = Self>
        where
            M: 'static,
        {
            let valid_sizes = (1..=256)
                .filter(|&n| ntt::next_order::<M::Source>(n) == Some(n))
                .collect::<Vec<_>>();

            let emb1 = embedding.clone();
            let emb2 = embedding;

            (select(valid_sizes), 0_usize..=3, bool::ANY)
                .prop_flat_map(move |(size, mask_length, masked)| {
                    let source_mask = if masked { mask_length.max(1) } else { 0 };
                    let source = IrsConfig::arbitrary(emb1.clone(), 1, size, source_mask, 1);
                    (source, Just(masked))
                })
                .prop_flat_map(|(source, masked)| {
                    let msg_len = source.message_length();
                    let rnd_len = source.mask_length * source.num_messages();
                    let target_mask = usize::from(masked);

                    let target = IrsConfig::<Identity<M::Target>>::arbitrary(
                        Identity::new(),
                        1,
                        msg_len,
                        target_mask,
                        1,
                    );

                    let mask = if masked && rnd_len > 0 {
                        IrsConfig::<Identity<M::Target>>::arbitrary(
                            Identity::new(),
                            1,
                            rnd_len,
                            0,
                            1,
                        )
                        .prop_map(Some)
                        .boxed()
                    } else {
                        Just(None).boxed()
                    };

                    (Just(source), target, mask, 0_usize..=5)
                })
                .prop_map(
                    move |(source, target, mask_commit, out_domain_samples)| Self {
                        embedding: Typed::new(emb2.clone()),
                        source,
                        target,
                        out_domain_samples,
                        mask_commit,
                    },
                )
        }
    }

    fn test_config<F: Field + Codec<[u8]>>(seed: u64, config: &Config<Identity<F>>)
    where
        Standard: Distribution<F>,
        Hash: ProverMessage<[u8]>,
    {
        crate::tests::init();

        let instance = U64(seed);
        let ds = DomainSeparator::protocol(config)
            .session(&format!("Test at {}:{}", file!(), line!()))
            .instance(&instance);
        let mut rng = StdRng::seed_from_u64(seed);
        let message: Vec<F> = random_vector(&mut rng, config.source.message_length());
        let target_mu: F = rng.gen();

        let msg_len = config.source.message_length();
        let mut covector: Vec<F> = random_vector(&mut rng, msg_len);
        covector.resize(config.covector_length(), F::ZERO);

        let mut prover_state = ProverState::new_std(&ds);
        let source_witness = config.source.commit(&mut prover_state, &[&message]);
        let witness = config.prove(
            &mut prover_state,
            message.clone(),
            &source_witness.masks,
            &source_witness,
            &mut covector,
        );
        let proof = prover_state.proof();

        let mut verifier_state = VerifierState::new_std(&ds, &proof);
        let source_commitment = config
            .source
            .receive_commitment(&mut verifier_state)
            .unwrap();
        let mut verifier_sum = target_mu;
        let commitments = config
            .verify(&mut verifier_state, &mut verifier_sum, &source_commitment)
            .unwrap();

        // Transcript fully consumed — prover and verifier are in sync
        verifier_state.check_eof().unwrap();
        // Mask commitment present if ZK mode
        assert_eq!(commitments.mask.is_some(), config.mask_commit.is_some());
        assert_eq!(witness.message, message);
        assert_eq!(witness.mask.is_some(), config.mask_commit.is_some());
    }

    fn test<F: Field + Codec<[u8]> + 'static>()
    where
        Standard: Distribution<F>,
        Hash: ProverMessage<[u8]>,
    {
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
}
