//! Code-switching IOR: R_{C, C_zk, sl} → R_{C', C_zk, sl'}
//!
//! Reduces a proximity claim about oracle f (source code C) to a proximity
//! claim about oracle g (target code C'). Supports optional ZK via mask oracle.

use std::fmt;

use ark_ff::{AdditiveGroup, Field};
use ark_std::rand::{distributions::Standard, prelude::Distribution, CryptoRng, RngCore};
use serde::{Deserialize, Serialize};
#[cfg(feature = "tracing")]
use tracing::instrument;

use crate::{
    algebra::{
        dot,
        embedding::{Embedding, Identity},
        mixed_dot, univariate_evaluate,
    },
    engines::EngineId,
    hash::Hash,
    protocols::{
        geometric_challenge::geometric_challenge,
        irs_commit::{Commitment as IrsCommitment, Config as IrsConfig, Witness as IrsWitness},
    },
    transcript::{
        Codec, Decoding, DuplexSpongeInterface, ProverMessage, ProverState, VerificationResult,
        VerifierMessage, VerifierState,
    },
    type_info::Typed,
    utils::zip_strict,
};

/// Code-switching IOR config with optional ZK.
#[derive(Clone, PartialEq, Eq, Debug, Hash, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct Config<M: Embedding> {
    pub embedding: Typed<M>,
    pub source: IrsConfig<M>,
    pub target: IrsConfig<Identity<M::Target>>,
    pub ood_samples: usize,
    pub zk: bool,
}

#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct TargetConfigParams {
    pub security_target: f64,
    pub unique_decoding: bool,
    pub target_log_inv_rate: usize,
    pub target_interleaving_depth: usize,
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
}

/// Verifier output from the code-switch.
#[derive(Clone, Debug)]
pub struct Commitment<F: Field> {
    pub target: IrsCommitment<F>,
}

impl<M: Embedding> Config<M> {
    /// `zk`: whether ZK mode is active (mask oracle expected in prove/verify).
    pub fn new(
        hash_id: EngineId,
        source_config: IrsConfig<M>,
        target_config_params: &TargetConfigParams,
        ood_samples: usize,
        zk: bool,
    ) -> Self
    where
        M: Default,
        M::Target: Default,
    {
        assert!(target_config_params.target_log_inv_rate > 0);
        assert!(target_config_params.target_interleaving_depth > 0);
        assert!(!target_config_params.unique_decoding);
        assert!(
            ood_samples > 0,
            "code-switch requires OOD samples for cross-oracle consistency"
        );

        let source_message_length = source_config.message_length();
        let target_rate = 0.5_f64.powf(target_config_params.target_log_inv_rate as f64);
        let target_vector_size =
            source_message_length * target_config_params.target_interleaving_depth;

        let target_config = IrsConfig::<Identity<M::Target>>::new(
            target_config_params.security_target,
            target_config_params.unique_decoding,
            hash_id,
            1,
            target_vector_size,
            target_config_params.target_interleaving_depth,
            target_rate,
        );

        Self {
            embedding: Typed::<M>::default(),
            source: source_config,
            target: target_config,
            ood_samples,
            zk,
        }
    }

    /// Prove the code-switch
    #[cfg_attr(feature = "tracing", instrument(skip_all))]
    pub fn prove<H, R>(
        &self,
        prover_state: &mut ProverState<H, R>,
        message: Vec<M::Target>,
        witness: &IrsWitness<M::Source, M::Target>,
        covector: &mut [M::Target],
        mask_oracle: Option<&MaskOracle<M::Target>>,
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
        assert_eq!(
            self.source.num_messages(),
            1,
            "code-switch only supports num_messages() == 1 (single polynomial)"
        );
        assert_eq!(
            mask_oracle.is_some(),
            self.zk,
            "mask_oracle presence must match zk flag"
        );

        // Step 1: g := Enc_{C'}(f, r') — Construction 9.7 Step 1, p.55
        let target_witness = self.target.commit(prover_state, &[&message]);

        // Step 2-3: OOD challenge + answers — Construction 9.7 Steps 2-3, p.55
        let ood_points: Vec<M::Target> = prover_state.verifier_message_vec(self.ood_samples);
        let msg_len = message.len();
        for &point in &ood_points {
            let f_eval = univariate_evaluate(&message, point);
            if let Some(mask_oracle) = mask_oracle {
                let mask_msg_eval = univariate_evaluate(&mask_oracle.message, point);
                let shift = point.pow([msg_len as u64]);
                prover_state.prover_message(&(f_eval + shift * mask_msg_eval));
            } else {
                prover_state.prover_message(&f_eval);
            }
        }

        // Step 4: in-domain queries — Construction 9.7 Step 4, p.55
        let source_evaluations = self.source.open(prover_state, &[witness]);

        // Step 5: batching — Construction 9.7 Step 5, p.55
        let num_ood = self.ood_samples;
        let num_in_domain = source_evaluations.matrix.len();
        let batching_coeffs =
            geometric_challenge::<_, M::Target>(prover_state, 1 + num_ood + num_in_domain);
        let (&original_sl_coeff, constraint_rlc_coeffs) = batching_coeffs.split_first().unwrap();
        let (ood_rlc_coeffs, in_domain_rlc_coeffs) = constraint_rlc_coeffs.split_at(num_ood);

        // Covector update — sl' from Completeness proof (p.55-56)
        let embedding = self.source.embedding();
        let eval_points: Vec<_> = source_evaluations
            .points
            .iter()
            .map(|&p| embedding.map(p))
            .collect();
        let all_points: Vec<_> = ood_points.iter().chain(&eval_points).copied().collect();
        let mut pows: Vec<_> = ood_rlc_coeffs
            .iter()
            .chain(in_domain_rlc_coeffs)
            .copied()
            .collect();
        for c in &mut *covector {
            let mut sum = M::Target::ZERO;
            for (pow, &point) in zip_strict(&mut pows, &all_points) {
                sum += *pow;
                *pow *= point;
            }
            *c = *c * original_sl_coeff + sum;
        }

        Witness {
            message,
            target_witness,
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

        // Step 1: target commitment — Construction 9.7 Step 1, p.55
        let target_commitment = self.target.receive_commitment(verifier_state)?;

        // Step 2-3: OOD — Construction 9.7 Steps 2-3, p.55
        let _ood_points: Vec<M::Target> = verifier_state.verifier_message_vec(self.ood_samples);
        let ood_answers: Vec<M::Target> = verifier_state.prover_messages_vec(self.ood_samples)?;

        // Step 4: source opening — Construction 9.7 Step 4, p.55
        let source_evaluations = self.source.verify(verifier_state, &[commitment])?;

        // Step 5-6: batching + μ' — Construction 9.7 Decision phase, p.55
        let num_ood = self.ood_samples;
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
        })
    }
}

impl<M: Embedding> fmt::Display for Config<M> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Config(source={}, target={}, ood_samples={}, zk={})",
            self.source, self.target, self.ood_samples, self.zk
        )
    }
}

#[cfg(test)]
mod tests {
    use ark_std::rand::{
        distributions::Standard, prelude::Distribution, rngs::StdRng, SeedableRng,
    };
    use proptest::{prelude::Strategy, proptest, sample::select, strategy::Just};

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

            let emb2 = embedding.clone();

            // Generate source config, then derive target from it
            (select(valid_sizes), 0_usize..=3, 0_usize..=5)
                .prop_flat_map(move |(size, mask_length, ood_samples)| {
                    let source = IrsConfig::arbitrary(embedding.clone(), 1, size, mask_length, 1);
                    (source, Just(ood_samples))
                })
                .prop_flat_map(|(source, ood_samples)| {
                    let msg_len = source.message_length();
                    let target = IrsConfig::<Identity<M::Target>>::arbitrary(
                        Identity::new(),
                        1,
                        msg_len,
                        0,
                        1,
                    );
                    (Just(source), target, Just(ood_samples))
                })
                .prop_map(move |(source, target, ood_samples)| Self {
                    embedding: Typed::new(emb2.clone()),
                    source,
                    target,
                    ood_samples,
                    // TODO : ZK path requires orchestrator; tested at integration level
                    zk: false,
                })
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
        let mut covector: Vec<F> = random_vector(&mut rng, config.source.message_length());

        // Honest input: μ = ⟨f, sl⟩
        let initial_mu = dot(&message, &covector);

        let mut prover_state = ProverState::new_std(&ds);
        let source_witness = config.source.commit(&mut prover_state, &[&message]);
        // Non-ZK test path: no mask oracle
        let witness = config.prove(
            &mut prover_state,
            message.clone(),
            &source_witness,
            &mut covector,
            None,
        );
        let proof = prover_state.proof();

        let mut verifier_state = VerifierState::new_std(&ds, &proof);
        let source_commitment = config
            .source
            .receive_commitment(&mut verifier_state)
            .unwrap();
        let mut verifier_sum = initial_mu;
        let _commitments = config
            .verify(&mut verifier_state, &mut verifier_sum, &source_commitment)
            .unwrap();

        // Transcript fully consumed — prover and verifier are in sync
        verifier_state.check_eof().unwrap();
        assert_eq!(witness.message, message);
        // Non ZK case check only
        if config.source.mask_length == 0 {
            assert_eq!(dot(&message, &covector), verifier_sum);
        }
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
