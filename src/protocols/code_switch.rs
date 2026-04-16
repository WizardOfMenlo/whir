//! Code-switching IOR: R_{C, C_zk, sl} → R_{C', C_zk, sl'}
//!
//! Reduces a proximity claim about oracle f (source code C) to a proximity
//! claim about oracle g (target code C'). Supports optional ZK via mask oracle.

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
        lift,
        linear_form::UnivariateEvaluation,
        mixed_dot, random_vector, univariate_evaluate,
    },
    hash::Hash,
    protocols::{
        geometric_challenge::geometric_challenge,
        irs_commit::{
            num_ood_samples, Commitment as IrsCommitment, Config as IrsConfig,
            Evaluations as IrsEvaluations, Witness as IrsWitness,
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
    pub ood_samples: usize,
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

/// Prover output from the code-switch.
#[derive(Clone, Debug)]
pub struct Witness<F: Field, G: Field = F> {
    pub target_witness: IrsWitness<F>,
    pub message: Vec<F>,
    pub source_evaluations: IrsEvaluations<G>,
    pub mask: Option<(Vec<F>, IrsWitness<F>)>,
}

/// OOD and in-domain constraint weights for a single oracle.
#[derive(Clone, Debug)]
pub struct ConstraintWeights<F: Field, G: Field = F> {
    pub ood: Vec<UnivariateEvaluation<F>>,
    pub in_domain: Vec<UnivariateEvaluation<G>>,
}

/// ZK mask oracle claim. OOD weights evaluate full mask; in-domain weights
/// evaluate only `mask_msg[..source_randomness_len]` (paper: ⟨(r,s), (G_C^s[x,·], 0)⟩).
#[derive(Clone, Debug)]
pub struct MaskClaimInfo<F: Field, G: Field = F> {
    pub commitment: IrsCommitment<F>,
    pub ood_rlc_coeffs: Vec<F>,
    pub in_domain_rlc_coeffs: Vec<F>,
    pub weights: ConstraintWeights<F, G>,
    pub source_randomness_len: usize,
}

impl<F: Field, G: Field> MaskClaimInfo<F, G> {
    /// Weighted mask contribution. In-domain points are lifted G → F via embedding.
    pub fn evaluate(
        &self,
        embedding: &impl Embedding<Source = G, Target = F>,
        mask_msg: &[F],
    ) -> F {
        // OOD: ⟨(r,s), ze^{→,i}_ood(ρ)⟩ = ρ^ℓ · (r,s)^(ρ)
        let ood_sum: F = self
            .ood_rlc_coeffs
            .iter()
            .zip(&self.weights.ood)
            .map(|(&c, w)| c * univariate_evaluate(mask_msg, w.point))
            .sum();

        // In-domain: ⟨(r,s), (G^s_C[x,·], 0)⟩ = φ(x)^ℓ · r̂(φ(x))
        let r_slice = &mask_msg[..self.source_randomness_len];
        let in_domain_sum: F = self
            .in_domain_rlc_coeffs
            .iter()
            .zip(&self.weights.in_domain)
            .map(|(&c, w)| c * univariate_evaluate(r_slice, embedding.map(w.point)))
            .sum();

        ood_sum + in_domain_sum
    }
}

/// Verifier output. Paper: Equation 9 (p.55).
#[derive(Clone, Debug)]
pub struct CodeSwitchClaim<F: Field, G: Field = F> {
    pub mu_prime: F,
    pub original_sl_coeff: F,
    pub target_commitment: IrsCommitment<F>,
    pub ood_rlc_coeffs: Vec<F>,
    pub in_domain_rlc_coeffs: Vec<F>,
    pub source_weights: ConstraintWeights<F, G>,
    pub mask_info: Option<MaskClaimInfo<F, G>>,
    pub source_evaluations: IrsEvaluations<G>,
}

impl<M: Embedding> Config<M> {
    /// `zk`: `None` for non-ZK, `Some(budget)` for ZK with next stage's query budgets.
    pub fn new(
        security_target: f64,
        unique_decoding: bool,
        hash_id: crate::engines::EngineId,
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
            target_config.mask_length = budget.target;
            target_config.reparameterise_security(security_target, unique_decoding);

            let source_randomness_len = source_config.mask_length * source_config.num_messages();
            assert!(
                source_randomness_len > 0,
                "ZK code-switch requires source_config.mask_length > 0"
            );
            let mut mask_config = IrsConfig::<Identity<M::Target>>::new(
                security_target,
                unique_decoding,
                hash_id,
                1,
                source_randomness_len,
                1,
                source_config.rate(),
            );
            mask_config.mask_length = budget.mask;
            mask_config.reparameterise_security(security_target, unique_decoding);
            mask_config
        });

        let (list_size, degree) = mask_commit.as_ref().map_or_else(
            || (target_config.list_size(), source_message_length),
            |mask_cfg| {
                (
                    target_config.list_size() * mask_cfg.list_size(),
                    source_message_length + mask_cfg.message_length(),
                )
            },
        );
        let ood_samples = num_ood_samples(
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
            ood_samples,
            mask_commit,
        }
    }

    #[cfg_attr(feature = "tracing", instrument(skip_all))]
    pub fn prove<H, R>(
        &self,
        prover_state: &mut ProverState<H, R>,
        message: Vec<M::Target>,
        masks: &[M::Source],
        witness: &IrsWitness<M::Source, M::Target>,
    ) -> Witness<M::Target, M::Source>
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
            masks.len(),
            self.source.mask_length * self.source.num_messages()
        );
        assert!(
            self.mask_commit.is_some() == (self.target.mask_length > 0),
            "mask config and target mask_length must agree"
        );

        // Step 1a: g := Enc_{C'}(f, r')
        let target_witness = self.target.commit(prover_state, &[&message]);

        // Step 1b: s := Enc_{C_zk}((r || padding), r'')
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
            Some((mask_msg, witness))
        } else {
            None
        };

        // Step 2-3: OOD challenge + answers
        let ood_points: Vec<M::Target> = prover_state.verifier_message_vec(self.ood_samples);
        let msg_len = message.len();
        for &point in &ood_points {
            let f_eval = univariate_evaluate(&message, point);
            if let Some((mask_msg, _)) = &mask {
                let mask_msg_eval = univariate_evaluate(mask_msg, point);
                let shift = point.pow([msg_len as u64]);
                prover_state.prover_message(&(f_eval + shift * mask_msg_eval));
            } else {
                prover_state.prover_message(&f_eval);
            }
        }

        // Step 4: in-domain queries
        let source_evaluations = self.source.open(prover_state, &[witness]);

        // Step 5: batching (advance transcript to stay in sync with verify)
        geometric_challenge::<_, M::Target>(
            prover_state,
            1 + self.ood_samples + source_evaluations.matrix.len(),
        );

        Witness {
            target_witness,
            message,
            source_evaluations,
            mask,
        }
    }

    #[cfg_attr(feature = "tracing", instrument(skip_all))]
    pub fn verify<H>(
        &self,
        verifier_state: &mut VerifierState<H>,
        mu: M::Target,
        commitment: &IrsCommitment<M::Target>,
    ) -> VerificationResult<CodeSwitchClaim<M::Target, M::Source>>
    where
        H: DuplexSpongeInterface,
        Standard: Distribution<M::Target>,
        M::Target: Codec<[H::U]>,
        u8: Decoding<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        assert!(
            self.mask_commit.is_some() == (self.target.mask_length > 0),
            "mask config and target mask_length must agree"
        );

        // Step 1: commitments
        let target_commitment = self.target.receive_commitment(verifier_state)?;
        let mask_commitment = self
            .mask_commit
            .as_ref()
            .map(|mask_cfg| mask_cfg.receive_commitment(verifier_state))
            .transpose()?;

        // Step 2-3: OOD
        let ood_points: Vec<M::Target> = verifier_state.verifier_message_vec(self.ood_samples);
        let ood_answers: Vec<M::Target> = verifier_state.prover_messages_vec(self.ood_samples)?;

        // Step 4: source opening
        let source_evaluations = self.source.verify(verifier_state, &[commitment])?;

        // Step 5: batching
        let num_ood = self.ood_samples;
        let num_in_domain = source_evaluations.matrix.len();
        let coeffs = geometric_challenge(verifier_state, 1 + num_ood + num_in_domain);
        let (&original_sl_coeff, all_rlc_coeffs) = coeffs.split_first().unwrap();
        let (ood_rlc_coeffs, in_domain_rlc_coeffs) = all_rlc_coeffs.split_at(num_ood);

        // Step 6: μ'
        let mu_prime = original_sl_coeff * mu
            + dot(ood_rlc_coeffs, &ood_answers)
            + mixed_dot(
                self.source.embedding(),
                in_domain_rlc_coeffs,
                &source_evaluations.matrix,
            );

        // Step 7: constraint weights
        let source_msg_len = self.source.message_length();
        let source_weights = ConstraintWeights {
            ood: ood_points
                .iter()
                .map(|&point| UnivariateEvaluation::new(point, source_msg_len))
                .collect(),
            in_domain: source_evaluations.evaluators(source_msg_len).collect(),
        };

        // Mask weights with shift ρ^ℓ baked into RLC coefficients.
        // In-domain shift lifted G → F via embedding (ring homomorphism).
        let mask_info = self.mask_commit.as_ref().map(|mask_config| {
            let source_randomness_len = self.source.mask_length * self.source.num_messages();
            let weights = ConstraintWeights {
                ood: ood_points
                    .iter()
                    .map(|&point| UnivariateEvaluation::new(point, mask_config.message_length()))
                    .collect(),
                in_domain: source_evaluations
                    .evaluators(source_randomness_len)
                    .collect(),
            };
            let embedding = self.source.embedding();
            let mask_ood_rlc_coeffs: Vec<M::Target> = ood_rlc_coeffs
                .iter()
                .zip(&weights.ood)
                .map(|(&c, w)| c * w.point.pow([source_msg_len as u64]))
                .collect();
            let mask_in_domain_rlc_coeffs: Vec<M::Target> = in_domain_rlc_coeffs
                .iter()
                .zip(&weights.in_domain)
                .map(|(&c, w)| c * embedding.map(w.point.pow([source_msg_len as u64])))
                .collect();

            MaskClaimInfo {
                commitment: mask_commitment.expect("mask commitment must exist when masked"),
                ood_rlc_coeffs: mask_ood_rlc_coeffs,
                in_domain_rlc_coeffs: mask_in_domain_rlc_coeffs,
                weights,
                source_randomness_len,
            }
        });

        Ok(CodeSwitchClaim {
            mu_prime,
            original_sl_coeff,
            target_commitment,
            ood_rlc_coeffs: ood_rlc_coeffs.to_vec(),
            in_domain_rlc_coeffs: in_domain_rlc_coeffs.to_vec(),
            source_weights,
            mask_info,
            source_evaluations,
        })
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
        algebra::{embedding::Identity, fields, linear_form::Evaluate, ntt, random_vector},
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
                    let target_mask = if masked { 1 } else { 0 };

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
                .prop_map(move |(source, target, mask_commit, ood_samples)| Self {
                    embedding: Typed::new(emb2.clone()),
                    source,
                    target,
                    ood_samples,
                    mask_commit,
                })
        }
    }

    /// Σ rlc[i] * weight[i].evaluate(vector)
    fn weighted_eval<F: Field>(rlc: &[F], weights: &[UnivariateEvaluation<F>], v: &[F]) -> F {
        let e = Identity::<F>::new();
        rlc.iter()
            .zip(weights)
            .map(|(&c, w)| c * w.evaluate(&e, v))
            .sum()
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

        // Prover
        let mut prover_state = ProverState::new_std(&ds);
        let source_witness = config.source.commit(&mut prover_state, &[&message]);
        let witness = config.prove(
            &mut prover_state,
            message.clone(),
            &source_witness.masks,
            &source_witness,
        );
        let proof = prover_state.proof();

        // Verifier
        let mut verifier_state = VerifierState::new_std(&ds, &proof);
        let commitment = config
            .source
            .receive_commitment(&mut verifier_state)
            .unwrap();
        let claim = config
            .verify(&mut verifier_state, target_mu, &commitment)
            .unwrap();
        verifier_state.check_eof().unwrap();

        // Completeness: μ' = ν_1·μ + ⟨f, source_weights⟩ + ⟨mask_msg, mask_weights⟩
        assert_eq!(witness.message, message);

        let msg_sum = weighted_eval(&claim.ood_rlc_coeffs, &claim.source_weights.ood, &message)
            + weighted_eval(
                &claim.in_domain_rlc_coeffs,
                &claim.source_weights.in_domain,
                &message,
            );

        let mask_sum = witness
            .mask
            .as_ref()
            .zip(claim.mask_info.as_ref())
            .map_or(F::ZERO, |((mm, _), info)| {
                info.evaluate(&Identity::<F>::new(), mm)
            });

        assert_eq!(
            msg_sum + mask_sum,
            claim.mu_prime - claim.original_sl_coeff * target_mu,
        );

        // Structural invariants
        assert_eq!(claim.ood_rlc_coeffs.len(), claim.source_weights.ood.len());
        assert_eq!(
            claim.in_domain_rlc_coeffs.len(),
            claim.source_weights.in_domain.len()
        );
        assert_eq!(claim.mask_info.is_some(), config.mask_commit.is_some());
        if let Some(ref info) = claim.mask_info {
            assert_eq!(info.ood_rlc_coeffs.len(), info.weights.ood.len());
            assert_eq!(
                info.in_domain_rlc_coeffs.len(),
                info.weights.in_domain.len()
            );
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
