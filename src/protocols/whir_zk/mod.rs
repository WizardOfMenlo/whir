pub mod api;
mod committer;
pub mod prefold;
mod prover;
pub mod utils;
mod verifier;

pub use api::{ProverInput, VerifierInput};
pub use prefold::{PrefoldGroupCommitments, PrefoldGroupInput, PrefoldLevelConfig};
pub use utils::{HelperEvaluations, ZkParams, ZkPreprocessingPolynomials, ZkWitness};

#[cfg(test)]
mod tests {

    use ark_std::rand::{rngs::StdRng, SeedableRng};

    use super::*;
    use crate::{
        algebra::{
            fields::{Field64, Field64_2},
            polynomials::{CoefficientList, MultilinearPoint},
            Weights,
        },
        hash,
        parameters::{FoldingFactor, MultivariateParameters, ProtocolParameters, SoundnessType},
        protocols::whir::Config,
        transcript::{codecs::Empty, DomainSeparator, ProverState, VerifierState},
    };

    /// Field type used in the tests.
    type F = Field64;

    /// Extension field type used in the tests.
    type EF = Field64_2;

    /// What to tamper with in soundness tests.
    #[derive(Clone, Copy, Debug, PartialEq)]
    enum Tamper {
        /// Honest run — no tampering.
        None,
        /// Corrupt a native evaluation passed to the verifier.
        NativeEval,
        /// Corrupt a byte in the middle of the serialised proof.
        ProofBytes,
        /// Corrupt a prefold group's evaluation claim.
        PrefoldEval,
    }

    /// Run a full `batch_prove_zk` → `batch_verify_zk` round-trip and return
    /// whether verification succeeds.
    ///
    /// Each element of `group_specs` is `(arity, num_polys, num_points)`.
    /// Exactly one group must have `arity == n_min` (the native group).
    fn batch_api_roundtrip(
        n_min: usize,
        folding_factor: FoldingFactor,
        soundness_type: SoundnessType,
        pow_bits: usize,
        group_specs: &[(usize, usize, usize)],
        tamper: Tamper,
    ) -> bool {
        let mut rng = StdRng::seed_from_u64(12345);

        let mv_params = MultivariateParameters::<EF>::new(n_min);
        let whir_params = ProtocolParameters {
            initial_statement: true,
            security_level: 32,
            pow_bits,
            folding_factor,
            soundness_type,
            starting_log_inv_rate: 1,
            batch_size: 1,
            hash_id: hash::SHA2,
        };
        let main_config = Config::new(mv_params, &whir_params);

        // Build polynomial groups
        let mut all_polys: Vec<Vec<CoefficientList<F>>> = Vec::new();
        let mut all_weights: Vec<Vec<Weights<EF>>> = Vec::new();
        let mut all_evals: Vec<Vec<EF>> = Vec::new();

        for (gi, &(arity, num_polys, num_points)) in group_specs.iter().enumerate() {
            let num_coeffs = 1usize << arity;
            let polys: Vec<CoefficientList<F>> = (0..num_polys)
                .map(|i| {
                    CoefficientList::new(
                        (0..num_coeffs)
                            .map(|j| F::from(((gi + 1) * 1000 + i * num_coeffs + j + 1) as u64))
                            .collect(),
                    )
                })
                .collect();

            let mut weights = Vec::new();
            let mut evals = Vec::new();
            for _ in 0..num_points {
                let point = MultilinearPoint::rand(&mut rng, arity);
                weights.push(Weights::evaluation(point.clone()));
                for poly in &polys {
                    evals.push(poly.mixed_evaluate(main_config.embedding(), &point));
                }
            }

            all_polys.push(polys);
            all_weights.push(weights);
            all_evals.push(evals);
        }

        // Build ProverInputs
        let poly_ref_vecs: Vec<Vec<&CoefficientList<F>>> =
            all_polys.iter().map(|ps| ps.iter().collect()).collect();

        let prover_inputs: Vec<ProverInput<'_, EF>> = group_specs
            .iter()
            .enumerate()
            .map(|(gi, _)| {
                ProverInput::new(
                    poly_ref_vecs[gi].clone(),
                    all_weights[gi].clone(),
                    all_evals[gi].clone(),
                )
            })
            .collect();

        // Prove
        let ds = DomainSeparator::protocol(&main_config)
            .session(&format!("Batch API Test at {}:{}", file!(), line!()))
            .instance(&Empty);
        let mut prover_state = ProverState::new_std(&ds);

        let _ =
            main_config.batch_prove_zk(&mut prover_state, &whir_params, &prover_inputs, &mut rng);
        let mut proof = prover_state.proof();

        // Apply tampering
        if tamper == Tamper::ProofBytes && proof.narg_string.len() > 10 {
            let mid = proof.narg_string.len() / 2;
            proof.narg_string[mid] ^= 0xFF;
        }

        let mut claims: Vec<VerifierInput<EF>> = prover_inputs
            .iter()
            .map(|g| g.to_verifier_input())
            .collect();

        match tamper {
            Tamper::NativeEval => {
                if let Some(c) = claims
                    .iter_mut()
                    .find(|c| c.arity == n_min && !c.evaluations.is_empty())
                {
                    c.evaluations[0] += EF::from(42u64);
                }
            }
            Tamper::PrefoldEval => {
                if let Some(c) = claims
                    .iter_mut()
                    .find(|c| c.arity > n_min && !c.evaluations.is_empty())
                {
                    c.evaluations[0] += EF::from(42u64);
                }
            }
            _ => {}
        }

        // Verify
        let mut verifier_state = VerifierState::new_std(&ds, &proof);
        main_config
            .batch_verify_zk(&mut verifier_state, &whir_params, &claims)
            .is_ok()
    }

    /// Various (num_variables, folding_factor) combos with 0 / 1 / 2 evaluation
    /// constraints, single polynomial per group.
    #[test]
    fn test_batch_api_basic_configs() {
        let configs: &[(usize, usize, usize)] = &[
            // (num_variables, folding_factor, num_points)
            (10, 2, 0),
            (10, 2, 1),
            (10, 2, 2),
            (12, 2, 1),
            (12, 3, 1),
            (12, 4, 1),
        ];

        for &(n, k, num_pts) in configs {
            eprintln!();
            dbg!(n, k, num_pts);

            let ok = batch_api_roundtrip(
                n,
                FoldingFactor::Constant(k),
                SoundnessType::ConjectureList,
                0,
                &[(n, 1, num_pts)],
                Tamper::None,
            );
            assert!(ok, "failed for n={n}, k={k}, num_pts={num_pts}");
        }
    }

    /// Multiple polynomials at native arity (batched proving / verification).
    #[test]
    fn test_batch_api_multi_polynomial() {
        let configs: &[(usize, usize, usize, usize)] = &[
            // (num_variables, folding_factor, num_points, num_polynomials)
            (10, 2, 1, 2),
            (10, 2, 2, 2),
            (12, 2, 1, 3),
            (12, 3, 2, 2),
        ];

        for &(n, k, num_pts, num_polys) in configs {
            eprintln!();
            dbg!(n, k, num_pts, num_polys);

            let ok = batch_api_roundtrip(
                n,
                FoldingFactor::Constant(k),
                SoundnessType::ConjectureList,
                0,
                &[(n, num_polys, num_pts)],
                Tamper::None,
            );
            assert!(
                ok,
                "failed for n={n}, k={k}, pts={num_pts}, polys={num_polys}"
            );
        }
    }

    /// Multi-arity prefold: polynomials across several arities, including a
    /// group with zero constraints and fold depths 1–3.
    #[test]
    fn test_batch_api_multi_arity() {
        let ok = batch_api_roundtrip(
            10,
            FoldingFactor::Constant(2),
            SoundnessType::ConjectureList,
            0,
            &[
                (10, 2, 1), // native: 2 polynomials, 1 constraint
                (11, 1, 1), // fold_depth = 1
                (12, 1, 1), // fold_depth = 2
                (13, 1, 0), // fold_depth = 3, no constraints
            ],
            Tamper::None,
        );
        assert!(ok, "multi-arity prefold must verify");
    }

    /// Proof-of-work, alternative soundness types, and mixed folding factors.
    #[test]
    fn test_batch_api_advanced_configs() {
        // PoW
        let ok = batch_api_roundtrip(
            12,
            FoldingFactor::Constant(2),
            SoundnessType::ConjectureList,
            5,
            &[(12, 1, 2)],
            Tamper::None,
        );
        assert!(ok, "PoW test failed");

        // Soundness types
        for st in [SoundnessType::ProvableList, SoundnessType::UniqueDecoding] {
            let ok = batch_api_roundtrip(
                12,
                FoldingFactor::Constant(2),
                st,
                0,
                &[(12, 1, 1)],
                Tamper::None,
            );
            assert!(ok, "soundness type {st:?} failed");
        }

        // Mixed folding
        let ok = batch_api_roundtrip(
            12,
            FoldingFactor::ConstantFromSecondRound(3, 3),
            SoundnessType::ConjectureList,
            0,
            &[(12, 1, 1)],
            Tamper::None,
        );
        assert!(ok, "mixed folding failed");
    }

    /// Soundness: tampered native eval, corrupted proof bytes, tampered prefold eval.
    #[test]
    fn test_batch_api_soundness() {
        let groups = &[
            (10, 1, 1), // native
            (11, 1, 1), // prefold
        ];

        for tamper in [Tamper::NativeEval, Tamper::ProofBytes, Tamper::PrefoldEval] {
            eprintln!();
            dbg!(tamper);

            let ok = batch_api_roundtrip(
                10,
                FoldingFactor::Constant(2),
                SoundnessType::ConjectureList,
                0,
                groups,
                tamper,
            );
            assert!(!ok, "verification must FAIL with tamper {tamper:?}");
        }
    }
}
