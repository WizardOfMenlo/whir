//! Allocation profiling binary for ZK WHIR.
//!
//! Run with:
//! ```bash
//! cargo run --bin alloc_report --features alloc-track --release
//! ```
//!
//! You can change `NUM_VARIABLES`, `NUM_POLYS`, and `NUM_POINTS` at the
//! top of `run()` to match the configuration you care about.

fn main() {
    #[cfg(feature = "alloc-track")]
    run();

    #[cfg(not(feature = "alloc-track"))]
    eprintln!(
        "This binary requires the `alloc-track` feature.\n\
         Run: cargo run --bin alloc_report --features alloc-track --release"
    );
}

#[cfg(feature = "alloc-track")]
#[global_allocator]
static ALLOC: whir::alloc_track::TrackingAllocator = whir::alloc_track::TrackingAllocator;

#[cfg(feature = "alloc-track")]
fn run() {
    use ark_std::rand::{rngs::StdRng, SeedableRng};
    use whir::{
        algebra::{
            fields::{Field64, Field64_2},
            polynomials::{CoefficientList, MultilinearPoint},
            Weights,
        },
        alloc_track, hash,
        parameters::{FoldingFactor, MultivariateParameters, ProtocolParameters, SoundnessType},
        protocols::{
            whir::Config,
            whir_zk::{ZkParams, ZkPreprocessingPolynomials},
        },
        transcript::{codecs::Empty, DomainSeparator, ProverState, VerifierState},
    };

    type F = Field64;
    type EF = Field64_2;

    /// ── Tunables ────────────────────────────────────────────────────
    const NUM_VARIABLES: usize = 20;
    const NUM_POLYS: usize = 2;
    const NUM_POINTS: usize = 10;

    let mut rng = StdRng::seed_from_u64(42);
    let num_coeffs = 1usize << NUM_VARIABLES;

    // ── Build config ─────────────────────────────────────────────────
    let mv = MultivariateParameters::new(NUM_VARIABLES);
    let params = ProtocolParameters {
        initial_statement: true,
        security_level: 32,
        pow_bits: 0,
        folding_factor: FoldingFactor::ConstantFromSecondRound(2, 4),
        soundness_type: SoundnessType::ConjectureList,
        starting_log_inv_rate: 1,
        batch_size: 1,
        hash_id: hash::SHA2,
    };
    let config = Config::<EF>::new(mv, &params);
    let zk_params = ZkParams::from_whir_params(&config);

    let helper_mv = MultivariateParameters::new(zk_params.ell + 1);
    let helper_params = ProtocolParameters {
        initial_statement: true,
        security_level: 32,
        pow_bits: 0,
        folding_factor: FoldingFactor::Constant(4),
        soundness_type: SoundnessType::ConjectureList,
        starting_log_inv_rate: 1,
        batch_size: zk_params.helper_batch_size(NUM_POLYS),
        hash_id: hash::SHA2,
    };
    let helper_config = Config::<EF>::new(helper_mv, &helper_params);

    eprintln!("╔══════════════════════════════════════════════════════════════════════╗");
    eprintln!("║  ZK WHIR Allocation Report                                         ║");
    eprintln!("╠══════════════════════════════════════════════════════════════════════╣");
    eprintln!(
        "║  num_variables = {:>4}                                              ║",
        NUM_VARIABLES
    );
    eprintln!(
        "║  num_polys     = {:>4}                                              ║",
        NUM_POLYS
    );
    eprintln!(
        "║  num_points    = {:>4}                                              ║",
        NUM_POINTS
    );
    eprintln!(
        "║  ell (ZK)      = {:>4}                                              ║",
        zk_params.ell
    );
    eprintln!(
        "║  mu  (ZK)      = {:>4}                                              ║",
        zk_params.mu
    );
    eprintln!(
        "║  WHIR rounds   = {:>4}                                              ║",
        config.n_rounds()
    );
    eprintln!("╚══════════════════════════════════════════════════════════════════════╝");
    eprintln!();

    // ── Build polynomials and preprocessings ─────────────────────────
    eprintln!("── setup ──────────────────────────────────────────────────────────");
    let mut snap = alloc_track::Snapshot::now();

    let polynomials: Vec<CoefficientList<F>> = (0..NUM_POLYS)
        .map(|i| {
            CoefficientList::new(
                (0..num_coeffs)
                    .map(|j| F::from((i * num_coeffs + j + 1) as u64))
                    .collect(),
            )
        })
        .collect();
    alloc_track::report("setup::build_polynomials", &snap);
    snap = alloc_track::Snapshot::now();

    let preprocessings: Vec<ZkPreprocessingPolynomials<EF>> = (0..NUM_POLYS)
        .map(|_| ZkPreprocessingPolynomials::<EF>::sample(&mut rng, zk_params.clone()))
        .collect();
    alloc_track::report("setup::sample_preprocessing", &snap);
    snap = alloc_track::Snapshot::now();

    // ── Build weights and evaluations ────────────────────────────────
    let mut weights = Vec::new();
    let mut evaluations = Vec::new();
    for _ in 0..NUM_POINTS {
        let point = MultilinearPoint::rand(&mut rng, NUM_VARIABLES);
        weights.push(Weights::evaluation(point.clone()));
        for poly in &polynomials {
            evaluations.push(poly.mixed_evaluate(config.embedding(), &point));
        }
    }
    let weight_refs: Vec<&Weights<EF>> = weights.iter().collect();
    let poly_refs: Vec<&CoefficientList<F>> = polynomials.iter().collect();
    alloc_track::report("setup::weights_and_evaluations", &snap);

    // ── Transcript setup ─────────────────────────────────────────────
    let ds = DomainSeparator::protocol(&config)
        .session(&String::from("alloc-report"))
        .instance(&Empty);

    // ══════════════════════════════════════════════════════════════════
    //  COMMIT
    // ══════════════════════════════════════════════════════════════════
    eprintln!();
    eprintln!("── commit_zk ──────────────────────────────────────────────────────");
    let mut prover_state = ProverState::new_std(&ds);
    snap = alloc_track::Snapshot::now();
    let zk_witness = config.commit_zk(
        &mut prover_state,
        &poly_refs,
        &helper_config,
        &preprocessings.iter().collect::<Vec<_>>(),
    );
    eprintln!("  ──────────────────────────────────────────────────────────────");
    alloc_track::report("commit_zk  TOTAL", &snap);

    // ══════════════════════════════════════════════════════════════════
    //  PROVE
    // ══════════════════════════════════════════════════════════════════
    eprintln!();
    eprintln!("── prove_zk ───────────────────────────────────────────────────────");
    snap = alloc_track::Snapshot::now();
    let (_point, _evals) = config.prove_zk(
        &mut prover_state,
        &poly_refs,
        &zk_witness,
        &helper_config,
        &weight_refs,
        &evaluations,
    );
    eprintln!("  ──────────────────────────────────────────────────────────────");
    alloc_track::report("prove_zk   TOTAL", &snap);

    // ══════════════════════════════════════════════════════════════════
    //  VERIFY
    // ══════════════════════════════════════════════════════════════════
    let proof = prover_state.proof();
    eprintln!();
    eprintln!("── verify_zk ──────────────────────────────────────────────────────");
    snap = alloc_track::Snapshot::now();

    let mut verifier_state = VerifierState::new_std(&ds, &proof);
    let f_hat_commitments: Vec<_> = (0..NUM_POLYS)
        .map(|_| config.receive_commitment(&mut verifier_state).unwrap())
        .collect();
    let f_hat_refs: Vec<_> = f_hat_commitments.iter().collect();
    let helper_commitment = helper_config
        .receive_commitment(&mut verifier_state)
        .unwrap();
    alloc_track::report("verify_zk::receive_commitments", &snap);
    snap = alloc_track::Snapshot::now();

    let result = config.verify_zk(
        &mut verifier_state,
        &f_hat_refs,
        &helper_commitment,
        &helper_config,
        &zk_params,
        &weight_refs,
        &evaluations,
    );
    assert!(result.is_ok(), "Verification failed: {result:?}");
    eprintln!("  ──────────────────────────────────────────────────────────────");
    alloc_track::report("verify_zk  TOTAL", &snap);

    eprintln!();
    eprintln!("✓ Verification passed. All allocation counts above are per full run.");
}
