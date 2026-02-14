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
            whir::Config as WhirConfig,
            whir_zk::{self, ZkParams},
        },
        transcript::{codecs::Empty, ProverState, VerifierState},
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
    // Compute ZK params from main config to size the helper
    let zk_params = ZkParams::from_whir_params(&WhirConfig::<EF>::new(mv, &params));

    let helper_mv = MultivariateParameters::new(zk_params.num_helper_variables + 1);
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
    let zk_config = whir_zk::Config::new(mv, &params, helper_mv, &helper_params);

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
        "║  helper_vars   = {:>4}                                              ║",
        zk_params.num_helper_variables
    );
    eprintln!(
        "║  witness_vars  = {:>4}                                              ║",
        zk_params.num_witness_variables
    );
    eprintln!(
        "║  WHIR rounds   = {:>4}                                              ║",
        zk_config.main.n_rounds()
    );
    eprintln!("╚══════════════════════════════════════════════════════════════════════╝");
    eprintln!();

    // ── Build polynomials ────────────────────────────────────────────
    eprintln!("── setup ──────────────────────────────────────────────────────────");
    let mut snap = alloc_track::Snapshot::now();

    let polynomials: Vec<CoefficientList<F>> = (0..NUM_POLYS)
        .map(|poly_idx| {
            CoefficientList::new(
                (0..num_coeffs)
                    .map(|coeff_idx| F::from((poly_idx * num_coeffs + coeff_idx + 1) as u64))
                    .collect(),
            )
        })
        .collect();
    alloc_track::report("setup::build_polynomials", &snap);
    snap = alloc_track::Snapshot::now();

    // ── Build weights and evaluations ────────────────────────────────
    let mut weights = Vec::new();
    let mut evaluations = Vec::new();
    for _ in 0..NUM_POINTS {
        let point = MultilinearPoint::rand(&mut rng, NUM_VARIABLES);
        weights.push(Weights::evaluation(point.clone()));
        for poly in &polynomials {
            evaluations.push(poly.mixed_evaluate(zk_config.main.embedding(), &point));
        }
    }
    let weight_refs: Vec<&Weights<EF>> = weights.iter().collect();
    let poly_refs: Vec<&CoefficientList<F>> = polynomials.iter().collect();
    alloc_track::report("setup::weights_and_evaluations", &snap);

    // ── Transcript setup ─────────────────────────────────────────────
    let ds = zk_config
        .domain_separator()
        .session(&String::from("alloc-report"))
        .instance(&Empty);

    // ══════════════════════════════════════════════════════════════════
    //  COMMIT
    // ══════════════════════════════════════════════════════════════════
    eprintln!();
    eprintln!("── commit ─────────────────────────────────────────────────────────");
    let mut prover_state = ProverState::new_std(&ds);
    snap = alloc_track::Snapshot::now();
    let zk_witness = zk_config.commit(&mut prover_state, &poly_refs);
    eprintln!("  ──────────────────────────────────────────────────────────────");
    alloc_track::report("commit     TOTAL", &snap);

    // ══════════════════════════════════════════════════════════════════
    //  PROVE
    // ══════════════════════════════════════════════════════════════════
    eprintln!();
    eprintln!("── prove ──────────────────────────────────────────────────────────");
    snap = alloc_track::Snapshot::now();
    let (_point, _evals) = zk_config.prove(
        &mut prover_state,
        &poly_refs,
        &zk_witness,
        &weight_refs,
        &evaluations,
    );
    eprintln!("  ──────────────────────────────────────────────────────────────");
    alloc_track::report("prove      TOTAL", &snap);

    // ══════════════════════════════════════════════════════════════════
    //  VERIFY
    // ══════════════════════════════════════════════════════════════════
    let proof = prover_state.proof();
    eprintln!();
    eprintln!("── verify ─────────────────────────────────────────────────────────");
    snap = alloc_track::Snapshot::now();

    let mut verifier_state = VerifierState::new_std(&ds, &proof);
    let (f_hat_commitments, helper_commitment) = zk_config
        .receive_commitments(&mut verifier_state, NUM_POLYS)
        .unwrap();
    let f_hat_refs: Vec<_> = f_hat_commitments.iter().collect();
    alloc_track::report("verify::receive_commitments", &snap);
    snap = alloc_track::Snapshot::now();

    let result = zk_config.verify(
        &mut verifier_state,
        &f_hat_refs,
        &helper_commitment,
        &weight_refs,
        &evaluations,
    );
    assert!(result.is_ok(), "Verification failed: {result:?}");
    eprintln!("  ──────────────────────────────────────────────────────────────");
    alloc_track::report("verify     TOTAL", &snap);

    eprintln!();
    eprintln!("✓ Verification passed. All allocation counts above are per full run.");
}
