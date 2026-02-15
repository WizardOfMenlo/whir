//! Allocation profiling binary for ZK WHIR.
//!
//! Uses the tracing-based allocation tracker ported from ProveKit.
//! Each `#[instrument]`-annotated function automatically prints timing and
//! memory stats (peak, local, allocation count) as a tree.
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
static ALLOCATOR: whir::alloc_track::ProfilingAllocator =
    whir::alloc_track::ProfilingAllocator::new();

#[cfg(feature = "alloc-track")]
#[allow(clippy::too_many_lines)]
fn run() {
    use ark_std::rand::{rngs::StdRng, SeedableRng};
    use whir::{
        algebra::{
            fields::{Field64, Field64_2},
            polynomials::{CoefficientList, MultilinearPoint},
            Weights,
        },
        hash,
        parameters::{FoldingFactor, MultivariateParameters, ProtocolParameters, SoundnessType},
        protocols::whir_zk,
        transcript::{codecs::Empty, DomainSeparator, ProverState, VerifierState},
    };

    type F = Field64;
    type EF = Field64_2;

    /// ── Tunables ────────────────────────────────────────────────────
    const NUM_VARIABLES: usize = 20;
    const NUM_POLYS: usize = 2;
    const NUM_POINTS: usize = 10;

    // Initialize the tracing subscriber with memory stats layer.
    whir::alloc_track::init_subscriber(&ALLOCATOR);

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
    let zk_config = whir_zk::Config::<EF>::new(mv, &params, FoldingFactor::Constant(4), NUM_POLYS);

    eprintln!("╔══════════════════════════════════════════════════════════════════════╗");
    eprintln!("║  ZK WHIR Allocation Report  (tracing-based)                        ║");
    eprintln!("╠══════════════════════════════════════════════════════════════════════╣");
    eprintln!(
        "║  num_variables     = {NUM_VARIABLES:>4}                                          ║",
    );
    eprintln!("║  num_polys         = {NUM_POLYS:>4}                                          ║",);
    eprintln!("║  num_points        = {NUM_POINTS:>4}                                          ║",);
    eprintln!(
        "║  blinding_vars     = {:>4}                                          ║",
        zk_config.num_blinding_variables()
    );
    eprintln!(
        "║  witness_vars      = {:>4}                                          ║",
        zk_config.num_witness_variables()
    );
    eprintln!(
        "║  WHIR rounds       = {:>4}                                          ║",
        zk_config.blinded_commitment.n_rounds()
    );
    eprintln!("╚══════════════════════════════════════════════════════════════════════╝");
    eprintln!();

    // ── Build polynomials ────────────────────────────────────────────
    let polynomials: Vec<CoefficientList<F>> = (0..NUM_POLYS)
        .map(|poly_idx| {
            CoefficientList::new(
                (0..num_coeffs)
                    .map(|coeff_idx| F::from((poly_idx * num_coeffs + coeff_idx + 1) as u64))
                    .collect(),
            )
        })
        .collect();

    // ── Build weights and evaluations ────────────────────────────────
    let mut weights = Vec::new();
    let mut evaluations = Vec::new();
    for _ in 0..NUM_POINTS {
        let point = MultilinearPoint::rand(&mut rng, NUM_VARIABLES);
        weights.push(Weights::evaluation(point.clone()));
        for poly in &polynomials {
            evaluations.push(poly.mixed_evaluate(zk_config.blinded_commitment.embedding(), &point));
        }
    }
    let weight_refs: Vec<&Weights<EF>> = weights.iter().collect();
    let poly_refs: Vec<&CoefficientList<F>> = polynomials.iter().collect();

    // ── Transcript setup ─────────────────────────────────────────────
    let ds = DomainSeparator::protocol(&zk_config)
        .session(&String::from("alloc-report"))
        .instance(&Empty);

    // ══════════════════════════════════════════════════════════════════
    //  COMMIT + PROVE
    // ══════════════════════════════════════════════════════════════════
    let mut prover_state = ProverState::new_std(&ds);
    let zk_witness = zk_config.commit(&mut prover_state, &poly_refs);
    let (_point, _evals) = zk_config.prove(
        &mut prover_state,
        &poly_refs,
        &zk_witness,
        &weight_refs,
        &evaluations,
    );

    // ══════════════════════════════════════════════════════════════════
    //  VERIFY
    // ══════════════════════════════════════════════════════════════════
    let proof = prover_state.proof();
    let mut verifier_state = VerifierState::new_std(&ds, &proof);
    let commitment = zk_config
        .receive_commitments(&mut verifier_state, NUM_POLYS)
        .unwrap();
    let result = zk_config.verify(&mut verifier_state, &commitment, &weight_refs, &evaluations);
    assert!(result.is_ok(), "Verification failed: {result:?}");

    eprintln!();
    eprintln!("✓ Verification passed.");
}
