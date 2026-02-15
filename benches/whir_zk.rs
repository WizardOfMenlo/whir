//! Benchmark: ZK v1 vs ZK v2 WHIR proving (2 polynomials).
//!
//! Run with:
//!   cargo bench --bench whir_zk
//!
//! Or filter to a specific group:
//!   cargo bench --bench whir_zk -- zk_v1
//!   cargo bench --bench whir_zk -- zk_v2

use ark_std::rand::{rngs::StdRng, SeedableRng};
use divan::{black_box, AllocProfiler, Bencher};
use whir::{
    algebra::{
        fields::{Field64, Field64_2},
        polynomials::{CoefficientList, MultilinearPoint},
        Weights,
    },
    hash,
    parameters::{FoldingFactor, MultivariateParameters, ProtocolParameters, SoundnessType},
    protocols::{whir::Config, whir_zk},
    transcript::{codecs::Empty, DomainSeparator, ProverState, VerifierState},
};

#[global_allocator]
static ALLOC: AllocProfiler = AllocProfiler::system();

type F = Field64;
type EF = Field64_2;

/// Polynomial sizes to benchmark (log₂ of number of coefficients).
const SIZES: &[usize] = &[20];

/// Number of polynomials for batched benchmarks.
const NUM_POLYS: usize = 2;

// ────────────────────────────────────────────────────────────────────────────
//  Shared setup helpers
// ────────────────────────────────────────────────────────────────────────────

/// Build `num_polynomials` deterministic polynomials with distinct coefficients.
fn make_polynomials(num_variables: usize, num_polynomials: usize) -> Vec<CoefficientList<F>> {
    let num_coeffs = 1usize << num_variables;
    (0..num_polynomials)
        .map(|poly_idx| {
            CoefficientList::new(
                (0..num_coeffs)
                    .map(|coeff_idx| F::from((poly_idx * num_coeffs + coeff_idx + 1) as u64))
                    .collect(),
            )
        })
        .collect()
}

/// Build weights + evaluations for multiple polynomials.
/// Layout: row-major [w₀_p₀, w₀_p₁, …] (one eval per polynomial per weight).
fn make_weights_and_evaluations_multi(
    polynomials: &[CoefficientList<F>],
    config: &Config<EF>,
    num_variables: usize,
) -> (Vec<Weights<EF>>, Vec<EF>) {
    let mut rng = StdRng::seed_from_u64(0xBEEF);
    let point = MultilinearPoint::rand(&mut rng, num_variables);
    let mut evaluations = Vec::with_capacity(polynomials.len());
    for poly in polynomials {
        evaluations.push(poly.mixed_evaluate(config.embedding(), &point));
    }
    let weights = vec![Weights::evaluation(point)];
    (weights, evaluations)
}

// ────────────────────────────────────────────────────────────────────────────
//  ZK v1 helpers
// ────────────────────────────────────────────────────────────────────────────

/// ZK v1: WHIR config for committing, μ+1 variables.
/// `batch_size` = number of polynomials × 2 (each poly contributes f̂ and g).
fn zk_v1_commit_config(num_variables: usize, num_polynomials: usize) -> Config<EF> {
    let extended = num_variables + 1;
    let mv = MultivariateParameters::new(extended);
    let params = ProtocolParameters {
        initial_statement: true,
        security_level: 32,
        pow_bits: 0,
        folding_factor: FoldingFactor::ConstantFromSecondRound(4, 4),
        soundness_type: SoundnessType::ConjectureList,
        starting_log_inv_rate: 1,
        batch_size: 2 * num_polynomials,
        hash_id: hash::SHA2,
    };
    Config::new(mv, &params)
}

/// ZK v1: WHIR config for proving P₁..Pₙ, μ+1 variables.
/// `batch_size` = number of P polynomials to prove.
fn zk_v1_prove_config(num_variables: usize, num_polynomials: usize) -> Config<EF> {
    let extended = num_variables + 1;
    let mv = MultivariateParameters::new(extended);
    let params = ProtocolParameters {
        initial_statement: true,
        security_level: 32,
        pow_bits: 0,
        folding_factor: FoldingFactor::ConstantFromSecondRound(4, 4),
        soundness_type: SoundnessType::ConjectureList,
        starting_log_inv_rate: 1,
        batch_size: num_polynomials,
        hash_id: hash::SHA2,
    };
    Config::new(mv, &params)
}

/// ZK v1 polynomial bundle: f̂(x,y) = f(x) + y·msk(x), random g(x,y), P = ρ·f̂ + g.
struct ZkV1Polys {
    f_hat: CoefficientList<F>,
    g_poly: CoefficientList<F>,
    p_poly: CoefficientList<F>,
}

/// Build N v1 polynomial bundles: for each polynomial, f̂, g, P = masking·f̂ + g.
fn make_zk_v1_polys(num_variables: usize, num_polynomials: usize) -> Vec<ZkV1Polys> {
    use ark_std::UniformRand;

    let mut rng = StdRng::seed_from_u64(0xCAFE);
    let num_coeffs = 1usize << num_variables;
    let extended_num_coeffs = 1usize << (num_variables + 1);
    let masking_challenge = F::rand(&mut rng);

    (0..num_polynomials)
        .map(|poly_idx| {
            // Deterministic base polynomial (distinct per polynomial).
            let base_coeffs: Vec<F> = (0..num_coeffs)
                .map(|coeff_idx| F::from((poly_idx * num_coeffs + coeff_idx + 1) as u64))
                .collect();

            // f̂(x,y) = base(x) + y·msk(x)
            let mut f_hat_coeffs = vec![F::from(0u64); extended_num_coeffs];
            for (coeff_idx, &coeff) in base_coeffs.iter().enumerate() {
                f_hat_coeffs[coeff_idx] = coeff;
            }
            for coeff_idx in 0..num_coeffs {
                f_hat_coeffs[num_coeffs + coeff_idx] = F::rand(&mut rng);
            }
            let f_hat = CoefficientList::new(f_hat_coeffs);

            // Random g(x,y)
            let g_coeffs: Vec<F> = (0..extended_num_coeffs)
                .map(|_| F::rand(&mut rng))
                .collect();
            let g_poly = CoefficientList::new(g_coeffs);

            // P = masking·f̂ + g
            let p_coeffs: Vec<F> = f_hat
                .coeffs()
                .iter()
                .zip(g_poly.coeffs().iter())
                .map(|(&f_hat_coeff, &g_coeff)| masking_challenge * f_hat_coeff + g_coeff)
                .collect();
            let p_poly = CoefficientList::new(p_coeffs);

            ZkV1Polys {
                f_hat,
                g_poly,
                p_poly,
            }
        })
        .collect()
}

/// Build weights + evaluations for multiple (μ+1)-variable P polynomials at (ā, 0).
/// Evaluations layout: row-major [w₀_P₀, w₀_P₁, …].
fn make_zk_v1_weights_and_evaluations(
    p_polys: &[CoefficientList<F>],
    config: &Config<EF>,
    num_variables: usize,
) -> (Vec<Weights<EF>>, Vec<EF>) {
    let mut rng = StdRng::seed_from_u64(0xBEEF);
    let base_point = MultilinearPoint::rand(&mut rng, num_variables);
    let mut coords = base_point.0;
    coords.push(EF::from(0u64)); // y = 0
    let extended_point = MultilinearPoint(coords);
    let mut evaluations = Vec::with_capacity(p_polys.len());
    for p_poly in p_polys {
        evaluations.push(p_poly.mixed_evaluate(config.embedding(), &extended_point));
    }
    (vec![Weights::evaluation(extended_point)], evaluations)
}

// ────────────────────────────────────────────────────────────────────────────
//  ZK v2 helpers
// ────────────────────────────────────────────────────────────────────────────

/// ZK v2 main WHIR parameters (round-0 fold = 2 for small k).
const fn zk_main_params(num_variables: usize) -> (MultivariateParameters<EF>, ProtocolParameters) {
    let mv = MultivariateParameters::new(num_variables);
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
    (mv, params)
}

/// Build a complete ZK v2 config for the given variable count and polynomial count.
fn make_zk_v2_config(num_variables: usize, num_polynomials: usize) -> whir_zk::Config<EF> {
    let (main_mv, main_params) = zk_main_params(num_variables);
    whir_zk::Config::new(
        main_mv,
        &main_params,
        FoldingFactor::Constant(1),
        num_polynomials,
    )
}

// ────────────────────────────────────────────────────────────────────────────
//  ZK v1 benchmarks – 2 polynomials (batched)
// ────────────────────────────────────────────────────────────────────────────

/// Commit [f̂₁, g₁, f̂₂, g₂] with batch_size=4.
#[divan::bench(args = SIZES)]
fn zk_v1_commit(bencher: Bencher, num_variables: usize) {
    let bundles = make_zk_v1_polys(num_variables, NUM_POLYS);
    let commit_config = zk_v1_commit_config(num_variables, NUM_POLYS);
    let ds = DomainSeparator::protocol(&commit_config)
        .session(&format!("bench-zk-v1-commit-{num_variables}"))
        .instance(&Empty);

    // Flatten: [f̂₁, g₁, f̂₂, g₂]
    let commit_polys: Vec<&CoefficientList<F>> = bundles
        .iter()
        .flat_map(|bundle| [&bundle.f_hat, &bundle.g_poly])
        .collect();

    bencher
        .with_inputs(|| ProverState::new_std(&ds))
        .bench_values(|mut prover_state| {
            let _ = black_box(commit_config.commit(&mut prover_state, &commit_polys));
        });
}

/// Prove [P₁, P₂] with batch_size=2, μ+1 variables.
#[divan::bench(args = SIZES)]
fn zk_v1_prove(bencher: Bencher, num_variables: usize) {
    let bundles = make_zk_v1_polys(num_variables, NUM_POLYS);
    let prove_config = zk_v1_prove_config(num_variables, NUM_POLYS);
    let p_polys: Vec<CoefficientList<F>> =
        bundles.iter().map(|bundle| bundle.p_poly.clone()).collect();
    let (weights, evaluations) =
        make_zk_v1_weights_and_evaluations(&p_polys, &prove_config, num_variables);
    let weight_refs: Vec<&Weights<EF>> = weights.iter().collect();

    let ds = DomainSeparator::protocol(&prove_config)
        .session(&format!("bench-zk-v1-prove-{num_variables}"))
        .instance(&Empty);

    let p_refs: Vec<&CoefficientList<F>> = p_polys.iter().collect();

    bencher
        .with_inputs(|| {
            let mut prover_state = ProverState::new_std(&ds);
            let witness = prove_config.commit(&mut prover_state, &p_refs);
            (prover_state, witness)
        })
        .bench_values(|(mut prover_state, witness)| {
            black_box(prove_config.prove(
                &mut prover_state,
                &p_refs,
                &[&witness],
                &weight_refs,
                &evaluations,
            ));
        });
}

/// Verify [P₁, P₂] via standard WHIR.
#[divan::bench(args = SIZES)]
fn zk_v1_verify(bencher: Bencher, num_variables: usize) {
    let bundles = make_zk_v1_polys(num_variables, NUM_POLYS);
    let prove_config = zk_v1_prove_config(num_variables, NUM_POLYS);
    let p_polys: Vec<CoefficientList<F>> =
        bundles.iter().map(|bundle| bundle.p_poly.clone()).collect();
    let (weights, evaluations) =
        make_zk_v1_weights_and_evaluations(&p_polys, &prove_config, num_variables);
    let weight_refs: Vec<&Weights<EF>> = weights.iter().collect();

    let ds = DomainSeparator::protocol(&prove_config)
        .session(&format!("bench-zk-v1-verify-{num_variables}"))
        .instance(&Empty);

    let p_refs: Vec<&CoefficientList<F>> = p_polys.iter().collect();

    // Generate a proof once.
    let proof = {
        let mut prover_state = ProverState::new_std(&ds);
        let witness = prove_config.commit(&mut prover_state, &p_refs);
        prove_config.prove(
            &mut prover_state,
            &p_refs,
            &[&witness],
            &weight_refs,
            &evaluations,
        );
        prover_state.proof()
    };

    bencher
        .with_inputs(|| {
            let mut verifier_state = VerifierState::new_std(&ds, &proof);
            let commitment = prove_config
                .receive_commitment(&mut verifier_state)
                .unwrap();
            (verifier_state, commitment)
        })
        .bench_values(|(mut verifier_state, commitment)| {
            black_box(
                prove_config
                    .verify(
                        &mut verifier_state,
                        &[&commitment],
                        &weight_refs,
                        &evaluations,
                    )
                    .unwrap(),
            );
        });
}

// ────────────────────────────────────────────────────────────────────────────
//  ZK v2 benchmarks – 2 polynomials (batched)
// ────────────────────────────────────────────────────────────────────────────

#[divan::bench(args = SIZES)]
fn zk_v2_commit(bencher: Bencher, num_variables: usize) {
    let polynomials = make_polynomials(num_variables, NUM_POLYS);
    let zk_config = make_zk_v2_config(num_variables, NUM_POLYS);

    let ds = DomainSeparator::protocol(&zk_config)
        .session(&format!("bench-zk-v2-commit-{num_variables}"))
        .instance(&Empty);

    bencher
        .with_inputs(|| ProverState::new_std(&ds))
        .bench_values(|mut prover_state| {
            let poly_refs: Vec<&CoefficientList<F>> = polynomials.iter().collect();
            black_box(zk_config.commit(&mut prover_state, &poly_refs));
        });
}

#[divan::bench(args = SIZES)]
fn zk_v2_prove(bencher: Bencher, num_variables: usize) {
    let polynomials = make_polynomials(num_variables, NUM_POLYS);
    let zk_config = make_zk_v2_config(num_variables, NUM_POLYS);

    let (weights, evaluations) = make_weights_and_evaluations_multi(
        &polynomials,
        &zk_config.blinded_commitment,
        num_variables,
    );
    let weight_refs: Vec<&Weights<EF>> = weights.iter().collect();

    let ds = DomainSeparator::protocol(&zk_config)
        .session(&format!("bench-zk-v2-prove-{num_variables}"))
        .instance(&Empty);

    bencher
        .with_inputs(|| {
            let mut prover_state = ProverState::new_std(&ds);
            let poly_refs: Vec<&CoefficientList<F>> = polynomials.iter().collect();
            let zk_witness = zk_config.commit(&mut prover_state, &poly_refs);
            (prover_state, zk_witness)
        })
        .bench_values(|(mut prover_state, zk_witness)| {
            let poly_refs: Vec<&CoefficientList<F>> = polynomials.iter().collect();
            black_box(zk_config.prove(
                &mut prover_state,
                &poly_refs,
                &zk_witness,
                &weight_refs,
                &evaluations,
            ));
        });
}

#[divan::bench(args = SIZES)]
fn zk_v2_verify(bencher: Bencher, num_variables: usize) {
    let polynomials = make_polynomials(num_variables, NUM_POLYS);
    let zk_config = make_zk_v2_config(num_variables, NUM_POLYS);

    let (weights, evaluations) = make_weights_and_evaluations_multi(
        &polynomials,
        &zk_config.blinded_commitment,
        num_variables,
    );
    let weight_refs: Vec<&Weights<EF>> = weights.iter().collect();

    let ds = DomainSeparator::protocol(&zk_config)
        .session(&format!("bench-zk-v2-verify-{num_variables}"))
        .instance(&Empty);

    // Generate a proof once (outside the benchmark loop).
    let proof = {
        let mut prover_state = ProverState::new_std(&ds);
        let poly_refs: Vec<&CoefficientList<F>> = polynomials.iter().collect();
        let zk_witness = zk_config.commit(&mut prover_state, &poly_refs);
        zk_config.prove(
            &mut prover_state,
            &poly_refs,
            &zk_witness,
            &weight_refs,
            &evaluations,
        );
        prover_state.proof()
    };

    bencher
        .with_inputs(|| {
            let mut verifier_state = VerifierState::new_std(&ds, &proof);
            let commitment = zk_config
                .receive_commitments(&mut verifier_state, NUM_POLYS)
                .unwrap();
            (verifier_state, commitment)
        })
        .bench_values(|(mut verifier_state, commitment)| {
            black_box(
                zk_config
                    .verify(&mut verifier_state, &commitment, &weight_refs, &evaluations)
                    .unwrap(),
            );
        });
}

fn main() {
    divan::main();
}
