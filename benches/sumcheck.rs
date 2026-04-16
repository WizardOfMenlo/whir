use ark_std::rand::{rngs::StdRng, SeedableRng};
use divan::{black_box, AllocProfiler, Bencher};
use efficient_sumcheck::{
    inner_product_sumcheck_partial_with_hook,
    transcript::{SanityTranscript, Transcript},
};
use whir::algebra::fields::{Field64 as G1, Field64_2 as G2, Field64_3 as G3};

#[global_allocator]
static ALLOC: AllocProfiler = AllocProfiler::system();

const SIZES: &[u64] = &[1 << 20, 1 << 24];
const SEED: u64 = 0xA110C8ED;

// ── effsc MSB fused path: inner_product_sumcheck_partial_with_hook runs all
//    rounds internally — round 0 = compute_sumcheck_polynomial, rounds ≥1 =
//    fused_fold_and_compute_polynomial (the 8R+4W-per-quadruple kernel).

#[divan::bench(args = SIZES)]
fn effsc_full_g1(bencher: Bencher, size: u64) {
    run_effsc_full::<G1>(bencher, size);
}
#[divan::bench(args = SIZES)]
fn effsc_full_g2(bencher: Bencher, size: u64) {
    run_effsc_full::<G2>(bencher, size);
}
#[divan::bench(args = SIZES)]
fn effsc_full_g3(bencher: Bencher, size: u64) {
    run_effsc_full::<G3>(bencher, size);
}

fn run_effsc_full<F: ark_ff::Field + From<u64>>(bencher: Bencher, size: u64) {
    let num_rounds = (size as u64).trailing_zeros() as usize;
    bencher
        .with_inputs(|| {
            let a: Vec<F> = (0..size).map(F::from).collect();
            let b: Vec<F> = (0..size).map(F::from).collect();
            (a, b)
        })
        .bench_values(|(mut a, mut b)| {
            let mut rng = StdRng::seed_from_u64(SEED);
            let mut t = SanityTranscript::<StdRng>::new(&mut rng);
            let result = inner_product_sumcheck_partial_with_hook(
                &mut a,
                &mut b,
                &mut t,
                num_rounds,
                |_, _| {},
            );
            black_box((result, a, b))
        });
}

fn main() {
    divan::main();
}
