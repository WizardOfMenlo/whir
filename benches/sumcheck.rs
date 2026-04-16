use divan::{black_box, AllocProfiler, Bencher};
use efficient_sumcheck::{order_strategy::MSBOrder, simd_ops as effsc_simd, streams::reorder_vec};
use whir::algebra::{
    fields::{Field64 as G1, Field64_2 as G2, Field64_3 as G3},
    sumcheck::{compute_sumcheck_polynomial, fold},
};

#[global_allocator]
static ALLOC: AllocProfiler = AllocProfiler::system();

const SIZES: &[u64] = &[1 << 16, 1 << 18, 1 << 20, 1 << 22];

// ── Whir baseline kernel: compute_sumcheck_polynomial + fold ───────────────

#[divan::bench(args = SIZES)]
fn whir_g1(bencher: Bencher, size: u64) {
    run_whir::<G1>(bencher, size);
}
#[divan::bench(args = SIZES)]
fn whir_g2(bencher: Bencher, size: u64) {
    run_whir::<G2>(bencher, size);
}
#[divan::bench(args = SIZES)]
fn whir_g3(bencher: Bencher, size: u64) {
    run_whir::<G3>(bencher, size);
}

fn run_whir<F: ark_ff::Field + From<u64>>(bencher: Bencher, size: u64) {
    bencher
        .with_inputs(|| {
            let a: Vec<F> = (0..size).map(F::from).collect();
            let b: Vec<F> = (0..size).map(F::from).collect();
            let r = F::from(42);
            (a, b, r)
        })
        .bench_values(|(mut a, mut b, r)| {
            let poly = compute_sumcheck_polynomial(&a, &b);
            fold(&mut a, r);
            fold(&mut b, r);
            black_box((poly, a, b))
        });
}

// ── effsc SIMD path: reorder + pairwise_product_sum + fold ─────────────────
// Includes the bit-reversal permutation cost on the first round.

#[divan::bench(args = SIZES)]
fn effsc_g1(bencher: Bencher, size: u64) {
    run_effsc::<G1>(bencher, size);
}
#[divan::bench(args = SIZES)]
fn effsc_g2(bencher: Bencher, size: u64) {
    run_effsc::<G2>(bencher, size);
}
#[divan::bench(args = SIZES)]
fn effsc_g3(bencher: Bencher, size: u64) {
    run_effsc::<G3>(bencher, size);
}

fn run_effsc<F: ark_ff::Field + From<u64>>(bencher: Bencher, size: u64) {
    bencher
        .with_inputs(|| {
            let a: Vec<F> = (0..size).map(F::from).collect();
            let b: Vec<F> = (0..size).map(F::from).collect();
            let r = F::from(42);
            (a, b, r)
        })
        .bench_values(|(a, b, r)| {
            // Bit-reverse to LSB-first so effsc's adjacent-pair kernels
            // bind whir's x_0, x_1, ... in order.
            let mut a = reorder_vec::<F, MSBOrder>(a);
            let mut b = reorder_vec::<F, MSBOrder>(b);
            let poly = effsc_simd::pairwise_product_sum(&a, &b);
            effsc_simd::fold(&mut a, r);
            effsc_simd::fold(&mut b, r);
            black_box((poly, a, b))
        });
}

// ── effsc SIMD path WITHOUT the entry permutation ──────────────────────────
// Isolates the kernel speedup from the permutation overhead, simulating
// mid-sumcheck rounds where the data is already in LSB-first layout.

#[divan::bench(args = SIZES)]
fn effsc_nopermute_g1(bencher: Bencher, size: u64) {
    run_effsc_nopermute::<G1>(bencher, size);
}
#[divan::bench(args = SIZES)]
fn effsc_nopermute_g2(bencher: Bencher, size: u64) {
    run_effsc_nopermute::<G2>(bencher, size);
}
#[divan::bench(args = SIZES)]
fn effsc_nopermute_g3(bencher: Bencher, size: u64) {
    run_effsc_nopermute::<G3>(bencher, size);
}

fn run_effsc_nopermute<F: ark_ff::Field + From<u64>>(bencher: Bencher, size: u64) {
    bencher
        .with_inputs(|| {
            let a: Vec<F> = (0..size).map(F::from).collect();
            let b: Vec<F> = (0..size).map(F::from).collect();
            let r = F::from(42);
            (a, b, r)
        })
        .bench_values(|(mut a, mut b, r)| {
            let poly = effsc_simd::pairwise_product_sum(&a, &b);
            effsc_simd::fold(&mut a, r);
            effsc_simd::fold(&mut b, r);
            black_box((poly, a, b))
        });
}

fn main() {
    divan::main();
}
