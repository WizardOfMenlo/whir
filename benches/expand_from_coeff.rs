use divan::{black_box, AllocProfiler, Bencher};
use whir::{crypto::fields::Field64, ntt};

#[global_allocator]
static ALLOC: AllocProfiler = AllocProfiler::system();

// Test cases with polynomial sizes defined as exponents of 2 and expansion factors
const TEST_CASES: &[(u32, usize)] = &[
    (18, 2),
    (20, 2),
    (22, 2),
    (24, 2),
    (26, 2),
    (18, 4),
    (20, 4),
    (22, 4),
    (24, 4),
    (26, 4),
];

#[divan::bench(args = TEST_CASES)]
fn expand_from_coeff(bencher: Bencher, case: &(u32, usize)) {
    bencher
        .with_inputs(|| {
            let (exp, expansion) = *case;

            let size = 1 << exp;
            let coeffs: Vec<_> = (0..size).map(Field64::from).collect();
            (coeffs, expansion)
        })
        .bench_values(|(coeffs, expansion)| black_box(ntt::expand_from_coeff(&coeffs, expansion)));
}

fn main() {
    divan::main();
}
