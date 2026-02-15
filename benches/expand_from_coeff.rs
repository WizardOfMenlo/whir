use divan::{black_box, AllocProfiler, Bencher};
use whir::algebra::{fields::Field64, ntt};

#[global_allocator]
static ALLOC: AllocProfiler = AllocProfiler::system();

//
// Test cases with tuple entries:
//      - polynomial sizes defined as exponents of 2,
//      - RS code expansion factors, and
//      - interleaved bloc size exponent of 2
//
const TEST_CASES: &[(u32, usize, usize)] = &[
    (16, 2, 2),
    (18, 2, 2),
    (20, 2, 3),
    (16, 4, 3),
    (18, 4, 3),
    (20, 4, 4),
    (22, 4, 4),
];

#[divan::bench(args = TEST_CASES)]
fn interleaved_rs_encode(bencher: Bencher, case: &(u32, usize, usize)) {
    bencher
        .with_inputs(|| {
            let (exp, expansion, coset_sz) = *case;

            let size = 1 << exp;
            let coeffs: Vec<_> = (0..size).map(Field64::from).collect();
            (coeffs, expansion, coset_sz)
        })
        .bench_values(|(coeffs, expansion, coset_sz)| {
            black_box(ntt::interleaved_rs_encode(
                &[&coeffs],
                expansion,
                1 << coset_sz,
            ))
        });
}

fn main() {
    divan::main();
}
