use divan::black_box;
use whir::{crypto::fields::Field64, ntt};

// Test cases with polynomial sizes defined as exponents of 2 and expansion factors
const TEST_CASES: &[(u32, usize)] = &[
    (1, 2),
    (2, 4),
    (3, 2),
    (4, 2),
    (5, 4),
    (6, 8),
    (7, 4),
    (10, 4),
    (11, 2),
    (20, 2),
];

#[divan::bench(args = TEST_CASES)]
fn expand_from_coeff(case: &(u32, usize)) {
    let (exp, expansion) = *case;
    
    // Compute 2^exp
    let size = 1 << exp;
    let coeffs: Vec<_> = (0..size).map(Field64::from).collect();

    ntt::expand_from_coeff(
        black_box(&coeffs),
        black_box(expansion)
    );
}

fn main() {
    divan::main();
}