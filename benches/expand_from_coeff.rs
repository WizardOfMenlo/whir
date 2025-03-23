use criterion::{black_box, criterion_group, criterion_main, Criterion};
use whir::{crypto::fields::Field64, ntt::expand_from_coeff};

fn bench_expand_from_coeff(c: &mut Criterion) {
    // Test cases with polynomial sizes defined as exponents of 2 and expansion factors
    let test_cases =
        [(1, 2), (2, 4), (3, 2), (4, 2), (5, 4), (6, 8), (7, 4), (10, 4), (11, 2), (20, 2)];

    for &(exp, expansion) in &test_cases {
        // Compute 2^exp
        let size = 1 << exp;
        let coeffs: Vec<_> = (0..size).map(Field64::from).collect();

        c.bench_function(&format!("expand_from_coeff size=2^{exp} exp={expansion}"), |b| {
            b.iter(|| expand_from_coeff(black_box(&coeffs), black_box(expansion)))
        });
    }
}

criterion_group!(benches, bench_expand_from_coeff);
criterion_main!(benches);
