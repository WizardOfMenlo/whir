use criterion::{black_box, criterion_group, criterion_main, Criterion};
use whir::crypto::fields::Field64;
use whir::ntt::expand_from_coeff;

fn bench_expand_from_coeff(c: &mut Criterion) {
    // Test cases with increasing polynomial sizes and expansion factors
    let test_cases = [
        (2, 2),
        (4, 4),
        (8, 2),
        (16, 2),
        (32, 4),
        (64, 8),
        (128, 4),
        (1024, 4),
        (2048, 2),
    ];

    for &(size, expansion) in &test_cases {
        let coeffs: Vec<_> = (0..size).map(Field64::from).collect();

        c.bench_function(
            &format!("expand_from_coeff size={} exp={}", size, expansion),
            |b| b.iter(|| expand_from_coeff(black_box(&coeffs), black_box(expansion))),
        );
    }
}

criterion_group!(benches, bench_expand_from_coeff);
criterion_main!(benches);
