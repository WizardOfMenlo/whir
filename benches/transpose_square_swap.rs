use criterion::{black_box, criterion_group, criterion_main, Criterion};
use whir::ntt::matrix::MatrixMut;
use whir::ntt::transpose::transpose_square_swap_parallel;

/// Creates an `N x N` matrix with elements `(row, col)` for benchmarking
fn create_matrix(rows: usize) -> Vec<(usize, usize)> {
    (0..rows)
        .flat_map(|i| (0..rows).map(move |j| (i, j)))
        .collect()
}

/// Benchmark function for `transpose_square_swap_parallel`
fn benchmark_transpose_square_swap(c: &mut Criterion) {
    let size = 1024; // Adjust for different test sizes
    let mut matrix_a = create_matrix(size);
    let mut matrix_b = create_matrix(size);

    let view_a = MatrixMut::from_mut_slice(&mut matrix_a, size, size);
    let view_b = MatrixMut::from_mut_slice(&mut matrix_b, size, size);

    c.bench_function(
        &format!("transpose_square_swap_parallel {size}x{size}"),
        |b| {
            b.iter(|| {
                transpose_square_swap_parallel(
                    black_box(view_a.clone()),
                    black_box(view_b.clone()),
                );
            });
        },
    );
}

// Register benchmark group
criterion_group!(benches, benchmark_transpose_square_swap);
criterion_main!(benches);
