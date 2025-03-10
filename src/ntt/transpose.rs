use super::super::utils::is_power_of_two;
use super::{utils::workload_size, MatrixMut};
use std::mem::swap;

#[cfg(feature = "parallel")]
use rayon::join;

// NOTE: The assumption that rows and cols are a power of two are actually only relevant for the square matrix case.
// (This is because the algorithm recurses into 4 sub-matrices of half dimension; we assume those to be square matrices as well, which only works for powers of two).

/// Transpose a matrix in-place.
/// Will batch transpose multiple matrices if the length of the slice is a multiple of rows * cols.
/// This algorithm assumes that both rows and cols are powers of two.
pub fn transpose<F: Sized + Copy + Send>(matrix: &mut [F], rows: usize, cols: usize) {
    debug_assert_eq!(matrix.len() % (rows * cols), 0);
    debug_assert!(is_power_of_two(rows));
    debug_assert!(is_power_of_two(cols));
    // eprintln!(
    //     "Transpose {} x {rows} x {cols} matrix.",
    //     matrix.len() / (rows * cols)
    // );
    if rows == cols {
        for matrix in matrix.chunks_exact_mut(rows * cols) {
            let matrix = MatrixMut::from_mut_slice(matrix, rows, cols);
            transpose_square(matrix);
        }
    } else {
        // TODO: Special case for rows = 2 * cols and cols = 2 * rows.
        // TODO: Special case for very wide matrices (e.g. n x 16).
        let mut scratch = vec![matrix[0]; rows * cols];
        for matrix in matrix.chunks_exact_mut(rows * cols) {
            scratch.copy_from_slice(matrix);
            let src = MatrixMut::from_mut_slice(scratch.as_mut_slice(), rows, cols);
            let dst = MatrixMut::from_mut_slice(matrix, cols, rows);
            transpose_copy(src, dst);
        }
    }
}

// The following function have both a parallel and a non-parallel implementation.
// We fuly split those in a parallel and a non-parallel functions (rather than using #[cfg] within a single function)
// and have main entry point fun that just calls the appropriate version (either fun_parallel or fun_not_parallel).
// The sole reason is that this simplifies unit tests: We otherwise would need to build twice to cover both cases.
// For effiency, we assume the compiler inlines away the extra "indirection" that we add to the entry point function.

// NOTE: We could lift the Send constraints on non-parallel build.

fn transpose_copy<F: Sized + Copy + Send>(src: MatrixMut<F>, dst: MatrixMut<F>) {
    #[cfg(not(feature = "parallel"))]
    transpose_copy_not_parallel(src, dst);
    #[cfg(feature = "parallel")]
    transpose_copy_parallel(src, dst);
}

/// Sets `dst` to the transpose of `src`. This will panic if the sizes of `src` and `dst` are not compatible.
#[cfg(feature = "parallel")]
fn transpose_copy_parallel<F: Sized + Copy + Send>(
    src: MatrixMut<'_, F>,
    mut dst: MatrixMut<'_, F>,
) {
    assert_eq!(src.rows(), dst.cols());
    assert_eq!(src.cols(), dst.rows());
    if src.rows() * src.cols() > workload_size::<F>() {
        // Split along longest axis and recurse.
        // This results in a cache-oblivious algorithm.
        let ((a, b), (x, y)) = if src.rows() > src.cols() {
            let n = src.rows() / 2;
            (src.split_vertical(n), dst.split_horizontal(n))
        } else {
            let n = src.cols() / 2;
            (src.split_horizontal(n), dst.split_vertical(n))
        };
        join(
            || transpose_copy_parallel(a, x),
            || transpose_copy_parallel(b, y),
        );
    } else {
        for i in 0..src.rows() {
            for j in 0..src.cols() {
                dst[(j, i)] = src[(i, j)];
            }
        }
    }
}

/// Sets `dst` to the transpose of `src`. This will panic if the sizes of `src` and `dst` are not compatible.
/// This is the non-parallel version
#[cfg(not(feature = "parallel"))]
fn transpose_copy_not_parallel<F: Sized + Copy>(src: MatrixMut<'_, F>, mut dst: MatrixMut<'_, F>) {
    assert_eq!(src.rows(), dst.cols());
    assert_eq!(src.cols(), dst.rows());
    if src.rows() * src.cols() > workload_size::<F>() {
        // Split along longest axis and recurse.
        // This results in a cache-oblivious algorithm.
        let ((a, b), (x, y)) = if src.rows() > src.cols() {
            let n = src.rows() / 2;
            (src.split_vertical(n), dst.split_horizontal(n))
        } else {
            let n = src.cols() / 2;
            (src.split_horizontal(n), dst.split_vertical(n))
        };
        transpose_copy_not_parallel(a, x);
        transpose_copy_not_parallel(b, y);
    } else {
        for i in 0..src.rows() {
            for j in 0..src.cols() {
                dst[(j, i)] = src[(i, j)];
            }
        }
    }
}

/// Transpose a square matrix in-place. Asserts that the size of the matrix is a power of two.
fn transpose_square<F: Sized + Send>(m: MatrixMut<F>) {
    #[cfg(feature = "parallel")]
    transpose_square_parallel(m);
    #[cfg(not(feature = "parallel"))]
    transpose_square_non_parallel(m);
}

/// Transpose a square matrix in-place. Asserts that the size of the matrix is a power of two.
/// This is the parallel version.
#[cfg(feature = "parallel")]
fn transpose_square_parallel<F: Sized + Send>(mut m: MatrixMut<F>) {
    debug_assert!(m.is_square());
    debug_assert!(m.rows().is_power_of_two());
    let size = m.rows();
    if size * size > workload_size::<F>() {
        // Recurse into quadrants.
        // This results in a cache-oblivious algorithm.
        let n = size / 2;
        let (a, b, c, d) = m.split_quadrants(n, n);

        join(
            || transpose_square_swap_parallel(b, c),
            || {
                join(
                    || transpose_square_parallel(a),
                    || transpose_square_parallel(d),
                )
            },
        );
    } else {
        for i in 0..size {
            for j in (i + 1)..size {
                // unsafe needed due to lack of bounds-check by swap. We are guaranteed that (i,j) and (j,i) are within the bounds.
                unsafe {
                    m.swap((i, j), (j, i));
                }
            }
        }
    }
}

/// Transpose a square matrix in-place. Asserts that the size of the matrix is a power of two.
/// This is the non-parallel version.
#[cfg(not(feature = "parallel"))]
fn transpose_square_non_parallel<F: Sized>(mut m: MatrixMut<F>) {
    debug_assert!(m.is_square());
    debug_assert!(m.rows().is_power_of_two());
    let size = m.rows();
    if size * size > workload_size::<F>() {
        // Recurse into quadrants.
        // This results in a cache-oblivious algorithm.
        let n = size / 2;
        let (a, b, c, d) = m.split_quadrants(n, n);
        transpose_square_non_parallel(a);
        transpose_square_non_parallel(d);
        transpose_square_swap_non_parallel(b, c);
    } else {
        for i in 0..size {
            for j in (i + 1)..size {
                // unsafe needed due to lack of bounds-check by swap. We are guaranteed that (i,j) and (j,i) are within the bounds.
                unsafe {
                    m.swap((i, j), (j, i));
                }
            }
        }
    }
}

/// Transpose and swap two square size matrices. Sizes must be equal and a power of two.
fn transpose_square_swap<F: Sized + Send>(a: MatrixMut<F>, b: MatrixMut<F>) {
    #[cfg(feature = "parallel")]
    transpose_square_swap_parallel(a, b);
    #[cfg(not(feature = "parallel"))]
    transpose_square_swap_non_parallel(a, b);
}

/// Transpose and swap two square size matrices (parallel version).
///
/// The size must be a power of two.
#[cfg(feature = "parallel")]
pub fn transpose_square_swap_parallel<F: Sized + Send>(
    mut a: MatrixMut<'_, F>,
    mut b: MatrixMut<'_, F>,
) {
    debug_assert!(a.is_square());
    debug_assert_eq!(a.rows(), b.cols());
    debug_assert_eq!(a.cols(), b.rows());
    debug_assert!(is_power_of_two(a.rows()));
    debug_assert!(workload_size::<F>() >= 2);

    let size = a.rows();

    // Direct swaps for small matrices (≤8x8)
    // - Avoids recursion overhead
    // - Uses basic element-wise swaps
    if size <= 8 {
        for i in 0..size {
            for j in 0..size {
                swap(&mut a[(i, j)], &mut b[(j, i)]);
            }
        }
        return;
    }

    // If the matrix is large, use recursive subdivision:
    // - Improves cache efficiency by working on smaller blocks
    // - Enables parallel execution
    if 2 * size * size > workload_size::<F>() {
        let n = size / 2;
        let (aa, ab, ac, ad) = a.split_quadrants(n, n);
        let (ba, bb, bc, bd) = b.split_quadrants(n, n);

        join(
            || {
                join(
                    || transpose_square_swap_parallel(aa, ba),
                    || transpose_square_swap_parallel(ab, bc),
                )
            },
            || {
                join(
                    || transpose_square_swap_parallel(ac, bb),
                    || transpose_square_swap_parallel(ad, bd),
                )
            },
        );
    } else {
        // Optimized 2×2 loop unrolling for larger blocks
        // - Reduces loop overhead
        // - Increases memory access efficiency
        for i in (0..size).step_by(2) {
            for j in (0..size).step_by(2) {
                swap(&mut a[(i, j)], &mut b[(j, i)]);
                swap(&mut a[(i + 1, j)], &mut b[(j, i + 1)]);
                swap(&mut a[(i, j + 1)], &mut b[(j + 1, i)]);
                swap(&mut a[(i + 1, j + 1)], &mut b[(j + 1, i + 1)]);
            }
        }
    }
}

/// Transpose and swap two square size matrices, whose sizes are a power of two (non-parallel version)
#[cfg(not(feature = "parallel"))]
fn transpose_square_swap_non_parallel<F: Sized>(mut a: MatrixMut<F>, mut b: MatrixMut<F>) {
    debug_assert!(a.is_square());
    debug_assert_eq!(a.rows(), b.cols());
    debug_assert_eq!(a.cols(), b.rows());
    debug_assert!(is_power_of_two(a.rows()));
    debug_assert!(workload_size::<F>() >= 2); // otherwise, we would recurse even if size == 1.

    let size = a.rows();
    if 2 * size * size > workload_size::<F>() {
        // Recurse into quadrants.
        // This results in a cache-oblivious algorithm.
        let n = size / 2;
        let (aa, ab, ac, ad) = a.split_quadrants(n, n);
        let (ba, bb, bc, bd) = b.split_quadrants(n, n);
        transpose_square_swap_non_parallel(aa, ba);
        transpose_square_swap_non_parallel(ab, bc);
        transpose_square_swap_non_parallel(ac, bb);
        transpose_square_swap_non_parallel(ad, bd);
    } else {
        for i in 0..size {
            for j in 0..size {
                swap(&mut a[(i, j)], &mut b[(j, i)]);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::utils::workload_size;
    use super::*;

    type Pair = (usize, usize);
    type Triple = (usize, usize, usize);

    /// Creates a `rows x columns` matrix stored as a flat vector.
    /// Each element `(i, j)` represents its row and column position.
    fn make_example_matrix(rows: usize, columns: usize) -> Vec<Pair> {
        (0..rows)
            .flat_map(|i| (0..columns).map(move |j| (i, j)))
            .collect()
    }

    /// Creates a sequence of `instances` matrices, each of size `rows x columns`.
    ///
    /// Each element in the `index`-th matrix is `(index, row, col)`, stored in a flat vector.
    fn make_example_matrices(rows: usize, columns: usize, instances: usize) -> Vec<Triple> {
        let mut matrices = Vec::with_capacity(rows * columns * instances);

        for index in 0..instances {
            for row in 0..rows {
                for col in 0..columns {
                    matrices.push((index, row, col));
                }
            }
        }

        matrices
    }

    #[test]
    fn test_transpose_copy() {
        // iterate over both parallel and non-parallel implementation.
        // Needs HRTB, otherwise it won't work.
        let mut funs: Vec<&dyn for<'a, 'b> Fn(MatrixMut<'a, Pair>, MatrixMut<'b, Pair>)> = vec![
            #[cfg(not(feature = "parallel"))]
            &transpose_copy_not_parallel::<Pair>,
            &transpose_copy::<Pair>,
        ];
        #[cfg(feature = "parallel")]
        funs.push(&transpose_copy_parallel::<Pair>);

        for f in funs {
            let rows: usize = workload_size::<Pair>() + 1; // intentionally not a power of two: The function is not described as only working for powers of two.
            let columns: usize = 4;
            let mut srcarray = make_example_matrix(rows, columns);
            let mut dstarray: Vec<(usize, usize)> = vec![(0, 0); rows * columns];

            let src1 = MatrixMut::<Pair>::from_mut_slice(&mut srcarray[..], rows, columns);
            let dst1 = MatrixMut::<Pair>::from_mut_slice(&mut dstarray[..], columns, rows);

            f(src1, dst1);
            let dst1 = MatrixMut::<Pair>::from_mut_slice(&mut dstarray[..], columns, rows);

            for i in 0..rows {
                for j in 0..columns {
                    assert_eq!(dst1[(j, i)], (i, j));
                }
            }
        }
    }

    #[test]
    fn test_transpose_square_swap() {
        // iterate over parallel and non-parallel variants:
        let mut funs: Vec<&dyn for<'a> Fn(MatrixMut<'a, Triple>, MatrixMut<'a, Triple>)> = vec![
            &transpose_square_swap::<Triple>,
            #[cfg(not(feature = "parallel"))]
            &transpose_square_swap_non_parallel::<Triple>,
        ];
        #[cfg(feature = "parallel")]
        funs.push(&transpose_square_swap_parallel::<Triple>);

        for f in funs {
            // Set rows manually. We want to be sure to trigger the actual recursion.
            // (Computing this from workload_size was too much hassle.)
            let rows = 1024; // workload_size::<Triple>();
            assert!(rows * rows > 2 * workload_size::<Triple>());

            let examples: Vec<Triple> = make_example_matrices(rows, rows, 2);
            // Make copies for simplicity, because we borrow different parts.
            let mut examples1 = Vec::from(&examples[0..rows * rows]);
            let mut examples2 = Vec::from(&examples[rows * rows..2 * rows * rows]);

            let view1 = MatrixMut::from_mut_slice(&mut examples1, rows, rows);
            let view2 = MatrixMut::from_mut_slice(&mut examples2, rows, rows);
            for i in 0..rows {
                for j in 0..rows {
                    assert_eq!(view1[(i, j)], (0, i, j));
                    assert_eq!(view2[(i, j)], (1, i, j));
                }
            }
            f(view1, view2);
            let view1 = MatrixMut::from_mut_slice(&mut examples1, rows, rows);
            let view2 = MatrixMut::from_mut_slice(&mut examples2, rows, rows);
            for i in 0..rows {
                for j in 0..rows {
                    assert_eq!(view1[(i, j)], (1, j, i));
                    assert_eq!(view2[(i, j)], (0, j, i));
                }
            }
        }
    }

    #[test]
    fn test_transpose_square() {
        let mut funs: Vec<&dyn for<'a> Fn(MatrixMut<'a, _>)> = vec![
            &transpose_square::<Pair>,
            &transpose_square_parallel::<Pair>,
        ];
        #[cfg(feature = "parallel")]
        funs.push(&transpose_square::<Pair>);
        for f in funs {
            // Set rows manually. We want to be sure to trigger the actual recursion.
            // (Computing this from workload_size was too much hassle.)
            let size = 1024;
            assert!(size * size > 2 * workload_size::<Pair>());

            let mut example = make_example_matrix(size, size);
            let view = MatrixMut::from_mut_slice(&mut example, size, size);
            f(view);
            let view = MatrixMut::from_mut_slice(&mut example, size, size);
            for i in 0..size {
                for j in 0..size {
                    assert_eq!(view[(i, j)], (j, i));
                }
            }
        }
    }

    #[test]
    fn test_transpose() {
        let size = 1024;

        // rectangular matrix:
        let rows = size;
        let cols = 16;
        let mut example = make_example_matrix(rows, cols);
        transpose(&mut example, rows, cols);
        let view = MatrixMut::from_mut_slice(&mut example, cols, rows);
        for i in 0..cols {
            for j in 0..rows {
                assert_eq!(view[(i, j)], (j, i));
            }
        }

        // square matrix:
        let rows = size;
        let cols = size;
        let mut example = make_example_matrix(rows, cols);
        transpose(&mut example, rows, cols);
        let view = MatrixMut::from_mut_slice(&mut example, cols, rows);
        for i in 0..cols {
            for j in 0..rows {
                assert_eq!(view[(i, j)], (j, i));
            }
        }

        // 20 rectangular matrices:
        let number_of_matrices = 20;
        let rows = size;
        let cols = 16;
        let mut example = make_example_matrices(rows, cols, number_of_matrices);
        transpose(&mut example, rows, cols);
        for index in 0..number_of_matrices {
            let view = MatrixMut::from_mut_slice(
                &mut example[index * rows * cols..(index + 1) * rows * cols],
                cols,
                rows,
            );
            for i in 0..cols {
                for j in 0..rows {
                    assert_eq!(view[(i, j)], (index, j, i));
                }
            }
        }

        // 20 square matrices:
        let number_of_matrices = 20;
        let rows = size;
        let cols = size;
        let mut example = make_example_matrices(rows, cols, number_of_matrices);
        transpose(&mut example, rows, cols);
        for index in 0..number_of_matrices {
            let view = MatrixMut::from_mut_slice(
                &mut example[index * rows * cols..(index + 1) * rows * cols],
                cols,
                rows,
            );
            for i in 0..cols {
                for j in 0..rows {
                    assert_eq!(view[(i, j)], (index, j, i));
                }
            }
        }
    }
}
