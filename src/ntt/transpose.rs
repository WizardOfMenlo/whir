use super::super::utils::is_power_of_two;
use super::{utils::workload_size, MatrixMut};
use std::mem::swap;

#[cfg(feature = "parallel")]
use rayon::join;

/// Transpose a matrix in-place.
/// Will batch transpose multiple matrices if the length of the slice is a multiple of rows * cols.
pub fn transpose<F: Sized + Copy + Send>(matrix: &mut [F], rows: usize, cols: usize) {
    debug_assert_eq!(matrix.len() % (rows * cols), 0);
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

// Fuly split those parallel and a non-parallel functions (rather than using #[cfg] within a single function).
// The reason is that this simplifies testing: We otherwise would need to build twice to cover both cases and
// we could not do differential tests.
// This assumes the compiler inlines away the extra "indirection".
// NOTE: Could lift the Send contraint on non-parallel build.

fn transpose_copy<F: Sized + Copy + Send>(src: MatrixMut<F>, dst: MatrixMut<F>) {
    #[cfg(not(feature = "parallel"))]
    transpose_copy_not_parallel(src, dst);
    #[cfg(feature = "parallel")]
    transpose_copy_parallel(src, dst);
}

#[cfg(feature = "parallel")]
fn transpose_copy_parallel<'a, 'b, F: Sized + Copy + Send>(
    src: MatrixMut<'a, F>,
    mut dst: MatrixMut<'b, F>,
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

fn transpose_copy_not_parallel<'a, 'b, F: Sized + Copy>(
    src: MatrixMut<'a, F>,
    mut dst: MatrixMut<'b, F>,
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
            || transpose_square_parallel(a),
            || {
                join(
                    || transpose_square_parallel(d),
                    || transpose_square_swap_parallel(b, c),
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

/// Transpose a square matrix in-place.
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

/// Transpose and swap two square size matrices.
#[cfg(feature = "parallel")]
fn transpose_square_swap_parallel<F: Sized + Send>(mut a: MatrixMut<F>, mut b: MatrixMut<F>) {
    debug_assert!(a.is_square());
    debug_assert_eq!(a.rows(), b.cols());
    debug_assert_eq!(a.cols(), b.rows());
    let size = a.rows();
    if 2 * size * size > workload_size::<F>() {
        // Recurse into quadrants.
        // This results in a cache-oblivious algorithm.
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
        for i in 0..size {
            for j in 0..size {
                swap(&mut a[(i, j)], &mut b[(j, i)])
            }
        }
    }
}

/// Transpose and swap two square size matrices, whose sizes are a power of two.
fn transpose_square_swap_non_parallel<F: Sized>(mut a: MatrixMut<F>, mut b: MatrixMut<F>) {
    debug_assert!(a.is_square());

    debug_assert_eq!(a.rows(), b.cols());
    debug_assert_eq!(a.cols(), b.rows());

    debug_assert!(is_power_of_two(a.rows()));

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
                swap(&mut a[(i, j)], &mut b[(j, i)])
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::utils::workload_size;
    use super::*;

    type F = i32; // for simplicity
    type Pair = (usize, usize);
    type Triple = (usize, usize, usize);

    // create a vector (intended to be viewed as a matrix) whose (i,j)'th entry is the pair (i,j) itself.
    // This is useful to debug transposition algorithms.
    fn make_example_matrix(rows: usize, columns: usize) -> Vec<Pair> {
        let mut v: Vec<Pair> = vec![(0, 0); rows * columns];
        let mut view = MatrixMut::from_mut_slice(&mut v, rows, columns);
        for i in 0..rows {
            for j in 0..columns {
                view[(i, j)] = (i, j);
            }
        }
        v
    }

    // create a vector (intended to be viewed as a sequence of `instances` matrix) where (i,j)'th entry of the `index`th matrix
    // is the triple (index, i,j).
    fn make_example_matrices(rows: usize, columns: usize, instances: usize) -> Vec<Triple> {
        let mut v: Vec<Triple> = vec![(0, 0, 0); rows * columns * instances];
        for index in 0..instances {
            let mut view = MatrixMut::from_mut_slice(
                &mut v[rows * columns * index..rows * columns * (index + 1)],
                rows,
                columns,
            );
            for i in 0..rows {
                for j in 0..columns {
                    view[(i, j)] = (index, i, j);
                }
            }
        }
        v
    }

    // not needed, actually. We compare the backing arrays.
    impl<'a, 'b> PartialEq for MatrixMut<'a, F> {
        fn eq(&self, other: &Self) -> bool {
            let r = self.rows();
            let c = self.cols();

            if other.rows() != r || other.cols() != c {
                return false;
            }
            for i in 0..r {
                for j in 0..c {
                    if self[(i, j)] != other[(i, j)] {
                        return false;
                    }
                }
            }
            true
        }
    }

    #[test]
    fn test_transpose_copy() {
        // iterate over both parallel and non-parallel implementation.
        // Needs HRTB, otherwise it won't work.
        let mut funs: Vec<&dyn for<'a, 'b> Fn(MatrixMut<'a, Pair>, MatrixMut<'b, Pair>)> = vec![
            &transpose_copy_not_parallel::<Pair>,
            &transpose_copy::<Pair>,
        ];
        #[cfg(feature = "parallel")]
        funs.push(&transpose_copy_parallel::<Pair>);

        for f in funs {
            let rows: usize = workload_size::<Pair>();
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
    fn test_transpose_square() {
        // iterate over parallel and non-parallel variants:
        let mut funs: Vec<&dyn for<'a> Fn(MatrixMut<'a, Triple>, MatrixMut<'a, Triple>)> = vec![
            &transpose_square_swap::<Triple>,
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
}
