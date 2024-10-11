use super::{utils::workload_size, MatrixMut};
use std::mem::swap;

#[cfg(feature = "parallel")]
use rayon::join;

/// Transpose a matrix in-place.
/// Will batch transpose multiple matrices if the length of the slice is a multiple of rows * cols.
pub fn transpose<F: Sized + Copy + Send>(matrix: &mut [F], rows: usize, cols: usize) {
    debug_assert_eq!(matrix.len() % rows * cols, 0);
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
fn transpose_copy_parallel<'a, 'b, F: Sized + Copy + Send>(src: MatrixMut<'a, F>, mut dst: MatrixMut<'b, F>) {
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
        join(|| transpose_copy_parallel(a, x), || transpose_copy_parallel(b, y));
    } else {
        for i in 0..src.rows() {
            for j in 0..src.cols() {
                dst[(j, i)] = src[(i, j)];
            }
        }
    }
}

fn transpose_copy_not_parallel<'a, 'b, F: Sized + Copy>(src: MatrixMut<'a, F>, mut dst: MatrixMut<'b, F>) {
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

/// Transpose a square matrix in-place.
fn transpose_square<F: Sized + Send>(mut m: MatrixMut<F>) {
    debug_assert!(m.is_square());
    debug_assert!(m.rows().is_power_of_two());
    let size = m.rows();
    if size * size > workload_size::<F>() {
        // Recurse into quadrants.
        // This results in a cache-oblivious algorithm.
        let n = size / 2;
        let (a, b, c, d) = m.split_quadrants(n, n);

        #[cfg(not(feature = "parallel"))]
        {
            transpose_square(a);
            transpose_square(d);
            transpose_square_swap(b, c);
        }
        #[cfg(feature = "parallel")]
        join(
            || transpose_square(a),
            || join(|| transpose_square(d), || transpose_square_swap(b, c)),
        );
    } else {
        for i in 0..size {
            for j in (i + 1)..size {
                // unsafe needed due to lack of bounds-check by swap. We are guaranteed that (i,j) and (j,i) are within the bounds.
                unsafe {m.swap((i, j), (j, i));}
            }
        }
    }
}

/// Transpose and swap two square size matrices.
fn transpose_square_swap<F: Sized + Send>(mut a: MatrixMut<F>, mut b: MatrixMut<F>) {
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

        #[cfg(not(feature = "parallel"))]
        {
            transpose_square_swap(aa, ba);
            transpose_square_swap(ab, bc);
            transpose_square_swap(ac, bb);
            transpose_square_swap(ad, bd);
        }
        #[cfg(feature = "parallel")]
        join(
            || {
                join(
                    || transpose_square_swap(aa, ba),
                    || transpose_square_swap(ab, bc),
                )
            },
            || {
                join(
                    || transpose_square_swap(ac, bb),
                    || transpose_square_swap(ad, bd),
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



#[cfg(test)]
mod tests{
    use std::ops::Index;

    use super::*;

    type F = i32; // for simplicity
    impl<'a,'b> PartialEq for MatrixMut::<'a, F>{
        fn eq(&self, other: &Self) -> bool{
            let r = self.rows();
            let c = self.cols();

            if other.rows() != r || other.cols() != c{
                return false;
            }
            for i in 0..r{
                for j in 0..c{
                 if self[(i,j)] != other[(i,j)]{
                    return false
                 }
                }
            }
            true

        }
    }

    #[test]
    fn test_transpose_copy(){

        // iterate over both parallel and non-parallel implementation.
        // Needs HRTB, otherwise it won't work.
        let mut funs: Vec<&dyn for<'a,'b> Fn(MatrixMut<'a,F>, MatrixMut<'b,F>) > = vec![ &transpose_copy_not_parallel::<F>];
        #[cfg(feature="parallel")]
        funs.push( &transpose_copy_parallel::<F>);

        for f in funs{
            
            let mut srcarray = [1,2,3,4,5,6];
            let mut src1 = MatrixMut::<F>::from_mut_slice(&mut srcarray[..], 2, 3);

            let mut dstarray = [0;6];
            let mut dst1 = MatrixMut::<F>::from_mut_slice(&mut dstarray[..], 3, 2);
            f(src1, dst1);
            
            let intended_result = [1,4,2,5,3,6];
            // assert_eq!(dstarray, intended_result);
        }
    }

}