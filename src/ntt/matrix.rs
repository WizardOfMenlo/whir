//! Minimal matrix class that supports strided access.
//! This abstracts over the unsafe pointer arithmetic required for transpose-like algorithms.

#![allow(unsafe_code)]

use std::{
    marker::PhantomData,
    ops::{Index, IndexMut},
    ptr, slice,
};

/// Mutable reference to a matrix.
///
/// The invariant this data structure maintains is that `data` has lifetime
/// `'a` and points to a collection of `rows` rowws, at intervals `row_stride`,
/// each of length `cols`.
pub struct MatrixMut<'a, T> {
    data: *mut T,
    rows: usize,
    cols: usize,
    row_stride: usize,
    _lifetime: PhantomData<&'a mut T>,
}

unsafe impl<'a, T: Send> Send for MatrixMut<'_, T> {}

unsafe impl<'a, T: Sync> Sync for MatrixMut<'_, T> {}

impl<'a, T> MatrixMut<'a, T> {
    /// creates a MatrixMut from `slice`, where slice is the concatenations of `rows` rows, each consisting of `cols` many entries.
    pub fn from_mut_slice(slice: &'a mut [T], rows: usize, cols: usize) -> Self {
        assert_eq!(slice.len(), rows * cols);
        // Safety: The input slice is valid for the lifetime `'a` and has
        // `rows` contiguous rows of length `cols`.
        Self {
            data: slice.as_mut_ptr(),
            rows,
            cols,
            row_stride: cols,
            _lifetime: PhantomData,
        }
    }

    /// returns the number of rows
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// returns the number of columns
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// checks whether the matrix is a square matrix
    pub fn is_square(&self) -> bool {
        self.rows == self.cols
    }

    /// returns a mutable reference to the `row`'th row of the MatrixMut
    pub fn row(&mut self, row: usize) -> &mut [T] {
        assert!(row < self.rows);
        // Safety: The structure invariant guarantees that at offset `row * self.row_stride`
        // there is valid data of length `self.cols`.
        unsafe { slice::from_raw_parts_mut(self.data.add(row * self.row_stride), self.cols) }
    }

    /// Split the matrix into two vertically at the `row`'th row (meaning that in the returned pair (A,B), the matrix A has `row` rows).
    ///
    /// [A]
    /// [ ] = self
    /// [B]
    pub fn split_vertical(self, row: usize) -> (Self, Self) {
        assert!(row <= self.rows);
        (
            Self {
                data: self.data,
                rows: row,
                cols: self.cols,
                row_stride: self.row_stride,
                _lifetime: PhantomData,
            },
            Self {
                data: unsafe { self.data.add(row * self.row_stride) },
                rows: self.rows - row,
                cols: self.cols,
                row_stride: self.row_stride,
                _lifetime: PhantomData,
            },
        )
    }

    /// Split the matrix into two horizontally at the `col`th column (meaning that in the returned pair (A,B), the matrix A has `col` columns).
    ///
    /// [A B] = self
    pub fn split_horizontal(self, col: usize) -> (Self, Self) {
        assert!(col <= self.cols);
        (
            // Safety: This reduces the number of cols, keeping all else the same.
            Self {
                data: self.data,
                rows: self.rows,
                cols: col,
                row_stride: self.row_stride,
                _lifetime: PhantomData,
            },
            // Safety: This reduces the number of cols and offsets and, keeping all else the same.
            Self {
                data: unsafe { self.data.add(col) },
                rows: self.rows,
                cols: self.cols - col,
                row_stride: self.row_stride,
                _lifetime: PhantomData,
            },
        )
    }

    /// Split the matrix into four quadrants at the indicated `row` and `col` (meaning that in the returned 4-tuple (A,B,C,D), the matrix A is a `row`x`col` )
    ///
    /// [A B] = self
    /// [C D]
    pub fn split_quadrants(self, row: usize, col: usize) -> (Self, Self, Self, Self) {
        let (u, l) = self.split_vertical(row); // split into upper and lower parts
        let (a, b) = u.split_horizontal(col);
        let (c, d) = l.split_horizontal(col);
        (a, b, c, d)
    }

    /// Swap two elements in the matrix.
    pub fn swap(&mut self, a: (usize, usize), b: (usize, usize)) {
        if a != b {
            unsafe {
                let a = self.ptr_at_mut(a.0, a.1);
                let b = self.ptr_at_mut(b.0, b.1);
                ptr::swap_nonoverlapping(a, b, 1)
            }
        }
    }

    unsafe fn ptr_at(&self, row: usize, col: usize) -> *const T {
        assert!(row < self.rows);
        assert!(col < self.cols);
        // Safety: The structure invariant guarantees that at offset `row * self.row_stride + col`
        // there is valid data.
        self.data.add(row * self.row_stride + col)
    }

    unsafe fn ptr_at_mut(&mut self, row: usize, col: usize) -> *mut T {
        assert!(row < self.rows);
        assert!(col < self.cols);
        // Safety: The structure invariant guarantees that at offset `row * self.row_stride + col`
        // there is valid data.
        self.data.add(row * self.row_stride + col)
    }
}

impl<T> Index<(usize, usize)> for MatrixMut<'_, T> {
    type Output = T;

    fn index(&self, (row, col): (usize, usize)) -> &T {
        assert!(row < self.rows);
        assert!(col < self.cols);
        // Safety: The structure invariant guarantees that at offset `row * self.row_stride + col`
        // there is valid data.
        unsafe { &*self.ptr_at(row, col) }
    }
}

impl<T> IndexMut<(usize, usize)> for MatrixMut<'_, T> {
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut T {
        assert!(row < self.rows);
        assert!(col < self.cols);
        // Safety: The structure invariant guarantees that at offset `row * self.row_stride + col`
        // there is valid data.
        unsafe { &mut *self.ptr_at_mut(row, col) }
    }
}
