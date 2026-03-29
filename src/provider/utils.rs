//! Shared utilities for hash-based polynomial commitment schemes.
//!
//! Provides a cache-oblivious matrix transpose and a strided matrix view (`MatrixMut`)
//! used by NTT-based Reed-Solomon encoding and other PCS infrastructure.
// Ported from https://github.com/WizardOfMenlo/whir

use std::{
  marker::PhantomData,
  mem::{size_of, swap},
  ops::{Index, IndexMut},
  ptr, slice,
};

/// Target single-thread workload size for `T`.
/// Should ideally be a multiple of a cache line (64 bytes)
/// and close to the L1 cache size.
pub const fn workload_size<T: Sized>() -> usize {
  #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
  const CACHE_SIZE: usize = 1 << 17; // 128KB for Apple Silicon

  #[cfg(all(
    target_arch = "aarch64",
    any(target_os = "ios", target_os = "android", target_os = "linux")
  ))]
  const CACHE_SIZE: usize = 1 << 16; // 64KB for mobile/server ARM

  #[cfg(target_arch = "x86_64")]
  const CACHE_SIZE: usize = 1 << 15; // 32KB for x86-64

  #[cfg(not(any(
    all(target_arch = "aarch64", target_os = "macos"),
    all(
      target_arch = "aarch64",
      any(target_os = "ios", target_os = "android", target_os = "linux")
    ),
    target_arch = "x86_64"
  )))]
  const CACHE_SIZE: usize = 1 << 15; // 32KB default

  CACHE_SIZE / size_of::<T>()
}

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

unsafe impl<T: Send> Send for MatrixMut<'_, T> {}

unsafe impl<T: Sync> Sync for MatrixMut<'_, T> {}

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
  pub const fn rows(&self) -> usize {
    self.rows
  }

  /// returns the number of columns
  pub const fn cols(&self) -> usize {
    self.cols
  }

  /// checks whether the matrix is a square matrix
  pub const fn is_square(&self) -> bool {
    self.rows == self.cols
  }

  /// returns a mutable reference to the `row`'th row of the MatrixMut
  #[allow(dead_code)]
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

  /// Split the matrix into four quadrants at the indicated `row` and `col` (meaning that in the returned 4-tuple (A,B,C,D), the matrix A is a `row`x`col` matrix)
  ///
  /// self = [A B]
  ///        [C D]
  pub fn split_quadrants(self, row: usize, col: usize) -> (Self, Self, Self, Self) {
    let (u, l) = self.split_vertical(row); // split into upper and lower parts
    let (a, b) = u.split_horizontal(col);
    let (c, d) = l.split_horizontal(col);
    (a, b, c, d)
  }

  /// Swap two elements `a` and `b` in the matrix.
  /// Each of `a`, `b` is given as (row,column)-pair.
  /// If the given coordinates are out-of-bounds, the behaviour is undefined.
  pub unsafe fn swap(&mut self, a: (usize, usize), b: (usize, usize)) {
    if a != b {
      unsafe {
        let a = self.ptr_at_mut(a.0, a.1);
        let b = self.ptr_at_mut(b.0, b.1);
        ptr::swap_nonoverlapping(a, b, 1);
      }
    }
  }

  /// returns an immutable pointer to the element at (`row`, `col`). This performs no bounds checking and provining indices out-of-bounds is UB.
  pub(crate) const unsafe fn ptr_at(&self, row: usize, col: usize) -> *const T {
    // Safety: The structure invariant guarantees that at offset `row * self.row_stride + col`
    // there is valid data.
    unsafe { self.data.add(row * self.row_stride + col) }
  }

  /// returns a mutable pointer to the element at (`row`, `col`). This performs no bounds checking and providing indices out-of-bounds is UB.
  pub(crate) const unsafe fn ptr_at_mut(&mut self, row: usize, col: usize) -> *mut T {
    // Safety: The structure invariant guarantees that at offset `row * self.row_stride + col`
    // there is valid data.
    unsafe { self.data.add(row * self.row_stride + col) }
  }
}

// Use MatrixMut::ptr_at and MatrixMut::ptr_at_mut to implement Index and IndexMut. The latter are not unsafe, since they contain bounds-checks.

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

// NOTE: The assumption that rows and cols are a power of two are actually only relevant for the square matrix case.
// (This is because the algorithm recurses into 4 sub-matrices of half dimension; we assume those to be square matrices as well, which only works for powers of two).

/// Compute the row-major index permutation for a transposition.
pub fn transpose_permute(index: usize, rows: usize, cols: usize) -> usize {
  debug_assert!(index < rows * cols);
  let (row, col) = (index / cols, index % cols);
  row + col * rows
}

/// Transposes a matrix in-place.
///
/// This function processes a batch of matrices if the slice length is a multiple of `rows * cols`.
/// Assumes that both `rows` and `cols` are powers of two.
pub fn transpose<F: Sized + Copy + Send>(matrix: &mut [F], rows: usize, cols: usize) {
  assert!(matrix.len().is_multiple_of(rows * cols));
  if !rows.is_power_of_two() || !cols.is_power_of_two() {
    // Fall back to non-recursive.
    if matrix.is_empty() {
      return;
    }
    let mut buffer = vec![matrix[0]; rows * cols];
    for matrix in matrix.chunks_exact_mut(rows * cols) {
      transpose_copy(
        MatrixMut::from_mut_slice(matrix, rows, cols),
        MatrixMut::from_mut_slice(buffer.as_mut_slice(), cols, rows),
      );
      matrix.copy_from_slice(&buffer);
    }
    return;
  }

  debug_assert!(rows.is_power_of_two());
  debug_assert!(cols.is_power_of_two());
  if rows == cols {
    for matrix in matrix.chunks_exact_mut(rows * cols) {
      let matrix = MatrixMut::from_mut_slice(matrix, rows, cols);
      transpose_square(matrix);
    }
  } else {
    // TODO: Special case for rows = 2 * cols and cols = 2 * rows.
    // TODO: Special case for very wide matrices (e.g. n x 16).
    let mut scratch = Vec::with_capacity(rows * cols);
    for matrix in matrix.chunks_exact_mut(rows * cols) {
      scratch.clear();
      scratch.extend_from_slice(matrix);
      let src = MatrixMut::from_mut_slice(scratch.as_mut_slice(), rows, cols);
      let dst = MatrixMut::from_mut_slice(matrix, cols, rows);
      transpose_copy(src, dst);
    }
  }
}

/// Transposes a rectangular matrix into another matrix.
fn transpose_copy<F: Sized + Copy + Send>(src: MatrixMut<'_, F>, mut dst: MatrixMut<'_, F>) {
  assert_eq!(src.rows(), dst.cols());
  assert_eq!(src.cols(), dst.rows());

  let (rows, cols) = (src.rows(), src.cols());

  // Direct element-wise transposition for small matrices (avoids recursion overhead)
  if rows * cols * 2 <= workload_size::<F>() {
    unsafe {
      for i in 0..rows {
        for j in 0..cols {
          *dst.ptr_at_mut(j, i) = *src.ptr_at(i, j);
        }
      }
    }
    return;
  }

  // Determine optimal split axis
  let (src_a, src_b, dst_a, dst_b) = if rows > cols {
    let split_size = rows / 2;
    let (s1, s2) = src.split_vertical(split_size);
    let (d1, d2) = dst.split_horizontal(split_size);
    (s1, s2, d1, d2)
  } else {
    let split_size = cols / 2;
    let (s1, s2) = src.split_horizontal(split_size);
    let (d1, d2) = dst.split_vertical(split_size);
    (s1, s2, d1, d2)
  };

  rayon::join(
    || transpose_copy(src_a, dst_a),
    || transpose_copy(src_b, dst_b),
  );
}

/// Transposes a square matrix in-place.
fn transpose_square<F: Sized + Send>(mut m: MatrixMut<'_, F>) {
  debug_assert!(m.is_square());
  debug_assert!(m.rows().is_power_of_two());
  let size = m.rows();

  if size * size > workload_size::<F>() {
    // Recurse into quadrants.
    // This results in a cache-oblivious algorithm.
    let n = size / 2;
    let (a, b, c, d) = m.split_quadrants(n, n);

    rayon::join(
      || transpose_square_swap(b, c),
      || rayon::join(|| transpose_square(a), || transpose_square(d)),
    );
  } else {
    for i in 0..size {
      for j in (i + 1)..size {
        unsafe {
          m.swap((i, j), (j, i));
        }
      }
    }
  }
}

/// Swaps two square sub-matrices in-place, transposing them simultaneously.
fn transpose_square_swap<F: Sized + Send>(mut a: MatrixMut<'_, F>, mut b: MatrixMut<'_, F>) {
  debug_assert!(a.is_square());
  debug_assert_eq!(a.rows(), b.cols());
  debug_assert_eq!(a.cols(), b.rows());
  debug_assert!(a.rows().is_power_of_two());
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
    let (a_tl, a_tr, a_bl, a_br) = a.split_quadrants(n, n);
    let (b_tl, b_tr, b_bl, b_br) = b.split_quadrants(n, n);

    rayon::join(
      || {
        rayon::join(
          || transpose_square_swap(a_tl, b_tl),
          || transpose_square_swap(a_tr, b_bl),
        )
      },
      || {
        rayon::join(
          || transpose_square_swap(a_bl, b_tr),
          || transpose_square_swap(a_br, b_br),
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

#[cfg(test)]
mod tests {
  use super::*;
  use proptest::prelude::*;

  #[test]
  fn test_matrix_creation() {
    let mut data = vec![1, 2, 3, 4, 5, 6];
    let matrix = MatrixMut::from_mut_slice(&mut data, 2, 3);
    assert_eq!(matrix.rows(), 2);
    assert_eq!(matrix.cols(), 3);
    assert!(!matrix.is_square());
    assert_eq!(matrix[(0, 0)], 1);
    assert_eq!(matrix[(1, 0)], 4);

    assert_eq!(matrix[(0, 1)], 2);
    assert_eq!(matrix[(1, 1)], 5);

    assert_eq!(matrix[(0, 2)], 3);
    assert_eq!(matrix[(1, 2)], 6);
  }

  #[test]
  fn test_row_access() {
    let mut data = vec![1, 2, 3, 4, 5, 6];
    let mut matrix = MatrixMut::from_mut_slice(&mut data, 2, 3);
    assert_eq!(matrix.row(0), &[1, 2, 3]);
    assert_eq!(matrix.row(1), &[4, 5, 6]);
  }

  #[test]
  fn test_split_vertical() {
    let mut data = vec![1, 2, 3, 4, 5, 6];
    let matrix = MatrixMut::from_mut_slice(&mut data, 2, 3);
    let (top, bottom) = matrix.split_vertical(1);
    assert_eq!(top.rows(), 1);
    assert_eq!(top[(0, 0)], 1);
    assert_eq!(top[(0, 1)], 2);
    assert_eq!(top[(0, 2)], 3);

    assert_eq!(bottom.rows(), 1);
    assert_eq!(bottom[(0, 0)], 4);
    assert_eq!(bottom[(0, 1)], 5);
    assert_eq!(bottom[(0, 2)], 6);
  }

  #[test]
  fn test_split_horizontal() {
    let mut data = vec![1, 2, 3, 4, 5, 6];
    let matrix = MatrixMut::from_mut_slice(&mut data, 2, 3);
    let (left, right) = matrix.split_horizontal(1);
    assert_eq!(left.cols(), 1);
    assert_eq!(left[(0, 0)], 1);
    assert_eq!(left[(1, 0)], 4);

    assert_eq!(right.cols(), 2);
    assert_eq!(right[(0, 0)], 2);
    assert_eq!(right[(0, 1)], 3);
    assert_eq!(right[(1, 0)], 5);
    assert_eq!(right[(1, 1)], 6);
  }

  #[test]
  fn test_element_access() {
    let mut data = vec![1, 2, 3, 4, 5, 6];
    let matrix = MatrixMut::from_mut_slice(&mut data, 2, 3);
    assert_eq!(matrix[(0, 1)], 2);
  }

  #[test]
  fn test_swap() {
    let mut data = vec![1, 2, 3, 4, 5, 6];
    let mut matrix = MatrixMut::from_mut_slice(&mut data, 2, 3);
    unsafe {
      matrix.swap((0, 0), (1, 1));
    }
    assert_eq!(matrix[(0, 0)], 5);
    assert_eq!(matrix[(1, 1)], 1);
  }

  #[test]
  fn test_split_quadrants_even() {
    let mut data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    let matrix = MatrixMut::from_mut_slice(&mut data, 4, 4);

    let (a, b, c, d) = matrix.split_quadrants(2, 2);

    // Check dimensions
    assert_eq!(a.rows(), 2);
    assert_eq!(a.cols(), 2);
    assert_eq!(b.rows(), 2);
    assert_eq!(b.cols(), 2);
    assert_eq!(c.rows(), 2);
    assert_eq!(c.cols(), 2);
    assert_eq!(d.rows(), 2);
    assert_eq!(d.cols(), 2);

    // Check values in quadrants
    assert_eq!(a[(0, 0)], 1);
    assert_eq!(a[(0, 1)], 2);
    assert_eq!(a[(1, 0)], 5);
    assert_eq!(a[(1, 1)], 6);

    assert_eq!(b[(0, 0)], 3);
    assert_eq!(b[(0, 1)], 4);
    assert_eq!(b[(1, 0)], 7);
    assert_eq!(b[(1, 1)], 8);

    assert_eq!(c[(0, 0)], 9);
    assert_eq!(c[(0, 1)], 10);
    assert_eq!(c[(1, 0)], 13);
    assert_eq!(c[(1, 1)], 14);

    assert_eq!(d[(0, 0)], 11);
    assert_eq!(d[(0, 1)], 12);
    assert_eq!(d[(1, 0)], 15);
    assert_eq!(d[(1, 1)], 16);
  }

  #[test]
  fn test_split_quadrants_odd_rows() {
    let mut data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
    let matrix = MatrixMut::from_mut_slice(&mut data, 3, 3);

    let (a, b, c, d) = matrix.split_quadrants(1, 1);

    // Check dimensions
    assert_eq!(a.rows(), 1);
    assert_eq!(a.cols(), 1);
    assert_eq!(b.rows(), 1);
    assert_eq!(b.cols(), 2);
    assert_eq!(c.rows(), 2);
    assert_eq!(c.cols(), 1);
    assert_eq!(d.rows(), 2);
    assert_eq!(d.cols(), 2);

    // Check values in quadrants
    assert_eq!(a[(0, 0)], 1);

    assert_eq!(b[(0, 0)], 2);
    assert_eq!(b[(0, 1)], 3);

    assert_eq!(c[(0, 0)], 4);
    assert_eq!(c[(1, 0)], 7);

    assert_eq!(d[(0, 0)], 5);
    assert_eq!(d[(0, 1)], 6);
    assert_eq!(d[(1, 0)], 8);
    assert_eq!(d[(1, 1)], 9);
  }

  #[test]
  fn test_split_quadrants_odd_cols() {
    let mut data = vec![1, 2, 3, 4, 5, 6];
    let matrix = MatrixMut::from_mut_slice(&mut data, 3, 2);

    let (a, b, c, d) = matrix.split_quadrants(1, 1);

    // Check dimensions
    assert_eq!(a.rows(), 1);
    assert_eq!(a.cols(), 1);
    assert_eq!(b.rows(), 1);
    assert_eq!(b.cols(), 1);
    assert_eq!(c.rows(), 2);
    assert_eq!(c.cols(), 1);
    assert_eq!(d.rows(), 2);
    assert_eq!(d.cols(), 1);

    // Check values in quadrants
    assert_eq!(a[(0, 0)], 1);

    assert_eq!(b[(0, 0)], 2);

    assert_eq!(c[(0, 0)], 3);
    assert_eq!(c[(1, 0)], 5);

    assert_eq!(d[(0, 0)], 4);
    assert_eq!(d[(1, 0)], 6);
  }

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
  #[allow(clippy::type_complexity)]
  fn test_transpose_copy() {
    let rows: usize = workload_size::<Pair>() + 1; // intentionally not a power of two: The function is not described as only working for powers of two.
    let columns: usize = 4;
    let mut srcarray = make_example_matrix(rows, columns);
    let mut dstarray: Vec<(usize, usize)> = vec![(0, 0); rows * columns];

    let src1 = MatrixMut::<Pair>::from_mut_slice(&mut srcarray[..], rows, columns);
    let dst1 = MatrixMut::<Pair>::from_mut_slice(&mut dstarray[..], columns, rows);

    transpose_copy(src1, dst1);
    let dst1 = MatrixMut::<Pair>::from_mut_slice(&mut dstarray[..], columns, rows);

    for i in 0..rows {
      for j in 0..columns {
        assert_eq!(dst1[(j, i)], (i, j));
      }
    }
  }

  #[test]
  fn test_transpose_square_swap() {
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
    transpose_square_swap(view1, view2);
    let view1 = MatrixMut::from_mut_slice(&mut examples1, rows, rows);
    let view2 = MatrixMut::from_mut_slice(&mut examples2, rows, rows);
    for i in 0..rows {
      for j in 0..rows {
        assert_eq!(view1[(i, j)], (1, j, i));
        assert_eq!(view2[(i, j)], (0, j, i));
      }
    }
  }

  #[test]
  fn test_transpose_square() {
    // Set rows manually. We want to be sure to trigger the actual recursion.
    // (Computing this from workload_size was too much hassle.)
    let size = 1024;
    assert!(size * size > 2 * workload_size::<Pair>());

    let mut example = make_example_matrix(size, size);
    let view = MatrixMut::from_mut_slice(&mut example, size, size);
    transpose_square(view);
    let view = MatrixMut::from_mut_slice(&mut example, size, size);
    for i in 0..size {
      for j in 0..size {
        assert_eq!(view[(i, j)], (j, i));
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

  fn arb_pow2() -> impl Strategy<Value = usize> {
    (0..=8).prop_map(|log2_size| 1_usize << log2_size)
  }

  /// Generates random square matrices with sizes that are powers of two.
  #[allow(clippy::cast_sign_loss)]
  fn arb_square_matrix() -> impl Strategy<Value = (Vec<usize>, usize)> {
    arb_pow2()
      .prop_map(|size| size * size)
      .prop_flat_map(|matrix_size| {
        prop::collection::vec(0usize..1000, matrix_size)
          .prop_map(move |matrix| (matrix, (matrix_size as f64).sqrt() as usize))
      })
  }

  /// Generates random rectangular matrices where rows and columns are powers of two.
  fn arb_rect_matrix() -> impl Strategy<Value = (Vec<usize>, usize, usize)> {
    (arb_pow2(), arb_pow2()).prop_flat_map(|(rows, cols)| {
      prop::collection::vec(0usize..1000, rows * cols).prop_map(move |matrix| (matrix, rows, cols))
    })
  }

  proptest! {
      #[test]
      fn proptest_transpose_square((mut matrix, size) in arb_square_matrix()) {
          let original = matrix.clone();
          transpose(&mut matrix, size, size);
          transpose(&mut matrix, size, size);
          prop_assert_eq!(matrix, original);
      }

      #[test]
      fn proptest_transpose_rect((mut matrix, rows, cols) in arb_rect_matrix()) {
          let original = matrix.clone();
          transpose(&mut matrix, rows, cols);

          let view = MatrixMut::from_mut_slice(&mut matrix, cols, rows);

          // Verify that each (i, j) moved to (j, i)
          for i in 0..cols {
              for j in 0..rows {
                  prop_assert_eq!(view[(i, j)], original[j * cols + i]);
              }
          }
      }
  }
}
