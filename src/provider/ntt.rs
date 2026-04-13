//! Number-theoretic transforms (NTTs) over fields with high two-adicity.
//!
//! Implements the √N Cooley-Tukey six-step algorithm to achieve parallelism with good locality.
//! A global cache is used for twiddle factors.
//! Reference: https://github.com/WizardOfMenlo/whir

use std::{
  any::{Any, TypeId},
  collections::HashMap,
  sync::{Arc, LazyLock, Mutex, RwLock, RwLockReadGuard},
};

use ff::PrimeField;
use rayon::prelude::*;
use std::cmp::max;

use super::utils::{transpose, workload_size};

/// Global cache for NTT engines, indexed by field.
// TODO: Skip `LazyLock` when `HashMap::with_hasher` becomes const.
// see https://github.com/rust-lang/rust/issues/102575
static ENGINE_CACHE: LazyLock<Mutex<HashMap<TypeId, Arc<dyn Any + Send + Sync>>>> =
  LazyLock::new(|| Mutex::new(HashMap::new()));

/// Engine for computing NTTs over arbitrary fields.
/// Assumes the field has large two-adicity.
#[derive(Debug)]
pub struct NttEngine<F: PrimeField> {
  order: usize,         // order of omega_orger
  divisors: Vec<usize>, // divisors of the order.
  omega_order: F,       // primitive order'th root.

  // Roots of small order (zero if unavailable). The naming convention is that omega_foo has order foo.
  half_omega_3_1_plus_2: F, // ½(ω₃ + ω₃²)
  half_omega_3_1_min_2: F,  // ½(ω₃ - ω₃²)
  omega_4_1: F,
  omega_8_1: F,
  omega_8_3: F,
  omega_16_1: F,
  omega_16_3: F,
  omega_16_9: F,

  // Root lookup table (extended on demand)
  roots: RwLock<Vec<F>>,
}

/// Returns the root-of-unity used for a domain of size `size`.
pub fn generator<F: PrimeField>(size: usize) -> Option<F> {
  NttEngine::<F>::new_from_cache().checked_root(size)
}

/// Compute the NTT of a slice of field elements using a cached engine.
pub fn ntt<F: PrimeField>(values: &mut [F]) {
  NttEngine::<F>::new_from_cache().ntt(values);
}

/// Compute the many NTTs of size `size` using a cached engine.
pub fn ntt_batch<F: PrimeField>(values: &mut [F], size: usize) {
  NttEngine::<F>::new_from_cache().ntt_batch(values, size);
}

/// Compute the inverse NTT of a slice of field element without the 1/n scaling factor, using a cached engine.
pub fn intt<F: PrimeField>(values: &mut [F]) {
  NttEngine::<F>::new_from_cache().intt(values);
}

/// Compute the inverse NTT of multiple slice of field elements, each of size `size`, without the 1/n scaling factor and using a cached engine.
pub fn intt_batch<F: PrimeField>(values: &mut [F], size: usize) {
  NttEngine::<F>::new_from_cache().intt_batch(values, size);
}

/// Compute an interleaved Reed-Solomon encoding using a cached engine.
pub fn interleaved_rs_encode<F: PrimeField>(
  messages: &[&[F]],
  masks: &[F],
  codeword_length: usize,
) -> Vec<F> {
  NttEngine::<F>::new_from_cache().interleaved_encode(messages, masks, codeword_length)
}

impl<F: PrimeField> NttEngine<F> {
  /// Get or create a cached engine for the field `F`.
  pub fn new_from_cache() -> Arc<Self> {
    let mut cache = ENGINE_CACHE.lock().unwrap();
    let type_id = TypeId::of::<F>();
    #[allow(clippy::option_if_let_else)]
    if let Some(engine) = cache.get(&type_id) {
      engine.clone().downcast::<Self>().unwrap()
    } else {
      let engine = Arc::new(Self::new_from_PrimeField());
      cache.insert(type_id, engine.clone());
      engine
    }
  }

  /// Construct a new engine from the field's `PrimeField` trait.
  pub(crate) fn new_from_PrimeField() -> Self {
    // TODO: Support SMALL_SUBGROUP
    if F::S <= 63 {
      Self::new(1 << F::S, F::ROOT_OF_UNITY)
    } else {
      let mut generator = F::ROOT_OF_UNITY;
      for _ in 0..(F::S - 63) {
        generator = generator.square();
      }
      Self::new(1usize << (usize::BITS - 1), generator)
    }
  }
}

/// Creates a new NttEngine. `omega_order` must be a primitive root of unity of even order `omega`.
impl<F: PrimeField> NttEngine<F> {
  pub fn new(order: usize, omega_order: F) -> Self {
    assert!(order.trailing_zeros() > 0, "Order must be a multiple of 2.");
    // TODO: Assert that omega factors into 2s and 3s.
    assert_eq!(omega_order.pow([order as u64]), F::ONE);
    assert_ne!(omega_order.pow([order as u64 / 2]), F::ONE);
    let mut res = Self {
      order,
      omega_order,
      half_omega_3_1_plus_2: F::ZERO,
      half_omega_3_1_min_2: F::ZERO,
      omega_4_1: F::ZERO,
      omega_8_1: F::ZERO,
      omega_8_3: F::ZERO,
      omega_16_1: F::ZERO,
      omega_16_3: F::ZERO,
      omega_16_9: F::ZERO,
      roots: RwLock::new(Vec::new()),
      divisors: divisors(order, &[2, 3]),
    };
    if order.is_multiple_of(3) {
      let omega_3_1 = res.root(3);
      let omega_3_2 = omega_3_1 * omega_3_1;
      // Note: char F cannot be 2 and so division by 2 works, because primitive roots of unity with even order exist.
      let two_inv = F::from(2u64).invert().unwrap();
      res.half_omega_3_1_min_2 = (omega_3_1 - omega_3_2) * two_inv;
      res.half_omega_3_1_plus_2 = (omega_3_1 + omega_3_2) * two_inv;
    }
    if order.is_multiple_of(4) {
      res.omega_4_1 = res.root(4);
    }
    if order.is_multiple_of(8) {
      res.omega_8_1 = res.root(8);
      res.omega_8_3 = res.omega_8_1.pow([3]);
    }
    if order.is_multiple_of(16) {
      res.omega_16_1 = res.root(16);
      res.omega_16_3 = res.omega_16_1.pow([3]);
      res.omega_16_9 = res.omega_16_1.pow([9]);
    }
    res
  }

  pub fn ntt(&self, values: &mut [F]) {
    self.ntt_batch(values, values.len());
  }

  pub fn ntt_batch(&self, values: &mut [F], size: usize) {
    assert!(values.len().is_multiple_of(size));
    let roots = self.roots_table(size);
    self.ntt_dispatch(values, &roots, size);
  }

  /// Inverse NTT. Does not apply 1/n scaling factor.
  pub fn intt(&self, values: &mut [F]) {
    values[1..].reverse();
    self.ntt(values);
  }

  /// Inverse batch NTT. Does not apply 1/n scaling factor.
  pub fn intt_batch(&self, values: &mut [F], size: usize) {
    assert!(values.len().is_multiple_of(size));
    values.par_chunks_exact_mut(size).for_each(|values| {
      values[1..].reverse();
    });

    self.ntt_batch(values, size);
  }

  pub fn checked_root(&self, order: usize) -> Option<F> {
    self
      .order
      .is_multiple_of(order)
      .then(|| self.omega_order.pow([(self.order / order) as u64]))
  }

  pub fn root(&self, order: usize) -> F {
    self
      .checked_root(order)
      .expect("Subgroup of requested order does not exist.")
  }

  /// Returns a cached table of roots of unity of the given order.
  fn roots_table(&self, order: usize) -> RwLockReadGuard<'_, Vec<F>> {
    // Precompute more roots of unity if requested.
    let roots = self.roots.read().unwrap();
    if roots.is_empty() || !roots.len().is_multiple_of(order) {
      // Obtain write lock to update the cache.
      drop(roots);
      let mut roots = self.roots.write().unwrap();
      // Race condition: check if another thread updated the cache.
      if roots.is_empty() || !roots.len().is_multiple_of(order) {
        // Compute minimal size to support all sizes seen so far.
        // TODO: Do we really need all of these? Can we leverage omege_2 = -1?
        let size = if roots.is_empty() {
          order
        } else {
          lcm(roots.len(), order)
        };
        roots.clear();
        roots.reserve_exact(size);

        // Compute powers of roots of unity.
        let root = self.root(size);
        roots.par_extend((0..size).into_par_iter().map_with(F::ZERO, |root_i, i| {
          if root_i.is_zero().into() {
            *root_i = root.pow([i as u64]);
          } else {
            *root_i *= root;
          }
          *root_i
        }));
      }
      // Back to read lock.
      drop(roots);
      self.roots.read().unwrap()
    } else {
      roots
    }
  }

  /// Compute NTTs in place by splititng into two factors.
  /// Recurses using the sqrt(N) Cooley-Tukey Six step NTT algorithm.
  fn ntt_recurse(&self, values: &mut [F], roots: &[F], size: usize) {
    debug_assert_eq!(values.len() % size, 0);
    let n1 = sqrt_factor(size);
    let n2 = size / n1;

    transpose(values, n1, n2);
    self.ntt_dispatch(values, roots, n1);
    transpose(values, n2, n1);
    // TODO: When (n1, n2) are coprime we can use the
    // Good-Thomas NTT algorithm and avoid the twiddle loop.
    apply_twiddles(values, roots, n1, n2);
    self.ntt_dispatch(values, roots, n2);
    transpose(values, n1, n2);
  }

  fn ntt_dispatch(&self, values: &mut [F], roots: &[F], size: usize) {
    debug_assert_eq!(values.len() % size, 0);
    debug_assert_eq!(roots.len() % size, 0);
    if values.len() > workload_size::<F>() && values.len() != size {
      // Multiple NTTs, compute in parallel.
      // Work size is largest multiple of `size` smaller than `WORKLOAD_SIZE`.
      let workload_size = size * max(1, workload_size::<F>() / size);
      return values.par_chunks_mut(workload_size).for_each(|values| {
        self.ntt_dispatch(values, roots, size);
      });
    }
    match size {
      0 | 1 => {}
      2 => {
        for v in values.chunks_exact_mut(2) {
          (v[0], v[1]) = (v[0] + v[1], v[0] - v[1]);
        }
      }
      3 => {
        for v in values.chunks_exact_mut(3) {
          // Rader NTT to reduce 3 to 2.
          let v0 = v[0];
          (v[1], v[2]) = (v[1] + v[2], v[1] - v[2]);
          v[0] += v[1];
          v[1] *= self.half_omega_3_1_plus_2; // ½(ω₃ + ω₃²)
          v[2] *= self.half_omega_3_1_min_2; // ½(ω₃ - ω₃²)
          v[1] += v0;
          (v[1], v[2]) = (v[1] + v[2], v[1] - v[2]);
        }
      }
      4 => {
        for v in values.chunks_exact_mut(4) {
          (v[0], v[2]) = (v[0] + v[2], v[0] - v[2]);
          (v[1], v[3]) = (v[1] + v[3], v[1] - v[3]);
          v[3] *= self.omega_4_1;
          (v[0], v[1]) = (v[0] + v[1], v[0] - v[1]);
          (v[2], v[3]) = (v[2] + v[3], v[2] - v[3]);
          (v[1], v[2]) = (v[2], v[1]);
        }
      }
      8 => {
        for v in values.chunks_exact_mut(8) {
          // Cooley-Tukey with v as 2x4 matrix.
          (v[0], v[4]) = (v[0] + v[4], v[0] - v[4]);
          (v[1], v[5]) = (v[1] + v[5], v[1] - v[5]);
          (v[2], v[6]) = (v[2] + v[6], v[2] - v[6]);
          (v[3], v[7]) = (v[3] + v[7], v[3] - v[7]);
          v[5] *= self.omega_8_1;
          v[6] *= self.omega_4_1; // == omega_8_2
          v[7] *= self.omega_8_3;
          (v[0], v[2]) = (v[0] + v[2], v[0] - v[2]);
          (v[1], v[3]) = (v[1] + v[3], v[1] - v[3]);
          v[3] *= self.omega_4_1;
          (v[0], v[1]) = (v[0] + v[1], v[0] - v[1]);
          (v[2], v[3]) = (v[2] + v[3], v[2] - v[3]);
          (v[4], v[6]) = (v[4] + v[6], v[4] - v[6]);
          (v[5], v[7]) = (v[5] + v[7], v[5] - v[7]);
          v[7] *= self.omega_4_1;
          (v[4], v[5]) = (v[4] + v[5], v[4] - v[5]);
          (v[6], v[7]) = (v[6] + v[7], v[6] - v[7]);
          (v[1], v[4]) = (v[4], v[1]);
          (v[3], v[6]) = (v[6], v[3]);
        }
      }
      16 => {
        for v in values.chunks_exact_mut(16) {
          // Cooley-Tukey with v as 4x4 matrix.
          for i in 0..4 {
            let v = &mut v[i..];
            (v[0], v[8]) = (v[0] + v[8], v[0] - v[8]);
            (v[4], v[12]) = (v[4] + v[12], v[4] - v[12]);
            v[12] *= self.omega_4_1;
            (v[0], v[4]) = (v[0] + v[4], v[0] - v[4]);
            (v[8], v[12]) = (v[8] + v[12], v[8] - v[12]);
            (v[4], v[8]) = (v[8], v[4]);
          }
          v[5] *= self.omega_16_1;
          v[6] *= self.omega_8_1;
          v[7] *= self.omega_16_3;
          v[9] *= self.omega_8_1;
          v[10] *= self.omega_4_1;
          v[11] *= self.omega_8_3;
          v[13] *= self.omega_16_3;
          v[14] *= self.omega_8_3;
          v[15] *= self.omega_16_9;
          for i in 0..4 {
            let v = &mut v[i * 4..];
            (v[0], v[2]) = (v[0] + v[2], v[0] - v[2]);
            (v[1], v[3]) = (v[1] + v[3], v[1] - v[3]);
            v[3] *= self.omega_4_1;
            (v[0], v[1]) = (v[0] + v[1], v[0] - v[1]);
            (v[2], v[3]) = (v[2] + v[3], v[2] - v[3]);
            (v[1], v[2]) = (v[2], v[1]);
          }
          (v[1], v[4]) = (v[4], v[1]);
          (v[2], v[8]) = (v[8], v[2]);
          (v[3], v[12]) = (v[12], v[3]);
          (v[6], v[9]) = (v[9], v[6]);
          (v[7], v[13]) = (v[13], v[7]);
          (v[11], v[14]) = (v[14], v[11]);
        }
      }
      size => self.ntt_recurse(values, roots, size),
    }
  }
}

// ---------------------------------------------------------------------------
// ReedSolomon trait implementation
// ---------------------------------------------------------------------------

use super::{pcs::ReedSolomon, utils::transpose_permute};

impl<F: PrimeField> ReedSolomon<F> for NttEngine<F> {
  fn next_order(&self, size: usize) -> Option<usize> {
    match self.divisors.binary_search(&size) {
      Ok(index) | Err(index) => self.divisors.get(index).copied(),
    }
  }

  fn evaluation_points(
    &self,
    message_length: usize,
    codeword_length: usize,
    indices: &[usize],
  ) -> Vec<F> {
    assert!(message_length <= codeword_length);
    assert!(self.order.is_multiple_of(codeword_length));
    let mut result = Vec::new();
    let generator = self
      .omega_order
      .pow([(self.order / codeword_length) as u64]);

    // Coset transformation
    let mut coset_size = self.next_order(message_length).unwrap();
    while !codeword_length.is_multiple_of(coset_size) {
      coset_size = self.next_order(coset_size + 1).unwrap();
    }
    let num_cosets = codeword_length / coset_size;

    for &index in indices {
      assert!(index < codeword_length);
      let index = transpose_permute(index, num_cosets, coset_size);
      result.push(generator.pow([index as u64]));
    }
    result
  }

  fn interleaved_encode(&self, messages: &[&[F]], masks: &[F], codeword_length: usize) -> Vec<F> {
    assert!(self.order.is_multiple_of(codeword_length));
    if messages.is_empty() {
      assert!(masks.is_empty());
      return Vec::new();
    }
    let num_messages = messages.len();
    let message_len = messages[0].len();
    assert!(messages.iter().all(|m| m.len() == message_len));
    assert!(masks.len().is_multiple_of(num_messages));
    let mask_length = masks.len() / num_messages;
    let message_length = message_len + mask_length;
    assert!(message_length <= codeword_length);

    // Coset-NTT: instead of doing one codeword-length NTT on mostly zeros,
    // do `num_cosets` many `coset_size`-point NTTs on twisted coefficient
    // vectors. For coset `c`, we evaluate on points
    //
    //     ω_N^{c + j * num_cosets} = ω_N^c · (ω_N^{num_cosets})^j
    //
    // so the coefficient of X^i must be multiplied by (ω_N^c)^i.
    //
    // You can also see this as applying a first round of Cooley-Tukey with
    // N = coset_size × num_cosets, and solving it directly by observing that
    // only the first coset is non-zero.
    let mut coset_size = self.next_order(message_length).unwrap();
    while !codeword_length.is_multiple_of(coset_size) {
      coset_size = self.next_order(coset_size + 1).unwrap();
    }
    let num_cosets = codeword_length / coset_size;
    let coset_padding = coset_size - message_length;

    // Lay out twisted coefficients in contiguous coset blocks of length
    // `coset_size`, zero-padding each block as needed.
    let mut result = Vec::with_capacity(num_messages * codeword_length);
    let mask_chunks: Vec<&[F]> = if mask_length > 0 {
      masks.chunks_exact(mask_length).collect()
    } else {
      vec![&[]; num_messages]
    };
    assert_eq!(messages.len(), mask_chunks.len());
    for (message, mask) in messages.iter().zip(mask_chunks.iter()) {
      // FFT[a 0 0 0] = [a a a a], so just replicate input in coset dimension.
      for _ in 0..num_cosets {
        result.extend_from_slice(message);
        result.extend_from_slice(mask);
        result.resize(result.len() + coset_padding, F::ZERO);
      }
    }
    assert_eq!(result.len(), num_messages * codeword_length);

    // NTT each coset block, then transpose each codeword block from
    // coset-major `(num_cosets × coset_size)` layout into standard codeword
    // order `(coset_size × num_cosets)`, where global index is
    // `c + j * num_cosets`.
    apply_twiddles(
      &mut result,
      self.roots_table(codeword_length).as_slice(),
      num_cosets,
      coset_size,
    );
    self.ntt_batch(&mut result, coset_size);

    // Transpose to row-major order with vectors stacked horizontally.
    transpose(&mut result, num_messages, codeword_length);
    result
  }
}

/// Applies twiddle factors to a slice of field elements in-place.
///
/// This is part of the six-step Cooley-Tukey NTT algorithm,
/// where after transposing and partially transforming a 2D matrix,
/// we multiply each non-zero row and column entry by a scalar "twiddle factor"
/// derived from powers of a root of unity.
///
/// Given:
/// - `values`: a flattened set of NTT matrices, each with shape `[rows × cols]`
/// - `roots`: the root-of-unity table (should have length divisible by `rows * cols`)
/// - `rows`, `cols`: the dimensions of each matrix
///
/// This function mutates `values` in-place, applying twiddle factors like so:
///
/// ```text
/// values[i][j] *= roots[(i * step + j * step) % roots.len()]
/// ```
///
/// More specifically:
/// - The first row and column are left untouched (twiddle factor is 1).
/// - For each row `i > 0`, each element `values[i][j > 0]` is multiplied by a twiddle factor.
/// - The factor is taken as `roots[index]`, where:
///   - `index` starts at `step = (i * roots.len()) / (rows * cols)`
///   - `index` increments by `step` for each column.
///
/// ### Parallelism
/// - If `parallel` is enabled and `values.len()` exceeds a threshold:
///   - Large matrices are split into workloads and processed in parallel.
///   - If a single matrix is present, its rows are parallelized directly.
///
/// ### Panics
/// - If `values.len() % (rows * cols) != 0`
/// - If `roots.len()` is not divisible by `rows * cols`
///
/// ### Example
/// Suppose you have a `2×4` matrix:
/// ```text
/// [ a0 a1 a2 a3 ]
/// [ b0 b1 b2 b3 ]
/// ```
/// and `roots = [r0, r1, ..., rN]`, then the transformed matrix becomes:
/// ```text
/// [ a0  a1       a2       a3       ]
/// [ b0  b1*rX1   b2*rX2   b3*rX3   ]
/// ```
/// where `rX1`, `rX2`, etc., are powers of root-of-unity determined by row/col indices.
pub fn apply_twiddles<F: PrimeField>(values: &mut [F], roots: &[F], rows: usize, cols: usize) {
  let size = rows * cols;
  debug_assert_eq!(values.len() % size, 0);
  let step = roots.len() / size;

  {
    if values.len() > workload_size::<F>() {
      if values.len() == size {
        // Only one matrix → parallelize rows directly
        values
          .par_chunks_exact_mut(cols)
          .enumerate()
          .skip(1)
          .for_each(|(i, row)| {
            let step = (i * step) % roots.len();
            let mut index = step;
            for value in row.iter_mut().skip(1) {
              index %= roots.len();
              *value *= roots[index];
              index += step;
            }
          });
        return;
      }
      // Multiple matrices → chunk and recurse
      let workload_size = size * max(1, workload_size::<F>() / size);
      values
        .par_chunks_mut(workload_size)
        .for_each(|chunk| apply_twiddles(chunk, roots, rows, cols));
      return;
    }
  }

  // Fallback (non-parallel or small workload)
  for matrix in values.chunks_exact_mut(size) {
    for (i, row) in matrix.chunks_exact_mut(cols).enumerate().skip(1) {
      let step = (i * step) % roots.len();
      let mut index = step;
      for value in row.iter_mut().skip(1) {
        index %= roots.len();
        *value *= roots[index];
        index += step;
      }
    }
  }
}

/// Compute the largest factor of `n` that is ≤ sqrt(n).
/// Assumes `n` is of the form `2^k * {1,3,9}`.
pub fn sqrt_factor(n: usize) -> usize {
  // Count the number of trailing zeros in `n`, i.e., the power of 2 in `n`
  let twos = n.trailing_zeros();

  // Divide `n` by the highest power of 2 to extract the base component
  let base = n >> twos;

  // Determine the largest factor ≤ sqrt(n) based on the extracted `base`
  match base {
    // Case: `n` is purely a power of 2 (base = 1)
    // The largest factor ≤ sqrt(n) is 2^(twos/2)
    1 => 1 << (twos / 2),

    // Case: `n = 2^k * 3`
    3 => {
      if twos == 0 {
        // sqrt(3) ≈ 1.73, so the largest integer factor ≤ sqrt(3) is 1
        1
      } else {
        // - If `twos` is even: The largest factor is `3 * 2^((twos - 1) / 2)`
        // - If `twos` is odd: The largest factor is `2^((twos / 2))`
        if twos.is_multiple_of(2) {
          3 << ((twos - 1) / 2)
        } else {
          2 << (twos / 2)
        }
      }
    }

    // Case: `n = 2^k * 9`
    9 => {
      if twos == 1 {
        // sqrt(9 * 2^1) = sqrt(18) ≈ 4.24, largest factor ≤ sqrt(18) is 3
        3
      } else {
        // - If `twos` is even: The largest factor is `3 * 2^(twos / 2)`
        // - If `twos` is odd: The largest factor is `4 * 2^(twos / 2)`
        if twos.is_multiple_of(2) {
          3 << (twos / 2)
        } else {
          4 << (twos / 2)
        }
      }
    }

    // If `base` is not in {1,3,9}, `n` is not in the expected form
    _ => panic!("n is not in the form 2^k * {{1,3,9}}"),
  }
}

/// Least common multiple.
///
/// Note that lcm(0,0) will panic (rather than give the correct answer 0).
pub const fn lcm(a: usize, b: usize) -> usize {
  a * (b / gcd(a, b))
}

/// Greatest common divisor.
pub const fn gcd(mut a: usize, mut b: usize) -> usize {
  while b != 0 {
    (a, b) = (b, a % b);
  }
  a
}

/// Asserts that `n` factors over `primes` and returns a sorted list of divisors.
pub fn divisors(n: usize, primes: &[usize]) -> Vec<usize> {
  let mut result = vec![1usize];
  let mut remaining = n;
  for &p in primes {
    let mut pk = 1usize;
    let existing = result.clone();
    while remaining.is_multiple_of(p) {
      pk *= p;
      remaining /= p;
      result.extend(existing.iter().map(|d| d * pk));
    }
  }
  assert_eq!(remaining, 1);
  result.sort_unstable();
  result
}

#[cfg(test)]
#[allow(clippy::significant_drop_tightening)]
mod tests {
  use super::*;
  use ff::{Field, PrimeField};
  use halo2curves::bn256::Fr;
  use proptest::prelude::*;

  #[test]
  fn test_new_from_PrimeField_basic() {
    // Ensure that an engine is created correctly from FFT field properties
    let engine = NttEngine::<Fr>::new_from_PrimeField();

    // Verify that the order of the engine is correctly set
    assert!(engine.order.is_power_of_two());

    // Verify that the root of unity is correctly initialized
    let expected_root = Fr::ROOT_OF_UNITY;
    let computed_root = engine.root(engine.order);
    assert_eq!(computed_root.pow([engine.order as u64]), Fr::ONE);
    assert_eq!(computed_root, expected_root);
  }

  #[test]
  fn test_root_computation() {
    let engine = NttEngine::<Fr>::new_from_PrimeField();

    // Ensure the root exponentiates correctly
    assert_eq!(engine.root(8).pow([8]), Fr::ONE);
    assert_eq!(engine.root(4).pow([4]), Fr::ONE);
    assert_eq!(engine.root(2).pow([2]), Fr::ONE);

    // Ensure it's not a lower-order root
    assert_ne!(engine.root(8).pow([4]), Fr::ONE);
    assert_ne!(engine.root(4).pow([2]), Fr::ONE);
  }

  #[test]
  fn test_root_of_unity_multiplication() {
    let engine = NttEngine::<Fr>::new_from_PrimeField();

    let root = engine.root(16);

    // Multiply root by itself repeatedly and verify expected outcomes
    assert_eq!(root.pow([2]), engine.root(8));
    assert_eq!(root.pow([4]), engine.root(4));
    assert_eq!(root.pow([8]), engine.root(2));
  }

  #[test]
  fn test_root_of_unity_inversion() {
    let engine = NttEngine::<Fr>::new_from_PrimeField();
    let root = engine.root(16);
    let inverse_root = root.invert().unwrap();
    assert_eq!(root * inverse_root, Fr::ONE);
  }

  #[test]
  fn test_precomputed_small_roots() {
    let engine = NttEngine::<Fr>::new_from_PrimeField();

    // Check that precomputed values are correctly initialized
    assert_eq!(engine.omega_8_1.pow([8]), Fr::ONE);
    assert_eq!(engine.omega_8_3.pow([8]), Fr::ONE);
    assert_eq!(engine.omega_16_1.pow([16]), Fr::ONE);
    assert_eq!(engine.omega_16_3.pow([16]), Fr::ONE);
    assert_eq!(engine.omega_16_9.pow([16]), Fr::ONE);
  }

  #[test]
  fn test_consistency_across_multiple_instances() {
    let engine1 = NttEngine::<Fr>::new_from_PrimeField();
    let engine2 = NttEngine::<Fr>::new_from_PrimeField();

    // Ensure that multiple instances yield the same results
    assert_eq!(engine1.root(8), engine2.root(8));
    assert_eq!(engine1.root(4), engine2.root(4));
    assert_eq!(engine1.root(2), engine2.root(2));
  }

  #[test]
  fn test_roots_table_basic() {
    let engine = NttEngine::<Fr>::new_from_PrimeField();
    let roots_4 = engine.roots_table(4);

    // Check hardcoded expected values (ω^i)
    assert_eq!(roots_4[0], Field::ONE);
    assert_eq!(roots_4[1], engine.root(4));
    assert_eq!(roots_4[2], engine.root(4).pow([2]));
    assert_eq!(roots_4[3], engine.root(4).pow([3]));
  }

  #[test]
  fn test_roots_table_minimal_order() {
    let engine = NttEngine::<Fr>::new_from_PrimeField();

    let roots_2 = engine.roots_table(2);

    // Must contain only ω^0 and ω^1
    assert_eq!(roots_2.len(), 2);
    assert_eq!(roots_2[0], Fr::ONE);
    assert_eq!(roots_2[1], engine.root(2));
  }

  #[test]
  fn test_roots_table_progression() {
    let engine = NttEngine::<Fr>::new_from_PrimeField();

    let roots_4 = engine.roots_table(4);

    // Ensure the sequence follows expected powers of the root of unity
    for i in 0..4 {
      assert_eq!(roots_4[i], engine.root(4).pow([i as u64]));
    }
  }

  #[test]
  fn test_roots_table_cached_results() {
    let engine = NttEngine::<Fr>::new_from_PrimeField();

    let first_access = engine.roots_table(4);
    let second_access = engine.roots_table(4);

    // The memory location should be the same, meaning it's cached
    assert!(std::ptr::eq(first_access.as_ptr(), second_access.as_ptr()));
  }

  #[test]
  fn test_roots_table_recompute_factor_order() {
    let engine = NttEngine::<Fr>::new_from_PrimeField();

    let roots_4 = engine.roots_table(4);
    let roots_2 = engine.roots_table(2);

    // Ensure first two elements of roots_4 match the first two elements of roots_2
    assert_eq!(&roots_4[..2], &roots_2[..2]);
  }

  #[test]
  fn test_apply_twiddles_basic() {
    let engine = NttEngine::<Fr>::new_from_PrimeField();

    let mut values = vec![
      Fr::from(1),
      Fr::from(2),
      Fr::from(3),
      Fr::from(4),
      Fr::from(5),
      Fr::from(6),
      Fr::from(7),
      Fr::from(8),
    ];

    // Mock roots
    let r1 = Fr::from(33);
    let roots = vec![r1];

    // Ensure the root of unity is correct
    assert_eq!(engine.root(4).pow([4]), Fr::ONE);

    apply_twiddles(&mut values, &roots, 2, 4);

    // The first row should remain unchanged
    assert_eq!(values[0], Fr::from(1));
    assert_eq!(values[1], Fr::from(2));
    assert_eq!(values[2], Fr::from(3));
    assert_eq!(values[3], Fr::from(4));

    // The second row should be multiplied by the correct twiddle factors
    assert_eq!(values[4], Fr::from(5)); // No change for first column
    assert_eq!(values[5], Fr::from(6) * r1);
    assert_eq!(values[6], Fr::from(7) * r1);
    assert_eq!(values[7], Fr::from(8) * r1);
  }

  #[test]
  fn test_apply_twiddles_single_row() {
    let mut values = vec![Fr::from(1), Fr::from(2)];

    // Mock roots
    let r1 = Fr::from(12);
    let roots = vec![r1];

    apply_twiddles(&mut values, &roots, 1, 2);

    // Everything should remain unchanged
    assert_eq!(values[0], Fr::from(1));
    assert_eq!(values[1], Fr::from(2));
  }

  #[test]
  fn test_apply_twiddles_varying_rows() {
    let mut values = vec![
      Fr::from(1),
      Fr::from(2),
      Fr::from(3),
      Fr::from(4),
      Fr::from(5),
      Fr::from(6),
      Fr::from(7),
      Fr::from(8),
      Fr::from(9),
    ];

    // Mock roots
    let roots = (2..100).map(Fr::from).collect::<Vec<_>>();

    apply_twiddles(&mut values, &roots, 3, 3);

    // First row remains unchanged
    assert_eq!(values[0], Fr::from(1));
    assert_eq!(values[1], Fr::from(2));
    assert_eq!(values[2], Fr::from(3));

    // Second row multiplied by twiddle factors
    assert_eq!(values[3], Fr::from(4));
    assert_eq!(values[4], Fr::from(5) * roots[10]);
    assert_eq!(values[5], Fr::from(6) * roots[20]);

    // Third row multiplied by twiddle factors
    assert_eq!(values[6], Fr::from(7));
    assert_eq!(values[7], Fr::from(8) * roots[20]);
    assert_eq!(values[8], Fr::from(9) * roots[40]);
  }

  #[test]
  fn test_apply_twiddles_large_table() {
    let rows = 320;
    let cols = 320;
    let size = rows * cols;

    let mut values: Vec<Fr> = (0..size as u64).map(Fr::from).collect();

    // Generate a large set of twiddle factors
    let roots: Vec<Fr> = (0..(size * 2) as u64).map(Fr::from).collect();

    apply_twiddles(&mut values, &roots, rows, cols);

    // Verify the first row remains unchanged
    for (i, &col) in values.iter().enumerate().take(cols) {
      assert_eq!(col, Fr::from(i as u64));
    }

    // Verify the first column remains unchanged
    for row in 1..rows {
      let index = row * cols;
      assert_eq!(
        values[index],
        Fr::from(index as u64),
        "Mismatch in first column at row={row}"
      );
    }

    // Verify that other rows have been modified using the twiddle factors
    for row in 1..rows {
      let mut idx = row * 2;
      for col in 1..cols {
        let index = row * cols + col;
        let expected = Fr::from(index as u64) * roots[idx];
        assert_eq!(values[index], expected, "Mismatch at row={row}, col={col}");
        idx += 2 * row;
      }
    }
  }

  #[test]
  fn test_new_from_cache_singleton() {
    // Retrieve two instances of the engine
    let engine1 = NttEngine::<Fr>::new_from_cache();
    let engine2 = NttEngine::<Fr>::new_from_cache();

    // Both instances should point to the same object in memory
    assert!(Arc::ptr_eq(&engine1, &engine2));

    // Verify that the cached instance has the expected properties
    assert!(engine1.order.is_power_of_two());

    let expected_root = Fr::ROOT_OF_UNITY;
    assert_eq!(engine1.root(engine1.order), expected_root);
  }

  #[test]
  fn test_ntt_batch_size_2() {
    let engine = NttEngine::<Fr>::new_from_PrimeField();

    // Input values: f(x) = [1, 2]
    let f0 = Fr::from(1);
    let f1 = Fr::from(2);
    let mut values = vec![f0, f1];

    // Compute the expected NTT manually:
    //
    //   F(0)  =  f0 + f1
    //   F(1)  =  f0 - f1
    //
    // ω is the 2nd root of unity: ω² = 1.

    let expected_f0 = f0 + f1;
    let expected_f1 = f0 - f1;

    let expected_values = vec![expected_f0, expected_f1];

    engine.ntt_batch(&mut values, 2);

    assert_eq!(values, expected_values);
  }

  #[test]
  fn test_ntt_batch_size_4() {
    let engine = NttEngine::<Fr>::new_from_PrimeField();

    // Input values: f(x) = [1, 2, 3, 4]
    let f0 = Fr::from(1);
    let f1 = Fr::from(2);
    let f2 = Fr::from(3);
    let f3 = Fr::from(4);
    let mut values = vec![f0, f1, f2, f3];

    // Compute the expected NTT manually:
    //
    //   F(0)  =  f0 + f1 + f2 + f3
    //   F(1)  =  f0 + f1 * ω + f2 * ω² + f3 * ω³
    //   F(2)  =  f0 + f1 * ω² + f2 * ω⁴ + f3 * ω⁶
    //   F(3)  =  f0 + f1 * ω³ + f2 * ω⁶ + f3 * ω⁹
    //
    // ω is the 4th root of unity: ω⁴ = 1, ω² = -1.

    let omega = engine.omega_4_1;
    let omega1 = omega; // ω
    let omega2 = omega * omega; // ω² = -1
    let omega3 = omega * omega2; // ω³ = -ω
    let omega4 = omega * omega3; // ω⁴ = 1

    let expected_f0 = f0 + f1 + f2 + f3;
    let expected_f1 = f0 + f1 * omega1 + f2 * omega2 + f3 * omega3;
    let expected_f2 = f0 + f1 * omega2 + f2 * omega4 + f3 * omega2;
    let expected_f3 = f0 + f1 * omega3 + f2 * omega2 + f3 * omega1;

    let expected_values = vec![expected_f0, expected_f1, expected_f2, expected_f3];

    engine.ntt_batch(&mut values, 4);

    assert_eq!(values, expected_values);
  }

  #[test]
  fn test_ntt_batch_size_8() {
    let engine = NttEngine::<Fr>::new_from_PrimeField();

    // Input values: f(x) = [1, 2, 3, 4, 5, 6, 7, 8]
    let f0 = Fr::from(1);
    let f1 = Fr::from(2);
    let f2 = Fr::from(3);
    let f3 = Fr::from(4);
    let f4 = Fr::from(5);
    let f5 = Fr::from(6);
    let f6 = Fr::from(7);
    let f7 = Fr::from(8);
    let mut values = vec![f0, f1, f2, f3, f4, f5, f6, f7];

    // Compute the expected NTT manually:
    //
    //   F(k) = ∑ f_j * ω^(j*k)  for k ∈ {0, ..., 7}
    //
    // ω is the 8th root of unity: ω⁸ = 1.

    let omega = engine.omega_8_1; // ω
    let omega1 = omega; // ω
    let omega2 = omega * omega; // ω²
    let omega3 = omega * omega2; // ω³
    let omega4 = omega * omega3; // ω⁴
    let omega5 = omega * omega4; // ω⁵
    let omega6 = omega * omega5; // ω⁶
    let omega7 = omega * omega6; // ω⁷

    let expected_f0 = f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7;
    let expected_f1 = f0
      + f1 * omega1
      + f2 * omega2
      + f3 * omega3
      + f4 * omega4
      + f5 * omega5
      + f6 * omega6
      + f7 * omega7;
    let expected_f2 = f0
      + f1 * omega2
      + f2 * omega4
      + f3 * omega6
      + f4 * Fr::ONE
      + f5 * omega2
      + f6 * omega4
      + f7 * omega6;
    let expected_f3 = f0
      + f1 * omega3
      + f2 * omega6
      + f3 * omega1
      + f4 * omega4
      + f5 * omega7
      + f6 * omega2
      + f7 * omega5;
    let expected_f4 = f0
      + f1 * omega4
      + f2 * Fr::ONE
      + f3 * omega4
      + f4 * Fr::ONE
      + f5 * omega4
      + f6 * Fr::ONE
      + f7 * omega4;
    let expected_f5 = f0
      + f1 * omega5
      + f2 * omega2
      + f3 * omega7
      + f4 * omega4
      + f5 * omega1
      + f6 * omega6
      + f7 * omega3;
    let expected_f6 = f0
      + f1 * omega6
      + f2 * omega4
      + f3 * omega2
      + f4 * Fr::ONE
      + f5 * omega6
      + f6 * omega4
      + f7 * omega2;
    let expected_f7 = f0
      + f1 * omega7
      + f2 * omega6
      + f3 * omega5
      + f4 * omega4
      + f5 * omega3
      + f6 * omega2
      + f7 * omega1;

    let expected_values = vec![
      expected_f0,
      expected_f1,
      expected_f2,
      expected_f3,
      expected_f4,
      expected_f5,
      expected_f6,
      expected_f7,
    ];

    engine.ntt_batch(&mut values, 8);

    assert_eq!(values, expected_values);
  }

  #[test]
  fn test_ntt_batch_size_16() {
    let engine = NttEngine::<Fr>::new_from_PrimeField();

    // Input values: f(x) = [1, 2, ..., 16]
    let values: Vec<_> = (1..=16).map(Fr::from).collect();
    let mut values_ntt = values.clone();

    // Compute the expected NTT manually:
    //
    //   F(k) = ∑ f_j * ω^(j*k)  for k ∈ {0, ..., 15}
    //
    // ω is the 16th root of unity: ω¹⁶ = 1.

    let omega = engine.omega_16_1; // ω
    let mut expected_values = vec![Fr::ZERO; 16];
    for (k, expected_value) in expected_values.iter_mut().enumerate().take(16) {
      let omega_k = omega.pow([k as u64]);
      *expected_value = values
        .iter()
        .enumerate()
        .map(|(j, &f_j)| f_j * omega_k.pow([j as u64]))
        .sum();
    }

    engine.ntt_batch(&mut values_ntt, 16);

    assert_eq!(values_ntt, expected_values);
  }

  #[test]
  fn test_ntt_batch_size_32() {
    let engine = NttEngine::<Fr>::new_from_PrimeField();

    // Input values: f(x) = [1, 2, ..., 32]
    let values: Vec<_> = (1..=32).map(Fr::from).collect();
    let mut values_ntt = values.clone();

    // Compute the expected NTT manually:
    //
    //   F(k) = ∑ f_j * ω^(j*k)  for k ∈ {0, ..., 31}
    //
    // ω is the 32nd root of unity: ω³² = 1.

    let omega = engine.root(32);
    let mut expected_values = vec![Fr::ZERO; 32];
    for (k, expected_value) in expected_values.iter_mut().enumerate().take(32) {
      let omega_k = omega.pow([k as u64]);
      *expected_value = values
        .iter()
        .enumerate()
        .map(|(j, &f_j)| f_j * omega_k.pow([j as u64]))
        .sum();
    }

    engine.ntt_batch(&mut values_ntt, 32);

    assert_eq!(values_ntt, expected_values);
  }

  /// Computes the largest factor of `x` that is ≤ sqrt(x).
  /// If `x` is 0, returns 0.
  fn get_largest_divisor_up_to_sqrt(x: usize) -> usize {
    if x == 0 {
      return 0;
    }

    let mut result = 1;

    // Compute integer square root of `x` using floating point arithmetic.
    #[allow(clippy::cast_sign_loss)]
    let isqrt_x = (x as f64).sqrt() as usize;

    // Iterate from 1 to `isqrt_x` to find the largest factor of `x`.
    for i in 1..=isqrt_x {
      if x.is_multiple_of(i) {
        // Update `result` with the largest divisor found.
        result = i;
      }
    }

    result
  }

  #[test]
  fn test_gcd() {
    assert_eq!(gcd(4, 6), 2);
    assert_eq!(gcd(0, 4), 4);
    assert_eq!(gcd(4, 0), 4);
    assert_eq!(gcd(1, 1), 1);
    assert_eq!(gcd(64, 16), 16);
    assert_eq!(gcd(81, 9), 9);
    assert_eq!(gcd(0, 0), 0);
  }

  #[test]
  fn test_lcm() {
    assert_eq!(lcm(5, 6), 30);
    assert_eq!(lcm(3, 7), 21);
    assert_eq!(lcm(0, 10), 0);
  }

  #[test]
  fn test_sqrt_factor() {
    // Cases where n = 2^k * 1
    assert_eq!(sqrt_factor(1), 1); // 1 = 2^0 * 1
    assert_eq!(sqrt_factor(4), 2); // 4 = 2^2 * 1
    assert_eq!(sqrt_factor(16), 4); // 16 = 2^4 * 1
    assert_eq!(sqrt_factor(32), 4); // 32 = 2^5 * 1
    assert_eq!(sqrt_factor(64), 8); // 64 = 2^6 * 1
    assert_eq!(sqrt_factor(256), 16); // 256 = 2^8 * 1

    // Cases where n = 2^k * 3
    assert_eq!(sqrt_factor(3), 1); // 3 = 2^0 * 3
    assert_eq!(sqrt_factor(12), 3); // 12 = 2^2 * 3
    assert_eq!(sqrt_factor(48), 6); // 48 = 2^4 * 3
    assert_eq!(sqrt_factor(192), 12); // 192 = 2^6 * 3
    assert_eq!(sqrt_factor(768), 24); // 768 = 2^8 * 3

    // Cases where n = 2^k * 9
    assert_eq!(sqrt_factor(9), 3); // 9 = 2^0 * 9
    assert_eq!(sqrt_factor(36), 6); // 36 = 2^2 * 9
    assert_eq!(sqrt_factor(144), 12); // 144 = 2^4 * 9
    assert_eq!(sqrt_factor(576), 24); // 576 = 2^6 * 9
    assert_eq!(sqrt_factor(2304), 48); // 2304 = 2^8 * 9
  }

  #[test]
  fn test_divisors() {
    assert_eq!(divisors(24, &[2, 3]), vec![1, 2, 3, 4, 6, 8, 12, 24]);
    assert_eq!(
      divisors(210, &[2, 3, 5, 7]),
      vec![1, 2, 3, 5, 6, 7, 10, 14, 15, 21, 30, 35, 42, 70, 105, 210]
    );
    assert_eq!(
      divisors(8 * 9 * 25, &[2, 3, 5]),
      vec![
        1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 18, 20, 24, 25, 30, 36, 40, 45, 50, 60, 72, 75, 90,
        100, 120, 150, 180, 200, 225, 300, 360, 450, 600, 900, 1800
      ]
    );
  }

  proptest! {
      #[test]
      fn proptest_sqrt_factor(k in 0usize..30, base in prop_oneof![Just(1), Just(3), Just(9)])
      {
          let n = (1 << k) * base;
          let expected = get_largest_divisor_up_to_sqrt(n);
          prop_assert_eq!(sqrt_factor(n), expected);
      }
  }

  // --- Reed-Solomon tests ---

  use proptest::{collection, prelude::Just, sample::select, strategy::Strategy};
  use rand::{SeedableRng, rngs::StdRng};
  use std::iter;

  /// Horner's method: evaluate polynomial `coeffs` at point `x`.
  fn univariate_evaluate<F: Field>(coeffs: &[F], x: F) -> F {
    coeffs.iter().rev().fold(F::ZERO, |acc, &c| acc * x + c)
  }

  /// Returns up to `count` NTT-smooth codeword lengths >= `size` for field `F`.
  fn valid_codeword_lengths<F: PrimeField>(
    engine: &NttEngine<F>,
    size: usize,
    count: usize,
  ) -> Vec<usize> {
    iter::successors(engine.next_order(size), |&s| engine.next_order(s + 1))
      .take(count)
      .collect()
  }

  fn test_rs<F: PrimeField>(engine: &NttEngine<F>) {
    let cases = (
      0_usize..10,
      0_usize..(1 << 8), // smaller than WHIR to keep test fast
      0_usize..(1 << 8),
      1_usize..=16,
    )
      .prop_flat_map(|(num_messages, message_length, mask_length, sample_size)| {
        let codeword_lengths = valid_codeword_lengths::<F>(
          &NttEngine::<F>::new_from_PrimeField(),
          message_length + mask_length,
          4,
        );
        if codeword_lengths.is_empty() {
          return Just((num_messages, message_length, mask_length, 0, vec![])).boxed();
        }
        select(codeword_lengths)
          .prop_flat_map(move |codeword_length| {
            let sample_size = sample_size.min(codeword_length.max(1));
            (
              Just(num_messages),
              Just(message_length),
              Just(mask_length),
              Just(codeword_length),
              collection::vec(0..codeword_length.max(1), sample_size),
            )
          })
          .boxed()
      });

    proptest!(|(
      seed: u64,
      (num_messages, message_length, mask_length, codeword_length, sampled_indices) in cases
    )| {
      if codeword_length == 0 { return Ok(()); }
      let mut rng = StdRng::seed_from_u64(seed);
      let messages: Vec<Vec<F>> = (0..num_messages)
        .map(|_| (0..message_length).map(|_| F::random(&mut rng)).collect())
        .collect();
      let masks: Vec<F> = (0..mask_length * num_messages).map(|_| F::random(&mut rng)).collect();
      let message_refs: Vec<&[F]> = messages.iter().map(|v| v.as_slice()).collect();

      let codeword = engine.interleaved_encode(&message_refs, &masks, codeword_length);

      // Output size check.
      assert_eq!(codeword.len(), codeword_length * num_messages);

      // Each codeword value == polynomial evaluation at the corresponding point.
      let evaluation_points = engine.evaluation_points(
        message_length + mask_length, codeword_length, &sampled_indices,
      );
      for (&index, &eval_point) in sampled_indices.iter().zip(evaluation_points.iter()) {
        let evaluations = &codeword[index * num_messages..(index + 1) * num_messages];
        let mask_chunks: Vec<&[F]> = if mask_length > 0 {
          masks.chunks_exact(mask_length).collect()
        } else {
          vec![&[]; num_messages]
        };
        for ((message, mask), &value) in messages.iter().zip(mask_chunks.iter()).zip(evaluations.iter()) {
          let expected = univariate_evaluate(message, eval_point)
            + eval_point.pow([message_length as u64]) * univariate_evaluate(mask, eval_point);
          prop_assert_eq!(value, expected);
        }
      }

      // Evaluation points are unique for unique indices.
      let mut deduped_indices = sampled_indices.clone();
      deduped_indices.sort_unstable();
      deduped_indices.dedup();
      let mut deduped_points: Vec<F> = engine
        .evaluation_points(message_length + mask_length, codeword_length, &deduped_indices);
      deduped_points.sort_unstable_by(|a, b| a.to_repr().as_ref().cmp(b.to_repr().as_ref()));
      deduped_points.dedup();
      prop_assert_eq!(deduped_indices.len(), deduped_points.len());
    });
  }

  #[test]
  fn test_rs_bn254_fr() {
    let engine = NttEngine::<Fr>::new_from_PrimeField();
    test_rs(&engine);
  }
}
