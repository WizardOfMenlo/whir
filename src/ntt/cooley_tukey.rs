//! Number-theoretic transforms (NTTs) over fields with high two-adicity.
//!
//! Implements the √N Cooley-Tukey six-step algorithm to achieve parallelism with good locality.
//! A global cache is used for twiddle factors.

use super::{
    transpose,
    utils::{lcm, sqrt_factor},
};
use ark_ff::{FftField, Field};
use std::{
    any::{Any, TypeId},
    collections::HashMap,
    sync::{Arc, LazyLock, Mutex, RwLock, RwLockReadGuard},
};

#[cfg(feature = "parallel")]
use {super::utils::workload_size, rayon::prelude::*, std::cmp::max};

/// Global cache for NTT engines, indexed by field.
// TODO: Skip `LazyLock` when `HashMap::with_hasher` becomes const.
// see https://github.com/rust-lang/rust/issues/102575
static ENGINE_CACHE: LazyLock<Mutex<HashMap<TypeId, Arc<dyn Any + Send + Sync>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Enginge for computing NTTs over arbitrary fields.
/// Assumes the field has large two-adicity.
pub struct NttEngine<F: Field> {
    order: usize,   // order of omega_orger
    omega_order: F, // primitive order'th root.

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

/// Compute the NTT of a slice of field elements using a cached engine.
pub fn ntt<F: FftField>(values: &mut [F]) {
    NttEngine::<F>::new_from_cache().ntt(values);
}

/// Compute the many NTTs of size `size` using a cached engine.
pub fn ntt_batch<F: FftField>(values: &mut [F], size: usize) {
    NttEngine::<F>::new_from_cache().ntt_batch(values, size);
}

/// Compute the inverse NTT of a slice of field element without the 1/n scaling factor, using a cached engine.
pub fn intt<F: FftField>(values: &mut [F]) {
    NttEngine::<F>::new_from_cache().intt(values);
}

/// Compute the inverse NTT of multiple slice of field elements, each of size `size`, without the 1/n scaling factor and using a cached engine.
pub fn intt_batch<F: FftField>(values: &mut [F], size: usize) {
    NttEngine::<F>::new_from_cache().intt_batch(values, size);
}

impl<F: FftField> NttEngine<F> {
    /// Get or create a cached engine for the field `F`.
    pub fn new_from_cache() -> Arc<Self> {
        let mut cache = ENGINE_CACHE.lock().unwrap();
        let type_id = TypeId::of::<F>();
        #[allow(clippy::option_if_let_else)]
        if let Some(engine) = cache.get(&type_id) {
            engine.clone().downcast::<Self>().unwrap()
        } else {
            let engine = Arc::new(Self::new_from_fftfield());
            cache.insert(type_id, engine.clone());
            engine
        }
    }

    /// Construct a new engine from the field's `FftField` trait.
    pub(crate) fn new_from_fftfield() -> Self {
        // TODO: Support SMALL_SUBGROUP
        if F::TWO_ADICITY <= 63 {
            Self::new(1 << F::TWO_ADICITY, F::TWO_ADIC_ROOT_OF_UNITY)
        } else {
            let mut generator = F::TWO_ADIC_ROOT_OF_UNITY;
            for _ in 0..(F::TWO_ADICITY - 63) {
                generator = generator.square();
            }
            Self::new(1 << 63, generator)
        }
    }
}

/// Creates a new NttEngine. `omega_order` must be a primitive root of unity of even order `omega`.
impl<F: Field> NttEngine<F> {
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
        };
        if order % 3 == 0 {
            let omega_3_1 = res.root(3);
            let omega_3_2 = omega_3_1 * omega_3_1;
            // Note: char F cannot be 2 and so division by 2 works, because primitive roots of unity with even order exist.
            res.half_omega_3_1_min_2 = (omega_3_1 - omega_3_2) / F::from(2u64);
            res.half_omega_3_1_plus_2 = (omega_3_1 + omega_3_2) / F::from(2u64);
        }
        if order % 4 == 0 {
            res.omega_4_1 = res.root(4);
        }
        if order % 8 == 0 {
            res.omega_8_1 = res.root(8);
            res.omega_8_3 = res.omega_8_1.pow([3]);
        }
        if order % 16 == 0 {
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
        assert!(values.len() % size == 0);
        let roots = self.roots_table(size);
        self.ntt_dispatch(values, &roots, size);
    }

    /// Inverse NTT. Does not aply 1/n scaling factor.
    pub fn intt(&self, values: &mut [F]) {
        values[1..].reverse();
        self.ntt(values);
    }

    /// Inverse batch NTT. Does not aply 1/n scaling factor.
    pub fn intt_batch(&self, values: &mut [F], size: usize) {
        assert!(values.len() % size == 0);

        #[cfg(not(feature = "parallel"))]
        values.chunks_exact_mut(size).for_each(|values| {
            values[1..].reverse();
        });

        #[cfg(feature = "parallel")]
        values.par_chunks_exact_mut(size).for_each(|values| {
            values[1..].reverse();
        });

        self.ntt_batch(values, size);
    }

    pub fn root(&self, order: usize) -> F {
        assert!(
            self.order % order == 0,
            "Subgroup of requested order does not exist."
        );
        self.omega_order.pow([(self.order / order) as u64])
    }

    /// Returns a cached table of roots of unity of the given order.
    fn roots_table(&self, order: usize) -> RwLockReadGuard<Vec<F>> {
        // Precompute more roots of unity if requested.
        let roots = self.roots.read().unwrap();
        if roots.is_empty() || roots.len() % order != 0 {
            // Obtain write lock to update the cache.
            drop(roots);
            let mut roots = self.roots.write().unwrap();
            // Race condition: check if another thread updated the cache.
            if roots.is_empty() || roots.len() % order != 0 {
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
                #[cfg(not(feature = "parallel"))]
                {
                    let mut root_i = F::ONE;
                    for _ in 0..size {
                        roots.push(root_i);
                        root_i *= root;
                    }
                }
                #[cfg(feature = "parallel")]
                roots.par_extend((0..size).into_par_iter().map_with(F::ZERO, |root_i, i| {
                    if root_i.is_zero() {
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
        #[cfg(feature = "parallel")]
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

#[cfg(not(feature = "parallel"))]
fn apply_twiddles<F: Field>(values: &mut [F], roots: &[F], rows: usize, cols: usize) {
    debug_assert_eq!(values.len() % (rows * cols), 0);
    let step = roots.len() / (rows * cols);
    for values in values.chunks_exact_mut(rows * cols) {
        for (i, row) in values.chunks_exact_mut(cols).enumerate().skip(1) {
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

#[cfg(feature = "parallel")]
fn apply_twiddles<F: Field>(values: &mut [F], roots: &[F], rows: usize, cols: usize) {
    let size = rows * cols;
    debug_assert_eq!(values.len() % size, 0);
    let step = roots.len() / size;

    if values.len() > workload_size::<F>() {
        if values.len() == size {
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
        } else {
            let workload_size = size * max(1, workload_size::<F>() / size);
            values.par_chunks_mut(workload_size).for_each(|values| {
                apply_twiddles(values, roots, rows, cols);
            });
        }
    } else {
        for values in values.chunks_exact_mut(size) {
            for (i, row) in values.chunks_exact_mut(cols).enumerate().skip(1) {
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
}

#[cfg(test)]
#[allow(clippy::significant_drop_tightening)]
mod tests {
    use super::*;
    use crate::crypto::fields::Field64;
    use ark_ff::AdditiveGroup;
    use ark_ff::{BigInteger, PrimeField};

    #[test]
    fn test_new_from_fftfield_basic() {
        // Ensure that an engine is created correctly from FFT field properties
        let engine = NttEngine::<Field64>::new_from_fftfield();

        // Verify that the order of the engine is correctly set
        assert!(engine.order.is_power_of_two());

        // Verify that the root of unity is correctly initialized
        let expected_root = Field64::TWO_ADIC_ROOT_OF_UNITY;
        let computed_root = engine.root(engine.order);
        assert_eq!(computed_root.pow([engine.order as u64]), Field64::ONE);
        assert_eq!(computed_root, expected_root);
    }

    #[test]
    fn test_root_computation() {
        let engine = NttEngine::<Field64>::new_from_fftfield();

        // Ensure the root exponentiates correctly
        assert_eq!(engine.root(8).pow([8]), Field64::ONE);
        assert_eq!(engine.root(4).pow([4]), Field64::ONE);
        assert_eq!(engine.root(2).pow([2]), Field64::ONE);

        // Ensure it's not a lower-order root
        assert_ne!(engine.root(8).pow([4]), Field64::ONE);
        assert_ne!(engine.root(4).pow([2]), Field64::ONE);
    }

    #[test]
    fn test_root_of_unity_multiplication() {
        let engine = NttEngine::<Field64>::new_from_fftfield();

        let root = engine.root(16);

        // Multiply root by itself repeatedly and verify expected outcomes
        assert_eq!(root.pow([2]), engine.root(8));
        assert_eq!(root.pow([4]), engine.root(4));
        assert_eq!(root.pow([8]), engine.root(2));
    }

    #[test]
    fn test_root_of_unity_inversion() {
        let engine = NttEngine::<Field64>::new_from_fftfield();
        let root = engine.root(16);

        // The inverse of ω is ω^{-1}, computed as ω^(p-2) in Field64.
        let p: u64 = u64::from_be_bytes(Field64::MODULUS.to_bytes_be().try_into().unwrap());
        let inverse_root = root.pow([p - 2]);
        assert_eq!(root * inverse_root, Field64::ONE);
    }

    #[test]
    fn test_precomputed_small_roots() {
        let engine = NttEngine::<Field64>::new_from_fftfield();

        // Check that precomputed values are correctly initialized
        assert_eq!(engine.omega_8_1.pow([8]), Field64::ONE);
        assert_eq!(engine.omega_8_3.pow([8]), Field64::ONE);
        assert_eq!(engine.omega_16_1.pow([16]), Field64::ONE);
        assert_eq!(engine.omega_16_3.pow([16]), Field64::ONE);
        assert_eq!(engine.omega_16_9.pow([16]), Field64::ONE);
    }

    #[test]
    fn test_consistency_across_multiple_instances() {
        let engine1 = NttEngine::<Field64>::new_from_fftfield();
        let engine2 = NttEngine::<Field64>::new_from_fftfield();

        // Ensure that multiple instances yield the same results
        assert_eq!(engine1.root(8), engine2.root(8));
        assert_eq!(engine1.root(4), engine2.root(4));
        assert_eq!(engine1.root(2), engine2.root(2));
    }

    #[test]
    fn test_roots_table_basic() {
        let engine = NttEngine::<Field64>::new_from_fftfield();
        let roots_4 = engine.roots_table(4);

        // Check hardcoded expected values (ω^i)
        assert_eq!(roots_4[0], Field::ONE);
        assert_eq!(roots_4[1], engine.root(4));
        assert_eq!(roots_4[2], engine.root(4).pow([2]));
        assert_eq!(roots_4[3], engine.root(4).pow([3]));
    }

    #[test]
    fn test_roots_table_minimal_order() {
        let engine = NttEngine::<Field64>::new_from_fftfield();

        let roots_2 = engine.roots_table(2);

        // Must contain only ω^0 and ω^1
        assert_eq!(roots_2.len(), 2);
        assert_eq!(roots_2[0], Field64::ONE);
        assert_eq!(roots_2[1], engine.root(2));
    }

    #[test]
    fn test_roots_table_progression() {
        let engine = NttEngine::<Field64>::new_from_fftfield();

        let roots_4 = engine.roots_table(4);

        // Ensure the sequence follows expected powers of the root of unity
        for i in 0..4 {
            assert_eq!(roots_4[i], engine.root(4).pow([i as u64]));
        }
    }

    #[test]
    fn test_roots_table_cached_results() {
        let engine = NttEngine::<Field64>::new_from_fftfield();

        let first_access = engine.roots_table(4);
        let second_access = engine.roots_table(4);

        // The memory location should be the same, meaning it's cached
        assert!(std::ptr::eq(first_access.as_ptr(), second_access.as_ptr()));
    }

    #[test]
    fn test_roots_table_recompute_factor_order() {
        let engine = NttEngine::<Field64>::new_from_fftfield();

        let roots_4 = engine.roots_table(4);
        let roots_2 = engine.roots_table(2);

        // Ensure first two elements of roots_4 match the first two elements of roots_2
        assert_eq!(&roots_4[..2], &roots_2[..2]);
    }

    #[test]
    fn test_apply_twiddles_basic() {
        let engine = NttEngine::<Field64>::new_from_fftfield();

        let mut values = vec![
            Field64::from(1),
            Field64::from(2),
            Field64::from(3),
            Field64::from(4),
            Field64::from(5),
            Field64::from(6),
            Field64::from(7),
            Field64::from(8),
        ];

        // Mock roots
        let r1 = Field64::from(33);
        let roots = vec![r1];

        // Ensure the root of unity is correct
        assert_eq!(engine.root(4).pow([4]), Field64::ONE);

        apply_twiddles(&mut values, &roots, 2, 4);

        // The first row should remain unchanged
        assert_eq!(values[0], Field64::from(1));
        assert_eq!(values[1], Field64::from(2));
        assert_eq!(values[2], Field64::from(3));
        assert_eq!(values[3], Field64::from(4));

        // The second row should be multiplied by the correct twiddle factors
        assert_eq!(values[4], Field64::from(5)); // No change for first column
        assert_eq!(values[5], Field64::from(6) * r1);
        assert_eq!(values[6], Field64::from(7) * r1);
        assert_eq!(values[7], Field64::from(8) * r1);
    }

    #[test]
    fn test_apply_twiddles_single_row() {
        let mut values = vec![Field64::from(1), Field64::from(2)];

        // Mock roots
        let r1 = Field64::from(12);
        let roots = vec![r1];

        apply_twiddles(&mut values, &roots, 1, 2);

        // Everything should remain unchanged
        assert_eq!(values[0], Field64::from(1));
        assert_eq!(values[1], Field64::from(2));
    }

    #[test]
    fn test_apply_twiddles_varying_rows() {
        let mut values = vec![
            Field64::from(1),
            Field64::from(2),
            Field64::from(3),
            Field64::from(4),
            Field64::from(5),
            Field64::from(6),
            Field64::from(7),
            Field64::from(8),
            Field64::from(9),
        ];

        // Mock roots
        let roots = (2..100).map(Field64::from).collect::<Vec<_>>();

        apply_twiddles(&mut values, &roots, 3, 3);

        // First row remains unchanged
        assert_eq!(values[0], Field64::from(1));
        assert_eq!(values[1], Field64::from(2));
        assert_eq!(values[2], Field64::from(3));

        // Second row multiplied by twiddle factors
        assert_eq!(values[3], Field64::from(4));
        assert_eq!(values[4], Field64::from(5) * roots[10]);
        assert_eq!(values[5], Field64::from(6) * roots[20]);

        // Third row multiplied by twiddle factors
        assert_eq!(values[6], Field64::from(7));
        assert_eq!(values[7], Field64::from(8) * roots[20]);
        assert_eq!(values[8], Field64::from(9) * roots[40]);
    }

    #[test]
    fn test_apply_twiddles_large_table() {
        let rows = 320;
        let cols = 320;
        let size = rows * cols;

        let mut values: Vec<Field64> = (0..size as u64).map(Field64::from).collect();

        // Generate a large set of twiddle factors
        let roots: Vec<Field64> = (0..(size * 2) as u64).map(Field64::from).collect();

        apply_twiddles(&mut values, &roots, rows, cols);

        // Verify the first row remains unchanged
        for (i, &col) in values.iter().enumerate().take(cols) {
            assert_eq!(col, Field64::from(i as u64));
        }

        // Verify the first column remains unchanged
        for row in 1..rows {
            let index = row * cols;
            assert_eq!(
                values[index],
                Field64::from(index as u64),
                "Mismatch in first column at row={row}"
            );
        }

        // Verify that other rows have been modified using the twiddle factors
        for row in 1..rows {
            let mut idx = row * 2;
            for col in 1..cols {
                let index = row * cols + col;
                let expected = Field64::from(index as u64) * roots[idx];
                assert_eq!(values[index], expected, "Mismatch at row={row}, col={col}");
                idx += 2 * row;
            }
        }
    }

    #[test]
    fn test_new_from_cache_singleton() {
        // Retrieve two instances of the engine
        let engine1 = NttEngine::<Field64>::new_from_cache();
        let engine2 = NttEngine::<Field64>::new_from_cache();

        // Both instances should point to the same object in memory
        assert!(Arc::ptr_eq(&engine1, &engine2));

        // Verify that the cached instance has the expected properties
        assert!(engine1.order.is_power_of_two());

        let expected_root = Field64::TWO_ADIC_ROOT_OF_UNITY;
        assert_eq!(engine1.root(engine1.order), expected_root);
    }

    #[test]
    fn test_ntt_batch_size_2() {
        let engine = NttEngine::<Field64>::new_from_fftfield();

        // Input values: f(x) = [1, 2]
        let f0 = Field64::from(1);
        let f1 = Field64::from(2);
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
        let engine = NttEngine::<Field64>::new_from_fftfield();

        // Input values: f(x) = [1, 2, 3, 4]
        let f0 = Field64::from(1);
        let f1 = Field64::from(2);
        let f2 = Field64::from(3);
        let f3 = Field64::from(4);
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
        let engine = NttEngine::<Field64>::new_from_fftfield();

        // Input values: f(x) = [1, 2, 3, 4, 5, 6, 7, 8]
        let f0 = Field64::from(1);
        let f1 = Field64::from(2);
        let f2 = Field64::from(3);
        let f3 = Field64::from(4);
        let f4 = Field64::from(5);
        let f5 = Field64::from(6);
        let f6 = Field64::from(7);
        let f7 = Field64::from(8);
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
            + f4 * Field64::ONE
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
            + f2 * Field64::ONE
            + f3 * omega4
            + f4 * Field64::ONE
            + f5 * omega4
            + f6 * Field64::ONE
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
            + f4 * Field64::ONE
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
        let engine = NttEngine::<Field64>::new_from_fftfield();

        // Input values: f(x) = [1, 2, ..., 16]
        let values: Vec<_> = (1..=16).map(Field64::from).collect();
        let mut values_ntt = values.clone();

        // Compute the expected NTT manually:
        //
        //   F(k) = ∑ f_j * ω^(j*k)  for k ∈ {0, ..., 15}
        //
        // ω is the 16th root of unity: ω¹⁶ = 1.

        let omega = engine.omega_16_1; // ω
        let mut expected_values = vec![Field64::ZERO; 16];
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
        let engine = NttEngine::<Field64>::new_from_fftfield();

        // Input values: f(x) = [1, 2, ..., 32]
        let values: Vec<_> = (1..=32).map(Field64::from).collect();
        let mut values_ntt = values.clone();

        // Compute the expected NTT manually:
        //
        //   F(k) = ∑ f_j * ω^(j*k)  for k ∈ {0, ..., 31}
        //
        // ω is the 32nd root of unity: ω³² = 1.

        let omega = engine.root(32);
        let mut expected_values = vec![Field64::ZERO; 32];
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
}
