//! Number-theoretic transforms (NTTs) over fields with high two-adicity.
//!
//! Implements the √N Cooley-Tukey six-step algorithm to achieve parallelism with good locality.
//! A global cache is used for twiddle factors.

use super::{
    transpose,
    utils::{lcm, sqrt_factor, workload_size},
};
use ark_ff::{FftField, Field};
use std::{
    any::{Any, TypeId},
    collections::HashMap,
    sync::{Arc, LazyLock, Mutex, RwLock, RwLockReadGuard},
};

#[cfg(feature = "parallel")]
use {rayon::prelude::*, std::cmp::max};

/// Global cache for NTT engines, indexed by field.
// TODO: Skip `LazyLock` when `HashMap::with_hasher` becomes const.
// see https://github.com/rust-lang/rust/issues/102575
static ENGINE_CACHE: LazyLock<Mutex<HashMap<TypeId, Arc<dyn Any + Send + Sync>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Enginge for computing NTTs over arbitrary fields.
/// Assumes the field has large two-adicity.
pub struct NttEngine<F: Field> {
    order: usize, // order of omega_orger
    omega_order: F,  // primitive order'th root.

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
        if let Some(engine) = cache.get(&type_id) {
            engine.clone().downcast::<NttEngine<F>>().unwrap()
        } else {
            let engine = Arc::new(NttEngine::new_from_fftfield());
            cache.insert(type_id, engine.clone());
            engine
        }
    }

    /// Construct a new engine from the field's `FftField` trait.
    fn new_from_fftfield() -> Self {
        // TODO: Support SMALL_SUBGROUP
        if F::TWO_ADICITY <= 63 {
            Self::new(1 << F::TWO_ADICITY, F::TWO_ADIC_ROOT_OF_UNITY)
        } else {
            let mut generator = F::TWO_ADIC_ROOT_OF_UNITY;
            for _ in 0..(63 - F::TWO_ADICITY) {
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
        let mut res = NttEngine {
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
        self.ntt_batch(values, values.len())
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
        self.apply_twiddles(values, roots, n1, n2);
        self.ntt_dispatch(values, roots, n2);
        transpose(values, n1, n2);
    }

    #[cfg(not(feature = "parallel"))]
    fn apply_twiddles(&self, values: &mut [F], roots: &[F], rows: usize, cols: usize) {
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
    fn apply_twiddles(&self, values: &mut [F], roots: &[F], rows: usize, cols: usize) {
        debug_assert_eq!(values.len() % (rows * cols), 0);
        if values.len() > workload_size::<F>() {
            let size = rows * cols;
            if values.len() != size {
                let workload_size = size * max(1, workload_size::<F>() / size);
                values.par_chunks_mut(workload_size).for_each(|values| {
                    self.apply_twiddles(values, roots, rows, cols);
                });
            } else {
                let step = roots.len() / (rows * cols);
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
            }
        } else {
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
