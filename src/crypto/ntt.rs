use ark_ff::{FftField, Field};
use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::sync::{Arc, LazyLock, Mutex, RwLock, RwLockReadGuard};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Global cache for NTT engines, indexed by field.
// TODO: Skip `LazyLock` when `HashMap::with_hasher` becomes const.
// see https://github.com/rust-lang/rust/issues/102575
static ENGINE_CACHE: LazyLock<Mutex<HashMap<TypeId, Arc<dyn Any + Send + Sync>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Enginge for computing NTTs over arbitrary fields.
/// Assumes the field has large two-adicity.
pub struct NttEngine<F: Field> {
    order: usize,
    omega_order: F,

    // Small roots (Zero if unavailable)
    half_omega_3_1_plus_2: F, // ½(ω₃ + ω₃²)
    half_omega_3_1_min_2: F,  // ½(ω₃ - ω₃²)
    omega_4_1: F,
    omega_8_1: F,
    omega_8_3: F,

    // Root lookup table (extended on demand)
    roots: RwLock<Vec<F>>,
}

/// Compute the NTT of a slice of field elements using a cached engine.
pub fn ntt<F: FftField>(values: &mut [F]) {
    NttEngine::new_from_cache().ntt(values);
}

/// Compute the many NTTs of size `size` using a cached engine.
pub fn ntt_batch<F: FftField>(values: &mut [F], size: usize) {
    NttEngine::new_from_cache().ntt_batch(values, size);
}

pub fn intt<F: FftField>(values: &mut [F]) {
    NttEngine::new_from_cache().intt(values);
}

pub fn intt_batch<F: FftField>(values: &mut [F], size: usize) {
    NttEngine::new_from_cache().intt_batch(values, size);
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

impl<F: Field> NttEngine<F> {
    pub fn new(order: usize, omega_order: F) -> Self {
        assert!(order.trailing_zeros() > 0, "Order must be a power of 2.");
        // TODO: Assert that omega_order factors into 2s and 3s.
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
            roots: RwLock::new(Vec::new()),
        };
        if order % 3 == 0 {
            let omega_3_1 = res.root(3);
            let omega_3_2 = omega_3_1 * omega_3_1;
            res.half_omega_3_1_min_2 = (omega_3_1 - omega_3_2) / F::from(2u64);
            res.half_omega_3_1_plus_2 = (omega_3_1 + omega_3_2) / F::from(2u64);
        }
        if order % 4 == 0 {
            res.omega_4_1 = res.root(4);
        }
        if order % 8 == 0 {
            res.omega_8_1 = res.root(8);
            res.omega_8_3 = res.omega_8_1.pow(&[3]);
        }
        res
    }

    pub fn ntt(&self, values: &mut [F]) {
        self.ntt_batch(values, values.len())
    }

    #[cfg(not(feature = "parallel"))]
    pub fn ntt_batch(&self, values: &mut [F], size: usize) {
        assert!(values.len() % size == 0);
        let roots = self.roots_table(size);
        self.ntt_dispatch(values, &roots, size);
    }

    #[cfg(feature = "parallel")]
    pub fn ntt_batch(&self, values: &mut [F], size: usize) {
        const PARALLEL_THRESHOLD: usize = 1 << 14;
        assert!(values.len() % size == 0);
        let roots = self.roots_table(size);
        if values.len() < PARALLEL_THRESHOLD {
            self.ntt_dispatch(values, &roots, size);
        } else {
            // Work size is next multiple of `size` larger than `PARALLEL_THRESHOLD`.
            let chunk_size = ((PARALLEL_THRESHOLD + size - 1) / size) * size;
            values.par_chunks_exact_mut(chunk_size).for_each(|values| {
                self.ntt_dispatch(values, &roots, size);
            });
        }
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
                let size = if roots.is_empty() {
                    order
                } else {
                    lcm(roots.len(), order)
                };
                roots.clear();
                roots.reserve_exact(size);

                // Compute powers of roots of unity.
                let root = self.root(size);
                let mut root_i = F::ONE;
                while roots.len() < size {
                    roots.push(root_i);
                    root_i *= root;
                }
            }
            // Back to read lock.
            drop(roots);
            self.roots.read().unwrap()
        } else {
            roots
        }
    }

    /// Compute an NTT in place by splititng into two factors.
    /// Recurses using the sqrt(N) Cooley-Tukey Six step NTT algorithm.
    fn ntt_recurse(&self, values: &mut [F], roots: &[F], size: usize) {
        let n1 = sqrt_factor(size);
        let n2 = size / n1;
        let step = roots.len() / size;

        // TODO: Lift recursion out of loop.
        for values in values.chunks_exact_mut(size) {
            transpose(values, n1, n2);
        }
        self.ntt_dispatch(values, roots, n1);
        for values in values.chunks_exact_mut(size) {
            transpose(values, n2, n1);

            // TODO: When (n1, n2) are coprime we can use the
            // Good-Thomas NTT algorithm and avoid the twiddle loop.
            for i in 1..n1 {
                let step = (i * step) % roots.len();
                let mut index = step;
                for j in 1..n2 {
                    index %= roots.len();
                    values[i * n2 + j] *= roots[index];
                    index += step;
                }
            }
        }
        self.ntt_dispatch(values, roots, n2);
        for values in values.chunks_exact_mut(size) {
            transpose(values, n1, n2);
        }
    }

    fn ntt_dispatch(&self, values: &mut [F], roots: &[F], size: usize) {
        debug_assert_eq!(values.len() % size, 0);
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
            } // TODO: 16
            size => self.ntt_recurse(values, roots, size),
        }
    }
}

/// Transpose a matrix in-place.
pub fn transpose<T: Copy>(matrix: &mut [T], rows: usize, cols: usize) {
    debug_assert_eq!(matrix.len(), rows * cols);
    if rows == cols {
        for i in 0..rows {
            for j in (i + 1)..cols {
                matrix.swap(i * cols + j, j * rows + i);
            }
        }
    } else {
        let copy = matrix.to_vec();
        for i in 0..rows {
            for j in 0..cols {
                matrix[j * rows + i] = copy[i * cols + j];
            }
        }
    }
}

/// Compute the largest factor of n that is <= sqrt(n).
/// Assumes n is of the form 2^k * {1,3,9}.
fn sqrt_factor(n: usize) -> usize {
    let twos = n.trailing_zeros();
    match n >> twos {
        1 => 1 << (twos / 2),
        3 | 9 => 3 << (twos / 2),
        _ => panic!(),
    }
}

/// Least common multiple.
fn lcm(a: usize, b: usize) -> usize {
    a * b / gcd(a, b)
}

// Greatest common divisor.
fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}
