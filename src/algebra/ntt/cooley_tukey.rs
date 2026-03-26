//! Number-theoretic transforms (NTTs) over fields with high two-adicity.
//!
//! Implements the √N Cooley-Tukey six-step algorithm to achieve parallelism with good locality.
//! A global cache is used for twiddle factors.

use std::{
    any::{Any, TypeId},
    collections::HashMap,
    sync::{Arc, LazyLock, Mutex, RwLock, RwLockReadGuard},
};

use ark_ff::{FftField, Field};
#[cfg(feature = "parallel")]
use {crate::utils::workload_size, rayon::prelude::*, std::cmp::max};

use super::{
    transpose,
    utils::{lcm, sqrt_factor},
};

/// Global cache for NTT engines, indexed by field.
// TODO: Skip `LazyLock` when `HashMap::with_hasher` becomes const.
// see https://github.com/rust-lang/rust/issues/102575
static ENGINE_CACHE: LazyLock<Mutex<HashMap<TypeId, Arc<dyn Any + Send + Sync>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Engine for computing NTTs over arbitrary fields.
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

    // Winograd constants for the 13-point DFT:
    // C_m = (ω₁₃^m + ω₁₃^{13-m}) / 2 for m = 1..6
    // S_m = (ω₁₃^m - ω₁₃^{13-m}) / 2 for m = 1..6
    omega_13_c: [F; 6],
    omega_13_s: [F; 6],

    // Root lookup table (extended on demand)
    roots: RwLock<Vec<F>>,
}

/// Returns the root-of-unity used for a domain of size `size`.
pub fn generator<F: FftField>(size: usize) -> Option<F> {
    NttEngine::<F>::new_from_cache().checked_root(size)
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
        if let Some(large_root) = F::LARGE_SUBGROUP_ROOT_OF_UNITY {
            let q = F::SMALL_SUBGROUP_BASE.unwrap() as usize;
            let q_adicity = F::SMALL_SUBGROUP_BASE_ADICITY.unwrap();
            let two_adicity = F::TWO_ADICITY.min(63) as u32;

            let mut order = 1usize << two_adicity;
            for _ in 0..q_adicity {
                order = order
                    .checked_mul(q)
                    .expect("Mixed-radix NTT order overflows usize");
            }

            let mut generator = large_root;
            if F::TWO_ADICITY > 63 {
                for _ in 0..(F::TWO_ADICITY - 63) {
                    generator = generator.square();
                }
            }

            let (order, generator) = try_extend_order_with_primes(order, generator, &[13]);

            Self::new(order, generator)
        } else if F::TWO_ADICITY <= 63 {
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

/// Extend the NTT order by extra small primes dividing |F*| beyond `order`.
fn try_extend_order_with_primes<F: FftField>(
    mut order: usize,
    mut root: F,
    extra_primes: &[usize],
) -> (usize, F) {
    let char_limbs = F::characteristic();
    let mut p_minus_1: Vec<u8> = char_limbs
        .iter()
        .flat_map(|limb| limb.to_le_bytes())
        .collect();
    let mut borrow = 1u16;
    for byte in &mut p_minus_1 {
        let diff = (*byte as u16).wrapping_sub(borrow);
        *byte = diff as u8;
        borrow = if diff > 255 { 1 } else { 0 };
    }
    while p_minus_1.last() == Some(&0) && p_minus_1.len() > 1 {
        p_minus_1.pop();
    }

    for &p in extra_primes {
        let extended_order = match order.checked_mul(p) {
            Some(ext) => ext,
            None => continue,
        };

        let mut cofactor = p_minus_1.clone();
        let mut remaining = extended_order;
        while remaining % 2 == 0 {
            if cofactor[0] & 1 != 0 {
                break;
            }
            let mut carry = 0u8;
            for byte in cofactor.iter_mut().rev() {
                let new_carry = *byte & 1;
                *byte = (*byte >> 1) | (carry << 7);
                carry = new_carry;
            }
            remaining /= 2;
        }
        if remaining != extended_order >> extended_order.trailing_zeros() {
            continue;
        }

        let mut divisible = true;
        if remaining > 1 {
            let divisor = remaining as u128;
            let mut carry: u128 = 0;
            for byte in cofactor.iter_mut().rev() {
                let cur = carry * 256 + *byte as u128;
                *byte = (cur / divisor) as u8;
                carry = cur % divisor;
            }
            if carry != 0 {
                divisible = false;
            }
        }
        if !divisible {
            continue;
        }

        while cofactor.last() == Some(&0) && cofactor.len() > 1 {
            cofactor.pop();
        }

        let cofactor_limbs: Vec<u64> = cofactor
            .chunks(8)
            .map(|chunk: &[u8]| {
                let mut bytes = [0u8; 8];
                bytes[..chunk.len()].copy_from_slice(chunk);
                u64::from_le_bytes(bytes)
            })
            .collect();

        for g in [2u64, 3, 5, 7, 11] {
            let candidate = F::from(g).pow(&cofactor_limbs);
            if candidate.pow([extended_order as u64]) != F::ONE {
                continue;
            }
            if candidate.pow([(extended_order / 2) as u64]) == F::ONE {
                continue;
            }
            if candidate.pow([(extended_order / p) as u64]) == F::ONE {
                continue;
            }
            if extended_order % 3 == 0 && candidate.pow([(extended_order / 3) as u64]) == F::ONE {
                continue;
            }
            order = extended_order;
            root = candidate;
            break;
        }
    }
    (order, root)
}

/// Creates a new NttEngine. `omega_order` must be a primitive root of unity of even order `omega`.
impl<F: Field> NttEngine<F> {
    pub fn new(order: usize, omega_order: F) -> Self {
        assert!(order.trailing_zeros() > 0, "Order must be a multiple of 2.");
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
            omega_13_c: [F::ZERO; 6],
            omega_13_s: [F::ZERO; 6],
            roots: RwLock::new(Vec::new()),
        };
        if order.is_multiple_of(3) {
            let omega_3_1 = res.root(3);
            let omega_3_2 = omega_3_1 * omega_3_1;
            // Note: char F cannot be 2 and so division by 2 works, because primitive roots of unity with even order exist.
            res.half_omega_3_1_min_2 = (omega_3_1 - omega_3_2) / F::from(2u64);
            res.half_omega_3_1_plus_2 = (omega_3_1 + omega_3_2) / F::from(2u64);
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
        if order.is_multiple_of(13) {
            let w = res.root(13);
            let two_inv = F::from(2u64).inverse().unwrap();
            for m in 1..=6 {
                let wm = w.pow([m as u64]);
                let wn = w.pow([(13 - m) as u64]);
                res.omega_13_c[m - 1] = (wm + wn) * two_inv;
                res.omega_13_s[m - 1] = (wm - wn) * two_inv;
            }
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

    pub fn checked_root(&self, order: usize) -> Option<F> {
        self.order
            .is_multiple_of(order)
            .then(|| self.omega_order.pow([(self.order / order) as u64]))
    }

    pub fn root(&self, order: usize) -> F {
        self.checked_root(order)
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
            13 => {
                // Winograd-style 13-point DFT exploiting ω^{13-m} = ω^{-m}.
                // Pairs outputs X[k] and X[13-k] via symmetric/antisymmetric
                // decomposition, reducing from 169 to 72 multiplications.
                let [c1, c2, c3, c4, c5, c6] = self.omega_13_c;
                let [t1, t2, t3, t4, t5, t6] = self.omega_13_s;
                for v in values.chunks_exact_mut(13) {
                    let v0 = v[0];
                    let (s1, d1) = (v[1] + v[12], v[1] - v[12]);
                    let (s2, d2) = (v[2] + v[11], v[2] - v[11]);
                    let (s3, d3) = (v[3] + v[10], v[3] - v[10]);
                    let (s4, d4) = (v[4] + v[9], v[4] - v[9]);
                    let (s5, d5) = (v[5] + v[8], v[5] - v[8]);
                    let (s6, d6) = (v[6] + v[7], v[6] - v[7]);

                    v[0] = v0 + s1 + s2 + s3 + s4 + s5 + s6;

                    // k=1: C-idx [1,2,3,4,5,6], S-signs [+,+,+,+,+,+]
                    let sc = c1 * s1 + c2 * s2 + c3 * s3 + c4 * s4 + c5 * s5 + c6 * s6;
                    let ss = t1 * d1 + t2 * d2 + t3 * d3 + t4 * d4 + t5 * d5 + t6 * d6;
                    (v[1], v[12]) = (v0 + sc + ss, v0 + sc - ss);

                    // k=2: C-idx [2,4,6,5,3,1], S-signs [+,+,+,-,-,-]
                    let sc = c2 * s1 + c4 * s2 + c6 * s3 + c5 * s4 + c3 * s5 + c1 * s6;
                    let ss = t2 * d1 + t4 * d2 + t6 * d3 - t5 * d4 - t3 * d5 - t1 * d6;
                    (v[2], v[11]) = (v0 + sc + ss, v0 + sc - ss);

                    // k=3: C-idx [3,6,4,1,2,5], S-signs [+,+,-,-,+,+]
                    let sc = c3 * s1 + c6 * s2 + c4 * s3 + c1 * s4 + c2 * s5 + c5 * s6;
                    let ss = t3 * d1 + t6 * d2 - t4 * d3 - t1 * d4 + t2 * d5 + t5 * d6;
                    (v[3], v[10]) = (v0 + sc + ss, v0 + sc - ss);

                    // k=4: C-idx [4,5,1,3,6,2], S-signs [+,-,-,+,-,-]
                    let sc = c4 * s1 + c5 * s2 + c1 * s3 + c3 * s4 + c6 * s5 + c2 * s6;
                    let ss = t4 * d1 - t5 * d2 - t1 * d3 + t3 * d4 - t6 * d5 - t2 * d6;
                    (v[4], v[9]) = (v0 + sc + ss, v0 + sc - ss);

                    // k=5: C-idx [5,3,2,6,1,4], S-signs [+,-,+,-,-,+]
                    let sc = c5 * s1 + c3 * s2 + c2 * s3 + c6 * s4 + c1 * s5 + c4 * s6;
                    let ss = t5 * d1 - t3 * d2 + t2 * d3 - t6 * d4 - t1 * d5 + t4 * d6;
                    (v[5], v[8]) = (v0 + sc + ss, v0 + sc - ss);

                    // k=6: C-idx [6,1,5,2,4,3], S-signs [+,-,+,-,+,-]
                    let sc = c6 * s1 + c1 * s2 + c5 * s3 + c2 * s4 + c4 * s5 + c3 * s6;
                    let ss = t6 * d1 - t1 * d2 + t5 * d3 - t2 * d4 + t4 * d5 - t3 * d6;
                    (v[6], v[7]) = (v0 + sc + ss, v0 + sc - ss);
                }
            }
            size => self.ntt_recurse(values, roots, size),
        }
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
pub fn apply_twiddles<F: Field>(values: &mut [F], roots: &[F], rows: usize, cols: usize) {
    let size = rows * cols;
    debug_assert_eq!(values.len() % size, 0);
    let step = roots.len() / size;

    #[cfg(feature = "parallel")]
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

#[cfg(test)]
#[allow(clippy::significant_drop_tightening)]
mod tests {
    use ark_ff::{AdditiveGroup as _, BigInteger, PrimeField};

    use super::*;
    use crate::algebra::fields::{Field256, Field64};

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

    #[test]
    fn test_field256_mixed_radix_engine() {
        // Field256 (BN254 Fr) should have a mixed-radix engine with order 2^28 * 3^2 * 13.
        let engine = NttEngine::<Field256>::new_from_fftfield();
        let expected_order = (1usize << 28) * 9 * 13;
        assert_eq!(engine.order, expected_order);

        // Verify the root of unity has the correct order.
        assert_eq!(
            engine.omega_order.pow([expected_order as u64]),
            Field256::ONE
        );
        assert_ne!(
            engine.omega_order.pow([(expected_order / 2) as u64]),
            Field256::ONE
        );
        assert_ne!(
            engine.omega_order.pow([(expected_order / 3) as u64]),
            Field256::ONE
        );
        assert_ne!(
            engine.omega_order.pow([(expected_order / 13) as u64]),
            Field256::ONE
        );
    }

    #[test]
    fn test_field256_mixed_radix_roots() {
        let engine = NttEngine::<Field256>::new_from_fftfield();

        // Verify roots for mixed-radix sizes.
        assert!(engine.checked_root(3).is_some());
        assert!(engine.checked_root(9).is_some());
        assert!(engine.checked_root(6).is_some()); // 2 * 3
        assert!(engine.checked_root(12).is_some()); // 4 * 3
        assert!(engine.checked_root(18).is_some()); // 2 * 9
        assert!(engine.checked_root(36).is_some()); // 4 * 9

        // Verify root properties.
        let root_3 = engine.root(3);
        assert_eq!(root_3.pow([3]), Field256::ONE);
        assert_ne!(root_3, Field256::ONE);

        let root_9 = engine.root(9);
        assert_eq!(root_9.pow([9]), Field256::ONE);
        assert_ne!(root_9.pow([3]), Field256::ONE);

        let root_36 = engine.root(36);
        assert_eq!(root_36.pow([36]), Field256::ONE);
        assert_ne!(root_36.pow([18]), Field256::ONE);
    }

    #[test]
    fn test_field256_ntt_size_3() {
        let engine = NttEngine::<Field256>::new_from_fftfield();

        // NTT of size 3 over Field256 (BN254 Fr).
        let values: Vec<_> = (1..=3).map(Field256::from).collect();
        let mut values_ntt = values.clone();

        let omega = engine.root(3);
        let mut expected = vec![Field256::ZERO; 3];
        for k in 0..3 {
            let omega_k = omega.pow([k as u64]);
            expected[k] = values
                .iter()
                .enumerate()
                .map(|(j, &v)| v * omega_k.pow([j as u64]))
                .sum();
        }

        engine.ntt_batch(&mut values_ntt, 3);
        assert_eq!(values_ntt, expected);
    }

    #[test]
    fn test_field256_ntt_size_9() {
        let engine = NttEngine::<Field256>::new_from_fftfield();

        // NTT of size 9 over Field256 (BN254 Fr).
        let values: Vec<_> = (1..=9).map(Field256::from).collect();
        let mut values_ntt = values.clone();

        let omega = engine.root(9);
        let mut expected = vec![Field256::ZERO; 9];
        for k in 0..9 {
            let omega_k = omega.pow([k as u64]);
            expected[k] = values
                .iter()
                .enumerate()
                .map(|(j, &v)| v * omega_k.pow([j as u64]))
                .sum();
        }

        engine.ntt_batch(&mut values_ntt, 9);
        assert_eq!(values_ntt, expected);
    }

    #[test]
    fn test_field256_ntt_size_12() {
        // 12 = 4 * 3, a mixed-radix size.
        let engine = NttEngine::<Field256>::new_from_fftfield();

        let values: Vec<_> = (1..=12).map(Field256::from).collect();
        let mut values_ntt = values.clone();

        let omega = engine.root(12);
        let mut expected = vec![Field256::ZERO; 12];
        for k in 0..12 {
            let omega_k = omega.pow([k as u64]);
            expected[k] = values
                .iter()
                .enumerate()
                .map(|(j, &v)| v * omega_k.pow([j as u64]))
                .sum();
        }

        engine.ntt_batch(&mut values_ntt, 12);
        assert_eq!(values_ntt, expected);
    }

    #[test]
    fn test_field256_ntt_size_13() {
        let engine = NttEngine::<Field256>::new_from_fftfield();

        // NTT of size 13 over Field256 (BN254 Fr).
        let values: Vec<_> = (1..=13).map(Field256::from).collect();
        let mut values_ntt = values.clone();

        let omega = engine.root(13);
        let mut expected = vec![Field256::ZERO; 13];
        for k in 0..13 {
            let omega_k = omega.pow([k as u64]);
            expected[k] = values
                .iter()
                .enumerate()
                .map(|(j, &v)| v * omega_k.pow([j as u64]))
                .sum();
        }

        engine.ntt_batch(&mut values_ntt, 13);
        assert_eq!(values_ntt, expected);
    }

    #[test]
    fn test_field256_ntt_size_26() {
        // 26 = 2 * 13, a mixed-radix size.
        let engine = NttEngine::<Field256>::new_from_fftfield();

        let values: Vec<_> = (1..=26).map(Field256::from).collect();
        let mut values_ntt = values.clone();

        let omega = engine.root(26);
        let mut expected = vec![Field256::ZERO; 26];
        for k in 0..26 {
            let omega_k = omega.pow([k as u64]);
            expected[k] = values
                .iter()
                .enumerate()
                .map(|(j, &v)| v * omega_k.pow([j as u64]))
                .sum();
        }

        engine.ntt_batch(&mut values_ntt, 26);
        assert_eq!(values_ntt, expected);
    }

    #[test]
    fn test_field256_ntt_size_39() {
        // 39 = 3 * 13, a mixed-radix size.
        let engine = NttEngine::<Field256>::new_from_fftfield();

        let values: Vec<_> = (1..=39).map(Field256::from).collect();
        let mut values_ntt = values.clone();

        let omega = engine.root(39);
        let mut expected = vec![Field256::ZERO; 39];
        for k in 0..39 {
            let omega_k = omega.pow([k as u64]);
            expected[k] = values
                .iter()
                .enumerate()
                .map(|(j, &v)| v * omega_k.pow([j as u64]))
                .sum();
        }

        engine.ntt_batch(&mut values_ntt, 39);
        assert_eq!(values_ntt, expected);
    }

    #[test]
    fn test_field256_ntt_roundtrip_mixed() {
        // Test NTT → INTT roundtrip for mixed-radix sizes.
        use ark_std::UniformRand;

        let engine = NttEngine::<Field256>::new_from_fftfield();
        let mut rng = ark_std::test_rng();

        for size in [
            3, 6, 9, 12, 13, 18, 26, 36, 39, 52, 72, 78, 104, 117, 144, 156,
        ] {
            let original: Vec<_> = (0..size).map(|_| Field256::rand(&mut rng)).collect();
            let mut values = original.clone();

            engine.ntt_batch(&mut values, size);
            engine.intt_batch(&mut values, size);

            // INTT omits the 1/n scaling factor, so we need to multiply by 1/n.
            let n_inv = Field256::from(size as u64).inverse().unwrap();
            for v in &mut values {
                *v *= n_inv;
            }

            assert_eq!(values, original, "Roundtrip failed for size {size}");
        }
    }
}
