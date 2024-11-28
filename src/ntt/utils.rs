/// Target single-thread workload size for `T`.
/// Should ideally be a multiple of a cache line (64 bytes)
/// and close to the L1 cache size (32 KB).
pub const fn workload_size<T: Sized>() -> usize {
    const CACHE_SIZE: usize = 1 << 15;
    CACHE_SIZE / size_of::<T>()
}

/// Cast a slice into chunks of size N.
///
/// TODO: Replace with `slice::as_chunks` when stable.
pub fn as_chunks_exact_mut<T, const N: usize>(slice: &mut [T]) -> &mut [[T; N]] {
    assert!(N != 0, "chunk size must be non-zero");
    assert_eq!(
        slice.len() % N,
        0,
        "slice length must be a multiple of chunk size"
    );
    // SAFETY: Caller must guarantee that `N` is nonzero and exactly divides the slice length
    let new_len = slice.len() / N;
    // SAFETY: We cast a slice of `new_len * N` elements into
    // a slice of `new_len` many `N` elements chunks.
    unsafe { std::slice::from_raw_parts_mut(slice.as_mut_ptr().cast(), new_len) }
}

/// Compute the largest factor of n that is <= sqrt(n).
/// Assumes n is of the form 2^k * {1,3,9}.
pub fn sqrt_factor(n: usize) -> usize {
    let twos = n.trailing_zeros();
    match n >> twos {
        1 => 1 << (twos / 2),
        3 | 9 => 3 << (twos / 2),
        _ => panic!(),
    }
}

/// Least common multiple.
///
/// Note that lcm(0,0) will panic (rather than give the correct answer 0).
pub fn lcm(a: usize, b: usize) -> usize {
    a * (b / gcd(a, b))
}

/// Greatest common divisor.
pub fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

#[cfg(test)]
mod tests {
    use super::{as_chunks_exact_mut, gcd, lcm, sqrt_factor};

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
        // naive brute-force search for largest divisor up to sqrt n.
        // This is not supposed to be efficient, but optimized for "ease of convincing yourself it's correct (provided none of the asserts trigger)".
        fn get_largest_divisor_up_to_sqrt(x: usize) -> usize {
            if x == 0 {
                return 0;
            }
            let mut result = 1;
            let isqrt_of_x: usize = {
                // use x.isqrt() once this is stabilized. That would be MUCH simpler.

                assert!(x < (1 << f64::MANTISSA_DIGITS)); // guarantees that each of {x, floor(sqrt(x)), ceil(sqrt(x))} can be represented exactly by f64.
                let x_as_float = x as f64;
                // sqrt is guaranteed to be the exact result, then rounded. Due to the above assert, the rounded value is between floor(sqrt(x)) and ceil(sqrt(x)).
                let sqrt_x = x_as_float.sqrt();
                // We return sqrt_x, rounded to 0; for correctness, we need to rule out that we rounded from a non-integer up to the integer ceil(sqrt(x)).
                if sqrt_x.fract() == 0.0 {
                    assert!(sqrt_x * sqrt_x == x_as_float);
                }
                unsafe { sqrt_x.to_int_unchecked() }
            };
            for i in 1..=isqrt_of_x {
                if x % i == 0 {
                    result = i;
                }
            }
            result
        }

        for i in 0..10 {
            assert_eq!(sqrt_factor(1 << i), get_largest_divisor_up_to_sqrt(1 << i));
        }
        for i in 0..10 {
            assert_eq!(sqrt_factor(1 << i), get_largest_divisor_up_to_sqrt(1 << i));
        }

        for i in 0..10 {
            assert_eq!(sqrt_factor(1 << i), get_largest_divisor_up_to_sqrt(1 << i));
        }
    }

    #[test]
    fn test_as_chunks_exact_mut() {
        let v = &mut [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        assert_eq!(
            as_chunks_exact_mut::<_, 12>(v),
            &[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]
        );
        assert_eq!(
            as_chunks_exact_mut::<_, 6>(v),
            &[[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]
        );
        assert_eq!(
            as_chunks_exact_mut::<_, 1>(v),
            &[
                [1],
                [2],
                [3],
                [4],
                [5],
                [6],
                [7],
                [8],
                [9],
                [10],
                [11],
                [12]
            ]
        );
        let should_not_work = std::panic::catch_unwind(|| {
            as_chunks_exact_mut::<_, 2>(&mut [1, 2, 3]);
        });
        assert!(should_not_work.is_err())
    }
}
