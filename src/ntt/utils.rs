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
pub fn lcm(a: usize, b: usize) -> usize {
    a * b / gcd(a, b)
}

// Greatest common divisor.
pub fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}
