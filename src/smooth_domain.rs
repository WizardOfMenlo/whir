//! Utilities for smooth-{2,3,13} numbers: values of the form `2^a * 3^b * 13^c`.
//!
//! BN254 Fr has `r-1 = 2^28 * 3^2 * 13 * …`, so NTT domains can be
//! `2^a * 3^b * 13^c` with `a ≤ 28`, `b ≤ 2`, and `c ≤ 1`.
//! Using smooth sizes instead of strict powers of two lets us pad closer
//! to the actual constraint count.

/// Returns `true` if `n` is a smooth-{2,3,13} number, i.e. `n = 2^a * 3^b * 13^c`.
///
/// `0` is NOT smooth.
#[inline]
pub const fn is_smooth(n: usize) -> bool {
    if n == 0 {
        return false;
    }
    let mut v = n;
    while v % 2 == 0 {
        v /= 2;
    }
    while v % 3 == 0 {
        v /= 3;
    }
    while v % 13 == 0 {
        v /= 13;
    }
    v == 1
}

/// Returns the odd part of `n`: after removing all factors of 2,
/// the remaining value `3^b * 13^c`.
#[inline]
pub const fn odd_part(n: usize) -> usize {
    assert!(n > 0);
    let mut v = n;
    while v % 2 == 0 {
        v /= 2;
    }
    v
}

/// Number of extra sumcheck rounds needed for the odd residual:
/// `ceil(log2(odd_part))` where `odd_part = 3^b * 13^c`.
#[inline]
pub const fn extra_rounds(n: usize) -> usize {
    let odd = odd_part(n);
    if odd == 1 {
        0
    } else {
        // ceil(log2(odd)) = 64 - (odd - 1).leading_zeros()
        (usize::BITS - (odd - 1).leading_zeros()) as usize
    }
}

/// Next power of two that is ≥ the odd part.
/// Equivalently `1 << extra_rounds(n)`.
#[inline]
pub const fn odd_part_padded(n: usize) -> usize {
    1 << extra_rounds(n)
}

/// Padded size: the power-of-two size that the final sumcheck operates on
/// for a smooth polynomial of size `n`.
///
/// Equal to `2^(a + ceil(log2(odd)))` where `n = 2^a * odd`.
#[inline]
pub const fn padded_size(n: usize) -> usize {
    let a = n.trailing_zeros() as usize;
    let extra = extra_rounds(n);
    1 << (a + extra)
}

/// Find the smallest smooth-{2,3,13} number ≥ `n`.
///
/// Enumerates all `2^a * 3^b * 13^c` with `b ≤ 2`, `c ≤ 1`.
pub const fn next_smooth(n: usize) -> usize {
    if n <= 1 {
        return n;
    }
    let mut best = n.next_power_of_two(); // worst case: pure power of 2

    // Try each combination of 3^b * 13^c
    let mut c: usize = 0;
    while c <= 1 {
        let pow13 = if c == 0 { 1 } else { 13 };
        let mut pow3: usize = 1;
        let mut b: usize = 0;
        while b <= 2 {
            let odd = pow3 * pow13;
            // Smallest 2^a such that 2^a * odd >= n
            let needed = (n + odd - 1) / odd; // ceil(n / odd)
            let pow2 = needed.next_power_of_two();
            let candidate = pow2 * odd;
            if candidate < best {
                best = candidate;
            }
            pow3 *= 3;
            b += 1;
        }
        c += 1;
    }
    best
}

/// Decompose a smooth-{2,3,13} number into `(a, b, c)` where `n = 2^a * 3^b * 13^c`.
///
/// Panics if `n` is not smooth-{2,3,13}.
#[inline]
pub const fn decompose(n: usize) -> (usize, usize, usize) {
    assert!(n > 0);
    let a = n.trailing_zeros() as usize;
    let mut odd = n >> a;
    let mut b = 0;
    while odd % 3 == 0 {
        odd /= 3;
        b += 1;
    }
    let mut c = 0;
    while odd % 13 == 0 {
        odd /= 13;
        c += 1;
    }
    assert!(odd == 1, "not a smooth-{{2,3,13}} number");
    (a, b, c)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_smooth() {
        assert!(!is_smooth(0));
        assert!(is_smooth(1));
        assert!(is_smooth(2));
        assert!(is_smooth(3));
        assert!(is_smooth(4));
        assert!(!is_smooth(5));
        assert!(is_smooth(6));
        assert!(!is_smooth(7));
        assert!(is_smooth(8));
        assert!(is_smooth(9));
        assert!(!is_smooth(10));
        assert!(is_smooth(12));
        assert!(is_smooth(13)); // 13^1
        assert!(is_smooth(16));
        assert!(is_smooth(18));
        assert!(is_smooth(24));
        assert!(is_smooth(26)); // 2 * 13
        assert!(is_smooth(39)); // 3 * 13
        assert!(is_smooth(78)); // 2 * 3 * 13
        assert!(is_smooth(117)); // 9 * 13
        assert!(!is_smooth(15));
        assert!(!is_smooth(20));
        assert!(is_smooth(1 << 28));
        assert!(is_smooth(9 * (1 << 18)));
        assert!(is_smooth(13 * (1 << 10)));
        assert!(is_smooth(117 * (1 << 5))); // 9 * 13 * 32
    }

    #[test]
    fn test_odd_part() {
        assert_eq!(odd_part(1), 1);
        assert_eq!(odd_part(2), 1);
        assert_eq!(odd_part(4), 1);
        assert_eq!(odd_part(8), 1);
        assert_eq!(odd_part(3), 3);
        assert_eq!(odd_part(6), 3);
        assert_eq!(odd_part(12), 3);
        assert_eq!(odd_part(9), 9);
        assert_eq!(odd_part(18), 9);
        assert_eq!(odd_part(36), 9);
        assert_eq!(odd_part(9 * (1 << 18)), 9);
        assert_eq!(odd_part(13), 13);
        assert_eq!(odd_part(26), 13);
        assert_eq!(odd_part(39), 39);
        assert_eq!(odd_part(78), 39);
        assert_eq!(odd_part(117), 117);
        assert_eq!(odd_part(117 * 4), 117);
    }

    #[test]
    fn test_extra_rounds() {
        // odd part = 1: no extra rounds
        assert_eq!(extra_rounds(1), 0);
        assert_eq!(extra_rounds(2), 0);
        assert_eq!(extra_rounds(1024), 0);

        // odd part = 3: ceil(log2(3)) = 2
        assert_eq!(extra_rounds(3), 2);
        assert_eq!(extra_rounds(6), 2);

        // odd part = 9: ceil(log2(9)) = 4
        assert_eq!(extra_rounds(9), 4);
        assert_eq!(extra_rounds(18), 4);

        // odd part = 13: ceil(log2(13)) = 4
        assert_eq!(extra_rounds(13), 4);
        assert_eq!(extra_rounds(26), 4);

        // odd part = 39 = 3*13: ceil(log2(39)) = 6
        assert_eq!(extra_rounds(39), 6);
        assert_eq!(extra_rounds(78), 6);

        // odd part = 117 = 9*13: ceil(log2(117)) = 7
        assert_eq!(extra_rounds(117), 7);
        assert_eq!(extra_rounds(234), 7);
    }

    #[test]
    fn test_odd_part_padded() {
        assert_eq!(odd_part_padded(1), 1);
        assert_eq!(odd_part_padded(8), 1);
        assert_eq!(odd_part_padded(3), 4);
        assert_eq!(odd_part_padded(6), 4);
        assert_eq!(odd_part_padded(9), 16);
        assert_eq!(odd_part_padded(18), 16);
        assert_eq!(odd_part_padded(13), 16); // ceil(log2(13)) = 4, 2^4 = 16
        assert_eq!(odd_part_padded(26), 16);
        assert_eq!(odd_part_padded(39), 64); // ceil(log2(39)) = 6, 2^6 = 64
        assert_eq!(odd_part_padded(117), 128); // ceil(log2(117)) = 7, 2^7 = 128
    }

    #[test]
    fn test_padded_size() {
        assert_eq!(padded_size(1), 1);
        assert_eq!(padded_size(8), 8);
        assert_eq!(padded_size(1024), 1024);
        // 3 * 2^2 → padded = 2^(2+2) = 16
        assert_eq!(padded_size(12), 16);
        // 9 * 2^1 → padded = 2^(1+4) = 32
        assert_eq!(padded_size(18), 32);
        // 13 * 2^1 → padded = 2^(1+4) = 32
        assert_eq!(padded_size(26), 32);
        // 39 * 2^1 → padded = 2^(1+6) = 128
        assert_eq!(padded_size(78), 128);
        // 117 * 2^2 → padded = 2^(2+7) = 512
        assert_eq!(padded_size(468), 512);
    }

    #[test]
    fn test_next_smooth() {
        assert_eq!(next_smooth(0), 0);
        assert_eq!(next_smooth(1), 1);
        assert_eq!(next_smooth(2), 2);
        assert_eq!(next_smooth(3), 3);
        assert_eq!(next_smooth(4), 4);
        assert_eq!(next_smooth(5), 6);
        assert_eq!(next_smooth(6), 6);
        assert_eq!(next_smooth(7), 8);
        assert_eq!(next_smooth(8), 8);
        assert_eq!(next_smooth(9), 9);
        assert_eq!(next_smooth(10), 12);
        assert_eq!(next_smooth(11), 12);
        assert_eq!(next_smooth(12), 12);
        assert_eq!(next_smooth(13), 13); // now 13 is smooth!
        assert_eq!(next_smooth(14), 16);
        assert_eq!(next_smooth(17), 18);
        assert_eq!(next_smooth(18), 18);
        assert_eq!(next_smooth(19), 24);
    }

    #[test]
    fn test_decompose() {
        assert_eq!(decompose(1), (0, 0, 0));
        assert_eq!(decompose(2), (1, 0, 0));
        assert_eq!(decompose(3), (0, 1, 0));
        assert_eq!(decompose(4), (2, 0, 0));
        assert_eq!(decompose(6), (1, 1, 0));
        assert_eq!(decompose(8), (3, 0, 0));
        assert_eq!(decompose(9), (0, 2, 0));
        assert_eq!(decompose(12), (2, 1, 0));
        assert_eq!(decompose(13), (0, 0, 1));
        assert_eq!(decompose(18), (1, 2, 0));
        assert_eq!(decompose(26), (1, 0, 1));
        assert_eq!(decompose(39), (0, 1, 1));
        assert_eq!(decompose(78), (1, 1, 1));
        assert_eq!(decompose(117), (0, 2, 1));
        assert_eq!(decompose(9 * (1 << 18)), (18, 2, 0));
        assert_eq!(decompose(13 * (1 << 10)), (10, 0, 1));
    }

    #[test]
    fn test_next_smooth_all_results_are_smooth() {
        for n in 1..=10_000 {
            let s = next_smooth(n);
            assert!(s >= n, "next_smooth({n}) = {s} < {n}");
            assert!(is_smooth(s), "next_smooth({n}) = {s} is not smooth");
        }
    }
}
