/// Compute the largest factor of `n` that is ≤ sqrt(n).
/// Assumes `n` is a smooth-{2,3,13} number, i.e. of the form `2^a * 3^b * 13^c`.
pub fn sqrt_factor(n: usize) -> usize {
    let twos = n.trailing_zeros() as usize;
    let odd = n >> twos;

    // Enumerate all divisors of the odd part and for each, find the largest
    // power-of-2 multiplier that keeps the product ≤ sqrt(n).
    let odd_divisors: &[usize] = match odd {
        1 => &[1],
        3 => &[1, 3],
        9 => &[1, 3, 9],
        13 => &[1, 13],
        39 => &[1, 3, 13, 39],
        117 => &[1, 3, 9, 13, 39, 117],
        _ => panic!("n is not a smooth-{{2,3,13}} number"),
    };

    let mut best = 1usize;
    for &d in odd_divisors {
        let d_sq = d * d;
        if d_sq > n {
            continue;
        }
        // We need d * 2^a ≤ sqrt(n), i.e. d² * 4^a ≤ n, i.e. 4^a ≤ n/d².
        let ratio = n / d_sq;
        // max a such that 4^a ≤ ratio: a = floor(log2(ratio)) / 2, capped at twos.
        let max_2a = (usize::BITS - 1 - ratio.leading_zeros()) as usize;
        let a = (max_2a / 2).min(twos);
        best = best.max(d << a);
    }
    best
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

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::{gcd, lcm, sqrt_factor};

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

        // Cases where n = 2^k * 13
        assert_eq!(sqrt_factor(13), 1); // 13 = 2^0 * 13
        assert_eq!(sqrt_factor(26), 2); // 26 = 2^1 * 13
        assert_eq!(sqrt_factor(52), 4); // 52 = 2^2 * 13
        assert_eq!(sqrt_factor(208), 13); // 208 = 2^4 * 13
        assert_eq!(sqrt_factor(832), 26); // 832 = 2^6 * 13
        assert_eq!(sqrt_factor(3328), 52); // 3328 = 2^8 * 13

        // Cases where n = 2^k * 39
        assert_eq!(sqrt_factor(39), 3); // 39 = 2^0 * 39
        assert_eq!(sqrt_factor(78), 6); // 78 = 2^1 * 39
        assert_eq!(sqrt_factor(156), 12); // 156 = 2^2 * 39
        assert_eq!(sqrt_factor(624), 24); // 624 = 2^4 * 39
        assert_eq!(sqrt_factor(2496), 48); // 2496 = 2^6 * 39

        // Cases where n = 2^k * 117
        assert_eq!(sqrt_factor(117), 9); // 117 = 2^0 * 117
        assert_eq!(sqrt_factor(234), 13); // 234 = 2^1 * 117
        assert_eq!(sqrt_factor(468), 18); // 468 = 2^2 * 117
        assert_eq!(sqrt_factor(1872), 39); // 1872 = 2^4 * 117
        assert_eq!(sqrt_factor(7488), 78); // 7488 = 2^6 * 117
    }

    proptest! {
        #[test]
        fn proptest_sqrt_factor(k in 0usize..30, base in prop_oneof![Just(1), Just(3), Just(9), Just(13), Just(39), Just(117)])
        {
            let n = (1 << k) * base;
            let expected = get_largest_divisor_up_to_sqrt(n);
            prop_assert_eq!(sqrt_factor(n), expected);
        }
    }
}
