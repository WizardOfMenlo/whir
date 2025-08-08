/// Returns sign, exponent and significant of an `f64`
///
/// The significant has the implicit leading one added for normal floats.
pub fn f64_parts(f: f64) -> (bool, i16, u64) {
    let bits = f.to_bits();
    let sign = (bits >> 63) != 0;
    let exp_bits = ((bits >> 52) & 0x7ff) as i16;
    let frac = bits & ((1 << 52) - 1);
    if exp_bits == 0 {
        // Subnormals and zero (no implicit 1)
        (sign, -1022, frac)
    } else {
        // Normal: add the implicit 1 at bit 52
        (sign, exp_bits - 1023, frac + (1 << 52))
    }
}

/// Convert a float to the nearest u256, clamping to zero and MAX.
pub fn f64_to_u256(f: f64) -> [u64; 4] {
    let (sign, exp, significand) = f64_parts(f);
    if sign {
        return [0; 4];
    }
    if exp > 256 {
        return [u64::MAX; 4];
    }
    let mut result = [0; 4];
    let shift = exp - 52;
    if shift < 0 {
        result[0] = f.round() as u64;
    } else {
        let shift = shift as u32;
        let (limb, shift) = ((shift / 64) as usize, shift % 64);
        result[limb] = significand << shift;
        if shift != 0 && limb < 3 {
            result[limb + 1] = significand >> (64 - shift);
        }
    }
    result
}

/// Little-endian comparison of two 256-bit integers represented as arrays of four u64s.
pub fn less_than(l: [u64; 4], r: [u64; 4]) -> bool {
    for (l, r) in l.into_iter().zip(r.into_iter()).rev() {
        if l < r {
            return true;
        } else if l > r {
            return false;
        }
    }
    false
}
