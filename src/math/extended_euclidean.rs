//! Extended Euclidean algorithm. Returns `(gcd(a, b), x, y)` such that
//! `a·x + b·y = gcd(a, b)`. Foundation for modular inverse and Bezout
//! identity computations.

/// Returns `(gcd, x, y)` with `a*x + b*y = gcd(a, b)`.
pub const fn ext_gcd(a: i64, b: i64) -> (i64, i64, i64) {
    let (mut old_r, mut r) = (a, b);
    let (mut old_s, mut s) = (1_i64, 0_i64);
    let (mut old_t, mut t) = (0_i64, 1_i64);
    while r != 0 {
        let q = old_r / r;
        let new_r = old_r - q * r;
        old_r = r;
        r = new_r;
        let new_s = old_s - q * s;
        old_s = s;
        s = new_s;
        let new_t = old_t - q * t;
        old_t = t;
        t = new_t;
    }
    (old_r, old_s, old_t)
}

/// Returns the modular inverse of `a` modulo `m` if it exists, i.e. some
/// `x` in `[0, m)` such that `a*x ≡ 1 (mod m)`. Requires `gcd(a, m) == 1`.
pub const fn mod_inverse(a: i64, m: i64) -> Option<i64> {
    let (g, x, _) = ext_gcd(a.rem_euclid(m), m);
    if g == 1 {
        Some(x.rem_euclid(m))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::{ext_gcd, mod_inverse};

    #[test]
    fn coprime_inputs() {
        let (g, x, y) = ext_gcd(35, 15);
        assert_eq!(g, 5);
        assert_eq!(35 * x + 15 * y, 5);
    }

    #[test]
    fn one_is_zero() {
        let (g, _, _) = ext_gcd(0, 5);
        assert_eq!(g, 5);
    }

    #[test]
    fn both_zero() {
        let (g, _, _) = ext_gcd(0, 0);
        assert_eq!(g, 0);
    }

    #[test]
    fn classic_31_and_99() {
        let (g, x, y) = ext_gcd(31, 99);
        assert_eq!(g, 1);
        assert_eq!(31 * x + 99 * y, 1);
    }

    #[test]
    fn inverse_exists() {
        // 3 * 5 = 15 ≡ 1 (mod 7) ?  15 mod 7 = 1 ✓
        assert_eq!(mod_inverse(3, 7), Some(5));
    }

    #[test]
    fn inverse_does_not_exist() {
        // gcd(2, 4) = 2 ≠ 1
        assert_eq!(mod_inverse(2, 4), None);
    }

    #[test]
    fn inverse_negative_input() {
        // -3 mod 7 = 4; inverse of 4 mod 7 = 2 (since 8 mod 7 = 1)
        assert_eq!(mod_inverse(-3, 7), Some(2));
    }

    #[test]
    fn against_brute_force_inverse() {
        // For prime m=11, every nonzero value should have an inverse.
        for a in 1..11 {
            let inv = mod_inverse(a, 11).unwrap();
            assert_eq!((a * inv).rem_euclid(11), 1);
        }
    }
}
