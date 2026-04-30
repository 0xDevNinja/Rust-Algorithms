//! Greatest Common Divisor (GCD) and Least Common Multiple (LCM) over `u64`.
//!
//! Both functions are `const fn` so they can be evaluated at compile time.
//!
//! # Algorithms
//! - **GCD**: iterative Euclidean algorithm — repeatedly replaces `(a, b)` with
//!   `(b, a % b)` until `b == 0`, at which point `a` is the GCD.
//! - **LCM**: computed as `a / gcd(a, b) * b` (division before multiplication)
//!   to keep intermediate values as small as possible and avoid overflow for
//!   inputs whose product would exceed `u64::MAX`.
//!
//! # Complexity
//! - Time:  O(log(min(a, b))) for GCD; O(log(min(a, b))) for LCM.
//! - Space: O(1) — no heap allocation.
//!
//! # Edge cases
//! - `gcd(0, n) == n` and `gcd(n, 0) == n` (0 is the identity for GCD).
//! - `gcd(0, 0) == 0`.
//! - `lcm(0, n) == 0` and `lcm(n, 0) == 0` by convention.

/// Returns the greatest common divisor of `a` and `b` using the Euclidean
/// algorithm.
///
/// # Examples
/// ```
/// use rust_algorithms::math::gcd_lcm::gcd;
/// assert_eq!(gcd(12, 18), 6);
/// assert_eq!(gcd(0, 7), 7);
/// assert_eq!(gcd(0, 0), 0);
/// ```
pub const fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

/// Returns the least common multiple of `a` and `b`.
///
/// Returns `0` if either argument is `0`.
///
/// The division `a / gcd(a, b)` is performed before the final multiplication
/// to avoid intermediate overflow for large inputs.
///
/// # Examples
/// ```
/// use rust_algorithms::math::gcd_lcm::lcm;
/// assert_eq!(lcm(4, 6), 12);
/// assert_eq!(lcm(0, 5), 0);
/// ```
pub const fn lcm(a: u64, b: u64) -> u64 {
    if a == 0 || b == 0 {
        return 0;
    }
    a / gcd(a, b) * b
}

#[cfg(test)]
mod tests {
    use super::{gcd, lcm};
    use quickcheck_macros::quickcheck;

    // --- GCD ---

    #[test]
    fn gcd_both_zero() {
        assert_eq!(gcd(0, 0), 0);
    }

    #[test]
    fn gcd_left_zero() {
        assert_eq!(gcd(0, 7), 7);
    }

    #[test]
    fn gcd_right_zero() {
        assert_eq!(gcd(7, 0), 7);
    }

    #[test]
    fn gcd_twelve_eighteen() {
        assert_eq!(gcd(12, 18), 6);
    }

    #[test]
    fn gcd_fortyeight_thirtysix() {
        assert_eq!(gcd(48, 36), 12);
    }

    #[test]
    fn gcd_coprime() {
        assert_eq!(gcd(13, 17), 1);
    }

    // --- LCM ---

    #[test]
    fn lcm_left_zero() {
        assert_eq!(lcm(0, 5), 0);
    }

    #[test]
    fn lcm_right_zero() {
        assert_eq!(lcm(5, 0), 0);
    }

    #[test]
    fn lcm_four_six() {
        assert_eq!(lcm(4, 6), 12);
    }

    #[test]
    fn lcm_twentyone_six() {
        assert_eq!(lcm(21, 6), 42);
    }

    /// `lcm(u64::MAX / 2, 2)` must not panic; the division-before-multiply
    /// formula keeps the intermediate value within u64 range.
    #[test]
    fn lcm_no_overflow_large_inputs() {
        // a = 2^32, b = 2^33; gcd = 2^32, lcm = 2^33 which fits in u64.
        let a: u64 = 1 << 32;
        let b: u64 = 1 << 33;
        assert_eq!(lcm(a, b), 1u64 << 33);

        // lcm(u64::MAX / 2, 2): u64::MAX is odd, so u64::MAX / 2 is also odd,
        // meaning gcd = 1 and the division-before-multiply formula produces
        // (u64::MAX / 2) * 2 without intermediate overflow.
        let half = u64::MAX / 2;
        assert_eq!(lcm(half, 2), half * 2);
    }

    // --- Property test ---

    /// For nonzero `u32` inputs cast to `u64`: gcd(a, b) * lcm(a, b) == a * b.
    /// Using `u32` inputs guarantees `a * b` fits in `u64`, so the product on
    /// the right-hand side never overflows.
    #[allow(clippy::needless_pass_by_value)]
    #[quickcheck]
    fn prop_gcd_lcm_product(a: u32, b: u32) -> bool {
        let a = a as u64;
        let b = b as u64;
        if a == 0 || b == 0 {
            // Convention: lcm is 0 for zero inputs; skip the product identity.
            return true;
        }
        gcd(a, b) * lcm(a, b) == a * b
    }
}
