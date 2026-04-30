//! Fast-doubling Fibonacci — compute F(n) in O(log n) multiplications.
//!
//! # Algorithm
//! Given F(k) and F(k+1), the next pair (F(2k), F(2k+1)) is derived from:
//! - F(2k)   = F(k) · (2·F(k+1) − F(k))
//! - F(2k+1) = F(k)² + F(k+1)²
//!
//! The `n`-th Fibonacci number is then computed by walking the bits of `n`
//! from the most significant bit downward, doubling the index at each step
//! and conditionally advancing by one when the current bit is set. Each step
//! performs a constant number of multiplications, so the total work is
//! O(log n) multiplications versus O(n) additions for the naive iterative
//! recurrence.
//!
//! # Complexity
//! - Time:  O(log n) word-sized multiplications.
//! - Space: O(1) — no heap allocation, no recursion.
//!
//! # Conventions
//! `F(0) = 0`, `F(1) = 1`, `F(n) = F(n-1) + F(n-2)`.
//!
//! # Safe range
//! Results are returned as [`u128`]. While `F(186)` itself still fits in a
//! `u128`, the doubling step computes both `F(2k)` and `F(2k+1)` at each
//! iteration, so reaching `F(186)` requires forming `F(187)` as an
//! intermediate — and `F(187)` overflows `u128`. Using checked arithmetic
//! end-to-end, [`fibonacci`] returns `Some(F(n))` for `n <= 185` and `None`
//! for `n >= 186`.

/// Returns `Some(F(n))` where `F(0) = 0`, `F(1) = 1`, computed via the
/// fast-doubling identities in O(log n) multiplications.
///
/// Returns `None` if any intermediate value (or the result itself) would
/// overflow [`u128`]. In practice this means `Some(F(n))` for `n <= 185`
/// and `None` for `n >= 186`.
///
/// # Examples
/// ```
/// use rust_algorithms::math::fast_doubling_fibonacci::fibonacci;
/// assert_eq!(fibonacci(0), Some(0));
/// assert_eq!(fibonacci(1), Some(1));
/// assert_eq!(fibonacci(10), Some(55));
/// assert_eq!(fibonacci(50), Some(12_586_269_025));
/// assert_eq!(fibonacci(186), None);
/// ```
#[must_use]
pub fn fibonacci(n: u64) -> Option<u128> {
    // (a, b) = (F(k), F(k+1)) starts at k = 0.
    let mut a: u128 = 0;
    let mut b: u128 = 1;

    // Walk bits of `n` from the most significant set bit downward. Skipping
    // the leading zero bits keeps the index doubling tight: after `i` steps,
    // (a, b) holds (F(k), F(k+1)) where `k` is the high `i` bits of `n`.
    let bits = u64::BITS - n.leading_zeros();
    for i in (0..bits).rev() {
        // Doubling: (F(k), F(k+1)) -> (F(2k), F(2k+1)).
        // c = F(2k)   = F(k) * (2*F(k+1) - F(k))
        // d = F(2k+1) = F(k)^2 + F(k+1)^2
        let two_b = b.checked_mul(2)?;
        let two_b_minus_a = two_b.checked_sub(a)?;
        let c = a.checked_mul(two_b_minus_a)?;
        let a_sq = a.checked_mul(a)?;
        let b_sq = b.checked_mul(b)?;
        let d = a_sq.checked_add(b_sq)?;

        if (n >> i) & 1 == 1 {
            // Bit set: advance one step -> (F(2k+1), F(2k+2)) where
            // F(2k+2) = F(2k) + F(2k+1) = c + d.
            a = d;
            b = c.checked_add(d)?;
        } else {
            a = c;
            b = d;
        }
    }

    Some(a)
}

#[cfg(test)]
mod tests {
    use super::fibonacci;
    use quickcheck_macros::quickcheck;

    #[test]
    fn f_0() {
        assert_eq!(fibonacci(0), Some(0));
    }

    #[test]
    fn f_1() {
        assert_eq!(fibonacci(1), Some(1));
    }

    #[test]
    fn f_2() {
        assert_eq!(fibonacci(2), Some(1));
    }

    #[test]
    fn f_3() {
        assert_eq!(fibonacci(3), Some(2));
    }

    #[test]
    fn f_10() {
        assert_eq!(fibonacci(10), Some(55));
    }

    #[test]
    fn f_50() {
        assert_eq!(fibonacci(50), Some(12_586_269_025));
    }

    /// F(93) is the largest Fibonacci number that still fits in a `u64`.
    #[test]
    fn f_93_last_fits_in_u64() {
        assert_eq!(fibonacci(93), Some(12_200_160_415_121_876_738));
    }

    /// F(185) is the largest index this routine can compute without an
    /// intermediate `u128` overflow (the doubling step would otherwise need
    /// to form F(187), which exceeds `u128::MAX`).
    #[test]
    fn f_185_largest_safe() {
        assert_eq!(
            fibonacci(185),
            Some(205_697_230_343_233_228_174_223_751_303_346_572_685),
        );
    }

    /// `F(186)` itself fits in `u128`, but the doubling step would compute
    /// `F(187)` as an intermediate, which overflows; checked arithmetic
    /// surfaces this as `None`.
    #[test]
    fn f_186_overflows_intermediate() {
        assert_eq!(fibonacci(186), None);
    }

    /// `F(187)` overflows `u128`.
    #[test]
    fn f_187_overflows() {
        assert_eq!(fibonacci(187), None);
    }

    /// For random `n` up to 80 (well within the u128 safe range), the
    /// fast-doubling result must match a straightforward O(n) iterative
    /// reference implementation.
    #[quickcheck]
    fn prop_matches_iterative_reference(n: u8) -> bool {
        let n = u64::from(n.min(80));
        let mut a: u128 = 0;
        let mut b: u128 = 1;
        for _ in 0..n {
            let next = a + b;
            a = b;
            b = next;
        }
        fibonacci(n) == Some(a)
    }
}
