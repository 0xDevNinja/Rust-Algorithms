//! Binary search on the answer (a.k.a. parametric search).
//!
//! Given a monotone predicate `check(x)` over a totally-ordered domain, binary
//! search finds the boundary where the predicate flips from `false` to `true`
//! (or vice versa). The integer variant runs in `O(log(hi - lo))` predicate
//! evaluations; the floating-point variant performs a fixed number of
//! bisection steps to drive the gap below an epsilon.
//!
//! Typical use: minimise `t` subject to `feasible(t)`, where `feasible` is
//! monotone in `t` (Aggressive Cows, Allocate-Books, K-th-magical-number,
//! capacity-to-ship problems).

/// Smallest integer `x` in the half-open range `[lo, hi)` such that
/// `check(x)` is `true`, assuming `check` is monotone non-decreasing in `x`
/// (`false` for small `x`, then becomes `true`). Returns `hi` if no such `x`
/// exists in the range.
///
/// - Time: `O(log(hi - lo))` predicate evaluations.
/// - Space: `O(1)`.
///
/// # Panics
/// Panics if `lo > hi`.
pub fn first_true_i64(lo: i64, hi: i64, mut check: impl FnMut(i64) -> bool) -> i64 {
    assert!(lo <= hi, "first_true_i64: lo ({lo}) must be <= hi ({hi})");
    let mut l = lo;
    let mut r = hi;
    while l < r {
        let mid = l + (r - l) / 2;
        if check(mid) {
            r = mid;
        } else {
            l = mid + 1;
        }
    }
    l
}

/// Largest integer `x` in `[lo, hi)` such that `check(x)` is `true`, assuming
/// `check` is monotone non-increasing in `x` (`true` for small `x`, then
/// becomes `false`). Returns `lo - 1` if no such `x` exists in the range.
///
/// - Time: `O(log(hi - lo))` predicate evaluations.
/// - Space: `O(1)`.
///
/// # Panics
/// Panics if `lo > hi`.
pub fn last_true_i64(lo: i64, hi: i64, mut check: impl FnMut(i64) -> bool) -> i64 {
    assert!(lo <= hi, "last_true_i64: lo ({lo}) must be <= hi ({hi})");
    // Find first false, then return it - 1.
    let first_false = first_true_i64(lo, hi, |x| !check(x));
    first_false - 1
}

/// Bisection search over `f64` with a fixed iteration budget. Returns an `x`
/// in `[lo, hi]` such that `check(x)` is `true` and `check(x - eps)` is
/// `false` for `eps -> 0`, assuming `check` is monotone non-decreasing.
///
/// - Time: `O(iterations)` predicate evaluations (default `100` is enough to
///   drive the interval below `1e-9` from a unit range).
/// - Space: `O(1)`.
///
/// # Panics
/// Panics if `lo > hi`.
pub fn first_true_f64(
    lo: f64,
    hi: f64,
    iterations: u32,
    mut check: impl FnMut(f64) -> bool,
) -> f64 {
    assert!(lo <= hi, "first_true_f64: lo ({lo}) must be <= hi ({hi})");
    let mut l = lo;
    let mut r = hi;
    for _ in 0..iterations {
        let mid = l + (r - l) / 2.0;
        if check(mid) {
            r = mid;
        } else {
            l = mid;
        }
    }
    r
}

#[cfg(test)]
mod tests {
    use super::{first_true_f64, first_true_i64, last_true_i64};
    use quickcheck_macros::quickcheck;

    #[test]
    fn first_true_basic() {
        // check(x) = (x >= 17)
        let r = first_true_i64(0, 100, |x| x >= 17);
        assert_eq!(r, 17);
    }

    #[test]
    fn first_true_at_lo() {
        let r = first_true_i64(5, 50, |_| true);
        assert_eq!(r, 5);
    }

    #[test]
    fn first_true_never_holds() {
        // returns hi when no x satisfies the predicate
        let r = first_true_i64(0, 10, |_| false);
        assert_eq!(r, 10);
    }

    #[test]
    fn first_true_singleton() {
        let r = first_true_i64(7, 8, |x| x >= 7);
        assert_eq!(r, 7);
        let r = first_true_i64(7, 8, |x| x >= 8);
        assert_eq!(r, 8);
    }

    #[test]
    fn first_true_empty_range_returns_lo() {
        let r = first_true_i64(5, 5, |_| true);
        assert_eq!(r, 5);
    }

    #[test]
    fn last_true_basic() {
        // monotone non-increasing: x <= 10
        let r = last_true_i64(0, 100, |x| x <= 10);
        assert_eq!(r, 10);
    }

    #[test]
    fn last_true_never_holds() {
        let r = last_true_i64(0, 10, |_| false);
        assert_eq!(r, -1);
    }

    #[test]
    fn float_sqrt_two() {
        // check(x) = (x*x >= 2)
        let r = first_true_f64(0.0, 2.0, 100, |x| x * x >= 2.0);
        assert!((r - 2.0_f64.sqrt()).abs() < 1e-9);
    }

    #[test]
    fn float_predicate_never_true_returns_hi() {
        let r = first_true_f64(0.0, 1.0, 60, |_| false);
        assert_eq!(r, 1.0);
    }

    #[test]
    #[should_panic(expected = "must be <= hi")]
    fn first_true_inverted_range_panics() {
        let _ = first_true_i64(10, 5, |_| true);
    }

    #[allow(clippy::needless_pass_by_value)]
    #[quickcheck]
    fn first_true_matches_linear(threshold: i32, lo: i32, hi: i32) -> bool {
        let lo = i64::from(lo % 200);
        let hi = i64::from(hi % 200);
        if lo > hi {
            return true;
        }
        let threshold = i64::from(threshold);
        let got = first_true_i64(lo, hi, |x| x >= threshold);
        let want = (lo..hi).find(|&x| x >= threshold).unwrap_or(hi);
        got == want
    }
}
