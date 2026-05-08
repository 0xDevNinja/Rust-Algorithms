//! Bisection method for locating a root of a continuous real function on an
//! interval whose endpoints bracket a sign change.
//!
//! Each iteration evaluates `f` at the midpoint and discards the half that
//! does not contain a sign change. The bracket width is halved every step,
//! giving linear convergence with one bit of accuracy per iteration. The
//! method only requires that `f` be continuous and that `f(lo)` and `f(hi)`
//! have opposite signs — it does not require differentiability.
//!
//! Time complexity: `O(log2((hi - lo) / tol))` evaluations of `f`.
//! Space complexity: `O(1)`.

/// Finds a root of `f` in `[lo, hi]` by repeated bisection.
///
/// Returns `Some(root)` once `|f(mid)| <= tol` or `(hi - lo) / 2 <= tol`.
/// Returns `None` if `f(lo)` and `f(hi)` share a sign (no bracketed root) or
/// if `max_iter` iterations elapse without converging. A zero at either
/// endpoint is reported immediately.
pub fn bisect<F: Fn(f64) -> f64>(
    f: F,
    mut lo: f64,
    mut hi: f64,
    tol: f64,
    max_iter: u32,
) -> Option<f64> {
    if lo > hi {
        core::mem::swap(&mut lo, &mut hi);
    }

    let f_lo = f(lo);
    let f_hi = f(hi);

    if f_lo == 0.0 {
        return Some(lo);
    }
    if f_hi == 0.0 {
        return Some(hi);
    }

    // Same sign at both endpoints: no guaranteed bracketed root.
    if f_lo.signum() == f_hi.signum() {
        return None;
    }

    let lo_sign = f_lo.signum();

    for _ in 0..max_iter {
        let mid = (hi - lo).mul_add(0.5, lo);
        let f_mid = f(mid);

        if f_mid.abs() <= tol || (hi - lo) * 0.5 <= tol {
            return Some(mid);
        }

        if f_mid.signum() == lo_sign {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    None
}

#[cfg(test)]
#[allow(clippy::suboptimal_flops, clippy::manual_let_else)]
mod tests {
    use super::bisect;
    use quickcheck::TestResult;
    use quickcheck_macros::quickcheck;

    #[test]
    fn sqrt_two() {
        let root = bisect(|x| x * x - 2.0, 1.0, 2.0, 1e-12, 200).unwrap();
        assert!((root - std::f64::consts::SQRT_2).abs() < 1e-9);
    }

    #[test]
    fn cos_x_minus_x() {
        // Dottie number ≈ 0.7390851332151607
        let root = bisect(|x: f64| x.cos() - x, 0.0, 1.0, 1e-12, 200).unwrap();
        assert!((root - 0.739_085_133_215_160_6).abs() < 1e-9);
    }

    #[test]
    fn cubic_root_at_zero() {
        let root = bisect(|x: f64| x * x * x, -1.0, 1.0, 1e-12, 200).unwrap();
        assert!(root.abs() < 1e-9);
    }

    #[test]
    fn same_sign_endpoints_returns_none() {
        // x^2 + 1 is strictly positive everywhere.
        assert!(bisect(|x: f64| x * x + 1.0, -1.0, 1.0, 1e-12, 200).is_none());
        // f(x) = x^2 - 4 has f(0) = -4 and f(1) = -3 (same sign).
        assert!(bisect(|x: f64| x * x - 4.0, 0.0, 1.0, 1e-12, 200).is_none());
    }

    #[test]
    fn max_iter_zero_returns_none() {
        // Even with a valid bracket, zero iterations cannot converge.
        assert!(bisect(|x: f64| x * x - 2.0, 1.0, 2.0, 1e-12, 0).is_none());
    }

    #[test]
    fn endpoint_is_root() {
        // f(2) = 0 should be reported without iterating.
        let root = bisect(|x: f64| x - 2.0, 0.0, 2.0, 1e-12, 0).unwrap();
        assert!((root - 2.0).abs() < 1e-12);
    }

    #[test]
    fn swapped_bounds() {
        // Passing hi < lo should still work after the internal swap.
        let root = bisect(|x| x * x - 2.0, 2.0, 1.0, 1e-12, 200).unwrap();
        assert!((root - std::f64::consts::SQRT_2).abs() < 1e-9);
    }

    #[quickcheck]
    fn prop_linear_root(a: i16, b: i16) -> TestResult {
        // Solve a*x + b = 0 → x = -b / a, on a wide bracket containing it.
        if a == 0 {
            return TestResult::discard();
        }
        let a = f64::from(a);
        let b = f64::from(b);
        let expected = -b / a;

        // Build a bracket that strictly contains `expected` and is wide
        // enough to admit non-trivial bisection work.
        let lo = expected - 100.0;
        let hi = expected + 100.0;

        let tol = 1e-9;
        let root = match bisect(|x: f64| a * x + b, lo, hi, tol, 200) {
            Some(r) => r,
            None => return TestResult::failed(),
        };

        // Half-width termination guarantees |root - expected| <= tol; allow
        // a small slack for the |f(mid)| <= tol branch where the absolute
        // x-error is bounded by tol / |a|.
        let allowed = tol / a.abs() + tol;
        TestResult::from_bool((root - expected).abs() <= allowed.max(1e-6))
    }
}
