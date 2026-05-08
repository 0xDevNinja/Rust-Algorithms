//! Newton's method (Newton–Raphson) for root finding on real-valued
//! differentiable functions, plus an integer-square-root variant using
//! Heron's iteration on integers.
//!
//! Given `f` and its derivative `f'`, Newton's method iterates
//! `x ← x - f(x)/f'(x)` until `|f(x)| ≤ tol`. Convergence is quadratic
//! near a simple root but the method can diverge or stall when the
//! derivative approaches zero or the starting point is poorly chosen.
//!
//! Complexity: `O(max_iter)` evaluations of `f` and `f'` for the floating
//! point version. `integer_sqrt` runs in `O(log n)` iterations.

/// Threshold below which the derivative is considered numerically zero.
const DERIV_EPS: f64 = 1e-300;

/// Newton's method root finder.
///
/// Iterates `x ← x - f(x)/f'(x)` from the initial guess `x` until either
/// `|f(x)| ≤ tol` (success) or `max_iter` iterations have elapsed without
/// convergence. Returns `None` if the derivative becomes too close to
/// zero at any step (`|f'(x)| < DERIV_EPS`) or if the iteration fails to
/// converge within `max_iter` steps.
pub fn newton<F, FP>(f: F, fp: FP, mut x: f64, tol: f64, max_iter: u32) -> Option<f64>
where
    F: Fn(f64) -> f64,
    FP: Fn(f64) -> f64,
{
    for _ in 0..max_iter {
        let fx = f(x);
        if fx.abs() <= tol {
            return Some(x);
        }
        let dfx = fp(x);
        if dfx.abs() < DERIV_EPS {
            return None;
        }
        x -= fx / dfx;
        if !x.is_finite() {
            return None;
        }
    }
    // Final tolerance check after the last update.
    if f(x).abs() <= tol {
        Some(x)
    } else {
        None
    }
}

/// Exact floor of the square root of `n` using Heron's iteration on
/// integers. Returns `⌊√n⌋` for any `u64`, including `u64::MAX`.
pub const fn integer_sqrt(n: u64) -> u64 {
    if n < 2 {
        return n;
    }
    // Initial guess: 2^(ceil(bits(n) / 2)) is an upper bound on sqrt(n).
    let bits = 64 - n.leading_zeros();
    let shift = bits.div_ceil(2);
    let mut x = 1_u64 << shift;
    loop {
        // Heron step: y = (x + n/x) / 2 written via `midpoint` to avoid
        // any intermediate overflow when x is near u64::MAX.
        let y = u64::midpoint(x, n / x);
        if y >= x {
            return x;
        }
        x = y;
    }
}

#[cfg(test)]
mod tests {
    use super::{integer_sqrt, newton};
    use quickcheck_macros::quickcheck;

    #[test]
    fn newton_sqrt_two() {
        // f(x) = x^2 - 2, f'(x) = 2x. Root at sqrt(2).
        let r = newton(|x| x.mul_add(x, -2.0), |x| 2.0 * x, 1.0, 1e-12, 100).unwrap();
        assert!((r - 2.0_f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn newton_no_real_root() {
        // f(x) = x^2 + 1 has no real root; iteration cannot reach |f| ≤ tol.
        let r = newton(|x| x.mul_add(x, 1.0), |x| 2.0 * x, 1.0, 1e-12, 100);
        assert!(r.is_none());
    }

    #[test]
    fn newton_cosine_root() {
        // cos(x) has a root at π/2 starting from x = 1.
        let r = newton(f64::cos, |x| -x.sin(), 1.0, 1e-12, 100).unwrap();
        assert!((r - std::f64::consts::FRAC_PI_2).abs() < 1e-12);
    }

    #[test]
    fn newton_zero_derivative_bails() {
        // f(x) = x^3 - x. f(0) = 0 already, so newton returns Some(0.0)
        // immediately without ever inspecting the derivative.
        let r = newton(
            |x| (x * x).mul_add(x, -x),
            |x| (3.0 * x).mul_add(x, -1.0),
            0.0,
            1e-12,
            100,
        );
        assert_eq!(r, Some(0.0));
    }

    #[test]
    fn newton_stalled_derivative_returns_none() {
        // f'(x) = 0 everywhere with f nonzero: should fail fast.
        let r = newton(|_| 1.0, |_| 0.0, 1.0, 1e-12, 100);
        assert!(r.is_none());
    }

    #[test]
    fn integer_sqrt_small() {
        assert_eq!(integer_sqrt(0), 0);
        assert_eq!(integer_sqrt(1), 1);
        assert_eq!(integer_sqrt(2), 1);
        assert_eq!(integer_sqrt(3), 1);
        assert_eq!(integer_sqrt(4), 2);
        assert_eq!(integer_sqrt(99), 9);
        assert_eq!(integer_sqrt(100), 10);
        assert_eq!(integer_sqrt(101), 10);
    }

    #[test]
    fn integer_sqrt_u64_max() {
        // ⌊√(2^64 - 1)⌋ = 2^32 - 1 = 4294967295.
        assert_eq!(integer_sqrt(u64::MAX), 4_294_967_295);
    }

    #[test]
    fn integer_sqrt_perfect_squares() {
        for k in 0_u64..1000 {
            assert_eq!(integer_sqrt(k * k), k);
        }
    }

    #[quickcheck]
    fn prop_integer_sqrt_bracket(n: u64) -> bool {
        // Clamp to avoid (s+1)^2 overflow when n is near u64::MAX.
        let n = n % (u64::MAX / 4);
        let s = integer_sqrt(n);
        s * s <= n && n < (s + 1) * (s + 1)
    }
}
