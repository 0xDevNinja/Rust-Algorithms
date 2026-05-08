//! Simpson's rule — numerical integration of a real-valued function on a
//! finite interval.
//!
//! # Composite 1/3 rule
//! Partition `[a, b]` into `n` equal subintervals of width `h = (b - a) / n`
//! with `n` even, and let `x_k = a + k·h`. Then
//! ```text
//! ∫_a^b f(x) dx ≈ (h / 3) · [ f(x_0)
//!                            + 4·(f(x_1) + f(x_3) + … + f(x_{n-1}))
//!                            + 2·(f(x_2) + f(x_4) + … + f(x_{n-2}))
//!                            + f(x_n) ].
//! ```
//! The weights run `1, 4, 2, 4, …, 2, 4, 1`. The rule is exact for
//! polynomials of degree ≤ 3 because each pair of subintervals fits a
//! quadratic interpolant whose integral coincides with that of any cubic
//! sharing the same three samples.
//!
//! # Adaptive variant
//! `adaptive_simpson` recursively bisects `[a, b]`. On each subinterval `S`
//! it compares the single-panel Simpson estimate `S(a, b)` against the sum
//! of the two halves `S(a, m) + S(m, b)`; if they agree to within `15·tol`
//! (Richardson extrapolation gives the `1/15` factor for the leading O(h⁴)
//! error), the refined sum plus that error correction is accepted.
//! Otherwise the routine recurses, splitting `tol` evenly across halves
//! until `max_depth` is reached.
//!
//! # Complexity
//! - `simpson(f, a, b, n)`: `n + 1` function evaluations, `O(n)` time, `O(1)`
//!   auxiliary space.
//! - `adaptive_simpson(f, a, b, tol, max_depth)`: at most `O(2^max_depth)`
//!   evaluations in the worst case, but typically far fewer; `O(max_depth)`
//!   stack space.

/// Approximates `∫_a^b f(x) dx` using the composite Simpson's 1/3 rule with
/// `n` subintervals.
///
/// `n` must be even. If an odd `n` is supplied it is rounded up to the next
/// even value, and `n = 0` is promoted to `2` (the smallest legal panel
/// count). When `b < a`, the bounds are swapped internally and the result
/// is negated, so `simpson(f, a, b, n) == -simpson(f, b, a, n)`.
///
/// # Examples
/// ```
/// use rust_algorithms::math::simpson::simpson;
///
/// // ∫_0^1 x^2 dx = 1/3, exact since x^2 is degree 2 ≤ 3.
/// let approx = simpson(|x| x * x, 0.0, 1.0, 10);
/// assert!((approx - 1.0 / 3.0).abs() < 1e-12);
/// ```
pub fn simpson<F: Fn(f64) -> f64>(f: F, a: f64, b: f64, n: usize) -> f64 {
    // Empty interval: the integral is zero regardless of `f`.
    if a == b {
        return 0.0;
    }

    // Reversed bounds: integrate forwards and negate. This keeps the panel
    // step `h` positive and avoids the cancellations a negative `h` would
    // introduce.
    if b < a {
        return -simpson(f, b, a, n);
    }

    // Round `n` up to the next even value, with a minimum of 2.
    let n = if n < 2 {
        2
    } else if n % 2 == 1 {
        n + 1
    } else {
        n
    };

    let h = (b - a) / n as f64;
    let mut sum = f(a) + f(b);

    // Odd indices carry weight 4; even interior indices carry weight 2.
    for k in 1..n {
        let x = (k as f64).mul_add(h, a);
        let w: f64 = if k % 2 == 1 { 4.0 } else { 2.0 };
        sum = w.mul_add(f(x), sum);
    }

    sum * h / 3.0
}

/// One-panel Simpson estimate over `[lo, hi]` given the integrand sampled at
/// the endpoints and midpoint. Shared by [`adaptive_simpson`] across each
/// recursion level so the three function values survive a bisection step.
#[inline]
fn simpson_panel(flo: f64, fmid: f64, fhi: f64, lo: f64, hi: f64) -> f64 {
    (hi - lo) * 4.0_f64.mul_add(fmid, flo + fhi) / 6.0
}

/// State threaded through one bisection step of [`adaptive_simpson`]. Bundled
/// to keep the recursive helper's signature short and avoid the
/// `too_many_arguments` lint.
#[derive(Clone, Copy)]
struct PanelState {
    a: f64,
    b: f64,
    fa: f64,
    fb: f64,
    fm: f64,
    whole: f64,
}

fn adaptive_recurse<F: Fn(f64) -> f64>(f: &F, state: PanelState, tol: f64, depth: u32) -> f64 {
    let m = 0.5 * (state.a + state.b);
    let lm = 0.5 * (state.a + m);
    let rm = 0.5 * (m + state.b);
    let flm = f(lm);
    let frm = f(rm);
    let left = simpson_panel(state.fa, flm, state.fm, state.a, m);
    let right = simpson_panel(state.fm, frm, state.fb, m, state.b);
    let diff = left + right - state.whole;

    // Richardson: the refined estimate's error ≈ diff / 15.
    if depth == 0 || diff.abs() <= 15.0 * tol {
        left + right + diff / 15.0
    } else {
        let half = 0.5 * tol;
        let left_state = PanelState {
            a: state.a,
            b: m,
            fa: state.fa,
            fb: state.fm,
            fm: flm,
            whole: left,
        };
        let right_state = PanelState {
            a: m,
            b: state.b,
            fa: state.fm,
            fb: state.fb,
            fm: frm,
            whole: right,
        };
        adaptive_recurse(f, left_state, half, depth - 1)
            + adaptive_recurse(f, right_state, half, depth - 1)
    }
}

/// Approximates `∫_a^b f(x) dx` adaptively to within `tol`, recursively
/// bisecting until either each subinterval's Simpson estimate agrees with
/// the sum of its two halves to that tolerance or `max_depth` is exhausted.
///
/// The acceptance test uses Richardson extrapolation: the leading O(h⁴)
/// error in Simpson's rule cancels when comparing one panel to two, so the
/// difference between the coarse and refined estimates is `15` times the
/// remaining error of the refined estimate.
///
/// As with [`simpson`], swapping the bounds negates the result.
///
/// # Examples
/// ```
/// use rust_algorithms::math::simpson::adaptive_simpson;
///
/// let approx = adaptive_simpson(|x: f64| x.sin(), 0.0, std::f64::consts::PI, 1e-9, 20);
/// assert!((approx - 2.0).abs() < 1e-9);
/// ```
pub fn adaptive_simpson<F: Fn(f64) -> f64>(f: F, a: f64, b: f64, tol: f64, max_depth: u32) -> f64 {
    if a == b {
        return 0.0;
    }
    if b < a {
        return -adaptive_simpson(f, b, a, tol, max_depth);
    }

    let m = 0.5 * (a + b);
    let fa = f(a);
    let fb = f(b);
    let fm = f(m);
    let whole = simpson_panel(fa, fm, fb, a, b);
    let state = PanelState {
        a,
        b,
        fa,
        fb,
        fm,
        whole,
    };
    adaptive_recurse(&f, state, tol, max_depth)
}

#[cfg(test)]
mod tests {
    use super::{adaptive_simpson, simpson};
    use quickcheck_macros::quickcheck;
    use std::f64::consts::{E, PI};

    /// Tolerance for tests where Simpson is exact (polynomials of degree
    /// ≤ 3): allow only round-off-level slack.
    const EXACT_TOL: f64 = 1e-12;
    /// Tolerance for smooth non-polynomial integrands at moderate `n`.
    const SMOOTH_TOL: f64 = 1e-8;

    #[test]
    fn x_squared_over_unit_interval() {
        // ∫_0^1 x^2 dx = 1/3.
        let result = simpson(|x| x * x, 0.0, 1.0, 100);
        assert!((result - 1.0 / 3.0).abs() < EXACT_TOL);
    }

    #[test]
    fn sin_over_zero_to_pi() {
        // ∫_0^π sin(x) dx = 2.
        let result = simpson(f64::sin, 0.0, PI, 200);
        assert!((result - 2.0).abs() < SMOOTH_TOL);
    }

    #[test]
    fn exp_over_unit_interval() {
        // ∫_0^1 e^x dx = e − 1.
        let result = simpson(f64::exp, 0.0, 1.0, 100);
        assert!((result - (E - 1.0)).abs() < SMOOTH_TOL);
    }

    #[test]
    fn swapping_bounds_negates() {
        // ∫_a^b f = −∫_b^a f.
        let f = |x: f64| x.cos();
        let forward = simpson(f, 0.0, 1.5, 50);
        let backward = simpson(f, 1.5, 0.0, 50);
        assert!((forward + backward).abs() < EXACT_TOL);
    }

    #[test]
    fn equal_bounds_is_zero() {
        // Empty interval has zero measure for any integrand.
        assert_eq!(simpson(|x: f64| x.mul_add(x, 1.0), 2.5, 2.5, 10), 0.0);
    }

    #[test]
    fn n_zero_promoted_to_two() {
        // n = 0 must not divide-by-zero or panic; it should be promoted to
        // the smallest valid even count, n = 2.
        let result = simpson(|x| x * x * x, 0.0, 1.0, 0);
        // Cubics are exact under Simpson, so even with n = 2 we land on 1/4.
        assert!((result - 0.25).abs() < EXACT_TOL);
    }

    #[test]
    fn odd_n_rounded_up() {
        // n = 3 must behave the same as n = 4. Use a cubic so both are
        // exact and any difference would mean different rounding paths.
        let f = |x: f64| (2.0 * x * x).mul_add(x, -x + 5.0);
        let with_three = simpson(f, -1.0, 2.0, 3);
        let with_four = simpson(f, -1.0, 2.0, 4);
        assert!((with_three - with_four).abs() < EXACT_TOL);
    }

    #[test]
    fn constant_function() {
        // ∫_a^b c dx = c · (b − a).
        let result = simpson(|_| 7.0, -2.0, 3.0, 6);
        assert!((result - 35.0).abs() < EXACT_TOL);
    }

    #[test]
    fn cubic_is_exact_with_minimal_panels() {
        // The composite 1/3 rule integrates cubics exactly, even with the
        // smallest valid panel count n = 2.
        // ∫_0^2 (x^3 - 3x^2 + 2x - 1) dx
        //   = [x^4/4 - x^3 + x^2 - x] from 0 to 2 = 4 - 8 + 4 - 2 = -2.
        let f = |x: f64| (x * x).mul_add(x, (-3.0 * x).mul_add(x, 2.0f64.mul_add(x, -1.0)));
        let result = simpson(f, 0.0, 2.0, 2);
        assert!((result - (-2.0)).abs() < EXACT_TOL);
    }

    #[test]
    fn adaptive_matches_known_integrals() {
        let i_sin = adaptive_simpson(f64::sin, 0.0, PI, 1e-10, 20);
        assert!((i_sin - 2.0).abs() < 1e-9);

        let i_exp = adaptive_simpson(f64::exp, 0.0, 1.0, 1e-10, 20);
        assert!((i_exp - (E - 1.0)).abs() < 1e-9);
    }

    #[test]
    fn adaptive_swapping_bounds_negates() {
        let f = |x: f64| (x * x).sin();
        let forward = adaptive_simpson(f, 0.0, 1.0, 1e-9, 20);
        let backward = adaptive_simpson(f, 1.0, 0.0, 1e-9, 20);
        assert!((forward + backward).abs() < 1e-9);
    }

    /// Property: Simpson is exact for polynomials of degree ≤ 3, regardless
    /// of the (legal) panel count or the interval. We sample small integer
    /// coefficients so the closed-form integral stays well within `f64`'s
    /// dynamic range and round-off remains negligible.
    #[quickcheck]
    fn prop_exact_on_cubics(c0: i8, c1: i8, c2: i8, c3: i8, a_raw: i8, span: u8) -> bool {
        // Build a non-degenerate interval [a, b] = [a_raw, a_raw + span + 1].
        let a = f64::from(a_raw);
        let b = a + f64::from(u32::from(span) + 1);

        let c0 = f64::from(c0);
        let c1 = f64::from(c1);
        let c2 = f64::from(c2);
        let c3 = f64::from(c3);

        // Horner evaluation: ((c3·x + c2)·x + c1)·x + c0.
        let f = |x: f64| c3.mul_add(x, c2).mul_add(x, c1).mul_add(x, c0);

        // Closed-form antiderivative evaluated at b minus at a, also via
        // Horner: ((c3/4 · x + c2/3)·x + c1/2)·x · x + c0 · x.
        let antideriv = |x: f64| {
            (c3 / 4.0)
                .mul_add(x, c2 / 3.0)
                .mul_add(x, c1 / 2.0)
                .mul_add(x, c0)
                * x
        };
        let exact = antideriv(b) - antideriv(a);

        // The composite rule is exact for cubics, but each sample picks up
        // O(eps) of rounding; tolerate a tiny multiple of the result's
        // magnitude.
        let approx = simpson(f, a, b, 4);
        let scale = exact.abs().max(1.0);
        (approx - exact).abs() <= 1e-9 * scale
    }
}
