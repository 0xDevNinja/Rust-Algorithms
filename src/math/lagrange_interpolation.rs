//! Lagrange polynomial interpolation over `f64`.
//!
//! Given `n` distinct sample points `(xs[i], ys[i])`, there is a unique
//! polynomial of degree at most `n - 1` that passes through all of them.
//! This module evaluates that polynomial at an arbitrary point `x` using the
//! direct Lagrange basis form
//!
//! ```text
//!   p(x) = Σ_i  ys[i] · Π_{j ≠ i} (x - xs[j]) / (xs[i] - xs[j]).
//! ```
//!
//! It also exposes a helper that returns the polynomial in monomial-basis
//! coefficients `[c_0, c_1, ..., c_{n-1}]` representing
//! `c_0 + c_1·x + ... + c_{n-1}·x^{n-1}`.
//!
//! # Complexity
//! `lagrange_eval`: O(n²) time, O(1) extra space per call.
//! `lagrange_coefficients`: O(n²) time and space.
//!
//! # Preconditions
//! All `xs` must be pairwise distinct; `xs.len() == ys.len()`. Both functions
//! `debug_assert!` these invariants and will panic in release builds via
//! division-by-zero if duplicates slip through.

/// Evaluates the unique degree-`(n - 1)` polynomial through the points
/// `(xs[i], ys[i])` at `x`.
///
/// # Panics
/// Panics on length mismatch (`xs.len() != ys.len()`) and, with debug
/// assertions enabled, on duplicate `xs`. With an empty input the function
/// returns `0.0` (the unique constant zero polynomial through no points).
pub fn lagrange_eval(xs: &[f64], ys: &[f64], x: f64) -> f64 {
    assert_eq!(
        xs.len(),
        ys.len(),
        "lagrange_eval: xs and ys must have the same length"
    );
    debug_assert!(distinct(xs), "lagrange_eval: xs must be pairwise distinct");

    let n = xs.len();
    let mut result = 0.0_f64;
    for i in 0..n {
        let mut term = ys[i];
        for j in 0..n {
            if j == i {
                continue;
            }
            term *= (x - xs[j]) / (xs[i] - xs[j]);
        }
        result += term;
    }
    result
}

/// Returns the monomial-basis coefficients `[c_0, ..., c_{n-1}]` of the unique
/// degree-`(n - 1)` interpolating polynomial. The polynomial evaluated at `x`
/// equals `c_0 + c_1·x + c_2·x² + ... + c_{n-1}·x^{n-1}`.
///
/// For `n == 0` returns the empty vector (the zero polynomial has no
/// coefficients in this representation).
///
/// # Panics
/// Panics on length mismatch; debug-asserts pairwise distinctness of `xs`.
pub fn lagrange_coefficients(xs: &[f64], ys: &[f64]) -> Vec<f64> {
    assert_eq!(
        xs.len(),
        ys.len(),
        "lagrange_coefficients: xs and ys must have the same length"
    );
    debug_assert!(
        distinct(xs),
        "lagrange_coefficients: xs must be pairwise distinct"
    );

    let n = xs.len();
    if n == 0 {
        return Vec::new();
    }

    let mut coeffs = vec![0.0_f64; n];

    for i in 0..n {
        // Build the Lagrange basis polynomial L_i(x) = Π_{j ≠ i} (x - xs[j]) / (xs[i] - xs[j])
        // as a vector of monomial coefficients of length `n` (degree n-1).
        let mut basis = vec![0.0_f64; n];
        basis[0] = 1.0;
        let mut deg = 0_usize; // current degree of `basis`
        let mut denom = 1.0_f64;

        for j in 0..n {
            if j == i {
                continue;
            }
            // Multiply current polynomial by (x - xs[j]):
            //   new[k] = old[k - 1] - xs[j] * old[k]
            // Walk from high to low so we don't overwrite values we still need.
            let xj = xs[j];
            deg += 1;
            // Shift up by one: new[k] = old[k-1], for k = deg..=1, then subtract xj*old[k].
            for k in (1..=deg).rev() {
                basis[k] = (-xj).mul_add(basis[k], basis[k - 1]);
            }
            basis[0] *= -xj;

            denom *= xs[i] - xs[j];
        }

        let scale = ys[i] / denom;
        for k in 0..n {
            coeffs[k] += scale * basis[k];
        }
    }

    coeffs
}

/// Returns `true` iff every value in `xs` is distinct. O(n²) but only used
/// inside `debug_assert!` so this never runs in release builds.
fn distinct(xs: &[f64]) -> bool {
    for i in 0..xs.len() {
        for j in (i + 1)..xs.len() {
            if xs[i] == xs[j] {
                return false;
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::{lagrange_coefficients, lagrange_eval};

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() <= eps * (1.0 + a.abs().max(b.abs()))
    }

    #[test]
    fn empty_returns_zero() {
        // No samples -> the (degenerate) zero polynomial.
        assert_eq!(lagrange_eval(&[], &[], 2.5), 0.0);
        assert!(lagrange_coefficients(&[], &[]).is_empty());
    }

    #[test]
    fn single_point_is_constant() {
        let xs = [2.5];
        let ys = [-7.0];
        for &x in &[-1.0, 0.0, 2.5, 100.0] {
            assert_eq!(lagrange_eval(&xs, &ys, x), -7.0);
        }
        let coeffs = lagrange_coefficients(&xs, &ys);
        assert_eq!(coeffs, vec![-7.0]);
    }

    #[test]
    fn two_points_form_a_line() {
        // y = 2x + 1 through (0, 1) and (3, 7)
        let xs = [0.0, 3.0];
        let ys = [1.0, 7.0];
        for &x in &[-2.0_f64, 0.0, 1.5, 3.0, 10.0] {
            let expected = 2.0_f64.mul_add(x, 1.0);
            assert!(approx_eq(lagrange_eval(&xs, &ys, x), expected, 1e-12));
        }
        let coeffs = lagrange_coefficients(&xs, &ys);
        assert!(approx_eq(coeffs[0], 1.0, 1e-12));
        assert!(approx_eq(coeffs[1], 2.0, 1e-12));
    }

    #[test]
    fn three_collinear_points_recover_line() {
        // y = -x + 4 sampled at three points -- the interpolant must still be linear.
        let xs = [-1.0, 2.0, 5.0];
        let ys = [5.0, 2.0, -1.0];
        for &x in &[-3.0, 0.0, 1.0, 4.5, 10.0] {
            let expected = -x + 4.0;
            assert!(approx_eq(lagrange_eval(&xs, &ys, x), expected, 1e-12));
        }
        // Quadratic coefficient should be (numerically) zero.
        let coeffs = lagrange_coefficients(&xs, &ys);
        assert!(coeffs[2].abs() < 1e-12, "got {}", coeffs[2]);
        assert!(approx_eq(coeffs[1], -1.0, 1e-12));
        assert!(approx_eq(coeffs[0], 4.0, 1e-12));
    }

    #[test]
    fn quadratic_recovery() {
        // p(x) = x^2, sampled at three distinct points.
        let xs = [-2.0, 0.0, 3.0];
        let ys = [4.0, 0.0, 9.0];
        for &x in &[-5.0, -1.5, 0.0, 1.0, 2.0, 7.0] {
            assert!(approx_eq(lagrange_eval(&xs, &ys, x), x * x, 1e-10));
        }
        let coeffs = lagrange_coefficients(&xs, &ys);
        assert!(approx_eq(coeffs[0], 0.0, 1e-12));
        assert!(approx_eq(coeffs[1], 0.0, 1e-12));
        assert!(approx_eq(coeffs[2], 1.0, 1e-12));
    }

    #[test]
    fn cubic_recovery_through_four_points() {
        // p(x) = 2x^3 - x^2 + 5
        let p = |x: f64| (2.0_f64 * x).mul_add(x * x, (-x).mul_add(x, 5.0));
        let xs = [-1.0, 0.0, 2.0, 3.5];
        let ys = [p(-1.0), p(0.0), p(2.0), p(3.5)];
        for &x in &[-3.0, -0.5, 1.25, 4.0, 10.0] {
            assert!(approx_eq(lagrange_eval(&xs, &ys, x), p(x), 1e-8));
        }
        let coeffs = lagrange_coefficients(&xs, &ys);
        assert!(approx_eq(coeffs[0], 5.0, 1e-10));
        assert!(coeffs[1].abs() < 1e-10);
        assert!(approx_eq(coeffs[2], -1.0, 1e-10));
        assert!(approx_eq(coeffs[3], 2.0, 1e-10));
    }

    #[test]
    #[should_panic(expected = "same length")]
    fn length_mismatch_panics() {
        let xs = [1.0, 2.0];
        let ys = [3.0];
        let _ = lagrange_eval(&xs, &ys, 0.0);
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "pairwise distinct")]
    fn duplicate_xs_panics_in_debug() {
        let xs = [1.0, 2.0, 1.0];
        let ys = [0.0, 0.0, 0.0];
        let _ = lagrange_eval(&xs, &ys, 0.5);
    }

    // ── property-based tests ─────────────────────────────────────────────────

    #[cfg(test)]
    mod property {
        use super::super::{lagrange_coefficients, lagrange_eval};
        use super::approx_eq;
        use quickcheck_macros::quickcheck;

        /// Pick a deterministic-but-varied degree-3 polynomial from four `i16`
        /// inputs, sample it at four fixed-but-distinct points, and verify
        /// `lagrange_eval` reproduces it at fresh test points.
        #[quickcheck]
        fn matches_random_cubic(a: i16, b: i16, c: i16, d: i16) -> bool {
            // Keep coefficients small so the resulting polynomial's values stay well
            // within f64 precision; the test is about correctness not magnitude.
            let a = f64::from(a) / 1000.0;
            let b = f64::from(b) / 1000.0;
            let c = f64::from(c) / 1000.0;
            let d = f64::from(d) / 1000.0;

            let p = |x: f64| {
                let t = a.mul_add(x, b);
                let t = t.mul_add(x, c);
                t.mul_add(x, d)
            };

            let xs = [-2.5_f64, -0.75, 1.25, 3.0];
            let ys = [p(xs[0]), p(xs[1]), p(xs[2]), p(xs[3])];

            // Verify at a few points distinct from the sample nodes.
            let probes = [-4.0_f64, -1.0, 0.0, 0.5, 2.0, 5.5];
            for &x in &probes {
                let got = lagrange_eval(&xs, &ys, x);
                let want = p(x);
                if !approx_eq(got, want, 1e-9) {
                    return false;
                }
            }

            // Also sanity-check the monomial coefficients reproduce the polynomial.
            let coeffs = lagrange_coefficients(&xs, &ys);
            for &x in &probes {
                let mut acc = 0.0_f64;
                for &k in coeffs.iter().rev() {
                    acc = acc.mul_add(x, k);
                }
                if !approx_eq(acc, p(x), 1e-8) {
                    return false;
                }
            }
            true
        }
    }
}
