//! Linear Diophantine equation solver.
//!
//! Solves `a·x + b·y = c` over the integers. Such an equation has integer
//! solutions iff `gcd(a, b) | c`. When it does, the extended Euclidean
//! algorithm yields a particular solution `(x0, y0)`, and every other
//! integer solution is obtained by walking along the kernel of the linear
//! form: for any integer `k`,
//!
//! ```text
//!     x = x0 + k · (b / g)
//!     y = y0 - k · (a / g)
//! ```
//!
//! where `g = gcd(a, b)`. The pair `(b/g, a/g)` is stored in [`DiophantineSolution`]
//! as the parametric step `(dx, dy)`, with the sign convention that all
//! solutions are `(x0 + k·dx, y0 - k·dy)`.
//!
//! Runtime is dominated by the single `O(log min(|a|, |b|))` extended
//! Euclidean call.
//!
//! # Degenerate cases
//!
//! * If `a == 0` and `b == 0`, the equation is `0 = c`. It is solvable iff
//!   `c == 0`, in which case every `(x, y)` is a solution; the chosen
//!   representative is `(x0, y0) = (0, 0)` with step `(dx, dy) = (1, 0)`,
//!   so `dy = 0` and incrementing `k` walks along the `x` axis. The
//!   parametric family does not generate every solution in this single
//!   degenerate case (it covers `{(k, 0)}` only); callers needing the full
//!   2D plane must handle `a == b == 0` themselves.
//! * If exactly one of `a`, `b` is zero, the formulas above still apply
//!   and the parametric step is `(0, ±1)` or `(±1, 0)` as appropriate.
//!
//! # Examples
//!
//! ```
//! use rust_algorithms::math::diophantine::solve;
//!
//! // 12·x + 18·y = 6, gcd = 6, divides 6.
//! let sol = solve(12, 18, 6).expect("solvable");
//! assert_eq!(12 * sol.x0 + 18 * sol.y0, 6);
//!
//! // gcd(4, 6) = 2 does not divide 5.
//! assert!(solve(4, 6, 5).is_none());
//! ```

use super::extended_euclidean::ext_gcd;

/// One particular solution `(x0, y0)` of `a·x + b·y = c`, together with
/// the parametric step `(dx, dy)` such that every integer solution has
/// the form `(x0 + k·dx, y0 - k·dy)` for some `k: i64`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DiophantineSolution {
    /// A particular `x` satisfying the equation.
    pub x0: i64,
    /// A particular `y` satisfying the equation.
    pub y0: i64,
    /// `b / gcd(a, b)`. Step added to `x0` per unit of `k`.
    pub dx: i64,
    /// `a / gcd(a, b)`. Step subtracted from `y0` per unit of `k`.
    pub dy: i64,
}

impl DiophantineSolution {
    /// Returns the solution at parameter `k`: `(x0 + k·dx, y0 - k·dy)`.
    pub const fn at(&self, k: i64) -> (i64, i64) {
        (self.x0 + k * self.dx, self.y0 - k * self.dy)
    }
}

/// Solves `a·x + b·y = c` over the integers.
///
/// Returns `Some(sol)` when `gcd(a, b)` divides `c`, and `None` otherwise.
/// The fully degenerate `a == b == 0` case is solvable iff `c == 0`; see
/// the module docs for the chosen representative.
pub const fn solve(a: i64, b: i64, c: i64) -> Option<DiophantineSolution> {
    if a == 0 && b == 0 {
        return if c == 0 {
            Some(DiophantineSolution {
                x0: 0,
                y0: 0,
                dx: 1,
                dy: 0,
            })
        } else {
            None
        };
    }

    let (g_signed, s, t) = ext_gcd(a, b);
    // ext_gcd may return a negative gcd when one of the inputs is negative;
    // normalize so `g` is positive, flipping the Bezout coefficients in lockstep.
    let (g, s, t) = if g_signed < 0 {
        (-g_signed, -s, -t)
    } else {
        (g_signed, s, t)
    };

    if c % g != 0 {
        return None;
    }

    let scale = c / g;
    Some(DiophantineSolution {
        x0: s * scale,
        y0: t * scale,
        dx: b / g,
        dy: a / g,
    })
}

#[cfg(test)]
mod tests {
    use super::{solve, DiophantineSolution};

    fn check_solution(a: i64, b: i64, c: i64, sol: &DiophantineSolution) {
        assert_eq!(
            a * sol.x0 + b * sol.y0,
            c,
            "particular solution failed for a={a}, b={b}, c={c}: {sol:?}"
        );
        for k in [-3_i64, -1, 0, 1, 3, 7] {
            let (x, y) = sol.at(k);
            assert_eq!(
                a * x + b * y,
                c,
                "k={k} solution failed for a={a}, b={b}, c={c}: ({x}, {y})"
            );
        }
    }

    #[test]
    fn classic_12x_plus_18y_eq_6() {
        let sol = solve(12, 18, 6).expect("solvable");
        check_solution(12, 18, 6, &sol);
        // Step direction: dx = 18/6 = 3, dy = 12/6 = 2.
        assert_eq!(sol.dx, 3);
        assert_eq!(sol.dy, 2);
    }

    #[test]
    fn no_solution_when_gcd_does_not_divide_c() {
        assert!(solve(4, 6, 5).is_none());
        assert!(solve(9, 6, 4).is_none());
        assert!(solve(-4, 6, 5).is_none());
    }

    #[test]
    fn coprime_coefficients_always_solvable() {
        let sol = solve(3, 5, 1).expect("gcd is 1");
        check_solution(3, 5, 1, &sol);

        let sol = solve(3, 5, 17).expect("gcd is 1");
        check_solution(3, 5, 17, &sol);
    }

    #[test]
    fn zero_a_with_b_dividing_c() {
        // 0·x + 5·y = 15 -> y = 3, x is free.
        let sol = solve(0, 5, 15).expect("5 divides 15");
        check_solution(0, 5, 15, &sol);
        assert_eq!(sol.y0, 3);
        // dy = a/g = 0, so y is fixed; dx = b/g = 1, so x walks freely.
        assert_eq!(sol.dy, 0);
        assert_eq!(sol.dx, 1);
    }

    #[test]
    fn zero_a_with_b_not_dividing_c() {
        assert!(solve(0, 5, 7).is_none());
    }

    #[test]
    fn zero_b_with_a_dividing_c() {
        // 4·x + 0·y = 12 -> x = 3, y is free.
        let sol = solve(4, 0, 12).expect("4 divides 12");
        check_solution(4, 0, 12, &sol);
        assert_eq!(sol.x0, 3);
        // dx = b/g = 0, dy = a/g = 1, so y walks freely as -k.
        assert_eq!(sol.dx, 0);
        assert_eq!(sol.dy, 1);
    }

    #[test]
    fn both_zero_with_c_zero_returns_origin() {
        let sol = solve(0, 0, 0).expect("trivially solvable");
        assert_eq!(sol.x0, 0);
        assert_eq!(sol.y0, 0);
    }

    #[test]
    fn both_zero_with_c_nonzero_is_none() {
        assert!(solve(0, 0, 1).is_none());
        assert!(solve(0, 0, -42).is_none());
    }

    #[test]
    fn negative_coefficients() {
        let sol = solve(-12, 18, 6).expect("solvable");
        check_solution(-12, 18, 6, &sol);

        let sol = solve(12, -18, -6).expect("solvable");
        check_solution(12, -18, -6, &sol);

        let sol = solve(-3, -5, -7).expect("gcd 1 divides -7");
        check_solution(-3, -5, -7, &sol);
    }

    #[test]
    fn large_coefficients() {
        let a = 1_000_003_i64;
        let b = 999_983_i64;
        let c = 12_345_i64;
        // a and b are both prime, so gcd = 1 and the equation is solvable.
        let sol = solve(a, b, c).expect("coprime primes");
        check_solution(a, b, c, &sol);

        // Multiples that share a common factor.
        let a2 = 2_000_000_i64;
        let b2 = 3_000_000_i64;
        let c2 = 1_000_000_i64; // gcd is 1_000_000, divides c.
        let sol = solve(a2, b2, c2).expect("solvable");
        check_solution(a2, b2, c2, &sol);
    }

    #[test]
    fn at_method_matches_explicit_formula() {
        let sol = solve(7, 11, 1).expect("gcd 1");
        for k in -10..=10 {
            let (x, y) = sol.at(k);
            assert_eq!(x, sol.x0 + k * sol.dx);
            assert_eq!(y, sol.y0 - k * sol.dy);
            assert_eq!(7 * x + 11 * y, 1);
        }
    }

    // Pseudo-random property test using a small linear-congruential sequence
    // so we stay within `i64` without overflow when checking `a·x + b·y`.
    #[test]
    fn property_random_inputs() {
        let mut state: u64 = 0x00C0_FFEE_BABE;
        let next = |s: &mut u64| -> i64 {
            *s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            // Map into a moderate range to keep products in range.
            ((*s >> 33) as i64 % 20_001) - 10_000
        };

        for _ in 0..500 {
            let a = next(&mut state);
            let b = next(&mut state);
            let c = next(&mut state);

            match solve(a, b, c) {
                Some(sol) => {
                    assert_eq!(
                        a * sol.x0 + b * sol.y0,
                        c,
                        "particular solution failed for a={a}, b={b}, c={c}"
                    );
                    for k in [-5_i64, -1, 0, 1, 4, 9] {
                        let (x, y) = sol.at(k);
                        assert_eq!(a * x + b * y, c, "k={k} failed for a={a}, b={b}, c={c}");
                    }
                }
                None => {
                    // No solution implies gcd(a, b) does not divide c (and
                    // a == b == 0 with c != 0 is the lone exception).
                    if a == 0 && b == 0 {
                        assert_ne!(c, 0);
                    } else {
                        let (g, _, _) = ext_gcd_abs(a, b);
                        assert!(c % g != 0, "expected gcd to not divide c, but g={g}, c={c}");
                    }
                }
            }
        }
    }

    fn ext_gcd_abs(a: i64, b: i64) -> (i64, i64, i64) {
        let (g, s, t) = super::ext_gcd(a, b);
        if g < 0 {
            (-g, -s, -t)
        } else {
            (g, s, t)
        }
    }
}
