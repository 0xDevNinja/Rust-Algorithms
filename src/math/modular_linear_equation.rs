//! Modular linear equation solver.
//!
//! Finds every `x` in `[0, m)` satisfying `a·x ≡ b (mod m)`.
//!
//! Let `g = gcd(a, m)`. The congruence has a solution iff `g | b`. When it
//! does, there are exactly `g` solutions modulo `m`: a particular solution
//! `x'` of `(a/g)·x ≡ (b/g) (mod m/g)` is recovered from the extended
//! Euclidean algorithm, and the full set is `{ x' + i·(m/g) : i = 0..g }`,
//! each reduced into `[0, m)` and returned in ascending order.
//!
//! Runtime is dominated by the `O(log m)` extended Euclidean call plus
//! `O(g)` work to enumerate solutions. The multi-solution case is the
//! reason for returning a `Vec<i64>` rather than an `Option<i64>`.
//!
//! # Behavior on non-positive moduli
//!
//! `m <= 0` is not a valid modulus for this routine and the function
//! returns an empty `Vec`.
//!
//! # Examples
//!
//! ```
//! use rust_algorithms::math::modular_linear_equation::solve;
//!
//! // 14·x ≡ 30 (mod 100): gcd(14, 100) = 2 divides 30, so two solutions.
//! assert_eq!(solve(14, 30, 100), vec![45, 95]);
//!
//! // gcd(2, 4) = 2 does not divide 3, so no solutions.
//! assert!(solve(2, 3, 4).is_empty());
//! ```

use super::extended_euclidean::ext_gcd;

/// Returns every `x` in `[0, m)` satisfying `a·x ≡ b (mod m)`, in
/// ascending order. Returns an empty `Vec` when no solution exists or
/// when `m <= 0`.
pub fn solve(a: i64, b: i64, m: i64) -> Vec<i64> {
    if m <= 0 {
        return Vec::new();
    }

    // Normalize a and b into [0, m) so subsequent arithmetic stays well-behaved.
    let a = a.rem_euclid(m);
    let b = b.rem_euclid(m);

    let (g, x0, _) = ext_gcd(a, m);
    // gcd from ext_gcd may be negative if a == 0 and m < 0, but we guard m > 0
    // and a is non-negative here, so g >= 0. When a == 0 and b == 0, g == m and
    // every x in [0, m) is a solution.
    if g == 0 {
        // a == 0 and m == 0 cannot occur (m > 0); a == 0 and b == 0 handled below.
        return (0..m).collect();
    }

    if b % g != 0 {
        return Vec::new();
    }

    let step = m / g;
    // Particular solution to a·x ≡ b (mod m): scale Bezout coefficient by b/g.
    let x_particular = (x0 * (b / g)).rem_euclid(step);

    let mut solutions: Vec<i64> = (0..g)
        .map(|i| (x_particular + i * step).rem_euclid(m))
        .collect();
    solutions.sort_unstable();
    solutions
}

#[cfg(test)]
mod tests {
    use super::solve;

    fn brute_force(a: i64, b: i64, m: i64) -> Vec<i64> {
        if m <= 0 {
            return Vec::new();
        }
        (0..m).filter(|x| (a * x - b).rem_euclid(m) == 0).collect()
    }

    #[test]
    fn classic_14x_eq_30_mod_100() {
        assert_eq!(solve(14, 30, 100), vec![45, 95]);
    }

    #[test]
    fn zero_a_zero_b_yields_full_residue_set() {
        let m = 7;
        assert_eq!(solve(0, 0, m), (0..m).collect::<Vec<_>>());
    }

    #[test]
    fn zero_a_nonzero_b_has_no_solution() {
        assert!(solve(0, 3, 10).is_empty());
    }

    #[test]
    fn identity_coefficient_gives_unique_solution() {
        for k in -20..20 {
            let m = 13;
            assert_eq!(solve(1, k, m), vec![k.rem_euclid(m)]);
        }
    }

    #[test]
    fn coprime_a_and_m_gives_unique_solution() {
        // 3·x ≡ 2 (mod 7); 3·3 = 9 ≡ 2 (mod 7), so x = 3.
        assert_eq!(solve(3, 2, 7), vec![3]);
    }

    #[test]
    fn b_not_divisible_by_gcd_is_empty() {
        // gcd(6, 9) = 3 does not divide 4.
        assert!(solve(6, 4, 9).is_empty());
    }

    #[test]
    fn negative_inputs_are_normalized() {
        // -14 ≡ 86 (mod 100), -30 ≡ 70 (mod 100); same solution set as classic case
        // mirrored: 14·x ≡ -30 (mod 100) ↔ x ≡ 5, 55 (mod 100).
        assert_eq!(solve(14, -30, 100), vec![5, 55]);
    }

    #[test]
    fn non_positive_modulus_returns_empty() {
        assert!(solve(1, 1, 0).is_empty());
        assert!(solve(1, 1, -7).is_empty());
    }

    #[test]
    fn quickcheck_against_brute_force() {
        for m in 1..=50_i64 {
            for a in -50..=50_i64 {
                for b in -50..=50_i64 {
                    let got = solve(a, b, m);
                    let expected = brute_force(a, b, m);
                    assert_eq!(
                        got, expected,
                        "mismatch for a={a}, b={b}, m={m}: got {got:?}, expected {expected:?}"
                    );
                }
            }
        }
    }
}
