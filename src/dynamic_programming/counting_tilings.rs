//! Counting domino tilings of fixed-width grids by dynamic programming.
//!
//! Each function below evaluates a constant-coefficient linear recurrence
//! iteratively in O(n) time and O(1) extra space (a small rolling window of
//! the last few terms). All accumulators are `u128`; the sequences grow
//! exponentially, so the result eventually overflows even `u128`. Concrete
//! safe-input bounds:
//!
//! * `tilings_2xn` — the 2×n domino-tiling count is the (n+1)-th Fibonacci
//!   number, which fits in `u128` for `n ≤ 184`.
//! * `tilings_4xn` — OEIS A005178; fits in `u128` for `n ≤ 96`.
//! * `tilings_2xn_with_trominoes` — fits in `u128` for `n ≤ 124`.
//!
//! Beyond those bounds the functions panic on overflow in debug builds and
//! wrap in release builds, the standard Rust integer behaviour. Callers that
//! need larger `n` should switch to a `BigUint` or a modular variant.

/// Number of ways to tile a 2×`n` grid with 1×2 dominoes.
///
/// Recurrence: `f(0) = 1`, `f(1) = 1`, `f(n) = f(n - 1) + f(n - 2)` — the
/// shifted Fibonacci sequence. Runs in O(n) time and O(1) space.
///
/// Stays within `u128` for `n ≤ 184`.
#[must_use]
pub fn tilings_2xn(n: u64) -> u128 {
    let (mut a, mut b): (u128, u128) = (1, 1);
    for _ in 0..n {
        let next = a + b;
        a = b;
        b = next;
    }
    a
}

/// Number of ways to tile a 4×`n` grid with 1×2 dominoes (OEIS A005178).
///
/// Recurrence: `f(n) = f(n - 1) + 5·f(n - 2) + f(n - 3) − f(n - 4)` with base
/// cases `f(0) = 1`, `f(1) = 1`, `f(2) = 5`, `f(3) = 11`. Runs in O(n) time
/// and O(1) space using a length-4 rolling window.
///
/// Stays within `u128` for `n ≤ 96`.
#[must_use]
pub fn tilings_4xn(n: u64) -> u128 {
    if n == 0 {
        return 1;
    }
    if n == 1 {
        return 1;
    }
    if n == 2 {
        return 5;
    }
    if n == 3 {
        return 11;
    }
    // Rolling window: (f(k-4), f(k-3), f(k-2), f(k-1)) for k starting at 4.
    let (mut a, mut b, mut c, mut d): (u128, u128, u128, u128) = (1, 1, 5, 11);
    for _ in 4..=n {
        // f(k) = d + 5·c + b − a. The recurrence guarantees a ≤ d + 5·c + b
        // for all k ≥ 4, so the subtraction never underflows.
        let next = d + 5 * c + b - a;
        a = b;
        b = c;
        c = d;
        d = next;
    }
    d
}

/// Number of ways to tile a 2×`n` grid using 1×2 dominoes **and** L-trominoes
/// (OEIS A001835 / A030186-style mix; here the recurrence
/// `f(n) = 2·f(n - 1) + f(n - 3)` with `f(0) = 1`, `f(1) = 1`, `f(2) = 2`).
///
/// Runs in O(n) time and O(1) space using a length-3 rolling window. Stays
/// within `u128` for `n ≤ 124`.
#[must_use]
pub fn tilings_2xn_with_trominoes(n: u64) -> u128 {
    if n == 0 {
        return 1;
    }
    if n == 1 {
        return 1;
    }
    if n == 2 {
        return 2;
    }
    let (mut a, mut b, mut c): (u128, u128, u128) = (1, 1, 2);
    for _ in 3..=n {
        let next = 2 * c + a;
        a = b;
        b = c;
        c = next;
    }
    c
}

#[cfg(test)]
mod tests {
    use super::{tilings_2xn, tilings_2xn_with_trominoes, tilings_4xn};
    use quickcheck_macros::quickcheck;

    #[test]
    fn tilings_2xn_known_small_values() {
        // f(n) for n = 0..=10 — Fibonacci shifted.
        let expected: [u128; 11] = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89];
        for (n, &v) in expected.iter().enumerate() {
            assert_eq!(tilings_2xn(n as u64), v, "tilings_2xn({n})");
        }
    }

    #[test]
    fn tilings_2xn_zero_is_one() {
        // The empty grid has one (empty) tiling.
        assert_eq!(tilings_2xn(0), 1);
    }

    #[test]
    fn tilings_4xn_known_small_values() {
        // OEIS A005178: 1, 1, 5, 11, 36, 95, 281, 781, 2245, 6336, ...
        let expected: [u128; 10] = [1, 1, 5, 11, 36, 95, 281, 781, 2245, 6336];
        for (n, &v) in expected.iter().enumerate() {
            assert_eq!(tilings_4xn(n as u64), v, "tilings_4xn({n})");
        }
    }

    #[test]
    fn tilings_4xn_zero_is_one() {
        assert_eq!(tilings_4xn(0), 1);
    }

    #[test]
    fn tilings_2xn_with_trominoes_known_small_values() {
        // f(0)=1, f(1)=1, f(2)=2, f(3)=2·2+1=5, f(4)=2·5+1=11, f(5)=2·11+2=24,
        // f(6)=2·24+5=53, f(7)=2·53+11=117.
        let expected: [u128; 8] = [1, 1, 2, 5, 11, 24, 53, 117];
        for (n, &v) in expected.iter().enumerate() {
            assert_eq!(
                tilings_2xn_with_trominoes(n as u64),
                v,
                "tilings_2xn_with_trominoes({n})"
            );
        }
    }

    /// Independent Fibonacci computation for cross-checking `tilings_2xn`.
    fn fib_shifted(n: u64) -> u128 {
        let (mut a, mut b): (u128, u128) = (1, 1);
        for _ in 0..n {
            let next = a + b;
            a = b;
            b = next;
        }
        a
    }

    #[allow(clippy::needless_pass_by_value)]
    #[quickcheck]
    fn tilings_2xn_matches_independent_fibonacci(n_pick: u8) -> bool {
        let n = u64::from(n_pick % 31); // n ∈ [0, 30]
        tilings_2xn(n) == fib_shifted(n)
    }

    #[allow(clippy::needless_pass_by_value)]
    #[quickcheck]
    fn tilings_2xn_recurrence_holds(n_pick: u8) -> bool {
        let n = u64::from(n_pick % 29) + 2; // n ∈ [2, 30]
        tilings_2xn(n) == tilings_2xn(n - 1) + tilings_2xn(n - 2)
    }

    #[allow(clippy::needless_pass_by_value)]
    #[quickcheck]
    fn tilings_4xn_recurrence_holds(n_pick: u8) -> bool {
        let n = u64::from(n_pick % 27) + 4; // n ∈ [4, 30]
        let lhs = tilings_4xn(n) + tilings_4xn(n - 4);
        let rhs = tilings_4xn(n - 1) + 5 * tilings_4xn(n - 2) + tilings_4xn(n - 3);
        lhs == rhs
    }

    #[allow(clippy::needless_pass_by_value)]
    #[quickcheck]
    fn tilings_2xn_with_trominoes_recurrence_holds(n_pick: u8) -> bool {
        let n = u64::from(n_pick % 28) + 3; // n ∈ [3, 30]
        tilings_2xn_with_trominoes(n)
            == 2 * tilings_2xn_with_trominoes(n - 1) + tilings_2xn_with_trominoes(n - 3)
    }
}
