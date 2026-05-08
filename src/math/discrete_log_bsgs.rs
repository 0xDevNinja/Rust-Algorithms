//! Discrete logarithm via baby-step giant-step (Cyrk variant).
//!
//! Given `a`, `b`, `m`, finds the smallest non-negative `x` such that
//! `a^x ≡ b (mod m)`, or returns `None` when no such `x` exists.
//!
//! The classical baby-step giant-step works only when `gcd(a, m) = 1`.
//! For arbitrary moduli we apply the Cyrk-style preprocessing: while
//! `g = gcd(a, m) > 1`, divide out the common factor (this consumes one
//! step of the exponent) until the residual base is coprime with the
//! reduced modulus. Then the standard BSGS solves the lifted instance.
//!
//! # Complexity
//!
//! `O(sqrt(m) · log m)` time, `O(sqrt(m))` memory. Uses `u128`
//! intermediates so that modular multiplications are safe for any
//! `m` fitting in a `u64`.
//!
//! # Examples
//!
//! ```
//! use rust_algorithms::math::discrete_log_bsgs::discrete_log;
//!
//! // 2^3 = 8 ≡ 3 (mod 5).
//! assert_eq!(discrete_log(2, 3, 5), Some(3));
//!
//! // a^0 = 1, so b = 1 always has solution x = 0.
//! assert_eq!(discrete_log(7, 1, 11), Some(0));
//!
//! // 2^x is always even modulo 4, so 2^x ≡ 3 (mod 4) has no solution.
//! assert_eq!(discrete_log(2, 3, 4), None);
//! ```

use std::collections::HashMap;

use super::gcd_lcm::gcd;
use super::modular_exponentiation::mod_pow;

/// Returns the smallest non-negative `x` with `a^x ≡ b (mod m)`, or
/// `None` if no such `x` exists. Panics if `m == 0`.
pub fn discrete_log(a: u64, b: u64, m: u64) -> Option<u64> {
    assert!(m > 0, "modulus must be positive");
    if m == 1 {
        // Every integer is congruent to 0 mod 1, so a^0 = 1 ≡ 0 ≡ b.
        return Some(0);
    }

    let a_mod = a % m;
    let b_mod = b % m;

    // a^0 = 1 by convention (matches mod_pow). Catch this first so the
    // preprocessing loop does not advance past a valid x = 0 solution.
    if b_mod == 1 {
        return Some(0);
    }

    // Cyrk preprocessing: extract common factors of `a` and the modulus.
    // Each iteration uses the identity
    //     a · a^(x-1) ≡ b (mod m)  ⇒  (a/g) · a^(x-1) ≡ (b/g) (mod m/g)
    // when g = gcd(a, m) divides b. After k such reductions we have a
    // residual `c · a^y ≡ b' (mod m')` where `c` is the running product
    // of the `a/g` factors, `m'` is the reduced modulus, `b'` is `b`
    // reduced modulo `m'`, and the answer to the original problem (if
    // any) is `k + y`. The reduction terminates as soon as
    // gcd(a, m') = 1, after which the residual problem is solvable by
    // the classical baby-step giant-step.
    let mut k: u64 = 0;
    let mut m_prime = m;
    let mut c: u128 = 1 % (m_prime as u128);
    let mut b_residual = b_mod;

    loop {
        let g = gcd(a_mod, m_prime);
        if g == 1 {
            break;
        }
        if !b_residual.is_multiple_of(g) {
            // a^x is divisible by g for x >= 1, but the residual b is not.
            return None;
        }
        m_prime /= g;
        b_residual /= g;
        b_residual %= m_prime;
        c = (c * ((a_mod / g) as u128)) % (m_prime as u128);
        k += 1;
        // After this reduction, the equation at exponent y = 0 reads
        // c ≡ b_residual (mod m'); satisfying it means x = k works.
        if c == b_residual as u128 {
            return Some(k);
        }
    }

    // Now solve `c · a^y ≡ b_residual (mod m')` with gcd(a, m') = 1.
    // Write y = i·n - j with i ∈ [1, n], j ∈ [0, n). The congruence
    // becomes `c · a^(i·n) ≡ b_residual · a^j (mod m')`, i.e. the giant
    // step `c · a^(i·n)` must coincide with a baby step
    // `b_residual · a^j`. The final answer to the original equation is
    // `k + y`. (`y = 0` is handled before this loop runs.)
    let m_prime_u128 = m_prime as u128;
    // Ceiling of sqrt(m').
    let n = ((m_prime as f64).sqrt() as u64) + 1;

    // Baby steps: store b_residual · a^j -> j for j in [0, n). Keep the
    // largest j on collisions so that y = i·n - j is the smallest
    // non-negative exponent compatible with the matched giant step.
    let a_u128 = a_mod as u128;
    let mut table: HashMap<u64, u64> = HashMap::with_capacity(n as usize);
    let mut baby: u128 = b_residual as u128 % m_prime_u128;
    for j in 0..n {
        table.insert(baby as u64, j);
        baby = (baby * a_u128) % m_prime_u128;
    }

    // Giant step factor: a^n mod m'.
    let factor = mod_pow(a_mod, n, m_prime) as u128;
    // Probe values: c · (a^n)^i mod m', for i in [1, n].
    let mut probe: u128 = c % m_prime_u128;
    for i in 1..=n {
        probe = (probe * factor) % m_prime_u128;
        if let Some(&j) = table.get(&(probe as u64)) {
            let y = i * n - j;
            let x = y + k;
            // Verify against the original equation. The lifted residual
            // is congruence-equivalent to the original problem, but a
            // spurious y might appear when `c` is not invertible modulo
            // m' (it can fail to satisfy the original equation). The
            // explicit check guarantees correctness.
            if mod_pow(a, x, m) == b_mod {
                return Some(x);
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::discrete_log;
    use crate::math::modular_exponentiation::mod_pow;

    #[test]
    fn basic_small_prime() {
        // 2^3 = 8 ≡ 3 (mod 5).
        assert_eq!(discrete_log(2, 3, 5), Some(3));
    }

    #[test]
    fn classic_3_pow_x_eq_13_mod_17() {
        // 3^4 = 81 ≡ 81 - 4*17 = 13 (mod 17).
        assert_eq!(discrete_log(3, 13, 17), Some(4));
    }

    #[test]
    fn no_solution_case() {
        // 2^x is even for x >= 1 and equals 1 for x = 0; 3 is unreachable mod 4.
        assert_eq!(discrete_log(2, 3, 4), None);
    }

    #[test]
    fn b_equals_one_returns_zero() {
        assert_eq!(discrete_log(5, 1, 13), Some(0));
        assert_eq!(discrete_log(2, 1, 7), Some(0));
        // Even when gcd(a, m) > 1, b = 1 still admits x = 0.
        assert_eq!(discrete_log(6, 1, 9), Some(0));
    }

    #[test]
    fn modulus_one_is_always_zero() {
        assert_eq!(discrete_log(0, 0, 1), Some(0));
        assert_eq!(discrete_log(7, 5, 1), Some(0));
    }

    #[test]
    fn large_prime_modulus() {
        // m = 58231 is prime. Compute b = 5^12345 mod m and recover x via
        // discrete_log; the solver must return some y with 5^y ≡ b.
        let m = 58_231_u64;
        let b = mod_pow(5, 12_345, m);
        let x = discrete_log(5, b, m).expect("solution should exist for this prime modulus");
        assert_eq!(mod_pow(5, x, m), b);
    }

    #[test]
    fn composite_modulus_with_shared_factor() {
        // m = 12, a = 6: 6^1 = 6, 6^2 = 36 ≡ 0, 6^3 ≡ 0, ... so b = 0 yields x = 2.
        assert_eq!(discrete_log(6, 0, 12), Some(2));
        // 6^1 ≡ 6 (mod 12).
        assert_eq!(discrete_log(6, 6, 12), Some(1));
    }

    #[test]
    fn composite_modulus_no_solution_via_gcd_filter() {
        // gcd(4, 12) = 4 does not divide 5, and 4^0 = 1 ≠ 5.
        assert_eq!(discrete_log(4, 5, 12), None);
    }

    #[test]
    fn finds_smallest_non_negative_x() {
        // 1^x ≡ 1 (mod 7) for every x; the smallest is 0.
        assert_eq!(discrete_log(1, 1, 7), Some(0));
    }

    #[test]
    fn against_brute_force_small_cases() {
        // For small m, enumerate every (a, b) and compare with brute force.
        for m in 2_u64..40 {
            for a in 0_u64..m {
                let mut reachable: std::collections::HashMap<u64, u64> =
                    std::collections::HashMap::new();
                let mut value = 1_u64 % m;
                // Walk up to 2m steps; the powers cycle within m steps once
                // we enter the recurrent part, so this is enough to spot every
                // attainable residue and the first time it appears.
                for x in 0..(2 * m) {
                    reachable.entry(value).or_insert(x);
                    value = ((value as u128 * a as u128) % m as u128) as u64;
                }
                for b in 0_u64..m {
                    let expected = reachable.get(&b).copied();
                    let got = discrete_log(a, b, m);
                    match (expected, got) {
                        (Some(e), Some(g)) => {
                            assert_eq!(
                                mod_pow(a, g, m),
                                b % m,
                                "wrong x for a={a}, b={b}, m={m}: got {g}"
                            );
                            // Returned x must not exceed the smallest brute-force x.
                            assert!(
                                g <= e,
                                "non-minimal x for a={a}, b={b}, m={m}: got {g}, expected <= {e}"
                            );
                        }
                        (None, None) => {}
                        (e, g) => {
                            panic!("disagreement for a={a}, b={b}, m={m}: brute={e:?} got={g:?}")
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn property_random_small_primes() {
        // For each small prime p and a few generators g, pick exponents x in
        // [0, p) and verify discrete_log recovers some y with g^y ≡ g^x.
        let primes = [7_u64, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47];
        for &p in &primes {
            for g in 2..p {
                for x in 0..p {
                    let b = mod_pow(g, x, p);
                    let y = discrete_log(g, b, p)
                        .unwrap_or_else(|| panic!("expected solution for {g}^x ≡ {b} (mod {p})"));
                    assert_eq!(
                        mod_pow(g, y, p),
                        b,
                        "wrong recovery for g={g}, x={x}, p={p}: y={y}"
                    );
                }
            }
        }
    }
}
