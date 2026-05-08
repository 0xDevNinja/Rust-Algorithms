//! Deterministic Miller-Rabin primality test for the full `u64` range.
//!
//! Uses the fixed witness set `[2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]`,
//! which is known (Sorenson & Webster, 2015) to deterministically classify
//! every 64-bit integer. Modular multiplications are performed in `u128` to
//! avoid overflow.
//!
//! - Time: `O(k · log^3 n)` with `k = 12` witnesses.
//! - Space: `O(1)`.

const WITNESSES: [u64; 12] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];

/// Returns `(base^exp) mod m` using `u128` intermediates. `m == 0` panics.
fn pow_mod(mut base: u64, mut exp: u64, m: u64) -> u64 {
    assert!(m > 0, "modulus must be positive");
    if m == 1 {
        return 0;
    }
    let modulus = u128::from(m);
    let mut result: u128 = 1;
    base = (u128::from(base) % modulus) as u64;
    while exp > 0 {
        let b = u128::from(base);
        if exp & 1 == 1 {
            result = (result * b) % modulus;
        }
        exp >>= 1;
        base = ((b * b) % modulus) as u64;
    }
    result as u64
}

/// Returns `true` iff `n` is prime. Deterministic across the full `u64` range.
///
/// - Time: `O(log^3 n)`.
/// - Space: `O(1)`.
pub fn is_prime(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    // Quick checks against the witness set itself.
    for &w in &WITNESSES {
        if n == w {
            return true;
        }
        if n.is_multiple_of(w) {
            return false;
        }
    }

    // Write n - 1 = d * 2^s with d odd.
    let mut d = n - 1;
    let mut s: u32 = 0;
    while d & 1 == 0 {
        d >>= 1;
        s += 1;
    }

    'witness: for &a in &WITNESSES {
        // Witnesses larger than or equal to n have already been handled above
        // (for n in the witness set we returned `true`; for other n smaller
        // than the largest witness, `n % w == 0` and `n != w` would have
        // returned `false`). So `a < n` holds here.
        let mut x = pow_mod(a, d, n);
        if x == 1 || x == n - 1 {
            continue;
        }
        for _ in 0..s - 1 {
            x = pow_mod(x, 2, n);
            if x == n - 1 {
                continue 'witness;
            }
        }
        return false;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::{is_prime, pow_mod};
    use quickcheck_macros::quickcheck;

    fn trial_division(n: u64) -> bool {
        if n < 2 {
            return false;
        }
        if n < 4 {
            return true;
        }
        if n.is_multiple_of(2) {
            return false;
        }
        let mut p: u64 = 3;
        while p.saturating_mul(p) <= n {
            if n.is_multiple_of(p) {
                return false;
            }
            p += 2;
        }
        true
    }

    #[test]
    fn edge_cases() {
        assert!(!is_prime(0));
        assert!(!is_prime(1));
        assert!(is_prime(2));
        assert!(is_prime(3));
        assert!(!is_prime(4));
    }

    #[test]
    fn small_primes() {
        for &p in &[
            2_u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 97,
        ] {
            assert!(is_prime(p), "{p} should be prime");
        }
    }

    #[test]
    fn small_composites() {
        for &c in &[4_u64, 6, 8, 9, 10, 15, 21, 25, 27, 33, 35, 49, 51, 100] {
            assert!(!is_prime(c), "{c} should be composite");
        }
    }

    #[test]
    fn carmichael_numbers() {
        // Carmichael numbers fool naive Fermat tests but Miller-Rabin catches them.
        for &c in &[561_u64, 1105, 1729, 2465, 2821, 6601, 8911, 41_041] {
            assert!(!is_prime(c), "Carmichael {c} should be composite");
        }
    }

    #[test]
    fn large_primes() {
        // 2^61 - 1 is the 9th Mersenne prime.
        assert!(is_prime(2_305_843_009_213_693_951));
        // 10^18 + 9 is prime.
        assert!(is_prime(1_000_000_000_000_000_009));
        // 2^31 - 1 (Mersenne).
        assert!(is_prime(2_147_483_647));
        // 10^9 + 7, the canonical CP modulus.
        assert!(is_prime(1_000_000_007));
    }

    #[test]
    fn large_composites() {
        // (10^9 + 7) * (10^9 + 9) — product of two large primes.
        assert!(!is_prime(1_000_000_007 * 1_000_000_009));
        // 2^61 - 1 squared overflows u64; instead use a Mersenne composite.
        // 2^11 - 1 = 2047 = 23 * 89, composite though Mersenne.
        assert!(!is_prime(2047));
        // 25_326_001 — strong pseudoprime to bases 2, 3, 5.
        assert!(!is_prime(25_326_001));
        // 3_215_031_751 — strong pseudoprime to bases 2, 3, 5, 7.
        assert!(!is_prime(3_215_031_751));
    }

    #[test]
    fn pow_mod_basic() {
        assert_eq!(pow_mod(2, 10, 1000), 24);
        assert_eq!(pow_mod(0, 0, 7), 1);
        assert_eq!(pow_mod(5, 0, 7), 1);
        assert_eq!(pow_mod(7, 4, 1), 0);
    }

    #[test]
    fn matches_trial_division_exhaustive() {
        for n in 0_u64..10_000 {
            assert_eq!(is_prime(n), trial_division(n), "mismatch at n = {n}");
        }
    }

    #[allow(clippy::needless_pass_by_value)]
    #[quickcheck]
    fn matches_trial_division_quickcheck(n: u16) -> bool {
        let n = u64::from(n);
        is_prime(n) == trial_division(n)
    }
}
