//! Euler's totient function `phi(n)`, counting integers in `[1, n]` coprime
//! to `n`. Single-value evaluation runs in `O(sqrt(n))` via trial-division
//! prime factorisation, while `phi_sieve` computes `phi(0..=n)` in
//! `O(n log log n)` using an Eratosthenes-style traversal. Foundational for
//! Euler's theorem (`a^phi(n) ≡ 1 (mod n)` for `gcd(a, n) = 1`) and the
//! Euler-Fermat / RSA family of identities.

/// Returns `phi(n)`, the count of integers in `[1, n]` coprime to `n`.
///
/// Implementation: extract each distinct prime `p` of `n` by trial division
/// up to `sqrt(n)`, applying `phi *= (p - 1) / p` (carried as
/// `phi -= phi / p`). Any prime factor greater than `sqrt(n)` remains as the
/// reduced `n` after the loop and contributes a final factor.
///
/// Edge case: `phi(0)` is conventionally `0` (the empty range has no
/// elements). `phi(1) = 1`.
pub const fn phi(mut n: u64) -> u64 {
    if n == 0 {
        return 0;
    }
    let mut result = n;
    let mut p: u64 = 2;
    while p.saturating_mul(p) <= n {
        if n.is_multiple_of(p) {
            while n.is_multiple_of(p) {
                n /= p;
            }
            result -= result / p;
        }
        p += 1;
    }
    if n > 1 {
        result -= result / n;
    }
    result
}

/// Returns `phi(0..=n)` as a vector of length `n + 1` using a linear-style
/// sieve. Initialises `phi[i] = i`, then for each prime `p` (detected when
/// `phi[p]` is still `p`) updates every multiple `k` of `p` via
/// `phi[k] = phi[k] / p * (p - 1)`. By convention `phi[0] = 0` and
/// `phi[1] = 1`.
#[must_use]
pub fn phi_sieve(n: usize) -> Vec<u64> {
    let mut phi: Vec<u64> = (0..=n as u64).collect();
    if n >= 1 {
        phi[1] = 1;
    }
    let mut p: usize = 2;
    while p <= n {
        if phi[p] == p as u64 {
            // p is prime; apply the (p-1)/p factor to every multiple.
            let mut k = p;
            while k <= n {
                phi[k] = phi[k] / p as u64 * (p as u64 - 1);
                k += p;
            }
        }
        p += 1;
    }
    phi
}

#[cfg(test)]
mod tests {
    use super::{phi, phi_sieve};
    use quickcheck::TestResult;
    use quickcheck_macros::quickcheck;

    /// Canonical `phi(n)` table for `n = 1..=20` (OEIS A000010).
    const PHI_TABLE: [u64; 20] = [
        1, 1, 2, 2, 4, 2, 6, 4, 6, 4, 10, 4, 12, 6, 8, 8, 16, 6, 18, 8,
    ];

    fn gcd(mut a: u64, mut b: u64) -> u64 {
        while b != 0 {
            let t = a % b;
            a = b;
            b = t;
        }
        a
    }

    #[test]
    fn phi_zero_is_zero() {
        assert_eq!(phi(0), 0);
    }

    #[test]
    fn phi_one_is_one() {
        assert_eq!(phi(1), 1);
    }

    #[test]
    fn phi_of_primes() {
        for &p in &[2_u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 97, 9973] {
            assert_eq!(phi(p), p - 1, "phi({p}) should be {p} - 1");
        }
    }

    #[test]
    fn phi_of_prime_powers() {
        // phi(p^k) = p^(k-1) * (p-1)
        for &(p, k) in &[(2_u64, 5_u32), (3, 4), (5, 3), (7, 3), (11, 2), (13, 2)] {
            let pk = p.pow(k);
            let expected = p.pow(k - 1) * (p - 1);
            assert_eq!(phi(pk), expected, "phi({p}^{k}) = {expected}");
        }
    }

    #[test]
    fn phi_canonical_table_1_to_20() {
        for (idx, &expected) in PHI_TABLE.iter().enumerate() {
            let n = (idx + 1) as u64;
            assert_eq!(phi(n), expected, "phi({n}) should be {expected}");
        }
    }

    #[test]
    fn phi_matches_brute_force_small() {
        for n in 1..=200_u64 {
            let brute = (1..=n).filter(|&k| gcd(k, n) == 1).count() as u64;
            assert_eq!(phi(n), brute, "phi({n}) brute mismatch");
        }
    }

    #[test]
    fn sieve_zero_yields_single_zero() {
        assert_eq!(phi_sieve(0), vec![0]);
    }

    #[test]
    fn sieve_canonical_table_1_to_20() {
        let s = phi_sieve(20);
        assert_eq!(s[0], 0);
        for (idx, &expected) in PHI_TABLE.iter().enumerate() {
            let n = idx + 1;
            assert_eq!(s[n], expected, "sieve phi({n}) should be {expected}");
        }
    }

    #[test]
    fn sieve_agrees_with_phi_first_1000() {
        let s = phi_sieve(1000);
        for n in 0..=1000_u64 {
            assert_eq!(s[n as usize], phi(n), "mismatch at n = {n}");
        }
    }

    #[quickcheck]
    fn prop_multiplicative_when_coprime(a: u32, b: u32) -> TestResult {
        // Restrict to a manageable range; widen to u64 to avoid overflow in a*b.
        if a == 0 || b == 0 {
            return TestResult::discard();
        }
        let a = u64::from(a % 5_000 + 1);
        let b = u64::from(b % 5_000 + 1);
        if gcd(a, b) != 1 {
            return TestResult::discard();
        }
        TestResult::from_bool(phi(a * b) == phi(a) * phi(b))
    }

    #[quickcheck]
    fn prop_sieve_matches_phi(n: u16) -> bool {
        // Bound n to keep the test cheap.
        let n = (n % 2_000) as usize;
        let s = phi_sieve(n);
        (0..=n as u64).all(|k| s[k as usize] == phi(k))
    }
}
