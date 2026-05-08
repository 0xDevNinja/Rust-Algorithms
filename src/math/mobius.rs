//! Möbius function μ(n) and a linear sieve over `[0, n]`.
//!
//! μ(1) = 1, μ(n) = 0 if n has a squared prime factor, otherwise μ(n) = (-1)^k
//! where k is the number of distinct prime factors of n. By convention this
//! implementation returns μ(0) = 0 (0 is not a positive integer, so the
//! standard definition does not apply).
//!
//! Complexity:
//! - [`mobius`]: O(√n) trial division.
//! - [`mobius_sieve`]: O(n) using a linear sieve that tracks the smallest
//!   prime factor of each index.

/// Computes μ(n) by trial division.
///
/// Returns `0` if any prime factor of `n` appears with multiplicity ≥ 2,
/// otherwise `(-1)^k` where `k` is the number of distinct primes dividing `n`.
/// μ(1) = 1 and, by convention, μ(0) = 0.
pub const fn mobius(n: u64) -> i8 {
    if n == 0 {
        return 0;
    }
    if n == 1 {
        return 1;
    }
    let mut n = n;
    let mut sign: i8 = 1;
    let mut p: u64 = 2;
    while p * p <= n {
        if n.is_multiple_of(p) {
            n /= p;
            if n.is_multiple_of(p) {
                return 0;
            }
            sign = -sign;
        }
        p += 1;
    }
    if n > 1 {
        // remaining prime factor
        sign = -sign;
    }
    sign
}

/// Returns `mu` such that `mu[i] = μ(i)` for `i` in `0..=n`.
///
/// Uses a linear sieve: for each `i ≥ 2`, let `p = spf[i]` be the smallest
/// prime factor of `i`. If `(i / p) % p == 0` then `p² | i` and `mu[i] = 0`;
/// otherwise `mu[i] = -mu[i / p]`. `mu[0] = 0` and `mu[1] = 1`.
pub fn mobius_sieve(n: usize) -> Vec<i8> {
    let mut mu: Vec<i8> = vec![0; n + 1];
    if n >= 1 {
        mu[1] = 1;
    }
    let mut spf: Vec<usize> = vec![0; n + 1];
    let mut primes: Vec<usize> = Vec::new();
    for i in 2..=n {
        if spf[i] == 0 {
            spf[i] = i;
            primes.push(i);
            mu[i] = -1;
        }
        for &p in &primes {
            let ip = i.checked_mul(p);
            match ip {
                Some(v) if v <= n => {
                    spf[v] = p;
                    if i.is_multiple_of(p) {
                        mu[v] = 0;
                        break;
                    }
                    mu[v] = -mu[i];
                }
                _ => break,
            }
        }
    }
    mu
}

#[cfg(test)]
mod tests {
    use super::{mobius, mobius_sieve};

    #[test]
    fn mu_zero_convention() {
        assert_eq!(mobius(0), 0);
    }

    #[test]
    fn mu_one() {
        assert_eq!(mobius(1), 1);
    }

    #[test]
    fn mu_primes_are_minus_one() {
        for &p in &[2u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 97, 9973] {
            assert_eq!(mobius(p), -1, "μ({p}) should be -1");
        }
    }

    #[test]
    fn mu_product_of_two_distinct_primes_is_one() {
        // p*q with p ≠ q
        for &(p, q) in &[(2u64, 3u64), (3, 5), (5, 7), (2, 7), (11, 13), (17, 23)] {
            assert_eq!(mobius(p * q), 1, "μ({}) should be 1", p * q);
        }
    }

    #[test]
    fn mu_squared_prime_factor_is_zero() {
        for &p in &[2u64, 3, 5, 7, 11, 13] {
            assert_eq!(mobius(p * p), 0);
            assert_eq!(mobius(p * p * 5), 0);
        }
    }

    #[test]
    fn mu_canonical_table_1_to_20() {
        let expected: [i8; 20] = [
            1, -1, -1, 0, -1, 1, -1, 0, 0, 1, -1, 0, -1, 1, 1, 0, -1, 0, -1, 0,
        ];
        for (i, &want) in expected.iter().enumerate() {
            let n = (i + 1) as u64;
            assert_eq!(mobius(n), want, "μ({n}) mismatch");
        }
    }

    #[test]
    fn sieve_agrees_with_direct_function() {
        let n: usize = 50;
        let mu = mobius_sieve(n);
        assert_eq!(mu.len(), n + 1);
        assert_eq!(mu[0], 0);
        for i in 0..=n {
            assert_eq!(mu[i], mobius(i as u64), "μ({i}) sieve vs direct");
        }
    }

    #[test]
    fn sieve_zero_is_zero_one_is_one() {
        let mu = mobius_sieve(10);
        assert_eq!(mu[0], 0);
        assert_eq!(mu[1], 1);
    }

    #[test]
    fn mobius_inversion_identity() {
        // Σ_{d|n} μ(d) == [n == 1]
        let mu = mobius_sieve(30);
        for n in 1usize..=30 {
            let mut sum: i32 = 0;
            for d in 1..=n {
                if n % d == 0 {
                    sum += mu[d] as i32;
                }
            }
            let expected = i32::from(n == 1);
            assert_eq!(sum, expected, "Σ_{{d|{n}}} μ(d) should be {expected}");
        }
    }

    #[test]
    fn sieve_small_lengths() {
        assert_eq!(mobius_sieve(0), vec![0]);
        assert_eq!(mobius_sieve(1), vec![0, 1]);
        assert_eq!(mobius_sieve(2), vec![0, 1, -1]);
    }
}
