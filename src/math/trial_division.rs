//! Trial-division prime factorization for `u64`.
//!
//! Divides the input by every candidate divisor up to `sqrt(n)`, peeling off
//! prime factors with their multiplicities. Tries 2, then odd candidates only,
//! which keeps the inner loop tight while remaining trivially correct. Runs in
//! `O(sqrt(n) / log n)` time and `O(log n)` output size.
//!
//! Suitable for `n` up to about `10^14`; for larger inputs use Pollard's rho.

/// Returns the prime factors of `n` paired with their exponents, sorted by
/// prime in ascending order. `factorize(0)` returns the empty vector and
/// `factorize(1)` returns the empty vector since neither has prime factors.
///
/// - Time: `O(sqrt(n) / log n)`.
/// - Space: `O(log n)`.
pub fn factorize(mut n: u64) -> Vec<(u64, u32)> {
    let mut out = Vec::new();
    if n < 2 {
        return out;
    }
    let peel = |n: &mut u64, p: u64, out: &mut Vec<(u64, u32)>| {
        if n.is_multiple_of(p) {
            let mut e = 0_u32;
            while n.is_multiple_of(p) {
                *n /= p;
                e += 1;
            }
            out.push((p, e));
        }
    };
    peel(&mut n, 2, &mut out);
    let mut p: u64 = 3;
    while p.saturating_mul(p) <= n {
        peel(&mut n, p, &mut out);
        p += 2;
    }
    if n > 1 {
        out.push((n, 1));
    }
    out
}

/// Returns the sorted, deduplicated list of prime factors of `n`.
///
/// - Time: `O(sqrt(n) / log n)`.
/// - Space: `O(log n)`.
pub fn distinct_prime_factors(n: u64) -> Vec<u64> {
    factorize(n).into_iter().map(|(p, _)| p).collect()
}

/// Returns the smallest prime factor of `n`, or `None` for `n < 2`.
///
/// - Time: `O(sqrt(n))` worst case.
pub const fn smallest_prime_factor(n: u64) -> Option<u64> {
    if n < 2 {
        return None;
    }
    if n.is_multiple_of(2) {
        return Some(2);
    }
    let mut p: u64 = 3;
    while p.saturating_mul(p) <= n {
        if n.is_multiple_of(p) {
            return Some(p);
        }
        p += 2;
    }
    Some(n)
}

#[cfg(test)]
mod tests {
    use super::{distinct_prime_factors, factorize, smallest_prime_factor};
    use quickcheck_macros::quickcheck;

    fn brute_factorize(mut n: u64) -> Vec<(u64, u32)> {
        let mut out = Vec::new();
        if n < 2 {
            return out;
        }
        let mut p = 2_u64;
        while p * p <= n {
            if n.is_multiple_of(p) {
                let mut e = 0_u32;
                while n.is_multiple_of(p) {
                    n /= p;
                    e += 1;
                }
                out.push((p, e));
            }
            p += 1;
        }
        if n > 1 {
            out.push((n, 1));
        }
        out
    }

    #[test]
    fn small_inputs() {
        assert!(factorize(0).is_empty());
        assert!(factorize(1).is_empty());
        assert_eq!(factorize(2), vec![(2, 1)]);
        assert_eq!(factorize(3), vec![(3, 1)]);
        assert_eq!(factorize(4), vec![(2, 2)]);
    }

    #[test]
    fn composite_with_repeated_primes() {
        assert_eq!(factorize(360), vec![(2, 3), (3, 2), (5, 1)]);
        assert_eq!(factorize(1024), vec![(2, 10)]);
        assert_eq!(
            factorize(2_310),
            vec![(2, 1), (3, 1), (5, 1), (7, 1), (11, 1)]
        );
    }

    #[test]
    fn large_prime() {
        assert_eq!(factorize(1_000_000_007), vec![(1_000_000_007, 1)]);
    }

    #[test]
    fn distinct_factors_are_unique_and_sorted() {
        let v = distinct_prime_factors(2 * 2 * 3 * 5 * 5 * 11);
        assert_eq!(v, vec![2, 3, 5, 11]);
    }

    #[test]
    fn smallest_prime_factor_basics() {
        assert_eq!(smallest_prime_factor(0), None);
        assert_eq!(smallest_prime_factor(1), None);
        assert_eq!(smallest_prime_factor(15), Some(3));
        assert_eq!(smallest_prime_factor(91), Some(7));
        assert_eq!(smallest_prime_factor(97), Some(97));
    }

    #[allow(clippy::needless_pass_by_value)]
    #[quickcheck]
    fn matches_brute(n: u32) -> bool {
        let n = u64::from(n);
        factorize(n) == brute_factorize(n)
    }

    #[allow(clippy::needless_pass_by_value)]
    #[quickcheck]
    fn product_round_trip(n: u32) -> bool {
        let n = u64::from(n);
        if n < 2 {
            return true;
        }
        let mut acc: u64 = 1;
        for (p, e) in factorize(n) {
            acc *= p.pow(e);
        }
        acc == n
    }
}
