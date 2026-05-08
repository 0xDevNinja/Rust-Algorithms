//! Linear (Euler) sieve. Computes the smallest-prime-factor table together
//! with the list of primes up to `n` in O(n) time and O(n) space. Each
//! composite is marked exactly once by its smallest prime factor.

/// Returns `(spf, primes)` where `spf[i]` is the smallest prime factor of
/// `i` for `2 <= i <= n` (with `spf[0] = spf[1] = 0`), and `primes` is the
/// ordered list of primes `p` with `p <= n`.
pub fn linear_sieve(n: usize) -> (Vec<usize>, Vec<u32>) {
    let mut spf = vec![0_usize; n + 1];
    let mut primes: Vec<u32> = Vec::new();
    if n < 2 {
        return (spf, primes);
    }
    for i in 2..=n {
        if spf[i] == 0 {
            spf[i] = i;
            primes.push(i as u32);
        }
        for &p in &primes {
            let p_us = p as usize;
            if p_us > spf[i] || i * p_us > n {
                break;
            }
            spf[i * p_us] = p_us;
        }
    }
    (spf, primes)
}

/// Returns the prime-power factorization of `n` using the precomputed
/// smallest-prime-factor table. The table must satisfy `spf.len() > n as usize`
/// and have been produced by [`linear_sieve`] for some bound `>= n`.
/// Returns an empty vector for `n < 2`.
pub fn factorize_with_spf(n: u32, spf: &[usize]) -> Vec<(u32, u32)> {
    let mut out: Vec<(u32, u32)> = Vec::new();
    let mut x = n as usize;
    if x < 2 {
        return out;
    }
    while x > 1 {
        let p = spf[x] as u32;
        let mut e = 0_u32;
        while spf[x] as u32 == p {
            x /= p as usize;
            e += 1;
        }
        out.push((p, e));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{factorize_with_spf, linear_sieve};
    use quickcheck_macros::quickcheck;

    #[test]
    fn empty_n_zero() {
        let (spf, primes) = linear_sieve(0);
        assert_eq!(spf, vec![0]);
        assert!(primes.is_empty());
    }

    #[test]
    fn n_one() {
        let (spf, primes) = linear_sieve(1);
        assert_eq!(spf, vec![0, 0]);
        assert!(primes.is_empty());
    }

    #[test]
    fn n_two() {
        let (spf, primes) = linear_sieve(2);
        assert_eq!(spf, vec![0, 0, 2]);
        assert_eq!(primes, vec![2]);
    }

    #[test]
    fn primes_up_to_30() {
        let (_, primes) = linear_sieve(30);
        assert_eq!(primes, vec![2, 3, 5, 7, 11, 13, 17, 19, 23, 29]);
    }

    #[test]
    fn spf_selected_composites() {
        let (spf, _) = linear_sieve(100);
        assert_eq!(spf[12], 2);
        assert_eq!(spf[15], 3);
        assert_eq!(spf[49], 7);
        assert_eq!(spf[77], 7);
        assert_eq!(spf[97], 97);
        assert_eq!(spf[100], 2);
    }

    #[test]
    fn spf_zero_for_below_two() {
        let (spf, _) = linear_sieve(50);
        assert_eq!(spf[0], 0);
        assert_eq!(spf[1], 0);
    }

    #[test]
    fn factorize_basic() {
        let (spf, _) = linear_sieve(1_000);
        assert!(factorize_with_spf(0, &spf).is_empty());
        assert!(factorize_with_spf(1, &spf).is_empty());
        assert_eq!(factorize_with_spf(2, &spf), vec![(2, 1)]);
        assert_eq!(factorize_with_spf(360, &spf), vec![(2, 3), (3, 2), (5, 1)]);
        assert_eq!(factorize_with_spf(997, &spf), vec![(997, 1)]);
    }

    #[test]
    fn prime_count_100() {
        // π(100) = 25
        let (_, primes) = linear_sieve(100);
        assert_eq!(primes.len(), 25);
    }

    #[allow(clippy::needless_pass_by_value)]
    #[quickcheck]
    fn spf_chain_reconstructs_n(seed: u32) -> bool {
        let n = 2 + (seed % 998) as usize;
        let (spf, _) = linear_sieve(n);
        let mut x = n;
        let mut acc = 1_usize;
        while x > 1 {
            let p = spf[x];
            acc *= p;
            x /= p;
        }
        acc == n
    }

    #[allow(clippy::needless_pass_by_value)]
    #[quickcheck]
    fn factorize_round_trip(seed: u32) -> bool {
        let n = 2 + (seed % 998);
        let (spf, _) = linear_sieve(n as usize);
        let mut acc: u64 = 1;
        for (p, e) in factorize_with_spf(n, &spf) {
            acc *= u64::from(p).pow(e);
        }
        acc == u64::from(n)
    }
}
