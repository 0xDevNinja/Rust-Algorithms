//! Sieve of Eratosthenes. Generates all primes up to (and including) `n` in
//! O(n log log n) time and O(n) space.

/// Returns a vector of all primes `p` with `p <= n`.
pub fn primes_up_to(n: u32) -> Vec<u32> {
    if n < 2 {
        return Vec::new();
    }
    let n_us = n as usize;
    let mut is_prime = vec![true; n_us + 1];
    is_prime[0] = false;
    is_prime[1] = false;
    let mut p = 2_usize;
    while p * p <= n_us {
        if is_prime[p] {
            let mut k = p * p;
            while k <= n_us {
                is_prime[k] = false;
                k += p;
            }
        }
        p += 1;
    }
    is_prime
        .iter()
        .enumerate()
        .filter_map(|(i, &flag)| if flag { Some(i as u32) } else { None })
        .collect()
}

/// Returns the boolean prime-mask `mask` of length `n + 1` such that
/// `mask[i]` is true iff `i` is prime.
pub fn prime_mask(n: u32) -> Vec<bool> {
    let n_us = n as usize;
    let mut mask = vec![true; n_us + 1];
    if !mask.is_empty() {
        mask[0] = false;
    }
    if mask.len() >= 2 {
        mask[1] = false;
    }
    let mut p = 2_usize;
    while p * p <= n_us {
        if mask[p] {
            let mut k = p * p;
            while k <= n_us {
                mask[k] = false;
                k += p;
            }
        }
        p += 1;
    }
    mask
}

#[cfg(test)]
mod tests {
    use super::{prime_mask, primes_up_to};

    #[test]
    fn small_n() {
        assert_eq!(primes_up_to(0), Vec::<u32>::new());
        assert_eq!(primes_up_to(1), Vec::<u32>::new());
        assert_eq!(primes_up_to(2), vec![2]);
    }

    #[test]
    fn classic_30() {
        assert_eq!(primes_up_to(30), vec![2, 3, 5, 7, 11, 13, 17, 19, 23, 29]);
    }

    #[test]
    fn prime_count_100() {
        // π(100) = 25
        assert_eq!(primes_up_to(100).len(), 25);
    }

    #[test]
    fn mask_aligns_with_list() {
        let n = 50;
        let mask = prime_mask(n);
        let listed = primes_up_to(n);
        for p in listed {
            assert!(mask[p as usize]);
        }
        let count = mask.iter().filter(|&&b| b).count();
        assert_eq!(count, primes_up_to(n).len());
    }

    #[test]
    fn n_is_prime_inclusive() {
        let primes = primes_up_to(13);
        assert_eq!(primes.last(), Some(&13));
    }

    #[test]
    fn larger_n_does_not_panic() {
        let primes = primes_up_to(10_000);
        // π(10_000) = 1229
        assert_eq!(primes.len(), 1229);
    }
}
