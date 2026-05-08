//! Inclusion-exclusion principle.
//!
//! Provides a generic driver that, given an oracle returning the size of
//! intersections of any subset of `n` finite sets, computes the size of the
//! union `|A_1 ∪ … ∪ A_n|` by summing the alternating series
//!
//! ```text
//! |A_1 ∪ … ∪ A_n| = Σ_{∅ ≠ S ⊆ {1..n}} (-1)^{|S|+1} · |⋂_{i ∈ S} A_i|.
//! ```
//!
//! Two canonical applications are included:
//!
//! - [`count_coprime_to`] counts integers in `1..=n` divisible by none of a
//!   list of distinct primes (a generalization of Euler's totient).
//! - [`derangements`] counts permutations of `n` elements with no fixed point.
//!
//! Complexity:
//! - [`inclusion_exclusion`]: enumerates all `2^n` subsets, calling the oracle
//!   once per non-empty subset, so `O(2^n · C)` where `C` is the cost of the
//!   oracle.
//! - [`count_coprime_to`]: `O(2^k)` where `k = primes.len()`.
//! - [`derangements`]: `O(n)` via the standard two-term recurrence.

/// Computes `|A_1 ∪ … ∪ A_n|` via inclusion-exclusion.
///
/// `intersect_size(&idx)` must return the size of the intersection of the
/// sets indexed by `idx` (a slice of distinct indices in `0..n`). For the
/// empty slice it is conventionally the size of the universe; that case is
/// never queried by this function.
///
/// Subsets are enumerated by bitmask in `1..(1 << n)`. The sign of each term
/// is `(-1)^{|S|+1}`, i.e. positive when `|S|` is odd.
pub fn inclusion_exclusion<F: Fn(&[usize]) -> i64>(n: usize, intersect_size: F) -> i64 {
    assert!(n < usize::BITS as usize, "n too large for bitmask");
    let mut total: i64 = 0;
    let mut indices: Vec<usize> = Vec::with_capacity(n);
    for mask in 1u64..(1u64 << n) {
        indices.clear();
        for i in 0..n {
            if (mask >> i) & 1 == 1 {
                indices.push(i);
            }
        }
        let term = intersect_size(&indices);
        if indices.len() % 2 == 1 {
            total += term;
        } else {
            total -= term;
        }
    }
    total
}

/// Counts integers in `1..=n` that are coprime to every prime in `primes`.
///
/// `primes` must be a slice of distinct primes; the function does not verify
/// primality. The count of integers in `1..=n` divisible by at least one
/// prime is computed by inclusion-exclusion over the divisor lattice, and
/// the answer is `n` minus that count.
pub fn count_coprime_to(n: u64, primes: &[u64]) -> u64 {
    let k = primes.len();
    assert!(k < u32::BITS as usize, "too many primes");
    let mut divisible: u64 = 0;
    // Iterate non-empty subsets of primes; add or subtract n / product.
    for mask in 1u32..(1u32 << k) {
        let mut product: u64 = 1;
        let mut bits = 0u32;
        let mut overflow = false;
        for (i, &p) in primes.iter().enumerate() {
            if (mask >> i) & 1 == 1 {
                bits += 1;
                match product.checked_mul(p) {
                    Some(v) if v <= n => product = v,
                    _ => {
                        overflow = true;
                        break;
                    }
                }
            }
        }
        if overflow {
            // Product exceeds n, so n / product = 0; nothing to add.
            continue;
        }
        let term = n / product;
        if bits % 2 == 1 {
            divisible += term;
        } else {
            divisible -= term;
        }
    }
    n - divisible
}

/// Counts derangements `!n` (permutations of `n` items with no fixed point).
///
/// Implemented via the linear recurrence `!n = (n - 1) · (!(n - 1) + !(n - 2))`
/// with bases `!0 = 1`, `!1 = 0`. This avoids the floating-point hazards of
/// the IE formula while remaining a direct consequence of it.
pub fn derangements(n: u32) -> u64 {
    if n == 0 {
        return 1;
    }
    if n == 1 {
        return 0;
    }
    let (mut prev_prev, mut prev) = (1u64, 0u64); // !0, !1
    for k in 2..=n {
        let next = (k as u64 - 1) * (prev + prev_prev);
        prev_prev = prev;
        prev = next;
    }
    prev
}

#[cfg(test)]
mod tests {
    use super::{count_coprime_to, derangements, inclusion_exclusion};

    #[test]
    fn union_of_two_sets() {
        // |A| = 3, |B| = 4, |A ∩ B| = 2  ⇒  |A ∪ B| = 5.
        let union = inclusion_exclusion(2, |idx| match idx {
            [0] => 3,
            [1] => 4,
            [0, 1] => 2,
            _ => unreachable!(),
        });
        assert_eq!(union, 5);
    }

    #[test]
    fn union_of_three_sets() {
        // |A|=|B|=|C|=10, pairwise intersections =3, triple =1.
        // |A∪B∪C| = 30 - 9 + 1 = 22.
        let union = inclusion_exclusion(3, |idx| match idx.len() {
            1 => 10,
            2 => 3,
            3 => 1,
            _ => 0,
        });
        assert_eq!(union, 22);
    }

    #[test]
    fn union_n_zero_is_zero() {
        let union = inclusion_exclusion(0, |_| 1);
        assert_eq!(union, 0);
    }

    #[test]
    fn coprime_to_30_is_phi_30() {
        // φ(30) = 30 · (1 - 1/2)(1 - 1/3)(1 - 1/5) = 8.
        assert_eq!(count_coprime_to(30, &[2, 3, 5]), 8);
    }

    #[test]
    fn coprime_to_100_via_first_four_primes() {
        // Verify against direct enumeration.
        let primes = [2u64, 3, 5, 7];
        let direct = (1u64..=100)
            .filter(|m| primes.iter().all(|p| m % p != 0))
            .count() as u64;
        assert_eq!(count_coprime_to(100, &primes), direct);
    }

    #[test]
    fn coprime_empty_prime_list() {
        // No primes excluded, so every integer in [1, n] qualifies.
        assert_eq!(count_coprime_to(50, &[]), 50);
    }

    #[test]
    fn coprime_handles_overflow_safe_products() {
        // Large primes whose product overflows u64 must not panic; their
        // contribution is zero because the product exceeds n.
        let primes = [2u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31];
        let direct = (1u64..=100)
            .filter(|m| primes.iter().all(|p| m % p != 0))
            .count() as u64;
        assert_eq!(count_coprime_to(100, &primes), direct);
    }

    #[test]
    fn derangement_table() {
        let expected = [
            (0u32, 1u64),
            (1, 0),
            (2, 1),
            (3, 2),
            (4, 9),
            (5, 44),
            (6, 265),
            (7, 1854),
            (8, 14833),
            (9, 133_496),
            (10, 1_334_961),
        ];
        for (n, want) in expected {
            assert_eq!(derangements(n), want, "!{n}");
        }
    }

    fn brute_force_derangements(n: u32) -> u64 {
        // Count permutations of 0..n with no fixed point by direct search.
        let n = n as usize;
        if n == 0 {
            return 1;
        }
        let mut perm: Vec<usize> = (0..n).collect();
        let mut count = 0u64;
        permute(&mut perm, 0, &mut count);
        count
    }

    fn permute(perm: &mut [usize], start: usize, count: &mut u64) {
        if start == perm.len() {
            if perm.iter().enumerate().all(|(i, &v)| i != v) {
                *count += 1;
            }
            return;
        }
        for i in start..perm.len() {
            perm.swap(start, i);
            permute(perm, start + 1, count);
            perm.swap(start, i);
        }
    }

    #[test]
    fn derangements_match_brute_force() {
        for n in 0..=7u32 {
            assert_eq!(
                derangements(n),
                brute_force_derangements(n),
                "mismatch at n={n}"
            );
        }
    }
}
