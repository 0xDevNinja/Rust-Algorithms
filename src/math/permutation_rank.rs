//! Permutation rank / unrank using the factorial number system (Lehmer code).
//!
//! Establishes a bijection between the `n!` lexicographically-ordered
//! permutations of `[0..n)` and integer ranks in `0..n!`. The Lehmer digit
//! `L[i]` counts how many `j > i` satisfy `perm[j] < perm[i]`; the rank is
//! `Σ L[i] · (n-1-i)!`. Unrank reverses the process by repeatedly extracting
//! a digit and selecting from a shrinking list of available values.
//!
//! Constraints: `n ≤ 20`, since `21! > u128::MAX`.
//!
//! Complexity: `O(n^2)` time, `O(n)` extra space (uses `Vec::remove`, which
//! is acceptable for `n ≤ 20`). A Fenwick-tree variant could bring this to
//! `O(n log n)` if needed.
//!
//! Reference: <https://en.wikipedia.org/wiki/Lehmer_code>.
//! See also: <https://en.wikipedia.org/wiki/Factorial_number_system>.
//!
//! Maximum supported permutation length (`20!` fits in `u128`).
const MAX_N: usize = 20;

/// Returns `k!` as a `u128`. Panics if `k > MAX_N` (would overflow `u128`).
fn factorial(k: usize) -> u128 {
    assert!(k <= MAX_N, "factorial overflows u128 for k > {MAX_N}");
    let mut f: u128 = 1;
    for i in 2..=k as u128 {
        f *= i;
    }
    f
}

/// Returns the lexicographic rank of `perm` among the `n!` permutations
/// of `[0..n)`, where `n = perm.len()`.
///
/// Debug-asserts that `perm` is a valid permutation of `[0..n)` and that
/// `n ≤ 20`.
pub fn rank(perm: &[usize]) -> u128 {
    let n = perm.len();
    debug_assert!(n <= MAX_N, "n must be <= {MAX_N} (21! exceeds u128)");
    debug_assert!(
        is_permutation(perm),
        "input must be a permutation of [0..n)"
    );

    // Maintain a sorted list of values still available; the Lehmer digit
    // for position `i` is the index of `perm[i]` in that list.
    let mut available: Vec<usize> = (0..n).collect();
    let mut r: u128 = 0;
    for (i, &v) in perm.iter().enumerate() {
        let idx = available
            .iter()
            .position(|&x| x == v)
            .expect("value must be in available list for a valid permutation");
        available.remove(idx);
        r += (idx as u128) * factorial(n - 1 - i);
    }
    r
}

/// Returns the permutation of `[0..n)` whose lexicographic rank is `r`.
///
/// Panics if `n > 20` or if `r >= n!`.
pub fn unrank(n: usize, mut r: u128) -> Vec<usize> {
    assert!(n <= MAX_N, "n must be <= {MAX_N} (21! exceeds u128)");
    let total = factorial(n);
    assert!(
        r < total,
        "rank {r} out of range for n = {n} (n! = {total})"
    );

    let mut available: Vec<usize> = (0..n).collect();
    let mut perm = Vec::with_capacity(n);
    for i in 0..n {
        let f = factorial(n - 1 - i);
        let idx = (r / f) as usize;
        r %= f;
        perm.push(available.remove(idx));
    }
    perm
}

/// Checks that `perm` contains exactly the values `0..perm.len()`.
fn is_permutation(perm: &[usize]) -> bool {
    let n = perm.len();
    let mut seen = vec![false; n];
    for &v in perm {
        if v >= n || seen[v] {
            return false;
        }
        seen[v] = true;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::{factorial, rank, unrank};

    #[test]
    fn rank_canonical_examples() {
        assert_eq!(rank(&[0, 1, 2]), 0);
        assert_eq!(rank(&[2, 1, 0]), 5);
    }

    #[test]
    fn unrank_canonical_example() {
        assert_eq!(unrank(3, 4), vec![2, 0, 1]);
    }

    #[test]
    fn empty_permutation() {
        let empty: Vec<usize> = Vec::new();
        assert_eq!(rank(&empty), 0);
        assert_eq!(unrank(0, 0), empty);
    }

    #[test]
    fn single_element_permutation() {
        assert_eq!(rank(&[0]), 0);
        assert_eq!(unrank(1, 0), vec![0]);
    }

    #[test]
    #[should_panic(expected = "rank 6 out of range")]
    fn unrank_out_of_range_panics() {
        // 3! = 6, so rank 6 is invalid.
        let _ = unrank(3, 6);
    }

    /// Generate every permutation of `[0..n)` in lexicographic order using
    /// the standard "next permutation" algorithm.
    fn lex_permutations(n: usize) -> Vec<Vec<usize>> {
        let mut out = Vec::new();
        if n == 0 {
            out.push(Vec::new());
            return out;
        }
        let mut p: Vec<usize> = (0..n).collect();
        loop {
            out.push(p.clone());
            // Find largest i with p[i] < p[i+1].
            let Some(mut i) = (0..n - 1).rev().find(|&i| p[i] < p[i + 1]) else {
                break;
            };
            // Find largest j > i with p[j] > p[i].
            let mut j = n - 1;
            while p[j] <= p[i] {
                j -= 1;
            }
            p.swap(i, j);
            // Reverse the suffix.
            i += 1;
            let mut k = n - 1;
            while i < k {
                p.swap(i, k);
                i += 1;
                k -= 1;
            }
        }
        out
    }

    #[test]
    fn rank_matches_lex_index_for_small_n() {
        for n in 0..=6 {
            let perms = lex_permutations(n);
            assert_eq!(perms.len() as u128, factorial(n));
            for (idx, perm) in perms.iter().enumerate() {
                assert_eq!(rank(perm), idx as u128, "rank mismatch for {perm:?}");
                assert_eq!(
                    unrank(n, idx as u128),
                    *perm,
                    "unrank mismatch at idx {idx}"
                );
            }
        }
    }

    #[test]
    fn round_trip_random_small_perms() {
        // Deterministic LCG so the property test is reproducible without deps.
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut next = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            state
        };
        for n in 0..=8usize {
            for _ in 0..50 {
                // Fisher-Yates shuffle of [0..n).
                let mut p: Vec<usize> = (0..n).collect();
                for i in (1..n).rev() {
                    let j = (next() as usize) % (i + 1);
                    p.swap(i, j);
                }
                let r = rank(&p);
                assert!(r < factorial(n));
                assert_eq!(unrank(n, r), p);
            }
        }
    }

    #[test]
    fn round_trip_large_n() {
        // n = 12 -> 12! = 479_001_600 ranks. Spot-check a few.
        let n = 12;
        let total = factorial(n);
        for &r in &[0u128, 1, 42, total / 2, total - 1] {
            let p = unrank(n, r);
            assert_eq!(rank(&p), r);
        }
    }

    #[test]
    fn round_trip_max_n() {
        // n = 20 sits at the u128 boundary.
        let n = 20;
        let total = factorial(n);
        let r = total - 1;
        let p = unrank(n, r);
        assert_eq!(rank(&p), r);
        // The last permutation in lex order is the descending sequence.
        let descending: Vec<usize> = (0..n).rev().collect();
        assert_eq!(p, descending);
        assert_eq!(unrank(n, 0), (0..n).collect::<Vec<_>>());
    }
}
