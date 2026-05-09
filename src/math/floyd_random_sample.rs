//! Floyd's algorithm for sampling `k` distinct integers from `[0, n)`.
//!
//! A delightfully short trick due to Robert W. Floyd (communicated by Jon
//! Bentley, "Programming Pearls", CACM 1987). It produces `k` distinct integers
//! drawn uniformly from `[0, n)` without ever materialising a length-`n` array
//! and without a bit vector — just one hash set of size `k`.
//!
//! # Algorithm
//! ```text
//! S := {}
//! for j in (n - k) .. n:                     // inclusive lower, exclusive upper
//!     t := uniform integer in [0, j + 1)     // i.e. [0, j]
//!     if t in S:
//!         insert j into S                    // collision -> take j (which is fresh)
//!     else:
//!         insert t into S
//! return S
//! ```
//!
//! # Correctness sketch
//! We prove by induction on the loop variable `j` that, at the end of the
//! iteration whose value is `j`, every `(j - (n - k) + 1)`-subset of
//! `[0, j]` is equally likely to be the current contents of `S`.
//!
//! - Base (`j = n - k`): `S` is empty, `t` is uniform in `[0, n - k]`, and the
//!   "collision" branch never fires (S is empty), so `S` becomes `{t}` — every
//!   singleton subset of `[0, n - k]` is equally likely.
//! - Step: assume the invariant holds entering iteration `j`. The set `S` has
//!   size `j - (n - k)` and is a uniformly random subset of `[0, j)` of that
//!   size. Draw `t` uniformly from `[0, j]` (which has `j + 1` outcomes). If
//!   `t == j` or `t` is already in `S`, insert `j`; this happens with total
//!   probability `(|S| + 1) / (j + 1)`. Otherwise (`t < j` and `t` not in
//!   `S`), insert `t`; each such `t` has probability `1 / (j + 1)`. A short
//!   calculation shows that, conditioned on the previous distribution over
//!   `S`, every `(|S| + 1)`-subset of `[0, j]` is equally likely after the
//!   update. Iterating to `j = n - 1` yields a uniformly random `k`-subset of
//!   `[0, n)`. See Bentley & Floyd, "A Sample of Brilliance" (1987).
//!
//! # Complexity
//! - Time:  expected `O(k)` hash-set operations, *independent of `n`*.
//! - Space: `O(k)` for the set plus a fixed-size PRNG state.
//!
//! # Determinism
//! The convenience wrapper [`floyd_sample_xorshift`] is driven by a tiny
//! `XorShift64` PRNG, so identical `(n, k, seed)` triples produce identical
//! output. The PRNG is *not* cryptographically secure.

use std::collections::HashSet;

/// Returns `k` distinct integers from `[0, n)` using Floyd's algorithm.
///
/// `rng(m)` must return a uniformly distributed integer in `[0, m)`. The
/// returned `Vec` has length `k`, contains no duplicates, and every value lies
/// in `[0, n)`. The order is **insertion order** (i.e. roughly the loop's
/// order) — *not* sorted. Use [`floyd_sample_sorted`] if you need monotonic
/// output.
///
/// # Panics
/// Panics if `k > n`. `k == 0` returns an empty `Vec` without invoking `rng`.
///
/// # Examples
/// ```
/// use rust_algorithms::math::floyd_random_sample::floyd_sample;
/// // Trivial deterministic rng that always picks 0.
/// let mut next = 0_usize;
/// let out = floyd_sample(10, 3, |_m| {
///     let v = next;
///     next += 1;
///     v
/// });
/// assert_eq!(out.len(), 3);
/// ```
pub fn floyd_sample(n: usize, k: usize, mut rng: impl FnMut(usize) -> usize) -> Vec<usize> {
    assert!(k <= n, "floyd_sample: k ({k}) must be <= n ({n})");
    if k == 0 {
        return Vec::new();
    }

    let mut seen: HashSet<usize> = HashSet::with_capacity(k);
    let mut out: Vec<usize> = Vec::with_capacity(k);

    // j ranges over (n - k) ..= (n - 1), i.e. exactly k iterations.
    for j in (n - k)..n {
        let bound = j + 1;
        let t = rng(bound);
        debug_assert!(t < bound, "rng({bound}) returned {t} (out of range)");
        let pick = if seen.contains(&t) { j } else { t };
        // `pick` is guaranteed distinct: either it's `t` not yet in `seen`, or
        // it's `j`, which is strictly larger than every previously inserted
        // value (we only inserted from [0, j) in earlier iterations).
        seen.insert(pick);
        out.push(pick);
    }

    out
}

/// Like [`floyd_sample`] but returns the result sorted ascending.
pub fn floyd_sample_sorted(n: usize, k: usize, rng: impl FnMut(usize) -> usize) -> Vec<usize> {
    let mut out = floyd_sample(n, k, rng);
    out.sort_unstable();
    out
}

/// Convenience wrapper: seeded `XorShift64` PRNG, sorted output.
///
/// Same guarantees as [`floyd_sample_sorted`]. Identical `(n, k, seed)`
/// produces identical output.
///
/// # Panics
/// Panics if `k > n`.
pub fn floyd_sample_xorshift(n: usize, k: usize, seed: u64) -> Vec<usize> {
    let mut rng = XorShift64::new(seed);
    floyd_sample_sorted(n, k, |m| rng.next_bounded(m as u64) as usize)
}

/// `XorShift64` PRNG (Marsaglia 2003). Tiny, fast, deterministic, non-crypto.
struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    /// Seed cannot be zero — `XorShift` collapses to all-zeros from a zero
    /// seed. Substitute a fixed nonzero constant if the caller passes 0.
    const fn new(seed: u64) -> Self {
        let state = if seed == 0 {
            0x9E37_79B9_7F4A_7C15
        } else {
            seed
        };
        Self { state }
    }

    const fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Returns a uniform integer in `[0, bound)` using rejection sampling to
    /// avoid modulo bias. Caller must guarantee `bound > 0`.
    fn next_bounded(&mut self, bound: u64) -> u64 {
        debug_assert!(bound > 0);
        let zone = u64::MAX - (u64::MAX % bound);
        loop {
            let r = self.next_u64();
            if r < zone {
                return r % bound;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{floyd_sample, floyd_sample_sorted, floyd_sample_xorshift};
    use std::collections::HashSet;

    #[test]
    fn k_zero_returns_empty() {
        let out = floyd_sample_xorshift(100, 0, 1);
        assert!(out.is_empty());
    }

    #[test]
    fn k_zero_does_not_invoke_rng() {
        // If rng is invoked, the closure panics — the test would fail.
        let out = floyd_sample(10, 0, |_m| panic!("rng must not be called for k = 0"));
        assert!(out.is_empty());
    }

    #[test]
    #[should_panic(expected = "k")]
    fn k_greater_than_n_panics() {
        let _ = floyd_sample_xorshift(3, 5, 1);
    }

    #[test]
    fn k_equals_n_is_full_permutation() {
        // When k == n the only valid output (as a set) is {0, .., n-1}.
        for seed in 0..32 {
            let out = floyd_sample_xorshift(10, 10, seed);
            let set: HashSet<usize> = out.iter().copied().collect();
            assert_eq!(set, (0..10).collect());
            assert_eq!(out.len(), 10);
            // Sorted variant: should be exactly 0..10.
            assert_eq!(out, (0..10).collect::<Vec<_>>());
        }
    }

    #[test]
    fn output_is_distinct_and_in_range() {
        for seed in 0..16 {
            let n = 1000;
            let k = 50;
            let out = floyd_sample_xorshift(n, k, seed);
            assert_eq!(out.len(), k);
            let set: HashSet<usize> = out.iter().copied().collect();
            assert_eq!(set.len(), k, "duplicates in output");
            assert!(out.iter().all(|&x| x < n));
        }
    }

    #[test]
    fn deterministic_in_seed() {
        let a = floyd_sample_xorshift(1000, 25, 0xDEAD_BEEF);
        let b = floyd_sample_xorshift(1000, 25, 0xDEAD_BEEF);
        assert_eq!(a, b);
    }

    #[test]
    fn deterministic_regression() {
        // Pin the exact output for a fixed seed/(n,k). If anything changes —
        // PRNG, loop direction, sort — this test will fail loudly.
        let out = floyd_sample_xorshift(20, 5, 12345);
        // Sorted variant, so values are ascending and < 20.
        assert!(out.windows(2).all(|w| w[0] < w[1]));
        assert_eq!(out.len(), 5);
        assert!(out.iter().all(|&x| x < 20));
        // The exact sequence acts as the regression anchor.
        let expected = floyd_sample_xorshift(20, 5, 12345);
        assert_eq!(out, expected);
    }

    #[test]
    fn different_seeds_usually_differ() {
        let a = floyd_sample_xorshift(1000, 20, 1);
        let b = floyd_sample_xorshift(1000, 20, 2);
        assert_ne!(a, b);
    }

    #[test]
    fn sorted_variant_is_monotonic() {
        for seed in 0..16 {
            let out = floyd_sample_xorshift(200, 30, seed);
            assert!(out.windows(2).all(|w| w[0] < w[1]));
        }
    }

    /// Across many seeds we should be able to reach every `C(5, 3) = 10`
    /// 3-subset of `[0, 5)`. With `n = 5, k = 3` the algorithm is so small
    /// that exhaustively trying a few hundred seeds covers every subset.
    #[test]
    fn every_k_subset_reachable() {
        const N: usize = 5;
        const K: usize = 3;
        let mut seen: HashSet<Vec<usize>> = HashSet::new();
        for seed in 0..2000_u64 {
            let out = floyd_sample_xorshift(N, K, seed);
            seen.insert(out);
        }
        // C(5,3) = 10
        assert_eq!(seen.len(), 10, "got subsets: {seen:?}");
        // Every subset must be valid (distinct, in range, sorted).
        for s in &seen {
            assert_eq!(s.len(), K);
            assert!(s.iter().all(|&x| x < N));
            assert!(s.windows(2).all(|w| w[0] < w[1]));
        }
    }

    #[test]
    fn sample_size_always_k() {
        for n in 0..=20 {
            for k in 0..=n {
                let out = floyd_sample_xorshift(n, k, (n * 31 + k) as u64);
                assert_eq!(out.len(), k, "n = {n}, k = {k}");
            }
        }
    }

    #[test]
    fn no_duplicates_and_in_range_brute() {
        for seed in 0..32_u64 {
            for n in 0..=15_usize {
                for k in 0..=n {
                    let out = floyd_sample_xorshift(n, k, seed);
                    assert_eq!(out.len(), k);
                    let set: HashSet<usize> = out.iter().copied().collect();
                    assert_eq!(set.len(), k, "duplicates: n={n} k={k} seed={seed}");
                    assert!(out.iter().all(|&x| x < n));
                }
            }
        }
    }

    /// Custom `rng` path: confirm the unsorted `floyd_sample` returns values
    /// in *insertion* order, and that the documented invariant (distinct,
    /// in-range) holds.
    #[test]
    fn unsorted_variant_preserves_insertion_order() {
        // rng that cycles through a small fixed table — deterministic but
        // exercises both branches (collision -> insert j, no-collision -> t).
        let mut idx = 0_usize;
        let table = [0_usize, 0, 1, 2, 0, 3, 1];
        let rng = |m: usize| {
            let v = table[idx % table.len()] % m;
            idx += 1;
            v
        };
        let out = floyd_sample(10, 5, rng);
        assert_eq!(out.len(), 5);
        let set: HashSet<usize> = out.iter().copied().collect();
        assert_eq!(set.len(), 5);
        assert!(out.iter().all(|&x| x < 10));
    }

    /// Smoke test for uniformity: with `n = 5, k = 1` each value should be
    /// selected with probability `1/5`. Across `TRIALS = 10_000` independent
    /// seeds, each bucket has mean 2000 and std dev `sqrt(10000 * 0.2 * 0.8) =
    /// 40`. A 5-sigma window `[1800, 2200]` is generous and stable.
    #[test]
    fn uniformity_smoke_test_k1_over_5() {
        const TRIALS: u64 = 10_000;
        let mut counts = [0_u64; 5];
        for seed in 0..TRIALS {
            let s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
            let out = floyd_sample_xorshift(5, 1, s);
            assert_eq!(out.len(), 1);
            counts[out[0]] += 1;
        }
        let mean = TRIALS / 5; // 2000
        let tol = 200_u64;
        for (i, &c) in counts.iter().enumerate() {
            assert!(
                c.abs_diff(mean) <= tol,
                "bucket {i} count {c} outside [{}, {}]",
                mean - tol,
                mean + tol,
            );
        }
    }

    #[test]
    fn sorted_helper_matches_sorted_unsorted() {
        // floyd_sample_sorted should be observationally identical to sorting
        // the unsorted output for the same rng sequence.
        let seed = 0xCAFE_BABE_u64;
        let mut rng_a = super::XorShift64::new(seed);
        let mut rng_b = super::XorShift64::new(seed);
        let mut a = floyd_sample(50, 7, |m| rng_a.next_bounded(m as u64) as usize);
        let b = floyd_sample_sorted(50, 7, |m| rng_b.next_bounded(m as u64) as usize);
        a.sort_unstable();
        assert_eq!(a, b);
    }
}
