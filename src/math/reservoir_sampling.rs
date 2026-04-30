//! Reservoir sampling — Vitter's Algorithm R.
//!
//! Selects `k` items uniformly at random from a stream of unknown length in a
//! single pass, using only `O(k)` auxiliary space. The stream is consumed once
//! as a generic [`Iterator`]; its length need not be known in advance.
//!
//! # Algorithm R
//! 1. Fill the reservoir with the first `k` items of the stream.
//! 2. For every subsequent item at 0-indexed position `i` (`i >= k`), draw a
//!    uniform integer `j` in `[0, i]`. If `j < k`, replace `reservoir[j]` with
//!    the new item. Otherwise discard the new item.
//!
//! # Uniform-probability property
//! After processing a stream of length `n >= k`, every `k`-subset of the input
//! appears in the reservoir with probability `1 / C(n, k)`, equivalently every
//! individual element is retained with probability `k / n`. Proof is by
//! induction on `n`: see Vitter, "Random Sampling with a Reservoir" (1985).
//!
//! # Complexity
//! - Time:  `O(n)` for a stream of length `n`, with one PRNG draw per item past
//!   the first `k`.
//! - Space: `O(k)` for the reservoir plus a fixed-size PRNG state.
//!
//! # Determinism
//! Sampling is driven by a small `XorShift64` PRNG seeded from the caller-supplied
//! `seed`, so two calls with identical `(stream, k, seed)` produce byte-for-byte
//! identical output. This is intentional — it keeps the public API free of any
//! `rand` dependency and makes tests reproducible. The PRNG is *not*
//! cryptographically secure and must not be used for security-sensitive work.

/// Returns up to `k` items sampled uniformly at random from `stream` using
/// Vitter's Algorithm R. The stream is consumed once.
///
/// If the stream yields fewer than `k` items, every item is returned (in stream
/// order). If `k == 0`, an empty `Vec` is returned without consuming the
/// stream. Sampling is deterministic in `seed`: same inputs produce the same
/// output.
///
/// # Examples
/// ```
/// use rust_algorithms::math::reservoir_sampling::reservoir_sample;
/// let sample = reservoir_sample(0..100, 5, 42);
/// assert_eq!(sample.len(), 5);
/// assert!(sample.iter().all(|x| (0..100).contains(x)));
/// ```
pub fn reservoir_sample<T: Clone, I: Iterator<Item = T>>(stream: I, k: usize, seed: u64) -> Vec<T> {
    if k == 0 {
        return Vec::new();
    }

    let mut reservoir: Vec<T> = Vec::with_capacity(k);
    let mut rng = XorShift64::new(seed);

    for (i, item) in stream.enumerate() {
        if i < k {
            reservoir.push(item);
        } else {
            // Draw j uniformly in [0, i]. If j < k, replace reservoir[j].
            let j = rng.next_bounded((i as u64) + 1) as usize;
            if j < k {
                reservoir[j] = item;
            }
        }
    }

    reservoir
}

/// `XorShift64` PRNG (Marsaglia 2003). Tiny, fast, deterministic, non-crypto.
struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    /// Seed cannot be zero — `XorShift` collapses to all-zeros from a zero seed.
    /// Substitute a fixed nonzero constant if the caller passes 0.
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
        // Largest multiple of `bound` that fits in u64; reject anything above.
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
    use super::reservoir_sample;
    use quickcheck_macros::quickcheck;
    use std::collections::HashSet;

    #[test]
    fn empty_stream_returns_empty() {
        let out: Vec<i32> = reservoir_sample(std::iter::empty(), 5, 1);
        assert!(out.is_empty());
    }

    #[test]
    fn k_zero_returns_empty() {
        let out = reservoir_sample(0..100, 0, 1);
        assert!(out.is_empty());
    }

    #[test]
    fn stream_shorter_than_k_returns_full_stream() {
        let out = reservoir_sample(0..3_i32, 10, 1);
        assert_eq!(out, vec![0, 1, 2]);
    }

    #[test]
    fn stream_longer_than_k_returns_exactly_k() {
        let out = reservoir_sample(0..1000_i32, 7, 1);
        assert_eq!(out.len(), 7);
        assert!(out.iter().all(|x| (0..1000).contains(x)));
    }

    #[test]
    fn output_items_come_from_stream() {
        let out = reservoir_sample(0..50_i32, 5, 12345);
        let source: HashSet<i32> = (0..50).collect();
        for x in &out {
            assert!(source.contains(x));
        }
    }

    #[test]
    fn deterministic_in_seed() {
        let a = reservoir_sample(0..1000_i32, 10, 0xDEAD_BEEF);
        let b = reservoir_sample(0..1000_i32, 10, 0xDEAD_BEEF);
        assert_eq!(a, b);
    }

    #[test]
    fn different_seeds_usually_differ() {
        // Not a hard guarantee, but with k=10 over 1000 elements the chance of
        // an exact collision is negligible.
        let a = reservoir_sample(0..1000_i32, 10, 1);
        let b = reservoir_sample(0..1000_i32, 10, 2);
        assert_ne!(a, b);
    }

    #[test]
    fn k_one_returns_single_item_from_stream() {
        let out = reservoir_sample(0..100_i32, 1, 7);
        assert_eq!(out.len(), 1);
        assert!((0..100).contains(&out[0]));
    }

    /// Smoke test for uniformity: with `k = 1` over the values `1..=10`, each
    /// value should be selected with probability `1/10`. Across `TRIALS = 10_000`
    /// independent runs (varying the seed) each value's count is ~1000 with
    /// standard deviation `sqrt(TRIALS * p * (1-p)) ≈ 30`. A 3-sigma window
    /// `[910, 1090]` should contain every count with probability `> 99.7 %` per
    /// bucket. We use a slightly looser bound to keep the test stable across
    /// platforms.
    #[test]
    fn uniformity_smoke_test_k1_over_10() {
        const TRIALS: u64 = 10_000;
        let mut counts = [0_u64; 10];
        for seed in 0..TRIALS {
            // Vary the seed nontrivially so consecutive runs are independent.
            let s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
            let out = reservoir_sample(1..=10_i32, 1, s);
            assert_eq!(out.len(), 1);
            counts[(out[0] - 1) as usize] += 1;
        }
        let mean = TRIALS / 10; // 1000
                                // 3 sigma ≈ 90; pad to 120 to leave headroom for a non-CSPRNG.
        let tol = 120_u64;
        for (i, &c) in counts.iter().enumerate() {
            assert!(
                c.abs_diff(mean) <= tol,
                "bucket {i} count {c} outside [{}, {}]",
                mean - tol,
                mean + tol,
            );
        }
    }

    #[quickcheck]
    fn prop_output_size_is_min_n_k(n: u8, k: u8, seed: u64) -> bool {
        // Bound n,k to keep test cheap; u8 already does that.
        let n = n as usize;
        let k = k as usize;
        let stream = 0..(n as i32);
        let out = reservoir_sample(stream, k, seed);
        out.len() == n.min(k)
    }

    #[quickcheck]
    fn prop_output_items_are_distinct_and_from_input(n: u8, k: u8, seed: u64) -> bool {
        let n = n as usize;
        let k = k as usize;
        let stream = 0..(n as i32);
        let out = reservoir_sample(stream, k, seed);
        let source: HashSet<i32> = (0..(n as i32)).collect();
        let unique: HashSet<i32> = out.iter().copied().collect();
        unique.len() == out.len() && out.iter().all(|x| source.contains(x))
    }
}
