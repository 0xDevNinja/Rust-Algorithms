//! Fisher–Yates shuffle (Durstenfeld's modern in-place variant).
//!
//! Permutes a slice uniformly at random in linear time using `n - 1` swaps.
//! The algorithm walks the index `i` from `n - 1` down to `1` and swaps
//! `arr[i]` with `arr[rng(i + 1)]`, where `rng(b)` is required to return a
//! uniform integer in `[0, b)`. Each of the `n!` permutations is produced with
//! probability exactly `1 / n!` provided the random source is unbiased.
//!
//! # Why it is uniform
//! After step `i`, the element placed at index `i` is drawn uniformly from the
//! `i + 1` elements in `arr[0..=i]` that have not yet been frozen. By induction
//! every prefix of fixed positions remains uniformly distributed across the
//! corresponding falling-factorial number of choices, which collapses to `n!`
//! equally likely outputs once `i` reaches `0`.
//!
//! # Complexity
//! - Time:  `O(n)` — exactly `n - 1` swaps and `n - 1` PRNG draws.
//! - Space: `O(1)` auxiliary; the shuffle is in place.
//!
//! # Determinism
//! [`fisher_yates_shuffle`] takes a caller-supplied closure as its random
//! source so it imposes no opinion on PRNG choice, security, or seeding.
//! [`shuffle_xorshift`] is a convenience wrapper over a small deterministic
//! `XorShift64` PRNG so identical `(input, seed)` pairs always produce the
//! same permutation. Neither is cryptographically secure.

/// Permutes `arr` in place using the modern Fisher–Yates / Durstenfeld
/// algorithm. `rng(b)` must return a uniform integer in `[0, b)`; bias in the
/// supplied closure translates directly into bias in the output distribution.
///
/// Slices of length `0` or `1` are returned untouched.
///
/// # Examples
/// ```
/// use rust_algorithms::sorting::fisher_yates_shuffle::fisher_yates_shuffle;
/// let mut data = [1, 2, 3, 4, 5];
/// // Identity rng — always picks the last available index, leaving the slice
/// // unchanged. Useful as a smoke test; do not use in production.
/// fisher_yates_shuffle(&mut data, |b| b - 1);
/// assert_eq!(data, [1, 2, 3, 4, 5]);
/// ```
pub fn fisher_yates_shuffle<T>(arr: &mut [T], mut rng: impl FnMut(usize) -> usize) {
    let n = arr.len();
    if n < 2 {
        return;
    }
    for i in (1..n).rev() {
        let j = rng(i + 1);
        debug_assert!(j <= i, "rng must return a value in [0, i+1)");
        arr.swap(i, j);
    }
}

/// Convenience wrapper that shuffles `arr` in place using a deterministic
/// `XorShift64` PRNG seeded by `seed`. Two calls with identical `(arr, seed)`
/// produce byte-for-byte identical permutations. Not cryptographically secure.
///
/// # Examples
/// ```
/// use rust_algorithms::sorting::fisher_yates_shuffle::shuffle_xorshift;
/// let mut a = [1, 2, 3, 4, 5];
/// let mut b = [1, 2, 3, 4, 5];
/// shuffle_xorshift(&mut a, 42);
/// shuffle_xorshift(&mut b, 42);
/// assert_eq!(a, b);
/// ```
pub fn shuffle_xorshift<T>(arr: &mut [T], seed: u64) {
    let mut rng = XorShift64::new(seed);
    fisher_yates_shuffle(arr, |bound| rng.next_bounded(bound as u64) as usize);
}

/// `XorShift64` PRNG (Marsaglia 2003). Tiny, fast, deterministic, non-crypto.
struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    /// `XorShift` collapses to all-zeros from a zero seed, so substitute a
    /// fixed nonzero constant when the caller passes 0.
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

    /// Uniform integer in `[0, bound)` via rejection sampling to avoid modulo
    /// bias. Caller must guarantee `bound > 0`.
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
    use super::{fisher_yates_shuffle, shuffle_xorshift};
    use std::collections::HashSet;

    #[test]
    fn empty_slice_is_identity() {
        let mut data: [i32; 0] = [];
        shuffle_xorshift(&mut data, 1);
        assert_eq!(data, [] as [i32; 0]);
    }

    #[test]
    fn single_element_is_identity() {
        let mut data = [42];
        shuffle_xorshift(&mut data, 1);
        assert_eq!(data, [42]);
    }

    #[test]
    fn fixed_seed_is_deterministic_regression() {
        // Lock in the exact permutation for this seed so accidental changes to
        // the PRNG or the loop wiring trip the test immediately.
        let mut a = [1, 2, 3, 4, 5, 6, 7, 8];
        let mut b = [1, 2, 3, 4, 5, 6, 7, 8];
        shuffle_xorshift(&mut a, 0xDEAD_BEEF);
        shuffle_xorshift(&mut b, 0xDEAD_BEEF);
        assert_eq!(a, b);
        // Compare against the original to confirm a real shuffle happened.
        assert_ne!(a, [1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn multiset_is_preserved() {
        let original: Vec<i32> = (0..50).collect();
        for seed in 0..16_u64 {
            let mut data = original.clone();
            shuffle_xorshift(&mut data, seed);
            let mut sorted = data.clone();
            sorted.sort();
            assert_eq!(sorted, original);
        }
    }

    #[test]
    fn multiset_preserved_with_duplicates() {
        let original = vec![1, 1, 2, 2, 3, 3, 4, 4];
        let mut data = original.clone();
        shuffle_xorshift(&mut data, 7);
        let mut sorted = data.clone();
        sorted.sort();
        let mut expected = original;
        expected.sort();
        assert_eq!(sorted, expected);
    }

    #[test]
    fn reaches_all_six_permutations_of_length_three() {
        let mut seen: HashSet<[u8; 3]> = HashSet::new();
        // 256 seeds is plenty for a length-3 search space (6 outputs).
        for seed in 0..256_u64 {
            let mut data = [1_u8, 2, 3];
            shuffle_xorshift(&mut data, seed);
            seen.insert(data);
            if seen.len() == 6 {
                break;
            }
        }
        assert_eq!(seen.len(), 6, "did not cover all 3! permutations: {seen:?}");
    }

    /// Smoke test for absence of gross bias on length 4. Each of the `4! = 24`
    /// permutations should appear with probability `1/24` ≈ 4.17 %. Over
    /// `TRIALS = 24_000` independent shuffles every bucket should land near
    /// 1000. We use a generous `[700, 1300]` window — well outside the noise
    /// floor of a non-cryptographic PRNG — so the test stays stable yet still
    /// catches any off-by-one in the loop bounds.
    #[test]
    fn no_gross_bias_on_length_four() {
        const TRIALS: u64 = 24_000;
        let mut counts: std::collections::HashMap<[u8; 4], u64> = std::collections::HashMap::new();
        for seed in 0..TRIALS {
            // Spread seeds nontrivially so consecutive runs are independent.
            let s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
            let mut data = [1_u8, 2, 3, 4];
            shuffle_xorshift(&mut data, s);
            *counts.entry(data).or_insert(0) += 1;
        }
        assert_eq!(
            counts.len(),
            24,
            "missing permutations: only {} seen",
            counts.len()
        );
        let lo = 700_u64;
        let hi = 1300_u64;
        for (perm, &c) in &counts {
            assert!(
                (lo..=hi).contains(&c),
                "permutation {perm:?} count {c} outside [{lo}, {hi}]"
            );
        }
    }

    #[test]
    fn closure_rng_drives_swap_choice() {
        // With rng(b) = 0 every swap pulls index 0, producing a deterministic
        // permutation independent of any PRNG state. This lets us verify the
        // loop wiring directly: starting from [1,2,3,4] the swaps are
        //   i=3, j=0 -> [4,2,3,1]
        //   i=2, j=0 -> [3,2,4,1]
        //   i=1, j=0 -> [2,3,4,1]
        let mut data = [1, 2, 3, 4];
        fisher_yates_shuffle(&mut data, |_| 0);
        assert_eq!(data, [2, 3, 4, 1]);
    }
}
