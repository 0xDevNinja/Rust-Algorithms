//! Randomized quicksort. In-place, O(n log n) expected, O(n²) worst case.
//!
//! Pivot index is drawn from a deterministic `XorShift64` PRNG seeded by the
//! caller. Random pivoting makes the bad O(n²) case vanishingly unlikely on
//! adversarial / already-sorted inputs that defeat the fixed-pivot
//! [`quick_sort`](super::quick_sort) variant, while keeping the expected work
//! at O(n log n) and recursion depth at O(log n) expected.
//!
//! Partitioning uses the Lomuto scheme — chosen for its simplicity and the
//! clean separation between "pick a pivot" and "partition around it". The
//! randomly selected pivot is swapped to the end of the active range first,
//! so the partition routine is identical to a textbook last-element Lomuto.
//!
//! Determinism: the same `seed` and the same input always produce the same
//! sequence of pivot picks, so runs are reproducible. Pass distinct seeds
//! across runs (e.g. system time) when reproducibility is not desired.
//!
//! Space: O(log n) expected stack from recursion. Not stable.

/// Sorts `values` in non-decreasing order using randomized quicksort.
///
/// `seed` drives the internal `XorShift64` PRNG used to pick pivots; the same
/// `(values, seed)` pair is fully deterministic. Empty and single-element
/// slices are no-ops.
pub fn randomized_quicksort<T: Ord>(values: &mut [T], seed: u64) {
    let len = values.len();
    if len < 2 {
        return;
    }
    let mut rng = XorShift64::new(seed);
    sort_range(values, 0, len - 1, &mut rng);
}

fn sort_range<T: Ord>(values: &mut [T], lo: usize, hi: usize, rng: &mut XorShift64) {
    if lo >= hi {
        return;
    }
    // Pick a uniform pivot in [lo, hi] and move it to `hi` so partition can
    // use the standard last-element Lomuto scheme.
    let span = (hi - lo + 1) as u64;
    let pivot = lo + rng.next_bounded(span) as usize;
    values.swap(pivot, hi);

    let p = partition(values, lo, hi);
    if p > 0 {
        sort_range(values, lo, p - 1, rng);
    }
    sort_range(values, p + 1, hi, rng);
}

fn partition<T: Ord>(values: &mut [T], lo: usize, hi: usize) -> usize {
    let mut i = lo;
    for j in lo..hi {
        if values[j] <= values[hi] {
            values.swap(i, j);
            i += 1;
        }
    }
    values.swap(i, hi);
    i
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
    use super::randomized_quicksort;
    use quickcheck_macros::quickcheck;

    const SEED: u64 = 0x00C0_FFEE_BABE;

    #[test]
    fn empty() {
        let mut v: Vec<i32> = vec![];
        randomized_quicksort(&mut v, SEED);
        assert!(v.is_empty());
    }

    #[test]
    fn single_element() {
        let mut v = vec![42];
        randomized_quicksort(&mut v, SEED);
        assert_eq!(v, vec![42]);
    }

    #[test]
    fn already_sorted() {
        let mut v: Vec<i32> = (0..50).collect();
        let expected = v.clone();
        randomized_quicksort(&mut v, SEED);
        assert_eq!(v, expected);
    }

    #[test]
    fn reverse_sorted() {
        let mut v: Vec<i32> = (0..50).rev().collect();
        let mut expected = v.clone();
        expected.sort();
        randomized_quicksort(&mut v, SEED);
        assert_eq!(v, expected);
    }

    #[test]
    fn all_equal() {
        let mut v = vec![7; 100];
        randomized_quicksort(&mut v, SEED);
        assert_eq!(v, vec![7; 100]);
    }

    #[test]
    fn duplicates() {
        let mut v = vec![3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5];
        let mut expected = v.clone();
        expected.sort();
        randomized_quicksort(&mut v, SEED);
        assert_eq!(v, expected);
    }

    #[test]
    fn alternating() {
        let mut v: Vec<i32> = (0..100).map(|i| i32::from(i % 2 != 0)).collect();
        let mut expected = v.clone();
        expected.sort();
        randomized_quicksort(&mut v, SEED);
        assert_eq!(v, expected);
    }

    #[test]
    fn large_n() {
        // 10_000 elements seeded deterministically with a homemade XorShift
        // so the test stays reproducible without `rand`.
        let mut state: u64 = 0xDEAD_BEEF_CAFE_F00D;
        let mut v: Vec<i32> = (0..10_000)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                state as i32
            })
            .collect();
        let mut expected = v.clone();
        expected.sort();
        randomized_quicksort(&mut v, SEED);
        assert_eq!(v, expected);
    }

    #[test]
    fn same_seed_is_deterministic() {
        // Same seed + same input must produce byte-identical results.
        let original: Vec<i32> = vec![5, -2, 11, 3, 0, 7, -8, 4, 9, 1, -3, 6, 2, 10, -1];
        let mut a = original.clone();
        let mut b = original.clone();
        randomized_quicksort(&mut a, 0x1234_5678_9ABC_DEF0);
        randomized_quicksort(&mut b, 0x1234_5678_9ABC_DEF0);
        assert_eq!(a, b);

        // And of course the result is sorted.
        let mut expected = original;
        expected.sort();
        assert_eq!(a, expected);
    }

    #[quickcheck]
    fn matches_std_sort(input: Vec<i32>) -> bool {
        if input.len() > 100 {
            return true;
        }
        let mut got = input.clone();
        let mut expected = input;
        expected.sort();
        randomized_quicksort(&mut got, 0xA5A5_5A5A_A5A5_5A5A);
        got == expected
    }
}
