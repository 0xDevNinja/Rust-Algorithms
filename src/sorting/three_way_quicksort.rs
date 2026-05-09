//! Three-way (Bentley/McIlroy fat-pivot) quicksort. In-place.
//!
//! Partitions the active range into three contiguous regions —
//! `< pivot`, `= pivot`, `> pivot` — using Dijkstra's Dutch-National-Flag
//! scheme, then recurses only on the strict `<` and `>` regions. Equal
//! keys collapse into the middle band and are never touched again, so
//! duplicate-heavy inputs (the classic worst case for vanilla quicksort)
//! sort in O(n) total comparisons.
//!
//! Complexity:
//! * Comparisons: O(n log n) average, O(n) when the input has only
//!   k = O(1) distinct keys, O(n²) worst case on adversarial input.
//! * Space: O(log n) expected stack from recursion.
//! * Stability: not stable.
//!
//! Pivot selection is randomized via a deterministic `XorShift64` PRNG
//! seeded from the slice length, so runs are reproducible without
//! pulling in a `rand` dependency while still avoiding the trivial
//! pathological cases of a fixed-position pivot.
//!
//! Reference: Bentley & `McIlroy`, "Engineering a Sort Function" (1993);
//! Sedgewick, "Quicksort with 3-way partitioning".
//!
//! See [`super::quick_sort`] for the textbook Lomuto variant and
//! [`super::randomized_quicksort`] for randomized two-way partitioning.

/// Sorts `arr` in non-decreasing order using three-way (fat-pivot) quicksort.
///
/// Empty and single-element slices are no-ops. The algorithm performs
/// O(n) work when every element is equal — the headline property of the
/// 3-way partition.
pub fn three_way_quicksort<T: Ord + Clone>(arr: &mut [T]) {
    let len = arr.len();
    if len < 2 {
        return;
    }
    // Seed the pivot PRNG from the input length so a given input shape
    // sorts identically across runs but differently from inputs of a
    // different size.
    let mut rng = XorShift64::new(len as u64);
    sort_range(arr, 0, len - 1, &mut rng);
}

fn sort_range<T: Ord + Clone>(arr: &mut [T], lo: usize, hi: usize, rng: &mut XorShift64) {
    if lo >= hi {
        return;
    }
    // Random pivot to dodge the sorted/reverse-sorted worst case.
    let span = (hi - lo + 1) as u64;
    let pivot_idx = lo + rng.next_bounded(span) as usize;
    arr.swap(lo, pivot_idx);
    let pivot = arr[lo].clone();

    // Dijkstra Dutch-National-Flag invariant over [lo, hi]:
    //   arr[lo  ..lt] < pivot
    //   arr[lt  ..=gt] not yet classified  (i scans from left, gt from right)
    //   arr[lt  ..=  i - 1] == pivot       (after each iteration)
    //   arr[gt+1..= hi] > pivot
    let mut lt = lo;
    let mut gt = hi;
    let mut i = lo + 1;
    while i <= gt {
        match arr[i].cmp(&pivot) {
            std::cmp::Ordering::Less => {
                arr.swap(lt, i);
                lt += 1;
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                arr.swap(i, gt);
                // gt may underflow on the way down; guard the loop with `i <= gt`
                // and break out via the underflow check below.
                if gt == 0 {
                    break;
                }
                gt -= 1;
            }
            std::cmp::Ordering::Equal => {
                i += 1;
            }
        }
    }

    // Recurse only on the strict regions; the equal band is already done.
    if lt > 0 {
        sort_range(arr, lo, lt - 1, rng);
    }
    sort_range(arr, gt + 1, hi, rng);
}

/// `XorShift64` PRNG (Marsaglia 2003). Deterministic, non-crypto.
struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    /// Zero seed would collapse the generator; substitute a fixed nonzero
    /// constant in that case.
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

    /// Uniform integer in `[0, bound)` via rejection sampling. `bound > 0`.
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
    use super::three_way_quicksort;
    use quickcheck_macros::quickcheck;
    use std::cell::Cell;
    use std::cmp::Ordering;

    #[test]
    fn empty() {
        let mut v: Vec<i32> = vec![];
        three_way_quicksort(&mut v);
        assert!(v.is_empty());
    }

    #[test]
    fn single_element() {
        let mut v = vec![42];
        three_way_quicksort(&mut v);
        assert_eq!(v, vec![42]);
    }

    #[test]
    fn already_sorted() {
        let mut v: Vec<i32> = (0..64).collect();
        let expected = v.clone();
        three_way_quicksort(&mut v);
        assert_eq!(v, expected);
    }

    #[test]
    fn reverse_sorted() {
        let mut v: Vec<i32> = (0..64).rev().collect();
        let mut expected = v.clone();
        expected.sort();
        three_way_quicksort(&mut v);
        assert_eq!(v, expected);
    }

    #[test]
    fn all_equal() {
        let mut v = vec![7; 256];
        three_way_quicksort(&mut v);
        assert_eq!(v, vec![7; 256]);
    }

    #[test]
    fn many_duplicates() {
        // Three distinct keys, lots of repeats — the case 3-way is built for.
        let mut v: Vec<i32> = (0..300).map(|i| i % 3).collect();
        let mut expected = v.clone();
        expected.sort();
        three_way_quicksort(&mut v);
        assert_eq!(v, expected);
    }

    #[test]
    fn strings() {
        let mut v = vec![
            String::from("pear"),
            String::from("apple"),
            String::from("banana"),
            String::from("apple"),
            String::from("cherry"),
        ];
        let mut expected = v.clone();
        expected.sort();
        three_way_quicksort(&mut v);
        assert_eq!(v, expected);
    }

    #[test]
    fn negative_and_zero() {
        let mut v = vec![0, -1, 5, -3, 2, 0, -1, 5];
        let mut expected = v.clone();
        expected.sort();
        three_way_quicksort(&mut v);
        assert_eq!(v, expected);
    }

    /// Wrapper that counts comparisons, for verifying the O(n) all-equal property.
    #[derive(Clone)]
    struct Counted<'a> {
        value: i32,
        counter: &'a Cell<usize>,
    }

    impl PartialEq for Counted<'_> {
        fn eq(&self, other: &Self) -> bool {
            self.counter.set(self.counter.get() + 1);
            self.value == other.value
        }
    }

    impl Eq for Counted<'_> {}

    impl PartialOrd for Counted<'_> {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    impl Ord for Counted<'_> {
        fn cmp(&self, other: &Self) -> Ordering {
            self.counter.set(self.counter.get() + 1);
            self.value.cmp(&other.value)
        }
    }

    #[test]
    fn all_equal_is_linear_in_comparisons() {
        // On an all-equal input the 3-way partition makes a single sweep of the
        // active range and recurses on two empty subranges, so it does < 2n
        // comparisons total. A 2-way Lomuto quicksort would do ~n²/2.
        const N: usize = 1024;
        let counter = Cell::new(0usize);
        let mut v: Vec<Counted> = (0..N)
            .map(|_| Counted {
                value: 42,
                counter: &counter,
            })
            .collect();
        three_way_quicksort(&mut v);
        let cmps = counter.get();
        assert!(
            cmps < 2 * N,
            "expected < 2n = {} comparisons on all-equal input, got {}",
            2 * N,
            cmps
        );
    }

    #[quickcheck]
    fn matches_std_sort_i32(mut input: Vec<i32>) -> bool {
        let mut expected = input.clone();
        expected.sort();
        three_way_quicksort(&mut input);
        input == expected
    }

    #[quickcheck]
    fn matches_std_sort_strings(mut input: Vec<String>) -> bool {
        let mut expected = input.clone();
        expected.sort();
        three_way_quicksort(&mut input);
        input == expected
    }

    #[quickcheck]
    fn matches_std_sort_with_duplicates(input: Vec<u8>) -> bool {
        // u8 -> dense duplicate distribution, exercises the equal band.
        let mut got: Vec<u8> = input.clone();
        let mut expected = input;
        expected.sort();
        three_way_quicksort(&mut got);
        got == expected
    }
}
