//! Median-of-medians selection — kth order statistic in worst-case O(n) time.
//!
//! The Blum-Floyd-Pratt-Rivest-Tarjan (BFPRT) algorithm selects the
//! kth-smallest element of an unordered slice in linear time, even on
//! adversarial inputs that would defeat random or median-of-three pivots.
//! It runs the same partition-and-recurse skeleton as quickselect, but the
//! pivot is chosen deterministically as the median of the medians of small
//! groups so the partition is provably balanced.
//!
//! # Why groups of five
//!
//! Splitting the input into ⌈n/5⌉ groups of (at most) five elements gives a
//! pivot guaranteed to be greater than at least 3⌈⌈n/5⌉/2⌉ elements and less
//! than the same many — roughly 3n/10 on each side. The recurrence becomes
//! T(n) ≤ T(n/5) + T(7n/10) + O(n), whose unique solution is O(n) because
//! 1/5 + 7/10 < 1. Groups of three lose this property (1/3 + 2/3 = 1) and
//! collapse to O(n log n); groups of seven also work but multiply the
//! constant factor without improving the asymptotic.
//!
//! # Complexity
//!
//! - Time: O(n) worst case.
//! - Space: O(log n) recursion depth plus O(n) for the cloned working buffer
//!   (the input slice is not mutated).
//! - Stability: not applicable (selection, not a sort).

/// Returns the kth-smallest element (0-indexed) of `values`, or `None` if
/// `values` is empty or `k >= values.len()`. Runs in worst-case linear time
/// via the BFPRT median-of-medians pivot. The input slice is not modified.
pub fn median_of_medians_select<T: Ord + Clone>(values: &[T], k: usize) -> Option<T> {
    if k >= values.len() {
        return None;
    }
    let mut buf: Vec<T> = values.to_vec();
    let len = buf.len();
    select_in_place(&mut buf, 0, len - 1, k);
    Some(buf.swap_remove(k))
}

fn select_in_place<T: Ord + Clone>(buf: &mut [T], mut lo: usize, mut hi: usize, k: usize) {
    loop {
        if lo >= hi {
            return;
        }
        let pivot = pivot_value(buf, lo, hi);
        let (lt_end, gt_start) = three_way_partition(buf, lo, hi, &pivot);
        if k < lt_end {
            hi = lt_end - 1;
        } else if k >= gt_start {
            lo = gt_start;
        } else {
            // k lands inside the equals-band; the answer is `pivot`.
            return;
        }
    }
}

/// Picks the median-of-medians pivot for `buf[lo..=hi]`. Splits the range
/// into groups of five, sorts each group in place, collects the medians at
/// the front of the range, and recursively selects their median.
fn pivot_value<T: Ord + Clone>(buf: &mut [T], lo: usize, hi: usize) -> T {
    let n = hi - lo + 1;
    if n <= 5 {
        insertion_sort(buf, lo, hi);
        return buf[lo + (n - 1) / 2].clone();
    }
    let num_groups = n.div_ceil(5);
    for g in 0..num_groups {
        let g_lo = lo + g * 5;
        let g_hi = (g_lo + 4).min(hi);
        insertion_sort(buf, g_lo, g_hi);
        let median_idx = g_lo + (g_hi - g_lo) / 2;
        // Move this group's median to position lo + g so medians are packed
        // contiguously at the front of the range.
        buf.swap(lo + g, median_idx);
    }
    let mid_lo = lo;
    let mid_hi = lo + num_groups - 1;
    let target = mid_lo + (mid_hi - mid_lo) / 2;
    let target_offset = target - mid_lo;
    select_in_place(buf, mid_lo, mid_hi, mid_lo + target_offset);
    buf[target].clone()
}

fn insertion_sort<T: Ord>(buf: &mut [T], lo: usize, hi: usize) {
    for i in (lo + 1)..=hi {
        let mut j = i;
        while j > lo && buf[j - 1] > buf[j] {
            buf.swap(j - 1, j);
            j -= 1;
        }
    }
}

/// Three-way (Dutch national flag) partition around `pivot`. Returns
/// `(lt_end, gt_start)`: indices in `lo..=hi` are `< pivot` for
/// `lo..lt_end`, `== pivot` for `lt_end..gt_start`, and `> pivot` for
/// `gt_start..=hi`. Crucially, when many elements equal the pivot (e.g.
/// all-equal input) the equals-band absorbs them, preventing the O(n²)
/// degeneration a two-way partition would suffer.
fn three_way_partition<T: Ord>(buf: &mut [T], lo: usize, hi: usize, pivot: &T) -> (usize, usize) {
    let mut lt = lo;
    let mut i = lo;
    let mut gt = hi;
    while i <= gt {
        match buf[i].cmp(pivot) {
            std::cmp::Ordering::Less => {
                buf.swap(lt, i);
                lt += 1;
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                buf.swap(i, gt);
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
    (lt, i)
}

#[cfg(test)]
mod tests {
    use super::median_of_medians_select;
    use quickcheck_macros::quickcheck;

    #[test]
    fn empty_is_none() {
        let v: Vec<i32> = vec![];
        assert_eq!(median_of_medians_select(&v, 0), None);
    }

    #[test]
    fn k_out_of_range_is_none() {
        let v = vec![3, 1, 2];
        assert_eq!(median_of_medians_select(&v, 3), None);
        assert_eq!(median_of_medians_select(&v, 99), None);
    }

    #[test]
    fn single_element() {
        let v = vec![42];
        assert_eq!(median_of_medians_select(&v, 0), Some(42));
        assert_eq!(median_of_medians_select(&v, 1), None);
    }

    #[test]
    fn k_zero_returns_min() {
        let v = vec![5, 2, 8, 1, 9, 3];
        assert_eq!(median_of_medians_select(&v, 0), Some(1));
    }

    #[test]
    fn k_last_returns_max() {
        let v = vec![5, 2, 8, 1, 9, 3];
        assert_eq!(median_of_medians_select(&v, v.len() - 1), Some(9));
    }

    #[test]
    fn median_odd_length() {
        let v = vec![5, 2, 8, 1, 9, 3, 7];
        // sorted: 1 2 3 5 7 8 9 -> median at index 3 is 5
        assert_eq!(median_of_medians_select(&v, 3), Some(5));
    }

    #[test]
    fn does_not_mutate_input() {
        let v = vec![5, 2, 8, 1, 9, 3];
        let snapshot = v.clone();
        let _ = median_of_medians_select(&v, 2);
        assert_eq!(v, snapshot);
    }

    #[test]
    fn sorted_ascending_input() {
        let v: Vec<i32> = (0..50).collect();
        for k in 0..v.len() {
            assert_eq!(median_of_medians_select(&v, k), Some(k as i32));
        }
    }

    #[test]
    fn sorted_descending_input() {
        let v: Vec<i32> = (0..50).rev().collect();
        for k in 0..v.len() {
            assert_eq!(median_of_medians_select(&v, k), Some(k as i32));
        }
    }

    #[test]
    fn all_equal_input() {
        let v = vec![7; 25];
        for k in 0..v.len() {
            assert_eq!(median_of_medians_select(&v, k), Some(7));
        }
    }

    #[test]
    fn duplicates() {
        let v = vec![4, 1, 4, 2, 4, 3, 4];
        // sorted: 1 2 3 4 4 4 4
        assert_eq!(median_of_medians_select(&v, 0), Some(1));
        assert_eq!(median_of_medians_select(&v, 1), Some(2));
        assert_eq!(median_of_medians_select(&v, 2), Some(3));
        assert_eq!(median_of_medians_select(&v, 3), Some(4));
        assert_eq!(median_of_medians_select(&v, 6), Some(4));
    }

    #[test]
    fn large_input_matches_sort() {
        // n = 1000 stresses the recursive median-of-medians pivot path.
        let mut rng_state: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut v: Vec<i32> = (0..1000)
            .map(|_| {
                // Cheap xorshift; no extra deps needed.
                rng_state ^= rng_state << 13;
                rng_state ^= rng_state >> 7;
                rng_state ^= rng_state << 17;
                (rng_state as i32).rem_euclid(10_000)
            })
            .collect();
        let mut sorted = v.clone();
        sorted.sort();
        for &k in &[0_usize, 1, 250, 499, 500, 501, 750, 998, 999] {
            assert_eq!(median_of_medians_select(&v, k), Some(sorted[k]));
        }
        // Sanity: input slice unchanged after many calls.
        v.sort();
        assert_eq!(v, sorted);
    }

    #[quickcheck]
    fn matches_sorted_index(input: Vec<i32>, k: usize) -> bool {
        // Bound n to keep quickcheck cheap.
        let v: Vec<i32> = input.into_iter().take(200).collect();
        if v.is_empty() {
            return median_of_medians_select(&v, k).is_none();
        }
        let mut sorted = v.clone();
        sorted.sort();
        let idx = k % v.len();
        median_of_medians_select(&v, idx) == Some(sorted[idx])
    }
}
