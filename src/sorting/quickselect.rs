//! Quickselect — kth order statistic in expected O(n) time.
//!
//! Selects the kth-smallest element (0-indexed) of an unordered slice without
//! fully sorting it. Uses Hoare partitioning and recurses only on the side
//! that contains index `k`, giving expected linear time. Worst case is O(n²)
//! when pivots split poorly.
//!
//! # Median-of-three pivot
//!
//! A naive first/last-element pivot collapses to O(n²) on already-sorted or
//! reverse-sorted inputs. Median-of-three picks the median of `lo`, `mid`,
//! and `hi`, which is sorted in practice and yields balanced partitions on
//! the common pathological inputs while staying O(1) per call.
//!
//! # Complexity
//!
//! - Time: O(n) average, O(n²) worst case.
//! - Space: O(log n) recursion depth on average; O(n) for the cloned working
//!   buffer (the input slice is not mutated).

/// Returns the kth-smallest element (0-indexed) of `values`, or `None` if
/// `values` is empty or `k >= values.len()`. The input slice is not modified.
pub fn quickselect<T: Ord + Clone>(values: &[T], k: usize) -> Option<T> {
    if k >= values.len() {
        return None;
    }
    let mut buf: Vec<T> = values.to_vec();
    let len = buf.len();
    select_in_place(&mut buf, 0, len - 1, k);
    Some(buf.swap_remove(k))
}

fn select_in_place<T: Ord>(buf: &mut [T], mut lo: usize, mut hi: usize, k: usize) {
    loop {
        if lo >= hi {
            return;
        }
        let p = partition(buf, lo, hi);
        match k.cmp(&p) {
            std::cmp::Ordering::Equal => return,
            std::cmp::Ordering::Less => {
                if p == 0 {
                    return;
                }
                hi = p - 1;
            }
            std::cmp::Ordering::Greater => {
                lo = p + 1;
            }
        }
    }
}

/// Lomuto partition with median-of-three pivot. Returns the final pivot index.
fn partition<T: Ord>(buf: &mut [T], lo: usize, hi: usize) -> usize {
    let mid = lo + (hi - lo) / 2;
    // Order buf[lo] <= buf[mid] <= buf[hi] so the median lands at `mid`.
    if buf[lo] > buf[mid] {
        buf.swap(lo, mid);
    }
    if buf[lo] > buf[hi] {
        buf.swap(lo, hi);
    }
    if buf[mid] > buf[hi] {
        buf.swap(mid, hi);
    }
    // Move chosen pivot to `hi` and run a standard Lomuto partition.
    buf.swap(mid, hi);
    let mut i = lo;
    for j in lo..hi {
        if buf[j] <= buf[hi] {
            buf.swap(i, j);
            i += 1;
        }
    }
    buf.swap(i, hi);
    i
}

#[cfg(test)]
mod tests {
    use super::quickselect;
    use quickcheck_macros::quickcheck;

    #[test]
    fn empty_is_none() {
        let v: Vec<i32> = vec![];
        assert_eq!(quickselect(&v, 0), None);
    }

    #[test]
    fn k_out_of_range_is_none() {
        let v = vec![3, 1, 2];
        assert_eq!(quickselect(&v, 3), None);
        assert_eq!(quickselect(&v, 99), None);
    }

    #[test]
    fn single_element() {
        let v = vec![42];
        assert_eq!(quickselect(&v, 0), Some(42));
    }

    #[test]
    fn k_zero_returns_min() {
        let v = vec![5, 2, 8, 1, 9, 3];
        assert_eq!(quickselect(&v, 0), Some(1));
    }

    #[test]
    fn k_last_returns_max() {
        let v = vec![5, 2, 8, 1, 9, 3];
        assert_eq!(quickselect(&v, v.len() - 1), Some(9));
    }

    #[test]
    fn median_odd_length() {
        let v = vec![5, 2, 8, 1, 9, 3, 7];
        // sorted: 1 2 3 5 7 8 9 -> median at index 3 is 5
        assert_eq!(quickselect(&v, 3), Some(5));
    }

    #[test]
    fn does_not_mutate_input() {
        let v = vec![5, 2, 8, 1, 9, 3];
        let snapshot = v.clone();
        let _ = quickselect(&v, 2);
        assert_eq!(v, snapshot);
    }

    #[test]
    fn duplicates() {
        let v = vec![4, 1, 4, 2, 4, 3, 4];
        // sorted: 1 2 3 4 4 4 4
        assert_eq!(quickselect(&v, 0), Some(1));
        assert_eq!(quickselect(&v, 1), Some(2));
        assert_eq!(quickselect(&v, 2), Some(3));
        assert_eq!(quickselect(&v, 3), Some(4));
        assert_eq!(quickselect(&v, 6), Some(4));
    }

    #[test]
    fn sorted_ascending_input() {
        let v: Vec<i32> = (0..20).collect();
        for k in 0..v.len() {
            assert_eq!(quickselect(&v, k), Some(k as i32));
        }
    }

    #[test]
    fn sorted_descending_input() {
        let v: Vec<i32> = (0..20).rev().collect();
        for k in 0..v.len() {
            assert_eq!(quickselect(&v, k), Some(k as i32));
        }
    }

    #[test]
    fn all_equal_input() {
        let v = vec![7; 10];
        for k in 0..v.len() {
            assert_eq!(quickselect(&v, k), Some(7));
        }
    }

    #[quickcheck]
    fn matches_sorted_index(input: Vec<i32>, k: usize) -> bool {
        // Bound n to keep quickcheck cheap.
        let v: Vec<i32> = input.into_iter().take(50).collect();
        let mut sorted = v.clone();
        sorted.sort();
        if v.is_empty() {
            return quickselect(&v, k).is_none();
        }
        let idx = k % v.len();
        quickselect(&v, idx) == Some(sorted[idx])
    }
}
