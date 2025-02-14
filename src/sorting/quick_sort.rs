//! Quick sort. In-place, O(n log n) average, O(n²) worst (degenerate pivots).
//!
//! Uses Lomuto partitioning with the last element as pivot. Adequate for
//! random data; pathological inputs (already sorted) hit the worst case.

/// Sorts `slice` in non-decreasing order using quick sort.
pub fn quick_sort<T: Ord>(slice: &mut [T]) {
    let len = slice.len();
    if len > 1 {
        sort_range(slice, 0, len - 1);
    }
}

fn sort_range<T: Ord>(slice: &mut [T], lo: usize, hi: usize) {
    if lo >= hi {
        return;
    }
    let p = partition(slice, lo, hi);
    if p > 0 {
        sort_range(slice, lo, p - 1);
    }
    sort_range(slice, p + 1, hi);
}

fn partition<T: Ord>(slice: &mut [T], lo: usize, hi: usize) -> usize {
    let mut i = lo;
    for j in lo..hi {
        if slice[j] <= slice[hi] {
            slice.swap(i, j);
            i += 1;
        }
    }
    slice.swap(i, hi);
    i
}

#[cfg(test)]
mod tests {
    use super::quick_sort;
    use quickcheck_macros::quickcheck;

    #[test]
    fn empty() {
        let mut v: Vec<i32> = vec![];
        quick_sort(&mut v);
        assert!(v.is_empty());
    }

    #[test]
    fn random() {
        let mut v = vec![10, 7, 8, 9, 1, 5];
        quick_sort(&mut v);
        assert_eq!(v, vec![1, 5, 7, 8, 9, 10]);
    }

    #[test]
    fn duplicates() {
        let mut v = vec![3, 3, 3, 3];
        quick_sort(&mut v);
        assert_eq!(v, vec![3, 3, 3, 3]);
    }

    #[quickcheck]
    fn matches_std_sort(mut input: Vec<i32>) -> bool {
        let mut expected = input.clone();
        expected.sort();
        quick_sort(&mut input);
        input == expected
    }
}
