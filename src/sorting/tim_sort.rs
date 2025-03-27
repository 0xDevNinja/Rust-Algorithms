//! A simplified Tim sort: insertion-sort small runs of `MIN_RUN`, then
//! iteratively merge runs. Stable, O(n log n) worst case.

const MIN_RUN: usize = 32;

/// Sorts `slice` in non-decreasing order using a simplified Tim sort.
pub fn tim_sort<T: Ord + Clone>(slice: &mut [T]) {
    let n = slice.len();
    if n < 2 {
        return;
    }
    let mut start = 0;
    while start < n {
        let end = (start + MIN_RUN).min(n);
        insertion_sort_range(slice, start, end);
        start = end;
    }
    let mut size = MIN_RUN;
    while size < n {
        let mut left = 0;
        while left < n {
            let mid = (left + size).min(n);
            let right = (left + 2 * size).min(n);
            if mid < right {
                merge_runs(slice, left, mid, right);
            }
            left += 2 * size;
        }
        size *= 2;
    }
}

fn insertion_sort_range<T: Ord>(slice: &mut [T], lo: usize, hi: usize) {
    for i in (lo + 1)..hi {
        let mut j = i;
        while j > lo && slice[j - 1] > slice[j] {
            slice.swap(j - 1, j);
            j -= 1;
        }
    }
}

fn merge_runs<T: Ord + Clone>(slice: &mut [T], lo: usize, mid: usize, hi: usize) {
    let left: Vec<T> = slice[lo..mid].to_vec();
    let right: Vec<T> = slice[mid..hi].to_vec();
    let (mut i, mut j, mut k) = (0, 0, lo);
    while i < left.len() && j < right.len() {
        if left[i] <= right[j] {
            slice[k] = left[i].clone();
            i += 1;
        } else {
            slice[k] = right[j].clone();
            j += 1;
        }
        k += 1;
    }
    while i < left.len() {
        slice[k] = left[i].clone();
        i += 1;
        k += 1;
    }
    while j < right.len() {
        slice[k] = right[j].clone();
        j += 1;
        k += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::tim_sort;
    use quickcheck_macros::quickcheck;

    #[test]
    fn empty_and_single() {
        let mut a: Vec<i32> = vec![];
        tim_sort(&mut a);
        let mut b = vec![1];
        tim_sort(&mut b);
        assert!(a.is_empty());
        assert_eq!(b, vec![1]);
    }

    #[test]
    fn larger() {
        let mut v: Vec<i32> = (0..200).rev().collect();
        tim_sort(&mut v);
        assert_eq!(v, (0..200).collect::<Vec<_>>());
    }

    #[quickcheck]
    fn matches_std_sort(mut input: Vec<i32>) -> bool {
        let mut expected = input.clone();
        expected.sort();
        tim_sort(&mut input);
        input == expected
    }
}
