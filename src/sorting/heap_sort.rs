//! Heap sort. In-place, O(n log n) for all inputs, not stable.

/// Sorts `slice` in non-decreasing order using a binary max-heap.
pub fn heap_sort<T: Ord>(slice: &mut [T]) {
    let n = slice.len();
    if n < 2 {
        return;
    }
    // Build max-heap.
    for i in (0..n / 2).rev() {
        sift_down(slice, i, n);
    }
    // Extract max repeatedly.
    for end in (1..n).rev() {
        slice.swap(0, end);
        sift_down(slice, 0, end);
    }
}

fn sift_down<T: Ord>(slice: &mut [T], mut root: usize, end: usize) {
    loop {
        let left = 2 * root + 1;
        if left >= end {
            return;
        }
        let right = left + 1;
        let mut largest = root;
        if slice[left] > slice[largest] {
            largest = left;
        }
        if right < end && slice[right] > slice[largest] {
            largest = right;
        }
        if largest == root {
            return;
        }
        slice.swap(root, largest);
        root = largest;
    }
}

#[cfg(test)]
mod tests {
    use super::heap_sort;
    use quickcheck_macros::quickcheck;

    #[test]
    fn examples() {
        let mut v = vec![12, 11, 13, 5, 6, 7];
        heap_sort(&mut v);
        assert_eq!(v, vec![5, 6, 7, 11, 12, 13]);
    }

    #[test]
    fn empty_and_single() {
        let mut a: Vec<i32> = vec![];
        heap_sort(&mut a);
        let mut b = vec![1];
        heap_sort(&mut b);
        assert!(a.is_empty());
        assert_eq!(b, vec![1]);
    }

    #[quickcheck]
    fn matches_std_sort(mut input: Vec<i32>) -> bool {
        let mut expected = input.clone();
        expected.sort();
        heap_sort(&mut input);
        input == expected
    }
}
