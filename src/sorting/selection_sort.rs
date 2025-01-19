//! Selection sort. In-place, O(n²) for all inputs, not stable.

/// Sorts `slice` in non-decreasing order using selection sort.
pub fn selection_sort<T: Ord>(slice: &mut [T]) {
    let n = slice.len();
    for i in 0..n {
        let mut min_idx = i;
        for j in (i + 1)..n {
            if slice[j] < slice[min_idx] {
                min_idx = j;
            }
        }
        if min_idx != i {
            slice.swap(i, min_idx);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::selection_sort;
    use quickcheck_macros::quickcheck;

    #[test]
    fn empty_and_single() {
        let mut a: Vec<i32> = vec![];
        selection_sort(&mut a);
        assert!(a.is_empty());

        let mut b = vec![7];
        selection_sort(&mut b);
        assert_eq!(b, vec![7]);
    }

    #[test]
    fn negatives() {
        let mut v = vec![-5, 12, 0, -33, 7, 7, -1];
        selection_sort(&mut v);
        assert_eq!(v, vec![-33, -5, -1, 0, 7, 7, 12]);
    }

    #[test]
    fn already_sorted() {
        let mut v = vec![1, 2, 3, 4];
        selection_sort(&mut v);
        assert_eq!(v, vec![1, 2, 3, 4]);
    }

    #[quickcheck]
    fn matches_std_sort(mut input: Vec<i64>) -> bool {
        let mut expected = input.clone();
        expected.sort();
        selection_sort(&mut input);
        input == expected
    }
}
