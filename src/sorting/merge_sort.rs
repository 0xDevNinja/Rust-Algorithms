//! Merge sort. Stable, O(n log n) worst-case, O(n) auxiliary space.

/// Sorts `slice` in non-decreasing order using top-down merge sort.
pub fn merge_sort<T: Ord + Clone>(slice: &mut [T]) {
    let n = slice.len();
    if n < 2 {
        return;
    }
    let mid = n / 2;
    merge_sort(&mut slice[..mid]);
    merge_sort(&mut slice[mid..]);
    let merged = merge(&slice[..mid], &slice[mid..]);
    slice.clone_from_slice(&merged);
}

fn merge<T: Ord + Clone>(left: &[T], right: &[T]) -> Vec<T> {
    let mut out = Vec::with_capacity(left.len() + right.len());
    let (mut i, mut j) = (0, 0);
    while i < left.len() && j < right.len() {
        if left[i] <= right[j] {
            out.push(left[i].clone());
            i += 1;
        } else {
            out.push(right[j].clone());
            j += 1;
        }
    }
    out.extend_from_slice(&left[i..]);
    out.extend_from_slice(&right[j..]);
    out
}

#[cfg(test)]
mod tests {
    use super::merge_sort;
    use quickcheck_macros::quickcheck;

    #[test]
    fn empty_and_single() {
        let mut a: Vec<i32> = vec![];
        merge_sort(&mut a);
        assert!(a.is_empty());

        let mut b = vec![1];
        merge_sort(&mut b);
        assert_eq!(b, vec![1]);
    }

    #[test]
    fn random() {
        let mut v = vec![5, 2, 4, 6, 1, 3, 2, 6];
        merge_sort(&mut v);
        assert_eq!(v, vec![1, 2, 2, 3, 4, 5, 6, 6]);
    }

    #[quickcheck]
    fn matches_std_sort(mut input: Vec<i32>) -> bool {
        let mut expected = input.clone();
        expected.sort();
        merge_sort(&mut input);
        input == expected
    }
}
