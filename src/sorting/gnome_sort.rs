//! Gnome sort. In-place, stable, O(n²) worst/average, O(n) best (already sorted).
//!
//! Walks forward when adjacent elements are in order and backwards when they
//! are not, swapping until they are. Equivalent to insertion sort but with a
//! single index rather than nested loops.

/// Sorts `slice` in non-decreasing order using gnome sort.
pub fn gnome_sort<T: Ord>(slice: &mut [T]) {
    let n = slice.len();
    let mut i = 0;
    while i < n {
        if i == 0 || slice[i - 1] <= slice[i] {
            i += 1;
        } else {
            slice.swap(i - 1, i);
            i -= 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::gnome_sort;
    use quickcheck_macros::quickcheck;

    #[test]
    fn empty() {
        let mut v: Vec<i32> = vec![];
        gnome_sort(&mut v);
        assert!(v.is_empty());
    }

    #[test]
    fn single() {
        let mut v = vec![7];
        gnome_sort(&mut v);
        assert_eq!(v, vec![7]);
    }

    #[test]
    fn already_sorted() {
        let mut v = vec![1, 2, 3, 4, 5];
        gnome_sort(&mut v);
        assert_eq!(v, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn reverse() {
        let mut v = vec![5, 4, 3, 2, 1];
        gnome_sort(&mut v);
        assert_eq!(v, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn duplicates_preserve_relative_order() {
        let mut v = vec![(2, 'a'), (1, 'b'), (2, 'c'), (1, 'd')];
        gnome_sort(&mut v);
        assert_eq!(v, vec![(1, 'b'), (1, 'd'), (2, 'a'), (2, 'c')]);
    }

    #[quickcheck]
    fn matches_std_sort(mut input: Vec<i32>) -> bool {
        let mut expected = input.clone();
        expected.sort();
        gnome_sort(&mut input);
        input == expected
    }
}
