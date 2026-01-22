//! Sublist (subarray) search. Naive O(n·m) test for whether `pattern`
//! appears as a contiguous subsequence of `haystack`. KMP and Rabin–Karp
//! offer better worst-case bounds and are tracked separately.

/// Returns the start index of the first occurrence of `pattern` inside
/// `haystack`, or `None` if no occurrence exists.
///
/// An empty pattern matches at index 0 (consistent with `str::find`).
pub fn sublist_search<T: PartialEq>(haystack: &[T], pattern: &[T]) -> Option<usize> {
    let m = pattern.len();
    let n = haystack.len();
    if m == 0 {
        return Some(0);
    }
    if m > n {
        return None;
    }
    'outer: for start in 0..=n - m {
        for j in 0..m {
            if haystack[start + j] != pattern[j] {
                continue 'outer;
            }
        }
        return Some(start);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::sublist_search;

    #[test]
    fn empty_pattern_matches_at_zero() {
        let h = [1, 2, 3];
        let p: [i32; 0] = [];
        assert_eq!(sublist_search(&h, &p), Some(0));
    }

    #[test]
    fn empty_haystack_no_match() {
        let h: [i32; 0] = [];
        let p = [1];
        assert_eq!(sublist_search(&h, &p), None);
    }

    #[test]
    fn pattern_longer_than_haystack() {
        let h = [1, 2];
        let p = [1, 2, 3];
        assert_eq!(sublist_search(&h, &p), None);
    }

    #[test]
    fn middle_match() {
        let h = vec![1, 2, 3, 4, 5, 6];
        let p = vec![3, 4, 5];
        assert_eq!(sublist_search(&h, &p), Some(2));
    }

    #[test]
    fn end_match() {
        let h = "abracadabra".bytes().collect::<Vec<_>>();
        let p = "abra".bytes().collect::<Vec<_>>();
        assert_eq!(sublist_search(&h, &p), Some(0));
    }

    #[test]
    fn overlapping_prefix() {
        let h = vec![0, 0, 0, 0, 1];
        let p = vec![0, 0, 1];
        assert_eq!(sublist_search(&h, &p), Some(2));
    }

    #[test]
    fn no_match() {
        let h = vec![1, 2, 3, 4];
        let p = vec![5];
        assert_eq!(sublist_search(&h, &p), None);
    }
}
