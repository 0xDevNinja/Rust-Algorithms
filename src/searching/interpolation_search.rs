//! Interpolation search on a uniformly distributed sorted slice of integers.
//! O(log log n) on uniform data, O(n) worst case.

/// Returns the index of an element equal to `target` in a sorted `slice` of
/// `i64`, or `None`. Slice MUST be sorted in non-decreasing order.
pub fn interpolation_search(slice: &[i64], target: i64) -> Option<usize> {
    if slice.is_empty() {
        return None;
    }
    let (mut lo, mut hi) = (0_isize, slice.len() as isize - 1);
    while lo <= hi && target >= slice[lo as usize] && target <= slice[hi as usize] {
        if slice[lo as usize] == slice[hi as usize] {
            return if slice[lo as usize] == target {
                Some(lo as usize)
            } else {
                None
            };
        }
        let span = slice[hi as usize] - slice[lo as usize];
        let pos = lo + (((hi - lo) as i64) * (target - slice[lo as usize]) / span) as isize;
        if pos < lo || pos > hi {
            return None;
        }
        let pos_u = pos as usize;
        match slice[pos_u].cmp(&target) {
            std::cmp::Ordering::Equal => return Some(pos_u),
            std::cmp::Ordering::Less => lo = pos + 1,
            std::cmp::Ordering::Greater => hi = pos - 1,
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::interpolation_search;

    #[test]
    fn empty() {
        assert_eq!(interpolation_search(&[], 0), None);
    }

    #[test]
    fn uniform() {
        let v: Vec<i64> = (0..100).map(|i| i * 3).collect();
        assert_eq!(interpolation_search(&v, 0), Some(0));
        assert_eq!(interpolation_search(&v, 99), Some(33));
        assert_eq!(interpolation_search(&v, 297), Some(99));
        assert_eq!(interpolation_search(&v, 298), None);
    }

    #[test]
    fn duplicates_constant() {
        let v = vec![5_i64; 10];
        assert!(interpolation_search(&v, 5).is_some());
        assert_eq!(interpolation_search(&v, 6), None);
    }
}
