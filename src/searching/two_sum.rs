//! Two-sum on an unsorted slice via a hash-map of seen values.
//!
//! Given a slice `nums` and a `target`, find indices `(i, j)` with `i < j`
//! such that `nums[i] + nums[j] == target`. Unlike the sorted two-pointers
//! variant in [`crate::searching::two_pointers`], this routine does not
//! require the input to be sorted: it streams the slice once, recording each
//! value and its index in a `HashMap<i64, usize>`. For each new element `x`
//! at index `j` it looks up `target - x`; on a hit the previously recorded
//! index `i < j` together with `j` form the answer.
//!
//! - Time: `O(n)` expected (single pass, average-case `O(1)` map ops).
//! - Space: `O(n)` for the auxiliary map.

use std::collections::HashMap;

/// Returns the first pair of indices `(i, j)` with `i < j` such that
/// `nums[i] + nums[j] == target`, or `None` if no such pair exists.
///
/// "First" is defined by the smaller right index `j`: we scan left-to-right
/// and report as soon as we discover the complementary value already seen.
/// On ties at the same `j`, the smallest valid `i` (the earliest occurrence
/// of the complement) is returned because the map records each value's
/// first index and is not overwritten on later occurrences.
///
/// # Examples
///
/// ```
/// use rust_algorithms::searching::two_sum::two_sum;
///
/// assert_eq!(two_sum(&[2, 7, 11, 15], 9), Some((0, 1)));
/// assert_eq!(two_sum(&[3, 3], 6), Some((0, 1)));
/// assert_eq!(two_sum(&[1, 2, 3], 7), None);
/// ```
pub fn two_sum(nums: &[i64], target: i64) -> Option<(usize, usize)> {
    let mut seen: HashMap<i64, usize> = HashMap::with_capacity(nums.len());
    for (j, &x) in nums.iter().enumerate() {
        let complement = target.checked_sub(x)?;
        if let Some(&i) = seen.get(&complement) {
            return Some((i, j));
        }
        // Record only the first occurrence of `x` so that the returned `i`
        // is as small as possible.
        seen.entry(x).or_insert(j);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::two_sum;

    #[test]
    fn empty_returns_none() {
        assert_eq!(two_sum(&[], 0), None);
    }

    #[test]
    fn singleton_returns_none() {
        assert_eq!(two_sum(&[5], 5), None);
        assert_eq!(two_sum(&[5], 10), None);
    }

    #[test]
    fn no_solution_returns_none() {
        assert_eq!(two_sum(&[1, 2, 3, 4], 100), None);
        assert_eq!(two_sum(&[1, 2, 3, 4], -1), None);
    }

    #[test]
    fn classic_example() {
        assert_eq!(two_sum(&[2, 7, 11, 15], 9), Some((0, 1)));
    }

    #[test]
    fn duplicates_pair_with_themselves() {
        assert_eq!(two_sum(&[3, 3], 6), Some((0, 1)));
    }

    #[test]
    fn duplicate_value_does_not_self_pair_when_unneeded() {
        // target=8 needs 5+3; the two 3s should not be treated as a pair.
        assert_eq!(two_sum(&[3, 5, 3], 8), Some((0, 1)));
    }

    #[test]
    fn negative_numbers() {
        assert_eq!(two_sum(&[-1, -2, -3, -4], -7), Some((2, 3)));
        assert_eq!(two_sum(&[-3, 4, 3, 90], 0), Some((0, 2)));
    }

    #[test]
    fn target_zero() {
        assert_eq!(two_sum(&[-5, 1, 2, 5], 0), Some((0, 3)));
        assert_eq!(two_sum(&[0, 0, 1], 0), Some((0, 1)));
        assert_eq!(two_sum(&[1, 2, 3], 0), None);
    }

    #[test]
    fn returns_first_pair_by_right_index() {
        // Both (0,3) and (1,2) sum to 5; the algorithm reports the pair
        // discovered first by scanning, which is (1, 2) (j=2).
        let nums = [1, 2, 3, 4];
        let (i, j) = two_sum(&nums, 5).unwrap();
        assert!(i < j);
        assert_eq!(nums[i] + nums[j], 5);
        assert_eq!((i, j), (1, 2));
    }

    #[test]
    fn extreme_values_do_not_overflow() {
        // target - x must not overflow; large negative target with a large
        // positive element would underflow without checked arithmetic, but
        // our implementation returns None instead of panicking.
        let nums = [i64::MAX, 0, 1];
        assert_eq!(two_sum(&nums, i64::MIN), None);
    }
}
