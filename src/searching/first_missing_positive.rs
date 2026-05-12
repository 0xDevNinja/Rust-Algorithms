//! First missing positive integer via in-place cyclic sort.
//!
//! Given a slice `nums`, return the smallest positive integer (>= 1) that
//! does **not** appear in it. The naive solutions either sort the slice
//! (`O(n log n)`) or build an auxiliary set (`O(n)` time, `O(n)` extra
//! space). Cyclic sort solves the problem in `O(n)` time and `O(1)`
//! auxiliary space by rearranging the input itself so that every value `v`
//! with `1 <= v <= n` lands at index `v - 1`.
//!
//! The key observation: in a slice of length `n`, the answer always lies in
//! `1..=n + 1`. Values outside `[1, n]` are irrelevant for placement and
//! can be ignored. After placing every in-range value at its target index,
//! the first index `i` whose slot is wrong (`nums[i] != i + 1`) gives the
//! answer `i + 1`. If every slot is correct, the answer is `n + 1`.
//!
//! - Time: `O(n)`. Each successful swap places a value at its final home,
//!   so the total number of swaps across the outer loop is at most `n`.
//! - Space: `O(1)` auxiliary; the rearrangement is performed in place.

/// Returns the smallest positive integer that is not present in `nums`.
///
/// The slice is rearranged in place during the computation: each value `v`
/// with `1 <= v <= n` (where `n = nums.len()`) is moved to index `v - 1`
/// using a sequence of swaps. After the placement pass, the slice is
/// scanned once to locate the first index `i` whose slot does not hold
/// `i + 1`; that index yields the answer. If every slot is correct, the
/// answer is `n + 1`.
///
/// # Examples
///
/// ```
/// use rust_algorithms::searching::first_missing_positive::first_missing_positive;
///
/// assert_eq!(first_missing_positive(&mut []), 1);
/// assert_eq!(first_missing_positive(&mut [1, 2, 3]), 4);
/// assert_eq!(first_missing_positive(&mut [3, 4, -1, 1]), 2);
/// assert_eq!(first_missing_positive(&mut [7, 8, 9, 11, 12]), 1);
/// ```
pub fn first_missing_positive(nums: &mut [i32]) -> i32 {
    let n = nums.len();
    let mut i = 0;
    while i < n {
        let v = nums[i];
        // A value `v` is in range when `1 <= v <= n`. Use an `i64` cast
        // only conceptually; here we check non-negativity and the upper
        // bound against `n` as `i32` (safe because `n <= isize::MAX`).
        if v >= 1 && (v as i64) <= n as i64 {
            let target = (v - 1) as usize;
            // Only swap when the destination does not already hold `v`;
            // otherwise we would loop forever on duplicates.
            if nums[target] != v {
                nums.swap(i, target);
                continue;
            }
        }
        i += 1;
    }

    for (idx, &val) in nums.iter().enumerate() {
        if val != idx as i32 + 1 {
            return idx as i32 + 1;
        }
    }
    n as i32 + 1
}

#[cfg(test)]
mod tests {
    use super::first_missing_positive;

    #[test]
    fn empty_slice_returns_one() {
        let mut nums: [i32; 0] = [];
        assert_eq!(first_missing_positive(&mut nums), 1);
    }

    #[test]
    fn dense_prefix_returns_next() {
        let mut nums = [1, 2, 3];
        assert_eq!(first_missing_positive(&mut nums), 4);
    }

    #[test]
    fn mixed_with_negatives_and_gap() {
        let mut nums = [3, 4, -1, 1];
        assert_eq!(first_missing_positive(&mut nums), 2);
    }

    #[test]
    fn all_above_range_returns_one() {
        let mut nums = [7, 8, 9, 11, 12];
        assert_eq!(first_missing_positive(&mut nums), 1);
    }

    #[test]
    fn singleton_one_returns_two() {
        let mut nums = [1];
        assert_eq!(first_missing_positive(&mut nums), 2);
    }

    #[test]
    fn duplicates_do_not_loop() {
        let mut nums = [1, 1, 1];
        assert_eq!(first_missing_positive(&mut nums), 2);
    }

    #[test]
    fn singleton_non_one_returns_one() {
        let mut nums = [2];
        assert_eq!(first_missing_positive(&mut nums), 1);
        let mut nums = [-5];
        assert_eq!(first_missing_positive(&mut nums), 1);
    }

    #[test]
    fn zero_is_ignored() {
        let mut nums = [0, 2, 2, 1, 0, 1];
        // Values 1 and 2 are present; n = 6 so answer is 3.
        assert_eq!(first_missing_positive(&mut nums), 3);
    }

    #[test]
    fn already_sorted_full_range() {
        let mut nums = [1, 2, 3, 4, 5];
        assert_eq!(first_missing_positive(&mut nums), 6);
    }

    #[test]
    fn reverse_sorted_full_range() {
        let mut nums = [5, 4, 3, 2, 1];
        assert_eq!(first_missing_positive(&mut nums), 6);
    }

    #[test]
    fn extreme_values_do_not_panic() {
        let mut nums = [i32::MAX, i32::MIN, 0, 2];
        // n = 4; values in range are just `2`. Missing: 1.
        assert_eq!(first_missing_positive(&mut nums), 1);
    }

    #[test]
    fn large_dense_input() {
        let mut nums: Vec<i32> = (1..=1000).collect();
        assert_eq!(first_missing_positive(&mut nums), 1001);
    }

    #[test]
    fn large_with_single_gap() {
        let mut nums: Vec<i32> = (1..=1000).filter(|&x| x != 537).collect();
        // Length is 999; missing value 537 is in `1..=999`.
        assert_eq!(first_missing_positive(&mut nums), 537);
    }
}
