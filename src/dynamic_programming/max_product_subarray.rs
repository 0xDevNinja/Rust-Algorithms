//! Maximum product of any contiguous subarray. O(n).
//!
//! Because a negative factor flips the sign of the running product, we track
//! both the largest and the smallest product ending at the current index. At
//! every step the new max/min is one of `x`, `max_so_far * x`, or
//! `min_so_far * x`. Sweeping these candidates left-to-right yields the global
//! optimum in linear time and constant extra space.

/// Returns the maximum product of any contiguous subarray of `nums`.
///
/// Returns `0` for an empty slice (documented convention; there is no
/// "empty product" answer that fits the typical use of this routine).
///
/// Runs in O(n) time and O(1) extra space.
pub fn max_product(nums: &[i64]) -> i64 {
    if nums.is_empty() {
        return 0;
    }
    let mut max_so_far = nums[0];
    let mut min_so_far = nums[0];
    let mut best = nums[0];
    for &x in &nums[1..] {
        let candidates = (x, max_so_far * x, min_so_far * x);
        max_so_far = candidates.0.max(candidates.1).max(candidates.2);
        min_so_far = candidates.0.min(candidates.1).min(candidates.2);
        best = best.max(max_so_far);
    }
    best
}

#[cfg(test)]
mod tests {
    use super::max_product;

    #[test]
    fn empty_returns_zero() {
        assert_eq!(max_product(&[]), 0);
    }

    #[test]
    fn single_positive() {
        assert_eq!(max_product(&[7]), 7);
    }

    #[test]
    fn single_negative() {
        assert_eq!(max_product(&[-7]), -7);
    }

    #[test]
    fn classic_example() {
        // [2,3,-2,4] — best subarray is [2,3] with product 6.
        assert_eq!(max_product(&[2, 3, -2, 4]), 6);
    }

    #[test]
    fn zero_dominates_negative() {
        // [-2,0,-1] — best is the singleton [0].
        assert_eq!(max_product(&[-2, 0, -1]), 0);
    }

    #[test]
    fn two_negatives_flip_sign() {
        // [-2,3,-4] — full array product (-2)*3*(-4) = 24.
        assert_eq!(max_product(&[-2, 3, -4]), 24);
    }

    #[test]
    fn all_negative_even_length() {
        // [-1,-2,-3,-4] — full product is (+24).
        assert_eq!(max_product(&[-1, -2, -3, -4]), 24);
    }

    #[test]
    fn all_negative_odd_length() {
        // [-1,-2,-3] — best is the trailing pair (-2)*(-3) = 6.
        assert_eq!(max_product(&[-1, -2, -3]), 6);
    }

    #[test]
    fn includes_zeros() {
        // [0,2,-3,5] — zero resets the run; best is the singleton [5].
        assert_eq!(max_product(&[0, 2, -3, 5]), 5);
    }
}
