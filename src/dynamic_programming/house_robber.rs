//! Maximum sum of non-adjacent elements (the "House Robber" problem).
//!
//! Given a sequence `nums`, pick a subset such that no two chosen indices are
//! adjacent and the sum of the picks is maximised. Equivalent to the classic
//! "rob houses on a street without alerting the police" formulation.
//!
//! Recurrence: `dp[i] = max(dp[i-1], dp[i-2] + nums[i])`, where `dp[i]` is the
//! best achievable sum considering the prefix `nums[..=i]`. The empty subset
//! is always allowed, so the answer is at least `0` — negative values are
//! never picked.
//!
//! Runs in `O(n)` time and `O(1)` additional space using two rolling scalars.
//!
//! # Examples
//!
//! ```
//! use rust_algorithms::dynamic_programming::house_robber::max_non_adjacent_sum;
//! assert_eq!(max_non_adjacent_sum(&[2, 7, 9, 3, 1]), 12); // 2 + 9 + 1
//! assert_eq!(max_non_adjacent_sum(&[1, 2, 3, 1]), 4);     // 1 + 3
//! ```

/// Returns the maximum sum obtainable by selecting a subset of `nums` with no
/// two chosen indices adjacent.
///
/// The empty subset (sum `0`) is always a valid choice, so the result is
/// `>= 0`. For an empty input, returns `0`.
pub fn max_non_adjacent_sum(nums: &[i64]) -> i64 {
    // `prev2` tracks dp[i-2], `prev1` tracks dp[i-1]. Both start at 0 to
    // encode the empty-subset baseline (so negative values are never picked).
    let mut prev2: i64 = 0;
    let mut prev1: i64 = 0;
    for &x in nums {
        let take = prev2 + x;
        let curr = prev1.max(take);
        prev2 = prev1;
        prev1 = curr;
    }
    prev1
}

#[cfg(test)]
mod tests {
    use super::max_non_adjacent_sum;

    #[test]
    fn empty_input() {
        assert_eq!(max_non_adjacent_sum(&[]), 0);
    }

    #[test]
    fn single_positive() {
        assert_eq!(max_non_adjacent_sum(&[7]), 7);
    }

    #[test]
    fn single_negative() {
        // Empty subset wins.
        assert_eq!(max_non_adjacent_sum(&[-7]), 0);
    }

    #[test]
    fn single_zero() {
        assert_eq!(max_non_adjacent_sum(&[0]), 0);
    }

    #[test]
    fn classic_example_a() {
        assert_eq!(max_non_adjacent_sum(&[2, 7, 9, 3, 1]), 12);
    }

    #[test]
    fn classic_example_b() {
        assert_eq!(max_non_adjacent_sum(&[1, 2, 3, 1]), 4);
    }

    #[test]
    fn all_negative() {
        assert_eq!(max_non_adjacent_sum(&[-1, -2, -3]), 0);
        assert_eq!(max_non_adjacent_sum(&[-5, -1, -8, -2]), 0);
    }

    #[test]
    fn two_elements_picks_max() {
        assert_eq!(max_non_adjacent_sum(&[5, 1]), 5);
        assert_eq!(max_non_adjacent_sum(&[1, 5]), 5);
    }

    #[test]
    fn mixed_signs() {
        // Best is 10 + 4 = 14 (skip the negatives).
        assert_eq!(max_non_adjacent_sum(&[10, -100, -100, 4]), 14);
        // Best is 6 + 5 = 11 (indices 0 and 2).
        assert_eq!(max_non_adjacent_sum(&[6, -3, 5, -2]), 11);
    }

    #[test]
    fn all_zeros() {
        assert_eq!(max_non_adjacent_sum(&[0, 0, 0, 0, 0]), 0);
    }

    #[test]
    fn alternating_large_small() {
        // Picking every other large element: 100 + 100 + 100 = 300.
        assert_eq!(max_non_adjacent_sum(&[100, 1, 100, 1, 100]), 300);
    }

    #[test]
    fn long_run_of_positives() {
        // [5,5,10,100,10,5] — best is 5 + 100 + 5 = 110 (indices 0, 3, 5).
        assert_eq!(max_non_adjacent_sum(&[5, 5, 10, 100, 10, 5]), 110);
    }
}
