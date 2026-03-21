//! Subset-sum DP. Decides whether some subset of `nums` sums to `target`.
//! O(n · target) time, O(target) space using the rolling 1-D DP.

/// Returns `true` if some (possibly empty) subset of `nums` sums to `target`.
///
/// `nums` is interpreted as non-negative integers. Empty subset sums to 0,
/// so `subset_sum(&[], 0) == true`.
pub fn subset_sum(nums: &[u32], target: u32) -> bool {
    let target = target as usize;
    let mut dp = vec![false; target + 1];
    dp[0] = true;
    for &x in nums {
        let x = x as usize;
        if x > target {
            continue;
        }
        // Iterate down so each item is used at most once.
        for cap in (x..=target).rev() {
            if dp[cap - x] {
                dp[cap] = true;
            }
        }
        if dp[target] {
            return true;
        }
    }
    dp[target]
}

/// Returns one subset of `nums` summing to `target`, or `None`.
pub fn find_subset(nums: &[u32], target: u32) -> Option<Vec<u32>> {
    let target_usize = target as usize;
    let n = nums.len();
    let mut dp = vec![vec![false; target_usize + 1]; n + 1];
    dp[0][0] = true;
    for i in 1..=n {
        let x = nums[i - 1] as usize;
        for cap in 0..=target_usize {
            dp[i][cap] = dp[i - 1][cap] || (cap >= x && dp[i - 1][cap - x]);
        }
    }
    if !dp[n][target_usize] {
        return None;
    }
    let mut result = Vec::new();
    let mut cap = target_usize;
    for i in (1..=n).rev() {
        let x = nums[i - 1] as usize;
        if cap >= x && dp[i - 1][cap - x] {
            result.push(nums[i - 1]);
            cap -= x;
        }
    }
    result.reverse();
    Some(result)
}

#[cfg(test)]
mod tests {
    use super::{find_subset, subset_sum};

    #[test]
    fn empty_zero_target() {
        assert!(subset_sum(&[], 0));
        assert_eq!(find_subset(&[], 0), Some(vec![]));
    }

    #[test]
    fn empty_nonzero_target() {
        assert!(!subset_sum(&[], 5));
        assert!(find_subset(&[], 5).is_none());
    }

    #[test]
    fn single_match() {
        assert!(subset_sum(&[7], 7));
        assert_eq!(find_subset(&[7], 7), Some(vec![7]));
    }

    #[test]
    fn multiple_elements() {
        let nums = [3, 34, 4, 12, 5, 2];
        assert!(subset_sum(&nums, 9)); // 4 + 5
        assert!(subset_sum(&nums, 14)); // 12 + 2 or 5 + 4 + 3 + 2
        assert!(!subset_sum(&nums, 30));
    }

    #[test]
    fn find_subset_recovers_a_witness() {
        let nums = [3, 34, 4, 12, 5, 2];
        let witness = find_subset(&nums, 9).unwrap();
        assert_eq!(witness.iter().sum::<u32>(), 9);
    }

    #[test]
    fn target_zero_is_always_reachable() {
        assert!(subset_sum(&[5, 7, 11], 0));
        assert_eq!(find_subset(&[5, 7, 11], 0), Some(vec![]));
    }

    #[test]
    fn item_larger_than_target_ignored() {
        assert!(subset_sum(&[100, 1, 2], 3));
    }
}
