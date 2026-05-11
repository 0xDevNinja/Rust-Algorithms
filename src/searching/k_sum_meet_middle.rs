//! k-sum problem via meet-in-the-middle.
//!
//! Given an array `nums` of integers, a non-negative integer `k`, and a target
//! `target`, decide whether there exists a subset of `nums` of size exactly `k`
//! whose elements sum to `target`. The brute-force enumeration of every
//! `C(n, k)` subset costs `O(C(n, k) * k)` and is impractical past `n` in the
//! mid-twenties. Meet-in-the-middle splits the input in half: enumerate every
//! `C(n_l, a)` subset of the left half together with its sum (for every
//! plausible split `a + b = k`), then for every right-half subset of size
//! `b = k - a` look up `target - sum_left` in a hash map keyed on
//! `(size, sum)`. The technique brings the cost down to roughly
//! `O(2^(n/2) * n)` time and `O(2^(n/2))` space, which is the textbook attack
//! on instances with `n <= 40`.

use std::collections::HashMap;

/// Returns `true` if some subset of `nums` of size exactly `k` sums to
/// `target`.
///
/// The search uses meet-in-the-middle: the left half contributes
/// `(size, sum)` keys to a hash map, and the right half queries it for the
/// complementary `(k - size, target - sum)` pair.
///
/// - Time: `O(2^(n/2) * n)` average (hash map operations are amortized `O(1)`).
/// - Space: `O(2^(n/2))`.
///
/// Returns `false` whenever `k > nums.len()`. The empty subset (`k == 0`)
/// matches `target == 0`.
///
/// # Panics
/// Panics if `nums.len() > 40`, since enumerating `2^20` subsets per half is
/// already on the edge of what fits comfortably in memory.
pub fn k_sum_exists(nums: &[i64], k: usize, target: i64) -> bool {
    let n = nums.len();
    assert!(
        n <= 40,
        "k_sum_exists: meet-in-the-middle is only feasible for n <= 40 (got {n})"
    );
    if k > n {
        return false;
    }
    if k == 0 {
        return target == 0;
    }

    let mid = n / 2;
    let left = &nums[..mid];
    let right = &nums[mid..];

    // Map (subset_size, subset_sum) from the left half to its multiplicity.
    let mut left_table: HashMap<(usize, i64), usize> = HashMap::new();
    enumerate(left, |size, sum| {
        if size <= k {
            *left_table.entry((size, sum)).or_insert(0) += 1;
        }
    });

    let mut found = false;
    enumerate(right, |size, sum| {
        if found || size > k {
            return;
        }
        let need_size = k - size;
        let need_sum = target - sum;
        if left_table.contains_key(&(need_size, need_sum)) {
            found = true;
        }
    });

    found
}

/// Enumerates every subset of `items` and invokes `visit(size, sum)` once per
/// subset (including the empty subset).
fn enumerate<F: FnMut(usize, i64)>(items: &[i64], mut visit: F) {
    let n = items.len();
    for mask in 0_u64..(1_u64 << n) {
        let mut size = 0_usize;
        let mut sum = 0_i64;
        let mut m = mask;
        while m != 0 {
            let bit = m.trailing_zeros() as usize;
            sum += items[bit];
            size += 1;
            m &= m - 1;
        }
        visit(size, sum);
    }
}

#[cfg(test)]
mod tests {
    use super::k_sum_exists;

    #[test]
    fn empty_with_k_zero_matches_zero() {
        assert!(k_sum_exists(&[], 0, 0));
        assert!(!k_sum_exists(&[], 0, 1));
    }

    #[test]
    fn k_greater_than_n_is_false() {
        assert!(!k_sum_exists(&[1, 2, 3], 4, 6));
        assert!(!k_sum_exists(&[], 1, 0));
    }

    #[test]
    fn pair_sum_hits_target() {
        // 1 + 4 = 5 and 2 + 3 = 5
        assert!(k_sum_exists(&[1, 2, 3, 4], 2, 5));
    }

    #[test]
    fn pair_sum_misses_target() {
        // No pair in [1,2,3,4] sums to 10.
        assert!(!k_sum_exists(&[1, 2, 3, 4], 2, 10));
    }

    #[test]
    fn full_array_sum() {
        let nums = [1_i64, 2, 3, 4];
        assert!(k_sum_exists(&nums, 4, 10));
        assert!(!k_sum_exists(&nums, 4, 9));
    }

    #[test]
    fn single_element_subset() {
        let nums = [5_i64, -3, 8, 2];
        assert!(k_sum_exists(&nums, 1, 5));
        assert!(k_sum_exists(&nums, 1, -3));
        assert!(k_sum_exists(&nums, 1, 8));
        assert!(!k_sum_exists(&nums, 1, 100));
    }

    #[test]
    fn handles_negative_values() {
        let nums = [-5_i64, -2, 3, 7, 10];
        // (-5) + (-2) + 7 = 0
        assert!(k_sum_exists(&nums, 3, 0));
        // No triple sums to 50.
        assert!(!k_sum_exists(&nums, 3, 50));
    }

    #[test]
    fn k_zero_with_nonempty_array() {
        assert!(k_sum_exists(&[1, 2, 3], 0, 0));
        assert!(!k_sum_exists(&[1, 2, 3], 0, 1));
    }

    #[test]
    fn duplicates_count_correctly() {
        // Two 3s, picking both gives sum 6.
        let nums = [3_i64, 3, 1, 4];
        assert!(k_sum_exists(&nums, 2, 6));
        // 3+3+1 = 7
        assert!(k_sum_exists(&nums, 3, 7));
    }

    #[test]
    fn larger_instance_meets_in_middle() {
        // n = 20, pick exactly 5 numbers summing to 50.
        let nums: Vec<i64> = (1..=20).collect();
        // 6 + 8 + 10 + 12 + 14 = 50
        assert!(k_sum_exists(&nums, 5, 50));
        // Maximum 5-subset is 16+17+18+19+20 = 90, anything above is impossible.
        assert!(!k_sum_exists(&nums, 5, 91));
        // Minimum 5-subset is 1+2+3+4+5 = 15.
        assert!(!k_sum_exists(&nums, 5, 14));
    }
}
