//! Meet-in-the-middle subset sum.
//!
//! Splits the input multiset into two halves, enumerates the `2^(n/2)` subset
//! sums of each half, sorts the second list, and answers existence /
//! count-of-subsets-summing-to-`target` queries by binary searching one half
//! against the other. The technique reduces the naive `O(2^n)` brute force to
//! `O(2^(n/2) * n)` time and `O(2^(n/2))` space and is the textbook attack on
//! 4-Sum, knapsack-with-large-weights, and similar `n <= 40` instances.

/// Returns `true` if some subset of `weights` sums exactly to `target`.
///
/// - Time: `O(2^(n/2) * n)` (the `n` factor is the cost of sorting the right
///   half).
/// - Space: `O(2^(n/2))`.
///
/// # Panics
/// Panics if `weights.len() > 60`, since `2^31` is the largest subset
/// enumeration that fits comfortably in memory.
pub fn subset_sum_exists(weights: &[i64], target: i64) -> bool {
    assert!(
        weights.len() <= 60,
        "subset_sum_exists: meet-in-the-middle is only feasible for n <= 60 (got {})",
        weights.len()
    );
    let mid = weights.len() / 2;
    let left = enumerate_sums(&weights[..mid]);
    let mut right = enumerate_sums(&weights[mid..]);
    right.sort_unstable();

    for s in &left {
        let need = target - s;
        if right.binary_search(&need).is_ok() {
            return true;
        }
    }
    false
}

/// Counts the number of subsets of `weights` (including the empty subset)
/// whose sum equals `target`.
///
/// - Time: `O(2^(n/2) * n)`.
/// - Space: `O(2^(n/2))`.
///
/// # Panics
/// Panics if `weights.len() > 60`.
pub fn subset_sum_count(weights: &[i64], target: i64) -> u64 {
    assert!(
        weights.len() <= 60,
        "subset_sum_count: meet-in-the-middle is only feasible for n <= 60 (got {})",
        weights.len()
    );
    let mid = weights.len() / 2;
    let left = enumerate_sums(&weights[..mid]);
    let mut right = enumerate_sums(&weights[mid..]);
    right.sort_unstable();

    let mut total = 0_u64;
    for s in &left {
        let need = target - s;
        let lo = right.partition_point(|x| *x < need);
        let hi = right.partition_point(|x| *x <= need);
        total += (hi - lo) as u64;
    }
    total
}

fn enumerate_sums(items: &[i64]) -> Vec<i64> {
    let n = items.len();
    let mut sums = Vec::with_capacity(1 << n);
    for mask in 0_u64..(1_u64 << n) {
        let mut s = 0_i64;
        let mut m = mask;
        while m != 0 {
            let bit = m.trailing_zeros() as usize;
            s += items[bit];
            m &= m - 1;
        }
        sums.push(s);
    }
    sums
}

#[cfg(test)]
mod tests {
    use super::{subset_sum_count, subset_sum_exists};
    use quickcheck_macros::quickcheck;

    fn brute_count(weights: &[i64], target: i64) -> u64 {
        let n = weights.len();
        let mut total = 0_u64;
        for mask in 0_u64..(1_u64 << n) {
            let mut s = 0_i64;
            let mut m = mask;
            while m != 0 {
                let b = m.trailing_zeros() as usize;
                s += weights[b];
                m &= m - 1;
            }
            if s == target {
                total += 1;
            }
        }
        total
    }

    #[test]
    fn empty_set_only_hits_zero() {
        assert!(subset_sum_exists(&[], 0));
        assert!(!subset_sum_exists(&[], 1));
        assert_eq!(subset_sum_count(&[], 0), 1);
        assert_eq!(subset_sum_count(&[], 5), 0);
    }

    #[test]
    fn single_element() {
        assert!(subset_sum_exists(&[7], 0));
        assert!(subset_sum_exists(&[7], 7));
        assert!(!subset_sum_exists(&[7], 3));
        assert_eq!(subset_sum_count(&[7], 0), 1);
        assert_eq!(subset_sum_count(&[7], 7), 1);
    }

    #[test]
    fn known_examples() {
        let w = [3_i64, 1, 4, 1, 5, 9, 2, 6];
        assert!(subset_sum_exists(&w, 10));
        assert!(subset_sum_exists(&w, 31)); // sum of all
        assert!(!subset_sum_exists(&w, 32));
        assert_eq!(subset_sum_count(&w, 5), brute_count(&w, 5));
    }

    #[test]
    fn handles_negative_weights() {
        let w = [-3_i64, 5, -2, 7];
        assert!(subset_sum_exists(&w, 0)); // empty subset or {-3, 5, -2}
        assert!(subset_sum_exists(&w, 12)); // {5, -2, 7, -3+5? } -> 5+7=12
        assert_eq!(subset_sum_count(&w, 0), brute_count(&w, 0));
    }

    #[allow(clippy::needless_pass_by_value)]
    #[quickcheck]
    fn count_matches_brute(weights: Vec<i8>, target: i32) -> bool {
        let weights: Vec<i64> = weights.into_iter().take(12).map(i64::from).collect();
        let target = i64::from(target);
        subset_sum_count(&weights, target) == brute_count(&weights, target)
    }

    #[allow(clippy::needless_pass_by_value)]
    #[quickcheck]
    fn exists_matches_brute(weights: Vec<i8>, target: i32) -> bool {
        let weights: Vec<i64> = weights.into_iter().take(12).map(i64::from).collect();
        let target = i64::from(target);
        subset_sum_exists(&weights, target) == (brute_count(&weights, target) > 0)
    }
}
