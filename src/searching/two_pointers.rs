//! Two-pointers technique on sorted slices.
//!
//! The two-pointers idiom walks two indices `lo` and `hi` over a sorted
//! sequence, advancing one or the other based on a target predicate. It runs
//! in `O(n)` time after the input is sorted (or already-sorted), uses `O(1)`
//! extra memory, and replaces an `O(n^2)` brute force in canonical problems
//! such as 2-Sum on sorted input, the smallest-difference pair, or
//! container-with-most-water style sweeps.

use core::cmp::Ordering;

/// Returns indices `(i, j)` with `i < j` such that `a[i] + a[j] == target`,
/// assuming `a` is sorted in non-decreasing order. `None` if no such pair
/// exists.
///
/// - Time: `O(n)`.
/// - Space: `O(1)`.
/// - Precondition: `a` is sorted non-decreasing.
pub fn two_sum_sorted(a: &[i64], target: i64) -> Option<(usize, usize)> {
    if a.len() < 2 {
        return None;
    }
    let (mut lo, mut hi) = (0_usize, a.len() - 1);
    while lo < hi {
        let sum = a[lo].checked_add(a[hi])?;
        match sum.cmp(&target) {
            Ordering::Equal => return Some((lo, hi)),
            Ordering::Less => lo += 1,
            Ordering::Greater => hi -= 1,
        }
    }
    None
}

/// Counts unordered index pairs `(i, j)` with `i < j` whose sum is at most
/// `target`, assuming `a` is sorted non-decreasing.
///
/// - Time: `O(n)`.
/// - Space: `O(1)`.
pub fn count_pairs_with_sum_at_most(a: &[i64], target: i64) -> usize {
    if a.len() < 2 {
        return 0;
    }
    let (mut lo, mut hi) = (0_usize, a.len() - 1);
    let mut count = 0_usize;
    while lo < hi {
        if a[lo] + a[hi] <= target {
            count += hi - lo;
            lo += 1;
        } else {
            hi -= 1;
        }
    }
    count
}

/// Smallest absolute difference between any pair of elements drawn one from
/// each of two non-decreasing slices `a` and `b`. Returns `None` if either
/// slice is empty.
///
/// - Time: `O(n + m)`.
/// - Space: `O(1)`.
pub fn min_abs_difference(a: &[i64], b: &[i64]) -> Option<i64> {
    if a.is_empty() || b.is_empty() {
        return None;
    }
    let (mut i, mut j) = (0_usize, 0_usize);
    let mut best = (a[0] - b[0]).abs();
    while i < a.len() && j < b.len() {
        let diff = (a[i] - b[j]).abs();
        if diff < best {
            best = diff;
        }
        if a[i] < b[j] {
            i += 1;
        } else {
            j += 1;
        }
    }
    Some(best)
}

#[cfg(test)]
mod tests {
    use super::{count_pairs_with_sum_at_most, min_abs_difference, two_sum_sorted};
    use quickcheck_macros::quickcheck;

    #[test]
    fn two_sum_empty_or_singleton() {
        assert_eq!(two_sum_sorted(&[], 0), None);
        assert_eq!(two_sum_sorted(&[5], 5), None);
    }

    #[test]
    fn two_sum_finds_pair() {
        let a = [-3_i64, 1, 4, 6, 10];
        let (i, j) = two_sum_sorted(&a, 7).unwrap();
        assert!(i < j && a[i] + a[j] == 7);
        let (i, j) = two_sum_sorted(&a, 14).unwrap();
        assert!(i < j && a[i] + a[j] == 14);
    }

    #[test]
    fn two_sum_missing() {
        let a = [1_i64, 2, 3, 4];
        assert_eq!(two_sum_sorted(&a, 100), None);
        assert_eq!(two_sum_sorted(&a, 0), None);
    }

    #[test]
    fn two_sum_with_negatives() {
        let a = [-5_i64, -2, 0, 3, 7];
        let (i, j) = two_sum_sorted(&a, 1).unwrap();
        assert!(i < j && a[i] + a[j] == 1);
        assert_eq!(two_sum_sorted(&a, -7), Some((0, 1)));
    }

    #[test]
    fn count_pairs_basic() {
        let a = [1_i64, 2, 3, 4, 5];
        // pairs with sum <= 5: (1,2) (1,3) (1,4) (2,3) -> 4
        assert_eq!(count_pairs_with_sum_at_most(&a, 5), 4);
        assert_eq!(count_pairs_with_sum_at_most(&a, 0), 0);
        assert_eq!(count_pairs_with_sum_at_most(&a, 100), 10);
    }

    #[test]
    fn count_pairs_empty_or_singleton() {
        assert_eq!(count_pairs_with_sum_at_most(&[], 0), 0);
        assert_eq!(count_pairs_with_sum_at_most(&[5], 5), 0);
    }

    #[test]
    fn min_diff_basic() {
        assert_eq!(
            min_abs_difference(&[1, 3, 15, 19], &[10, 20, 30, 40]),
            Some(1)
        );
        assert_eq!(min_abs_difference(&[5, 5, 5], &[5]), Some(0));
    }

    #[test]
    fn min_diff_empty() {
        assert_eq!(min_abs_difference(&[], &[1, 2, 3]), None);
        assert_eq!(min_abs_difference(&[1], &[]), None);
    }

    #[allow(clippy::needless_pass_by_value)]
    #[quickcheck]
    fn two_sum_matches_brute(a: Vec<i32>, target: i32) -> bool {
        let mut a: Vec<i64> = a.into_iter().take(40).map(i64::from).collect();
        a.sort_unstable();
        let target = i64::from(target);
        let got = two_sum_sorted(&a, target);
        let mut brute = None;
        'outer: for i in 0..a.len() {
            for j in (i + 1)..a.len() {
                if a[i] + a[j] == target {
                    brute = Some(i < j);
                    break 'outer;
                }
            }
        }
        match (got, brute) {
            (Some((i, j)), Some(_)) => i < j && a[i] + a[j] == target,
            (None, None) => true,
            _ => false,
        }
    }

    #[allow(clippy::needless_pass_by_value)]
    #[quickcheck]
    fn count_pairs_matches_brute(a: Vec<i16>, target: i32) -> bool {
        let mut a: Vec<i64> = a.into_iter().take(40).map(i64::from).collect();
        a.sort_unstable();
        let target = i64::from(target);
        let want = {
            let mut c = 0_usize;
            for i in 0..a.len() {
                for j in (i + 1)..a.len() {
                    if a[i] + a[j] <= target {
                        c += 1;
                    }
                }
            }
            c
        };
        count_pairs_with_sum_at_most(&a, target) == want
    }

    #[allow(clippy::needless_pass_by_value)]
    #[quickcheck]
    fn min_diff_matches_brute(a: Vec<i16>, b: Vec<i16>) -> bool {
        let mut a: Vec<i64> = a.into_iter().take(20).map(i64::from).collect();
        let mut b: Vec<i64> = b.into_iter().take(20).map(i64::from).collect();
        a.sort_unstable();
        b.sort_unstable();
        let want = {
            let mut best: Option<i64> = None;
            for &x in &a {
                for &y in &b {
                    let d = (x - y).abs();
                    best = Some(best.map_or(d, |b| b.min(d)));
                }
            }
            best
        };
        min_abs_difference(&a, &b) == want
    }
}
