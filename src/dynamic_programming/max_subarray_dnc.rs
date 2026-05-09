//! Maximum-subarray sum via divide and conquer. O(n log n).
//!
//! Bentley's *Programming Pearls*, Column 8: split the array at its midpoint
//! and recurse. The maximum-sum contiguous subarray is whichever is largest
//! of:
//!
//! 1. the best subarray fully inside the left half,
//! 2. the best subarray fully inside the right half,
//! 3. the best subarray that *crosses* the midpoint — found in O(n) by
//!    sweeping outward from the midpoint in each direction.
//!
//! Recurrence `T(n) = 2 T(n/2) + O(n)` solves to `O(n log n)`. Slower than
//! the linear [`super::kadane`] solution, but a textbook example of the
//! divide-and-conquer paradigm.
//!
//! The empty subarray is allowed and contributes `0`, so an all-negative
//! input returns `0` (rather than the largest single element as in the
//! Kadane variant in this crate).
//!
//! See also: [`crate::dynamic_programming::kadane`] for the O(n) variant.
//!
//! # Complexity
//!
//! - Time: `O(n log n)`
//! - Space: `O(log n)` recursion depth.

/// Returns the maximum sum of any contiguous subarray of `arr`. The empty
/// subarray is allowed, so the result is always `>= 0`.
pub fn max_subarray_dnc(arr: &[i64]) -> i64 {
    if arr.is_empty() {
        return 0;
    }
    rec_sum(arr, 0, arr.len() - 1).max(0)
}

/// Returns `(sum, lo, hi)` where `arr[lo..=hi]` is a maximum-sum contiguous
/// subarray. If the empty subarray wins (all elements negative, or `arr`
/// empty), returns the sentinel `(0, 0, 0)`.
pub fn max_subarray_dnc_with_indices(arr: &[i64]) -> (i64, usize, usize) {
    if arr.is_empty() {
        return (0, 0, 0);
    }
    let (sum, lo, hi) = rec_with_indices(arr, 0, arr.len() - 1);
    if sum < 0 {
        (0, 0, 0)
    } else {
        (sum, lo, hi)
    }
}

/// Maximum subarray sum on the inclusive range `[lo, hi]`. Sum-only variant.
fn rec_sum(arr: &[i64], lo: usize, hi: usize) -> i64 {
    if lo == hi {
        return arr[lo];
    }
    let mid = lo + (hi - lo) / 2;
    let left = rec_sum(arr, lo, mid);
    let right = rec_sum(arr, mid + 1, hi);
    let cross = max_crossing_sum(arr, lo, mid, hi);
    left.max(right).max(cross)
}

/// Maximum subarray sum on the inclusive range `[lo, hi]`, returning the
/// indices of an optimal subarray as well.
fn rec_with_indices(arr: &[i64], lo: usize, hi: usize) -> (i64, usize, usize) {
    if lo == hi {
        return (arr[lo], lo, lo);
    }
    let mid = lo + (hi - lo) / 2;
    let left = rec_with_indices(arr, lo, mid);
    let right = rec_with_indices(arr, mid + 1, hi);
    let cross = max_crossing_with_indices(arr, lo, mid, hi);
    // Prefer left, then cross, then right on ties — any choice is correct.
    if left.0 >= cross.0 && left.0 >= right.0 {
        left
    } else if cross.0 >= right.0 {
        cross
    } else {
        right
    }
}

/// Best subarray sum that spans the midpoint, i.e. uses `arr[mid]` and
/// `arr[mid + 1]`. Linear in `hi - lo + 1`.
fn max_crossing_sum(arr: &[i64], lo: usize, mid: usize, hi: usize) -> i64 {
    let mut left_best = i64::MIN;
    let mut acc: i64 = 0;
    let mut i = mid;
    loop {
        acc += arr[i];
        if acc > left_best {
            left_best = acc;
        }
        if i == lo {
            break;
        }
        i -= 1;
    }
    let mut right_best = i64::MIN;
    acc = 0;
    for j in (mid + 1)..=hi {
        acc += arr[j];
        if acc > right_best {
            right_best = acc;
        }
    }
    left_best + right_best
}

/// Best crossing subarray together with its inclusive `[lo, hi]` indices.
fn max_crossing_with_indices(arr: &[i64], lo: usize, mid: usize, hi: usize) -> (i64, usize, usize) {
    let mut left_best = i64::MIN;
    let mut left_idx = mid;
    let mut acc: i64 = 0;
    let mut i = mid;
    loop {
        acc += arr[i];
        if acc > left_best {
            left_best = acc;
            left_idx = i;
        }
        if i == lo {
            break;
        }
        i -= 1;
    }
    let mut right_best = i64::MIN;
    let mut right_idx = mid + 1;
    acc = 0;
    for j in (mid + 1)..=hi {
        acc += arr[j];
        if acc > right_best {
            right_best = acc;
            right_idx = j;
        }
    }
    (left_best + right_best, left_idx, right_idx)
}

#[cfg(test)]
mod tests {
    use super::{max_subarray_dnc, max_subarray_dnc_with_indices};
    use crate::dynamic_programming::kadane::max_subarray_sum;

    #[test]
    fn empty_returns_zero() {
        assert_eq!(max_subarray_dnc(&[]), 0);
        assert_eq!(max_subarray_dnc_with_indices(&[]), (0, 0, 0));
    }

    #[test]
    fn single_positive() {
        assert_eq!(max_subarray_dnc(&[7]), 7);
        assert_eq!(max_subarray_dnc_with_indices(&[7]), (7, 0, 0));
    }

    #[test]
    fn single_negative_uses_empty_subarray() {
        assert_eq!(max_subarray_dnc(&[-4]), 0);
        assert_eq!(max_subarray_dnc_with_indices(&[-4]), (0, 0, 0));
    }

    #[test]
    fn all_positive() {
        let v = [1_i64, 2, 3, 4, 5];
        assert_eq!(max_subarray_dnc(&v), 15);
        let (sum, lo, hi) = max_subarray_dnc_with_indices(&v);
        assert_eq!(sum, 15);
        assert_eq!(lo, 0);
        assert_eq!(hi, 4);
    }

    #[test]
    fn all_negative_returns_zero() {
        let v = [-3_i64, -1, -4, -1, -5, -9, -2, -6];
        assert_eq!(max_subarray_dnc(&v), 0);
        assert_eq!(max_subarray_dnc_with_indices(&v), (0, 0, 0));
    }

    #[test]
    fn bentley_canonical_example() {
        // From Programming Pearls Column 8.
        // [-2,1,-3,4,-1,2,1,-5,4] — best contiguous slice is [4,-1,2,1] = 6.
        let v = [-2_i64, 1, -3, 4, -1, 2, 1, -5, 4];
        assert_eq!(max_subarray_dnc(&v), 6);
        let (sum, lo, hi) = max_subarray_dnc_with_indices(&v);
        assert_eq!(sum, 6);
        assert_eq!(&v[lo..=hi], &[4, -1, 2, 1]);
    }

    #[test]
    fn indices_form_a_valid_subarray() {
        // Whatever indices we report, the slice they describe must sum to
        // the reported sum.
        let cases: &[&[i64]] = &[
            &[2, -1, 2, 3, -9, 4, 4, -10, 5],
            &[5, -3, 5],
            &[-1, -2, -3, 10, -4, -5],
            &[10, -2, 3, -1, 2, -100, 8],
        ];
        for arr in cases {
            let (sum, lo, hi) = max_subarray_dnc_with_indices(arr);
            if sum == 0 && lo == 0 && hi == 0 {
                // Empty-subarray sentinel — only legal if no positive sum exists.
                assert!(max_subarray_dnc(arr) == 0);
            } else {
                let actual: i64 = arr[lo..=hi].iter().sum();
                assert_eq!(actual, sum, "indices [{lo},{hi}] of {arr:?}");
            }
        }
    }

    #[test]
    fn agrees_with_kadane_on_fixed_inputs() {
        // The kadane in this crate returns Option<i64> with the best non-empty
        // subarray sum; the DnC variant allows the empty subarray. They agree
        // whenever the optimum is non-negative.
        let cases: &[&[i64]] = &[
            &[1, 2, 3, 4, 5],
            &[-2, 1, -3, 4, -1, 2, 1, -5, 4],
            &[5],
            &[0, 0, 0],
            &[3, -2, 5, -1],
            &[-1, 4, -2, 3, -5, 4, 0, 7, -3],
        ];
        for arr in cases {
            let dnc = max_subarray_dnc(arr);
            let kadane = max_subarray_sum(arr).unwrap_or(0).max(0);
            assert_eq!(dnc, kadane, "mismatch on {arr:?}");
        }
    }

    #[test]
    fn agrees_with_kadane_on_pseudo_random_inputs() {
        // Deterministic LCG so this stays a regular unit test (no rand dep).
        const MUL: u64 = 6_364_136_223_846_793_005;
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        for _ in 0..40 {
            state = state.wrapping_mul(MUL).wrapping_add(1);
            let n = ((state >> 32) as usize) % 50;
            let mut v = Vec::with_capacity(n);
            for _ in 0..n {
                state = state.wrapping_mul(MUL).wrapping_add(1);
                let raw = (state >> 32) as i64;
                v.push((raw % 21) - 10); // [-10, 10]
            }
            let dnc = max_subarray_dnc(&v);
            let kadane = max_subarray_sum(&v).unwrap_or(0).max(0);
            assert_eq!(dnc, kadane, "mismatch on {v:?}");
        }
    }
}
