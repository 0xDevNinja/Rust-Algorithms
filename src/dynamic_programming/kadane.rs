//! Kadane's algorithm for the maximum-sum contiguous subarray problem. O(n).

/// Returns the largest sum of any contiguous subarray of `nums`. Returns
/// `None` for an empty input.
pub fn max_subarray_sum(nums: &[i64]) -> Option<i64> {
    let mut best: Option<i64> = None;
    let mut current = 0_i64;
    for &x in nums {
        current = match best {
            Some(_) => (current + x).max(x),
            None => x,
        };
        best = Some(best.map_or(current, |b| b.max(current)));
    }
    best
}

/// Returns `(sum, start, end_exclusive)` of the maximum-sum contiguous
/// subarray. Returns `None` for empty input.
pub fn max_subarray_with_indices(nums: &[i64]) -> Option<(i64, usize, usize)> {
    if nums.is_empty() {
        return None;
    }
    let mut best_sum = nums[0];
    let mut best_start = 0;
    let mut best_end = 1;
    let mut current_sum = nums[0];
    let mut current_start = 0;
    for (i, &x) in nums.iter().enumerate().skip(1) {
        if current_sum + x < x {
            current_sum = x;
            current_start = i;
        } else {
            current_sum += x;
        }
        if current_sum > best_sum {
            best_sum = current_sum;
            best_start = current_start;
            best_end = i + 1;
        }
    }
    Some((best_sum, best_start, best_end))
}

#[cfg(test)]
mod tests {
    use super::{max_subarray_sum, max_subarray_with_indices};

    #[test]
    fn empty_input() {
        assert_eq!(max_subarray_sum(&[]), None);
        assert_eq!(max_subarray_with_indices(&[]), None);
    }

    #[test]
    fn single_positive() {
        assert_eq!(max_subarray_sum(&[5]), Some(5));
    }

    #[test]
    fn single_negative() {
        assert_eq!(max_subarray_sum(&[-3]), Some(-3));
    }

    #[test]
    fn classic_example() {
        // [-2,1,-3,4,-1,2,1,-5,4] — max-sum subarray is [4,-1,2,1] = 6
        let v = [-2, 1, -3, 4, -1, 2, 1, -5, 4];
        assert_eq!(max_subarray_sum(&v), Some(6));
        let (sum, start, end) = max_subarray_with_indices(&v).unwrap();
        assert_eq!(sum, 6);
        assert_eq!(&v[start..end], &[4, -1, 2, 1]);
    }

    #[test]
    fn all_negative() {
        // Best is the single largest element.
        assert_eq!(max_subarray_sum(&[-3, -1, -4, -1, -5]), Some(-1));
    }

    #[test]
    fn all_positive_full_array() {
        let v = [1_i64, 2, 3, 4, 5];
        let (sum, start, end) = max_subarray_with_indices(&v).unwrap();
        assert_eq!(sum, 15);
        assert_eq!(start, 0);
        assert_eq!(end, 5);
    }
}
