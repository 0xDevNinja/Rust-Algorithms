//! Sliding-window minimum (and maximum) via a monotonic deque.
//!
//! Given an array `a` and a window size `k`, returns the minimum (or maximum)
//! over every length-`k` contiguous window in `O(n)` total time and `O(k)`
//! extra memory. The deque holds indices in strictly increasing order such
//! that the corresponding values form a strictly increasing (for min) or
//! decreasing (for max) sequence; each index is enqueued and dequeued at most
//! once.

use std::collections::VecDeque;

/// Returns `a[i .. i+k].iter().min()` (cloned) for every starting index
/// `i = 0 .. a.len() - k + 1`.
///
/// - Time: `O(n)`.
/// - Space: `O(k)`.
///
/// # Panics
/// Panics if `k == 0` or `k > a.len()`.
pub fn sliding_window_min<T: Ord + Clone>(a: &[T], k: usize) -> Vec<T> {
    assert!(k > 0, "sliding_window_min: window size must be >= 1");
    assert!(
        k <= a.len(),
        "sliding_window_min: window size {k} exceeds slice length {}",
        a.len()
    );

    let mut deque: VecDeque<usize> = VecDeque::with_capacity(k);
    let mut out = Vec::with_capacity(a.len() - k + 1);

    for i in 0..a.len() {
        // Drop indices that fell out of the window on the left.
        while let Some(&front) = deque.front() {
            if front + k <= i {
                deque.pop_front();
            } else {
                break;
            }
        }
        // Maintain a strictly increasing-value monotonic deque.
        while let Some(&back) = deque.back() {
            if a[back] >= a[i] {
                deque.pop_back();
            } else {
                break;
            }
        }
        deque.push_back(i);
        if i + 1 >= k {
            let front = *deque.front().expect("non-empty after push");
            out.push(a[front].clone());
        }
    }
    out
}

/// Returns `a[i .. i+k].iter().max()` (cloned) for every starting index
/// `i = 0 .. a.len() - k + 1`.
///
/// - Time: `O(n)`.
/// - Space: `O(k)`.
///
/// # Panics
/// Panics if `k == 0` or `k > a.len()`.
pub fn sliding_window_max<T: Ord + Clone>(a: &[T], k: usize) -> Vec<T> {
    assert!(k > 0, "sliding_window_max: window size must be >= 1");
    assert!(
        k <= a.len(),
        "sliding_window_max: window size {k} exceeds slice length {}",
        a.len()
    );

    let mut deque: VecDeque<usize> = VecDeque::with_capacity(k);
    let mut out = Vec::with_capacity(a.len() - k + 1);

    for i in 0..a.len() {
        while let Some(&front) = deque.front() {
            if front + k <= i {
                deque.pop_front();
            } else {
                break;
            }
        }
        while let Some(&back) = deque.back() {
            if a[back] <= a[i] {
                deque.pop_back();
            } else {
                break;
            }
        }
        deque.push_back(i);
        if i + 1 >= k {
            let front = *deque.front().expect("non-empty after push");
            out.push(a[front].clone());
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{sliding_window_max, sliding_window_min};
    use quickcheck_macros::quickcheck;

    fn brute_min(a: &[i32], k: usize) -> Vec<i32> {
        (0..=a.len() - k)
            .map(|i| *a[i..i + k].iter().min().unwrap())
            .collect()
    }

    fn brute_max(a: &[i32], k: usize) -> Vec<i32> {
        (0..=a.len() - k)
            .map(|i| *a[i..i + k].iter().max().unwrap())
            .collect()
    }

    #[test]
    fn single_element_window_returns_input() {
        let a = [3, 1, 4, 1, 5, 9, 2, 6];
        assert_eq!(sliding_window_min(&a, 1), a.to_vec());
        assert_eq!(sliding_window_max(&a, 1), a.to_vec());
    }

    #[test]
    fn full_window_returns_min_or_max() {
        let a = [3, 1, 4, 1, 5, 9, 2, 6];
        assert_eq!(sliding_window_min(&a, 8), vec![1]);
        assert_eq!(sliding_window_max(&a, 8), vec![9]);
    }

    #[test]
    fn classic_leetcode_example() {
        let a = [1, 3, -1, -3, 5, 3, 6, 7];
        assert_eq!(sliding_window_max(&a, 3), vec![3, 3, 5, 5, 6, 7]);
        assert_eq!(sliding_window_min(&a, 3), vec![-1, -3, -3, -3, 3, 3]);
    }

    #[test]
    #[should_panic(expected = "window size must be >= 1")]
    fn window_zero_panics() {
        let _ = sliding_window_min(&[1, 2, 3], 0);
    }

    #[test]
    #[should_panic(expected = "window size 5 exceeds slice length 3")]
    fn window_too_big_panics() {
        let _ = sliding_window_min(&[1, 2, 3], 5);
    }

    #[allow(clippy::needless_pass_by_value)]
    #[quickcheck]
    fn min_matches_brute(a: Vec<i32>, k: u8) -> bool {
        let a: Vec<i32> = a.into_iter().take(60).collect();
        if a.is_empty() {
            return true;
        }
        let k = (usize::from(k) % a.len()) + 1;
        sliding_window_min(&a, k) == brute_min(&a, k)
    }

    #[allow(clippy::needless_pass_by_value)]
    #[quickcheck]
    fn max_matches_brute(a: Vec<i32>, k: u8) -> bool {
        let a: Vec<i32> = a.into_iter().take(60).collect();
        if a.is_empty() {
            return true;
        }
        let k = (usize::from(k) % a.len()) + 1;
        sliding_window_max(&a, k) == brute_max(&a, k)
    }
}
