//! Largest rectangle in a histogram via a monotonic stack.
//!
//! Given a histogram whose `i`-th bar has height `heights[i]` and unit width,
//! the largest axis-aligned rectangle that fits entirely under the silhouette
//! has area `max over i of heights[i] * (right_i - left_i - 1)`, where
//! `left_i` and `right_i` are the nearest indices on either side whose height
//! is strictly less than `heights[i]` (or `-1` / `n` when no such index
//! exists).
//!
//! A single left-to-right sweep with an index stack that keeps heights in
//! non-decreasing order computes both boundaries on the fly: when the current
//! bar is shorter than the stack top, we pop and finalise the popped bar's
//! rectangle, using the current index as its right boundary and the new stack
//! top as its left boundary. Each index is pushed and popped at most once, so
//! the algorithm runs in `O(n)` time and `O(n)` worst-case extra space.

/// Returns the area of the largest axis-aligned rectangle that fits under the
/// histogram described by `heights` (each bar has unit width).
///
/// Returns `0` for an empty input.
///
/// - Time: `O(n)`.
/// - Space: `O(n)`.
pub fn largest_rectangle(heights: &[u64]) -> u64 {
    let n = heights.len();
    let mut stack: Vec<usize> = Vec::with_capacity(n + 1);
    let mut best: u64 = 0;
    // Sweep with a sentinel index `n` whose virtual height is `0`, which
    // forces the stack to drain at the end.
    for i in 0..=n {
        let cur = if i == n { 0 } else { heights[i] };
        while let Some(&top) = stack.last() {
            if heights[top] <= cur {
                break;
            }
            stack.pop();
            let height = heights[top];
            let width = match stack.last() {
                Some(&left) => (i - left - 1) as u64,
                None => i as u64,
            };
            let area = height * width;
            if area > best {
                best = area;
            }
        }
        stack.push(i);
    }
    best
}

#[cfg(test)]
mod tests {
    use super::largest_rectangle;

    #[test]
    fn empty_is_zero() {
        let h: [u64; 0] = [];
        assert_eq!(largest_rectangle(&h), 0);
    }

    #[test]
    fn single_bar_returns_height() {
        assert_eq!(largest_rectangle(&[0]), 0);
        assert_eq!(largest_rectangle(&[7]), 7);
        assert_eq!(largest_rectangle(&[u64::MAX]), u64::MAX);
    }

    #[test]
    fn all_equal_bars() {
        let h = [4u64; 5];
        assert_eq!(largest_rectangle(&h), 20);
    }

    #[test]
    fn classic_example_is_ten() {
        // Canonical LeetCode 84 instance: bar of height 5 and 6 gives 5*2=10.
        assert_eq!(largest_rectangle(&[2, 1, 5, 6, 2, 3]), 10);
    }

    #[test]
    fn strictly_increasing() {
        // For [1,2,3,4,5] the optimum is 3*3 = 9 (bars 3,4,5 trimmed to 3).
        assert_eq!(largest_rectangle(&[1, 2, 3, 4, 5]), 9);
    }

    #[test]
    fn strictly_decreasing() {
        // For [5,4,3,2,1] the optimum is also 3*3 = 9.
        assert_eq!(largest_rectangle(&[5, 4, 3, 2, 1]), 9);
    }

    #[test]
    fn zeros_split_histogram() {
        // The zero acts as a hard wall; best is from the right side: 4*2 = 8
        // (bars 4,5 of heights 4,5 trimmed to 4) which beats the left side's
        // best of 3*2 = 6.
        assert_eq!(largest_rectangle(&[2, 3, 0, 4, 5]), 8);
    }

    #[test]
    fn all_zeros() {
        assert_eq!(largest_rectangle(&[0, 0, 0, 0]), 0);
    }

    #[test]
    fn leading_and_trailing_zeros() {
        assert_eq!(largest_rectangle(&[0, 3, 3, 3, 0]), 9);
    }
}
