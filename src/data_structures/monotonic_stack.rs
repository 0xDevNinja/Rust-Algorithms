//! Monotonic-stack helpers for nearest smaller / greater element queries.
//!
//! A monotonic stack maintains its elements in strictly increasing or strictly
//! decreasing order while sweeping a slice once. Each element is pushed and
//! popped at most once, giving `O(n)` total time and `O(n)` worst-case space
//! for both the stack and the output. Classical applications include
//! computing the previous/next smaller (or greater) element for every index,
//! the largest rectangle in a histogram, stock-span queries, and Cartesian
//! tree construction.

/// For each index `i`, returns the index of the nearest element to the left
/// of `i` that is strictly smaller than `a[i]`, or `None` if no such index
/// exists.
///
/// - Time: `O(n)`.
/// - Space: `O(n)`.
pub fn previous_smaller<T: Ord>(a: &[T]) -> Vec<Option<usize>> {
    let mut out = vec![None; a.len()];
    let mut stack: Vec<usize> = Vec::with_capacity(a.len());
    for i in 0..a.len() {
        while let Some(&top) = stack.last() {
            if a[top] >= a[i] {
                stack.pop();
            } else {
                break;
            }
        }
        out[i] = stack.last().copied();
        stack.push(i);
    }
    out
}

/// For each index `i`, returns the index of the nearest element to the right
/// of `i` that is strictly smaller than `a[i]`, or `None` if no such index
/// exists.
///
/// - Time: `O(n)`.
/// - Space: `O(n)`.
pub fn next_smaller<T: Ord>(a: &[T]) -> Vec<Option<usize>> {
    let mut out = vec![None; a.len()];
    let mut stack: Vec<usize> = Vec::with_capacity(a.len());
    for i in (0..a.len()).rev() {
        while let Some(&top) = stack.last() {
            if a[top] >= a[i] {
                stack.pop();
            } else {
                break;
            }
        }
        out[i] = stack.last().copied();
        stack.push(i);
    }
    out
}

/// For each index `i`, returns the index of the nearest element to the left
/// of `i` that is strictly greater than `a[i]`, or `None`.
///
/// - Time: `O(n)`.
/// - Space: `O(n)`.
pub fn previous_greater<T: Ord>(a: &[T]) -> Vec<Option<usize>> {
    let mut out = vec![None; a.len()];
    let mut stack: Vec<usize> = Vec::with_capacity(a.len());
    for i in 0..a.len() {
        while let Some(&top) = stack.last() {
            if a[top] <= a[i] {
                stack.pop();
            } else {
                break;
            }
        }
        out[i] = stack.last().copied();
        stack.push(i);
    }
    out
}

/// For each index `i`, returns the index of the nearest element to the right
/// of `i` that is strictly greater than `a[i]`, or `None`.
///
/// - Time: `O(n)`.
/// - Space: `O(n)`.
pub fn next_greater<T: Ord>(a: &[T]) -> Vec<Option<usize>> {
    let mut out = vec![None; a.len()];
    let mut stack: Vec<usize> = Vec::with_capacity(a.len());
    for i in (0..a.len()).rev() {
        while let Some(&top) = stack.last() {
            if a[top] <= a[i] {
                stack.pop();
            } else {
                break;
            }
        }
        out[i] = stack.last().copied();
        stack.push(i);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{next_greater, next_smaller, previous_greater, previous_smaller};
    use quickcheck_macros::quickcheck;

    fn brute_prev_smaller(a: &[i32]) -> Vec<Option<usize>> {
        (0..a.len())
            .map(|i| (0..i).rev().find(|&j| a[j] < a[i]))
            .collect()
    }

    fn brute_next_smaller(a: &[i32]) -> Vec<Option<usize>> {
        (0..a.len())
            .map(|i| ((i + 1)..a.len()).find(|&j| a[j] < a[i]))
            .collect()
    }

    fn brute_prev_greater(a: &[i32]) -> Vec<Option<usize>> {
        (0..a.len())
            .map(|i| (0..i).rev().find(|&j| a[j] > a[i]))
            .collect()
    }

    fn brute_next_greater(a: &[i32]) -> Vec<Option<usize>> {
        (0..a.len())
            .map(|i| ((i + 1)..a.len()).find(|&j| a[j] > a[i]))
            .collect()
    }

    #[test]
    fn empty() {
        let v: Vec<i32> = vec![];
        assert!(previous_smaller(&v).is_empty());
        assert!(next_smaller(&v).is_empty());
        assert!(previous_greater(&v).is_empty());
        assert!(next_greater(&v).is_empty());
    }

    #[test]
    fn single() {
        assert_eq!(previous_smaller(&[5]), vec![None]);
        assert_eq!(next_smaller(&[5]), vec![None]);
        assert_eq!(previous_greater(&[5]), vec![None]);
        assert_eq!(next_greater(&[5]), vec![None]);
    }

    #[test]
    fn ascending() {
        let a = [1, 2, 3, 4, 5];
        assert_eq!(
            previous_smaller(&a),
            vec![None, Some(0), Some(1), Some(2), Some(3)]
        );
        assert_eq!(next_smaller(&a), vec![None, None, None, None, None]);
    }

    #[test]
    fn descending() {
        let a = [5, 4, 3, 2, 1];
        assert_eq!(previous_smaller(&a), vec![None, None, None, None, None]);
        assert_eq!(
            next_smaller(&a),
            vec![Some(1), Some(2), Some(3), Some(4), None]
        );
    }

    #[test]
    fn duplicates_use_strict_less() {
        // ties don't count as "smaller" -> equal neighbours give None.
        let a = [3, 3, 3];
        assert_eq!(previous_smaller(&a), vec![None, None, None]);
        assert_eq!(next_smaller(&a), vec![None, None, None]);
    }

    #[test]
    fn known_pattern() {
        let a = [4, 2, 5, 1, 6];
        assert_eq!(
            previous_smaller(&a),
            vec![None, None, Some(1), None, Some(3)]
        );
        assert_eq!(
            next_smaller(&a),
            vec![Some(1), Some(3), Some(3), None, None]
        );
        assert_eq!(
            previous_greater(&a),
            vec![None, Some(0), None, Some(2), None]
        );
        assert_eq!(
            next_greater(&a),
            vec![Some(2), Some(2), Some(4), Some(4), None]
        );
    }

    #[allow(clippy::needless_pass_by_value)]
    #[quickcheck]
    fn prev_smaller_matches_brute(a: Vec<i32>) -> bool {
        let a: Vec<i32> = a.into_iter().take(60).collect();
        previous_smaller(&a) == brute_prev_smaller(&a)
    }

    #[allow(clippy::needless_pass_by_value)]
    #[quickcheck]
    fn next_smaller_matches_brute(a: Vec<i32>) -> bool {
        let a: Vec<i32> = a.into_iter().take(60).collect();
        next_smaller(&a) == brute_next_smaller(&a)
    }

    #[allow(clippy::needless_pass_by_value)]
    #[quickcheck]
    fn prev_greater_matches_brute(a: Vec<i32>) -> bool {
        let a: Vec<i32> = a.into_iter().take(60).collect();
        previous_greater(&a) == brute_prev_greater(&a)
    }

    #[allow(clippy::needless_pass_by_value)]
    #[quickcheck]
    fn next_greater_matches_brute(a: Vec<i32>) -> bool {
        let a: Vec<i32> = a.into_iter().take(60).collect();
        next_greater(&a) == brute_next_greater(&a)
    }
}
