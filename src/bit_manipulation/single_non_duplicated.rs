//! Find the unique element(s) in a slice using XOR.
//!
//! Two classic interview problems solved with the algebraic structure of
//! XOR over `u64`:
//!
//! 1. [`single_non_duplicated`] — given a slice in which exactly one
//!    element appears once and every other element appears exactly
//!    twice, return the lone element. Because XOR is commutative,
//!    associative, and self-inverse (`x ^ x == 0`, `x ^ 0 == x`),
//!    folding the entire slice with XOR cancels every duplicated pair
//!    and leaves the unique value.
//!
//! 2. [`two_non_duplicated`] — given a slice in which exactly two
//!    elements appear once and every other element appears exactly
//!    twice, return the pair as `(smaller, larger)`. The XOR fold
//!    yields `a ^ b` where `a` and `b` are the two unique values.
//!    Since `a != b`, that XOR is non-zero, and any of its set bits
//!    distinguishes `a` from `b`. Partitioning the slice by that bit
//!    splits the problem into two independent instances of case 1.
//!
//! ## Complexity
//!
//! Both routines run in **O(n)** time and **O(1)** auxiliary space,
//! making a single linear pass for [`single_non_duplicated`] and two
//! linear passes for [`two_non_duplicated`].
//!
//! ## Preconditions
//!
//! The input shape is **not** validated. If the caller violates the
//! "exactly one / exactly two unique" contract, the returned value is
//! unspecified (it is whatever falls out of the XOR algebra) but the
//! routines never panic on well-typed input.

/// Returns the single element that appears an odd number of times in
/// `nums`, assuming exactly one element appears once and every other
/// element appears exactly twice.
///
/// # Algorithm
///
/// XOR is commutative, associative, and self-inverse, so folding the
/// slice with `^` cancels every duplicated pair (`x ^ x == 0`) and the
/// fold's identity (`0`) leaves the unique value untouched.
///
/// # Complexity
///
/// O(n) time, O(1) space. Single pass over the input.
///
/// # Contract
///
/// The caller is responsible for ensuring that exactly one element
/// appears once and every other element appears exactly twice. On
/// malformed input the result is unspecified (it is the XOR fold of
/// the slice, modulo whatever cancellations apply); this function
/// never panics.
///
/// # Examples
///
/// ```
/// use rust_algorithms::bit_manipulation::single_non_duplicated::single_non_duplicated;
///
/// assert_eq!(single_non_duplicated(&[2, 2, 1]), 1);
/// assert_eq!(single_non_duplicated(&[4, 1, 2, 1, 2]), 4);
/// ```
#[inline]
#[must_use]
pub fn single_non_duplicated(nums: &[u64]) -> u64 {
    nums.iter().fold(0u64, |acc, &x| acc ^ x)
}

/// Returns the two elements that each appear exactly once in `nums`,
/// assuming exactly two such elements exist and every other element
/// appears exactly twice. The result is ordered as `(smaller, larger)`.
///
/// # Algorithm
///
/// 1. XOR-fold the slice; the result is `a ^ b` where `a` and `b` are
///    the two unique values. Because `a != b`, this XOR is non-zero.
/// 2. Pick any set bit of `a ^ b` — concretely the lowest, isolated as
///    `xor & xor.wrapping_neg()`. That bit is set in exactly one of
///    `a`, `b`, so it partitions the slice into two halves: each half
///    contains one of the unique values plus only paired elements.
/// 3. XOR-fold each half independently to recover `a` and `b`, then
///    sort the pair.
///
/// # Complexity
///
/// O(n) time, O(1) space. Two passes over the input.
///
/// # Contract
///
/// Caller must ensure exactly two elements appear once and every other
/// element appears exactly twice. The two unique elements must be
/// distinct (which is implied by the contract: if they were equal,
/// that value would appear an even number of times).
///
/// On malformed input the result is unspecified; this function never
/// panics on well-typed slices.
///
/// # Examples
///
/// ```
/// use rust_algorithms::bit_manipulation::single_non_duplicated::two_non_duplicated;
///
/// assert_eq!(two_non_duplicated(&[1, 2, 1, 3, 2, 5]), (3, 5));
/// ```
#[inline]
#[must_use]
pub fn two_non_duplicated(nums: &[u64]) -> (u64, u64) {
    let xor = nums.iter().fold(0u64, |acc, &x| acc ^ x);
    // Isolate the lowest set bit of `xor`. This bit is guaranteed to
    // be present in exactly one of the two unique values.
    let distinguishing = xor & xor.wrapping_neg();

    let mut a: u64 = 0;
    let mut b: u64 = 0;
    for &x in nums {
        if x & distinguishing == 0 {
            a ^= x;
        } else {
            b ^= x;
        }
    }

    if a <= b {
        (a, b)
    } else {
        (b, a)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ----- single_non_duplicated -----

    #[test]
    fn single_basic_two_two_one() {
        assert_eq!(single_non_duplicated(&[2, 2, 1]), 1);
    }

    #[test]
    fn single_basic_four_one_two_one_two() {
        assert_eq!(single_non_duplicated(&[4, 1, 2, 1, 2]), 4);
    }

    #[test]
    fn single_singleton_slice() {
        // Sole element appears once; the fold returns it directly.
        assert_eq!(single_non_duplicated(&[42]), 42);
    }

    #[test]
    fn single_unique_is_zero() {
        // The unique element may legitimately be zero.
        assert_eq!(single_non_duplicated(&[7, 7, 0, 9, 9]), 0);
    }

    #[test]
    fn single_large_values() {
        let unique = u64::MAX;
        let nums = vec![123, 456, 123, 456, unique, 789, 789];
        assert_eq!(single_non_duplicated(&nums), unique);
    }

    #[test]
    fn single_order_invariance() {
        // XOR is commutative, so any permutation gives the same answer.
        let mut a = vec![1u64, 2, 3, 1, 2];
        let b = vec![3u64, 2, 1, 2, 1];
        a.reverse();
        assert_eq!(single_non_duplicated(&a), single_non_duplicated(&b));
        assert_eq!(single_non_duplicated(&a), 3);
    }

    // ----- two_non_duplicated -----

    #[test]
    fn two_basic_three_five() {
        assert_eq!(two_non_duplicated(&[1, 2, 1, 3, 2, 5]), (3, 5));
    }

    #[test]
    fn two_returns_ordered_pair() {
        // Same multiset, different order — result is still (3, 5).
        assert_eq!(two_non_duplicated(&[5, 3, 2, 2, 1, 1]), (3, 5));
    }

    #[test]
    fn two_minimum_slice() {
        // Smallest legal input: just the two unique elements.
        assert_eq!(two_non_duplicated(&[7, 4]), (4, 7));
        assert_eq!(two_non_duplicated(&[4, 7]), (4, 7));
    }

    #[test]
    fn two_one_unique_is_zero() {
        // Zero is a valid unique element; (0, 5) must be ordered correctly.
        assert_eq!(two_non_duplicated(&[0, 5, 3, 3, 9, 9]), (0, 5));
    }

    #[test]
    fn two_large_values() {
        let nums = vec![u64::MAX, 1u64 << 63, 42, 42, 100, 100];
        // 1 << 63 < u64::MAX, so the smaller comes first.
        assert_eq!(two_non_duplicated(&nums), (1u64 << 63, u64::MAX));
    }

    #[test]
    fn two_distinguishing_bit_is_high() {
        // Construct a case where the two uniques differ only in a
        // high bit, exercising the wrapping_neg isolation.
        let a: u64 = 1 << 40;
        let b: u64 = (1 << 40) | (1 << 50);
        let nums = vec![a, b, 7, 7, 9, 9, 11, 11];
        assert_eq!(two_non_duplicated(&nums), (a, b));
    }

    #[test]
    fn two_pairs_only_around_uniques() {
        // Many duplicated pairs; the two uniques are buried.
        let nums = vec![1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 99, 100];
        assert_eq!(two_non_duplicated(&nums), (99, 100));
    }
}
