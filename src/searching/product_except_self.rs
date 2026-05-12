//! Product of array except self, computed without division.
//!
//! Given a slice `nums` of length `n`, return a `Vec<i64>` `out` such that
//! `out[i]` equals the product of every element of `nums` except `nums[i]`.
//! The classical one-line solution divides the total product by `nums[i]`,
//! which fails as soon as any zero is present (and changes precision for
//! floating point). This routine instead exploits the identity
//!
//! ```text
//! out[i] = (prefix product of nums[..i]) * (suffix product of nums[i+1..])
//! ```
//!
//! and computes it in two linear sweeps. The first pass walks left-to-right
//! filling `out[i]` with the prefix product strictly to the left of `i`. The
//! second pass walks right-to-left, multiplying each `out[i]` by a running
//! suffix product strictly to the right of `i`. Aside from the output buffer
//! demanded by the signature, only a single accumulator is used.
//!
//! - Time: `O(n)` (two passes over the input).
//! - Space: `O(1)` extra (the returned vector is not counted).
//!
//! No `unsafe`, no division, and no allocations beyond the output `Vec`.

/// Returns a vector `out` where `out[i]` is the product of every element of
/// `nums` other than `nums[i]`.
///
/// Edge cases follow the standard "empty product equals one" convention:
///
/// - An empty input returns an empty vector.
/// - A single-element input returns `vec![1]`, since the product over the
///   empty set of "other" elements is the multiplicative identity.
///
/// The implementation is division-free, so inputs containing one or more
/// zeros are handled correctly. Integer overflow follows the usual
/// debug-panic / release-wrap semantics of `i64` multiplication; callers
/// working with potentially huge magnitudes should pre-validate their data.
///
/// # Examples
///
/// ```
/// use rust_algorithms::searching::product_except_self::product_except_self;
///
/// assert_eq!(product_except_self(&[1, 2, 3, 4]), vec![24, 12, 8, 6]);
/// assert_eq!(product_except_self(&[0, 0, 3]), vec![0, 0, 0]);
/// assert_eq!(product_except_self(&[]), Vec::<i64>::new());
/// ```
pub fn product_except_self(nums: &[i64]) -> Vec<i64> {
    let n = nums.len();
    let mut out = vec![1_i64; n];
    if n == 0 {
        return out;
    }

    // Left-to-right pass: out[i] = product of nums[0..i].
    let mut prefix: i64 = 1;
    for i in 0..n {
        out[i] = prefix;
        prefix = prefix.wrapping_mul(nums[i]);
    }

    // Right-to-left pass: multiply by product of nums[i+1..].
    let mut suffix: i64 = 1;
    for i in (0..n).rev() {
        out[i] = out[i].wrapping_mul(suffix);
        suffix = suffix.wrapping_mul(nums[i]);
    }

    out
}

#[cfg(test)]
mod tests {
    use super::product_except_self;

    #[test]
    fn empty_input_returns_empty_vec() {
        let out = product_except_self(&[]);
        assert!(out.is_empty());
    }

    #[test]
    fn single_element_returns_one() {
        // The empty product is 1, regardless of the lone value.
        assert_eq!(product_except_self(&[42]), vec![1]);
        assert_eq!(product_except_self(&[0]), vec![1]);
        assert_eq!(product_except_self(&[-7]), vec![1]);
    }

    #[test]
    fn classic_example() {
        assert_eq!(product_except_self(&[1, 2, 3, 4]), vec![24, 12, 8, 6]);
    }

    #[test]
    fn two_elements_swap() {
        assert_eq!(product_except_self(&[2, 5]), vec![5, 2]);
    }

    #[test]
    fn single_zero_zeros_everywhere_except_its_index() {
        // With exactly one zero, only the zero's slot retains the product
        // of the remaining elements; every other slot is zero.
        assert_eq!(product_except_self(&[1, 2, 0, 4]), vec![0, 0, 8, 0]);
    }

    #[test]
    fn two_zeros_yield_all_zeros() {
        // Two or more zeros annihilate every slot.
        assert_eq!(product_except_self(&[0, 0, 3]), vec![0, 0, 0]);
        assert_eq!(product_except_self(&[0, 1, 0, 2]), vec![0, 0, 0, 0]);
    }

    #[test]
    fn negatives_handled_correctly() {
        // (-1)*2*3 = -6, etc.
        assert_eq!(product_except_self(&[-1, 2, 3]), vec![6, -3, -2]);
        // Even count of negatives -> positive total.
        assert_eq!(
            product_except_self(&[-1, -2, -3, -4]),
            vec![-24, -12, -8, -6]
        );
    }

    #[test]
    fn output_length_matches_input() {
        let nums: Vec<i64> = (1..=10).collect();
        assert_eq!(product_except_self(&nums).len(), nums.len());
    }

    #[test]
    fn ones_are_idempotent() {
        assert_eq!(product_except_self(&[1, 1, 1, 1]), vec![1, 1, 1, 1]);
    }
}
