//! One-dimensional prefix sum (immutable range-sum helper).
//!
//! Given an array `a[0..n]`, builds a prefix table `P` of length `n + 1` such
//! that `P[i] = a[0] + a[1] + .. + a[i - 1]` and `P[0] = 0`. Range sums on the
//! half-open interval `[l, r)` are answered in `O(1)` as `P[r] - P[l]`. The
//! table is built once in `O(n)` and never mutated; for online updates use a
//! Fenwick tree instead.

/// Immutable 1-D prefix-sum table over `i64`.
///
/// - Build: `O(n)`.
/// - Query: `O(1)` for any half-open range `[l, r)`.
/// - Space: `O(n)`.
pub struct PrefixSum {
    prefix: Vec<i64>,
}

impl PrefixSum {
    /// Builds a prefix-sum table from a slice. Empty input yields a table
    /// containing just the sentinel `0` at index 0.
    pub fn new(values: &[i64]) -> Self {
        let mut prefix = Vec::with_capacity(values.len() + 1);
        prefix.push(0);
        let mut acc: i64 = 0;
        for &v in values {
            acc += v;
            prefix.push(acc);
        }
        Self { prefix }
    }

    /// Length of the underlying array.
    pub const fn len(&self) -> usize {
        self.prefix.len() - 1
    }

    /// True if the underlying array is empty.
    pub const fn is_empty(&self) -> bool {
        self.prefix.len() == 1
    }

    /// Returns the sum over the half-open range `a[l..r]` in `O(1)`. Empty
    /// ranges (`l == r`) return `0`.
    ///
    /// # Panics
    /// Panics if `l > r` or `r > len()`.
    pub fn range_sum(&self, l: usize, r: usize) -> i64 {
        assert!(l <= r, "PrefixSum::range_sum: l ({l}) > r ({r})");
        assert!(
            r <= self.len(),
            "PrefixSum::range_sum: r ({r}) exceeds len {}",
            self.len()
        );
        self.prefix[r] - self.prefix[l]
    }

    /// Returns the inclusive range sum `a[l..=r]`. Convenience for
    /// closed-interval callers.
    ///
    /// # Panics
    /// Panics if `l > r` or `r >= len()`.
    pub fn range_sum_inclusive(&self, l: usize, r: usize) -> i64 {
        assert!(
            r < self.len(),
            "PrefixSum::range_sum_inclusive: r ({r}) out of bounds for len {}",
            self.len()
        );
        self.range_sum(l, r + 1)
    }
}

#[cfg(test)]
mod tests {
    use super::PrefixSum;
    use quickcheck_macros::quickcheck;

    #[test]
    fn empty() {
        let p = PrefixSum::new(&[]);
        assert!(p.is_empty());
        assert_eq!(p.len(), 0);
        assert_eq!(p.range_sum(0, 0), 0);
    }

    #[test]
    fn single() {
        let p = PrefixSum::new(&[7_i64]);
        assert_eq!(p.range_sum(0, 1), 7);
        assert_eq!(p.range_sum(0, 0), 0);
        assert_eq!(p.range_sum(1, 1), 0);
        assert_eq!(p.range_sum_inclusive(0, 0), 7);
    }

    #[test]
    fn known_sequence() {
        let a = [1_i64, 2, 3, 4, 5];
        let p = PrefixSum::new(&a);
        assert_eq!(p.range_sum(0, 5), 15);
        assert_eq!(p.range_sum(1, 4), 9);
        assert_eq!(p.range_sum_inclusive(2, 4), 12);
    }

    #[test]
    fn negatives_and_mix() {
        let p = PrefixSum::new(&[-3_i64, 1, -4, 1, 5, -9, 2, 6]);
        assert_eq!(p.range_sum(0, 8), -1);
        assert_eq!(p.range_sum(2, 6), -7);
    }

    #[test]
    #[should_panic(expected = "exceeds len")]
    fn out_of_bounds_panics() {
        let p = PrefixSum::new(&[1_i64, 2, 3]);
        let _ = p.range_sum(0, 10);
    }

    #[test]
    #[should_panic(expected = "l (3) > r (1)")]
    fn inverted_range_panics() {
        let p = PrefixSum::new(&[1_i64, 2, 3, 4]);
        let _ = p.range_sum(3, 1);
    }

    #[allow(clippy::needless_pass_by_value)]
    #[quickcheck]
    fn matches_iter_sum(values: Vec<i32>, l: u32, r: u32) -> bool {
        let v: Vec<i64> = values.into_iter().take(50).map(i64::from).collect();
        if v.is_empty() {
            return true;
        }
        let n = v.len();
        let l = (l as usize) % (n + 1);
        let r = (r as usize) % (n + 1);
        let (l, r) = if l <= r { (l, r) } else { (r, l) };
        PrefixSum::new(&v).range_sum(l, r) == v[l..r].iter().sum::<i64>()
    }
}
