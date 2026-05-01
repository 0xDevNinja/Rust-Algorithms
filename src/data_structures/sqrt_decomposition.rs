//! Sqrt decomposition over `i64` supporting point updates and inclusive range
//! sum queries on `[l, r]`. The array is split into blocks of size
//! `floor(sqrt(n)).max(1)` and each block keeps a precomputed sum, which gives
//! O(sqrt(n)) per query and O(1) per point update with O(n) space. All public
//! indices are 0-based; range queries use the closed interval `[l, r]`.

/// Sqrt-decomposed array of `i64` supporting point set + closed-range sum.
///
/// - Time: `O(1)` per [`Self::update`], `O(sqrt(n))` per [`Self::range_sum`].
/// - Space: `O(n)`.
/// - Indexing: 0-based; [`Self::range_sum`] takes the inclusive interval
///   `[l, r]` with `l <= r < n`.
pub struct SqrtDecomposition {
    data: Vec<i64>,
    block_sum: Vec<i64>,
    block_size: usize,
}

impl SqrtDecomposition {
    /// Builds a sqrt-decomposed structure from `values`. Empty input is
    /// allowed and yields a structure on which only no-op operations make
    /// sense (any indexed access will panic).
    pub fn new(values: &[i64]) -> Self {
        let n = values.len();
        let block_size = ((n as f64).sqrt() as usize).max(1);
        let num_blocks = n.div_ceil(block_size);
        let mut block_sum = vec![0_i64; num_blocks];
        for (i, &v) in values.iter().enumerate() {
            block_sum[i / block_size] += v;
        }
        Self {
            data: values.to_vec(),
            block_sum,
            block_size,
        }
    }

    /// Number of elements in the underlying array.
    pub const fn len(&self) -> usize {
        self.data.len()
    }

    /// True if the structure was built from an empty slice.
    pub const fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Sets `data[idx] = value` in `O(1)`.
    ///
    /// # Panics
    /// Panics with a descriptive message if `idx >= len()`.
    pub fn update(&mut self, idx: usize, value: i64) {
        assert!(
            idx < self.data.len(),
            "SqrtDecomposition::update: index {idx} out of bounds for len {}",
            self.data.len()
        );
        let block = idx / self.block_size;
        self.block_sum[block] += value - self.data[idx];
        self.data[idx] = value;
    }

    /// Returns the inclusive range sum `data[l] + .. + data[r]` in
    /// `O(sqrt(n))`.
    ///
    /// # Panics
    /// Panics with a descriptive message if `l > r` or `r >= len()`.
    pub fn range_sum(&self, l: usize, r: usize) -> i64 {
        assert!(
            l <= r,
            "SqrtDecomposition::range_sum: empty range [{l}, {r}]"
        );
        assert!(
            r < self.data.len(),
            "SqrtDecomposition::range_sum: range [{l}, {r}] out of bounds for len {}",
            self.data.len()
        );

        let bs = self.block_size;
        let left_block = l / bs;
        let right_block = r / bs;

        if left_block == right_block {
            return self.data[l..=r].iter().sum();
        }

        let mut sum: i64 = 0;
        // Left partial block: [l, end_of_left_block].
        let left_end = (left_block + 1) * bs - 1;
        sum += self.data[l..=left_end].iter().sum::<i64>();
        // Full middle blocks.
        for b in (left_block + 1)..right_block {
            sum += self.block_sum[b];
        }
        // Right partial block: [start_of_right_block, r].
        let right_start = right_block * bs;
        sum += self.data[right_start..=r].iter().sum::<i64>();
        sum
    }
}

#[cfg(test)]
mod tests {
    use super::SqrtDecomposition;
    use quickcheck::TestResult;
    use quickcheck_macros::quickcheck;

    /// Bound for values used in the property test, chosen so cumulative sums
    /// over up to 64 values stay well within `i64`.
    const BOUND: i64 = 1_000_000;

    #[test]
    fn empty() {
        let s = SqrtDecomposition::new(&[]);
        assert!(s.is_empty());
        assert_eq!(s.len(), 0);
    }

    #[test]
    fn single_element() {
        let mut s = SqrtDecomposition::new(&[42]);
        assert_eq!(s.range_sum(0, 0), 42);
        s.update(0, -7);
        assert_eq!(s.range_sum(0, 0), -7);
    }

    #[test]
    fn full_array_query() {
        let s = SqrtDecomposition::new(&[1, 2, 3, 4, 5, 6, 7, 8, 9]);
        assert_eq!(s.range_sum(0, 8), 45);
    }

    #[test]
    fn single_element_range() {
        let s = SqrtDecomposition::new(&[3, 1, 4, 1, 5, 9, 2, 6]);
        for (i, &v) in [3, 1, 4, 1, 5, 9, 2, 6].iter().enumerate() {
            assert_eq!(s.range_sum(i, i), v);
        }
    }

    #[test]
    fn range_within_one_block() {
        // n = 16 -> block_size = 4. Block 1 spans indices 4..=7.
        let s = SqrtDecomposition::new(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
        assert_eq!(s.range_sum(4, 6), 5 + 6 + 7);
        assert_eq!(s.range_sum(8, 11), 9 + 10 + 11 + 12);
    }

    #[test]
    fn range_spanning_multiple_blocks() {
        // n = 16 -> block_size = 4. Range covers a left partial, middle full
        // blocks, and a right partial.
        let s = SqrtDecomposition::new(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
        assert_eq!(s.range_sum(2, 13), (3..=14).sum::<i64>());
        assert_eq!(s.range_sum(1, 14), (2..=15).sum::<i64>());
    }

    #[test]
    fn updates_followed_by_queries() {
        let mut s = SqrtDecomposition::new(&[0; 10]);
        for i in 0..10 {
            s.update(i, (i as i64) + 1);
        }
        assert_eq!(s.range_sum(0, 9), 55);
        s.update(4, 100);
        // Replaced value 5 with 100, delta = +95.
        assert_eq!(s.range_sum(0, 9), 55 + 95);
        assert_eq!(s.range_sum(2, 6), 3 + 4 + 100 + 6 + 7);
    }

    #[test]
    fn random_ops_against_brute_force() {
        // n = 1024 random ops vs brute force.
        let n = 1024;
        let mut reference: Vec<i64> = (0..n).map(|i| (i as i64) * 3 - 17).collect();
        let mut s = SqrtDecomposition::new(&reference);

        // Deterministic xorshift64 to avoid pulling in `rand`.
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };

        for _ in 0..1024 {
            let r = next();
            if r & 1 == 0 {
                let idx = (r >> 1) as usize % n;
                let val = ((next() % 2001) as i64) - 1000;
                reference[idx] = val;
                s.update(idx, val);
            } else {
                let a = (r >> 1) as usize % n;
                let b = (next() as usize) % n;
                let (l, hi) = if a <= b { (a, b) } else { (b, a) };
                let expected: i64 = reference[l..=hi].iter().sum();
                assert_eq!(s.range_sum(l, hi), expected);
            }
        }
    }

    #[quickcheck]
    #[allow(clippy::needless_pass_by_value)]
    fn prop_matches_brute_force(values: Vec<i64>, ops: Vec<(bool, u8, u8, i64)>) -> TestResult {
        if values.is_empty() || values.len() > 64 {
            return TestResult::discard();
        }
        if ops.len() > 100 {
            return TestResult::discard();
        }
        // Clamp values to a range whose cumulative sums comfortably fit in
        // `i64` for any sequence of operations the property test produces.
        let bounded: Vec<i64> = values.iter().map(|v| v % BOUND).collect();
        let n = bounded.len();
        let mut reference = bounded.clone();
        let mut s = SqrtDecomposition::new(&bounded);

        for &(is_query, a, b, val) in &ops {
            if is_query {
                let lo = (a as usize) % n;
                let hi_raw = (b as usize) % n;
                let (l, r) = if lo <= hi_raw {
                    (lo, hi_raw)
                } else {
                    (hi_raw, lo)
                };
                let expected: i64 = reference[l..=r].iter().sum();
                if s.range_sum(l, r) != expected {
                    return TestResult::failed();
                }
            } else {
                let idx = (a as usize) % n;
                let bounded_val = val % BOUND;
                reference[idx] = bounded_val;
                s.update(idx, bounded_val);
            }
        }
        TestResult::passed()
    }
}
