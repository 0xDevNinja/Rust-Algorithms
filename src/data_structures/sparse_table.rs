//! Sparse table for static range queries under an idempotent associative
//! operator. Given an array of `n` values and a binary operator `op` that is
//! both **associative** and **idempotent** (i.e. `op(x, x) == x`), the table
//! answers closed-range queries `op(a[l], a[l+1], ..., a[r])` in `O(1)` after
//! `O(n log n)` preprocessing and `O(n log n)` space.
//!
//! Idempotency is required because the `O(1)` query overlaps two power-of-two
//! windows that together cover `[l, r]` and may double-count elements; with
//! `op(x, x) == x` the overlap is harmless. Valid operators include `min`,
//! `max`, `gcd`, bitwise `AND`, and bitwise `OR`. **Sum is not idempotent**
//! (`x + x != x` in general), so this structure must not be used for range
//! sums — use a Fenwick tree or segment tree for those.
//!
//! All public indices are 0-based and queries take the inclusive interval
//! `[l, r]`. Building from an empty slice yields an empty table on which any
//! query panics.

/// Sparse table over an idempotent associative operator.
///
/// - Time: `O(n log n)` to build (see [`Self::new`]), `O(1)` per
///   [`Self::query`].
/// - Space: `O(n log n)`.
/// - Operator requirements: `op` MUST be associative and idempotent
///   (`op(x, x) == x`). Suitable for `min`, `max`, `gcd`, bitwise `AND`,
///   bitwise `OR`. Not valid for sum or product.
/// - Indexing: 0-based; [`Self::query`] takes the inclusive interval
///   `[l, r]` with `l <= r < n`.
pub struct SparseTable<T: Copy, F: Fn(T, T) -> T> {
    table: Vec<Vec<T>>,
    op: F,
}

impl<T: Copy, F: Fn(T, T) -> T> SparseTable<T, F> {
    /// Builds a sparse table from `values` under the idempotent associative
    /// operator `op` in `O(n log n)` time and space.
    ///
    /// Empty input is allowed and yields an empty table; any subsequent
    /// [`Self::query`] call will panic.
    ///
    /// The caller is responsible for ensuring `op` is associative and
    /// idempotent. Passing a non-idempotent operator (e.g. addition) will
    /// produce silently incorrect results, not a panic.
    pub fn new(values: &[T], op: F) -> Self {
        let n = values.len();
        if n == 0 {
            return Self {
                table: Vec::new(),
                op,
            };
        }

        // Number of rows = floor(log2(n)) + 1.
        let log = (usize::BITS - n.leading_zeros()) as usize;
        let mut table: Vec<Vec<T>> = Vec::with_capacity(log);
        table.push(values.to_vec());

        let mut k = 1;
        while (1usize << k) <= n {
            let len = 1usize << k;
            let half = len >> 1;
            let row_len = n - len + 1;
            let mut row = Vec::with_capacity(row_len);
            for i in 0..row_len {
                let left = table[k - 1][i];
                let right = table[k - 1][i + half];
                row.push(op(left, right));
            }
            table.push(row);
            k += 1;
        }

        Self { table, op }
    }

    /// Number of elements in the underlying array.
    pub fn len(&self) -> usize {
        self.table.first().map_or(0, Vec::len)
    }

    /// True if the table was built from an empty slice.
    pub const fn is_empty(&self) -> bool {
        self.table.is_empty()
    }

    /// Returns `op(values[l], values[l+1], ..., values[r])` in `O(1)` over the
    /// inclusive range `[l, r]`.
    ///
    /// # Panics
    /// Panics with a descriptive message if the table is empty, if `l > r`,
    /// or if `r >= len()`.
    pub fn query(&self, l: usize, r: usize) -> T {
        assert!(
            !self.table.is_empty(),
            "SparseTable::query: cannot query an empty table"
        );
        assert!(l <= r, "SparseTable::query: empty range [{l}, {r}]");
        let n = self.table[0].len();
        assert!(
            r < n,
            "SparseTable::query: range [{l}, {r}] out of bounds for len {n}"
        );

        let len = r - l + 1;
        // k = floor(log2(len)).
        let k = (usize::BITS - 1 - len.leading_zeros()) as usize;
        let half = 1usize << k;
        let left = self.table[k][l];
        let right = self.table[k][r + 1 - half];
        (self.op)(left, right)
    }
}

#[cfg(test)]
mod tests {
    use super::SparseTable;
    use quickcheck::TestResult;
    use quickcheck_macros::quickcheck;

    fn brute_min(values: &[i64], l: usize, r: usize) -> i64 {
        *values[l..=r].iter().min().unwrap()
    }

    fn brute_max(values: &[i64], l: usize, r: usize) -> i64 {
        *values[l..=r].iter().max().unwrap()
    }

    fn gcd(a: u64, b: u64) -> u64 {
        if b == 0 {
            a
        } else {
            gcd(b, a % b)
        }
    }

    fn brute_gcd(values: &[u64], l: usize, r: usize) -> u64 {
        values[l..=r].iter().copied().fold(0_u64, gcd)
    }

    #[test]
    fn empty_table_reports_empty() {
        let st: SparseTable<i64, _> = SparseTable::new(&[], i64::min);
        assert!(st.is_empty());
        assert_eq!(st.len(), 0);
    }

    #[test]
    #[should_panic(expected = "cannot query an empty table")]
    fn empty_query_panics() {
        let st: SparseTable<i64, _> = SparseTable::new(&[], i64::min);
        let _ = st.query(0, 0);
    }

    #[test]
    fn single_element() {
        let st = SparseTable::new(&[42_i64], i64::min);
        assert_eq!(st.len(), 1);
        assert_eq!(st.query(0, 0), 42);
    }

    #[test]
    fn full_array_min() {
        let values = [3_i64, 1, 4, 1, 5, 9, 2, 6, 5, 3];
        let st = SparseTable::new(&values, i64::min);
        assert_eq!(st.query(0, values.len() - 1), 1);
    }

    #[test]
    fn full_array_max() {
        let values = [3_i64, 1, 4, 1, 5, 9, 2, 6, 5, 3];
        let st = SparseTable::new(&values, i64::max);
        assert_eq!(st.query(0, values.len() - 1), 9);
    }

    #[test]
    fn range_min_matches_brute_force() {
        let values: Vec<i64> = vec![7, 2, 5, 9, 1, 8, 3, 6, 4, 0, 11, -3, 12, 5, 8];
        let st = SparseTable::new(&values, i64::min);
        for l in 0..values.len() {
            for r in l..values.len() {
                assert_eq!(st.query(l, r), brute_min(&values, l, r));
            }
        }
    }

    #[test]
    fn range_max_matches_brute_force() {
        let values: Vec<i64> = vec![7, 2, 5, 9, 1, 8, 3, 6, 4, 0, 11, -3, 12, 5, 8];
        let st = SparseTable::new(&values, i64::max);
        for l in 0..values.len() {
            for r in l..values.len() {
                assert_eq!(st.query(l, r), brute_max(&values, l, r));
            }
        }
    }

    #[test]
    fn range_gcd_matches_brute_force() {
        let values: Vec<u64> = vec![12, 18, 24, 36, 48, 60, 9, 27, 81, 6];
        let st = SparseTable::new(&values, gcd);
        for l in 0..values.len() {
            for r in l..values.len() {
                assert_eq!(st.query(l, r), brute_gcd(&values, l, r));
            }
        }
    }

    #[test]
    fn single_element_range_each_index() {
        let values: Vec<i64> = vec![3, 1, 4, 1, 5, 9, 2, 6];
        let st = SparseTable::new(&values, i64::min);
        for (i, &v) in values.iter().enumerate() {
            assert_eq!(st.query(i, i), v);
        }
    }

    #[test]
    fn repeated_queries_are_consistent() {
        let values: Vec<i64> = vec![5, 2, 7, 1, 9, 4, 6, 3, 8, 0];
        let st = SparseTable::new(&values, i64::min);
        for _ in 0..5 {
            assert_eq!(st.query(2, 7), 1);
            assert_eq!(st.query(0, 9), 0);
            assert_eq!(st.query(4, 8), 3);
        }
    }

    #[test]
    fn large_n_range_min_against_brute_force() {
        // n = 1024, deterministic xorshift to fill values.
        let n = 1024_usize;
        let mut state: u64 = 0xDEAD_BEEF_CAFE_BABE;
        let values: Vec<i64> = (0..n)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                ((state as i64) % 200_000) - 100_000
            })
            .collect();

        let st = SparseTable::new(&values, i64::min);

        // Sample a slice of (l, r) pairs rather than the full O(n^2) grid.
        let mut samples = 0;
        let mut probe: u64 = 0x1234_5678_9ABC_DEF0;
        while samples < 4096 {
            probe ^= probe << 13;
            probe ^= probe >> 7;
            probe ^= probe << 17;
            let a = (probe as usize) % n;
            probe ^= probe << 13;
            probe ^= probe >> 7;
            probe ^= probe << 17;
            let b = (probe as usize) % n;
            let (l, r) = if a <= b { (a, b) } else { (b, a) };
            assert_eq!(st.query(l, r), brute_min(&values, l, r));
            samples += 1;
        }
    }

    #[quickcheck]
    #[allow(clippy::needless_pass_by_value)]
    fn prop_range_min_matches_brute_force(values: Vec<i64>, queries: Vec<(u8, u8)>) -> TestResult {
        if values.is_empty() || values.len() > 64 {
            return TestResult::discard();
        }
        if queries.len() > 100 {
            return TestResult::discard();
        }
        let n = values.len();
        let st = SparseTable::new(&values, i64::min);
        for &(a, b) in &queries {
            let lo = (a as usize) % n;
            let hi = (b as usize) % n;
            let (l, r) = if lo <= hi { (lo, hi) } else { (hi, lo) };
            if st.query(l, r) != brute_min(&values, l, r) {
                return TestResult::failed();
            }
        }
        TestResult::passed()
    }
}
