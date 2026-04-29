//! # 2-D Segment Tree (point-update / rectangle-query)
//!
//! An outer segment tree over rows, where each outer node stores a 1-D inner
//! segment tree over the columns of its row range.
//!
//! ## Complexity
//! | Operation      | Time                 | Space     |
//! |----------------|----------------------|-----------|
//! | Build          | O(n · m)             | O(n · m)  |
//! | Point update   | O(log n · log m)     | —         |
//! | Rectangle query| O(log n · log m)     | —         |
//!
//! ## Preconditions
//! Row and column indices must be 0-based and within bounds.
//!
//! ## Limitations
//! **No lazy propagation.** Range-update / range-query in 2-D with lazy
//! propagation requires a significantly more complex "fractional cascading" or
//! persistent approach and is not implemented here.

use std::marker::PhantomData;

// ── Monoid trait ──────────────────────────────────────────────────────────────

/// An associative binary operation with an identity element.
///
/// Implementors must satisfy: `combine(identity(), x) == x`,
/// `combine(x, identity()) == x`, and associativity.
pub trait Monoid<T> {
    /// Returns the identity element (neutral element for `combine`).
    fn identity() -> T;
    /// Combines two values associatively.
    fn combine(a: T, b: T) -> T;
}

// ── Concrete monoid instances ─────────────────────────────────────────────────

/// Sum monoid over `i64`.
pub struct SumMonoid;

impl Monoid<i64> for SumMonoid {
    #[inline]
    fn identity() -> i64 {
        0
    }
    #[inline]
    fn combine(a: i64, b: i64) -> i64 {
        a + b
    }
}

/// Min monoid over `i64` (identity = `i64::MAX`).
pub struct MinMonoid;

impl Monoid<i64> for MinMonoid {
    #[inline]
    fn identity() -> i64 {
        i64::MAX
    }
    #[inline]
    fn combine(a: i64, b: i64) -> i64 {
        a.min(b)
    }
}

/// Max monoid over `i64` (identity = `i64::MIN`).
pub struct MaxMonoid;

impl Monoid<i64> for MaxMonoid {
    #[inline]
    fn identity() -> i64 {
        i64::MIN
    }
    #[inline]
    fn combine(a: i64, b: i64) -> i64 {
        a.max(b)
    }
}

// ── Helper: 1-D inner segment tree ops ───────────────────────────────────────

/// Query a 1-D segment tree stored in `tree[..]` over column range `[c1, c2]`.
fn inner_query<T, M>(tree: &[T], node: usize, lo: usize, hi: usize, c1: usize, c2: usize) -> T
where
    T: Copy,
    M: Monoid<T>,
{
    if c2 < lo || hi < c1 {
        return M::identity();
    }
    if c1 <= lo && hi <= c2 {
        return tree[node];
    }
    let mid = lo + (hi - lo) / 2;
    M::combine(
        inner_query::<T, M>(tree, 2 * node, lo, mid, c1, c2),
        inner_query::<T, M>(tree, 2 * node + 1, mid + 1, hi, c1, c2),
    )
}

/// Point-set a single column in a 1-D segment tree, re-aggregating upward.
fn inner_update<T, M>(tree: &mut [T], node: usize, lo: usize, hi: usize, col: usize, value: T)
where
    T: Copy,
    M: Monoid<T>,
{
    if lo == hi {
        tree[node] = value;
        return;
    }
    let mid = lo + (hi - lo) / 2;
    if col <= mid {
        inner_update::<T, M>(tree, 2 * node, lo, mid, col, value);
    } else {
        inner_update::<T, M>(tree, 2 * node + 1, mid + 1, hi, col, value);
    }
    tree[node] = M::combine(tree[2 * node], tree[2 * node + 1]);
}

/// Build a 1-D segment tree from a row slice.
fn inner_build<T, M>(tree: &mut [T], node: usize, lo: usize, hi: usize, row: &[T])
where
    T: Copy,
    M: Monoid<T>,
{
    if lo == hi {
        tree[node] = row[lo];
        return;
    }
    let mid = lo + (hi - lo) / 2;
    inner_build::<T, M>(tree, 2 * node, lo, mid, row);
    inner_build::<T, M>(tree, 2 * node + 1, mid + 1, hi, row);
    tree[node] = M::combine(tree[2 * node], tree[2 * node + 1]);
}

/// Build an internal outer node's inner tree by pairwise-combining two children.
fn inner_merge<T, M>(parent: &mut [T], left: &[T], right: &[T], node: usize, lo: usize, hi: usize)
where
    T: Copy,
    M: Monoid<T>,
{
    if lo == hi {
        parent[node] = M::combine(left[node], right[node]);
        return;
    }
    let mid = lo + (hi - lo) / 2;
    inner_merge::<T, M>(parent, left, right, 2 * node, lo, mid);
    inner_merge::<T, M>(parent, left, right, 2 * node + 1, mid + 1, hi);
    parent[node] = M::combine(parent[2 * node], parent[2 * node + 1]);
}

// ── SegmentTree2D ─────────────────────────────────────────────────────────────

/// 2-D segment tree for point-update / rectangle-query on an `n × m` grid.
///
/// The outer tree covers rows (size `4 * n`); each outer node `k` owns a
/// `Vec<T>` of size `4 * m` representing the inner 1-D segment tree over
/// the combined column values of that outer node's row range.
pub struct SegmentTree2D<T, M> {
    n: usize,
    m: usize,
    /// Flat array: `trees[outer_node]` is the inner seg-tree for that outer node.
    trees: Vec<Vec<T>>,
    _marker: PhantomData<M>,
}

impl<T, M> SegmentTree2D<T, M>
where
    T: Copy,
    M: Monoid<T>,
{
    // ── size helpers ──────────────────────────────────────────────────────

    fn outer_size(n: usize) -> usize {
        4 * n.max(1)
    }

    fn inner_size(m: usize) -> usize {
        4 * m.max(1)
    }

    // ── constructors ──────────────────────────────────────────────────────

    /// Creates an identity-filled `n × m` tree.
    pub fn new(n: usize, m: usize) -> Self {
        let id = M::identity();
        let trees = vec![vec![id; Self::inner_size(m)]; Self::outer_size(n)];
        Self {
            n,
            m,
            trees,
            _marker: PhantomData,
        }
    }

    /// Builds from a `values[0..n][0..m]` grid.
    ///
    /// Panics if any inner `Vec` has a length != `m`.
    pub fn from_grid(values: &[Vec<T>]) -> Self {
        let n = values.len();
        if n == 0 {
            return Self::new(0, 0);
        }
        let m = values[0].len();
        let mut st = Self::new(n, m);
        if m > 0 {
            st.build_outer(1, 0, n - 1, values);
        }
        st
    }

    // ── public API ────────────────────────────────────────────────────────

    /// Returns `(rows, cols)`.
    pub const fn dims(&self) -> (usize, usize) {
        (self.n, self.m)
    }

    /// Sets `grid[row][col] = value`.
    ///
    /// Out-of-bounds indices are silently ignored.
    pub fn point_update(&mut self, row: usize, col: usize, value: T) {
        if row >= self.n || col >= self.m {
            return;
        }
        self.update_outer(1, 0, self.n - 1, row, col, value);
    }

    /// Returns the aggregate over the inclusive rectangle
    /// `[r1, r2] × [c1, c2]`.
    ///
    /// Returns the identity if the range is empty or out of bounds.
    pub fn range_query(&self, r1: usize, c1: usize, r2: usize, c2: usize) -> T {
        if self.n == 0 || self.m == 0 || r1 > r2 || c1 > c2 {
            return M::identity();
        }
        let r2 = r2.min(self.n - 1);
        let c2 = c2.min(self.m - 1);
        self.query_outer(1, 0, self.n - 1, r1, r2, c1, c2)
    }

    // ── outer tree recursion ──────────────────────────────────────────────

    fn build_outer(&mut self, node: usize, lo: usize, hi: usize, values: &[Vec<T>]) {
        if lo == hi {
            // Leaf: build inner tree from the single row `values[lo]`.
            let row = &values[lo];
            inner_build::<T, M>(&mut self.trees[node], 1, 0, self.m - 1, row);
            return;
        }
        let mid = lo + (hi - lo) / 2;
        self.build_outer(2 * node, lo, mid, values);
        self.build_outer(2 * node + 1, mid + 1, hi, values);

        // Merge children into parent by splitting borrow.
        let (left_tree, right_tree) = {
            // We need immutable slices of children while mutably borrowing parent.
            // Collect into temporaries to avoid aliasing.
            let l = self.trees[2 * node].clone();
            let r = self.trees[2 * node + 1].clone();
            (l, r)
        };
        inner_merge::<T, M>(
            &mut self.trees[node],
            &left_tree,
            &right_tree,
            1,
            0,
            self.m - 1,
        );
    }

    fn update_outer(
        &mut self,
        node: usize,
        lo: usize,
        hi: usize,
        row: usize,
        col: usize,
        value: T,
    ) {
        if lo == hi {
            // Leaf: point-update the inner tree.
            inner_update::<T, M>(&mut self.trees[node], 1, 0, self.m - 1, col, value);
            return;
        }
        let mid = lo + (hi - lo) / 2;
        if row <= mid {
            self.update_outer(2 * node, lo, mid, row, col, value);
        } else {
            self.update_outer(2 * node + 1, mid + 1, hi, row, col, value);
        }
        // Re-aggregate: read children, write parent.
        let left_val = inner_query::<T, M>(&self.trees[2 * node], 1, 0, self.m - 1, col, col);
        let right_val = inner_query::<T, M>(&self.trees[2 * node + 1], 1, 0, self.m - 1, col, col);
        let merged = M::combine(left_val, right_val);
        inner_update::<T, M>(&mut self.trees[node], 1, 0, self.m - 1, col, merged);
    }

    #[allow(clippy::too_many_arguments)]
    fn query_outer(
        &self,
        node: usize,
        lo: usize,
        hi: usize,
        r1: usize,
        r2: usize,
        c1: usize,
        c2: usize,
    ) -> T {
        if r2 < lo || hi < r1 {
            return M::identity();
        }
        if r1 <= lo && hi <= r2 {
            return inner_query::<T, M>(&self.trees[node], 1, 0, self.m - 1, c1, c2);
        }
        let mid = lo + (hi - lo) / 2;
        M::combine(
            self.query_outer(2 * node, lo, mid, r1, r2, c1, c2),
            self.query_outer(2 * node + 1, mid + 1, hi, r1, r2, c1, c2),
        )
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::{MaxMonoid, MinMonoid, SegmentTree2D, SumMonoid};

    // Convenience alias
    type SumTree = SegmentTree2D<i64, SumMonoid>;

    // ── structural edge cases ─────────────────────────────────────────────

    #[test]
    fn one_by_one() {
        let grid = vec![vec![42_i64]];
        let st = SumTree::from_grid(&grid);
        assert_eq!(st.range_query(0, 0, 0, 0), 42);
    }

    #[test]
    fn one_by_n_degenerates_to_1d() {
        let grid = vec![vec![1_i64, 2, 3, 4, 5]];
        let st = SumTree::from_grid(&grid);
        assert_eq!(st.range_query(0, 0, 0, 4), 15);
        assert_eq!(st.range_query(0, 1, 0, 3), 9);
        assert_eq!(st.range_query(0, 2, 0, 2), 3);
    }

    #[test]
    fn n_by_one_degenerates_to_1d() {
        let grid: Vec<Vec<i64>> = (1..=5_i64).map(|v| vec![v]).collect();
        let st = SumTree::from_grid(&grid);
        assert_eq!(st.range_query(0, 0, 4, 0), 15);
        assert_eq!(st.range_query(1, 0, 3, 0), 9);
        assert_eq!(st.range_query(2, 0, 2, 0), 3);
    }

    // ── hand-computed sum example ─────────────────────────────────────────

    #[test]
    fn sum_monoid_small_grid() {
        // Grid:
        //  1  2  3
        //  4  5  6
        //  7  8  9
        let grid = vec![vec![1_i64, 2, 3], vec![4, 5, 6], vec![7, 8, 9]];
        let st = SumTree::from_grid(&grid);
        assert_eq!(st.range_query(0, 0, 2, 2), 45); // full grid
        assert_eq!(st.range_query(0, 0, 1, 1), 12); // top-left 2×2: 1+2+4+5
        assert_eq!(st.range_query(1, 1, 2, 2), 28); // bottom-right 2×2: 5+6+8+9
        assert_eq!(st.range_query(0, 2, 2, 2), 18); // right col: 3+6+9
        assert_eq!(st.range_query(2, 0, 2, 2), 24); // bottom row: 7+8+9
    }

    // ── min monoid ────────────────────────────────────────────────────────

    #[test]
    fn min_monoid_sub_rectangle() {
        let grid = vec![vec![9_i64, 3, 5], vec![2, 8, 1], vec![7, 4, 6]];
        let st: SegmentTree2D<i64, MinMonoid> = SegmentTree2D::from_grid(&grid);
        assert_eq!(st.range_query(0, 0, 2, 2), 1); // overall min
        assert_eq!(st.range_query(0, 0, 1, 1), 2); // top-left 2×2
        assert_eq!(st.range_query(0, 0, 0, 2), 3); // top row
        assert_eq!(st.range_query(1, 1, 2, 2), 1); // bottom-right 2×2
    }

    // ── max monoid ────────────────────────────────────────────────────────

    #[test]
    fn max_monoid_sub_rectangle() {
        let grid = vec![vec![1_i64, 5, 3], vec![4, 2, 9], vec![7, 8, 6]];
        let st: SegmentTree2D<i64, MaxMonoid> = SegmentTree2D::from_grid(&grid);
        assert_eq!(st.range_query(0, 0, 2, 2), 9); // overall max
        assert_eq!(st.range_query(0, 0, 1, 1), 5); // top-left 2×2
        assert_eq!(st.range_query(1, 1, 2, 2), 9); // bottom-right 2×2
    }

    // ── point update ─────────────────────────────────────────────────────

    #[test]
    fn point_update_affects_only_queried_cell() {
        let grid = vec![vec![1_i64, 2, 3], vec![4, 5, 6], vec![7, 8, 9]];
        let mut st = SumTree::from_grid(&grid);
        st.point_update(1, 1, 100); // set (1,1) from 5 → 100
        assert_eq!(st.range_query(1, 1, 1, 1), 100); // the cell itself
        assert_eq!(st.range_query(0, 0, 0, 2), 6); // top row unchanged
        assert_eq!(st.range_query(2, 0, 2, 2), 24); // bottom row unchanged
                                                    // full grid should reflect the delta (+95)
        assert_eq!(st.range_query(0, 0, 2, 2), 45 - 5 + 100);
    }

    #[test]
    fn single_cell_query_equals_cell_value() {
        let grid: Vec<Vec<i64>> = (0..4)
            .map(|r| (0..4).map(|c| (r * 4 + c) as i64).collect())
            .collect();
        let st = SumTree::from_grid(&grid);
        for r in 0..4 {
            for c in 0..4 {
                assert_eq!(
                    st.range_query(r, c, r, c),
                    grid[r][c],
                    "cell ({r},{c}) mismatch"
                );
            }
        }
    }

    // ── full-grid query equals naive sum ──────────────────────────────────

    #[test]
    fn full_grid_query_equals_naive_sum() {
        let grid: Vec<Vec<i64>> = (1..=4).map(|r| (1..=4).map(|c| r * c).collect()).collect();
        let naive: i64 = grid.iter().flatten().sum();
        let st = SumTree::from_grid(&grid);
        assert_eq!(st.range_query(0, 0, 3, 3), naive);
    }

    // ── dims accessor ─────────────────────────────────────────────────────

    #[test]
    fn dims_accessor() {
        let st = SumTree::new(5, 7);
        assert_eq!(st.dims(), (5, 7));
    }

    // ── quickcheck property test ──────────────────────────────────────────

    #[cfg(test)]
    mod prop {
        use super::SumTree;
        use quickcheck::TestResult;
        use quickcheck_macros::quickcheck;

        /// Brute-force rectangle sum over a 2-D array.
        fn brute_sum(grid: &[Vec<i64>], r1: usize, c1: usize, r2: usize, c2: usize) -> i64 {
            let mut s = 0_i64;
            for row in grid.iter().take(r2 + 1).skip(r1) {
                for &v in row.iter().take(c2 + 1).skip(c1) {
                    s += v;
                }
            }
            s
        }

        /// Random 5×5 grid, up to 8 point updates, then a random rectangle query,
        /// checked against brute force.
        ///
        /// Values are clamped to `[-100, 100]` so the sum over 25 cells never
        /// overflows `i64`.
        #[allow(clippy::needless_pass_by_value)]
        #[quickcheck]
        fn sum_matches_brute_force(
            flat_init: Vec<i8>,
            updates: Vec<(u8, u8, i8)>,
            r1: u8,
            c1: u8,
            r2: u8,
            c2: u8,
        ) -> TestResult {
            const N: usize = 5;
            const M: usize = 5;

            // Build initial 5×5 grid (pad/truncate flat_init).
            let mut grid: Vec<Vec<i64>> = (0..N)
                .map(|r| {
                    (0..M)
                        .map(|c| flat_init.get(r * M + c).copied().unwrap_or(0) as i64)
                        .collect()
                })
                .collect();

            let mut st = SumTree::from_grid(&grid);

            // Apply up to 8 updates.
            for (ur, uc, val) in updates.iter().take(8) {
                let row = (*ur as usize) % N;
                let col = (*uc as usize) % M;
                let v = *val as i64;
                grid[row][col] = v;
                st.point_update(row, col, v);
            }

            // Build an in-bounds rectangle.
            let r1 = (r1 as usize) % N;
            let c1 = (c1 as usize) % M;
            let r2 = r1 + (r2 as usize) % (N - r1);
            let c2 = c1 + (c2 as usize) % (M - c1);

            let expected = brute_sum(&grid, r1, c1, r2, c2);
            let got = st.range_query(r1, c1, r2, c2);

            if expected == got {
                TestResult::passed()
            } else {
                TestResult::error(format!(
                    "r={r1}..{r2} c={c1}..{c2}: expected {expected} got {got}"
                ))
            }
        }
    }
}
