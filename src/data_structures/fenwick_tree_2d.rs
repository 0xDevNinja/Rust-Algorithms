//! 2-D Fenwick tree (2-D Binary Indexed Tree).
//!
//! Supports point updates and prefix/rectangle sum queries on an `n × m` grid.
//!
//! # Complexities
//! | Operation      | Time                  | Space    |
//! |----------------|-----------------------|----------|
//! | `new`          | O(n · m)              | O(n · m) |
//! | `from_grid`    | O(n · m · log n · log m) | O(n · m) |
//! | `point_update` | O(log n · log m)      | O(1)     |
//! | `prefix_sum`   | O(log n · log m)      | O(1)     |
//! | `range_sum`    | O(log n · log m)      | O(1)     |
//!
//! # Preconditions
//! All row/column indices passed to public methods must be in-bounds (asserted).

use std::ops::{Add, AddAssign, Sub};

/// 2-D Binary Indexed Tree over a generic additive type `T`.
///
/// The public API is **0-indexed**. Internally the tree uses 1-based indexing
/// (mirroring the existing 1-D `FenwickTree` in this crate).
///
/// # Type constraints
/// `T` must implement `Default` (acts as additive zero), `Copy`, `AddAssign`,
/// `Add<Output = T>`, and `Sub<Output = T>` (both needed for inclusion-exclusion
/// in `range_sum`).
pub struct FenwickTree2D<T> {
    tree: Vec<Vec<T>>,
    /// number of rows (public dimension)
    n: usize,
    /// number of columns (public dimension)
    m: usize,
}

impl<T: Default + Copy + AddAssign + Add<Output = T> + Sub<Output = T>> FenwickTree2D<T> {
    /// Creates a zero-initialised tree for an `n × m` grid.
    ///
    /// # Panics
    /// Never (no preconditions on `n` or `m`).
    pub fn new(n: usize, m: usize) -> Self {
        Self {
            tree: vec![vec![T::default(); m + 1]; n + 1],
            n,
            m,
        }
    }

    /// Builds the tree from an existing grid in
    /// `O(n · m · log n · log m)` time.
    ///
    /// Each cell of `values` is added via `point_update`, so the resulting
    /// tree is identical to calling `new` then updating every cell.
    ///
    /// # Panics
    /// Never (empty grids are handled; jagged rows are handled gracefully).
    pub fn from_grid(values: &[Vec<T>]) -> Self {
        let n = values.len();
        let m = if n == 0 { 0 } else { values[0].len() };
        let mut ft = Self::new(n, m);
        for (r, row) in values.iter().enumerate() {
            for (c, &v) in row.iter().enumerate() {
                ft.point_update(r, c, v);
            }
        }
        ft
    }

    /// Adds `delta` to the cell at `(row, col)` (0-indexed). O(log n · log m).
    ///
    /// # Panics
    /// Panics if `row >= self.n` or `col >= self.m`.
    pub fn point_update(&mut self, row: usize, col: usize, delta: T) {
        assert!(
            row < self.n,
            "point_update: row {row} is out of bounds (n={})",
            self.n
        );
        assert!(
            col < self.m,
            "point_update: col {col} is out of bounds (m={})",
            self.m
        );
        let mut i = row + 1;
        while i <= self.n {
            let mut j = col + 1;
            while j <= self.m {
                self.tree[i][j] += delta;
                j += j & j.wrapping_neg();
            }
            i += i & i.wrapping_neg();
        }
    }

    /// Returns the sum over the rectangle `[0..=row][0..=col]` (0-indexed,
    /// inclusive). O(log n · log m).
    ///
    /// # Panics
    /// Panics if `row >= self.n` or `col >= self.m`.
    pub fn prefix_sum(&self, row: usize, col: usize) -> T {
        assert!(
            row < self.n,
            "prefix_sum: row {row} is out of bounds (n={})",
            self.n
        );
        assert!(
            col < self.m,
            "prefix_sum: col {col} is out of bounds (m={})",
            self.m
        );
        let mut sum = T::default();
        let mut i = row + 1;
        while i > 0 {
            let mut j = col + 1;
            while j > 0 {
                sum += self.tree[i][j];
                j -= j & j.wrapping_neg();
            }
            i -= i & i.wrapping_neg();
        }
        sum
    }

    /// Returns the sum over the inclusive rectangle
    /// `[r1..=r2][c1..=c2]` (0-indexed) via inclusion-exclusion of four
    /// prefix sums. O(log n · log m).
    ///
    /// # Panics
    /// Panics if any index is out of bounds or if `r1 > r2` / `c1 > c2`.
    pub fn range_sum(&self, r1: usize, c1: usize, r2: usize, c2: usize) -> T {
        assert!(r1 <= r2, "range_sum: r1={r1} > r2={r2}");
        assert!(c1 <= c2, "range_sum: c1={c1} > c2={c2}");
        assert!(
            r2 < self.n,
            "range_sum: r2={r2} is out of bounds (n={})",
            self.n
        );
        assert!(
            c2 < self.m,
            "range_sum: c2={c2} is out of bounds (m={})",
            self.m
        );

        // Inclusion-exclusion:
        //   sum(r1..=r2, c1..=c2)
        //   = P(r2, c2) - P(r1-1, c2) - P(r2, c1-1) + P(r1-1, c1-1)
        // where terms with negative indices are zero.
        let p = |r: usize, c: usize| -> T { self.prefix_sum(r, c) };

        let total = p(r2, c2);
        let sub_r = if r1 > 0 { p(r1 - 1, c2) } else { T::default() };
        let sub_c = if c1 > 0 { p(r2, c1 - 1) } else { T::default() };
        let add_corner = if r1 > 0 && c1 > 0 {
            p(r1 - 1, c1 - 1)
        } else {
            T::default()
        };

        total - sub_r - sub_c + add_corner
    }

    /// Returns `(n, m)` — the number of rows and columns.
    pub const fn dims(&self) -> (usize, usize) {
        (self.n, self.m)
    }
}

#[cfg(test)]
mod tests {
    use super::FenwickTree2D;
    use quickcheck::TestResult;
    use quickcheck_macros::quickcheck;

    // -----------------------------------------------------------------------
    // Unit tests
    // -----------------------------------------------------------------------

    #[test]
    fn new_1x1_is_zero() {
        let ft: FenwickTree2D<i32> = FenwickTree2D::new(1, 1);
        assert_eq!(ft.prefix_sum(0, 0), 0);
        assert_eq!(ft.dims(), (1, 1));
    }

    #[test]
    fn new_0x0_dims() {
        let ft: FenwickTree2D<i32> = FenwickTree2D::new(0, 0);
        assert_eq!(ft.dims(), (0, 0));
    }

    #[test]
    fn single_update_then_prefix_sum() {
        let mut ft: FenwickTree2D<i64> = FenwickTree2D::new(4, 4);
        ft.point_update(2, 3, 7);
        assert_eq!(ft.prefix_sum(2, 3), 7);
        // Prefix sums that do not include (2,3) must still be 0.
        assert_eq!(ft.prefix_sum(1, 3), 0);
        assert_eq!(ft.prefix_sum(2, 2), 0);
    }

    #[test]
    fn build_from_grid_spot_checks() {
        // 3×3 grid:
        //  1 2 3
        //  4 5 6
        //  7 8 9
        let grid = vec![vec![1i32, 2, 3], vec![4, 5, 6], vec![7, 8, 9]];
        let ft = FenwickTree2D::from_grid(&grid);

        // prefix_sum(0,0) == 1
        assert_eq!(ft.prefix_sum(0, 0), 1);
        // prefix_sum(1,1) == 1+2+4+5 = 12
        assert_eq!(ft.prefix_sum(1, 1), 12);
        // prefix_sum(2,2) == total = 45
        assert_eq!(ft.prefix_sum(2, 2), 45);
    }

    #[test]
    fn range_sum_full_grid_equals_total() {
        let grid = vec![vec![1i32, 2, 3], vec![4, 5, 6], vec![7, 8, 9]];
        let ft = FenwickTree2D::from_grid(&grid);
        assert_eq!(ft.range_sum(0, 0, 2, 2), 45);
    }

    #[test]
    fn range_sum_single_cell() {
        let grid = vec![vec![1i32, 2, 3], vec![4, 5, 6], vec![7, 8, 9]];
        let ft = FenwickTree2D::from_grid(&grid);
        for r in 0..3usize {
            for c in 0..3usize {
                assert_eq!(
                    ft.range_sum(r, c, r, c),
                    grid[r][c],
                    "single-cell range_sum failed at ({r},{c})"
                );
            }
        }
    }

    #[test]
    fn range_sum_sub_rectangle_hand_computed() {
        // Grid:
        //  1  2  3  4
        //  5  6  7  8
        //  9 10 11 12
        // range_sum(1,1, 2,2) = 6+7+10+11 = 34
        let grid = vec![vec![1i32, 2, 3, 4], vec![5, 6, 7, 8], vec![9, 10, 11, 12]];
        let ft = FenwickTree2D::from_grid(&grid);
        assert_eq!(ft.range_sum(1, 1, 2, 2), 34);
        // range_sum(0,0, 1,3) = 1+2+3+4+5+6+7+8 = 36
        assert_eq!(ft.range_sum(0, 0, 1, 3), 36);
    }

    #[test]
    fn point_updates_with_negative_deltas() {
        let mut ft: FenwickTree2D<i64> = FenwickTree2D::new(3, 3);
        ft.point_update(1, 1, 10);
        ft.point_update(1, 1, -4);
        assert_eq!(ft.prefix_sum(1, 1), 6);
        assert_eq!(ft.range_sum(1, 1, 1, 1), 6);
    }

    #[test]
    fn i64_with_negatives() {
        let grid = vec![vec![-5i64, 3], vec![7, -2]];
        let ft = FenwickTree2D::from_grid(&grid);
        // total = -5+3+7-2 = 3
        assert_eq!(ft.prefix_sum(1, 1), 3);
        // range_sum(0,0,0,1) = -5+3 = -2
        assert_eq!(ft.range_sum(0, 0, 0, 1), -2);
        // range_sum(1,0,1,1) = 7-2 = 5
        assert_eq!(ft.range_sum(1, 0, 1, 1), 5);
    }

    #[test]
    fn multiple_updates_accumulate() {
        let mut ft: FenwickTree2D<i32> = FenwickTree2D::new(2, 2);
        ft.point_update(0, 0, 1);
        ft.point_update(0, 0, 2);
        ft.point_update(1, 1, 5);
        // prefix_sum(1,1) = 1+2+5 = 8
        assert_eq!(ft.prefix_sum(1, 1), 8);
        // range_sum over top-left cell only = 3
        assert_eq!(ft.range_sum(0, 0, 0, 0), 3);
    }

    // -----------------------------------------------------------------------
    // Property-based test
    // -----------------------------------------------------------------------

    /// Brute-force rectangle sum for validation.
    fn brute_rect(grid: &[Vec<i32>], r1: usize, c1: usize, r2: usize, c2: usize) -> i64 {
        let mut s: i64 = 0;
        for row in grid.iter().take(r2 + 1).skip(r1) {
            for &v in row.iter().take(c2 + 1).skip(c1) {
                s += i64::from(v);
            }
        }
        s
    }

    #[quickcheck]
    #[allow(clippy::needless_pass_by_value)]
    fn prop_matches_brute_force(
        // 25 i8 values → flat 5×5 grid (values small enough to avoid overflow)
        flat: Vec<i8>,
        // Operations: (row, col, delta) for updates; (r1,c1,r2,c2) queries
        updates: Vec<(u8, u8, i8)>,
        queries: Vec<(u8, u8, u8, u8)>,
    ) -> TestResult {
        const N: usize = 5;
        const M: usize = 5;

        if flat.len() != N * M {
            return TestResult::discard();
        }

        // Build a 5×5 reference grid and the Fenwick tree.
        let mut ref_grid: Vec<Vec<i32>> = (0..N)
            .map(|r| (0..M).map(|c| i32::from(flat[r * M + c])).collect())
            .collect();

        let mut ft: FenwickTree2D<i64> = FenwickTree2D::new(N, M);
        for r in 0..N {
            for c in 0..M {
                ft.point_update(r, c, i64::from(ref_grid[r][c]));
            }
        }

        // Apply updates to both reference and Fenwick tree.
        for &(r, c, d) in &updates {
            let r = (r as usize) % N;
            let c = (c as usize) % M;
            let d = i32::from(d);
            ref_grid[r][c] += d;
            ft.point_update(r, c, i64::from(d));
        }

        // Check every query.
        for &(r1, c1, r2, c2) in &queries {
            let r1 = (r1 as usize) % N;
            let c1 = (c1 as usize) % M;
            let r2 = (r2 as usize) % N;
            let c2 = (c2 as usize) % M;
            let (r1, r2) = if r1 <= r2 { (r1, r2) } else { (r2, r1) };
            let (c1, c2) = if c1 <= c2 { (c1, c2) } else { (c2, c1) };

            let expected = brute_rect(&ref_grid, r1, c1, r2, c2);
            let got = ft.range_sum(r1, c1, r2, c2);
            if expected != got {
                return TestResult::failed();
            }
        }

        TestResult::passed()
    }
}
