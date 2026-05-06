//! Two-dimensional prefix sum (immutable summed-area table).
//!
//! Builds a `(rows + 1) × (cols + 1)` table `S` such that
//! `S[i+1][j+1] = sum(grid[r][c]) for r in 0..=i, c in 0..=j`. After the
//! `O(rows * cols)` preprocessing, the sum of any axis-aligned submatrix
//! `grid[r1..=r2][c1..=c2]` is answered in `O(1)` via four lookups:
//! `S[r2+1][c2+1] - S[r1][c2+1] - S[r2+1][c1] + S[r1][c1]`.

/// Immutable 2-D summed-area table over `i64`.
///
/// - Build: `O(rows * cols)`.
/// - Query: `O(1)` for any inclusive submatrix `[r1, r2] × [c1, c2]`.
/// - Space: `O((rows + 1) * (cols + 1))`.
pub struct PrefixSum2D {
    rows: usize,
    cols: usize,
    sat: Vec<i64>,
}

impl PrefixSum2D {
    /// Builds a 2-D prefix-sum table from a row-major grid `grid[row][col]`.
    /// All inner rows must have the same length. An empty grid (zero rows or
    /// zero columns) is allowed and produces a table with `rows == 0` or
    /// `cols == 0`; only the `len`/`is_empty` accessors are meaningful in
    /// that case.
    ///
    /// # Panics
    /// Panics if rows have inconsistent column counts.
    pub fn from_grid(grid: &[Vec<i64>]) -> Self {
        let rows = grid.len();
        let cols = grid.first().map_or(0, Vec::len);
        for row in grid {
            assert!(
                row.len() == cols,
                "PrefixSum2D::from_grid: ragged row (expected {cols} cols, got {})",
                row.len()
            );
        }
        let stride = cols + 1;
        let mut sat = vec![0_i64; (rows + 1) * stride];
        for i in 0..rows {
            let mut row_sum: i64 = 0;
            for j in 0..cols {
                row_sum += grid[i][j];
                sat[(i + 1) * stride + (j + 1)] = sat[i * stride + (j + 1)] + row_sum;
            }
        }
        Self { rows, cols, sat }
    }

    /// Number of rows in the underlying grid.
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Number of columns in the underlying grid.
    pub const fn cols(&self) -> usize {
        self.cols
    }

    /// True if the underlying grid has no cells.
    pub const fn is_empty(&self) -> bool {
        self.rows == 0 || self.cols == 0
    }

    /// Returns the inclusive submatrix sum `grid[r1..=r2][c1..=c2]`.
    ///
    /// - Time: `O(1)`.
    ///
    /// # Panics
    /// Panics if any index is out of bounds, or if `r1 > r2` or `c1 > c2`.
    pub fn range_sum(&self, r1: usize, c1: usize, r2: usize, c2: usize) -> i64 {
        assert!(
            r1 <= r2 && c1 <= c2,
            "PrefixSum2D::range_sum: empty rectangle ({r1},{c1})..=({r2},{c2})"
        );
        assert!(
            r2 < self.rows && c2 < self.cols,
            "PrefixSum2D::range_sum: ({r2},{c2}) out of bounds for {}x{}",
            self.rows,
            self.cols
        );
        let stride = self.cols + 1;
        let s = |i: usize, j: usize| self.sat[i * stride + j];
        s(r2 + 1, c2 + 1) - s(r1, c2 + 1) - s(r2 + 1, c1) + s(r1, c1)
    }
}

#[cfg(test)]
mod tests {
    use super::PrefixSum2D;
    use quickcheck_macros::quickcheck;

    fn brute(grid: &[Vec<i64>], r1: usize, c1: usize, r2: usize, c2: usize) -> i64 {
        let mut s = 0;
        for r in r1..=r2 {
            for c in c1..=c2 {
                s += grid[r][c];
            }
        }
        s
    }

    #[test]
    fn empty_grid() {
        let p = PrefixSum2D::from_grid(&[]);
        assert!(p.is_empty());
        assert_eq!(p.rows(), 0);
        assert_eq!(p.cols(), 0);
    }

    #[test]
    fn single_cell() {
        let g = vec![vec![7_i64]];
        let p = PrefixSum2D::from_grid(&g);
        assert_eq!(p.range_sum(0, 0, 0, 0), 7);
    }

    #[test]
    fn known_grid() {
        let g = vec![
            vec![3, 0, 1, 4, 2],
            vec![5, 6, 3, 2, 1],
            vec![1, 2, 0, 1, 5],
        ];
        let p = PrefixSum2D::from_grid(&g);
        assert_eq!(p.range_sum(0, 0, 2, 4), 36);
        assert_eq!(p.range_sum(1, 1, 2, 3), 14);
        assert_eq!(p.range_sum(0, 0, 0, 0), 3);
        assert_eq!(p.range_sum(2, 4, 2, 4), 5);
    }

    #[test]
    fn negative_entries() {
        let g = vec![vec![-1_i64, 2, -3], vec![4, -5, 6], vec![-7, 8, -9]];
        let p = PrefixSum2D::from_grid(&g);
        assert_eq!(p.range_sum(0, 0, 2, 2), -5);
        assert_eq!(p.range_sum(0, 1, 1, 2), 0);
    }

    #[test]
    #[should_panic(expected = "ragged row")]
    fn ragged_row_panics() {
        let _ = PrefixSum2D::from_grid(&[vec![1, 2, 3], vec![4, 5]]);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn out_of_bounds_panics() {
        let g = vec![vec![1_i64, 2], vec![3, 4]];
        let p = PrefixSum2D::from_grid(&g);
        let _ = p.range_sum(0, 0, 5, 5);
    }

    #[allow(clippy::needless_pass_by_value)]
    #[quickcheck]
    fn matches_brute(rows: u8, cols: u8, fills: Vec<i32>, queries: Vec<u32>) -> bool {
        let rows = (usize::from(rows) % 7) + 1;
        let cols = (usize::from(cols) % 7) + 1;
        let total = rows * cols;
        let mut grid: Vec<Vec<i64>> = vec![vec![0_i64; cols]; rows];
        for (i, v) in fills.iter().take(total).enumerate() {
            grid[i / cols][i % cols] = i64::from(*v);
        }
        let p = PrefixSum2D::from_grid(&grid);
        for chunk in queries.chunks(4) {
            if chunk.len() < 4 {
                break;
            }
            let r1 = (chunk[0] as usize) % rows;
            let r2 = (chunk[1] as usize) % rows;
            let c1 = (chunk[2] as usize) % cols;
            let c2 = (chunk[3] as usize) % cols;
            let (r1, r2) = if r1 <= r2 { (r1, r2) } else { (r2, r1) };
            let (c1, c2) = if c1 <= c2 { (c1, c2) } else { (c2, c1) };
            if p.range_sum(r1, c1, r2, c2) != brute(&grid, r1, c1, r2, c2) {
                return false;
            }
        }
        true
    }
}
