//! Largest all-ones square in a 0/1 grid via dynamic programming. O(m·n) time
//! and O(n) extra space.
//!
//! For each cell `(i, j)` containing a `1`, the side length of the largest
//! all-ones square whose bottom-right corner is `(i, j)` satisfies
//!
//! ```text
//! dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
//! ```
//!
//! and is `0` whenever `grid[i][j] == 0`. The answer is the maximum value of
//! `dp` over the whole grid; the area of the largest all-ones square is the
//! square of that side length.

/// Returns the side length of the largest square sub-grid whose cells are all
/// `1`s. Returns `0` when the grid is empty or contains no `1`s.
///
/// # Panics
///
/// Panics if `grid` is ragged (rows of differing lengths) or if any cell holds
/// a value other than `0` or `1`.
pub fn largest_ones_square_side(grid: &[Vec<u8>]) -> usize {
    let rows = grid.len();
    if rows == 0 {
        return 0;
    }
    let cols = grid[0].len();
    if cols == 0 {
        // All rows must agree, even when the first row is empty.
        for row in grid.iter().skip(1) {
            assert_eq!(row.len(), 0, "ragged grid: rows have differing lengths");
        }
        return 0;
    }

    // Rolling 1-D DP: `prev[j]` holds dp[i-1][j], `curr[j]` holds dp[i][j].
    let mut prev = vec![0_usize; cols];
    let mut curr = vec![0_usize; cols];
    let mut best = 0_usize;

    for (i, row) in grid.iter().enumerate() {
        assert_eq!(row.len(), cols, "ragged grid: rows have differing lengths");
        for (j, &cell) in row.iter().enumerate() {
            assert!(cell <= 1, "grid cells must be 0 or 1, got {cell}");
            curr[j] = if cell == 0 {
                0
            } else if i == 0 || j == 0 {
                1
            } else {
                let up = prev[j];
                let left = curr[j - 1];
                let diag = prev[j - 1];
                up.min(left).min(diag) + 1
            };
            if curr[j] > best {
                best = curr[j];
            }
        }
        std::mem::swap(&mut prev, &mut curr);
        // `curr` is now the old `prev` row; reset it for the next iteration so
        // stale values cannot leak in (only matters if we ever read it before
        // overwriting, but keeps the invariant clear).
        curr.fill(0);
    }

    best
}

#[cfg(test)]
mod tests {
    use super::largest_ones_square_side;

    #[test]
    fn empty_grid() {
        let g: Vec<Vec<u8>> = vec![];
        assert_eq!(largest_ones_square_side(&g), 0);
    }

    #[test]
    fn empty_rows() {
        let g: Vec<Vec<u8>> = vec![vec![], vec![], vec![]];
        assert_eq!(largest_ones_square_side(&g), 0);
    }

    #[test]
    fn all_zeros() {
        let g = vec![vec![0_u8; 4]; 3];
        assert_eq!(largest_ones_square_side(&g), 0);
    }

    #[test]
    fn all_ones_returns_min_dimension() {
        let g = vec![vec![1_u8; 5]; 3];
        assert_eq!(largest_ones_square_side(&g), 3);

        let g = vec![vec![1_u8; 2]; 7];
        assert_eq!(largest_ones_square_side(&g), 2);
    }

    #[test]
    fn classic_clrs_like_example() {
        // CLRS-style 4x5 example; largest all-ones square has side 2.
        let g = vec![
            vec![1, 0, 1, 0, 0],
            vec![1, 0, 1, 1, 1],
            vec![1, 1, 1, 1, 1],
            vec![1, 0, 0, 1, 0],
        ];
        assert_eq!(largest_ones_square_side(&g), 2);

        // Bigger example with a clear 3x3 block of ones.
        let g = vec![
            vec![0, 1, 1, 1, 0],
            vec![1, 1, 1, 1, 0],
            vec![0, 1, 1, 1, 1],
            vec![0, 1, 1, 1, 1],
            vec![0, 0, 0, 0, 0],
        ];
        assert_eq!(largest_ones_square_side(&g), 3);
    }

    #[test]
    fn single_row() {
        let g = vec![vec![0, 1, 1, 1, 0, 1]];
        assert_eq!(largest_ones_square_side(&g), 1);
    }

    #[test]
    fn single_column() {
        let g = vec![vec![1], vec![1], vec![0], vec![1]];
        assert_eq!(largest_ones_square_side(&g), 1);
    }

    #[test]
    fn area_is_side_squared() {
        let g = vec![vec![1_u8; 4]; 4];
        let side = largest_ones_square_side(&g);
        assert_eq!(side, 4);
        assert_eq!(side * side, 16);
    }

    #[test]
    #[should_panic(expected = "ragged grid")]
    fn ragged_input_panics() {
        let g = vec![vec![1, 1, 1], vec![1, 1], vec![1, 1, 1]];
        let _ = largest_ones_square_side(&g);
    }

    #[test]
    #[should_panic(expected = "grid cells must be 0 or 1")]
    fn non_binary_value_panics() {
        let g = vec![vec![0, 1, 2], vec![1, 1, 1]];
        let _ = largest_ones_square_side(&g);
    }
}
