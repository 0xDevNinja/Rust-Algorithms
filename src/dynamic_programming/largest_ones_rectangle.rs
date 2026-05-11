//! Largest rectangle of `1`s in a binary matrix.
//!
//! Given an `m x n` grid whose cells are `0` or `1`, find the area of the
//! largest axis-aligned rectangle that contains only `1`s.
//!
//! ## Approach
//!
//! Walk the grid row by row, maintaining a histogram where the bar at column
//! `c` is the count of consecutive `1`s ending at the current row in that
//! column.  Whenever the current cell is `0` the bar resets to `0`.  After
//! updating the histogram for a row, the answer for rectangles whose bottom
//! edge lies on this row is the **largest rectangle in the histogram**, which
//! we compute in linear time with a monotonic stack.
//!
//! The overall answer is the maximum across all rows.
//!
//! ## Complexity
//!
//! For an `m x n` grid:
//!
//! * Time:  **O(m · n)** — each cell is pushed and popped from the stack at
//!   most once per row.
//! * Space: **O(n)**     — one histogram and one stack of width `n`.
//!
//! ## Panics
//!
//! Panics if the grid is non-rectangular (rows of differing length) or if any
//! cell holds a value other than `0` or `1`.

/// Largest-rectangle-in-histogram via monotonic stack.
///
/// Returns the maximum area of a rectangle whose top is bounded by the bar
/// heights in `heights`.  Runs in O(n) time.
fn largest_rectangle_in_histogram(heights: &[u64]) -> u64 {
    let n = heights.len();
    let mut stack: Vec<usize> = Vec::with_capacity(n + 1);
    let mut best: u64 = 0;

    // Iterate one past the end with a sentinel height of 0 to flush the stack.
    for i in 0..=n {
        let h = if i == n { 0 } else { heights[i] };
        while let Some(&top) = stack.last() {
            if heights[top] <= h {
                break;
            }
            stack.pop();
            let height = heights[top];
            let width = match stack.last() {
                Some(&prev) => (i - prev - 1) as u64,
                None => i as u64,
            };
            let area = height * width;
            if area > best {
                best = area;
            }
        }
        stack.push(i);
    }

    best
}

/// Returns the area of the largest rectangle of `1`s in `grid`.
///
/// `grid` is treated as a binary matrix; each cell must be `0` or `1`.
/// Returns `0` for an empty grid (no rows or zero-width rows).
///
/// # Panics
///
/// * If rows have differing lengths (non-rectangular grid).
/// * If any cell contains a value other than `0` or `1`.
///
/// # Examples
///
/// ```
/// use rust_algorithms::dynamic_programming::largest_ones_rectangle::largest_ones_rectangle;
///
/// let grid = vec![
///     vec![1, 0, 1, 0, 0],
///     vec![1, 0, 1, 1, 1],
///     vec![1, 1, 1, 1, 1],
///     vec![1, 0, 0, 1, 0],
/// ];
/// assert_eq!(largest_ones_rectangle(&grid), 6);
/// ```
pub fn largest_ones_rectangle(grid: &[Vec<u8>]) -> u64 {
    if grid.is_empty() {
        return 0;
    }
    let cols = grid[0].len();
    if cols == 0 {
        return 0;
    }

    let mut heights = vec![0u64; cols];
    let mut best: u64 = 0;

    for row in grid {
        assert!(
            row.len() == cols,
            "largest_ones_rectangle: non-rectangular grid"
        );
        for (c, &cell) in row.iter().enumerate() {
            match cell {
                0 => heights[c] = 0,
                1 => heights[c] += 1,
                other => panic!("largest_ones_rectangle: cell value {other} is not 0 or 1"),
            }
        }
        let row_best = largest_rectangle_in_histogram(&heights);
        if row_best > best {
            best = row_best;
        }
    }

    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_grid() {
        let grid: Vec<Vec<u8>> = Vec::new();
        assert_eq!(largest_ones_rectangle(&grid), 0);
    }

    #[test]
    fn empty_rows() {
        let grid: Vec<Vec<u8>> = vec![vec![], vec![]];
        assert_eq!(largest_ones_rectangle(&grid), 0);
    }

    #[test]
    fn all_zeros() {
        let grid = vec![vec![0u8; 4]; 3];
        assert_eq!(largest_ones_rectangle(&grid), 0);
    }

    #[test]
    fn all_ones() {
        let rows = 5usize;
        let cols = 7usize;
        let grid = vec![vec![1u8; cols]; rows];
        assert_eq!(largest_ones_rectangle(&grid), (rows as u64) * (cols as u64));
    }

    #[test]
    fn classic_mixed_grid() {
        // Standard LeetCode "maximal rectangle" example — expected area 6.
        let grid = vec![
            vec![1, 0, 1, 0, 0],
            vec![1, 0, 1, 1, 1],
            vec![1, 1, 1, 1, 1],
            vec![1, 0, 0, 1, 0],
        ];
        assert_eq!(largest_ones_rectangle(&grid), 6);
    }

    #[test]
    fn single_row() {
        let grid = vec![vec![0, 1, 1, 1, 0, 1, 1]];
        // Longest run of 1s has length 3.
        assert_eq!(largest_ones_rectangle(&grid), 3);
    }

    #[test]
    fn single_column() {
        let grid = vec![
            vec![1u8],
            vec![1u8],
            vec![0u8],
            vec![1u8],
            vec![1u8],
            vec![1u8],
            vec![0u8],
        ];
        // Longest vertical run is 3.
        assert_eq!(largest_ones_rectangle(&grid), 3);
    }

    #[test]
    fn single_cell_one() {
        assert_eq!(largest_ones_rectangle(&[vec![1u8]]), 1);
    }

    #[test]
    fn single_cell_zero() {
        assert_eq!(largest_ones_rectangle(&[vec![0u8]]), 0);
    }

    #[test]
    #[should_panic(expected = "non-rectangular grid")]
    fn panics_on_non_rectangular() {
        let grid = vec![vec![1u8, 0, 1], vec![1u8, 1]];
        let _ = largest_ones_rectangle(&grid);
    }

    #[test]
    #[should_panic(expected = "is not 0 or 1")]
    fn panics_on_bad_value() {
        let grid = vec![vec![1u8, 2, 0]];
        let _ = largest_ones_rectangle(&grid);
    }

    #[test]
    fn histogram_helper_basic() {
        // Classic histogram example: [2,1,5,6,2,3] -> 10.
        assert_eq!(largest_rectangle_in_histogram(&[2, 1, 5, 6, 2, 3]), 10);
        assert_eq!(largest_rectangle_in_histogram(&[]), 0);
        assert_eq!(largest_rectangle_in_histogram(&[0, 0, 0]), 0);
        assert_eq!(largest_rectangle_in_histogram(&[4]), 4);
    }
}
