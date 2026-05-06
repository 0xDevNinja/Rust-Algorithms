//! Broken-profile (a.k.a. "profile DP") for tiling problems on n×m grids.
//!
//! The canonical problem solved here is: count the number of ways to tile an
//! n×m grid with 1×2 dominoes.
//!
//! # Algorithm
//!
//! The DP processes cells in row-major order. At each cell `(i, j)` we
//! maintain a bitmask of which cells in the **current row** are already covered
//! by a vertical domino placed in the previous row. The "broken" profile refers
//! to the fact that at any given moment the frontier cuts through the middle of
//! the current row, so the profile looks jagged.
//!
//! State: `dp[mask]` = number of ways to fill all cells before `(i, j)` such
//! that `mask` describes which cells at column index `j` onward in the current
//! row are pre-filled by vertical dominoes reaching down from row `i − 1`.
//!
//! At each cell the transition is:
//! * **Already covered** (bit `j` is set in mask): advance to the next cell
//!   with the bit cleared.
//! * **Place a vertical domino** (if `i + 1 < n`): covers `(i, j)` and
//!   `(i + 1, j)`; the cell in the next row is marked in the new mask.
//! * **Place a horizontal domino** (if `j + 1 < m` and bit `j + 1` is not set):
//!   covers `(i, j)` and `(i, j + 1)`; both bits are consumed / not emitted.
//!
//! # Complexity
//!
//! * Time: O(n · m · 2^m)
//! * Space: O(2^m)
//!
//! # Preconditions
//!
//! * `cols ≤ 20` (enforced at runtime; `1 << 20` = 1 048 576 states).
//! * The smaller dimension is always used as `cols` internally.

/// Counts the number of ways to tile an `rows × cols` grid with 1×2 dominoes.
///
/// The function swaps dimensions internally so the bitmask always spans the
/// smaller axis, keeping `1 << cols` manageable (cols ≤ 20 after the swap).
///
/// Returns 0 when `rows * cols` is odd (impossible parity) or when either
/// dimension is 0 (except the 0×0 case, which returns 1 by convention — the
/// empty tiling).
///
/// # Panics
///
/// Panics if `min(rows, cols) > 20`.
pub fn count_domino_tilings(rows: usize, cols: usize) -> u64 {
    // Canonical 0×0 edge case.
    if rows == 0 && cols == 0 {
        return 1;
    }
    // A grid with a zero dimension (but not both zero) has no cells; one empty
    // tiling exists only for the truly empty 0×0 grid handled above.
    if rows == 0 || cols == 0 {
        return 1;
    }
    // Odd total cells → impossible.
    if (rows * cols) % 2 == 1 {
        return 0;
    }

    // Use the smaller dimension as the bitmask width so 2^m stays small.
    let (n, m) = if rows <= cols {
        (cols, rows)
    } else {
        (rows, cols)
    };

    assert!(m <= 20, "profile DP requires min(rows, cols) ≤ 20");

    let states = 1_usize << m;
    // dp[mask] = number of partial tilings where `mask` encodes which cells in
    // the current column (of the n×m view) are pre-filled from a vertical
    // domino placed in the previous column.
    let mut dp = vec![0_u64; states];
    dp[0] = 1;

    // Process each cell in column-major order (iterate columns first, then
    // rows within each column).  We think of the grid as n rows × m cols, but
    // iterate column by column so the bitmask tracks which rows in the *next*
    // column are already occupied.
    for _col in 0..n {
        // Process all m rows within this column.
        for row in 0..m {
            let mut next_dp = vec![0_u64; states];
            for mask in 0..states {
                let ways = dp[mask];
                if ways == 0 {
                    continue;
                }
                let occupied = (mask >> row) & 1 == 1;
                if occupied {
                    // This cell is already covered by a vertical domino from
                    // the previous column; clear the bit and move on.
                    let new_mask = mask & !(1 << row);
                    next_dp[new_mask] = next_dp[new_mask].saturating_add(ways);
                } else {
                    // Option 1: place a vertical domino covering (row) and
                    // (row+1) in the *same* column — i.e., current row and
                    // next row.  Mark next row as occupied.
                    if row + 1 < m && (mask >> (row + 1)) & 1 == 0 {
                        let new_mask = mask | (1 << (row + 1));
                        next_dp[new_mask] = next_dp[new_mask].saturating_add(ways);
                    }
                    // Option 2: place a horizontal domino that extends into
                    // the next column.  Mark this row in the next column's
                    // mask (i.e., set bit `row` so it will be "pre-filled"
                    // when we process the next column).
                    // The current cell and its horizontal partner in the next
                    // column are both consumed; the partner is recorded by
                    // keeping bit `row` set in the outgoing mask.
                    let new_mask = mask | (1 << row);
                    next_dp[new_mask] = next_dp[new_mask].saturating_add(ways);
                }
            }
            dp = next_dp;
        }
    }

    // After processing all n columns × m rows, only the all-zero mask (no
    // dangling horizontal dominoes) is a valid complete tiling.
    dp[0]
}

#[cfg(test)]
mod tests {
    use super::count_domino_tilings;

    #[test]
    fn empty_grid() {
        assert_eq!(count_domino_tilings(0, 0), 1);
    }

    #[test]
    fn one_by_one() {
        assert_eq!(count_domino_tilings(1, 1), 0);
    }

    #[test]
    fn two_by_two() {
        assert_eq!(count_domino_tilings(2, 2), 2);
    }

    #[test]
    fn two_by_three() {
        assert_eq!(count_domino_tilings(2, 3), 3);
    }

    #[test]
    fn four_by_four() {
        assert_eq!(count_domino_tilings(4, 4), 36);
    }

    #[test]
    fn eight_by_eight() {
        assert_eq!(count_domino_tilings(8, 8), 12_988_816);
    }

    #[test]
    fn symmetry() {
        // Tiling count is symmetric in dimensions.
        for r in 1_usize..=6 {
            for c in 1_usize..=6 {
                assert_eq!(
                    count_domino_tilings(r, c),
                    count_domino_tilings(c, r),
                    "symmetry failed for {r}×{c}"
                );
            }
        }
    }

    /// Known Fibonacci values for tilings(2, n): the count equals fib(n+1)
    /// where fib is 1-indexed with fib(1)=1, fib(2)=1, fib(3)=2, fib(4)=3, ...
    ///
    /// tilings(2,1)=1=fib(2), tilings(2,2)=2=fib(3), tilings(2,3)=3=fib(4), ...
    #[test]
    fn two_by_n_fibonacci() {
        // 1-indexed Fibonacci: fib[1]=1, fib[2]=1, fib[3]=2, ..., fib[11]=89
        let fib: [u64; 12] = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89];
        for n in 1_usize..=10 {
            // tilings(2, n) = fib(n+1)
            assert_eq!(
                count_domino_tilings(2, n),
                fib[n + 1],
                "tilings(2,{n}) should equal fib({})",
                n + 1
            );
        }
    }
}
