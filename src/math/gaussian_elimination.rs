//! Gaussian elimination over `f64` with partial pivoting.
//!
//! Solves a linear system `A x = b` by reducing the augmented matrix
//! `[A | b]` to row-echelon form, swapping in the row with the
//! largest-magnitude pivot at each step for numerical stability, then
//! back-substituting. Singular systems are classified by inspecting any
//! all-zero pivot rows: a non-zero right-hand side on such a row signals
//! an inconsistent system, otherwise the system has infinitely many
//! solutions and the rank plus the column index of every pivot is
//! reported.
//!
//! Complexity: `O(n^3)` time and `O(n^2)` extra space for the working
//! copy of the augmented matrix on an `n x n` system.

/// Tolerance used when comparing floating-point values to zero.
pub const EPS: f64 = 1e-9;

/// Outcome of solving a linear system `A x = b`.
#[derive(Debug, Clone, PartialEq)]
pub enum GaussResult {
    /// The system has a single solution `x`.
    Unique(Vec<f64>),
    /// The system is consistent but under-determined. `rank` is the
    /// rank of `A` and `pivots` lists the column index of each pivot.
    Infinite { rank: usize, pivots: Vec<usize> },
    /// The system has no solution.
    Inconsistent,
}

/// Solves `A x = b` for `x` using Gaussian elimination with partial
/// pivoting. The matrix `A` must be square and `rhs` must have the same
/// length as the number of rows of `A`; otherwise `Inconsistent` is
/// returned.
pub fn solve(matrix: &[Vec<f64>], rhs: &[f64]) -> GaussResult {
    let n = matrix.len();
    if n == 0 {
        return GaussResult::Unique(Vec::new());
    }
    if rhs.len() != n || matrix.iter().any(|row| row.len() != n) {
        return GaussResult::Inconsistent;
    }

    // Build the augmented matrix `[A | b]` as a fresh working copy.
    let mut aug: Vec<Vec<f64>> = matrix
        .iter()
        .zip(rhs.iter())
        .map(|(row, &b)| {
            let mut r = row.clone();
            r.push(b);
            r
        })
        .collect();

    solve_in_place(&mut aug)
}

/// In-place variant. `aug` must be an `n` by `n + 1` augmented matrix
/// `[A | b]`; on return `aug` is in row-echelon form. Returns
/// `Inconsistent` if `aug` is malformed.
pub fn solve_in_place(aug: &mut [Vec<f64>]) -> GaussResult {
    let n = aug.len();
    if n == 0 {
        return GaussResult::Unique(Vec::new());
    }
    if aug.iter().any(|row| row.len() != n + 1) {
        return GaussResult::Inconsistent;
    }

    let mut pivots: Vec<usize> = Vec::with_capacity(n);
    let mut row = 0_usize;

    for col in 0..n {
        // Partial pivot: find the row at or below `row` with the
        // largest |aug[r][col]| and swap it up.
        let mut best_row = row;
        let mut best_abs = aug[row][col].abs();
        for r in (row + 1)..n {
            let v = aug[r][col].abs();
            if v > best_abs {
                best_abs = v;
                best_row = r;
            }
        }
        if best_abs <= EPS {
            // Column has no usable pivot — it represents a free
            // variable. Move on without advancing `row`.
            continue;
        }
        aug.swap(row, best_row);
        pivots.push(col);

        // Eliminate column `col` from every other row.
        let pivot = aug[row][col];
        for r in 0..n {
            if r == row {
                continue;
            }
            let factor = aug[r][col] / pivot;
            if factor == 0.0 {
                continue;
            }
            for c in col..=n {
                aug[r][c] -= factor * aug[row][c];
            }
        }
        row += 1;
    }

    let rank = pivots.len();

    // Inconsistency check: any row whose A-part is all zero but whose
    // rhs is non-zero is `0 = c` with `c != 0`.
    for r in rank..n {
        if aug[r][n].abs() > EPS {
            return GaussResult::Inconsistent;
        }
    }

    if rank < n {
        return GaussResult::Infinite { rank, pivots };
    }

    // Unique solution: each pivot row is now `pivots[i] * x[i] = rhs`.
    let mut x = vec![0.0_f64; n];
    for i in 0..n {
        let col = pivots[i];
        x[col] = aug[i][n] / aug[i][col];
    }
    GaussResult::Unique(x)
}

#[cfg(test)]
mod tests {
    use super::{solve, solve_in_place, GaussResult, EPS};
    use quickcheck_macros::quickcheck;

    fn approx_eq_vec(a: &[f64], b: &[f64], eps: f64) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() <= eps)
    }

    fn matvec(matrix: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
        matrix
            .iter()
            .map(|row| row.iter().zip(x).map(|(a, b)| a * b).sum())
            .collect()
    }

    #[test]
    fn unique_3x3_system() {
        // x + y + z = 6
        // 2x + 5y + z = 15
        // 2x + 3y + 8z = 32
        // Solution: x = 1, y = 2, z = 3.
        let a = vec![
            vec![1.0, 1.0, 1.0],
            vec![2.0, 5.0, 1.0],
            vec![2.0, 3.0, 8.0],
        ];
        let b = vec![6.0, 15.0, 32.0];
        let GaussResult::Unique(x) = solve(&a, &b) else {
            panic!("expected unique");
        };
        assert!(approx_eq_vec(&x, &[1.0, 2.0, 3.0], 1e-9));
    }

    #[test]
    fn unique_2x2_system() {
        // 2x + y = 5, x + 3y = 10  =>  x = 1, y = 3.
        let a = vec![vec![2.0, 1.0], vec![1.0, 3.0]];
        let b = vec![5.0, 10.0];
        let GaussResult::Unique(x) = solve(&a, &b) else {
            panic!("expected unique");
        };
        assert!(approx_eq_vec(&x, &[1.0, 3.0], 1e-9));
    }

    #[test]
    fn identity_matrix_returns_rhs() {
        let a = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
        ];
        let b = vec![3.5, -2.0, 7.25, 0.0];
        let GaussResult::Unique(x) = solve(&a, &b) else {
            panic!("expected unique");
        };
        assert!(approx_eq_vec(&x, &b, EPS));
    }

    #[test]
    fn singular_consistent_system() {
        // Row 3 = row 1 + row 2 on both sides.
        // x + y + z = 1
        // x - y + z = 3
        // 2x + 0y + 2z = 4   (consistent dependent equation)
        let a = vec![
            vec![1.0, 1.0, 1.0],
            vec![1.0, -1.0, 1.0],
            vec![2.0, 0.0, 2.0],
        ];
        let b = vec![1.0, 3.0, 4.0];
        match solve(&a, &b) {
            GaussResult::Infinite { rank, pivots } => {
                assert_eq!(rank, 2);
                assert_eq!(pivots.len(), 2);
            }
            other => panic!("expected Infinite, got {other:?}"),
        }
    }

    #[test]
    fn singular_inconsistent_system() {
        // Same lhs as above, rhs no longer consistent (row 3 != row 1 + row 2).
        let a = vec![
            vec![1.0, 1.0, 1.0],
            vec![1.0, -1.0, 1.0],
            vec![2.0, 0.0, 2.0],
        ];
        let b = vec![1.0, 3.0, 5.0];
        assert!(matches!(solve(&a, &b), GaussResult::Inconsistent));
    }

    #[test]
    fn zero_matrix_zero_rhs_is_infinite() {
        let a = vec![vec![0.0, 0.0], vec![0.0, 0.0]];
        let b = vec![0.0, 0.0];
        match solve(&a, &b) {
            GaussResult::Infinite { rank, pivots } => {
                assert_eq!(rank, 0);
                assert!(pivots.is_empty());
            }
            other => panic!("expected Infinite, got {other:?}"),
        }
    }

    #[test]
    fn zero_matrix_nonzero_rhs_is_inconsistent() {
        let a = vec![vec![0.0, 0.0], vec![0.0, 0.0]];
        let b = vec![0.0, 1.0];
        assert!(matches!(solve(&a, &b), GaussResult::Inconsistent));
    }

    #[test]
    fn empty_system_is_unique_empty() {
        let a: Vec<Vec<f64>> = Vec::new();
        let b: Vec<f64> = Vec::new();
        match solve(&a, &b) {
            GaussResult::Unique(x) => assert!(x.is_empty()),
            other => panic!("expected Unique, got {other:?}"),
        }
    }

    #[test]
    fn dimension_mismatch_is_inconsistent() {
        let a = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let b = vec![1.0]; // wrong length
        assert!(matches!(solve(&a, &b), GaussResult::Inconsistent));
    }

    #[test]
    fn in_place_matches_solve() {
        let a = vec![
            vec![3.0, 2.0, -1.0],
            vec![2.0, -2.0, 4.0],
            vec![-1.0, 0.5, -1.0],
        ];
        let b = vec![1.0, -2.0, 0.0];
        let mut aug: Vec<Vec<f64>> = a
            .iter()
            .zip(&b)
            .map(|(row, &v)| {
                let mut r = row.clone();
                r.push(v);
                r
            })
            .collect();
        let GaussResult::Unique(x_in_place) = solve_in_place(&mut aug) else {
            panic!("expected unique");
        };
        let GaussResult::Unique(x_solve) = solve(&a, &b) else {
            panic!("expected unique");
        };
        assert!(approx_eq_vec(&x_in_place, &x_solve, 1e-9));
        // Sanity: A * x ~ b.
        let bb = matvec(&a, &x_solve);
        assert!(approx_eq_vec(&bb, &b, 1e-9));
    }

    // Property test: build a deliberately diagonally-dominant matrix
    // (which is non-singular) from random integer entries, pick a
    // random integer x, set b = A * x, and verify the solver recovers
    // x within tolerance.
    #[allow(clippy::needless_pass_by_value)]
    #[quickcheck]
    fn quickcheck_random_nonsingular(entries: Vec<i8>, x_seed: Vec<i8>) -> bool {
        let n = (entries.len() / 4).clamp(1, 5);
        if x_seed.len() < n || entries.len() < n * n {
            return true;
        }
        let mut a: Vec<Vec<f64>> = (0..n)
            .map(|i| (0..n).map(|j| f64::from(entries[i * n + j])).collect())
            .collect();
        // Make A diagonally dominant: a[i][i] = sum_{j!=i} |a[i][j]| + 1.
        for i in 0..n {
            let off: f64 = (0..n).filter(|&j| j != i).map(|j| a[i][j].abs()).sum();
            a[i][i] = off + 1.0;
        }
        let x_true: Vec<f64> = x_seed.iter().take(n).map(|&v| f64::from(v)).collect();
        let b = matvec(&a, &x_true);

        let GaussResult::Unique(x) = solve(&a, &b) else {
            return false;
        };
        // Verify A * x ~ b with a magnitude-scaled tolerance.
        let bb = matvec(&a, &x);
        let mag = b.iter().fold(1.0_f64, |m, v| m.max(v.abs()));
        let tol = 1e-7 * mag.max(1.0);
        approx_eq_vec(&bb, &b, tol)
    }
}
