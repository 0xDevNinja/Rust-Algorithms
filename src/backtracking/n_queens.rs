//! N-queens backtracking solver. Each solution is encoded as a vector of
//! column indices, one per row.

/// Returns all distinct N-queens solutions. Each solution is a `Vec<usize>`
/// of length `n` where `solution[r]` is the column of the queen on row `r`.
pub fn solve_n_queens(n: usize) -> Vec<Vec<usize>> {
    let mut solutions = Vec::new();
    if n == 0 {
        solutions.push(Vec::new());
        return solutions;
    }
    let mut placement = vec![0_usize; n];
    let mut cols_used = vec![false; n];
    let mut diag1_used = vec![false; 2 * n - 1]; // r + c
    let mut diag2_used = vec![false; 2 * n - 1]; // r - c + n - 1
    backtrack(
        0,
        n,
        &mut placement,
        &mut cols_used,
        &mut diag1_used,
        &mut diag2_used,
        &mut solutions,
    );
    solutions
}

/// Returns the count of distinct N-queens solutions without materialising them.
pub fn count_solutions(n: usize) -> usize {
    if n == 0 {
        return 1;
    }
    let mut placement = vec![0_usize; n];
    let mut cols_used = vec![false; n];
    let mut diag1_used = vec![false; 2 * n - 1];
    let mut diag2_used = vec![false; 2 * n - 1];
    let mut count = 0_usize;
    count_backtrack(
        0,
        n,
        &mut placement,
        &mut cols_used,
        &mut diag1_used,
        &mut diag2_used,
        &mut count,
    );
    count
}

#[allow(clippy::too_many_arguments)]
fn backtrack(
    row: usize,
    n: usize,
    placement: &mut [usize],
    cols: &mut [bool],
    d1: &mut [bool],
    d2: &mut [bool],
    out: &mut Vec<Vec<usize>>,
) {
    if row == n {
        out.push(placement.to_vec());
        return;
    }
    for col in 0..n {
        let i1 = row + col;
        let i2 = row + n - 1 - col;
        if cols[col] || d1[i1] || d2[i2] {
            continue;
        }
        cols[col] = true;
        d1[i1] = true;
        d2[i2] = true;
        placement[row] = col;
        backtrack(row + 1, n, placement, cols, d1, d2, out);
        cols[col] = false;
        d1[i1] = false;
        d2[i2] = false;
    }
}

#[allow(clippy::too_many_arguments)]
fn count_backtrack(
    row: usize,
    n: usize,
    placement: &mut [usize],
    cols: &mut [bool],
    d1: &mut [bool],
    d2: &mut [bool],
    count: &mut usize,
) {
    if row == n {
        *count += 1;
        return;
    }
    for col in 0..n {
        let i1 = row + col;
        let i2 = row + n - 1 - col;
        if cols[col] || d1[i1] || d2[i2] {
            continue;
        }
        cols[col] = true;
        d1[i1] = true;
        d2[i2] = true;
        placement[row] = col;
        count_backtrack(row + 1, n, placement, cols, d1, d2, count);
        cols[col] = false;
        d1[i1] = false;
        d2[i2] = false;
    }
}

#[cfg(test)]
mod tests {
    use super::{count_solutions, solve_n_queens};

    fn is_valid(p: &[usize]) -> bool {
        let n = p.len();
        for i in 0..n {
            for j in i + 1..n {
                if p[i] == p[j] {
                    return false;
                }
                if (p[i] as isize - p[j] as isize).abs() == (j - i) as isize {
                    return false;
                }
            }
        }
        true
    }

    #[test]
    fn n_zero_one_solution() {
        // The empty board has one trivial "placement".
        assert_eq!(count_solutions(0), 1);
        assert_eq!(solve_n_queens(0).len(), 1);
    }

    #[test]
    fn n_one() {
        let s = solve_n_queens(1);
        assert_eq!(s, vec![vec![0_usize]]);
    }

    #[test]
    fn n_two_and_three_have_no_solutions() {
        assert!(solve_n_queens(2).is_empty());
        assert!(solve_n_queens(3).is_empty());
        assert_eq!(count_solutions(2), 0);
        assert_eq!(count_solutions(3), 0);
    }

    #[test]
    fn n_four_has_two_solutions() {
        let s = solve_n_queens(4);
        assert_eq!(s.len(), 2);
        for placement in &s {
            assert!(is_valid(placement));
        }
    }

    #[test]
    fn n_eight_has_92_solutions() {
        assert_eq!(count_solutions(8), 92);
    }

    #[test]
    fn solutions_are_valid() {
        for n in 4..=6 {
            for placement in solve_n_queens(n) {
                assert!(is_valid(&placement));
            }
        }
    }
}
