//! Knight's tour on an `n × n` chessboard via backtracking guided by
//! Warnsdorff's heuristic.
//!
//! A knight's tour visits every square of the board exactly once using the
//! standard chess knight move (±1, ±2) / (±2, ±1). At each step we prefer
//! the next square with the fewest unvisited onward moves (Warnsdorff's
//! rule); on the rare ties / dead-ends we fall back to ordered backtracking.
//!
//! ## Open vs closed tours
//! This solver returns *open* tours — the last square need not be a knight's
//! move away from the start. Closed tours exist only on boards with `n` even
//! and `n ≥ 6` (and on a few non-square boards we don't handle here). For
//! `n < 5` most starting squares admit no tour at all; for `n ≥ 5` an open
//! tour exists from every starting square and Warnsdorff usually finds one
//! without backtracking.
//!
//! ## Complexity
//! Worst-case time is exponential in `n²` (it's a Hamiltonian-path search),
//! but Warnsdorff's heuristic makes the typical case effectively linear in
//! the number of squares for `n ≥ 5`. Space is `O(n²)` for the board plus
//! recursion depth `O(n²)`.

/// The eight knight move offsets.
const MOVES: [(i32, i32); 8] = [
    (-2, -1),
    (-2, 1),
    (-1, -2),
    (-1, 2),
    (1, -2),
    (1, 2),
    (2, -1),
    (2, 1),
];

/// Find a knight's tour on an `n × n` board starting from `start = (row, col)`.
///
/// Returns `Some(board)` where `board[r][c]` is the move number on which the
/// knight visits `(r, c)` (`0` is the starting square, `n*n - 1` is the last).
/// Returns `None` if `n == 0`, the start is out of range, or no tour exists.
///
/// The search is backtracking guided by Warnsdorff's rule: from the current
/// square the knight tries unvisited neighbours in order of increasing
/// onward-move count, falling back to other neighbours if the preferred
/// pick fails.
pub fn knights_tour(n: usize, start: (usize, usize)) -> Option<Vec<Vec<usize>>> {
    if n == 0 {
        return None;
    }
    let (sr, sc) = start;
    if sr >= n || sc >= n {
        return None;
    }
    let mut board = vec![vec![usize::MAX; n]; n];
    board[sr][sc] = 0;
    if n == 1 {
        return Some(board);
    }
    if backtrack(&mut board, n, sr as i32, sc as i32, 1) {
        Some(board)
    } else {
        None
    }
}

/// Count the number of unvisited squares the knight can reach from `(r, c)`.
fn onward_count(board: &[Vec<usize>], n: i32, r: i32, c: i32) -> usize {
    let mut count = 0;
    for (dr, dc) in MOVES {
        let nr = r + dr;
        let nc = c + dc;
        if nr >= 0 && nr < n && nc >= 0 && nc < n && board[nr as usize][nc as usize] == usize::MAX {
            count += 1;
        }
    }
    count
}

/// Recursive backtracking step. `move_no` is the index to assign to the next
/// square the knight visits. Returns `true` when a complete tour has been
/// written into `board`.
fn backtrack(board: &mut [Vec<usize>], n: usize, r: i32, c: i32, move_no: usize) -> bool {
    if move_no == n * n {
        return true;
    }
    let ni = n as i32;
    // Gather unvisited neighbours ordered by Warnsdorff's degree.
    let mut candidates: Vec<(usize, i32, i32)> = Vec::with_capacity(8);
    for (dr, dc) in MOVES {
        let nr = r + dr;
        let nc = c + dc;
        if nr >= 0 && nr < ni && nc >= 0 && nc < ni && board[nr as usize][nc as usize] == usize::MAX
        {
            let deg = onward_count(board, ni, nr, nc);
            candidates.push((deg, nr, nc));
        }
    }
    candidates.sort_by_key(|&(deg, _, _)| deg);

    for (_, nr, nc) in candidates {
        board[nr as usize][nc as usize] = move_no;
        if backtrack(board, n, nr, nc, move_no + 1) {
            return true;
        }
        board[nr as usize][nc as usize] = usize::MAX;
    }
    false
}

#[cfg(test)]
mod tests {
    use super::{knights_tour, MOVES};

    /// Verify that `board` is a valid knight's tour: every value in
    /// `0..n*n` appears exactly once, and consecutive values are a single
    /// knight move apart.
    fn assert_valid_tour(board: &[Vec<usize>]) {
        let n = board.len();
        let total = n * n;
        // Build move_no -> (r, c) map and check coverage.
        let mut positions = vec![(usize::MAX, usize::MAX); total];
        for (r, row) in board.iter().enumerate() {
            assert_eq!(row.len(), n, "non-square board");
            for (c, &v) in row.iter().enumerate() {
                assert!(v < total, "value {v} out of range at ({r},{c})");
                assert_eq!(
                    positions[v],
                    (usize::MAX, usize::MAX),
                    "duplicate move number {v}"
                );
                positions[v] = (r, c);
            }
        }
        // Check every consecutive pair is a knight move.
        for k in 1..total {
            let (pr, pc) = positions[k - 1];
            let (cr, cc) = positions[k];
            let dr = cr as i32 - pr as i32;
            let dc = cc as i32 - pc as i32;
            assert!(
                MOVES.contains(&(dr, dc)),
                "step {k}: ({pr},{pc}) -> ({cr},{cc}) is not a knight move (d=({dr},{dc}))"
            );
        }
    }

    #[test]
    fn n_zero_returns_none() {
        assert!(knights_tour(0, (0, 0)).is_none());
    }

    #[test]
    fn out_of_range_start_returns_none() {
        assert!(knights_tour(5, (5, 0)).is_none());
        assert!(knights_tour(5, (0, 5)).is_none());
        assert!(knights_tour(8, (8, 8)).is_none());
    }

    #[test]
    fn n_one_trivial_tour() {
        let tour = knights_tour(1, (0, 0)).expect("trivial 1x1 tour");
        assert_eq!(tour, vec![vec![0_usize]]);
    }

    #[test]
    fn n_five_from_origin() {
        let board = knights_tour(5, (0, 0)).expect("5x5 tour from (0,0) must exist");
        assert_valid_tour(&board);
        assert_eq!(board[0][0], 0);
    }

    #[test]
    fn n_five_from_center() {
        let board = knights_tour(5, (2, 2)).expect("5x5 tour from (2,2) must exist");
        assert_valid_tour(&board);
        assert_eq!(board[2][2], 0);
    }

    #[test]
    fn n_six_open_tour() {
        let board = knights_tour(6, (0, 0)).expect("6x6 open tour from (0,0)");
        assert_valid_tour(&board);
    }

    #[test]
    fn n_eight_smoke_tests() {
        for start in [(0, 0), (3, 3), (0, 7), (7, 0), (4, 2)] {
            let board = knights_tour(8, start).unwrap_or_else(|| {
                panic!("8x8 tour from {start:?} should be findable by Warnsdorff")
            });
            assert_valid_tour(&board);
            assert_eq!(board[start.0][start.1], 0);
        }
    }
}
