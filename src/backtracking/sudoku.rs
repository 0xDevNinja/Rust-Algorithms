//! Sudoku 9×9 backtracking solver. Uses bitmask bookkeeping for fast
//! "candidate" lookups (rows, columns, boxes).
//!
//! `0` represents an empty cell; `1..=9` are filled values.

const N: usize = 9;

/// Solves the puzzle in place. Returns `true` if a solution was found.
///
/// If the puzzle has multiple solutions, the lexicographically-first one
/// is returned.
pub fn solve(board: &mut [[u8; N]; N]) -> bool {
    let mut rows = [0_u16; N];
    let mut cols = [0_u16; N];
    let mut boxes = [0_u16; N];
    for r in 0..N {
        for c in 0..N {
            let v = board[r][c];
            if v != 0 {
                let bit = 1_u16 << v;
                if rows[r] & bit != 0 || cols[c] & bit != 0 || boxes[box_idx(r, c)] & bit != 0 {
                    return false; // contradiction in input
                }
                rows[r] |= bit;
                cols[c] |= bit;
                boxes[box_idx(r, c)] |= bit;
            }
        }
    }
    backtrack(board, &mut rows, &mut cols, &mut boxes, 0)
}

const fn box_idx(r: usize, c: usize) -> usize {
    (r / 3) * 3 + c / 3
}

fn backtrack(
    board: &mut [[u8; N]; N],
    rows: &mut [u16; N],
    cols: &mut [u16; N],
    boxes: &mut [u16; N],
    pos: usize,
) -> bool {
    if pos == N * N {
        return true;
    }
    let r = pos / N;
    let c = pos % N;
    if board[r][c] != 0 {
        return backtrack(board, rows, cols, boxes, pos + 1);
    }
    let used = rows[r] | cols[c] | boxes[box_idx(r, c)];
    for v in 1_u8..=9 {
        let bit = 1_u16 << v;
        if used & bit != 0 {
            continue;
        }
        board[r][c] = v;
        rows[r] |= bit;
        cols[c] |= bit;
        boxes[box_idx(r, c)] |= bit;
        if backtrack(board, rows, cols, boxes, pos + 1) {
            return true;
        }
        board[r][c] = 0;
        rows[r] &= !bit;
        cols[c] &= !bit;
        boxes[box_idx(r, c)] &= !bit;
    }
    false
}

#[cfg(test)]
mod tests {
    use super::{solve, N};

    fn check_valid(board: &[[u8; N]; N]) -> bool {
        for i in 0..N {
            let mut row = [false; 10];
            let mut col = [false; 10];
            for j in 0..N {
                let r = board[i][j] as usize;
                let c = board[j][i] as usize;
                if r == 0 || row[r] {
                    return false;
                }
                if c == 0 || col[c] {
                    return false;
                }
                row[r] = true;
                col[c] = true;
            }
        }
        for br in 0..3 {
            for bc in 0..3 {
                let mut used = [false; 10];
                for r in br * 3..br * 3 + 3 {
                    for c in bc * 3..bc * 3 + 3 {
                        let v = board[r][c] as usize;
                        if v == 0 || used[v] {
                            return false;
                        }
                        used[v] = true;
                    }
                }
            }
        }
        true
    }

    #[test]
    fn empty_board_solves() {
        let mut b = [[0_u8; N]; N];
        assert!(solve(&mut b));
        assert!(check_valid(&b));
    }

    #[test]
    fn classic_easy() {
        // Wikipedia "easy" example.
        let mut b: [[u8; N]; N] = [
            [5, 3, 0, 0, 7, 0, 0, 0, 0],
            [6, 0, 0, 1, 9, 5, 0, 0, 0],
            [0, 9, 8, 0, 0, 0, 0, 6, 0],
            [8, 0, 0, 0, 6, 0, 0, 0, 3],
            [4, 0, 0, 8, 0, 3, 0, 0, 1],
            [7, 0, 0, 0, 2, 0, 0, 0, 6],
            [0, 6, 0, 0, 0, 0, 2, 8, 0],
            [0, 0, 0, 4, 1, 9, 0, 0, 5],
            [0, 0, 0, 0, 8, 0, 0, 7, 9],
        ];
        assert!(solve(&mut b));
        assert!(check_valid(&b));
        assert_eq!(b[0], [5, 3, 4, 6, 7, 8, 9, 1, 2]);
    }

    #[test]
    fn already_solved_input() {
        let mut b: [[u8; N]; N] = [
            [5, 3, 4, 6, 7, 8, 9, 1, 2],
            [6, 7, 2, 1, 9, 5, 3, 4, 8],
            [1, 9, 8, 3, 4, 2, 5, 6, 7],
            [8, 5, 9, 7, 6, 1, 4, 2, 3],
            [4, 2, 6, 8, 5, 3, 7, 9, 1],
            [7, 1, 3, 9, 2, 4, 8, 5, 6],
            [9, 6, 1, 5, 3, 7, 2, 8, 4],
            [2, 8, 7, 4, 1, 9, 6, 3, 5],
            [3, 4, 5, 2, 8, 6, 1, 7, 9],
        ];
        assert!(solve(&mut b));
        assert!(check_valid(&b));
    }

    #[test]
    fn invalid_input_returns_false() {
        // Two 5s in the first row.
        let mut b = [[0_u8; N]; N];
        b[0][0] = 5;
        b[0][1] = 5;
        assert!(!solve(&mut b));
    }

    #[test]
    fn near_solved_one_blank() {
        let mut b: [[u8; N]; N] = [
            [5, 3, 4, 6, 7, 8, 9, 1, 2],
            [6, 7, 2, 1, 9, 5, 3, 4, 8],
            [1, 9, 8, 3, 4, 2, 5, 6, 7],
            [8, 5, 9, 7, 6, 1, 4, 2, 3],
            [4, 2, 6, 8, 5, 3, 7, 9, 1],
            [7, 1, 3, 9, 2, 4, 8, 5, 6],
            [9, 6, 1, 5, 3, 7, 2, 8, 4],
            [2, 8, 7, 4, 1, 9, 6, 3, 5],
            [3, 4, 5, 2, 8, 6, 1, 7, 0],
        ];
        assert!(solve(&mut b));
        assert_eq!(b[8][8], 9);
    }
}
