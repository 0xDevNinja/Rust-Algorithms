//! Dancing Links (DLX) implementation of Knuth's Algorithm X for the exact
//! cover problem.
//!
//! Given a 0/1 matrix with `r` rows and `c` columns, the exact cover problem
//! asks for a subset of rows such that every column contains exactly one `1`
//! among the selected rows. Algorithm X is a recursive, non-deterministic
//! search; Dancing Links is the trick of representing the matrix as a sparse
//! toroidal doubly linked list whose nodes can be unlinked and relinked in
//! O(1), making the search efficient and easy to undo on backtrack.
//!
//! This implementation avoids `unsafe` and raw pointers by storing all nodes
//! in a `Vec<Node>` arena and using `usize` indices in place of pointers.
//!
//! Complexity: exact cover is NP-complete, so the worst-case running time is
//! exponential in the number of rows. In practice DLX with the
//! smallest-column heuristic explores far less than the naive bound; each
//! cover/uncover step is O(rows * `nnz_in_column`).

/// A node in the toroidal doubly linked list. Column headers and matrix
/// entries share the same node layout; column headers additionally use
/// `size` and `column` (set to its own index) and live above row 0.
#[derive(Clone, Copy)]
struct Node {
    left: usize,
    right: usize,
    up: usize,
    down: usize,
    column: usize, // index of the column header
    row: usize,    // original row index (unused for headers)
    size: usize,   // only meaningful for column headers
}

struct Dlx {
    nodes: Vec<Node>,
    header: usize, // index of the root header
    num_columns: usize,
}

impl Dlx {
    fn new(matrix: &[Vec<bool>], num_columns: usize) -> Self {
        // Allocate the root header plus one header per column.
        let mut nodes: Vec<Node> = Vec::with_capacity(1 + num_columns);
        // Root header at index 0.
        nodes.push(Node {
            left: 0,
            right: 0,
            up: 0,
            down: 0,
            column: 0,
            row: 0,
            size: 0,
        });
        // Column headers indexed 1..=num_columns.
        for _ in 0..num_columns {
            let idx = nodes.len();
            nodes.push(Node {
                left: idx - 1,
                right: 0, // patched below
                up: idx,
                down: idx,
                column: idx,
                row: 0,
                size: 0,
            });
            nodes[idx - 1].right = idx;
        }
        if num_columns > 0 {
            // Close the horizontal cycle on the header row.
            let last = num_columns; // index of last header
            nodes[last].right = 0;
            nodes[0].left = last;
        }

        let mut dlx = Self {
            nodes,
            header: 0,
            num_columns,
        };

        // Insert one node per `true` cell, row by row.
        for (row_idx, row) in matrix.iter().enumerate() {
            let mut first_in_row: Option<usize> = None;
            for (c, &cell) in row.iter().enumerate() {
                if !cell {
                    continue;
                }
                let col_header = c + 1; // headers start at index 1
                let new_idx = dlx.nodes.len();
                let col_up = dlx.nodes[col_header].up;
                // Vertical splice: insert above the column header (i.e. at
                // the bottom of the column's circular list).
                dlx.nodes.push(Node {
                    left: new_idx,
                    right: new_idx,
                    up: col_up,
                    down: col_header,
                    column: col_header,
                    row: row_idx,
                    size: 0,
                });
                dlx.nodes[col_up].down = new_idx;
                dlx.nodes[col_header].up = new_idx;
                dlx.nodes[col_header].size += 1;

                // Horizontal splice into the current row.
                match first_in_row {
                    None => first_in_row = Some(new_idx),
                    Some(first) => {
                        let last = dlx.nodes[first].left;
                        dlx.nodes[new_idx].left = last;
                        dlx.nodes[new_idx].right = first;
                        dlx.nodes[last].right = new_idx;
                        dlx.nodes[first].left = new_idx;
                    }
                }
            }
        }

        dlx
    }

    /// Cover column `c`: remove its header from the header list and remove
    /// every row that has a `1` in column `c` from all other columns.
    fn cover(&mut self, c: usize) {
        let r = self.nodes[c].right;
        let l = self.nodes[c].left;
        self.nodes[r].left = l;
        self.nodes[l].right = r;

        let mut i = self.nodes[c].down;
        while i != c {
            let mut j = self.nodes[i].right;
            while j != i {
                let d = self.nodes[j].down;
                let u = self.nodes[j].up;
                self.nodes[d].up = u;
                self.nodes[u].down = d;
                let col = self.nodes[j].column;
                self.nodes[col].size -= 1;
                j = self.nodes[j].right;
            }
            i = self.nodes[i].down;
        }
    }

    /// Inverse of `cover`. Restores the column and all rows that had been
    /// removed when it was covered. Done in reverse order.
    fn uncover(&mut self, c: usize) {
        let mut i = self.nodes[c].up;
        while i != c {
            let mut j = self.nodes[i].left;
            while j != i {
                let col = self.nodes[j].column;
                self.nodes[col].size += 1;
                let d = self.nodes[j].down;
                let u = self.nodes[j].up;
                self.nodes[d].up = j;
                self.nodes[u].down = j;
                j = self.nodes[j].left;
            }
            i = self.nodes[i].up;
        }
        let r = self.nodes[c].right;
        let l = self.nodes[c].left;
        self.nodes[r].left = c;
        self.nodes[l].right = c;
    }

    /// Pick the column header with the fewest remaining nodes (Knuth's "S"
    /// heuristic). Returns `None` if every column is already covered.
    fn choose_column(&self) -> Option<usize> {
        if self.num_columns == 0 {
            return None;
        }
        let mut best: Option<usize> = None;
        let mut best_size = usize::MAX;
        let mut c = self.nodes[self.header].right;
        while c != self.header {
            let s = self.nodes[c].size;
            if s < best_size {
                best_size = s;
                best = Some(c);
                if s <= 1 {
                    break;
                }
            }
            c = self.nodes[c].right;
        }
        best
    }

    fn search(&mut self, partial: &mut Vec<usize>, solutions: &mut Vec<Vec<usize>>) {
        // If no columns remain in the header list, every column is covered.
        if self.nodes[self.header].right == self.header {
            let mut sol = partial.clone();
            sol.sort_unstable();
            solutions.push(sol);
            return;
        }

        let Some(c) = self.choose_column() else {
            return;
        };
        // Dead end: column with no remaining rows.
        if self.nodes[c].size == 0 {
            return;
        }

        self.cover(c);

        let mut r = self.nodes[c].down;
        while r != c {
            partial.push(self.nodes[r].row);
            // Cover all other columns in this row.
            let mut j = self.nodes[r].right;
            while j != r {
                self.cover(self.nodes[j].column);
                j = self.nodes[j].right;
            }

            self.search(partial, solutions);

            // Undo: walk left and uncover.
            partial.pop();
            let mut j = self.nodes[r].left;
            while j != r {
                self.uncover(self.nodes[j].column);
                j = self.nodes[j].left;
            }

            r = self.nodes[r].down;
        }

        self.uncover(c);
    }
}

/// Solve the exact cover problem for the given 0/1 `matrix` (rows by
/// columns) and return every solution as a sorted vector of original row
/// indices.
///
/// Conventions:
/// - A matrix with `0` columns trivially admits the empty selection: the
///   returned value is `vec![vec![]]`.
/// - A matrix with `> 0` columns but `0` rows has no solution: returns
///   `vec![]`.
/// - Each returned solution is sorted ascending so that result equality
///   does not depend on search order.
pub fn solve_exact_cover(matrix: &[Vec<bool>]) -> Vec<Vec<usize>> {
    // Determine the number of columns. If the matrix is empty (no rows),
    // we conservatively treat it as 0 columns, which yields the trivial
    // empty cover.
    let num_columns = matrix.first().map_or(0, Vec::len);

    // Validate that all rows agree on the column count; reject inconsistent
    // input by returning no solutions rather than panicking on malformed
    // matrices.
    if matrix.iter().any(|r| r.len() != num_columns) {
        return Vec::new();
    }

    if num_columns == 0 {
        // The empty subset of rows trivially covers zero columns.
        return vec![Vec::new()];
    }

    let mut dlx = Dlx::new(matrix, num_columns);
    let mut solutions = Vec::new();
    let mut partial = Vec::new();
    dlx.search(&mut partial, &mut solutions);
    solutions
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sorted(mut v: Vec<Vec<usize>>) -> Vec<Vec<usize>> {
        for row in &mut v {
            row.sort_unstable();
        }
        v.sort();
        v
    }

    #[test]
    fn empty_matrix_has_trivial_cover() {
        let matrix: Vec<Vec<bool>> = Vec::new();
        assert_eq!(solve_exact_cover(&matrix), vec![Vec::<usize>::new()]);
    }

    #[test]
    fn zero_columns_with_rows_is_trivial() {
        // A matrix shaped 3x0 still has zero columns, so the empty
        // selection covers everything.
        let matrix: Vec<Vec<bool>> = vec![vec![], vec![], vec![]];
        assert_eq!(solve_exact_cover(&matrix), vec![Vec::<usize>::new()]);
    }

    #[test]
    fn no_rows_but_columns_has_no_solution() {
        // No rows means no way to cover any column; expect no solutions.
        // Represent "0 rows, 3 columns" as a single dummy row to fix the
        // column count, then assert the dummy alone cannot cover.
        let matrix: Vec<Vec<bool>> = vec![vec![false, false, false]];
        // The all-false row covers nothing, so no exact cover exists.
        assert_eq!(solve_exact_cover(&matrix), Vec::<Vec<usize>>::new());
    }

    #[test]
    fn knuth_seven_column_example_has_unique_solution() {
        // Knuth's canonical exact cover example from "Dancing Links"
        // (arXiv:cs/0011047), 7 columns and 6 rows:
        //   A = {1, 4, 7}
        //   B = {1, 4}
        //   C = {4, 5, 7}
        //   D = {3, 5, 6}
        //   E = {2, 3, 6, 7}
        //   F = {2, 7}
        // The unique exact cover is {B, D, F} = rows {1, 3, 5} (0-indexed),
        // covering columns {1,4} ∪ {3,5,6} ∪ {2,7} = {1..=7}.
        // Columns are zero-indexed below (1→0, …, 7→6).
        let matrix: Vec<Vec<bool>> = vec![
            // A = cols 1,4,7 (idx 0,3,6)
            vec![true, false, false, true, false, false, true],
            // B = cols 1,4 (idx 0,3)
            vec![true, false, false, true, false, false, false],
            // C = cols 4,5,7 (idx 3,4,6)
            vec![false, false, false, true, true, false, true],
            // D = cols 3,5,6 (idx 2,4,5)
            vec![false, false, true, false, true, true, false],
            // E = cols 2,3,6,7 (idx 1,2,5,6)
            vec![false, true, true, false, false, true, true],
            // F = cols 2,7 (idx 1,6)
            vec![false, true, false, false, false, false, true],
        ];
        let sols = sorted(solve_exact_cover(&matrix));
        assert_eq!(sols, vec![vec![1, 3, 5]]);

        // Also verify each reported solution actually partitions the
        // columns (defence-in-depth against an off-by-one in the solver).
        for sol in &sols {
            let mut seen = [false; 7];
            for &r in sol {
                for (c, &v) in matrix[r].iter().enumerate() {
                    if v {
                        assert!(!seen[c], "column {c} covered twice");
                        seen[c] = true;
                    }
                }
            }
            assert!(seen.iter().all(|&b| b), "not every column covered");
        }
    }

    #[test]
    fn identity_matrix_has_unique_solution() {
        // The n×n identity matrix has exactly one exact cover: all rows.
        let n = 5;
        let mut matrix = vec![vec![false; n]; n];
        for (i, row) in matrix.iter_mut().enumerate() {
            row[i] = true;
        }
        let sols = solve_exact_cover(&matrix);
        assert_eq!(sols, vec![vec![0, 1, 2, 3, 4]]);
    }

    #[test]
    fn redundant_rows_yield_multiple_solutions() {
        // Three rows, each covering one of three columns, and a duplicate
        // of row 0. There should be two exact covers: {0,1,2} and {1,2,3}.
        let matrix: Vec<Vec<bool>> = vec![
            vec![true, false, false],
            vec![false, true, false],
            vec![false, false, true],
            vec![true, false, false],
        ];
        let sols = sorted(solve_exact_cover(&matrix));
        assert_eq!(sols, vec![vec![0, 1, 2], vec![1, 2, 3]]);
    }

    #[test]
    fn no_exact_cover_returns_empty() {
        // Two rows, three columns; no combination covers column 2.
        let matrix: Vec<Vec<bool>> = vec![vec![true, false, false], vec![false, true, false]];
        assert!(solve_exact_cover(&matrix).is_empty());
    }

    #[test]
    fn overlapping_rows_excluded() {
        // Rows: {0,1}, {1,2}, {0,2}, {2}. The only exact cover of {0,1,2}
        // is {0} and {1,2}? Let's enumerate:
        // {0,1} ∪ {2} = {0,1,2} ✓ → rows {0, 3}
        // {1,2} ∪ ?     needs to cover 0, no row {0} alone exists; row 2 is {0,2}
        //   overlaps at 2.
        // {0,2} ∪ ?     needs to cover 1, no singleton {1}.
        // So unique solution is rows {0, 3}.
        let matrix: Vec<Vec<bool>> = vec![
            vec![true, true, false],
            vec![false, true, true],
            vec![true, false, true],
            vec![false, false, true],
        ];
        let sols = sorted(solve_exact_cover(&matrix));
        assert_eq!(sols, vec![vec![0, 3]]);
    }

    #[test]
    fn tiny_sudoku_style_partition() {
        // A 4-column problem encoding "place exactly one of two pieces in
        // each of two slots." Columns: slot0, slot1, pieceA, pieceB.
        // Rows: (A in slot0), (A in slot1), (B in slot0), (B in slot1).
        let matrix: Vec<Vec<bool>> = vec![
            vec![true, false, true, false], // A@0
            vec![false, true, true, false], // A@1
            vec![true, false, false, true], // B@0
            vec![false, true, false, true], // B@1
        ];
        // Exact covers: {A@0, B@1} = rows {0,3}; {A@1, B@0} = rows {1,2}.
        let sols = sorted(solve_exact_cover(&matrix));
        assert_eq!(sols, vec![vec![0, 3], vec![1, 2]]);
    }

    #[test]
    fn inconsistent_row_widths_rejected() {
        let matrix: Vec<Vec<bool>> = vec![vec![true, false], vec![true]];
        assert!(solve_exact_cover(&matrix).is_empty());
    }
}
