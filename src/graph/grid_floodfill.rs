//! Grid floodfill: connected-component labelling on an implicit grid graph.
//!
//! Each cell is treated as a node and edges connect cells with equal colour
//! sharing an edge (4-connectivity) or an edge or a corner (8-connectivity).
//! The labeller scans cells in row-major order and runs an iterative BFS from
//! every still-unlabelled cell, painting all reachable cells with a fresh
//! component id. Time and space are linear in the number of grid cells.

use std::collections::VecDeque;

/// Connectivity used for floodfill.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Connectivity {
    /// Up / down / left / right neighbours.
    Four,
    /// All 8 neighbours including diagonals.
    Eight,
}

const D4: [(isize, isize); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];
const D8: [(isize, isize); 8] = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
];

/// Returns `(labels, num_components)` where `labels[r][c]` is the 0-based
/// component id of cell `(r, c)`, and `num_components` is the total number of
/// components. Two cells belong to the same component iff they have equal
/// values and are reachable through neighbouring same-coloured cells under
/// the given connectivity.
///
/// - Time: `O(rows * cols)`.
/// - Space: `O(rows * cols)` for the labels and BFS frontier.
///
/// # Panics
/// Panics if rows have inconsistent lengths.
pub fn label_components<T: Eq>(grid: &[Vec<T>], conn: Connectivity) -> (Vec<Vec<usize>>, usize) {
    let rows = grid.len();
    let cols = grid.first().map_or(0, Vec::len);
    for row in grid {
        assert!(
            row.len() == cols,
            "label_components: ragged row (expected {cols} cols, got {})",
            row.len()
        );
    }
    let mut labels = vec![vec![usize::MAX; cols]; rows];
    if rows == 0 || cols == 0 {
        return (labels, 0);
    }
    let dirs: &[(isize, isize)] = match conn {
        Connectivity::Four => &D4,
        Connectivity::Eight => &D8,
    };

    let mut next_id = 0_usize;
    let mut queue: VecDeque<(usize, usize)> = VecDeque::new();
    for r in 0..rows {
        for c in 0..cols {
            if labels[r][c] != usize::MAX {
                continue;
            }
            let id = next_id;
            next_id += 1;
            labels[r][c] = id;
            queue.push_back((r, c));
            while let Some((cr, cc)) = queue.pop_front() {
                for &(dr, dc) in dirs {
                    let nr = cr as isize + dr;
                    let nc = cc as isize + dc;
                    if nr < 0 || nc < 0 {
                        continue;
                    }
                    let (nr, nc) = (nr as usize, nc as usize);
                    if nr >= rows || nc >= cols {
                        continue;
                    }
                    if labels[nr][nc] != usize::MAX {
                        continue;
                    }
                    if grid[nr][nc] != grid[cr][cc] {
                        continue;
                    }
                    labels[nr][nc] = id;
                    queue.push_back((nr, nc));
                }
            }
        }
    }
    (labels, next_id)
}

/// Floodfills the component containing `(start_row, start_col)` by
/// overwriting every reachable same-coloured cell with `new_value`. Returns
/// the number of cells repainted. A no-op if the source colour already equals
/// `new_value`.
///
/// - Time: `O(rows * cols)`.
/// - Space: `O(rows * cols)`.
///
/// # Panics
/// Panics if `(start_row, start_col)` is out of bounds.
pub fn floodfill<T: Eq + Clone>(
    grid: &mut [Vec<T>],
    start_row: usize,
    start_col: usize,
    new_value: &T,
    conn: Connectivity,
) -> usize {
    let rows = grid.len();
    let cols = grid.first().map_or(0, Vec::len);
    assert!(
        start_row < rows && start_col < cols,
        "floodfill: start ({start_row},{start_col}) out of bounds for {rows}x{cols}"
    );
    let source = grid[start_row][start_col].clone();
    if source == *new_value {
        return 0;
    }
    let dirs: &[(isize, isize)] = match conn {
        Connectivity::Four => &D4,
        Connectivity::Eight => &D8,
    };

    let mut painted = 0_usize;
    let mut queue: VecDeque<(usize, usize)> = VecDeque::new();
    queue.push_back((start_row, start_col));
    grid[start_row][start_col] = new_value.clone();
    painted += 1;

    while let Some((cr, cc)) = queue.pop_front() {
        for &(dr, dc) in dirs {
            let nr = cr as isize + dr;
            let nc = cc as isize + dc;
            if nr < 0 || nc < 0 {
                continue;
            }
            let (nr, nc) = (nr as usize, nc as usize);
            if nr >= rows || nc >= cols {
                continue;
            }
            if grid[nr][nc] != source {
                continue;
            }
            grid[nr][nc] = new_value.clone();
            painted += 1;
            queue.push_back((nr, nc));
        }
    }
    painted
}

#[cfg(test)]
mod tests {
    use super::{floodfill, label_components, Connectivity};

    #[test]
    fn empty_grid() {
        let g: Vec<Vec<u8>> = vec![];
        let (labels, n) = label_components(&g, Connectivity::Four);
        assert!(labels.is_empty());
        assert_eq!(n, 0);
    }

    #[test]
    fn single_cell() {
        let g = vec![vec![1_u8]];
        let (labels, n) = label_components(&g, Connectivity::Four);
        assert_eq!(labels, vec![vec![0]]);
        assert_eq!(n, 1);
    }

    #[test]
    fn checkerboard_4_connected() {
        let g = vec![vec![0_u8, 1, 0], vec![1, 0, 1], vec![0, 1, 0]];
        let (_, n) = label_components(&g, Connectivity::Four);
        assert_eq!(n, 9);
    }

    #[test]
    fn checkerboard_8_connected() {
        let g = vec![vec![0_u8, 1, 0], vec![1, 0, 1], vec![0, 1, 0]];
        let (_, n) = label_components(&g, Connectivity::Eight);
        // Both colours form a single 8-connected blob each.
        assert_eq!(n, 2);
    }

    #[test]
    fn islands_4_connected() {
        let g = vec![
            vec![1_u8, 1, 0, 0, 0],
            vec![1, 1, 0, 0, 1],
            vec![0, 0, 0, 1, 1],
            vec![0, 1, 0, 1, 0],
        ];
        let (_, n) = label_components(&g, Connectivity::Four);
        // ones: top-left blob, right-middle blob, bottom singleton (3)
        // zeros: big blob + isolated (3,4) (2) -> 5 components total
        assert_eq!(n, 5);
    }

    #[test]
    fn floodfill_paints_only_component() {
        let mut g = vec![vec![1_u8, 1, 0], vec![1, 1, 0], vec![0, 0, 0]];
        let painted = floodfill(&mut g, 0, 0, &9, Connectivity::Four);
        assert_eq!(painted, 4);
        assert_eq!(g, vec![vec![9, 9, 0], vec![9, 9, 0], vec![0, 0, 0]]);
    }

    #[test]
    fn floodfill_noop_when_color_unchanged() {
        let mut g = vec![vec![5_u8, 5], vec![5, 5]];
        let painted = floodfill(&mut g, 0, 0, &5, Connectivity::Four);
        assert_eq!(painted, 0);
        assert_eq!(g, vec![vec![5, 5], vec![5, 5]]);
    }

    #[test]
    fn floodfill_diagonal_via_eight_connectivity() {
        let mut g = vec![vec![1_u8, 0], vec![0, 1]];
        let painted = floodfill(&mut g, 0, 0, &9, Connectivity::Eight);
        assert_eq!(painted, 2);
        assert_eq!(g, vec![vec![9, 0], vec![0, 9]]);
    }

    #[test]
    #[should_panic(expected = "ragged row")]
    fn ragged_row_panics() {
        let g = vec![vec![1_u8, 2], vec![3]];
        let _ = label_components(&g, Connectivity::Four);
    }
}
