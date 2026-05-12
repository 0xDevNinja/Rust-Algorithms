//! Conway's Game of Life — bounded grid and sparse infinite-plane variants.
//!
//! Conway's Game of Life is a cellular automaton on a 2D lattice where each
//! cell is either alive (`1`) or dead (`0`). Each generation is produced by
//! applying the following rules simultaneously to every cell:
//!
//! - A live cell with **2 or 3** live neighbours stays alive.
//! - A dead cell with **exactly 3** live neighbours becomes alive.
//! - All other cells become or remain dead.
//!
//! Two variants are exposed:
//!
//! - [`step_bounded`] operates on a fixed-size rectangular grid; cells outside
//!   the boundary are treated as dead.
//! - [`step_sparse`] operates on a [`HashSet`] of `(i64, i64)` coordinates of
//!   live cells on the unbounded integer plane. It only inspects cells that
//!   are adjacent to at least one live cell, which is efficient for sparse
//!   patterns such as gliders or spaceships in otherwise empty space.
//!
//! # Complexity
//!
//! Let `R` and `C` be the bounded grid dimensions and `k` the number of live
//! cells in the sparse variant.
//!
//! - [`step_bounded`]: **O(R · C)** time, **O(R · C)** space.
//! - [`step_sparse`]: **O(k)** expected time and space (each live cell
//!   contributes at most nine candidate cells to the neighbour-count map).

use std::collections::HashMap;
use std::collections::HashSet;

/// Compute the next generation of a fixed-size rectangular grid.
///
/// Each entry is treated as alive if it equals `1` and dead otherwise. Cells
/// beyond the grid edges are considered dead. The returned grid has the same
/// dimensions as the input.
///
/// An empty input (zero rows, or zero columns in the first row) produces an
/// empty output.
pub fn step_bounded(grid: &[Vec<u8>]) -> Vec<Vec<u8>> {
    let rows = grid.len();
    if rows == 0 {
        return Vec::new();
    }
    let cols = grid[0].len();
    if cols == 0 {
        return vec![Vec::new(); rows];
    }

    let alive = |r: i64, c: i64| -> u8 {
        if r < 0 || c < 0 || r >= rows as i64 || c >= cols as i64 {
            0
        } else {
            u8::from(grid[r as usize][c as usize] == 1)
        }
    };

    let mut next = vec![vec![0u8; cols]; rows];
    for r in 0..rows {
        for c in 0..cols {
            let mut neighbours = 0u8;
            for dr in -1..=1i64 {
                for dc in -1..=1i64 {
                    if dr == 0 && dc == 0 {
                        continue;
                    }
                    neighbours += alive(r as i64 + dr, c as i64 + dc);
                }
            }
            let is_alive = grid[r][c] == 1;
            next[r][c] = u8::from(neighbours == 3 || (is_alive && neighbours == 2));
        }
    }
    next
}

/// Compute the next generation of a sparse, unbounded board.
///
/// The input contains the coordinates of all currently live cells. The output
/// contains the coordinates of all cells alive in the next generation. Only
/// cells adjacent to a live cell are examined, so the cost scales with the
/// population rather than the area of the bounding box.
#[allow(clippy::implicit_hasher)]
pub fn step_sparse(alive: &HashSet<(i64, i64)>) -> HashSet<(i64, i64)> {
    let mut counts: HashMap<(i64, i64), u8> = HashMap::with_capacity(alive.len() * 8);
    for &(x, y) in alive {
        for dx in -1..=1i64 {
            for dy in -1..=1i64 {
                if dx == 0 && dy == 0 {
                    continue;
                }
                *counts.entry((x + dx, y + dy)).or_insert(0) += 1;
            }
        }
    }

    let mut next = HashSet::with_capacity(alive.len());
    for (cell, n) in counts {
        let was_alive = alive.contains(&cell);
        if n == 3 || (was_alive && n == 2) {
            next.insert(cell);
        }
    }
    next
}

#[cfg(test)]
mod tests {
    use super::*;

    fn to_set<I: IntoIterator<Item = (i64, i64)>>(iter: I) -> HashSet<(i64, i64)> {
        iter.into_iter().collect()
    }

    #[test]
    fn empty_grid_stays_empty() {
        let grid: Vec<Vec<u8>> = Vec::new();
        assert!(step_bounded(&grid).is_empty());
    }

    #[test]
    fn all_dead_stays_dead() {
        let grid = vec![vec![0u8; 4]; 4];
        let next = step_bounded(&grid);
        assert_eq!(next, vec![vec![0u8; 4]; 4]);
    }

    #[test]
    fn block_is_still_life() {
        // 4x4 grid with a 2x2 block in the middle.
        let grid = vec![
            vec![0, 0, 0, 0],
            vec![0, 1, 1, 0],
            vec![0, 1, 1, 0],
            vec![0, 0, 0, 0],
        ];
        assert_eq!(step_bounded(&grid), grid);
    }

    #[test]
    fn beehive_is_still_life() {
        // 5x6 grid with a beehive.
        let grid = vec![
            vec![0, 0, 0, 0, 0, 0],
            vec![0, 0, 1, 1, 0, 0],
            vec![0, 1, 0, 0, 1, 0],
            vec![0, 0, 1, 1, 0, 0],
            vec![0, 0, 0, 0, 0, 0],
        ];
        assert_eq!(step_bounded(&grid), grid);
    }

    #[test]
    fn blinker_oscillates_with_period_two() {
        let horizontal = vec![
            vec![0, 0, 0, 0, 0],
            vec![0, 0, 0, 0, 0],
            vec![0, 1, 1, 1, 0],
            vec![0, 0, 0, 0, 0],
            vec![0, 0, 0, 0, 0],
        ];
        let vertical = vec![
            vec![0, 0, 0, 0, 0],
            vec![0, 0, 1, 0, 0],
            vec![0, 0, 1, 0, 0],
            vec![0, 0, 1, 0, 0],
            vec![0, 0, 0, 0, 0],
        ];
        let one = step_bounded(&horizontal);
        assert_eq!(one, vertical);
        let two = step_bounded(&one);
        assert_eq!(two, horizontal);
    }

    #[test]
    fn sparse_empty_stays_empty() {
        let alive: HashSet<(i64, i64)> = HashSet::new();
        assert!(step_sparse(&alive).is_empty());
    }

    #[test]
    fn sparse_blinker_oscillates() {
        let horizontal = to_set([(0, 0), (1, 0), (2, 0)]);
        let vertical = to_set([(1, -1), (1, 0), (1, 1)]);
        let one = step_sparse(&horizontal);
        assert_eq!(one, vertical);
        let two = step_sparse(&one);
        assert_eq!(two, horizontal);
    }

    #[test]
    fn sparse_glider_translates() {
        // A standard glider returns to its initial shape, shifted by (+1, +1)
        // (with y growing downward, this is one cell right and one cell down),
        // after exactly four generations.
        //
        // Initial pattern (x = column, y = row, y grows downward):
        //   .#.
        //   ..#
        //   ###
        let start = to_set([(1, 0), (2, 1), (0, 2), (1, 2), (2, 2)]);
        let mut state = start.clone();
        for _ in 0..4 {
            state = step_sparse(&state);
        }
        let expected: HashSet<(i64, i64)> = start.iter().map(|&(x, y)| (x + 1, y + 1)).collect();
        assert_eq!(state, expected);
    }

    #[test]
    fn sparse_block_is_still_life() {
        let block = to_set([(0, 0), (0, 1), (1, 0), (1, 1)]);
        assert_eq!(step_sparse(&block), block);
    }
}
