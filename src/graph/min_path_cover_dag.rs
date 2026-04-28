//! Minimum vertex-disjoint path cover on a DAG via Dilworth's theorem.
//!
//! A *vertex-disjoint path cover* of a directed acyclic graph `G = (V, E)`
//! is a set of directed simple paths in `G` such that every vertex of `V`
//! lies on exactly one path. The *minimum path cover* is one of smallest
//! cardinality.
//!
//! # Algorithm
//! Build the *split bipartite graph* `B` whose left side `L` and right
//! side `R` are each a copy of `V` (think of `u_out ∈ L` and `v_in ∈ R`):
//! for every DAG edge `u → v` add the bipartite edge `u_out — v_in`. Let
//! `M` be a maximum matching in `B`.
//!
//! **Dilworth's theorem (path-cover form).** The minimum number of
//! vertex-disjoint paths needed to cover all `n` vertices of a DAG equals
//! `n − |M|`.
//!
//! Sketch: starting with the trivial cover of `n` singleton paths, every
//! matched edge `u_out — v_in` lets us splice the path ending at `u` and
//! the path starting at `v` into one longer path, dropping the count by
//! one. Bipartite-matching constraints (each `u_out` and each `v_in`
//! appears in `M` at most once) are exactly the constraints that this
//! splicing produces a valid vertex-disjoint path cover. Maximizing `|M|`
//! therefore minimizes the number of paths.
//!
//! This implementation uses [`hopcroft_karp`](super::hopcroft_karp::hopcroft_karp)
//! for the bipartite-matching step.
//!
//! # Complexity
//! Let `n = |V|` and `m = |E|`.
//! - Time:  O(m · √n) — dominated by Hopcroft–Karp on a bipartite graph
//!   with `2n` vertices and `m` edges.
//! - Space: O(n + m).
//!
//! # Preconditions
//! - The input graph must be a DAG. The algorithm does not check this and
//!   results on graphs with cycles are not meaningful as a path cover.
//! - Edge endpoints must lie in `0..n`. Out-of-range endpoints are
//!   **undefined behaviour** (they will panic on out-of-bounds access in
//!   the underlying matching routine).
//! - Parallel edges and self-loops are tolerated but redundant: duplicates
//!   never improve the matching, and a self-loop `v → v` would match
//!   `v_out — v_in` and incorrectly drop the cover count, so callers
//!   should not pass self-loops.

use super::hopcroft_karp::hopcroft_karp;

/// Returns the minimum number of vertex-disjoint paths needed to cover
/// every vertex of the DAG with `n` vertices and the given directed
/// `edges` (each `(u, v)` denoting `u → v`).
///
/// Returns `0` for the empty graph (`n == 0`). For `n` isolated vertices
/// (no edges) returns `n`. For a single Hamiltonian-like chain that
/// already visits every vertex, returns `1`.
pub fn min_path_cover(n: usize, edges: &[(usize, usize)]) -> usize {
    if n == 0 {
        return 0;
    }
    let mut left_adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for &(u, v) in edges {
        left_adj[u].push(v);
    }
    let (matching_size, _, _) = hopcroft_karp(&left_adj, n);
    n - matching_size
}

#[cfg(test)]
mod tests {
    use super::min_path_cover;
    use quickcheck_macros::quickcheck;

    /// Brute-force reference: enumerate every permutation of the `n`
    /// vertices, partition the permutation into maximal contiguous runs
    /// where each consecutive pair `(a, b)` is an edge of the DAG, and
    /// take the minimum number of runs over all permutations.
    ///
    /// A vertex-disjoint path cover corresponds exactly to listing the
    /// vertices in some order and segmenting wherever the next pair is
    /// not an edge. The minimum segmentation over all orderings is the
    /// minimum path cover. Feasible only for very small `n` (at most 6
    /// here, i.e. 720 permutations).
    fn brute_min_path_cover(n: usize, edges: &[(usize, usize)]) -> usize {
        if n == 0 {
            return 0;
        }
        let mut adj_set = vec![vec![false; n]; n];
        for &(u, v) in edges {
            if u < n && v < n {
                adj_set[u][v] = true;
            }
        }
        let mut perm: Vec<usize> = (0..n).collect();
        let mut best = n;
        permute(&mut perm, 0, &mut |p| {
            let mut count = 1;
            for i in 1..p.len() {
                if !adj_set[p[i - 1]][p[i]] {
                    count += 1;
                }
            }
            if count < best {
                best = count;
            }
        });
        best
    }

    fn permute<F: FnMut(&[usize])>(arr: &mut [usize], start: usize, f: &mut F) {
        if start == arr.len() {
            f(arr);
            return;
        }
        for i in start..arr.len() {
            arr.swap(start, i);
            permute(arr, start + 1, f);
            arr.swap(start, i);
        }
    }

    #[test]
    fn empty_graph() {
        assert_eq!(min_path_cover(0, &[]), 0);
    }

    #[test]
    fn single_vertex() {
        assert_eq!(min_path_cover(1, &[]), 1);
    }

    #[test]
    fn antichain_no_edges() {
        // 5 isolated vertices: each its own path.
        assert_eq!(min_path_cover(5, &[]), 5);
    }

    #[test]
    fn single_chain() {
        // 0 -> 1 -> 2 -> 3 covers everything in one path.
        assert_eq!(min_path_cover(4, &[(0, 1), (1, 2), (2, 3)]), 1);
    }

    #[test]
    fn two_disjoint_chains() {
        // 0 -> 1 -> 2 and 3 -> 4 -> 5: two paths.
        assert_eq!(min_path_cover(6, &[(0, 1), (1, 2), (3, 4), (4, 5)]), 2);
    }

    #[test]
    fn y_shape() {
        // 0 -> 2, 1 -> 2, 2 -> 3: one branch must start its own path.
        // Optimal: {0 -> 2 -> 3, 1} or {1 -> 2 -> 3, 0} = 2 paths.
        assert_eq!(min_path_cover(4, &[(0, 2), (1, 2), (2, 3)]), 2);
    }

    #[test]
    fn butterfly() {
        // Two sources {0, 1}, middle {2}, two sinks {3, 4}.
        // Edges: 0->2, 1->2, 2->3, 2->4. Vertex 2 can only chain with one
        // neighbour on each side, but the splicing 0-2-3 leaves {1, 4} as
        // singletons giving 3 paths.
        assert_eq!(min_path_cover(5, &[(0, 2), (1, 2), (2, 3), (2, 4)]), 3);
    }

    #[test]
    fn three_plus_three_dag() {
        // Bipartite-style DAG: left {0,1,2} -> right {3,4,5},
        // edges (0,3), (0,4), (1,4), (1,5), (2,5).
        // Maximum matching = 3 (e.g. 0-3, 1-4, 2-5), so cover = 6 - 3 = 3.
        let edges = [(0, 3), (0, 4), (1, 4), (1, 5), (2, 5)];
        assert_eq!(min_path_cover(6, &edges), 3);
    }

    #[test]
    fn classic_example() {
        // CSES-style example: 7 vertices, edges form a small DAG whose
        // optimal path cover uses 3 paths.
        // Vertices 0..7. Edges:
        //   0->1, 0->2, 1->3, 2->3, 3->4, 3->5, 5->6
        // One optimal cover: (0 -> 1 -> 3 -> 5 -> 6), (2), (4)  = 3 paths.
        let n = 7;
        let edges = [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (3, 5), (5, 6)];
        let got = min_path_cover(n, &edges);
        assert_eq!(got, brute_min_path_cover(n, &edges));
        assert_eq!(got, 3);
    }

    #[test]
    fn complete_dag_is_one_path() {
        // Tournament-style complete DAG on 0..4 with edges i -> j for i < j.
        // 0 -> 1 -> 2 -> 3 already exists, so one path suffices.
        let mut edges = Vec::new();
        for i in 0..4 {
            for j in (i + 1)..4 {
                edges.push((i, j));
            }
        }
        assert_eq!(min_path_cover(4, &edges), 1);
    }

    /// Generate a random DAG on `n` vertices using the seed; only edges
    /// `i -> j` with `i < j` are emitted, guaranteeing acyclicity.
    fn random_dag(n: usize, seed: u64) -> Vec<(usize, usize)> {
        let mut state = seed.wrapping_add(1).wrapping_mul(0x9e37_79b9_7f4a_7c15);
        let mut xorshift = move || -> u64 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let mut edges = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                if xorshift() & 1 == 1 {
                    edges.push((i, j));
                }
            }
        }
        edges
    }

    /// Property test: the bipartite-matching answer must agree with the
    /// permutation-based brute force on small random DAGs.
    #[allow(clippy::needless_pass_by_value)]
    #[quickcheck]
    fn matches_brute_force_small(nn: u8, seed: u64) -> bool {
        let n = (nn as usize) % 6 + 1; // 1..=6
        let edges = random_dag(n, seed);
        min_path_cover(n, &edges) == brute_min_path_cover(n, &edges)
    }
}
