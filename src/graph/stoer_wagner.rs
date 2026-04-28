//! Stoer–Wagner global minimum cut for an undirected weighted graph.
//!
//! The algorithm runs `n - 1` *minimum-cut phases*. Inside one phase we
//! grow a set `A` starting from an arbitrary active vertex by repeatedly
//! adding the vertex outside `A` whose total edge weight back into `A`
//! is largest ("most tightly connected"). Let `t` be the last vertex
//! added and `s` the second-to-last. The "cut-of-the-phase" is the
//! sum of edge weights from `t` to all other active vertices, i.e.
//! `w(A \ {t}, {t})`, and is provably a candidate for the global min
//! cut. We then *contract* `t` into `s` (merge their incident weights)
//! and repeat. The smallest cut-of-the-phase ever seen is the answer.
//!
//! Total cost: `O(V^3)` time, `O(V^2)` space, fully deterministic — no
//! randomness and no heap, just two `Vec`s of size `n` per phase.
//!
//! ## Input format
//!
//! The graph is supplied as an `n × n` adjacency matrix
//! `weights[u][v]` of non-negative `u64` edge weights. The matrix must
//! be square and symmetric (`weights[u][v] == weights[v][u]`); a weight
//! of `0` denotes the absence of an edge. Self-loops (`weights[v][v]`)
//! are ignored — they contribute to no cut.
//!
//! ## Edge cases
//!
//! - `n < 2` → returns `0` (no cut exists with fewer than two
//!   vertices to separate).
//! - Disconnected graph → returns `0`. The cheapest cut is the empty
//!   set of edges between two existing components, and the algorithm
//!   discovers it naturally because some phase will end with a
//!   detached `t` whose total back-weight is `0`.
//!
//! ## Reference
//!
//! Stoer, M. & Wagner, F. (1997). *A simple min-cut algorithm.* Journal
//! of the ACM 44(4): 585–591.

/// Computes the global minimum cut of an undirected weighted graph
/// using the Stoer–Wagner algorithm.
///
/// `weights` is a square symmetric adjacency matrix of non-negative
/// edge weights; `weights[u][v] == 0` means "no edge". Self-loop
/// entries on the diagonal are ignored. Returns `0` for graphs with
/// fewer than two vertices and for disconnected graphs (the cut
/// between two existing components has weight `0`).
///
/// Runs in `O(V^3)` time and `O(V^2)` space, deterministically.
///
/// # Panics
///
/// Panics in debug builds if `weights` is not square. The implementation
/// does not verify symmetry — passing an asymmetric matrix is a
/// programmer error and will produce undefined-but-safe output.
pub fn stoer_wagner(weights: &[Vec<u64>]) -> u64 {
    let n = weights.len();
    if n < 2 {
        return 0;
    }
    debug_assert!(
        weights.iter().all(|row| row.len() == n),
        "stoer_wagner: adjacency matrix must be square"
    );

    // Working copy of the adjacency matrix — phases mutate it by
    // contracting one vertex into another, so we cannot touch the
    // caller's slice.
    let mut graph: Vec<Vec<u64>> = weights.to_vec();

    // `alive[v]` tracks whether vertex `v` still represents its own
    // supervertex; once it is contracted into another vertex it is
    // marked dead and skipped by every subsequent phase.
    let mut alive = vec![true; n];

    let mut best = u64::MAX;

    // Each phase shrinks the active set by one (via contraction), so
    // after `n - 1` phases only one supervertex remains and we are
    // done.
    for _ in 0..n - 1 {
        let cut = minimum_cut_phase(&mut graph, &mut alive);
        if cut < best {
            best = cut;
        }
        // Early-out: a cut of zero is already the smallest possible
        // value for non-negative weights, so further phases cannot
        // improve on it. This also short-circuits disconnected graphs.
        if best == 0 {
            return 0;
        }
    }

    if best == u64::MAX {
        0
    } else {
        best
    }
}

/// Runs one Stoer–Wagner minimum-cut phase on the live subgraph,
/// contracts the last vertex added (`t`) into the second-to-last (`s`),
/// and returns the cut-of-the-phase weight `w({t}, A \ {t})`.
fn minimum_cut_phase(graph: &mut [Vec<u64>], alive: &mut [bool]) -> u64 {
    let n = graph.len();

    // `in_a[v]` flips to true once `v` is absorbed into the growing set
    // `A`; `weight_to_a[v]` is the running sum of edge weights from
    // `v` into `A` (only meaningful while `v` is alive and not yet
    // in `A`).
    let mut in_a = vec![false; n];
    let mut weight_to_a = vec![0u64; n];

    // Pick any live vertex as the seed of `A` — the algorithm's
    // correctness does not depend on which one. We take the first.
    let start = alive
        .iter()
        .position(|&a| a)
        .expect("phase called with no live vertices");
    in_a[start] = true;
    for v in 0..n {
        if v != start && alive[v] {
            weight_to_a[v] = graph[start][v];
        }
    }

    // Track the live vertex count so we know when to stop adding to
    // `A` (we add every live vertex except `start`, i.e. count - 1
    // additions).
    let live_count = alive.iter().filter(|&&a| a).count();

    // `s` and `t` are the second-to-last and last vertices added to
    // `A` respectively; the cut-of-the-phase is `weight_to_a[t]` at
    // the moment `t` is added.
    let mut s = start;
    let mut t = start;
    let mut cut_of_the_phase = 0u64;

    for _ in 0..live_count - 1 {
        // Pick the live, not-yet-in-A vertex with the largest
        // weight-back-into-A. O(V) per pick, V picks per phase ⇒
        // O(V^2) per phase ⇒ O(V^3) overall.
        let mut next = usize::MAX;
        let mut best_w: i128 = -1;
        for v in 0..n {
            if alive[v] && !in_a[v] && i128::from(weight_to_a[v]) > best_w {
                best_w = i128::from(weight_to_a[v]);
                next = v;
            }
        }
        debug_assert!(next != usize::MAX, "no candidate vertex found in phase");

        s = t;
        t = next;
        cut_of_the_phase = weight_to_a[next];

        in_a[next] = true;
        // Update remaining vertices' weight-to-A by adding edges to
        // the freshly admitted `next`.
        for v in 0..n {
            if alive[v] && !in_a[v] {
                weight_to_a[v] = weight_to_a[v].saturating_add(graph[next][v]);
            }
        }
    }

    // Contract `t` into `s`: every edge incident to `t` becomes an
    // edge of `s` with the combined weight, and `t` is retired.
    if s != t {
        for v in 0..n {
            if v != s && v != t && alive[v] {
                let merged = graph[s][v].saturating_add(graph[t][v]);
                graph[s][v] = merged;
                graph[v][s] = merged;
            }
        }
        // Zero out `t`'s row and column so any stale reads are
        // harmless even though `alive[t]` already gates access.
        for v in 0..n {
            graph[t][v] = 0;
            graph[v][t] = 0;
        }
        alive[t] = false;
    }

    cut_of_the_phase
}

#[cfg(test)]
mod tests {
    use super::stoer_wagner;
    use quickcheck_macros::quickcheck;

    /// Brute-force global min cut by enumerating every non-trivial
    /// bipartition of the vertex set. O(2^n · n^2) — only viable for
    /// tiny `n`.
    fn brute_force_min_cut(weights: &[Vec<u64>]) -> u64 {
        let n = weights.len();
        if n < 2 {
            return 0;
        }
        let mut best = u64::MAX;
        // Fix vertex 0 on the "S" side to halve the work and avoid
        // counting each cut twice; iterate masks for the remaining
        // vertices.
        for mask in 0u32..(1u32 << (n - 1)) {
            // Decode: bit `i` set ⇒ vertex `i + 1` is on the T side.
            let mut on_t = vec![false; n];
            for i in 0..n - 1 {
                if mask & (1 << i) != 0 {
                    on_t[i + 1] = true;
                }
            }
            // Skip the trivial case where T is empty (mask = 0).
            if !on_t.iter().any(|&b| b) {
                continue;
            }
            let mut cut: u64 = 0;
            for u in 0..n {
                for v in (u + 1)..n {
                    if on_t[u] != on_t[v] {
                        cut = cut.saturating_add(weights[u][v]);
                    }
                }
            }
            if cut < best {
                best = cut;
            }
        }
        best
    }

    #[test]
    fn fewer_than_two_vertices_is_zero() {
        assert_eq!(stoer_wagner(&[]), 0);
        assert_eq!(stoer_wagner(&[vec![0]]), 0);
    }

    #[test]
    fn two_vertices_single_edge() {
        let g = vec![vec![0, 7], vec![7, 0]];
        assert_eq!(stoer_wagner(&g), 7);
    }

    #[test]
    fn triangle_unit_weights_min_cut_is_two() {
        let g = vec![vec![0, 1, 1], vec![1, 0, 1], vec![1, 1, 0]];
        assert_eq!(stoer_wagner(&g), 2);
    }

    #[test]
    fn complete_k4_unit_weights_min_cut_is_three() {
        let g = vec![
            vec![0, 1, 1, 1],
            vec![1, 0, 1, 1],
            vec![1, 1, 0, 1],
            vec![1, 1, 1, 0],
        ];
        assert_eq!(stoer_wagner(&g), 3);
    }

    #[test]
    fn weighted_bridge_between_cliques() {
        // Two K3 cliques on {0,1,2} and {3,4,5}, joined by a single
        // edge (2, 3) of weight 1. All clique edges have weight 5.
        // The global min cut is the bridge ⇒ 1.
        let mut g = vec![vec![0u64; 6]; 6];
        for a in 0..3 {
            for b in (a + 1)..3 {
                g[a][b] = 5;
                g[b][a] = 5;
            }
        }
        for a in 3..6 {
            for b in (a + 1)..6 {
                g[a][b] = 5;
                g[b][a] = 5;
            }
        }
        g[2][3] = 1;
        g[3][2] = 1;
        assert_eq!(stoer_wagner(&g), 1);
    }

    #[test]
    fn stoer_wagner_paper_example() {
        // The eight-vertex worked example from Stoer & Wagner (1997),
        // Figure 1. Vertices labelled 1..=8 in the paper map to
        // indices 0..=7 here. Documented global min cut is 4 (the
        // {3, 4, 7, 8} vs {1, 2, 5, 6} partition in paper labels,
        // i.e. {2, 3, 6, 7} vs {0, 1, 4, 5} in zero-based indices).
        let edges: &[(usize, usize, u64)] = &[
            (0, 1, 2),
            (0, 4, 3),
            (1, 2, 3),
            (1, 4, 2),
            (1, 5, 2),
            (2, 3, 4),
            (2, 6, 2),
            (3, 6, 2),
            (3, 7, 2),
            (4, 5, 3),
            (5, 6, 1),
            (6, 7, 3),
        ];
        let mut g = vec![vec![0u64; 8]; 8];
        for &(u, v, w) in edges {
            g[u][v] = w;
            g[v][u] = w;
        }
        assert_eq!(stoer_wagner(&g), 4);
    }

    #[test]
    fn disconnected_graph_is_zero() {
        // Two K2 components on {0,1} and {2,3} with no edge between
        // them ⇒ cut between components is 0.
        let g = vec![
            vec![0, 5, 0, 0],
            vec![5, 0, 0, 0],
            vec![0, 0, 0, 9],
            vec![0, 0, 9, 0],
        ];
        assert_eq!(stoer_wagner(&g), 0);
    }

    #[test]
    fn isolated_vertex_yields_zero() {
        // Vertex 2 has no edges to anyone — global min cut is 0
        // (separate {2} from the rest).
        let g = vec![vec![0, 4, 0], vec![4, 0, 0], vec![0, 0, 0]];
        assert_eq!(stoer_wagner(&g), 0);
    }

    #[test]
    fn self_loops_are_ignored() {
        // Diagonal entries must not affect the answer; the underlying
        // graph is a triangle with unit weights ⇒ min cut is 2.
        let g = vec![vec![9, 1, 1], vec![1, 9, 1], vec![1, 1, 9]];
        assert_eq!(stoer_wagner(&g), 2);
    }

    #[allow(clippy::needless_pass_by_value)]
    #[quickcheck]
    fn quickcheck_matches_brute_force(seed: Vec<u8>, n_seed: u8) -> bool {
        // Bound n ∈ [0, 5] and weights ∈ [0, 5] so the brute force is
        // tractable (2^5 = 32 subsets).
        let n = (n_seed as usize) % 6;
        let mut g = vec![vec![0u64; n]; n];
        let mut idx = 0usize;
        for u in 0..n {
            for v in (u + 1)..n {
                let w = if seed.is_empty() {
                    0
                } else {
                    u64::from(seed[idx % seed.len()] % 6)
                };
                g[u][v] = w;
                g[v][u] = w;
                idx += 1;
            }
        }
        let expected = brute_force_min_cut(&g);
        let got = stoer_wagner(&g);
        // Brute force returns u64::MAX if no valid bipartition was
        // explored (i.e. n < 2); stoer_wagner returns 0 in that case.
        let expected_norm = if expected == u64::MAX { 0 } else { expected };
        got == expected_norm
    }
}
