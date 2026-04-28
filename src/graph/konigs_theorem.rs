//! König's theorem: minimum vertex cover in a bipartite graph.
//!
//! **König's theorem.** In any bipartite graph, the size of a maximum
//! matching equals the size of a minimum vertex cover. Moreover, given a
//! maximum matching, a minimum vertex cover can be constructed in linear
//! time using an alternating-path argument.
//!
//! A *vertex cover* is a set of vertices that touches (covers) every edge:
//! for every edge `(u, v)` at least one of `u`, `v` lies in the set.
//!
//! # Algorithm
//! Given a bipartite graph with left set `L = 0..n_left` and right set
//! `R = 0..n_right`:
//!
//! 1. Compute a maximum matching `M` (this implementation calls
//!    [`hopcroft_karp`](super::hopcroft_karp::hopcroft_karp)).
//! 2. Let `U` be the set of left vertices unmatched by `M`. Run a BFS that
//!    alternates: from a left vertex follow any graph edge to the right;
//!    from a right vertex follow only the matching edge back to the left.
//!    Let `Z_L`, `Z_R` be the left and right vertices visited.
//! 3. The minimum vertex cover is `(L \ Z_L) ∪ Z_R` — i.e. the matched left
//!    vertices that are *not* visited together with the right vertices that
//!    *are* visited.
//!
//! Correctness sketch. Every edge `(u, v)` with `u ∈ L`, `v ∈ R` falls into
//! one of two cases:
//!
//! * `u ∈ Z_L`. Then because BFS from `Z_L` follows every graph edge to the
//!   right, `v ∈ Z_R`, so the edge is covered by `v`.
//! * `u ∉ Z_L`. Then `u` is matched (otherwise `u ∈ U ⊆ Z_L`), and `u` is
//!   in the cover, so the edge is covered by `u`.
//!
//! The size of the cover equals `|M|`: every matched edge contributes
//! exactly one endpoint to the cover (a matched right vertex `v ∈ Z_R`
//! implies its mate is in `Z_L`, and a matched left vertex `u ∉ Z_L` is in
//! the cover directly), and unmatched edges cannot exist by saturation.
//! By weak duality (matching ≤ cover) this size is optimal.
//!
//! # Complexity
//! - Time:  O(E · √V) — dominated by Hopcroft–Karp; the alternating BFS is
//!   O(V + E).
//! - Space: O(V + E).
//!
//! # Preconditions
//! - `left_adj.len() == n_left`. Each `left_adj[u]` lists right-vertex
//!   indices in `0..n_right`. Out-of-range right indices panic.
//! - The graph is bipartite with edges only between L and R (this is
//!   automatic from the adjacency-list representation).

use super::hopcroft_karp::hopcroft_karp;
use std::collections::VecDeque;

/// Returns `(left_cover, right_cover)`: the left and right vertices of a
/// minimum vertex cover for the bipartite graph described by `left_adj`
/// (left side) and `n_right` (size of the right side).
///
/// The total cover size equals the maximum-matching size of the graph
/// (König's theorem). Each returned vector is sorted ascending and free of
/// duplicates. For an empty graph (`left_adj` empty and `n_right == 0`)
/// returns `(vec![], vec![])`.
pub fn min_vertex_cover(left_adj: &[Vec<usize>], n_right: usize) -> (Vec<usize>, Vec<usize>) {
    let n_left = left_adj.len();
    if n_left == 0 {
        return (Vec::new(), Vec::new());
    }

    // 1. Maximum matching.
    let (_, match_l, match_r) = hopcroft_karp(left_adj, n_right);

    // 2. Alternating BFS from unmatched left vertices.
    //    Left -> right: follow any edge in `left_adj`.
    //    Right -> left: follow only the matching edge.
    let mut visited_l = vec![false; n_left];
    let mut visited_r = vec![false; n_right];
    let mut queue: VecDeque<usize> = VecDeque::new();
    for u in 0..n_left {
        if match_l[u].is_none() {
            visited_l[u] = true;
            queue.push_back(u);
        }
    }
    while let Some(u) = queue.pop_front() {
        for &v in &left_adj[u] {
            if visited_r[v] {
                continue;
            }
            visited_r[v] = true;
            if let Some(pair) = match_r[v] {
                if !visited_l[pair] {
                    visited_l[pair] = true;
                    queue.push_back(pair);
                }
            }
        }
    }

    // 3. Cover = (matched-left NOT visited) ∪ (right visited).
    let mut left_cover: Vec<usize> = (0..n_left)
        .filter(|&u| !visited_l[u] && match_l[u].is_some())
        .collect();
    let mut right_cover: Vec<usize> = (0..n_right).filter(|&v| visited_r[v]).collect();
    left_cover.sort_unstable();
    right_cover.sort_unstable();
    (left_cover, right_cover)
}

#[cfg(test)]
mod tests {
    use super::super::hopcroft_karp::hopcroft_karp;
    use super::min_vertex_cover;
    use quickcheck_macros::quickcheck;

    /// Asserts `(left_cover, right_cover)` is a valid vertex cover of the
    /// bipartite graph: every edge has at least one endpoint in the cover.
    fn assert_is_cover(
        left_adj: &[Vec<usize>],
        n_right: usize,
        left_cover: &[usize],
        right_cover: &[usize],
    ) {
        let mut in_left = vec![false; left_adj.len()];
        for &u in left_cover {
            in_left[u] = true;
        }
        let mut in_right = vec![false; n_right];
        for &v in right_cover {
            in_right[v] = true;
        }
        for (u, neighbours) in left_adj.iter().enumerate() {
            for &v in neighbours {
                assert!(in_left[u] || in_right[v], "edge ({u}, {v}) is not covered");
            }
        }
    }

    /// Brute-force minimum vertex cover by enumerating every subset of the
    /// vertex set `L ∪ R`. Only feasible when `n_left + n_right` is small.
    fn brute_force_min_cover(left_adj: &[Vec<usize>], n_right: usize) -> usize {
        let n_left = left_adj.len();
        let total = n_left + n_right;
        let mut best = total;
        for mask in 0u32..(1u32 << total) {
            let mut covered = true;
            for (u, neighbours) in left_adj.iter().enumerate() {
                if !covered {
                    break;
                }
                for &v in neighbours {
                    let u_in = (mask >> u) & 1 == 1;
                    let v_in = (mask >> (n_left + v)) & 1 == 1;
                    if !u_in && !v_in {
                        covered = false;
                        break;
                    }
                }
            }
            if covered {
                let size = mask.count_ones() as usize;
                if size < best {
                    best = size;
                }
            }
        }
        best
    }

    #[test]
    fn empty_graph() {
        let left_adj: Vec<Vec<usize>> = vec![];
        let (l, r) = min_vertex_cover(&left_adj, 0);
        assert!(l.is_empty());
        assert!(r.is_empty());
    }

    #[test]
    fn empty_left_nonempty_right() {
        // No left vertices => no edges => empty cover.
        let left_adj: Vec<Vec<usize>> = vec![];
        let (l, r) = min_vertex_cover(&left_adj, 5);
        assert!(l.is_empty());
        assert!(r.is_empty());
    }

    #[test]
    fn no_edges_returns_empty_cover() {
        let left_adj = vec![vec![], vec![], vec![]];
        let (l, r) = min_vertex_cover(&left_adj, 3);
        assert!(l.is_empty());
        assert!(r.is_empty());
    }

    #[test]
    fn k2_cover_one() {
        // Single edge (0,0): cover size = matching size = 1.
        let left_adj = vec![vec![0]];
        let (l, r) = min_vertex_cover(&left_adj, 1);
        assert_eq!(l.len() + r.len(), 1);
        assert_is_cover(&left_adj, 1, &l, &r);
    }

    #[test]
    fn k_2_3_cover_two() {
        // K_{2,3}: max matching = 2, min cover = 2 (the two left vertices).
        let left_adj = vec![vec![0, 1, 2], vec![0, 1, 2]];
        let (l, r) = min_vertex_cover(&left_adj, 3);
        assert_eq!(l.len() + r.len(), 2);
        assert_is_cover(&left_adj, 3, &l, &r);
    }

    #[test]
    fn two_disjoint_k2_cover_two() {
        // Two independent edges (0,0) and (1,1): matching = 2, cover = 2.
        let left_adj = vec![vec![0], vec![1]];
        let (l, r) = min_vertex_cover(&left_adj, 2);
        assert_eq!(l.len() + r.len(), 2);
        assert_is_cover(&left_adj, 2, &l, &r);
    }

    #[test]
    fn classic_alternating_path_example() {
        // Same shape as the Hopcroft–Karp classic test: matching size 4,
        // so cover size must also be 4 by König's theorem.
        let left_adj = vec![vec![0, 3], vec![0, 1], vec![1, 2], vec![2]];
        let (l, r) = min_vertex_cover(&left_adj, 4);
        let (m_size, _, _) = hopcroft_karp(&left_adj, 4);
        assert_eq!(l.len() + r.len(), m_size);
        assert_eq!(l.len() + r.len(), 4);
        assert_is_cover(&left_adj, 4, &l, &r);
    }

    #[test]
    fn isolated_vertices_excluded() {
        // Isolated left vertex 1 and isolated right vertex 2 must not appear
        // in any minimum cover.
        let left_adj = vec![vec![0], vec![], vec![1]];
        let (l, r) = min_vertex_cover(&left_adj, 3);
        assert_eq!(l.len() + r.len(), 2);
        assert!(!l.contains(&1));
        assert!(!r.contains(&2));
        assert_is_cover(&left_adj, 3, &l, &r);
    }

    #[test]
    fn cover_is_sorted_and_unique() {
        let left_adj = vec![vec![0, 1, 2], vec![0, 1, 2], vec![0, 1, 2]];
        let (l, r) = min_vertex_cover(&left_adj, 3);
        for w in l.windows(2) {
            assert!(w[0] < w[1]);
        }
        for w in r.windows(2) {
            assert!(w[0] < w[1]);
        }
        assert_is_cover(&left_adj, 3, &l, &r);
    }

    /// Build a random bipartite adjacency list from a deterministic seed.
    fn random_bipartite(n_left: usize, n_right: usize, seed: u64) -> Vec<Vec<usize>> {
        let mut state = seed.wrapping_add(1).wrapping_mul(0x9e37_79b9_7f4a_7c15);
        let mut xorshift = move || -> u64 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let mut g = vec![vec![]; n_left];
        for u in 0..n_left {
            for v in 0..n_right {
                if xorshift() & 1 == 1 {
                    g[u].push(v);
                }
            }
        }
        g
    }

    /// Property test: on small random bipartite graphs (up to 5 vertices per
    /// side), the cover returned by `min_vertex_cover` must (a) actually be
    /// a vertex cover, (b) match the maximum-matching size (König's
    /// theorem), and (c) match the optimum found by brute-force subset
    /// enumeration.
    #[allow(clippy::needless_pass_by_value)]
    #[quickcheck]
    fn matches_brute_force_small(nl: u8, nr: u8, seed: u64) -> bool {
        let n_left = (nl as usize) % 5 + 1;
        let n_right = (nr as usize) % 5 + 1;
        let g = random_bipartite(n_left, n_right, seed);

        let (l, r) = min_vertex_cover(&g, n_right);

        // (a) every edge is covered.
        let mut in_left = vec![false; n_left];
        for &u in &l {
            in_left[u] = true;
        }
        let mut in_right = vec![false; n_right];
        for &v in &r {
            in_right[v] = true;
        }
        for (u, neighbours) in g.iter().enumerate() {
            for &v in neighbours {
                if !in_left[u] && !in_right[v] {
                    return false;
                }
            }
        }

        // (b) cover size equals matching size.
        let (m_size, _, _) = hopcroft_karp(&g, n_right);
        if l.len() + r.len() != m_size {
            return false;
        }

        // (c) cover size equals brute-force optimum.
        let opt = brute_force_min_cover(&g, n_right);
        if l.len() + r.len() != opt {
            return false;
        }

        true
    }
}
