//! Hopcroft–Karp maximum cardinality bipartite matching.
//!
//! Given a bipartite graph with left vertex set `L = 0..n_left` and right
//! vertex set `R = 0..n_right`, finds a maximum-cardinality matching: a set
//! of edges with no two sharing an endpoint, of largest possible size.
//!
//! # Algorithm
//! Repeatedly alternates two phases until no augmenting path exists:
//!   1. **BFS** from all currently unmatched left vertices, building a layered
//!      graph where edges alternate between unmatched (L → R) and matched
//!      (R → L). The BFS records, for each right vertex, the shortest distance
//!      to it in this layered graph and stops at the first layer that contains
//!      an unmatched right vertex.
//!   2. **DFS** from each unmatched left vertex, finding vertex-disjoint
//!      augmenting paths along the layered graph and flipping the matching
//!      along every path it finds.
//!
//! Each iteration either grows the matching by at least one or terminates,
//! and the length of the shortest augmenting path strictly increases between
//! phases. A standard analysis shows the algorithm finishes in O(√V) phases,
//! so the overall complexity is **O(E · √V)**.
//!
//! # Preconditions
//! - `left_adj.len() == n_left`. Each `left_adj[u]` lists right-vertex
//!   indices in `0..n_right`. Right indices outside this range are
//!   **undefined behaviour** (they will panic on out-of-bounds access).
//! - The graph is bipartite with edges only between L and R; there are no
//!   left-left or right-right edges.
//!
//! # Complexity
//! - Time:  O(E · √V) where `V = n_left + n_right` and `E` is the number
//!   of edges.
//! - Space: O(V + E).

use std::collections::VecDeque;

const INF: usize = usize::MAX;

/// Returns `(matching_size, match_l, match_r)` for the bipartite graph
/// described by `left_adj` (left side) and `n_right` (size of the right side).
///
/// `match_l[u] = Some(v)` means left vertex `u` is matched to right vertex
/// `v`, and symmetrically `match_r[v] = Some(u)`. Unmatched vertices are
/// `None`. The returned `matching_size` equals the number of `Some` entries
/// in either array.
pub fn hopcroft_karp(
    left_adj: &[Vec<usize>],
    n_right: usize,
) -> (usize, Vec<Option<usize>>, Vec<Option<usize>>) {
    let n_left = left_adj.len();
    if n_left == 0 {
        return (0, Vec::new(), vec![None; n_right]);
    }

    let mut match_l: Vec<Option<usize>> = vec![None; n_left];
    let mut match_r: Vec<Option<usize>> = vec![None; n_right];
    let mut dist: Vec<usize> = vec![INF; n_left];

    let mut matching_size = 0;
    while bfs(left_adj, &match_l, &match_r, &mut dist) {
        for u in 0..n_left {
            if match_l[u].is_none() && dfs(u, left_adj, &mut match_l, &mut match_r, &mut dist) {
                matching_size += 1;
            }
        }
    }
    (matching_size, match_l, match_r)
}

/// Builds the layered graph by BFS over unmatched left vertices.
/// Returns `true` iff at least one augmenting path was found (i.e. an
/// unmatched right vertex is reachable in the layered graph).
fn bfs(
    left_adj: &[Vec<usize>],
    match_l: &[Option<usize>],
    match_r: &[Option<usize>],
    dist: &mut [usize],
) -> bool {
    let mut queue: VecDeque<usize> = VecDeque::new();
    for u in 0..left_adj.len() {
        if match_l[u].is_none() {
            dist[u] = 0;
            queue.push_back(u);
        } else {
            dist[u] = INF;
        }
    }
    let mut found = false;
    while let Some(u) = queue.pop_front() {
        let du = dist[u];
        for &v in &left_adj[u] {
            // Walk one step right (u -> v), then if v is matched, follow the
            // matching edge back to a left vertex `pair = match_r[v]` and
            // enqueue it at layer du + 1.
            match match_r[v] {
                None => {
                    // Unmatched right vertex: an augmenting path exists.
                    found = true;
                }
                Some(pair) => {
                    if dist[pair] == INF {
                        dist[pair] = du + 1;
                        queue.push_back(pair);
                    }
                }
            }
        }
    }
    found
}

/// Tries to find an augmenting path starting from left vertex `u` along the
/// layered graph constructed by [`bfs`]. If found, flips the matching along
/// the path and returns `true`.
fn dfs(
    u: usize,
    left_adj: &[Vec<usize>],
    match_l: &mut [Option<usize>],
    match_r: &mut [Option<usize>],
    dist: &mut [usize],
) -> bool {
    for i in 0..left_adj[u].len() {
        let v = left_adj[u][i];
        let ok = match_r[v].is_none_or(|pair| {
            dist[pair] == dist[u].wrapping_add(1) && dfs(pair, left_adj, match_l, match_r, dist)
        });
        if ok {
            match_l[u] = Some(v);
            match_r[v] = Some(u);
            return true;
        }
    }
    // Mark `u` exhausted so other DFS calls in this phase skip it.
    dist[u] = INF;
    false
}

#[cfg(test)]
mod tests {
    use super::hopcroft_karp;
    use quickcheck_macros::quickcheck;

    /// Asserts the returned matching is internally consistent.
    fn assert_consistent(
        left_adj: &[Vec<usize>],
        n_right: usize,
        size: usize,
        match_l: &[Option<usize>],
        match_r: &[Option<usize>],
    ) {
        assert_eq!(match_l.len(), left_adj.len());
        assert_eq!(match_r.len(), n_right);
        let count_l = match_l.iter().filter(|m| m.is_some()).count();
        let count_r = match_r.iter().filter(|m| m.is_some()).count();
        assert_eq!(count_l, size);
        assert_eq!(count_r, size);
        for (u, m) in match_l.iter().enumerate() {
            if let Some(v) = *m {
                assert!(
                    left_adj[u].contains(&v),
                    "matched edge ({u},{v}) not in graph"
                );
                assert_eq!(match_r[v], Some(u), "match_r[{v}] != Some({u})");
            }
        }
        for (v, m) in match_r.iter().enumerate() {
            if let Some(u) = *m {
                assert_eq!(match_l[u], Some(v), "match_l[{u}] != Some({v})");
            }
        }
    }

    /// Reference maximum-cardinality bipartite matching via Kuhn's
    /// algorithm (simple DFS augmenting paths). O(V·E). Independent of the
    /// Hopcroft–Karp implementation under test, so disagreement signals a
    /// real bug.
    fn kuhn_matching(left_adj: &[Vec<usize>], n_right: usize) -> usize {
        fn try_kuhn(
            u: usize,
            left_adj: &[Vec<usize>],
            visited: &mut [bool],
            match_r: &mut [Option<usize>],
        ) -> bool {
            for &v in &left_adj[u] {
                if visited[v] {
                    continue;
                }
                visited[v] = true;
                if match_r[v].is_none() || try_kuhn(match_r[v].unwrap(), left_adj, visited, match_r)
                {
                    match_r[v] = Some(u);
                    return true;
                }
            }
            false
        }
        let mut match_r: Vec<Option<usize>> = vec![None; n_right];
        let mut size = 0;
        for u in 0..left_adj.len() {
            let mut visited = vec![false; n_right];
            if try_kuhn(u, left_adj, &mut visited, &mut match_r) {
                size += 1;
            }
        }
        size
    }

    /// Subset-enumeration brute force used for the small unit cases
    /// alongside Kuhn's algorithm; restricted to graphs with at most
    /// 20 edges so `2^m` stays tractable.
    fn brute_force_matching(left_adj: &[Vec<usize>], n_right: usize) -> Option<usize> {
        let mut edges: Vec<(usize, usize)> = Vec::new();
        for (u, neigh) in left_adj.iter().enumerate() {
            for &v in neigh {
                if v < n_right {
                    edges.push((u, v));
                }
            }
        }
        let m = edges.len();
        if m > 20 {
            return None;
        }
        let mut best = 0;
        for mask in 0u32..(1u32 << m) {
            let mut used_l = vec![false; left_adj.len()];
            let mut used_r = vec![false; n_right];
            let mut ok = true;
            let mut size = 0;
            for (i, &(u, v)) in edges.iter().enumerate() {
                if (mask >> i) & 1 == 1 {
                    if used_l[u] || used_r[v] {
                        ok = false;
                        break;
                    }
                    used_l[u] = true;
                    used_r[v] = true;
                    size += 1;
                }
            }
            if ok && size > best {
                best = size;
            }
        }
        Some(best)
    }

    #[test]
    fn empty_graph() {
        let left_adj: Vec<Vec<usize>> = vec![];
        let (size, match_l, match_r) = hopcroft_karp(&left_adj, 0);
        assert_eq!(size, 0);
        assert!(match_l.is_empty());
        assert!(match_r.is_empty());
    }

    #[test]
    fn empty_left_nonempty_right() {
        let left_adj: Vec<Vec<usize>> = vec![];
        let (size, match_l, match_r) = hopcroft_karp(&left_adj, 3);
        assert_eq!(size, 0);
        assert!(match_l.is_empty());
        assert_eq!(match_r, vec![None, None, None]);
    }

    #[test]
    fn no_edges() {
        let left_adj = vec![vec![], vec![], vec![]];
        let (size, match_l, match_r) = hopcroft_karp(&left_adj, 3);
        assert_eq!(size, 0);
        assert!(match_l.iter().all(Option::is_none));
        assert!(match_r.iter().all(Option::is_none));
    }

    #[test]
    fn single_edge() {
        let left_adj = vec![vec![0]];
        let (size, match_l, match_r) = hopcroft_karp(&left_adj, 1);
        assert_eq!(size, 1);
        assert_eq!(match_l, vec![Some(0)]);
        assert_eq!(match_r, vec![Some(0)]);
    }

    #[test]
    fn k_2_2_perfect_matching() {
        // Complete bipartite K_{2,2}: matching size 2.
        let left_adj = vec![vec![0, 1], vec![0, 1]];
        let (size, match_l, match_r) = hopcroft_karp(&left_adj, 2);
        assert_eq!(size, 2);
        assert_consistent(&left_adj, 2, size, &match_l, &match_r);
    }

    #[test]
    fn k_2_3_matching_size_two() {
        // K_{2,3}: 2 left vertices, 3 right vertices; max matching = 2.
        let left_adj = vec![vec![0, 1, 2], vec![0, 1, 2]];
        let (size, match_l, match_r) = hopcroft_karp(&left_adj, 3);
        assert_eq!(size, 2);
        assert_consistent(&left_adj, 3, size, &match_l, &match_r);
    }

    #[test]
    fn k_3_3_perfect_matching() {
        let left_adj = vec![vec![0, 1, 2], vec![0, 1, 2], vec![0, 1, 2]];
        let (size, match_l, match_r) = hopcroft_karp(&left_adj, 3);
        assert_eq!(size, 3);
        assert_consistent(&left_adj, 3, size, &match_l, &match_r);
    }

    #[test]
    fn classic_augmenting_path_example() {
        // 4 left, 4 right. The greedy match {0-0, 1-1, 2-2} blocks left vertex
        // 3 (only neighbour is 2), so an augmenting path must be found:
        //   3 - 2 = 2 - 0 = 0 - 3   (= are matched edges)
        // Optimal matching size = 4.
        let left_adj = vec![vec![0, 3], vec![0, 1], vec![1, 2], vec![2]];
        let (size, match_l, match_r) = hopcroft_karp(&left_adj, 4);
        assert_eq!(size, 4);
        assert_consistent(&left_adj, 4, size, &match_l, &match_r);
    }

    #[test]
    fn isolated_left_vertices() {
        // Left vertices 1 and 3 have no edges; only vertex 0 and 2 can match.
        let left_adj = vec![vec![0], vec![], vec![1], vec![]];
        let (size, match_l, match_r) = hopcroft_karp(&left_adj, 3);
        assert_eq!(size, 2);
        assert_eq!(match_l[1], None);
        assert_eq!(match_l[3], None);
        assert_consistent(&left_adj, 3, size, &match_l, &match_r);
    }

    #[test]
    fn isolated_right_vertices() {
        // Right vertex 2 has no incoming edges.
        let left_adj = vec![vec![0], vec![1]];
        let (size, match_l, match_r) = hopcroft_karp(&left_adj, 3);
        assert_eq!(size, 2);
        assert_eq!(match_r[2], None);
        assert_consistent(&left_adj, 3, size, &match_l, &match_r);
    }

    #[test]
    fn duplicate_edges_are_handled() {
        // Parallel edges in the adjacency list must not break the algorithm.
        let left_adj = vec![vec![0, 0, 1], vec![1, 1]];
        let (size, match_l, match_r) = hopcroft_karp(&left_adj, 2);
        assert_eq!(size, 2);
        assert_consistent(&left_adj, 2, size, &match_l, &match_r);
    }

    /// Build an adjacency list from a deterministic seed.
    /// Result has `n_left` left vertices and `n_right` right vertices, with
    /// each potential edge present independently with probability ~50%.
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

    /// Property test: Hopcroft–Karp must agree with Kuhn's algorithm on
    /// small random bipartite graphs (and with full subset enumeration when
    /// the graph is tiny enough for it to be feasible).
    #[allow(clippy::needless_pass_by_value)]
    #[quickcheck]
    fn matches_reference_small(nl: u8, nr: u8, seed: u64) -> bool {
        let n_left = (nl as usize) % 6 + 1;
        let n_right = (nr as usize) % 6 + 1;
        let g = random_bipartite(n_left, n_right, seed);
        let (size, match_l, match_r) = hopcroft_karp(&g, n_right);
        if size != kuhn_matching(&g, n_right) {
            return false;
        }
        if let Some(brute) = brute_force_matching(&g, n_right) {
            if size != brute {
                return false;
            }
        }
        if match_l.len() != n_left || match_r.len() != n_right {
            return false;
        }
        let mut count = 0;
        for (u, m) in match_l.iter().enumerate() {
            if let Some(v) = *m {
                if !g[u].contains(&v) {
                    return false;
                }
                if match_r[v] != Some(u) {
                    return false;
                }
                count += 1;
            }
        }
        count == size
    }
}
