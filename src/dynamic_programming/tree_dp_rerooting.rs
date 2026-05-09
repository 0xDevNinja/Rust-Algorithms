//! Tree DP with **rerooting**.
//!
//! Given an unrooted tree, many quantities of the form
//!
//! ```text
//! f(v) = aggregate over u of g(v, u)
//! ```
//!
//! can be computed for **every** vertex `v` in linear time by combining a
//! downward DP from an arbitrary root with a second pass that "reroots" the
//! tree edge by edge.
//!
//! This module implements the canonical application: the **sum of distances
//! from each vertex** to all other vertices.
//!
//! # Algorithm (sum of distances)
//!
//! Root the tree at `0` and compute, with an iterative post-order DFS:
//! * `sz[v]`  — number of nodes in the subtree of `v`;
//! * `sum[v]` — Σ over `u` in the subtree of `v` of `dist(v, u)`.
//!
//! Then `result[0] = sum[0]`. To shift the root from a parent `p` to its
//! child `c`, every node in `c`'s subtree gets one closer (saving `sz[c]`
//! distance), and every node outside `c`'s subtree gets one farther
//! (adding `n - sz[c]`):
//!
//! ```text
//! result[c] = result[p] + (n - sz[c]) - sz[c]
//!           = result[p] + n - 2 * sz[c]
//! ```
//!
//! A second iterative pre-order DFS propagates this recurrence from the root
//! to all descendants.
//!
//! # Complexity
//! - Time:  O(n) — two linear DFS passes.
//! - Space: O(n) — `sz`, `result`, parent, and an explicit DFS stack.
//!
//! # Preconditions
//! `adj` must encode an undirected **tree**: connected, `n − 1` edges for
//! `n` vertices, every neighbour index in `0..n`. Behaviour on a forest or
//! cyclic graph is unspecified — a cycle will cause incorrect results, and a
//! forest will only sum within the component containing the chosen root.
//!
//! Iterative DFS is used throughout to avoid stack overflow on long paths.

/// Returns `sz[v]`, the number of nodes in the subtree of `v` when the tree
/// is rooted at `root`.
///
/// Uses an iterative post-order DFS so the call stack stays O(1) regardless
/// of tree depth.
///
/// # Panics
/// Panics if `root >= adj.len()` and `adj` is non-empty.
pub fn subtree_sizes(adj: &[Vec<usize>], root: usize) -> Vec<u64> {
    let n = adj.len();
    if n == 0 {
        return Vec::new();
    }
    assert!(root < n, "root index out of bounds");

    let mut parent = vec![usize::MAX; n];
    let mut order = Vec::with_capacity(n);
    // Iterative DFS to record a parent array and a traversal order. The
    // root is marked with itself as parent so that the `parent[v] != usize::MAX`
    // check correctly skips the back-edge in every neighbour scan.
    parent[root] = root;
    let mut stack = Vec::with_capacity(n);
    stack.push(root);
    while let Some(u) = stack.pop() {
        order.push(u);
        for &v in &adj[u] {
            if parent[v] == usize::MAX {
                parent[v] = u;
                stack.push(v);
            }
        }
    }

    // Process nodes in reverse traversal order so children are summed before
    // their parent — a post-order accumulation.
    let mut sz = vec![1_u64; n];
    for &u in order.iter().rev() {
        if u != root {
            sz[parent[u]] += sz[u];
        }
    }
    sz
}

/// For an unrooted tree on `n` vertices, returns `result` where
/// `result[v] = Σ_u dist(v, u)` is the sum of edge-distances from `v` to
/// every other vertex.
///
/// Runs in O(n) total via the rerooting technique.
///
/// Returns an empty vector for an empty tree and `[0]` for a single vertex.
pub fn sum_of_distances(adj: &[Vec<usize>]) -> Vec<u64> {
    let n = adj.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![0];
    }

    // ── Pass 1: iterative DFS to record parent pointers and a top-down order.
    let root = 0_usize;
    let mut parent = vec![usize::MAX; n];
    let mut order = Vec::with_capacity(n);
    parent[root] = root;
    let mut stack = Vec::with_capacity(n);
    stack.push(root);
    while let Some(u) = stack.pop() {
        order.push(u);
        for &v in &adj[u] {
            if parent[v] == usize::MAX && v != root {
                parent[v] = u;
                stack.push(v);
            }
        }
    }

    // ── Pass 2: post-order accumulation of subtree sizes and distance sums
    // from the rooted DP.
    //   sz[v]        = nodes in the subtree of v;
    //   sum_root[v]  = Σ_{u in subtree(v)} dist(v, u).
    // Recurrence (child c of v):
    //   sz[v]       += sz[c]
    //   sum_root[v] += sum_root[c] + sz[c]   (each subtree node moves up by 1)
    let mut sz = vec![1_u64; n];
    let mut sum_root = vec![0_u64; n];
    for &u in order.iter().rev() {
        if u != root {
            let p = parent[u];
            sz[p] += sz[u];
            sum_root[p] += sum_root[u] + sz[u];
        }
    }

    // ── Pass 3: rerooting in top-down order.
    //   result[child] = result[parent] + n - 2 * sz[child]
    let n_u64 = n as u64;
    let mut result = vec![0_u64; n];
    result[root] = sum_root[root];
    // `order` is already a valid top-down traversal (parents before children).
    for &u in &order {
        if u != root {
            let p = parent[u];
            // n is at least 2 here, and sz[u] <= n - 1, so n + (n - 2*sz[u])
            // never underflows when computed as result[p] + n - 2*sz[u].
            result[u] = result[p] + n_u64 - 2 * sz[u];
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::{subtree_sizes, sum_of_distances};
    use std::collections::VecDeque;

    /// Build an undirected adjacency list of size `n` from an edge list.
    fn build(n: usize, edges: &[(usize, usize)]) -> Vec<Vec<usize>> {
        let mut g = vec![vec![]; n];
        for &(u, v) in edges {
            g[u].push(v);
            g[v].push(u);
        }
        g
    }

    /// O(n²) reference: BFS from every vertex and sum distances.
    fn brute_sum_of_distances(adj: &[Vec<usize>]) -> Vec<u64> {
        let n = adj.len();
        let mut out = vec![0_u64; n];
        for src in 0..n {
            let mut dist = vec![u64::MAX; n];
            dist[src] = 0;
            let mut q = VecDeque::from([src]);
            while let Some(u) = q.pop_front() {
                for &v in &adj[u] {
                    if dist[v] == u64::MAX {
                        dist[v] = dist[u] + 1;
                        q.push_back(v);
                    }
                }
            }
            out[src] = dist.iter().filter(|&&d| d != u64::MAX).sum();
        }
        out
    }

    /// Deterministic `XorShift` random-tree builder; each node `i >= 1` picks a
    /// random parent in `0..i`.
    fn random_tree(n: usize, seed: u64) -> Vec<Vec<usize>> {
        if n == 0 {
            return vec![];
        }
        let mut state = seed ^ 0x9e37_79b9_7f4a_7c15;
        let mut xorshift = move || -> u64 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let mut g = vec![vec![]; n];
        for i in 1..n {
            let parent = (xorshift() as usize) % i;
            g[i].push(parent);
            g[parent].push(i);
        }
        g
    }

    #[test]
    fn empty_tree() {
        let g: Vec<Vec<usize>> = vec![];
        assert_eq!(sum_of_distances(&g), Vec::<u64>::new());
    }

    #[test]
    fn single_vertex() {
        let g = build(1, &[]);
        assert_eq!(sum_of_distances(&g), vec![0]);
    }

    #[test]
    fn two_vertices() {
        let g = build(2, &[(0, 1)]);
        assert_eq!(sum_of_distances(&g), vec![1, 1]);
    }

    #[test]
    fn path_five_vertices() {
        // 0 -- 1 -- 2 -- 3 -- 4
        // From 0: 0+1+2+3+4 = 10
        // From 1: 1+0+1+2+3 = 7
        // From 2: 2+1+0+1+2 = 6
        // Symmetric for 3 and 4.
        let g = build(5, &[(0, 1), (1, 2), (2, 3), (3, 4)]);
        assert_eq!(sum_of_distances(&g), vec![10, 7, 6, 7, 10]);
    }

    #[test]
    fn star_centre_zero_four_leaves() {
        // 0 is centre, 1..=4 are leaves.
        // From 0: 1+1+1+1 = 4
        // From any leaf: 1 (centre) + 3 leaves at distance 2 = 1 + 6 = 7
        let g = build(5, &[(0, 1), (0, 2), (0, 3), (0, 4)]);
        assert_eq!(sum_of_distances(&g), vec![4, 7, 7, 7, 7]);
    }

    #[test]
    fn caterpillar_matches_brute() {
        // Spine 0-1-2-3, legs 4@0, 5@1, 6@2, 7@3.
        let g = build(8, &[(0, 1), (1, 2), (2, 3), (0, 4), (1, 5), (2, 6), (3, 7)]);
        assert_eq!(sum_of_distances(&g), brute_sum_of_distances(&g));
    }

    #[test]
    fn random_small_trees_match_brute() {
        for seed in 0u64..200 {
            let n = ((seed % 30) + 1) as usize;
            let g = random_tree(n, seed);
            assert_eq!(
                sum_of_distances(&g),
                brute_sum_of_distances(&g),
                "seed={seed} n={n}"
            );
        }
    }

    #[test]
    fn long_path_does_not_overflow_stack() {
        // 5 000-node path — would overflow a recursive DFS on most systems.
        const N: usize = 5_000;
        let edges: Vec<(usize, usize)> = (0..N - 1).map(|i| (i, i + 1)).collect();
        let g = build(N, &edges);
        let res = sum_of_distances(&g);
        // Sanity check the endpoints: from vertex 0 the sum is 0+1+...+(N-1).
        let expected_endpoint = (N as u64 - 1) * (N as u64) / 2;
        assert_eq!(res[0], expected_endpoint);
        assert_eq!(res[N - 1], expected_endpoint);
    }

    #[test]
    fn subtree_sizes_path() {
        // 0 -- 1 -- 2 -- 3 -- 4 rooted at 0.
        let g = build(5, &[(0, 1), (1, 2), (2, 3), (3, 4)]);
        assert_eq!(subtree_sizes(&g, 0), vec![5, 4, 3, 2, 1]);
    }

    #[test]
    fn subtree_sizes_star_centre_root() {
        let g = build(5, &[(0, 1), (0, 2), (0, 3), (0, 4)]);
        assert_eq!(subtree_sizes(&g, 0), vec![5, 1, 1, 1, 1]);
    }

    #[test]
    fn subtree_sizes_star_leaf_root() {
        // Rooted at leaf 1: 1 -> 0 -> {2, 3, 4}.
        let g = build(5, &[(0, 1), (0, 2), (0, 3), (0, 4)]);
        assert_eq!(subtree_sizes(&g, 1), vec![4, 5, 1, 1, 1]);
    }

    #[test]
    fn subtree_sizes_empty() {
        let g: Vec<Vec<usize>> = vec![];
        assert_eq!(subtree_sizes(&g, 0), Vec::<u64>::new());
    }

    #[test]
    fn subtree_sizes_single() {
        let g = build(1, &[]);
        assert_eq!(subtree_sizes(&g, 0), vec![1]);
    }
}
