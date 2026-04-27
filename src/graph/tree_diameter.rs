//! Tree diameter via two BFS passes.
//!
//! The **diameter** of a tree is the number of edges on the longest path
//! between any two vertices.
//!
//! # Algorithm
//! 1. From an arbitrary start node, BFS to find the farthest reachable node `u`.
//! 2. From `u`, BFS again to find the farthest reachable node `v`.
//! 3. `dist(u, v)` is the diameter.
//!
//! # Complexity
//! - Time:  O(N) — two linear BFS passes.
//! - Space: O(N) — distance and queue arrays proportional to node count.
//!
//! # Preconditions
//! The input must be a **tree**: undirected, connected, with exactly N − 1
//! edges for N nodes, and every neighbour index in `graph[i]` must be in
//! `0..N`.
//!
//! Out-of-precondition behaviour:
//! - **Neighbour index `≥ N`** — panics with index-out-of-bounds.
//! - **Disconnected forest** — returns the diameter of the component that
//!   contains node 0, not the global maximum across components.
//! - **Cyclic graph** — terminates and returns a finite value, but the
//!   value is not a meaningful "diameter" in the tree sense.
//! - **Self-loops / duplicate edges** — handled by the visited check, so
//!   they do not affect the result on otherwise-tree-like input.

use std::collections::VecDeque;

/// Computes the diameter (longest path in edges) of an undirected tree given
/// as an adjacency list.
///
/// `graph[i]` contains the neighbours of node `i`. Returns `0` for graphs
/// with fewer than 2 nodes.
pub fn tree_diameter(graph: &[Vec<usize>]) -> usize {
    let n = graph.len();
    if n < 2 {
        return 0;
    }
    // First BFS from node 0 to find one endpoint of the diameter.
    let (u, _) = farthest(graph, 0);
    // Second BFS from that endpoint; the distance to the new farthest node is
    // the diameter.
    let (_, diameter) = farthest(graph, u);
    diameter
}

/// Returns `(farthest_node, distance)` from `start` via BFS.
fn farthest(graph: &[Vec<usize>], start: usize) -> (usize, usize) {
    let n = graph.len();
    let mut dist = vec![usize::MAX; n];
    dist[start] = 0;
    let mut queue = VecDeque::from([start]);
    let mut farthest_node = start;
    let mut farthest_dist = 0;
    while let Some(u) = queue.pop_front() {
        for &v in &graph[u] {
            if dist[v] == usize::MAX {
                dist[v] = dist[u] + 1;
                queue.push_back(v);
                if dist[v] > farthest_dist {
                    farthest_dist = dist[v];
                    farthest_node = v;
                }
            }
        }
    }
    (farthest_node, farthest_dist)
}

#[cfg(test)]
mod tests {
    use super::tree_diameter;
    use quickcheck_macros::quickcheck;

    // Helper: build an undirected edge list into an adjacency list of size `n`.
    fn build(n: usize, edges: &[(usize, usize)]) -> Vec<Vec<usize>> {
        let mut g = vec![vec![]; n];
        for &(u, v) in edges {
            g[u].push(v);
            g[v].push(u);
        }
        g
    }

    // Brute-force diameter via all-pairs BFS; only correct on connected graphs.
    fn brute_force_diameter(graph: &[Vec<usize>]) -> usize {
        let n = graph.len();
        if n < 2 {
            return 0;
        }
        let mut best = 0;
        for src in 0..n {
            let mut dist = vec![usize::MAX; n];
            dist[src] = 0;
            let mut queue = std::collections::VecDeque::from([src]);
            while let Some(u) = queue.pop_front() {
                for &v in &graph[u] {
                    if dist[v] == usize::MAX {
                        dist[v] = dist[u] + 1;
                        queue.push_back(v);
                        if dist[v] > best {
                            best = dist[v];
                        }
                    }
                }
            }
        }
        best
    }

    // Build a random tree on `n` nodes from a seed using a deterministic XorShift
    // PRNG. Each node i (i >= 1) is connected to a random parent in 0..i.
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
    fn empty_graph() {
        let g: Vec<Vec<usize>> = vec![];
        assert_eq!(tree_diameter(&g), 0);
    }

    #[test]
    fn single_node() {
        let g = build(1, &[]);
        assert_eq!(tree_diameter(&g), 0);
    }

    #[test]
    fn two_nodes_one_edge() {
        // 0 -- 1  →  diameter = 1
        let g = build(2, &[(0, 1)]);
        assert_eq!(tree_diameter(&g), 1);
    }

    #[test]
    fn path_five_nodes() {
        // 0 -- 1 -- 2 -- 3 -- 4  →  diameter = 4
        let g = build(5, &[(0, 1), (1, 2), (2, 3), (3, 4)]);
        assert_eq!(tree_diameter(&g), 4);
    }

    #[test]
    fn star_four_leaves() {
        // Centre = 0, leaves = 1, 2, 3, 4  →  diameter = 2
        let g = build(5, &[(0, 1), (0, 2), (0, 3), (0, 4)]);
        assert_eq!(tree_diameter(&g), 2);
    }

    #[test]
    fn branching_tree() {
        // Tree structure:
        //        0
        //       / \
        //      1   2
        //     / \
        //    3   4
        //   /
        //  5
        // Longest path: 5 → 3 → 1 → 0 → 2, length 4.
        let g = build(6, &[(0, 1), (0, 2), (1, 3), (1, 4), (3, 5)]);
        assert_eq!(tree_diameter(&g), 4);
    }

    #[test]
    fn linear_chain_diameter_equals_n_minus_one() {
        // n-node path graph always has diameter n-1.
        for n in 2..=10 {
            let edges: Vec<(usize, usize)> = (0..n - 1).map(|i| (i, i + 1)).collect();
            let g = build(n, &edges);
            assert_eq!(tree_diameter(&g), n - 1, "n={n}");
        }
    }

    #[test]
    fn caterpillar_tree() {
        // Spine: 0-1-2-3; legs hanging off each spine node.
        // 4 hangs off 0, 5 off 1, 6 off 2, 7 off 3.
        // Longest path: 4 → 0 → 1 → 2 → 3 → 7, length 5.
        let g = build(8, &[(0, 1), (1, 2), (2, 3), (0, 4), (1, 5), (2, 6), (3, 7)]);
        assert_eq!(tree_diameter(&g), 5);
    }

    // --- new tests below ---

    // Node 0 is the middle node of a 5-node path: 1-2-0-3-4.
    // Diameter is still 4 (path 1→2→0→3→4).
    #[test]
    fn node_zero_interior_to_path() {
        // Physical path: 1 -- 2 -- 0 -- 3 -- 4
        let g = build(5, &[(1, 2), (2, 0), (0, 3), (3, 4)]);
        assert_eq!(tree_diameter(&g), 4);
    }

    // 10 000-node path: diameter must be 9 999. Smoke / regression for BFS
    // scalability (no stack overflow, no allocation failure).
    #[test]
    fn large_path_smoke() {
        const N: usize = 10_000;
        let edges: Vec<(usize, usize)> = (0..N - 1).map(|i| (i, i + 1)).collect();
        let g = build(N, &edges);
        assert_eq!(tree_diameter(&g), N - 1);
    }

    // Two disjoint paths: 0-1-2 and 3-4-5-6.
    // The function is documented to return the diameter of node 0's component,
    // which is 2 (the path 0-1-2), not 3 (the longer component 3-4-5-6).
    #[test]
    fn disconnected_forest_returns_component_zero_diameter() {
        // Component of node 0: 0 -- 1 -- 2  (diameter 2)
        // Component of node 3: 3 -- 4 -- 5 -- 6  (diameter 3)
        let g = build(7, &[(0, 1), (1, 2), (3, 4), (4, 5), (5, 6)]);
        assert_eq!(tree_diameter(&g), 2);
    }

    // Property test: for 200 deterministically seeded random trees of up to 30
    // nodes, tree_diameter must agree with the all-pairs BFS reference.
    #[test]
    fn random_trees_match_brute_force() {
        for seed in 0u64..200 {
            // n ranges from 1 to 30, cycling through the seed space.
            let n = ((seed % 30) + 1) as usize;
            let g = random_tree(n, seed);
            assert_eq!(
                tree_diameter(&g),
                brute_force_diameter(&g),
                "seed={seed} n={n}"
            );
        }
    }

    // quickcheck variant: generate n as u8, clamp to [1, 30], build a random
    // tree and compare against the brute-force reference.
    #[allow(clippy::needless_pass_by_value)]
    #[quickcheck]
    fn quickcheck_random_tree_matches_brute_force(n: u8, seed: u64) -> bool {
        let n = ((n as usize) % 30) + 1;
        let g = random_tree(n, seed);
        tree_diameter(&g) == brute_force_diameter(&g)
    }
}
