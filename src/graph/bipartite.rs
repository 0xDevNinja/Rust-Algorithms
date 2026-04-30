//! Bipartite check / 2-coloring of an undirected graph.
//!
//! A graph is **bipartite** iff its vertex set can be split into two disjoint
//! sets `A` and `B` such that every edge has one endpoint in each set —
//! equivalently, iff it contains no odd-length cycle.
//!
//! # Algorithm
//! BFS-based 2-coloring. For every unvisited node, start a BFS that paints
//! the source `0`, then alternates colors layer by layer. If the BFS ever
//! reaches a neighbour that has already been painted with the same color as
//! the current node, an odd cycle has been found and the graph is not
//! bipartite. Looping over every unvisited node handles disconnected graphs
//! correctly.
//!
//! # Complexity
//! - Time:  O(N + M) — each node and edge is examined a constant number of
//!   times.
//! - Space: O(N) — color and queue arrays proportional to node count.
//!
//! # Preconditions
//! Input is an undirected adjacency list: for every edge `(u, v)`, both
//! `v ∈ graph[u]` and `u ∈ graph[v]` are present. Every neighbour index
//! must lie in `0..graph.len()`. Self-loops are treated as odd cycles of
//! length 1, so any graph containing one returns `None`.

use std::collections::VecDeque;

/// Returns a 2-coloring of `graph` (values `0` or `1` per node) if the graph
/// is bipartite, or `None` if any odd cycle is found.
///
/// `graph[i]` lists the neighbours of node `i`. Disconnected components are
/// each colored independently; the function returns `None` as soon as any
/// component is found non-bipartite.
pub fn bipartite_coloring(graph: &[Vec<usize>]) -> Option<Vec<u8>> {
    let n = graph.len();
    // Sentinel value 2 means "uncolored". Final output only contains 0 / 1.
    let mut color = vec![2u8; n];
    let mut queue = VecDeque::new();
    for start in 0..n {
        if color[start] != 2 {
            continue;
        }
        color[start] = 0;
        queue.clear();
        queue.push_back(start);
        while let Some(u) = queue.pop_front() {
            let next = 1 - color[u];
            for &v in &graph[u] {
                if color[v] == 2 {
                    color[v] = next;
                    queue.push_back(v);
                } else if color[v] == color[u] {
                    return None;
                }
            }
        }
    }
    Some(color)
}

/// Returns `true` iff `graph` is bipartite. Thin wrapper around
/// [`bipartite_coloring`].
pub fn is_bipartite(graph: &[Vec<usize>]) -> bool {
    bipartite_coloring(graph).is_some()
}

#[cfg(test)]
mod tests {
    use super::{bipartite_coloring, is_bipartite};
    use quickcheck_macros::quickcheck;

    // Helper: build an undirected edge list into an adjacency list of size `n`.
    fn build(n: usize, edges: &[(usize, usize)]) -> Vec<Vec<usize>> {
        let mut g = vec![vec![]; n];
        for &(u, v) in edges {
            g[u].push(v);
            if u != v {
                g[v].push(u);
            }
        }
        g
    }

    // Verify that the returned coloring is consistent: every edge connects
    // nodes of different colors, and every value is 0 or 1.
    fn assert_valid_coloring(graph: &[Vec<usize>], color: &[u8]) {
        assert_eq!(graph.len(), color.len());
        for (u, neighbours) in graph.iter().enumerate() {
            assert!(color[u] < 2, "color out of range at node {u}");
            for &v in neighbours {
                assert_ne!(
                    color[u], color[v],
                    "edge ({u}, {v}) has same-colored endpoints"
                );
            }
        }
    }

    // Build a random tree on `n` nodes from a seed using a deterministic
    // XorShift PRNG. Each node i (i >= 1) attaches to a random parent in 0..i.
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
    fn empty_graph_is_bipartite() {
        let g: Vec<Vec<usize>> = vec![];
        let c = bipartite_coloring(&g).expect("empty graph is bipartite");
        assert!(c.is_empty());
        assert!(is_bipartite(&g));
    }

    #[test]
    fn single_node_is_bipartite() {
        let g = build(1, &[]);
        let c = bipartite_coloring(&g).expect("single node is bipartite");
        assert_eq!(c.len(), 1);
        assert!(c[0] < 2);
    }

    #[test]
    fn k2_is_bipartite() {
        // 0 -- 1
        let g = build(2, &[(0, 1)]);
        let c = bipartite_coloring(&g).expect("K2 is bipartite");
        assert_ne!(c[0], c[1]);
        assert_valid_coloring(&g, &c);
    }

    #[test]
    fn path_graph_is_bipartite() {
        // 0 -- 1 -- 2 -- 3 -- 4
        let g = build(5, &[(0, 1), (1, 2), (2, 3), (3, 4)]);
        let c = bipartite_coloring(&g).expect("path is bipartite");
        assert_valid_coloring(&g, &c);
    }

    #[test]
    fn even_cycle_c4_is_bipartite() {
        // 0 -- 1 -- 2 -- 3 -- 0
        let g = build(4, &[(0, 1), (1, 2), (2, 3), (3, 0)]);
        let c = bipartite_coloring(&g).expect("C4 is bipartite");
        assert_valid_coloring(&g, &c);
    }

    #[test]
    fn even_cycle_c6_is_bipartite() {
        let g = build(6, &[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]);
        let c = bipartite_coloring(&g).expect("C6 is bipartite");
        assert_valid_coloring(&g, &c);
    }

    #[test]
    fn odd_cycle_c3_is_not_bipartite() {
        // Triangle: 0 -- 1 -- 2 -- 0
        let g = build(3, &[(0, 1), (1, 2), (2, 0)]);
        assert!(bipartite_coloring(&g).is_none());
        assert!(!is_bipartite(&g));
    }

    #[test]
    fn odd_cycle_c5_is_not_bipartite() {
        let g = build(5, &[(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]);
        assert!(bipartite_coloring(&g).is_none());
    }

    #[test]
    fn k_2_3_is_bipartite() {
        // Left side: {0, 1}; right side: {2, 3, 4}; every left connected to
        // every right.
        let edges = [(0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4)];
        let g = build(5, &edges);
        let c = bipartite_coloring(&g).expect("K_{2,3} is bipartite");
        assert_eq!(c[0], c[1], "left partition shares a color");
        assert_eq!(c[2], c[3], "right partition shares a color");
        assert_eq!(c[3], c[4], "right partition shares a color");
        assert_ne!(c[0], c[2], "partitions have different colors");
        assert_valid_coloring(&g, &c);
    }

    #[test]
    fn disjoint_bipartite_components_return_some() {
        // Component A: 0 -- 1 (bipartite)
        // Component B: 2 -- 3 -- 4 -- 2 (triangle) — but here we keep both
        // bipartite. This test covers two disjoint bipartite components.
        // Component A: 0 -- 1
        // Component B: 2 -- 3 -- 4 -- 5 (path)
        let g = build(6, &[(0, 1), (2, 3), (3, 4), (4, 5)]);
        let c = bipartite_coloring(&g).expect("disjoint bipartite -> Some");
        assert_valid_coloring(&g, &c);
    }

    #[test]
    fn disconnected_with_one_odd_cycle_returns_none() {
        // Component A: 0 -- 1 (bipartite)
        // Component B: 2 -- 3 -- 4 -- 2 (triangle, not bipartite)
        let g = build(5, &[(0, 1), (2, 3), (3, 4), (4, 2)]);
        assert!(bipartite_coloring(&g).is_none());
    }

    #[test]
    fn self_loop_is_not_bipartite() {
        // Single node with a self-loop: odd cycle of length 1.
        let g = build(1, &[(0, 0)]);
        assert!(bipartite_coloring(&g).is_none());
    }

    // Property test: a random tree is always bipartite. Verify Some(_) and
    // that the returned coloring is valid (every edge has differently
    // colored endpoints).
    #[allow(clippy::needless_pass_by_value)]
    #[quickcheck]
    fn random_tree_is_bipartite(n: u8, seed: u64) -> bool {
        let n = ((n as usize) % 30) + 1;
        let g = random_tree(n, seed);
        match bipartite_coloring(&g) {
            None => false,
            Some(color) => {
                if color.len() != g.len() {
                    return false;
                }
                for (u, neighbours) in g.iter().enumerate() {
                    if color[u] >= 2 {
                        return false;
                    }
                    for &v in neighbours {
                        if color[u] == color[v] {
                            return false;
                        }
                    }
                }
                true
            }
        }
    }
}
