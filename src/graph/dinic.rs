//! Dinic's maximum-flow algorithm.
//!
//! Computes the maximum `s`-`t` flow in a directed network with non-negative
//! integer capacities. It is order-of-magnitude faster than Edmonds–Karp on
//! dense graphs.
//!
//! # Algorithm
//! Each phase performs:
//! 1. **BFS from the source** over edges with positive residual capacity,
//!    assigning each reachable vertex its shortest-path *level* (number of
//!    edges from the source). If the sink is unreachable, the current flow is
//!    optimal.
//! 2. **Blocking-flow DFS** along the layered graph (only edges that go from
//!    level `L` to level `L + 1`). A per-vertex "next edge" pointer (`iter`)
//!    advances as edges are saturated, so each edge is examined a constant
//!    number of times per phase.
//!
//! ## Reverse-edge trick
//! Edges are stored in a flat `Vec<Edge>`. Each forward edge at index `i` has
//! a paired reverse edge at index `i ^ 1` (since edges are inserted in pairs)
//! whose `capacity` starts at `0`. Sending `f` units along edge `i` does
//! `edges[i].capacity -= f` and `edges[i ^ 1].capacity += f`. The reverse
//! edge represents the *option to cancel* prior flow, which is precisely what
//! makes residual-network search find the true maximum flow even on graphs
//! where a greedy path is suboptimal.
//!
//! Antiparallel edges (`u -> v` and `v -> u`) are handled correctly because
//! each user-supplied edge gets its *own* paired reverse edge — the two
//! directions never alias.
//!
//! # Complexity
//! - Time:  `O(V^2 · E)` general; `O(E · sqrt(V))` on unit-capacity graphs
//!   (so bipartite matching via Dinic's runs in `O(E · sqrt(V))`).
//! - Space: `O(V + E)` for the adjacency / edge list.
//!
//! # Preconditions
//! `src` and `sink` must be in `0..n`; otherwise `max_flow` panics. Flow is
//! `u64`; total flow must fit in `u64`. Capacities are summed for parallel
//! edges (each is stored as its own pair, so the algorithm naturally treats
//! them as a combined channel).

use std::collections::VecDeque;

/// One half of a residual edge. Edges are stored in pairs: index `2k` is the
/// forward edge, index `2k + 1` is its reverse, so `rev_idx = idx ^ 1`.
#[derive(Copy, Clone, Debug)]
struct Edge {
    to: usize,
    capacity: u64,
    rev_idx: usize,
}

/// A flow network supporting incremental edge insertion and a single-shot
/// `max_flow` query.
///
/// Internally stores all residual edges in a flat `Vec<Edge>` plus a per-node
/// `Vec<usize>` of edge indices, which is the standard idiomatic Rust
/// max-flow layout (no nested adjacency lists, cache-friendly).
#[derive(Clone, Debug)]
pub struct DinicNetwork {
    num_nodes: usize,
    edges: Vec<Edge>,
    adj: Vec<Vec<usize>>,
    level: Vec<i32>,
    iter: Vec<usize>,
}

impl DinicNetwork {
    /// Creates an empty network on `n` nodes labelled `0..n`.
    pub fn new(n: usize) -> Self {
        Self {
            num_nodes: n,
            edges: Vec::new(),
            adj: vec![Vec::new(); n],
            level: vec![-1; n],
            iter: vec![0; n],
        }
    }

    /// Adds a directed edge `from -> to` with the given non-negative
    /// `capacity`. Internally inserts the paired reverse edge with capacity
    /// `0`. Parallel calls add up: two `add_edge(u, v, 3)` calls behave the
    /// same as a single `add_edge(u, v, 6)` for max-flow purposes.
    ///
    /// # Panics
    /// Panics if `from` or `to` is out of range (`>= n`).
    pub fn add_edge(&mut self, from: usize, to: usize, capacity: u64) {
        assert!(
            from < self.num_nodes && to < self.num_nodes,
            "DinicNetwork::add_edge: endpoint out of range"
        );
        let m = self.edges.len();
        self.edges.push(Edge {
            to,
            capacity,
            rev_idx: m + 1,
        });
        self.edges.push(Edge {
            to: from,
            capacity: 0,
            rev_idx: m,
        });
        self.adj[from].push(m);
        self.adj[to].push(m + 1);
    }

    /// Returns the maximum flow value from `src` to `sink`. The network is
    /// mutated: residual capacities reflect the resulting flow assignment, so
    /// the same network should not be reused for a different `(src, sink)`
    /// pair without rebuilding.
    ///
    /// Returns `0` immediately if `src == sink`.
    ///
    /// # Panics
    /// Panics if `src` or `sink` is out of range (`>= n`).
    pub fn max_flow(&mut self, src: usize, sink: usize) -> u64 {
        assert!(
            src < self.num_nodes && sink < self.num_nodes,
            "DinicNetwork::max_flow: endpoint out of range"
        );
        if src == sink {
            return 0;
        }
        let mut total: u64 = 0;
        while self.bfs(src, sink) {
            // Reset the per-vertex edge iterator for the new blocking-flow phase.
            for x in &mut self.iter {
                *x = 0;
            }
            loop {
                let pushed = self.dfs(src, sink, u64::MAX);
                if pushed == 0 {
                    break;
                }
                total = total.saturating_add(pushed);
            }
        }
        total
    }

    /// BFS from `src` over positive-residual edges. Fills `self.level`;
    /// returns `true` iff `sink` is reachable.
    fn bfs(&mut self, src: usize, sink: usize) -> bool {
        for x in &mut self.level {
            *x = -1;
        }
        self.level[src] = 0;
        let mut queue = VecDeque::new();
        queue.push_back(src);
        while let Some(u) = queue.pop_front() {
            for &eid in &self.adj[u] {
                let e = self.edges[eid];
                if e.capacity > 0 && self.level[e.to] < 0 {
                    self.level[e.to] = self.level[u] + 1;
                    queue.push_back(e.to);
                }
            }
        }
        self.level[sink] >= 0
    }

    /// DFS along the layered graph from `u` toward `sink`, pushing up to
    /// `pushed` units of flow. Advances `self.iter[u]` past saturated /
    /// dead-end edges so each edge is touched at most twice per phase.
    fn dfs(&mut self, u: usize, sink: usize, pushed: u64) -> u64 {
        if u == sink {
            return pushed;
        }
        while self.iter[u] < self.adj[u].len() {
            let eid = self.adj[u][self.iter[u]];
            let e = self.edges[eid];
            if e.capacity > 0 && self.level[e.to] == self.level[u] + 1 {
                let d = self.dfs(e.to, sink, pushed.min(e.capacity));
                if d > 0 {
                    self.edges[eid].capacity -= d;
                    let rev = self.edges[eid].rev_idx;
                    self.edges[rev].capacity = self.edges[rev].capacity.saturating_add(d);
                    return d;
                }
            }
            self.iter[u] += 1;
        }
        0
    }
}

#[cfg(test)]
mod tests {
    use super::DinicNetwork;
    use crate::graph::edmonds_karp::{edmonds_karp, Edge as EkEdge};
    use quickcheck_macros::quickcheck;

    #[test]
    fn empty_network() {
        // No edges: max flow is trivially 0 even when sink is reachable in
        // index space.
        let mut g = DinicNetwork::new(2);
        assert_eq!(g.max_flow(0, 1), 0);
    }

    #[test]
    #[should_panic(expected = "endpoint out of range")]
    fn out_of_range_src_panics() {
        let mut g = DinicNetwork::new(2);
        let _ = g.max_flow(5, 1);
    }

    #[test]
    fn source_equals_sink() {
        let mut g = DinicNetwork::new(2);
        g.add_edge(0, 1, 5);
        assert_eq!(g.max_flow(0, 0), 0);
    }

    #[test]
    fn single_edge() {
        let mut g = DinicNetwork::new(2);
        g.add_edge(0, 1, 7);
        assert_eq!(g.max_flow(0, 1), 7);
    }

    #[test]
    fn unreachable_sink() {
        let mut g = DinicNetwork::new(3);
        g.add_edge(0, 1, 5);
        assert_eq!(g.max_flow(0, 2), 0);
    }

    #[test]
    fn classic_clrs_example() {
        // 6-node CLRS network; well-known max flow = 23.
        let mut g = DinicNetwork::new(6);
        g.add_edge(0, 1, 16);
        g.add_edge(0, 2, 13);
        g.add_edge(1, 2, 10);
        g.add_edge(2, 1, 4);
        g.add_edge(1, 3, 12);
        g.add_edge(2, 4, 14);
        g.add_edge(3, 2, 9);
        g.add_edge(3, 5, 20);
        g.add_edge(4, 3, 7);
        g.add_edge(4, 5, 4);
        assert_eq!(g.max_flow(0, 5), 23);
    }

    #[test]
    fn parallel_edges_sum() {
        let mut g = DinicNetwork::new(2);
        g.add_edge(0, 1, 3);
        g.add_edge(0, 1, 4);
        assert_eq!(g.max_flow(0, 1), 7);
    }

    #[test]
    fn antiparallel_edges_have_independent_reverse() {
        // u -> v capacity 5, v -> u capacity 5: each must keep its own paired
        // reverse edge; otherwise the two directions would alias and the max
        // flow from 0 to 2 would be wrong.
        // 0 -> 1 (5), 1 -> 0 (5), 1 -> 2 (5).
        let mut g = DinicNetwork::new(3);
        g.add_edge(0, 1, 5);
        g.add_edge(1, 0, 5);
        g.add_edge(1, 2, 5);
        assert_eq!(g.max_flow(0, 2), 5);
    }

    #[test]
    fn bipartite_matching_reduction() {
        // Left side: nodes 1..=3. Right side: nodes 4..=6. Source 0, sink 7.
        // Edges: 1-4, 1-5, 2-5, 3-5, 3-6. Max matching size = 3
        // (e.g. 1-4, 2-5, 3-6).
        let n = 8;
        let mut g = DinicNetwork::new(n);
        for l in 1..=3 {
            g.add_edge(0, l, 1);
        }
        for r in 4..=6 {
            g.add_edge(r, 7, 1);
        }
        let pairs = [(1, 4), (1, 5), (2, 5), (3, 5), (3, 6)];
        for (l, r) in pairs {
            g.add_edge(l, r, 1);
        }
        assert_eq!(g.max_flow(0, 7), 3);
    }

    /// Decode a deterministic pseudo-random graph from the `QuickCheck` inputs
    /// and return both the Dinic network and an Edmonds–Karp edge list for
    /// the same graph.
    fn build_random(
        n_seed: u8,
        mask: u64,
        weight_seed: u64,
    ) -> (DinicNetwork, Vec<EkEdge>, usize, usize, usize) {
        let n = ((n_seed as usize) % 5) + 2; // 2..=6
        let mut g = DinicNetwork::new(n);
        let mut ek = Vec::new();
        // Use bit positions of `mask` to decide which directed edges exist.
        // Use `weight_seed` rotated per edge to choose capacities in 0..=5.
        let mut bit = 0;
        let mut w = weight_seed;
        for u in 0..n {
            for v in 0..n {
                if u == v {
                    continue;
                }
                let present = (mask >> (bit % 64)) & 1 == 1;
                bit += 1;
                if !present {
                    continue;
                }
                let cap = w % 6;
                w = w.rotate_left(7).wrapping_add(0x9E37_79B9_7F4A_7C15);
                if cap == 0 {
                    continue;
                }
                g.add_edge(u, v, cap);
                ek.push(EkEdge {
                    from: u,
                    to: v,
                    capacity: cap,
                });
            }
        }
        let src = (weight_seed as usize) % n;
        let mut sink = ((weight_seed >> 8) as usize) % n;
        if sink == src {
            sink = (sink + 1) % n;
        }
        (g, ek, n, src, sink)
    }

    #[quickcheck]
    fn quickcheck_matches_edmonds_karp(n_seed: u8, mask: u64, weight_seed: u64) -> bool {
        let (mut dinic, ek, n, src, sink) = build_random(n_seed, mask, weight_seed);
        let dinic_flow = dinic.max_flow(src, sink);
        let ek_flow = edmonds_karp(n, &ek, src, sink);
        dinic_flow == ek_flow
    }
}
