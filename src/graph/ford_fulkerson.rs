//! Ford–Fulkerson maximum-flow with DFS-found augmenting paths.
//!
//! Computes the maximum `s`-`t` flow in a directed network with non-negative
//! integer capacities. This is the classic textbook formulation: repeatedly
//! find *any* augmenting path from source to sink in the residual network
//! (here via DFS), saturate the bottleneck, and stop when no augmenting path
//! remains.
//!
//! # Algorithm
//! Each round:
//! 1. Run a DFS from `src` in the residual network, recording the residual
//!    edge taken to reach each visited node.
//! 2. If `sink` was reached, walk back along the recorded edges to find the
//!    bottleneck residual capacity and add that much flow along the path
//!    (decrementing forward residuals and incrementing paired reverse
//!    residuals).
//! 3. Otherwise, the current flow is optimal.
//!
//! ## Reverse-edge trick
//! Edges are stored in a flat `Vec<Edge>` in pairs: index `2k` is the forward
//! residual edge, index `2k + 1` its reverse, so `rev_idx = idx ^ 1`. The
//! reverse edge encodes the option to *cancel* previously routed flow, which
//! is what allows residual-network search to find the true maximum flow even
//! on graphs where any single greedy path is suboptimal.
//!
//! # Complexity
//! - Time:  `O(E · max_flow)` for integer capacities — each DFS is `O(V + E)`,
//!   and each augmenting path increases the integer flow by at least `1`.
//! - Space: `O(V + E)`.
//!
//! # Termination caveat
//! With *integer* capacities the algorithm always terminates because each
//! augmentation increases the flow by at least one unit, bounded by the value
//! of the maximum flow. With *irrational* capacities Ford–Fulkerson can fail
//! to terminate at all (and may converge to a value strictly less than the
//! true max flow); use BFS-based Edmonds–Karp or Dinic's for guaranteed
//! polynomial-time termination on arbitrary real capacities. Because this
//! implementation uses `u64` capacities, that pathology cannot arise here.
//!
//! # Preconditions
//! `src` and `sink` must be in `0..n`; otherwise `max_flow` panics. Flow is
//! `u64`; total flow must fit in `u64`. Parallel edges between the same pair
//! of nodes are stored as independent residual pairs and behave as a single
//! channel whose capacity is their sum.

/// One half of a residual edge. Edges are inserted in pairs (forward, then
/// reverse), so the reverse of an edge at index `i` lives at `i ^ 1`.
#[derive(Copy, Clone, Debug)]
struct Edge {
    to: usize,
    capacity: u64,
    rev_idx: usize,
}

/// A flow network supporting incremental edge insertion and a single
/// `max_flow` query.
#[derive(Clone, Debug)]
pub struct FordFulkerson {
    num_nodes: usize,
    edges: Vec<Edge>,
    adj: Vec<Vec<usize>>,
    visited: Vec<bool>,
    parent_edge: Vec<usize>,
}

impl FordFulkerson {
    /// Creates an empty network on `n` nodes labelled `0..n`.
    pub fn new(n: usize) -> Self {
        Self {
            num_nodes: n,
            edges: Vec::new(),
            adj: vec![Vec::new(); n],
            visited: vec![false; n],
            parent_edge: vec![usize::MAX; n],
        }
    }

    /// Adds a directed edge `u -> v` with the given non-negative `cap`.
    /// Internally inserts the paired reverse edge with capacity `0`. Multiple
    /// calls on the same `(u, v)` are treated as parallel edges and their
    /// capacities sum naturally during the max-flow computation.
    ///
    /// # Panics
    /// Panics if `u` or `v` is out of range (`>= n`).
    pub fn add_edge(&mut self, u: usize, v: usize, cap: u64) {
        assert!(
            u < self.num_nodes && v < self.num_nodes,
            "FordFulkerson::add_edge: endpoint out of range"
        );
        let m = self.edges.len();
        self.edges.push(Edge {
            to: v,
            capacity: cap,
            rev_idx: m + 1,
        });
        self.edges.push(Edge {
            to: u,
            capacity: 0,
            rev_idx: m,
        });
        self.adj[u].push(m);
        self.adj[v].push(m + 1);
    }

    /// Returns the maximum flow value from `s` to `t`. The network is
    /// mutated: residual capacities reflect the resulting flow assignment, so
    /// the same network should not be reused for a different `(s, t)` pair
    /// without rebuilding.
    ///
    /// Returns `0` immediately if `s == t`.
    ///
    /// # Panics
    /// Panics if `s` or `t` is out of range (`>= n`).
    pub fn max_flow(&mut self, s: usize, t: usize) -> u64 {
        assert!(
            s < self.num_nodes && t < self.num_nodes,
            "FordFulkerson::max_flow: endpoint out of range"
        );
        if s == t {
            return 0;
        }
        let mut total: u64 = 0;
        loop {
            for x in &mut self.visited {
                *x = false;
            }
            for x in &mut self.parent_edge {
                *x = usize::MAX;
            }
            self.visited[s] = true;
            if !self.dfs(s, t) {
                break;
            }
            // Walk back from `t` to `s` to find the bottleneck.
            let mut bottleneck = u64::MAX;
            let mut v = t;
            while v != s {
                let eid = self.parent_edge[v];
                bottleneck = bottleneck.min(self.edges[eid].capacity);
                // Predecessor is the tail of edge `eid`, i.e. the head of its
                // reverse edge.
                v = self.edges[self.edges[eid].rev_idx].to;
            }
            // Apply the augmentation.
            let mut v = t;
            while v != s {
                let eid = self.parent_edge[v];
                self.edges[eid].capacity -= bottleneck;
                let rev = self.edges[eid].rev_idx;
                self.edges[rev].capacity = self.edges[rev].capacity.saturating_add(bottleneck);
                v = self.edges[rev].to;
            }
            total = total.saturating_add(bottleneck);
        }
        total
    }

    /// Iterative DFS over residual edges. Records the edge id used to reach
    /// each visited node in `self.parent_edge`. Returns `true` iff `sink` is
    /// reachable from `src`.
    fn dfs(&mut self, src: usize, sink: usize) -> bool {
        let mut stack = vec![src];
        while let Some(u) = stack.pop() {
            if u == sink {
                return true;
            }
            for &eid in &self.adj[u] {
                let e = self.edges[eid];
                if e.capacity > 0 && !self.visited[e.to] {
                    self.visited[e.to] = true;
                    self.parent_edge[e.to] = eid;
                    if e.to == sink {
                        return true;
                    }
                    stack.push(e.to);
                }
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::FordFulkerson;

    #[test]
    fn trivial_two_node() {
        let mut g = FordFulkerson::new(2);
        g.add_edge(0, 1, 7);
        assert_eq!(g.max_flow(0, 1), 7);
    }

    #[test]
    fn empty_two_node() {
        // No edges: flow is 0 even though the index space contains both nodes.
        let mut g = FordFulkerson::new(2);
        assert_eq!(g.max_flow(0, 1), 0);
    }

    #[test]
    fn source_equals_sink() {
        let mut g = FordFulkerson::new(2);
        g.add_edge(0, 1, 5);
        assert_eq!(g.max_flow(0, 0), 0);
    }

    #[test]
    fn classic_clrs_example() {
        // 6-node CLRS network; well-known max flow = 23.
        let mut g = FordFulkerson::new(6);
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
    fn bottleneck_graph() {
        // 0 -> 1 (100), 1 -> 2 (1), 2 -> 3 (100): the middle edge caps the
        // total flow at 1 regardless of the wide outer pipes.
        let mut g = FordFulkerson::new(4);
        g.add_edge(0, 1, 100);
        g.add_edge(1, 2, 1);
        g.add_edge(2, 3, 100);
        assert_eq!(g.max_flow(0, 3), 1);
    }

    #[test]
    fn no_path_returns_zero() {
        // Sink (node 2) is unreachable from source (node 0).
        let mut g = FordFulkerson::new(3);
        g.add_edge(0, 1, 5);
        assert_eq!(g.max_flow(0, 2), 0);
    }

    #[test]
    fn multi_edge_between_same_nodes() {
        // Three parallel edges 0 -> 1 should sum to capacity 9.
        let mut g = FordFulkerson::new(2);
        g.add_edge(0, 1, 2);
        g.add_edge(0, 1, 3);
        g.add_edge(0, 1, 4);
        assert_eq!(g.max_flow(0, 1), 9);
    }

    #[test]
    fn antiparallel_edges_independent() {
        // Forward 0 -> 1 (5) and backward 1 -> 0 (5) plus 1 -> 2 (5).
        // Each user edge keeps its own paired reverse, so the two directions
        // do not alias and the max 0 -> 2 flow is exactly 5.
        let mut g = FordFulkerson::new(3);
        g.add_edge(0, 1, 5);
        g.add_edge(1, 0, 5);
        g.add_edge(1, 2, 5);
        assert_eq!(g.max_flow(0, 2), 5);
    }

    #[test]
    fn dfs_must_use_reverse_edges() {
        // Classic counterexample where greedy DFS picks the "diagonal" edge
        // first and only completes by routing flow back along its reverse.
        // Source 0, sink 3.
        // 0 -> 1 (10), 0 -> 2 (10), 1 -> 2 (1), 1 -> 3 (10), 2 -> 3 (10).
        // Max flow = 20.
        let mut g = FordFulkerson::new(4);
        g.add_edge(0, 1, 10);
        g.add_edge(0, 2, 10);
        g.add_edge(1, 2, 1);
        g.add_edge(1, 3, 10);
        g.add_edge(2, 3, 10);
        assert_eq!(g.max_flow(0, 3), 20);
    }

    #[test]
    #[should_panic(expected = "endpoint out of range")]
    fn out_of_range_endpoint_panics() {
        let mut g = FordFulkerson::new(2);
        let _ = g.max_flow(5, 1);
    }
}
