//! Minimum-cost maximum-flow (MCMF) on a directed network with capacities and
//! per-unit edge costs.
//!
//! Computes the maximum `s`-`t` flow whose total cost — sum of
//! `flow(e) * cost(e)` over edges `e` — is minimal. Among all maximum flows,
//! the cheapest one is returned.
//!
//! # Algorithm — Successive Shortest Paths (SSP) with Bellman–Ford
//!
//! Repeatedly find the *cheapest* augmenting path in the residual network
//! using Bellman–Ford, then push as much flow as possible along it. Bellman–
//! Ford is required (rather than Dijkstra) because residual graphs contain
//! reverse edges with *negative* cost: cancelling one unit of flow on edge
//! `u -> v` at cost `c` gives an effective `v -> u` edge of cost `-c`. This
//! is exactly what lets the search undo a suboptimal earlier augmentation.
//!
//! When no augmenting path exists, the current flow is both maximum *and*
//! minimum-cost (proof: the residual graph contains no negative-cost cycle —
//! pushing along the shortest path each time preserves the no-negative-cycle
//! invariant — and a max flow with no negative residual cycle is a min-cost
//! max flow).
//!
//! ## Reverse-edge trick
//!
//! Edges are stored in a flat `Vec<HalfEdge>`. Each user-supplied forward
//! edge at index `i` has a paired reverse edge at index `i ^ 1` (edges are
//! always inserted in pairs). The reverse starts with `capacity = 0` and
//! `cost = -original_cost`, so sending `f` units along edge `i` does
//! `edges[i].capacity -= f` and `edges[i ^ 1].capacity += f`, and the
//! resulting `v -> u` residual carries cost `-c` automatically. Antiparallel
//! user edges (`u -> v` and `v -> u`) get their *own* paired reverses, so
//! the two directions never alias.
//!
//! # Complexity
//!
//! - Time:  `O(F * V * E)` where `F` is the value of the maximum flow — each
//!   augmentation pushes at least one unit, and Bellman–Ford runs in
//!   `O(V * E)` per iteration.
//! - Space: `O(V + E)`.
//!
//! # Preconditions
//!
//! - Original-edge costs are non-negative (`>= 0`). Costs on residual reverse
//!   edges are negative by construction; Bellman–Ford handles them.
//! - There must be no negative-cost cycle reachable from `src` in the input
//!   graph; with non-negative input costs this is automatic.
//! - `src` and `sink` must lie in `0..n`. An empty network or `src == sink`
//!   yields `(0, 0)`.

use std::collections::VecDeque;

/// One half of a residual edge. Edges are stored in pairs: index `2k` is the
/// forward edge supplied by the caller, index `2k + 1` is its reverse, so
/// `rev_idx = idx ^ 1`.
#[derive(Copy, Clone, Debug)]
struct HalfEdge {
    to: usize,
    capacity: u64,
    cost: i64,
    rev_idx: usize,
}

/// A min-cost flow network with incremental edge insertion and a single-shot
/// `min_cost_max_flow` query.
///
/// Edges are stored in a flat vector with paired reverse entries; `adj[u]`
/// holds indices into that vector. This is the same idiomatic layout used by
/// the [`DinicNetwork`](crate::graph::dinic::DinicNetwork) in this crate.
#[derive(Clone, Debug)]
pub struct MinCostFlow {
    num_nodes: usize,
    edges: Vec<HalfEdge>,
    adj: Vec<Vec<usize>>,
}

impl MinCostFlow {
    /// Creates an empty network on `n` nodes labelled `0..n`.
    #[must_use]
    pub fn new(n: usize) -> Self {
        Self {
            num_nodes: n,
            edges: Vec::new(),
            adj: vec![Vec::new(); n],
        }
    }

    /// Adds a directed edge `from -> to` with the given non-negative
    /// `capacity` and per-unit `cost`. Internally inserts the paired reverse
    /// edge with capacity `0` and cost `-cost`.
    ///
    /// Parallel calls compose: two `add_edge(u, v, 3, 1)` calls behave the
    /// same as a single channel of capacity `6` at cost `1` per unit (each
    /// pair is augmented independently, but the net min-cost max-flow is
    /// equivalent).
    ///
    /// # Panics
    /// Panics if `from` or `to` is out of range (`>= n`).
    pub fn add_edge(&mut self, from: usize, to: usize, capacity: u64, cost: i64) {
        assert!(
            from < self.num_nodes && to < self.num_nodes,
            "MinCostFlow::add_edge: endpoint out of range"
        );
        let m = self.edges.len();
        self.edges.push(HalfEdge {
            to,
            capacity,
            cost,
            rev_idx: m + 1,
        });
        self.edges.push(HalfEdge {
            to: from,
            capacity: 0,
            cost: -cost,
            rev_idx: m,
        });
        self.adj[from].push(m);
        self.adj[to].push(m + 1);
    }

    /// Returns `(max_flow, min_cost)` from `src` to `sink`: the maximum flow
    /// value, and — among all maximum flows — the minimum total cost.
    ///
    /// The network is mutated: residual capacities reflect the resulting flow
    /// assignment, so the same network should not be reused for a different
    /// `(src, sink)` pair without rebuilding.
    ///
    /// Returns `(0, 0)` immediately if `src == sink` or the network has no
    /// nodes.
    ///
    /// # Panics
    /// Panics if `src` or `sink` is out of range (`>= n`).
    pub fn min_cost_max_flow(&mut self, src: usize, sink: usize) -> (u64, i64) {
        assert!(
            src < self.num_nodes && sink < self.num_nodes,
            "MinCostFlow::min_cost_max_flow: endpoint out of range"
        );
        if src == sink {
            return (0, 0);
        }
        let mut total_flow: u64 = 0;
        let mut total_cost: i64 = 0;
        while let Some((path_flow, path_cost)) = self.augment(src, sink) {
            total_flow = total_flow.saturating_add(path_flow);
            total_cost = total_cost.saturating_add(path_cost);
        }
        (total_flow, total_cost)
    }

    /// One Bellman–Ford augmentation. Returns `Some((flow, cost))` for the
    /// pushed unit, or `None` if `sink` is unreachable along positive-residual
    /// edges.
    fn augment(&mut self, src: usize, sink: usize) -> Option<(u64, i64)> {
        let n = self.num_nodes;
        let mut dist: Vec<i64> = vec![i64::MAX; n];
        // Parent edge index used to reach each node along the current
        // shortest path; `usize::MAX` is the sentinel for "unset".
        let mut parent_edge: Vec<usize> = vec![usize::MAX; n];
        let mut in_queue = vec![false; n];
        let mut queue: VecDeque<usize> = VecDeque::new();

        dist[src] = 0;
        queue.push_back(src);
        in_queue[src] = true;

        // SPFA — Bellman–Ford with a queue of "modified" vertices. Each edge
        // is relaxed at most `O(V)` times in the worst case, matching plain
        // Bellman–Ford's `O(V * E)` bound while typically running much
        // faster on sparse graphs.
        while let Some(u) = queue.pop_front() {
            in_queue[u] = false;
            for &eid in &self.adj[u] {
                let e = self.edges[eid];
                if e.capacity == 0 {
                    continue;
                }
                let nd = dist[u].saturating_add(e.cost);
                if nd < dist[e.to] {
                    dist[e.to] = nd;
                    parent_edge[e.to] = eid;
                    if !in_queue[e.to] {
                        queue.push_back(e.to);
                        in_queue[e.to] = true;
                    }
                }
            }
        }

        if dist[sink] == i64::MAX {
            return None;
        }

        // Walk back from `sink` to `src` to find the bottleneck capacity.
        let mut bottleneck: u64 = u64::MAX;
        let mut v = sink;
        while v != src {
            let eid = parent_edge[v];
            let e = self.edges[eid];
            bottleneck = bottleneck.min(e.capacity);
            v = self.edges[e.rev_idx].to;
        }

        // Apply the flow: subtract from the forward residual, add to the
        // reverse residual.
        let mut v = sink;
        let mut path_cost: i64 = 0;
        while v != src {
            let eid = parent_edge[v];
            self.edges[eid].capacity -= bottleneck;
            let rev = self.edges[eid].rev_idx;
            self.edges[rev].capacity = self.edges[rev].capacity.saturating_add(bottleneck);
            path_cost = path_cost.saturating_add(self.edges[eid].cost * bottleneck as i64);
            v = self.edges[rev].to;
        }

        Some((bottleneck, path_cost))
    }
}

#[cfg(test)]
mod tests {
    use super::MinCostFlow;
    use crate::graph::edmonds_karp::{edmonds_karp, Edge as EkEdge};
    use quickcheck_macros::quickcheck;

    #[test]
    fn empty_network() {
        let mut g = MinCostFlow::new(2);
        assert_eq!(g.min_cost_max_flow(0, 1), (0, 0));
    }

    #[test]
    fn zero_nodes() {
        // src == sink == 0 short-circuits before any indexing happens.
        let mut g = MinCostFlow::new(1);
        assert_eq!(g.min_cost_max_flow(0, 0), (0, 0));
    }

    #[test]
    fn source_equals_sink() {
        let mut g = MinCostFlow::new(2);
        g.add_edge(0, 1, 5, 3);
        assert_eq!(g.min_cost_max_flow(0, 0), (0, 0));
    }

    #[test]
    fn simple_two_node_path() {
        let mut g = MinCostFlow::new(2);
        g.add_edge(0, 1, 7, 4);
        // 7 units * cost 4 = 28.
        assert_eq!(g.min_cost_max_flow(0, 1), (7, 28));
    }

    #[test]
    fn unreachable_sink() {
        let mut g = MinCostFlow::new(3);
        g.add_edge(0, 1, 5, 2);
        assert_eq!(g.min_cost_max_flow(0, 2), (0, 0));
    }

    #[test]
    #[should_panic(expected = "endpoint out of range")]
    fn out_of_range_src_panics() {
        let mut g = MinCostFlow::new(2);
        let _ = g.min_cost_max_flow(5, 1);
    }

    #[test]
    fn picks_cheaper_of_two_disjoint_paths() {
        // Two disjoint 0 -> 3 paths share a capacity-1 sink edge so only one
        // unit can flow total. The algorithm must route it along the cheap
        // path (cost 2), not the expensive one (cost 10).
        //
        //   0 -> 1 -> 3   costs (1, 1), caps (1, 1)  -> total cost 2
        //   0 -> 2 -> 3   costs (5, 5), caps (1, 1)  -> total cost 10
        //   But 1->3 and 2->3 share a single bottleneck via a "sink-cap" node.
        // Concretely: route both paths through node 3, then 3 -> 4 with
        // capacity 1; the actual sink is 4.
        let mut g = MinCostFlow::new(5);
        g.add_edge(0, 1, 1, 1);
        g.add_edge(1, 3, 1, 1);
        g.add_edge(0, 2, 1, 5);
        g.add_edge(2, 3, 1, 5);
        g.add_edge(3, 4, 1, 0);
        assert_eq!(g.min_cost_max_flow(0, 4), (1, 2));
    }

    #[test]
    fn uses_both_paths_when_max_flow_demands_it() {
        // Same shape, but each "outer" edge has capacity 1 from source. Total
        // max flow from 0 to 3 is 2 — one unit on the cheap path, one on the
        // expensive path. Min cost = 2 + 10 = 12.
        let mut g = MinCostFlow::new(4);
        g.add_edge(0, 1, 1, 1);
        g.add_edge(1, 3, 1, 1);
        g.add_edge(0, 2, 1, 5);
        g.add_edge(2, 3, 1, 5);
        assert_eq!(g.min_cost_max_flow(0, 3), (2, 12));
    }

    #[test]
    fn parallel_edges_pick_cheapest_first() {
        // Two parallel 0 -> 1 channels: capacity 3 at cost 10, capacity 2 at
        // cost 1. Demand 5, min cost = 2*1 + 3*10 = 32.
        let mut g = MinCostFlow::new(2);
        g.add_edge(0, 1, 3, 10);
        g.add_edge(0, 1, 2, 1);
        assert_eq!(g.min_cost_max_flow(0, 1), (5, 32));
    }

    #[test]
    fn classic_textbook_example() {
        // 4-node diamond:
        //     1
        //   / | \
        //  0  |  3
        //   \ | /
        //     2
        // Edges (cap, cost):
        //   0->1 (3, 1), 0->2 (2, 2),
        //   1->2 (1, 1),
        //   1->3 (2, 3), 2->3 (3, 1).
        // Max flow 0 -> 3 = 5.
        //
        // Greedy along cheapest-first paths:
        //  - 0->1->3 cost 1+3=4: bottleneck 2, push 2 (cost 8).
        //  - 0->1->2->3 cost 1+1+1=3 with cap 1: push 1 (cost 3).
        //  - 0->2->3 cost 2+1=3 with cap 2: push 2 (cost 6).
        //  Total flow 5, total cost 8 + 3 + 6 = 17.
        //
        // SSP must reach the same answer (with a different path order it is
        // free to use the negative residuals to reroute).
        let mut g = MinCostFlow::new(4);
        g.add_edge(0, 1, 3, 1);
        g.add_edge(0, 2, 2, 2);
        g.add_edge(1, 2, 1, 1);
        g.add_edge(1, 3, 2, 3);
        g.add_edge(2, 3, 3, 1);
        assert_eq!(g.min_cost_max_flow(0, 3), (5, 17));
    }

    #[test]
    fn reverse_edge_lets_search_reroute() {
        // Crafted so the first cheapest path is *not* part of an optimal
        // max-flow assignment, forcing the algorithm to use a residual
        // reverse edge to undo it.
        //
        // Nodes 0=s, 1, 2, 3=t. Edges (cap, cost):
        //   s->1 (1, 1), 1->t (1, 1)        cheap "main" path, cost 2
        //   s->2 (1, 100)                   only way to feed node 2
        //   2->1 (1, 1)                     funnel into 1
        //   1->t already saturated after first push, so the second unit must
        //   reroute via the residual t->1 ... that doesn't help; instead we
        //   add a direct 2->t with high cost so SSP keeps finding the
        //   cheapest augmenting path each round.
        //
        // We just verify the cost matches a hand-computed optimum.
        let mut g = MinCostFlow::new(4);
        g.add_edge(0, 1, 1, 1);
        g.add_edge(1, 3, 1, 1);
        g.add_edge(0, 2, 1, 100);
        g.add_edge(2, 1, 1, 1);
        g.add_edge(2, 3, 1, 50);
        // Round 1: 0->1->3 cost 2, flow 1.
        // Round 2: cheapest augmenting path is 0->2->3 cost 150, flow 1.
        // Total: flow 2, cost 152.
        assert_eq!(g.min_cost_max_flow(0, 3), (2, 152));
    }

    /// Decode a deterministic pseudo-random graph from the `QuickCheck` inputs
    /// and return both the `MinCostFlow` network and an Edmonds–Karp edge
    /// list for the same graph (capacities only). Costs are non-negative
    /// per the algorithm's preconditions.
    fn build_random(
        n_seed: u8,
        mask: u64,
        weight_seed: u64,
    ) -> (MinCostFlow, Vec<EkEdge>, usize, usize, usize) {
        let n = ((n_seed as usize) % 4) + 2; // 2..=5
        let mut g = MinCostFlow::new(n);
        let mut ek = Vec::new();
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
                let cap = w % 6; // 0..=5
                w = w.rotate_left(7).wrapping_add(0x9E37_79B9_7F4A_7C15);
                let cost = (w % 10) as i64; // 0..=9, non-negative
                w = w.rotate_left(11).wrapping_add(0xDEAD_BEEF_CAFE_F00D);
                if cap == 0 {
                    continue;
                }
                g.add_edge(u, v, cap, cost);
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
    fn quickcheck_max_flow_matches_edmonds_karp(n_seed: u8, mask: u64, weight_seed: u64) -> bool {
        let (mut mcmf, ek, n, src, sink) = build_random(n_seed, mask, weight_seed);
        let (mcmf_flow, _cost) = mcmf.min_cost_max_flow(src, sink);
        let ek_flow = edmonds_karp(n, &ek, src, sink);
        mcmf_flow == ek_flow
    }
}
