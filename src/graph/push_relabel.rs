//! Goldberg–Tarjan push-relabel maximum-flow algorithm (FIFO variant).
//!
//! Computes the maximum `s`-`t` flow in a directed network with non-negative
//! integer (`u64`) capacities. Push-relabel maintains a *preflow* (each vertex
//! may temporarily have more flow coming in than going out) and resolves the
//! excess by repeatedly *pushing* admissible flow along residual edges and
//! *relabeling* (raising the height of) vertices that still carry excess but
//! have no admissible outgoing edge.
//!
//! # Algorithm
//! Each vertex `v` has a non-negative integer **height** `h(v)` and an integer
//! **excess** `e(v) = inflow - outflow`. An edge `u -> v` in the residual
//! graph is **admissible** iff `cap_residual(u, v) > 0` *and*
//! `h(u) == h(v) + 1`.
//!
//! Initialization (preflow):
//! - `h(s) = n`, `h(v) = 0` for `v != s`.
//! - Saturate every edge out of the source: push `cap(s, v)` units along it,
//!   which gives every neighbour of `s` an excess equal to that capacity.
//!
//! Main loop (FIFO selection): keep a queue of *active* vertices (those with
//! positive excess, excluding `s` and `t`). Repeatedly dequeue an active
//! vertex `u` and **discharge** it:
//! 1. Walk `u`'s outgoing residual edges. For each edge `u -> v` with
//!    positive residual capacity and `h(u) == h(v) + 1`, push
//!    `min(e(u), cap_residual(u, v))` units along it. If `v` is not `s` or
//!    `t` and was inactive before, enqueue it.
//! 2. If after scanning every edge `u` still has excess, **relabel**: set
//!    `h(u) = 1 + min{h(v) : (u, v) has positive residual capacity}` and
//!    re-enqueue `u`.
//!
//! When the queue is empty no vertex other than `s` / `t` has excess, so the
//! preflow is in fact a valid flow, and `e(t)` is the maximum flow value.
//!
//! ## Invariants (each maintained at every step)
//! - `h(s) = n`, `h(t) = 0` always.
//! - For every residual edge `u -> v`, `h(u) <= h(v) + 1` (height function is
//!   *valid* w.r.t. the residual graph). This is what makes admissible
//!   pushes well-defined and bounds the height of any vertex by `2n - 1`.
//! - `e(v) >= 0` for `v != s` (preflow property).
//!
//! ## Reverse-edge trick
//! Identical to Dinic / Edmonds–Karp: edges are stored in pairs, forward at
//! index `2k`, reverse at `2k + 1`. Pushing `f` units does
//! `edges[i].capacity -= f` and `edges[i ^ 1].capacity += f`. Antiparallel
//! user edges keep their *own* paired reverses, so the two directions never
//! alias.
//!
//! # Complexity
//! - Time:  `O(V^2 · E)` for FIFO push-relabel (this implementation). Highest-
//!   label selection improves this to `O(V^2 · sqrt(E))` but adds bookkeeping
//!   overhead; FIFO is preferred here for clarity.
//! - Space: `O(V + E)`.
//!
//! # Preconditions
//! `src` and `sink` must be in `0..n`; otherwise `max_flow` panics. Total flow
//! must fit in `u64`.

use std::collections::VecDeque;

/// Mutable per-run scratch state shared between the main loop and `discharge`.
struct State {
    height: Vec<usize>,
    excess: Vec<u64>,
    in_queue: Vec<bool>,
    queue: VecDeque<usize>,
}

/// One half of a residual edge. Edges are stored in pairs: index `2k` is the
/// forward edge, index `2k + 1` its reverse, so `rev_idx = idx ^ 1`.
#[derive(Copy, Clone, Debug)]
struct Edge {
    to: usize,
    capacity: u64,
    rev_idx: usize,
}

/// A flow network solved with FIFO push-relabel.
///
/// Build the network with [`PushRelabelNetwork::new`] and
/// [`PushRelabelNetwork::add_edge`], then call
/// [`PushRelabelNetwork::max_flow`]. The network is mutated by `max_flow`
/// (residual capacities reflect the resulting flow), so it should not be
/// reused for a different `(src, sink)` pair without rebuilding.
#[derive(Clone, Debug)]
pub struct PushRelabelNetwork {
    num_nodes: usize,
    edges: Vec<Edge>,
    adj: Vec<Vec<usize>>,
}

impl PushRelabelNetwork {
    /// Creates an empty network on `n` nodes labelled `0..n`.
    pub fn new(n: usize) -> Self {
        Self {
            num_nodes: n,
            edges: Vec::new(),
            adj: vec![Vec::new(); n],
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
            "PushRelabelNetwork::add_edge: endpoint out of range"
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

    /// Returns the maximum flow value from `src` to `sink`.
    ///
    /// Returns `0` immediately if `src == sink` or if the network has fewer
    /// than two vertices.
    ///
    /// # Panics
    /// Panics if `src` or `sink` is out of range (`>= n`).
    pub fn max_flow(&mut self, src: usize, sink: usize) -> u64 {
        assert!(
            src < self.num_nodes && sink < self.num_nodes,
            "PushRelabelNetwork::max_flow: endpoint out of range"
        );
        if src == sink || self.num_nodes < 2 {
            return 0;
        }
        let n = self.num_nodes;
        let mut st = State {
            height: vec![0_usize; n],
            excess: vec![0_u64; n],
            in_queue: vec![false; n],
            queue: VecDeque::new(),
        };

        // Preflow: source height = n, saturate every edge out of source.
        st.height[src] = n;
        // Snapshot src's adjacency indices to avoid aliasing the borrow while
        // we mutate `self.edges`.
        let src_adj: Vec<usize> = self.adj[src].clone();
        for eid in src_adj {
            let cap = self.edges[eid].capacity;
            if cap == 0 {
                continue;
            }
            let to = self.edges[eid].to;
            let rev = self.edges[eid].rev_idx;
            self.edges[eid].capacity = 0;
            self.edges[rev].capacity = self.edges[rev].capacity.saturating_add(cap);
            st.excess[to] = st.excess[to].saturating_add(cap);
            // Source's excess is conceptually -inf; we don't track it.
            if to != src && to != sink && !st.in_queue[to] {
                st.queue.push_back(to);
                st.in_queue[to] = true;
            }
        }

        while let Some(u) = st.queue.pop_front() {
            st.in_queue[u] = false;
            self.discharge(u, src, sink, &mut st);
        }

        st.excess[sink]
    }

    /// Push as much flow as possible out of `u` along admissible edges; if
    /// excess remains, relabel `u` and re-enqueue it. Newly active neighbours
    /// (other than `src` / `sink`) are pushed onto the queue.
    fn discharge(&mut self, u: usize, src: usize, sink: usize, st: &mut State) {
        while st.excess[u] > 0 {
            let mut pushed_any = false;
            // Borrow the adjacency list by length to avoid aliasing while we
            // mutate edges.
            let deg = self.adj[u].len();
            for i in 0..deg {
                if st.excess[u] == 0 {
                    break;
                }
                let eid = self.adj[u][i];
                let e = self.edges[eid];
                if e.capacity > 0 && st.height[u] == st.height[e.to] + 1 {
                    let send = st.excess[u].min(e.capacity);
                    self.edges[eid].capacity -= send;
                    let rev = e.rev_idx;
                    self.edges[rev].capacity = self.edges[rev].capacity.saturating_add(send);
                    st.excess[u] -= send;
                    st.excess[e.to] = st.excess[e.to].saturating_add(send);
                    if e.to != src && e.to != sink && !st.in_queue[e.to] {
                        st.queue.push_back(e.to);
                        st.in_queue[e.to] = true;
                    }
                    pushed_any = true;
                }
            }
            if st.excess[u] == 0 {
                break;
            }
            if !pushed_any {
                // Relabel: minimum height of a residual neighbour, plus one.
                // If no residual neighbour exists the vertex is disconnected
                // from the sink in the residual graph; bail out (its excess
                // will eventually flow back to the source).
                let mut min_h: Option<usize> = None;
                for &eid in &self.adj[u] {
                    let e = self.edges[eid];
                    if e.capacity > 0 {
                        min_h = Some(min_h.map_or(st.height[e.to], |h| h.min(st.height[e.to])));
                    }
                }
                match min_h {
                    Some(h) => st.height[u] = h + 1,
                    None => {
                        // No outgoing residual capacity at all: stuck.
                        return;
                    }
                }
                // Re-enqueue self for another discharge round.
                if !st.in_queue[u] {
                    st.queue.push_back(u);
                    st.in_queue[u] = true;
                }
                return;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::PushRelabelNetwork;
    use crate::graph::edmonds_karp::{edmonds_karp, Edge as EkEdge};
    use quickcheck_macros::quickcheck;

    #[test]
    fn empty_network() {
        let mut g = PushRelabelNetwork::new(2);
        assert_eq!(g.max_flow(0, 1), 0);
    }

    #[test]
    #[should_panic(expected = "endpoint out of range")]
    fn out_of_range_src_panics() {
        let mut g = PushRelabelNetwork::new(2);
        let _ = g.max_flow(5, 1);
    }

    #[test]
    #[should_panic(expected = "endpoint out of range")]
    fn out_of_range_add_edge_panics() {
        let mut g = PushRelabelNetwork::new(2);
        g.add_edge(0, 7, 1);
    }

    #[test]
    fn source_equals_sink() {
        let mut g = PushRelabelNetwork::new(2);
        g.add_edge(0, 1, 5);
        assert_eq!(g.max_flow(0, 0), 0);
    }

    #[test]
    fn single_edge_path() {
        let mut g = PushRelabelNetwork::new(2);
        g.add_edge(0, 1, 7);
        assert_eq!(g.max_flow(0, 1), 7);
    }

    #[test]
    fn unreachable_sink() {
        let mut g = PushRelabelNetwork::new(3);
        g.add_edge(0, 1, 5);
        assert_eq!(g.max_flow(0, 2), 0);
    }

    #[test]
    fn classic_clrs_example() {
        // 6-node CLRS network; well-known max flow = 23.
        let mut g = PushRelabelNetwork::new(6);
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
        let mut g = PushRelabelNetwork::new(2);
        g.add_edge(0, 1, 3);
        g.add_edge(0, 1, 4);
        assert_eq!(g.max_flow(0, 1), 7);
    }

    #[test]
    fn antiparallel_edges_have_independent_reverse() {
        // 0 -> 1 (5), 1 -> 0 (5), 1 -> 2 (5). Max flow 0 -> 2 should be 5.
        let mut g = PushRelabelNetwork::new(3);
        g.add_edge(0, 1, 5);
        g.add_edge(1, 0, 5);
        g.add_edge(1, 2, 5);
        assert_eq!(g.max_flow(0, 2), 5);
    }

    #[test]
    fn diamond_with_cancellation() {
        // 0 -> 1 -> 3, 0 -> 2 -> 3, plus a misleading 1 -> 2 link. Forces the
        // algorithm to use reverse edges to undo flow if it misroutes early.
        let mut g = PushRelabelNetwork::new(4);
        g.add_edge(0, 1, 10);
        g.add_edge(0, 2, 10);
        g.add_edge(1, 2, 1);
        g.add_edge(1, 3, 10);
        g.add_edge(2, 3, 10);
        assert_eq!(g.max_flow(0, 3), 20);
    }

    #[test]
    fn grid_smoke() {
        // 4x4 grid: each cell connects right and down with capacity 1. Max
        // flow from (0,0) to (3,3) equals the number of edge-disjoint
        // paths, which on a unit-capacity grid is bounded by the min row /
        // column degree at the corners (= 2 for a 2D grid corner). This is
        // primarily a "doesn't blow up / agrees with EK" smoke test.
        let rows = 4;
        let cols = 4;
        let n = rows * cols;
        let idx = |r: usize, c: usize| r * cols + c;
        let mut g = PushRelabelNetwork::new(n);
        let mut ek = Vec::new();
        for r in 0..rows {
            for c in 0..cols {
                if c + 1 < cols {
                    g.add_edge(idx(r, c), idx(r, c + 1), 1);
                    ek.push(EkEdge {
                        from: idx(r, c),
                        to: idx(r, c + 1),
                        capacity: 1,
                    });
                }
                if r + 1 < rows {
                    g.add_edge(idx(r, c), idx(r + 1, c), 1);
                    ek.push(EkEdge {
                        from: idx(r, c),
                        to: idx(r + 1, c),
                        capacity: 1,
                    });
                }
            }
        }
        let src = idx(0, 0);
        let sink = idx(rows - 1, cols - 1);
        let pr_flow = g.max_flow(src, sink);
        let ek_flow = edmonds_karp(n, &ek, src, sink);
        assert_eq!(pr_flow, ek_flow);
        assert_eq!(pr_flow, 2);
    }

    #[test]
    fn bipartite_matching_reduction() {
        // Same shape as the Dinic test: 3 left nodes, 3 right nodes, source
        // and sink wrap the bipartite graph. Max matching = 3.
        let n = 8;
        let mut g = PushRelabelNetwork::new(n);
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

    /// Decode a deterministic pseudo-random graph from `QuickCheck` inputs and
    /// return both a push-relabel network and an Edmonds–Karp edge list for
    /// the same graph, plus `(n, src, sink)`.
    fn build_random(
        n_seed: u8,
        mask: u64,
        weight_seed: u64,
    ) -> (PushRelabelNetwork, Vec<EkEdge>, usize, usize, usize) {
        let n = ((n_seed as usize) % 5) + 2; // 2..=6
        let mut g = PushRelabelNetwork::new(n);
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
        let (mut pr, ek, n, src, sink) = build_random(n_seed, mask, weight_seed);
        let pr_flow = pr.max_flow(src, sink);
        let ek_flow = edmonds_karp(n, &ek, src, sink);
        pr_flow == ek_flow
    }
}
