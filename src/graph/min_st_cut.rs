//! Minimum `s`-`t` cut extraction from a maximum flow.
//!
//! Given a directed graph with non-negative integer edge capacities, this
//! module computes the **minimum `s`-`t` cut**: a partition of the vertices
//! into `(S, T)` with `s in S`, `t in T` minimising the total capacity of
//! edges crossing from `S` to `T`. By the max-flow / min-cut theorem this
//! capacity equals the value of the maximum `s`-`t` flow.
//!
//! # Algorithm
//! 1. Build a residual network and run a private Dinic-style max-flow.
//! 2. After the max-flow saturates every `s`-`t` augmenting path, BFS from
//!    `s` over edges with strictly positive residual capacity. The set of
//!    visited vertices forms the `S`-side of any minimum cut; everything
//!    else is the `T`-side.
//! 3. Walk the original (input) edges and emit `(u, v)` whenever
//!    `u in S`, `v in T`, and the original capacity is positive — those are
//!    exactly the saturated forward edges crossing the cut.
//!
//! The module is intentionally self-contained: it embeds its own max-flow
//! solver so callers do not pull in unrelated graph machinery.
//!
//! # Complexity
//! - Time:  `O(V^2 · E)` — dominated by Dinic's max-flow phase.
//! - Space: `O(V + E)`.
//!
//! # Returned cut
//! The summed capacity of the returned `(u, v)` edges always equals the
//! max-flow value, and the cut separates `s` from `t` in the original graph.

use std::collections::VecDeque;

/// One half of a residual edge. Edges are stored in pairs: index `2k` is the
/// forward edge, index `2k + 1` is its reverse, so `rev_idx = idx ^ 1`.
#[derive(Copy, Clone, Debug)]
struct Edge {
    to: usize,
    capacity: u64,
    rev_idx: usize,
}

/// Internal Dinic-style residual network used solely to compute max-flow.
struct Network {
    edges: Vec<Edge>,
    adj: Vec<Vec<usize>>,
    level: Vec<i32>,
    iter: Vec<usize>,
}

impl Network {
    fn new(n: usize) -> Self {
        Self {
            edges: Vec::new(),
            adj: vec![Vec::new(); n],
            level: vec![-1; n],
            iter: vec![0; n],
        }
    }

    fn add_edge(&mut self, from: usize, to: usize, capacity: u64) {
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

    fn max_flow(&mut self, src: usize, sink: usize) -> u64 {
        if src == sink {
            return 0;
        }
        let mut total: u64 = 0;
        while self.bfs(src, sink) {
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

    /// BFS from `src` over edges with positive residual capacity. Returns the
    /// reachability mask: `mask[v] = true` iff `v` is reachable.
    fn reachable_from(&self, src: usize) -> Vec<bool> {
        let n = self.adj.len();
        let mut visited = vec![false; n];
        if src >= n {
            return visited;
        }
        visited[src] = true;
        let mut queue = VecDeque::new();
        queue.push_back(src);
        while let Some(u) = queue.pop_front() {
            for &eid in &self.adj[u] {
                let e = self.edges[eid];
                if e.capacity > 0 && !visited[e.to] {
                    visited[e.to] = true;
                    queue.push_back(e.to);
                }
            }
        }
        visited
    }
}

/// Computes the minimum `s`-`t` cut of the directed network described by
/// `edges` on `n` vertices `0..n`.
///
/// `edges` is a slice of `(from, to, capacity)` triples; parallel edges are
/// allowed and accumulate. The returned tuple is
/// `(min_cut_capacity, cut_edges)` where `cut_edges` lists every original
/// edge `(u, v)` whose removal helps disconnect `s` from `t` — concretely,
/// every input edge with positive capacity such that `u` is reachable from
/// `s` in the final residual graph and `v` is not.
///
/// The summed capacity of `cut_edges` (counting parallel duplicates) equals
/// `min_cut_capacity`, and equals the maximum `s`-`t` flow value.
///
/// # Panics
/// Panics if `s`, `t`, or any edge endpoint is `>= n`.
pub fn min_st_cut(
    n: usize,
    edges: &[(usize, usize, u64)],
    s: usize,
    t: usize,
) -> (u64, Vec<(usize, usize)>) {
    assert!(s < n && t < n, "min_st_cut: s and t must be < n");
    for &(u, v, _) in edges {
        assert!(u < n && v < n, "min_st_cut: edge endpoint out of range");
    }
    if s == t {
        return (0, Vec::new());
    }

    let mut net = Network::new(n);
    for &(u, v, cap) in edges {
        net.add_edge(u, v, cap);
    }
    let flow = net.max_flow(s, t);

    let in_s = net.reachable_from(s);
    let mut cut_edges = Vec::new();
    for &(u, v, cap) in edges {
        if cap > 0 && in_s[u] && !in_s[v] {
            cut_edges.push((u, v));
        }
    }
    (flow, cut_edges)
}

#[cfg(test)]
mod tests {
    use super::min_st_cut;

    fn cut_capacity(edges: &[(usize, usize, u64)], cut: &[(usize, usize)]) -> u64 {
        // Sum capacities of every input edge matching a cut endpoint pair.
        // Handles duplicates: each cut-listed `(u,v)` consumes one matching
        // input edge so parallel edges are counted exactly once.
        let mut taken = vec![false; edges.len()];
        let mut total: u64 = 0;
        for &(cu, cv) in cut {
            for (i, &(u, v, c)) in edges.iter().enumerate() {
                if !taken[i] && u == cu && v == cv && c > 0 {
                    taken[i] = true;
                    total = total.saturating_add(c);
                    break;
                }
            }
        }
        total
    }

    /// Verify that removing exactly `cut` from `edges` disconnects `s` from
    /// `t`. Performs forward BFS in the residual-free original graph after
    /// deleting one occurrence of every listed edge.
    fn cut_separates(
        n: usize,
        edges: &[(usize, usize, u64)],
        cut: &[(usize, usize)],
        s: usize,
        t: usize,
    ) -> bool {
        let mut removed = vec![false; edges.len()];
        for &(cu, cv) in cut {
            for (i, &(u, v, c)) in edges.iter().enumerate() {
                if !removed[i] && u == cu && v == cv && c > 0 {
                    removed[i] = true;
                    break;
                }
            }
        }
        let mut adj = vec![Vec::new(); n];
        for (i, &(u, v, c)) in edges.iter().enumerate() {
            if !removed[i] && c > 0 {
                adj[u].push(v);
            }
        }
        let mut visited = vec![false; n];
        visited[s] = true;
        let mut stack = vec![s];
        while let Some(u) = stack.pop() {
            for &v in &adj[u] {
                if !visited[v] {
                    visited[v] = true;
                    stack.push(v);
                }
            }
        }
        !visited[t]
    }

    #[test]
    fn s_equals_t() {
        let (flow, cut) = min_st_cut(3, &[(0, 1, 5)], 1, 1);
        assert_eq!(flow, 0);
        assert!(cut.is_empty());
    }

    #[test]
    fn two_node_simple_cut() {
        let edges = [(0_usize, 1_usize, 9_u64)];
        let (flow, cut) = min_st_cut(2, &edges, 0, 1);
        assert_eq!(flow, 9);
        assert_eq!(cut, vec![(0, 1)]);
        assert_eq!(cut_capacity(&edges, &cut), flow);
    }

    #[test]
    fn unreachable_sink_has_empty_cut() {
        // No path from s to t -> max flow is 0 and the cut is empty.
        let edges = [(0_usize, 1_usize, 5_u64)];
        let (flow, cut) = min_st_cut(3, &edges, 0, 2);
        assert_eq!(flow, 0);
        assert!(cut.is_empty());
    }

    #[test]
    fn classic_four_node_cut() {
        // s=0, t=3.
        //        2
        //   0 -------> 1
        //   |          |
        //  10          3
        //   v          v
        //   2 -------> 3
        //        4
        // Plus a cross edge 1->2 of capacity 1.
        // Max flow = 5 (saturates 0->1=2 and 0->2 routed as 0->2 only up to 5
        // since 2->3 = 4 and 1->3 = 3, so flow = min cut). Min cut edges: the
        // outgoing edges from {0,1} to {2,3} after saturation.
        let edges = [
            (0_usize, 1_usize, 2_u64),
            (0, 2, 10),
            (1, 3, 3),
            (2, 3, 4),
            (1, 2, 1),
        ];
        let (flow, cut) = min_st_cut(4, &edges, 0, 3);
        // Sanity: capacity of any cut equals flow value.
        assert_eq!(cut_capacity(&edges, &cut), flow);
        assert!(cut_separates(4, &edges, &cut, 0, 3));
    }

    #[test]
    fn parallel_edges_all_in_cut() {
        // Three parallel edges 0->1, sink reached only through them.
        let edges = [(0_usize, 1_usize, 2_u64), (0, 1, 3), (0, 1, 4), (1, 2, 100)];
        let (flow, cut) = min_st_cut(3, &edges, 0, 2);
        assert_eq!(flow, 9);
        // All three parallel edges must appear in the cut (the bottleneck).
        let parallel_cut_count = cut.iter().filter(|&&(u, v)| u == 0 && v == 1).count();
        assert_eq!(parallel_cut_count, 3);
        assert_eq!(cut_capacity(&edges, &cut), flow);
        assert!(cut_separates(3, &edges, &cut, 0, 2));
    }

    #[test]
    fn disjoint_paths_each_contributes_one_edge() {
        // Two parallel disjoint paths s -> a -> t and s -> b -> t.
        // Capacities: s->a=3, a->t=3, s->b=4, b->t=4. Max flow = 7.
        // Min cut = 7. Either side of each path may be the cut edge depending
        // on residual structure; the union must still total 7.
        let edges = [
            (0_usize, 1_usize, 3_u64), // s -> a
            (1, 4, 3),                 // a -> t
            (0, 2, 4),                 // s -> b
            (2, 4, 4),                 // b -> t
        ];
        let (flow, cut) = min_st_cut(5, &edges, 0, 4);
        assert_eq!(flow, 7);
        assert_eq!(cut_capacity(&edges, &cut), flow);
        assert!(cut_separates(5, &edges, &cut, 0, 4));
        // Each disjoint path must contribute exactly one cut edge.
        assert_eq!(cut.len(), 2);
    }

    #[test]
    fn classic_clrs_cut_capacity_matches_max_flow() {
        // 6-node CLRS network; well-known max flow = 23.
        let edges = [
            (0_usize, 1_usize, 16_u64),
            (0, 2, 13),
            (1, 2, 10),
            (2, 1, 4),
            (1, 3, 12),
            (2, 4, 14),
            (3, 2, 9),
            (3, 5, 20),
            (4, 3, 7),
            (4, 5, 4),
        ];
        let (flow, cut) = min_st_cut(6, &edges, 0, 5);
        assert_eq!(flow, 23);
        assert_eq!(cut_capacity(&edges, &cut), 23);
        assert!(cut_separates(6, &edges, &cut, 0, 5));
    }

    #[test]
    fn zero_capacity_edges_ignored() {
        // A 0-capacity edge from s directly to t must not appear in the cut.
        let edges = [
            (0_usize, 2_usize, 0_u64), // s -> t with 0 cap
            (0, 1, 5),                 // s -> a
            (1, 2, 5),                 // a -> t
        ];
        let (flow, cut) = min_st_cut(3, &edges, 0, 2);
        assert_eq!(flow, 5);
        assert!(!cut.contains(&(0, 2)));
        assert_eq!(cut_capacity(&edges, &cut), 5);
        assert!(cut_separates(3, &edges, &cut, 0, 2));
    }

    #[test]
    #[should_panic(expected = "s and t must be < n")]
    fn out_of_range_endpoint_panics() {
        let _ = min_st_cut(2, &[], 0, 5);
    }

    #[test]
    #[should_panic(expected = "edge endpoint out of range")]
    fn out_of_range_edge_panics() {
        let _ = min_st_cut(2, &[(0, 7, 1)], 0, 1);
    }
}
