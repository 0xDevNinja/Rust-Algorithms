//! Johnson's all-pairs shortest paths. Handles graphs with negative edge
//! weights as long as no negative cycle exists.
//!
//! Algorithm:
//! 1. Add a virtual source node `q` with zero-weight edges to every vertex.
//! 2. Run Bellman-Ford from `q` to compute potentials `h[v]`. If a negative
//!    cycle is reachable from `q` (equivalently, from anywhere), abort.
//! 3. Reweight every edge `(u, v)` with weight `w` to `w + h[u] - h[v]`. The
//!    triangle inequality on potentials guarantees the new weights are
//!    non-negative.
//! 4. Run Dijkstra from each vertex over the reweighted graph.
//! 5. Recover original distances via `dist(u, v) = dist'(u, v) - h[u] + h[v]`.
//!
//! Time complexity: O(V·E·log V) using a binary-heap Dijkstra. Space: O(V²)
//! for the output matrix plus O(V + E) working memory.
//!
//! The whole procedure returns `None` when a negative cycle is detected;
//! individual unreachable pairs are returned as `None` inside the matrix.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

#[derive(Copy, Clone, Eq, PartialEq)]
struct State {
    cost: u64,
    node: usize,
}

impl Ord for State {
    fn cmp(&self, other: &Self) -> Ordering {
        // Min-heap: reverse cost ordering, ties broken by node id.
        other
            .cost
            .cmp(&self.cost)
            .then_with(|| self.node.cmp(&other.node))
    }
}

impl PartialOrd for State {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Returns the all-pairs shortest-path matrix. `graph[u]` lists outgoing
/// edges as `(neighbor, weight)` pairs; weights may be negative.
///
/// The outer `Option` is `None` if the graph contains a negative-weight cycle.
/// Otherwise, entry `[u][v]` is `Some(d)` for the shortest distance from `u`
/// to `v`, or `None` if `v` is unreachable from `u`. Diagonal entries are
/// `Some(0)`.
pub fn johnsons(graph: &[Vec<(usize, i64)>]) -> Option<Vec<Vec<Option<i64>>>> {
    let n = graph.len();
    if n == 0 {
        return Some(Vec::new());
    }

    // --- Step 1+2: Bellman-Ford from a virtual source `q` (index n).
    // We don't materialise q in the adjacency list; instead we initialise
    // every potential to 0 (equivalent to a 0-weight edge q -> v) and relax
    // the real edges of `graph`. After at most n-1 rounds the potentials are
    // final; one more pass detects a negative cycle.
    let mut h = vec![0_i64; n];
    for _ in 0..n.saturating_sub(1) {
        let mut updated = false;
        for (u, adj) in graph.iter().enumerate() {
            for &(v, w) in adj {
                if v >= n {
                    continue;
                }
                let candidate = h[u].saturating_add(w);
                if candidate < h[v] {
                    h[v] = candidate;
                    updated = true;
                }
            }
        }
        if !updated {
            break;
        }
    }
    for (u, adj) in graph.iter().enumerate() {
        for &(v, w) in adj {
            if v >= n {
                continue;
            }
            if h[u].saturating_add(w) < h[v] {
                return None;
            }
        }
    }

    // --- Step 3: build a reweighted graph with non-negative weights.
    // w'(u, v) = w(u, v) + h[u] - h[v]  >=  0.
    let mut reweighted: Vec<Vec<(usize, u64)>> = vec![Vec::new(); n];
    for (u, adj) in graph.iter().enumerate() {
        for &(v, w) in adj {
            if v >= n {
                continue;
            }
            // h[u] + w - h[v] is non-negative; cast safely via i128 to avoid
            // any saturating-arithmetic surprises on adversarial input.
            let new_w = i128::from(h[u]) + i128::from(w) - i128::from(h[v]);
            debug_assert!(new_w >= 0, "reweighting should produce non-negative edges");
            let new_w = if new_w < 0 { 0 } else { new_w as u64 };
            reweighted[u].push((v, new_w));
        }
    }

    // --- Step 4: Dijkstra from every source on the reweighted graph.
    // --- Step 5: undo the reweighting per pair.
    let mut result: Vec<Vec<Option<i64>>> = vec![vec![None; n]; n];
    for src in 0..n {
        let dist = dijkstra_reweighted(&reweighted, src);
        for v in 0..n {
            if dist[v] == u64::MAX {
                continue;
            }
            // original = dist'(src, v) - h[src] + h[v]
            let original = i128::from(dist[v]) - i128::from(h[src]) + i128::from(h[v]);
            result[src][v] = Some(original as i64);
        }
    }
    Some(result)
}

/// Internal Dijkstra over the non-negative reweighted graph. Returns
/// `u64::MAX` for unreachable vertices.
fn dijkstra_reweighted(graph: &[Vec<(usize, u64)>], start: usize) -> Vec<u64> {
    let n = graph.len();
    let mut dist = vec![u64::MAX; n];
    if start >= n {
        return dist;
    }
    dist[start] = 0;
    let mut heap = BinaryHeap::new();
    heap.push(State {
        cost: 0,
        node: start,
    });
    while let Some(State { cost, node }) = heap.pop() {
        if cost > dist[node] {
            continue;
        }
        for &(v, w) in &graph[node] {
            let next = cost.saturating_add(w);
            if next < dist[v] {
                dist[v] = next;
                heap.push(State {
                    cost: next,
                    node: v,
                });
            }
        }
    }
    dist
}

#[cfg(test)]
mod tests {
    use super::johnsons;
    use crate::graph::floyd_warshall::{floyd_warshall, INF};
    use quickcheck_macros::quickcheck;

    #[test]
    fn empty_graph() {
        let g: Vec<Vec<(usize, i64)>> = vec![];
        assert_eq!(johnsons(&g), Some(vec![]));
    }

    #[test]
    fn single_node() {
        let g: Vec<Vec<(usize, i64)>> = vec![vec![]];
        assert_eq!(johnsons(&g), Some(vec![vec![Some(0)]]));
    }

    #[test]
    fn disjoint_nodes_unreachable_is_none() {
        // Three isolated nodes: only the diagonals are reachable.
        let g: Vec<Vec<(usize, i64)>> = vec![vec![], vec![], vec![]];
        let want = vec![
            vec![Some(0), None, None],
            vec![None, Some(0), None],
            vec![None, None, Some(0)],
        ];
        assert_eq!(johnsons(&g), Some(want));
    }

    #[test]
    fn classic_positive_weights() {
        // 0 --1--> 1 --2--> 2
        //  \--4-----------> 2
        let g = vec![vec![(1, 1), (2, 4)], vec![(2, 2)], vec![]];
        let got = johnsons(&g).unwrap();
        assert_eq!(got[0][0], Some(0));
        assert_eq!(got[0][1], Some(1));
        assert_eq!(got[0][2], Some(3));
        assert_eq!(got[1][2], Some(2));
        assert_eq!(got[2][0], None);
        assert_eq!(got[2][1], None);
    }

    #[test]
    fn negative_edge_no_cycle() {
        // 0 --4--> 1 --(-3)--> 2
        // 0 --5--> 2
        let g = vec![vec![(1, 4), (2, 5)], vec![(2, -3)], vec![]];
        let got = johnsons(&g).unwrap();
        assert_eq!(got[0][2], Some(1)); // 4 + (-3) beats 5
        assert_eq!(got[0][1], Some(4));
        assert_eq!(got[1][2], Some(-3));
        assert_eq!(got[2][0], None);
    }

    #[test]
    fn negative_cycle_returns_none() {
        // 0 -> 1 -> 2 -> 0 with total weight -1.
        let g = vec![vec![(1, 1)], vec![(2, -1)], vec![(0, -1)]];
        assert_eq!(johnsons(&g), None);
    }

    #[test]
    fn unreachable_pairs_are_none() {
        // 0 -> 1, 2 is isolated.
        let g = vec![vec![(1, 7)], vec![], vec![]];
        let got = johnsons(&g).unwrap();
        assert_eq!(got[0][1], Some(7));
        assert_eq!(got[0][2], None);
        assert_eq!(got[1][0], None);
        assert_eq!(got[2][0], None);
        assert_eq!(got[2][1], None);
    }

    #[test]
    fn parallel_edges_take_minimum() {
        // 0 -> 1 with weights 5 and 2.
        let g = vec![vec![(1, 5), (1, 2)], vec![]];
        let got = johnsons(&g).unwrap();
        assert_eq!(got[0][1], Some(2));
    }

    // ---- property test: random small graphs vs Floyd-Warshall ----

    /// Convert `graph` into the dense matrix Floyd-Warshall expects. Returns
    /// `None` if any edge weight, when summed with itself n times, could
    /// blow past `INF`.
    fn to_dense_matrix(graph: &[Vec<(usize, i64)>]) -> Vec<Vec<i64>> {
        let n = graph.len();
        let mut m = vec![vec![INF; n]; n];
        for (u, row) in m.iter_mut().enumerate().take(n) {
            row[u] = 0;
        }
        for (u, adj) in graph.iter().enumerate() {
            for &(v, w) in adj {
                if w < m[u][v] {
                    m[u][v] = w;
                }
            }
        }
        m
    }

    /// Build a random small directed graph with `n` nodes (1..=6). Edges are
    /// drawn between every ordered pair (including reverse directions, so
    /// negative cycles are possible) using `mask` bits, with weights in
    /// [-5, 10] derived from `weight_seed` via a tiny LCG.
    fn random_graph(n: usize, mask: u64, weight_seed: u64) -> Vec<Vec<(usize, i64)>> {
        let mut g: Vec<Vec<(usize, i64)>> = vec![vec![]; n];
        let mut bit = 0u32;
        let mut wseed = weight_seed;
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                let present = (mask >> (bit % 64)) & 1 == 1;
                bit += 1;
                if present {
                    wseed = wseed
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1);
                    let w = (wseed >> 33) as i64 % 16 - 5; // [-5, 10]
                    g[i].push((j, w));
                }
            }
        }
        g
    }

    #[allow(clippy::needless_pass_by_value)]
    #[quickcheck]
    fn quickcheck_matches_floyd_warshall(n: u8, mask: u64, weight_seed: u64) -> bool {
        let n = ((n as usize) % 6) + 1; // 1..=6
        let g = random_graph(n, mask, weight_seed);

        let johnson = johnsons(&g);
        let fw = floyd_warshall(to_dense_matrix(&g));

        match (johnson, fw) {
            (None, Err(_)) => true,
            (Some(j), Ok(f)) => {
                for u in 0..n {
                    for v in 0..n {
                        let want = if f[u][v] >= INF { None } else { Some(f[u][v]) };
                        if j[u][v] != want {
                            return false;
                        }
                    }
                }
                true
            }
            // Disagreement on whether a negative cycle exists is a bug.
            _ => false,
        }
    }
}
