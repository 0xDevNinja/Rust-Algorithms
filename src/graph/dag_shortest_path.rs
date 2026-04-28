//! Single-source shortest path on a DAG via topological-order relaxation.
//! O(V + E) time and space. Negative edge weights are allowed because every
//! vertex is relaxed in topological order, so each edge is examined exactly
//! once after its source's distance is final.
//!
//! Returns `None` if the input graph contains a cycle.

use std::collections::VecDeque;

/// Returns shortest distances from `src` on a DAG. `graph[u]` lists outgoing
/// edges as `(neighbor, weight)` pairs. The result `dist` has `dist[src] =
/// Some(0)`, `dist[v] = Some(d)` for every reachable `v`, and `None` for
/// unreachable vertices. Returns `None` if `graph` contains a cycle.
///
/// # Panics
/// Panics if `src` is out of bounds for `graph`.
pub fn dag_shortest_path(graph: &[Vec<(usize, i64)>], src: usize) -> Option<Vec<Option<i64>>> {
    let n = graph.len();
    if n == 0 {
        return Some(Vec::new());
    }
    assert!(
        src < n,
        "dag_shortest_path: src {src} is out of bounds for graph of length {n}"
    );

    // Kahn's algorithm: compute in-degrees, then BFS.
    let mut in_degree = vec![0_usize; n];
    for adj in graph {
        for &(v, _) in adj {
            in_degree[v] += 1;
        }
    }
    let mut queue: VecDeque<usize> = (0..n).filter(|&i| in_degree[i] == 0).collect();
    let mut order: Vec<usize> = Vec::with_capacity(n);
    while let Some(u) = queue.pop_front() {
        order.push(u);
        for &(v, _) in &graph[u] {
            in_degree[v] -= 1;
            if in_degree[v] == 0 {
                queue.push_back(v);
            }
        }
    }
    if order.len() != n {
        return None;
    }

    // Relax in topological order. dist holds Option<i64>; None means unreached.
    let mut dist: Vec<Option<i64>> = vec![None; n];
    dist[src] = Some(0);
    for &u in &order {
        if let Some(du) = dist[u] {
            for &(v, w) in &graph[u] {
                let candidate = du.saturating_add(w);
                match dist[v] {
                    Some(dv) if dv <= candidate => {}
                    _ => dist[v] = Some(candidate),
                }
            }
        }
    }
    Some(dist)
}

#[cfg(test)]
mod tests {
    use super::dag_shortest_path;
    use quickcheck_macros::quickcheck;

    #[test]
    fn empty_graph() {
        let g: Vec<Vec<(usize, i64)>> = vec![];
        assert_eq!(dag_shortest_path(&g, 0), Some(vec![]));
    }

    #[test]
    fn single_node() {
        let g: Vec<Vec<(usize, i64)>> = vec![vec![]];
        assert_eq!(dag_shortest_path(&g, 0), Some(vec![Some(0)]));
    }

    #[test]
    fn simple_path() {
        // 0 --2--> 1 --3--> 2
        let g = vec![vec![(1, 2)], vec![(2, 3)], vec![]];
        assert_eq!(
            dag_shortest_path(&g, 0),
            Some(vec![Some(0), Some(2), Some(5)])
        );
    }

    #[test]
    fn multiple_paths_pick_min() {
        // 0 --1--> 1 --2--> 3
        // 0 --10-> 3
        // 0 --4--> 2 --1--> 3
        let g = vec![
            vec![(1, 1), (3, 10), (2, 4)],
            vec![(3, 2)],
            vec![(3, 1)],
            vec![],
        ];
        // best 0->3 is via 1: 1+2 = 3.
        assert_eq!(
            dag_shortest_path(&g, 0),
            Some(vec![Some(0), Some(1), Some(4), Some(3)])
        );
    }

    #[test]
    fn negative_weight_edge() {
        // 0 --5--> 1 --(-3)--> 2
        // 0 --10-> 2
        let g = vec![vec![(1, 5), (2, 10)], vec![(2, -3)], vec![]];
        assert_eq!(
            dag_shortest_path(&g, 0),
            Some(vec![Some(0), Some(5), Some(2)])
        );
    }

    #[test]
    fn unreachable_node() {
        // 0 -> 1, but 2 is isolated (no incoming, no outgoing useful from src).
        let g = vec![vec![(1, 4)], vec![], vec![]];
        assert_eq!(dag_shortest_path(&g, 0), Some(vec![Some(0), Some(4), None]));
    }

    #[test]
    fn cycle_returns_none() {
        // 0 -> 1 -> 2 -> 0
        let g = vec![vec![(1, 1)], vec![(2, 1)], vec![(0, 1)]];
        assert_eq!(dag_shortest_path(&g, 0), None);
    }

    #[test]
    fn parallel_edges() {
        // 0 -> 1 with weights 7 and 3 (parallel); shorter wins.
        let g = vec![vec![(1, 7), (1, 3)], vec![]];
        assert_eq!(dag_shortest_path(&g, 0), Some(vec![Some(0), Some(3)]));
    }

    #[test]
    fn src_not_zero() {
        // 0 -> 1 -> 2; start at 1.
        let g = vec![vec![(1, 5)], vec![(2, 6)], vec![]];
        assert_eq!(dag_shortest_path(&g, 1), Some(vec![None, Some(0), Some(6)]));
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn src_out_of_bounds_panics() {
        let g = vec![vec![(1, 1)], vec![]];
        let _ = dag_shortest_path(&g, 5);
    }

    // ---- property test against Bellman-Ford brute reference ----

    /// Bellman-Ford reference. Same `Option<i64>` shape as the function under
    /// test. Assumes no negative cycle (caller-controlled in tests).
    fn bellman_ford_reference(graph: &[Vec<(usize, i64)>], src: usize) -> Vec<Option<i64>> {
        let n = graph.len();
        let mut dist: Vec<Option<i64>> = vec![None; n];
        if n == 0 {
            return dist;
        }
        dist[src] = Some(0);
        for _ in 0..n {
            let mut updated = false;
            for u in 0..n {
                if let Some(du) = dist[u] {
                    for &(v, w) in &graph[u] {
                        let candidate = du.saturating_add(w);
                        match dist[v] {
                            Some(dv) if dv <= candidate => {}
                            _ => {
                                dist[v] = Some(candidate);
                                updated = true;
                            }
                        }
                    }
                }
            }
            if !updated {
                break;
            }
        }
        dist
    }

    /// Build a random DAG with `n` nodes (n in [1, 8]) where the only allowed
    /// edges are i -> j with i < j. Weights in [-10, 10]. The presence of each
    /// edge is determined by `mask` bits.
    fn random_dag(n: usize, mask: u64, weight_seed: u64) -> Vec<Vec<(usize, i64)>> {
        let mut g: Vec<Vec<(usize, i64)>> = vec![vec![]; n];
        let mut bit = 0u32;
        let mut wseed = weight_seed;
        for i in 0..n {
            for j in (i + 1)..n {
                let present = (mask >> (bit % 64)) & 1 == 1;
                bit += 1;
                if present {
                    // simple LCG to generate a weight in [-10, 10]
                    wseed = wseed
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1);
                    let w = (wseed >> 33) as i64 % 21 - 10;
                    g[i].push((j, w));
                }
            }
        }
        g
    }

    #[allow(clippy::needless_pass_by_value)]
    #[quickcheck]
    fn quickcheck_random_dag_matches_bellman_ford(n: u8, mask: u64, weight_seed: u64) -> bool {
        let n = ((n as usize) % 8) + 1; // 1..=8
        let g = random_dag(n, mask, weight_seed);
        let src = (weight_seed as usize) % n;
        let got = dag_shortest_path(&g, src).expect("constructed graph is acyclic");
        let want = bellman_ford_reference(&g, src);
        got == want
    }
}
