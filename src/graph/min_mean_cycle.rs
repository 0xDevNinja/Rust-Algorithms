//! Karp's minimum mean weight cycle.
//!
//! Given a directed graph with arbitrary (possibly negative) edge weights,
//! finds the minimum mean weight `min_C (sum w(e)/|C|)` over all directed
//! cycles `C`. Returns `None` if the graph is acyclic.
//!
//! Algorithm (Karp 1978): augment the graph with a virtual source `s` that
//! has a zero-weight edge to every real vertex. Let `n` be the number of real
//! vertices and `d_k(v)` the minimum weight of an `s -> v` walk of exactly
//! `k` edges. Then
//!
//! ```text
//!     mu* = min_v  max_{0 <= k < n}  (d_n(v) - d_k(v)) / (n - k)
//! ```
//!
//! ranging over vertices `v` with `d_n(v)` finite. Complexity: `O(V·E)` time
//! and `O(V^2)` extra space for the dp table.
//!
//! Input is an adjacency list `adj[u] = [(v, w), ...]` of outgoing edges.

/// Returns the minimum mean weight over all directed cycles in `adj`, or
/// `None` if the graph contains no cycle. Edge weights may be negative.
pub fn karp_min_mean_cycle(adj: &[Vec<(usize, i64)>]) -> Option<f64> {
    let n = adj.len();
    if n == 0 {
        return None;
    }

    // d[k][v] = min weight of a walk of exactly k edges from virtual source
    // (which connects to every vertex with weight 0) to v. None = unreachable.
    let mut d: Vec<Vec<Option<i64>>> = vec![vec![None; n]; n + 1];
    for v in 0..n {
        d[0][v] = Some(0);
    }

    for k in 1..=n {
        for u in 0..n {
            if let Some(du) = d[k - 1][u] {
                for &(v, w) in &adj[u] {
                    let cand = du.saturating_add(w);
                    match d[k][v] {
                        Some(cur) if cur <= cand => {}
                        _ => d[k][v] = Some(cand),
                    }
                }
            }
        }
    }

    let mut best: Option<f64> = None;
    for v in 0..n {
        let dn = match d[n][v] {
            Some(x) => x as f64,
            None => continue,
        };
        // max_{0 <= k < n} (d_n(v) - d_k(v)) / (n - k)
        let mut worst: Option<f64> = None;
        for k in 0..n {
            if let Some(dk) = d[k][v] {
                let mean = (dn - dk as f64) / ((n - k) as f64);
                worst = Some(worst.map_or(mean, |cur| cur.max(mean)));
            }
        }
        if let Some(w) = worst {
            best = Some(best.map_or(w, |cur| cur.min(w)));
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::karp_min_mean_cycle;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn dag_has_no_cycle() {
        // 0 -> 1 -> 2, no cycles.
        let adj = vec![vec![(1usize, 5i64)], vec![(2, 7)], vec![]];
        assert_eq!(karp_min_mean_cycle(&adj), None);
    }

    #[test]
    fn empty_graph() {
        let adj: Vec<Vec<(usize, i64)>> = vec![];
        assert_eq!(karp_min_mean_cycle(&adj), None);
    }

    #[test]
    fn simple_two_cycle() {
        // 0 -> 1 (weight 1), 1 -> 0 (weight 3). Mean = (1+3)/2 = 2.
        let adj = vec![vec![(1usize, 1i64)], vec![(0, 3)]];
        let mu = karp_min_mean_cycle(&adj).expect("cycle exists");
        assert!(approx_eq(mu, 2.0), "got {mu}");
    }

    #[test]
    fn triangle_cycle() {
        // 0 -> 1 -> 2 -> 0 with weights 2, 4, 6. Mean = 12/3 = 4.
        let adj = vec![vec![(1usize, 2i64)], vec![(2, 4)], vec![(0, 6)]];
        let mu = karp_min_mean_cycle(&adj).expect("cycle exists");
        assert!(approx_eq(mu, 4.0), "got {mu}");
    }

    #[test]
    fn negative_weight_cycle() {
        // Triangle 0->1->2->0 with weights -1, -2, -3. Mean = -6/3 = -2.
        let adj = vec![vec![(1usize, -1i64)], vec![(2, -2)], vec![(0, -3)]];
        let mu = karp_min_mean_cycle(&adj).expect("cycle exists");
        assert!(approx_eq(mu, -2.0), "got {mu}");
    }

    #[test]
    fn picks_minimum_mean_among_multiple_cycles() {
        // Two cycles sharing nothing:
        //   A: 0 -> 1 -> 0 weights 1, 3 (mean 2)
        //   B: 2 -> 3 -> 2 weights 0, 2 (mean 1)
        // Algorithm should pick the smaller mean, 1.0.
        let adj = vec![
            vec![(1usize, 1i64)],
            vec![(0, 3)],
            vec![(3, 0)],
            vec![(2, 2)],
        ];
        let mu = karp_min_mean_cycle(&adj).expect("cycle exists");
        assert!(approx_eq(mu, 1.0), "got {mu}");
    }

    #[test]
    fn multi_component_one_acyclic_one_cyclic() {
        // Component A is a DAG: 0 -> 1.
        // Component B has a cycle: 2 -> 3 -> 2 with weights 5, 7. Mean = 6.
        let adj = vec![vec![(1usize, 4i64)], vec![], vec![(3, 5)], vec![(2, 7)]];
        let mu = karp_min_mean_cycle(&adj).expect("cycle exists in component B");
        assert!(approx_eq(mu, 6.0), "got {mu}");
    }

    #[test]
    fn self_loop_is_a_cycle() {
        // 0 -> 0 with weight -4. Mean = -4.
        let adj = vec![vec![(0usize, -4i64)]];
        let mu = karp_min_mean_cycle(&adj).expect("self-loop is a cycle");
        assert!(approx_eq(mu, -4.0), "got {mu}");
    }
}
