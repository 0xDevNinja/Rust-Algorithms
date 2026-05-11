//! Minimum cost-to-time ratio cycle in a directed graph (parametric search).
//!
//! Each directed edge carries a cost `c_e` (any real) and a strictly positive
//! time `t_e`. For every directed cycle `C` define
//!
//! ```text
//!     ratio(C) = (sum_{e in C} c_e) / (sum_{e in C} t_e)
//! ```
//!
//! The goal is to compute `min_C ratio(C)` over all directed cycles in the
//! graph. The classical reduction observes that a cycle with `ratio(C) < λ`
//! exists iff the graph reweighted with `w_e = c_e - λ · t_e` contains a
//! negative cycle. We binary-search on `λ` and use Bellman–Ford as the
//! negative-cycle oracle.
//!
//! Complexity: `O((V · E) · log((hi - lo) / tol))` time, `O(V)` extra space.

const ITER_CAP: usize = 200;

/// Returns `true` iff the directed graph with edge weights `c_e - lambda * t_e`
/// contains a negative cycle. We attach a virtual source 0-weight to every
/// vertex so that any negative cycle in the graph is detected, regardless of
/// reachability from a single chosen source.
fn has_negative_cycle(n: usize, edges: &[(usize, usize, f64, f64)], lambda: f64) -> bool {
    if n == 0 {
        return false;
    }
    let mut dist = vec![0.0f64; n];
    for _ in 0..n - 1 {
        let mut updated = false;
        for &(u, v, c, t) in edges {
            let w = lambda.mul_add(-t, c);
            let nd = dist[u] + w;
            if nd < dist[v] {
                dist[v] = nd;
                updated = true;
            }
        }
        if !updated {
            return false;
        }
    }
    for &(u, v, c, t) in edges {
        let w = lambda.mul_add(-t, c);
        if dist[u] + w < dist[v] {
            return true;
        }
    }
    false
}

/// Computes the minimum cost-to-time ratio over directed cycles of the graph.
///
/// `edges` lists directed arcs as `(from, to, cost, time)` with `time > 0`.
/// Returns `Some(ratio)` such that `|ratio - optimum| <= tol`, or `None` if
/// the graph is acyclic. Edges with non-positive `time` and out-of-range
/// endpoints are ignored.
///
/// The search bounds are derived from the edge data: any cycle ratio lies in
/// `[min(c_e / t_e), max(c_e / t_e)]`, so we initialise `lo`/`hi` from those
/// per-edge ratios and tighten by binary search until the window shrinks
/// below `tol` (or the iteration cap fires).
pub fn min_cost_time_ratio_cycle(
    n: usize,
    edges: &[(usize, usize, f64, f64)],
    tol: f64,
) -> Option<f64> {
    if n == 0 || edges.is_empty() {
        return None;
    }

    // Filter to valid edges (positive time, endpoints in range).
    let valid: Vec<(usize, usize, f64, f64)> = edges
        .iter()
        .copied()
        .filter(|&(u, v, _, t)| u < n && v < n && t > 0.0 && t.is_finite())
        .collect();
    if valid.is_empty() {
        return None;
    }

    // Establish search bounds from per-edge ratios; pad slightly so the
    // initial `hi` strictly upper-bounds any feasible cycle ratio.
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for &(_, _, c, t) in &valid {
        let r = c / t;
        if r < lo {
            lo = r;
        }
        if r > hi {
            hi = r;
        }
    }
    if !lo.is_finite() || !hi.is_finite() {
        return None;
    }
    let pad = (hi - lo).abs().max(1.0);
    lo -= pad;
    hi += pad;

    // No cycle at all if `c - hi*t` (very large positive λ pushed up) still
    // does not yield a negative cycle. Equivalently: feasibility check at the
    // padded upper bound.
    if !has_negative_cycle(n, &valid, hi) {
        return None;
    }

    let tol = tol.max(1e-12);
    let mut iter = 0;
    while hi - lo > tol && iter < ITER_CAP {
        let mid = (hi - lo).mul_add(0.5, lo);
        if has_negative_cycle(n, &valid, mid) {
            hi = mid;
        } else {
            lo = mid;
        }
        iter += 1;
    }
    Some((hi - lo).mul_add(0.5, lo))
}

#[cfg(test)]
mod tests {
    use super::min_cost_time_ratio_cycle;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn simple_two_cycle() {
        // 0 -> 1 (cost 3, time 2), 1 -> 0 (cost 1, time 2). Sum cost = 4,
        // sum time = 4 → ratio = 1.0.
        let edges = [(0usize, 1usize, 3.0, 2.0), (1, 0, 1.0, 2.0)];
        let r = min_cost_time_ratio_cycle(2, &edges, 1e-9).expect("cycle exists");
        assert!(approx(r, 1.0, 1e-6), "got {r}");
    }

    #[test]
    fn triangle_cycle() {
        // 0 -> 1 -> 2 -> 0 with costs 2, 4, 6 and times 1, 1, 4. Total cost
        // 12, total time 6 → ratio = 2.0.
        let edges = [
            (0usize, 1usize, 2.0, 1.0),
            (1, 2, 4.0, 1.0),
            (2, 0, 6.0, 4.0),
        ];
        let r = min_cost_time_ratio_cycle(3, &edges, 1e-9).expect("cycle exists");
        assert!(approx(r, 2.0, 1e-6), "got {r}");
    }

    #[test]
    fn dag_returns_none() {
        // Linear DAG: 0 -> 1 -> 2 -> 3.
        let edges = [
            (0usize, 1usize, 1.0, 1.0),
            (1, 2, 2.0, 1.0),
            (2, 3, 3.0, 1.0),
        ];
        assert!(min_cost_time_ratio_cycle(4, &edges, 1e-9).is_none());
    }

    #[test]
    fn picks_min_among_multiple_cycles() {
        // Two disjoint cycles: triangle 0-1-2-0 with ratio 3.0, and 2-cycle
        // 3-4-3 with ratio 0.5. The minimum is 0.5.
        let edges = [
            (0usize, 1usize, 3.0, 1.0),
            (1, 2, 3.0, 1.0),
            (2, 0, 3.0, 1.0),
            (3, 4, 1.0, 2.0),
            (4, 3, 1.0, 2.0),
        ];
        let r = min_cost_time_ratio_cycle(5, &edges, 1e-9).expect("cycle exists");
        assert!(approx(r, 0.5, 1e-6), "got {r}");
    }

    #[test]
    fn negative_ratio_cycle() {
        // Cycle with negative cost: 0 -> 1 (cost -4, time 1), 1 -> 0
        // (cost 1, time 1). Ratio = -3 / 2 = -1.5.
        let edges = [(0usize, 1usize, -4.0, 1.0), (1, 0, 1.0, 1.0)];
        let r = min_cost_time_ratio_cycle(2, &edges, 1e-9).expect("cycle exists");
        assert!(approx(r, -1.5, 1e-6), "got {r}");
    }

    #[test]
    fn self_loop_cycle() {
        // Self loop is the cheapest cycle: ratio = 2 / 5 = 0.4.
        let edges = [
            (0usize, 0usize, 2.0, 5.0),
            (0, 1, 10.0, 1.0),
            (1, 0, 10.0, 1.0),
        ];
        let r = min_cost_time_ratio_cycle(2, &edges, 1e-9).expect("cycle exists");
        assert!(approx(r, 0.4, 1e-6), "got {r}");
    }
}
