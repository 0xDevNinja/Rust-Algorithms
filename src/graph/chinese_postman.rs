//! Chinese Postman Problem on an undirected, weighted, connected graph.
//!
//! Given a connected undirected multigraph with non-negative edge weights, the
//! *Chinese Postman Problem* asks for the minimum-weight closed walk that
//! traverses every edge at least once. The classical solution is:
//!
//! 1. Sum the weights of all edges — this is the cost paid even in the best
//!    case, since every edge must be used at least once.
//! 2. If every vertex has even degree, the graph is Eulerian and a single
//!    closed walk uses every edge exactly once. The answer is the total
//!    weight from step 1.
//! 3. Otherwise, let `O` be the (always even-sized) set of odd-degree
//!    vertices. Compute all-pairs shortest paths and find a perfect matching
//!    `M` of `O` that minimises the sum of shortest-path distances between
//!    matched pairs. Each matched pair contributes a duplicated path that
//!    "re-routes" the walk to make every degree even; the answer is
//!    `total_edge_weight + cost(M)`.
//!
//! All-pairs shortest paths are computed with Floyd–Warshall in `O(V^3)`
//! time. The minimum-weight perfect matching on the odd set is solved by a
//! bitmask DP over its `k = |O|` vertices in `O(2^k * k)` time. Because the
//! number of odd-degree vertices is bounded in practice (the implementation
//! supports up to `k = 20`, i.e. roughly `2 * 10^7` DP transitions), the
//! overall complexity is `O(V^3 + 2^k * k)`.
//!
//! The graph must be connected (ignoring isolated vertices is *not*
//! sufficient — every vertex `0..n` is treated as part of the graph). A
//! disconnected input causes a panic since no closed walk traversing every
//! edge can exist.

use super::floyd_warshall::{floyd_warshall, INF};

/// Maximum number of odd-degree vertices the bitmask DP will accept. With
/// `k = 20` the DP visits `2^20 ≈ 10^6` masks, each doing `O(k)` work.
const MAX_ODD: usize = 20;

/// Returns the minimum total weight of a closed walk that traverses every
/// edge of the undirected weighted graph at least once.
///
/// `edges` is a list of `(u, v, w)` triples with `0 <= u, v < n` and
/// non-negative weight `w`. Self-loops (`u == v`) and parallel edges are
/// allowed; a self-loop contributes `2` to its endpoint's degree and so does
/// not affect parity.
///
/// # Panics
///
/// * If any endpoint is out of range (`>= n`).
/// * If `n == 0`.
/// * If the graph is disconnected (treating every vertex in `0..n` as part of
///   the graph).
/// * If the number of odd-degree vertices exceeds [`MAX_ODD`] (`= 20`).
///
/// # Complexity
///
/// `O(V^3 + 2^k * k)` time and `O(V^2 + 2^k)` space, where `k` is the number
/// of odd-degree vertices.
pub fn chinese_postman_cost(n: usize, edges: &[(usize, usize, u64)]) -> u64 {
    assert!(n > 0, "graph must have at least one vertex");

    let mut degree = vec![0_usize; n];
    let mut total: u64 = 0;
    // Build the dense distance matrix, keeping the minimum direct edge weight
    // for parallel edges. Self-loops add to degree by 2 but never tighten the
    // distance from a vertex to itself (already 0).
    let mut dist = vec![vec![INF; n]; n];
    for i in 0..n {
        dist[i][i] = 0;
    }
    for &(u, v, w) in edges {
        assert!(u < n && v < n, "edge endpoint out of range");
        total = total
            .checked_add(w)
            .expect("total edge weight overflows u64");
        degree[u] += 1;
        degree[v] += 1;
        if u == v {
            continue;
        }
        let w_i = i64::try_from(w).expect("edge weight does not fit in i64");
        if w_i < dist[u][v] {
            dist[u][v] = w_i;
            dist[v][u] = w_i;
        }
    }

    // Connectivity check via undirected BFS over the original edges.
    assert!(is_connected(n, edges), "graph must be connected");

    let odd: Vec<usize> = (0..n).filter(|&u| !degree[u].is_multiple_of(2)).collect();
    if odd.is_empty() {
        return total;
    }
    assert!(
        odd.len() <= MAX_ODD,
        "too many odd-degree vertices ({}) for bitmask matching (max {})",
        odd.len(),
        MAX_ODD,
    );

    let sp = floyd_warshall(dist).expect("non-negative weights cannot create a negative cycle");

    let extra = min_weight_perfect_matching(&odd, &sp);
    total + extra
}

/// BFS-based undirected connectivity check on vertices `0..n`. Every vertex
/// in that range — including isolated ones — must be reachable from vertex
/// `0` for the function to return `true`.
fn is_connected(n: usize, edges: &[(usize, usize, u64)]) -> bool {
    let mut adj = vec![Vec::<usize>::new(); n];
    for &(u, v, _) in edges {
        if u != v {
            adj[u].push(v);
            adj[v].push(u);
        }
    }
    let mut visited = vec![false; n];
    let mut stack = vec![0_usize];
    visited[0] = true;
    let mut count = 1_usize;
    while let Some(u) = stack.pop() {
        for &v in &adj[u] {
            if !visited[v] {
                visited[v] = true;
                count += 1;
                stack.push(v);
            }
        }
    }
    count == n
}

/// Minimum-weight perfect matching on the vertex set `odd` using pairwise
/// distances from the all-pairs shortest-path matrix `sp`. Implemented as a
/// bitmask DP: `dp[mask]` is the minimum cost to perfectly match the subset
/// of `odd` indicated by the set bits of `mask`. The lowest set bit of each
/// mask is paired with every other set bit in turn.
///
/// `odd.len()` must be even and at most [`MAX_ODD`].
fn min_weight_perfect_matching(odd: &[usize], sp: &[Vec<i64>]) -> u64 {
    let k = odd.len();
    debug_assert!(k.is_multiple_of(2));
    debug_assert!(k <= MAX_ODD);

    let size = 1_usize << k;
    let mut dp = vec![u64::MAX; size];
    dp[0] = 0;

    for mask in 1..size {
        // The mask must have an even number of set bits to be reachable.
        if !(mask as u32).count_ones().is_multiple_of(2) {
            continue;
        }
        // Pair the lowest set bit with every other set bit.
        let i = mask.trailing_zeros() as usize;
        let rest = mask & !(1 << i);
        let mut bits = rest;
        while bits != 0 {
            let j = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            let prev_mask = mask & !(1 << i) & !(1 << j);
            let prev = dp[prev_mask];
            if prev == u64::MAX {
                continue;
            }
            let d = sp[odd[i]][odd[j]];
            assert!(
                d < INF,
                "graph is connected but odd vertices have no finite shortest path",
            );
            let candidate = prev + d as u64;
            if candidate < dp[mask] {
                dp[mask] = candidate;
            }
        }
    }

    let full = size - 1;
    assert!(dp[full] != u64::MAX, "no perfect matching found");
    dp[full]
}

#[cfg(test)]
mod tests {
    use super::chinese_postman_cost;

    #[test]
    fn single_edge_must_be_traversed_twice() {
        // K_2: degrees 1 and 1 are both odd, so the single edge has to be
        // walked once in each direction. Cost = 2 * w.
        assert_eq!(chinese_postman_cost(2, &[(0, 1, 7)]), 14);
    }

    #[test]
    fn triangle_is_eulerian() {
        // K_3 with weights 1, 2, 3. All degrees even -> answer is the sum.
        let edges = [(0, 1, 1_u64), (1, 2, 2), (2, 0, 3)];
        assert_eq!(chinese_postman_cost(3, &edges), 6);
    }

    #[test]
    fn bowtie_is_eulerian() {
        // Bowtie: two triangles glued at vertex 0. Vertex 0 has degree 4,
        // the others have degree 2 — all even, so the graph is Eulerian and
        // the answer is the sum of edge weights.
        let edges = [
            (0, 1, 1_u64),
            (1, 2, 2),
            (2, 0, 3),
            (0, 3, 4),
            (3, 4, 5),
            (4, 0, 6),
        ];
        assert_eq!(chinese_postman_cost(5, &edges), 1 + 2 + 3 + 4 + 5 + 6);
    }

    #[test]
    fn k4_complete_graph_needs_one_extra_edge() {
        // K_4 with unit weights: every vertex has degree 3 (odd). The four
        // odd vertices are 0,1,2,3. Minimum-weight perfect matching pairs
        // them so the duplicated paths sum to 2 (any two disjoint edges).
        // Total walk weight = 6 (sum of edges) + 2 = 8.
        let edges = [
            (0, 1, 1_u64),
            (0, 2, 1),
            (0, 3, 1),
            (1, 2, 1),
            (1, 3, 1),
            (2, 3, 1),
        ];
        assert_eq!(chinese_postman_cost(4, &edges), 8);
    }

    #[test]
    fn path_graph_duplicates_middle() {
        // Path 0 - 1 - 2 - 3 with weights 5, 1, 5. Odd vertices: 0 and 3.
        // Shortest path 0 -> 3 = 11. Walk cost = (5+1+5) + 11 = 22.
        let edges = [(0, 1, 5_u64), (1, 2, 1), (2, 3, 5)];
        assert_eq!(chinese_postman_cost(4, &edges), 22);
    }

    #[test]
    fn parallel_edges_keep_min_distance() {
        // Two vertices joined by two parallel edges of weights 3 and 5.
        // Each vertex has degree 2 (even), so the graph is Eulerian: walk
        // both edges and return. Cost = 3 + 5 = 8.
        let edges = [(0, 1, 3_u64), (0, 1, 5)];
        assert_eq!(chinese_postman_cost(2, &edges), 8);
    }

    #[test]
    fn self_loop_does_not_change_parity() {
        // Triangle 0-1-2 with a self-loop at vertex 0. Self-loop adds 2 to
        // vertex 0's degree, parities stay even, so the graph is Eulerian.
        let edges = [(0, 1, 2_u64), (1, 2, 2), (2, 0, 2), (0, 0, 7)];
        assert_eq!(chinese_postman_cost(3, &edges), 13);
    }

    #[test]
    fn matching_picks_cheaper_pairing() {
        // Four odd-degree vertices arranged so that one pairing is much
        // cheaper than the other. Path 0 - 1 - 2 - 3 with weight 1 each,
        // then a long detour edge 0 - 3 with weight 100.
        //
        // Edges and weights:
        //   (0,1)=1, (1,2)=1, (2,3)=1, (0,3)=100.
        // Degrees: 0:2, 1:2, 2:2, 3:2 -> all even, Eulerian, total = 103.
        //
        // To get odd vertices, add two spurs: (1,4)=2, (2,5)=2.
        // Degrees: 0:2, 1:3, 2:3, 3:2, 4:1, 5:1 -> odd set {1,2,4,5}.
        //
        // Shortest paths in the original (unaugmented) graph:
        //   d(1,2) = 1
        //   d(1,4) = 2,  d(1,5) = 1+2 = 3
        //   d(2,4) = 1+2 = 3,  d(2,5) = 2
        //   d(4,5) = 2+1+2 = 5
        // Pairings of {1,2,4,5}:
        //   {1-2, 4-5}: 1 + 5 = 6
        //   {1-4, 2-5}: 2 + 2 = 4   <- best
        //   {1-5, 2-4}: 3 + 3 = 6
        // Total edge weight = 1+1+1+100+2+2 = 107. Answer = 107 + 4 = 111.
        let edges = [
            (0, 1, 1_u64),
            (1, 2, 1),
            (2, 3, 1),
            (0, 3, 100),
            (1, 4, 2),
            (2, 5, 2),
        ];
        assert_eq!(chinese_postman_cost(6, &edges), 111);
    }

    #[test]
    fn six_odd_vertices_chooses_best_matching() {
        // Backbone path 0-1-2-3-4-5 (weight 1 each) with three weight-10
        // spurs at vertices 0, 2, 4.
        //
        //   Edges: (0,1)=1,(1,2)=1,(2,3)=1,(3,4)=1,(4,5)=1,
        //          (0,6)=10,(2,7)=10,(4,8)=10.
        //
        // Degrees: 0:2, 1:2, 2:3, 3:2, 4:3, 5:1, 6:1, 7:1, 8:1.
        // Odd set = {2, 4, 5, 6, 7, 8} (six vertices).
        //
        // Total edge weight = 5*1 + 3*10 = 35. The optimal perfect matching
        // on the odd set has cost 35 (multiple pairings tie, e.g.
        // {2-4, 5-8, 6-7} = 2 + 11 + 22). Final answer = 35 + 35 = 70.
        let edges = [
            (0, 1, 1_u64),
            (1, 2, 1),
            (2, 3, 1),
            (3, 4, 1),
            (4, 5, 1),
            (0, 6, 10),
            (2, 7, 10),
            (4, 8, 10),
        ];
        assert_eq!(chinese_postman_cost(9, &edges), 70);
    }

    #[test]
    #[should_panic(expected = "connected")]
    fn disconnected_panics() {
        // Two disjoint edges {0,1} and {2,3} — disconnected.
        let edges = [(0, 1, 1_u64), (2, 3, 1)];
        let _ = chinese_postman_cost(4, &edges);
    }

    #[test]
    #[should_panic(expected = "connected")]
    fn isolated_vertex_panics() {
        // Edge between 0 and 1, with vertex 2 isolated -> disconnected.
        let edges = [(0, 1, 5_u64)];
        let _ = chinese_postman_cost(3, &edges);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn out_of_range_endpoint_panics() {
        let edges = [(0, 5, 1_u64)];
        let _ = chinese_postman_cost(3, &edges);
    }

    #[test]
    #[should_panic(expected = "at least one vertex")]
    fn zero_vertices_panics() {
        let _ = chinese_postman_cost(0, &[]);
    }

    #[test]
    fn single_vertex_no_edges() {
        // n = 1, no edges: the empty closed walk has cost 0. Vertex 0 has
        // degree 0 (even) so the graph is trivially Eulerian.
        assert_eq!(chinese_postman_cost(1, &[]), 0);
    }
}
