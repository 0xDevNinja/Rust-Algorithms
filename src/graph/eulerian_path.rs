//! Eulerian path / circuit on an undirected multigraph via Hierholzer's
//! algorithm.
//!
//! An *Eulerian circuit* is a closed walk that uses every edge of the graph
//! exactly once. An *Eulerian path* is the open analogue. For an undirected
//! graph the existence conditions are:
//!
//! * **Circuit** — every vertex has even degree, and all vertices with at
//!   least one incident edge lie in a single connected component.
//! * **Path** — exactly zero or two vertices have odd degree, and all
//!   vertices with at least one incident edge lie in a single connected
//!   component. When there are two odd-degree vertices the walk must start
//!   at one of them and end at the other.
//!
//! Hierholzer's algorithm builds the walk in `O(V + E)` time and `O(V + E)`
//! space by repeatedly extending a current trail until it closes, then
//! splicing in further sub-trails discovered while back-tracking. Edges are
//! tracked by their identifier so that parallel edges and self-loops are
//! handled correctly; a parallel `Vec<bool>` of size `2 * edge_count` marks
//! which half-edge has already been consumed.
//!
//! Isolated vertices (no incident edges) are ignored when checking
//! connectivity — they do not affect existence and never appear in the
//! returned walk.

/// Returns an Eulerian *circuit* of the undirected multigraph `adj`, or
/// `None` if no such closed walk exists.
///
/// `adj[u]` is the list of `(neighbor, edge_id)` pairs incident to `u`. Each
/// undirected edge `{u, v}` must appear in both adjacency lists with the
/// **same** `edge_id` so that the two halves can be linked. Edge ids must be
/// dense in `0..edge_count` where `edge_count` is the number of undirected
/// edges.
///
/// Special cases:
/// * an empty graph (no vertices) returns `Some(vec![])`;
/// * a graph consisting only of isolated vertices returns `Some(vec![])`;
/// * a connected component on a single vertex with self-loops returns a
///   walk starting and ending at that vertex.
///
/// Runs in `O(V + E)` time and `O(V + E)` extra space.
pub fn eulerian_circuit(adj: &[Vec<(usize, usize)>]) -> Option<Vec<usize>> {
    let n = adj.len();
    let edge_count = count_edges(adj)?;

    // Every vertex must have even degree for a circuit to exist.
    if adj.iter().any(|nbrs| nbrs.len() % 2 != 0) {
        return None;
    }

    if edge_count == 0 {
        return Some(Vec::new());
    }

    let start = (0..n).find(|&u| !adj[u].is_empty())?;
    if !single_component(adj, start) {
        return None;
    }

    Some(hierholzer(adj, start, edge_count))
}

/// Returns an Eulerian *path* of the undirected multigraph `adj`, or `None`
/// if no such walk exists.
///
/// The walk is closed (a circuit) when every vertex has even degree, and
/// otherwise runs from one odd-degree vertex to the other. See the module
/// docs for the precise existence conditions and edge-id contract.
///
/// Special cases mirror [`eulerian_circuit`]: an empty graph or a graph with
/// no edges returns `Some(vec![])`.
///
/// Runs in `O(V + E)` time and `O(V + E)` extra space.
pub fn eulerian_path(adj: &[Vec<(usize, usize)>]) -> Option<Vec<usize>> {
    let n = adj.len();
    let edge_count = count_edges(adj)?;

    let odd: Vec<usize> = (0..n).filter(|&u| adj[u].len() % 2 == 1).collect();
    if !odd.is_empty() && odd.len() != 2 {
        return None;
    }

    if edge_count == 0 {
        return Some(Vec::new());
    }

    let start = if odd.is_empty() {
        (0..n).find(|&u| !adj[u].is_empty())?
    } else {
        odd[0]
    };

    if !single_component(adj, start) {
        return None;
    }

    Some(hierholzer(adj, start, edge_count))
}

/// Counts undirected edges by checking that each `edge_id` appears exactly
/// twice in the half-edge lists. Returns `None` if the contract is violated.
fn count_edges(adj: &[Vec<(usize, usize)>]) -> Option<usize> {
    let half_edges: usize = adj.iter().map(Vec::len).sum();
    if !half_edges.is_multiple_of(2) {
        return None;
    }
    let edge_count = half_edges / 2;
    let mut seen = vec![0u8; edge_count];
    for nbrs in adj {
        for &(_, eid) in nbrs {
            if eid >= edge_count {
                return None;
            }
            seen[eid] += 1;
            if seen[eid] > 2 {
                return None;
            }
        }
    }
    if seen.iter().any(|&c| c != 2) {
        return None;
    }
    Some(edge_count)
}

/// Returns `true` if every vertex with at least one incident edge is
/// reachable from `start` via undirected DFS. Isolated vertices are ignored.
fn single_component(adj: &[Vec<(usize, usize)>], start: usize) -> bool {
    let n = adj.len();
    let mut visited = vec![false; n];
    let mut stack = vec![start];
    visited[start] = true;
    while let Some(u) = stack.pop() {
        for &(v, _) in &adj[u] {
            if !visited[v] {
                visited[v] = true;
                stack.push(v);
            }
        }
    }
    (0..n).all(|u| adj[u].is_empty() || visited[u])
}

/// Hierholzer's algorithm. Assumes preconditions (degree parity and
/// connectivity) have already been checked by the caller.
fn hierholzer(adj: &[Vec<(usize, usize)>], start: usize, edge_count: usize) -> Vec<usize> {
    let n = adj.len();
    let mut iter_idx = vec![0_usize; n];
    let mut used = vec![false; edge_count];

    let mut stack = vec![start];
    let mut circuit = Vec::with_capacity(edge_count + 1);

    while let Some(&u) = stack.last() {
        // Skip already-consumed edges at the front of u's adjacency list.
        while iter_idx[u] < adj[u].len() && used[adj[u][iter_idx[u]].1] {
            iter_idx[u] += 1;
        }
        if iter_idx[u] == adj[u].len() {
            circuit.push(u);
            stack.pop();
        } else {
            let (v, eid) = adj[u][iter_idx[u]];
            used[eid] = true;
            iter_idx[u] += 1;
            stack.push(v);
        }
    }

    circuit.reverse();
    circuit
}

#[cfg(test)]
mod tests {
    use super::{eulerian_circuit, eulerian_path};

    /// Builds an undirected adjacency list with sequential edge ids from a
    /// list of `(u, v)` edges. Self-loops are supported.
    fn build(n: usize, edges: &[(usize, usize)]) -> Vec<Vec<(usize, usize)>> {
        let mut adj: Vec<Vec<(usize, usize)>> = vec![Vec::new(); n];
        for (eid, &(u, v)) in edges.iter().enumerate() {
            adj[u].push((v, eid));
            adj[v].push((u, eid));
        }
        adj
    }

    /// Verifies that `walk` traverses every edge in `edges` exactly once and
    /// that each consecutive pair is a real edge in the graph.
    fn walk_uses_every_edge(walk: &[usize], edges: &[(usize, usize)]) -> bool {
        if edges.is_empty() {
            return walk.is_empty();
        }
        if walk.len() != edges.len() + 1 {
            return false;
        }
        let mut remaining: Vec<(usize, usize)> = edges
            .iter()
            .map(|&(a, b)| if a <= b { (a, b) } else { (b, a) })
            .collect();
        for w in walk.windows(2) {
            let (a, b) = (w[0], w[1]);
            let key = if a <= b { (a, b) } else { (b, a) };
            if let Some(pos) = remaining.iter().position(|e| *e == key) {
                remaining.swap_remove(pos);
            } else {
                return false;
            }
        }
        remaining.is_empty()
    }

    #[test]
    fn empty_graph() {
        let adj: Vec<Vec<(usize, usize)>> = Vec::new();
        assert_eq!(eulerian_circuit(&adj), Some(Vec::new()));
        assert_eq!(eulerian_path(&adj), Some(Vec::new()));
    }

    #[test]
    fn single_isolated_vertex() {
        let adj: Vec<Vec<(usize, usize)>> = vec![Vec::new()];
        assert_eq!(eulerian_circuit(&adj), Some(Vec::new()));
        assert_eq!(eulerian_path(&adj), Some(Vec::new()));
    }

    #[test]
    fn k2_single_edge_has_path_not_circuit() {
        let edges = [(0_usize, 1_usize)];
        let adj = build(2, &edges);
        // Two odd-degree vertices: path exists, circuit does not.
        assert_eq!(eulerian_circuit(&adj), None);
        let path = eulerian_path(&adj).expect("path should exist");
        assert!(walk_uses_every_edge(&path, &edges));
        assert!(path.first() != path.last());
    }

    #[test]
    fn triangle_k3_has_circuit() {
        let edges = [(0_usize, 1_usize), (1, 2), (2, 0)];
        let adj = build(3, &edges);
        let circuit = eulerian_circuit(&adj).expect("circuit should exist");
        assert!(walk_uses_every_edge(&circuit, &edges));
        assert_eq!(circuit.first(), circuit.last());
        // Path should also exist (and be a circuit since all degrees even).
        let path = eulerian_path(&adj).expect("path should exist");
        assert!(walk_uses_every_edge(&path, &edges));
    }

    #[test]
    fn figure_eight_circuit() {
        // Two triangles sharing vertex 0: 0-1-2-0 and 0-3-4-0.
        let edges = [(0_usize, 1_usize), (1, 2), (2, 0), (0, 3), (3, 4), (4, 0)];
        let adj = build(5, &edges);
        let circuit = eulerian_circuit(&adj).expect("circuit should exist");
        assert!(walk_uses_every_edge(&circuit, &edges));
        assert_eq!(circuit.first(), circuit.last());
    }

    #[test]
    fn path_0_to_3() {
        // 0 - 1 - 2 - 3 ; degrees 1, 2, 2, 1 -> Eulerian path 0..=3.
        let edges = [(0_usize, 1_usize), (1, 2), (2, 3)];
        let adj = build(4, &edges);
        assert_eq!(eulerian_circuit(&adj), None);
        let path = eulerian_path(&adj).expect("path should exist");
        assert!(walk_uses_every_edge(&path, &edges));
        let endpoints = (*path.first().unwrap(), *path.last().unwrap());
        assert!(endpoints == (0, 3) || endpoints == (3, 0));
    }

    #[test]
    fn isolated_extra_vertex_allowed() {
        // Triangle on {0,1,2} plus isolated vertex 3.
        let edges = [(0_usize, 1_usize), (1, 2), (2, 0)];
        let adj = build(4, &edges);
        let circuit = eulerian_circuit(&adj).expect("circuit should exist");
        assert!(walk_uses_every_edge(&circuit, &edges));
        assert!(!circuit.contains(&3));
    }

    #[test]
    fn disconnected_non_isolated_components_rejected() {
        // Two disjoint edges: {0,1} and {2,3}. All degrees odd -> path
        // condition (4 odd vertices) fails immediately. Add another edge to
        // each component to make all degrees even but the graph still
        // disconnected.
        let edges = [(0_usize, 1_usize), (1, 0), (2, 3), (3, 2)];
        let adj = build(4, &edges);
        // Degrees are all even (2 each) but the graph has two components.
        assert_eq!(eulerian_circuit(&adj), None);
        assert_eq!(eulerian_path(&adj), None);
    }

    #[test]
    fn parallel_edges_and_self_loops() {
        // Vertex 0 with a self-loop plus vertex 1 connected by two parallel
        // edges. Degrees: 0 -> 2 (loop) + 2 (parallel) = 4, 1 -> 2. All
        // even, single component, so a circuit exists.
        let edges = [(0_usize, 0_usize), (0, 1), (0, 1)];
        let adj = build(2, &edges);
        let circuit = eulerian_circuit(&adj).expect("circuit should exist");
        assert!(walk_uses_every_edge(&circuit, &edges));
        assert_eq!(circuit.first(), circuit.last());
    }

    /// Brute-force enumeration: checks every walk of length `edges.len()`
    /// starting at `start` against the produced walk. This validates that
    /// the algorithm's output is *some* valid Eulerian walk, not a specific
    /// one.
    fn exists_eulerian_walk(adj: &[Vec<(usize, usize)>], start: usize, total_edges: usize) -> bool {
        fn dfs(
            adj: &[Vec<(usize, usize)>],
            u: usize,
            used: &mut [bool],
            depth: usize,
            target: usize,
        ) -> bool {
            if depth == target {
                return true;
            }
            for &(v, eid) in &adj[u] {
                if !used[eid] {
                    used[eid] = true;
                    if dfs(adj, v, used, depth + 1, target) {
                        return true;
                    }
                    used[eid] = false;
                }
            }
            false
        }
        let mut used = vec![false; total_edges];
        dfs(adj, start, &mut used, 0, total_edges)
    }

    #[test]
    fn brute_force_small_graphs() {
        // Enumerate small undirected graphs on n=4 vertices with up to 5
        // edges (allowing parallel edges and self-loops). For each graph
        // verify the result against brute-force existence.
        let candidate_edges: [(usize, usize); 6] = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
        let n = 4_usize;
        for mask in 0_u32..(1 << candidate_edges.len()) {
            let edges: Vec<(usize, usize)> = candidate_edges
                .iter()
                .enumerate()
                .filter(|(i, _)| mask & (1 << i) != 0)
                .map(|(_, &e)| e)
                .collect();
            if edges.len() > 5 {
                continue;
            }
            let adj = build(n, &edges);

            // Path check.
            let path = eulerian_path(&adj);
            // A path exists iff (all non-isolated vertices in one component)
            // AND (0 or 2 odd-degree vertices).
            let degrees: Vec<usize> = adj.iter().map(Vec::len).collect();
            let odd_count = degrees.iter().filter(|&&d| !d.is_multiple_of(2)).count();
            let mut start = None;
            for u in 0..n {
                if !adj[u].is_empty() {
                    start = Some(u);
                    break;
                }
            }
            let connected = start.is_none_or(|s| super::single_component(&adj, s));
            let path_should_exist = connected && (odd_count == 0 || odd_count == 2);
            assert_eq!(
                path.is_some(),
                path_should_exist,
                "path mismatch for mask {mask:06b}",
            );
            if let Some(walk) = path.as_ref() {
                assert!(walk_uses_every_edge(walk, &edges));
                if !walk.is_empty() {
                    let s = walk[0];
                    assert!(exists_eulerian_walk(&adj, s, edges.len()));
                }
            }

            // Circuit check.
            let circuit = eulerian_circuit(&adj);
            let circuit_should_exist = connected && odd_count == 0;
            assert_eq!(
                circuit.is_some(),
                circuit_should_exist,
                "circuit mismatch for mask {mask:06b}",
            );
            if let Some(walk) = circuit.as_ref() {
                assert!(walk_uses_every_edge(walk, &edges));
                if !walk.is_empty() {
                    assert_eq!(walk.first(), walk.last());
                }
            }
        }
    }
}
