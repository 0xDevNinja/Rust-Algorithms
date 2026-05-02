//! Hamiltonian path and cycle search via backtracking.
//!
//! A *Hamiltonian path* visits every vertex of a graph exactly once.
//! A *Hamiltonian cycle* additionally returns to its starting vertex,
//! forming a closed tour.
//!
//! Deciding whether either exists is **NP-hard** in general, so the worst-case
//! running time of these routines is `O(n!)` where `n` is the number of
//! vertices. The implementations below are intended for small graphs (rule of
//! thumb: `n` up to ~20). Do not pass huge graphs — the search will not finish
//! in reasonable time.
//!
//! Graphs are represented as adjacency lists: `adj[u]` is the list of vertices
//! adjacent to `u`. The graph is assumed simple and undirected (the caller is
//! responsible for symmetry), though directed graphs work too as long as the
//! adjacency list reflects the directed edges.

/// Returns a Hamiltonian path of the graph if one exists, as a vector of
/// vertex indices in visitation order, or `None` otherwise.
///
/// The search tries every vertex as a possible starting point and backtracks
/// on the first vertex from which a complete tour is found.
///
/// Edge cases:
/// - `n == 0` (empty graph) → `None`.
/// - `n == 1` → `Some(vec![0])` (the single vertex is a trivial path).
///
/// Worst-case complexity: `O(n!)` — Hamiltonian path is NP-hard.
pub fn hamiltonian_path(adj: &[Vec<usize>]) -> Option<Vec<usize>> {
    let n = adj.len();
    if n == 0 {
        return None;
    }
    if n == 1 {
        return Some(vec![0]);
    }
    let mut visited = vec![false; n];
    let mut path = Vec::with_capacity(n);
    for start in 0..n {
        visited[start] = true;
        path.push(start);
        if dfs_path(start, adj, &mut visited, &mut path, n) {
            return Some(path);
        }
        path.pop();
        visited[start] = false;
    }
    None
}

/// Returns a Hamiltonian cycle of the graph if one exists, as a vector of
/// vertex indices in visitation order. The starting vertex is **not**
/// repeated at the end; the cycle closes implicitly from the last element
/// back to the first.
///
/// By convention the search is rooted at vertex `0`: any Hamiltonian cycle
/// must contain every vertex, so fixing the start avoids redundant rotations.
///
/// Edge cases:
/// - `n == 0` → `None`.
/// - `n == 1` → `None` (a single vertex has no cycle without self-loops).
/// - `n == 2` → `None`. Closing the cycle would reuse the lone edge, which
///   is a multigraph cycle, not a simple Hamiltonian cycle. By convention a
///   Hamiltonian cycle requires `n >= 3`.
///
/// Worst-case complexity: `O(n!)` — Hamiltonian cycle is NP-hard.
pub fn hamiltonian_cycle(adj: &[Vec<usize>]) -> Option<Vec<usize>> {
    let n = adj.len();
    if n < 3 {
        return None;
    }
    let mut visited = vec![false; n];
    let mut path = Vec::with_capacity(n);
    visited[0] = true;
    path.push(0);
    if dfs_cycle(0, adj, &mut visited, &mut path, n) {
        Some(path)
    } else {
        None
    }
}

fn dfs_path(
    u: usize,
    adj: &[Vec<usize>],
    visited: &mut [bool],
    path: &mut Vec<usize>,
    n: usize,
) -> bool {
    if path.len() == n {
        return true;
    }
    for &v in &adj[u] {
        if !visited[v] {
            visited[v] = true;
            path.push(v);
            if dfs_path(v, adj, visited, path, n) {
                return true;
            }
            path.pop();
            visited[v] = false;
        }
    }
    false
}

fn dfs_cycle(
    u: usize,
    adj: &[Vec<usize>],
    visited: &mut [bool],
    path: &mut Vec<usize>,
    n: usize,
) -> bool {
    if path.len() == n {
        // Close back to the start vertex (0).
        return adj[u].contains(&0);
    }
    for &v in &adj[u] {
        if !visited[v] {
            visited[v] = true;
            path.push(v);
            if dfs_cycle(v, adj, visited, path, n) {
                return true;
            }
            path.pop();
            visited[v] = false;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::{hamiltonian_cycle, hamiltonian_path};

    /// Build an undirected adjacency list from an edge list and vertex count.
    fn undirected(n: usize, edges: &[(usize, usize)]) -> Vec<Vec<usize>> {
        let mut adj = vec![Vec::new(); n];
        for &(u, v) in edges {
            adj[u].push(v);
            adj[v].push(u);
        }
        adj
    }

    /// Is `p` a Hamiltonian path of `adj`?
    fn is_ham_path(adj: &[Vec<usize>], p: &[usize]) -> bool {
        let n = adj.len();
        if p.len() != n {
            return false;
        }
        let mut seen = vec![false; n];
        for &v in p {
            if v >= n || seen[v] {
                return false;
            }
            seen[v] = true;
        }
        for w in p.windows(2) {
            if !adj[w[0]].contains(&w[1]) {
                return false;
            }
        }
        true
    }

    /// Is `c` a Hamiltonian cycle (no repeated start at end)?
    fn is_ham_cycle(adj: &[Vec<usize>], c: &[usize]) -> bool {
        if !is_ham_path(adj, c) {
            return false;
        }
        let first = *c.first().unwrap();
        let last = *c.last().unwrap();
        adj[last].contains(&first)
    }

    /// Brute-force reference: enumerate every permutation of `0..n` and check
    /// whether any of them is a Hamiltonian path.
    fn brute_has_ham_path(adj: &[Vec<usize>]) -> bool {
        let n = adj.len();
        if n == 0 {
            return false;
        }
        let mut perm: Vec<usize> = (0..n).collect();
        permute_check(&mut perm, 0, &|p| is_ham_path(adj, p))
    }

    fn brute_has_ham_cycle(adj: &[Vec<usize>]) -> bool {
        let n = adj.len();
        // Hamiltonian cycle by convention requires n >= 3 (no reusing edges).
        if n < 3 {
            return false;
        }
        let mut perm: Vec<usize> = (0..n).collect();
        permute_check(&mut perm, 0, &|p| is_ham_cycle(adj, p))
    }

    fn permute_check(perm: &mut [usize], k: usize, pred: &dyn Fn(&[usize]) -> bool) -> bool {
        if k == perm.len() {
            return pred(perm);
        }
        for i in k..perm.len() {
            perm.swap(k, i);
            if permute_check(perm, k + 1, pred) {
                return true;
            }
            perm.swap(k, i);
        }
        false
    }

    #[test]
    fn empty_graph() {
        let adj: Vec<Vec<usize>> = Vec::new();
        assert_eq!(hamiltonian_path(&adj), None);
        assert_eq!(hamiltonian_cycle(&adj), None);
    }

    #[test]
    fn single_vertex() {
        let adj: Vec<Vec<usize>> = vec![Vec::new()];
        assert_eq!(hamiltonian_path(&adj), Some(vec![0]));
        assert_eq!(hamiltonian_cycle(&adj), None);
    }

    #[test]
    fn k2_path_yes_cycle_no() {
        // Two vertices with a single edge: path exists, cycle does not
        // (would require revisiting the only edge).
        let adj = undirected(2, &[(0, 1)]);
        let p = hamiltonian_path(&adj).expect("K2 has a Hamiltonian path");
        assert!(is_ham_path(&adj, &p));
        assert_eq!(hamiltonian_cycle(&adj), None);
    }

    #[test]
    fn triangle_k3() {
        let adj = undirected(3, &[(0, 1), (1, 2), (0, 2)]);
        let p = hamiltonian_path(&adj).expect("K3 has a Hamiltonian path");
        assert!(is_ham_path(&adj, &p));
        let c = hamiltonian_cycle(&adj).expect("K3 has a Hamiltonian cycle");
        assert!(is_ham_cycle(&adj, &c));
    }

    #[test]
    fn complete_k4() {
        let adj = undirected(4, &[(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]);
        let p = hamiltonian_path(&adj).expect("K4 has a Hamiltonian path");
        assert!(is_ham_path(&adj, &p));
        let c = hamiltonian_cycle(&adj).expect("K4 has a Hamiltonian cycle");
        assert!(is_ham_cycle(&adj, &c));
    }

    #[test]
    fn path_graph_p5() {
        // 0 - 1 - 2 - 3 - 4 : Hamiltonian path yes, cycle no.
        let adj = undirected(5, &[(0, 1), (1, 2), (2, 3), (3, 4)]);
        let p = hamiltonian_path(&adj).expect("P5 has a Hamiltonian path");
        assert!(is_ham_path(&adj, &p));
        assert_eq!(hamiltonian_cycle(&adj), None);
    }

    #[test]
    fn disconnected_graph() {
        // Two K2 components: no Hamiltonian path (cannot cross components),
        // hence no Hamiltonian cycle either.
        let adj = undirected(4, &[(0, 1), (2, 3)]);
        assert_eq!(hamiltonian_path(&adj), None);
        assert_eq!(hamiltonian_cycle(&adj), None);
    }

    #[test]
    fn bipartite_k_2_3_no_cycle() {
        // K_{2,3}: parts {0,1} and {2,3,4}. A Hamiltonian cycle in a bipartite
        // graph requires both parts to have equal size, so K_{2,3} has none.
        // A Hamiltonian path is also impossible: any path alternates parts,
        // so a 5-vertex path needs parts of size 3 and 2 starting and ending
        // in the larger one, but no edges exist within {2,3,4}.
        let adj = undirected(5, &[(0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4)]);
        // A Hamiltonian path *does* exist in K_{2,3}: e.g. 2-0-3-1-4.
        let p = hamiltonian_path(&adj).expect("K_{2,3} has a Hamiltonian path");
        assert!(is_ham_path(&adj, &p));
        // But no Hamiltonian cycle.
        assert_eq!(hamiltonian_cycle(&adj), None);
    }

    #[test]
    fn matches_brute_force_small_graphs() {
        // Hand-picked small graphs (n <= 6); compare backtracker against
        // brute-force permutation enumeration.
        let cases: Vec<Vec<Vec<usize>>> = vec![
            // empty edges, n=3
            undirected(3, &[]),
            // path on 4
            undirected(4, &[(0, 1), (1, 2), (2, 3)]),
            // star on 5 (center 0)
            undirected(5, &[(0, 1), (0, 2), (0, 3), (0, 4)]),
            // cycle C6
            undirected(6, &[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]),
            // K4 minus one edge
            undirected(4, &[(0, 1), (0, 2), (0, 3), (1, 2), (2, 3)]),
            // K_{2,3}
            undirected(5, &[(0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4)]),
            // Two triangles sharing a vertex (n=5)
            undirected(5, &[(0, 1), (1, 2), (0, 2), (2, 3), (3, 4), (2, 4)]),
        ];
        for adj in &cases {
            let path_found = hamiltonian_path(adj);
            let path_brute = brute_has_ham_path(adj);
            assert_eq!(path_found.is_some(), path_brute, "path mismatch on {adj:?}");
            if let Some(ref p) = path_found {
                assert!(is_ham_path(adj, p));
            }

            let cycle_found = hamiltonian_cycle(adj);
            let cycle_brute = brute_has_ham_cycle(adj);
            assert_eq!(
                cycle_found.is_some(),
                cycle_brute,
                "cycle mismatch on {adj:?}"
            );
            if let Some(ref c) = cycle_found {
                assert!(is_ham_cycle(adj, c));
            }
        }
    }
}
