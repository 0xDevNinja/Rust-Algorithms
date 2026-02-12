//! Bridges and articulation points of an undirected graph via DFS with
//! discovery / low-link arrays. O(V + E).

/// Returns `(bridges, articulation_points)`.
///
/// `graph[u]` is the adjacency list of node `u`. The graph is treated as
/// undirected; every edge `{u, v}` must appear in both lists.
///
/// Bridges are returned as `(u, v)` pairs with `u < v`. Articulation points
/// are returned in ascending order, with no duplicates.
pub fn bridges_and_articulation(graph: &[Vec<usize>]) -> (Vec<(usize, usize)>, Vec<usize>) {
    let n = graph.len();
    let mut disc = vec![usize::MAX; n];
    let mut low = vec![0_usize; n];
    let mut is_articulation = vec![false; n];
    let mut bridges: Vec<(usize, usize)> = Vec::new();
    let mut timer = 0_usize;

    for u in 0..n {
        if disc[u] == usize::MAX {
            dfs(
                graph,
                u,
                usize::MAX,
                &mut disc,
                &mut low,
                &mut is_articulation,
                &mut bridges,
                &mut timer,
            );
        }
    }

    let mut articulation: Vec<usize> = is_articulation
        .iter()
        .enumerate()
        .filter_map(|(i, &flag)| if flag { Some(i) } else { None })
        .collect();
    articulation.sort_unstable();
    bridges.sort_unstable();
    (bridges, articulation)
}

#[allow(clippy::too_many_arguments)]
fn dfs(
    graph: &[Vec<usize>],
    u: usize,
    parent: usize,
    disc: &mut [usize],
    low: &mut [usize],
    is_articulation: &mut [bool],
    bridges: &mut Vec<(usize, usize)>,
    timer: &mut usize,
) {
    disc[u] = *timer;
    low[u] = *timer;
    *timer += 1;
    let mut child_count = 0_usize;
    for &v in &graph[u] {
        if disc[v] == usize::MAX {
            child_count += 1;
            dfs(graph, v, u, disc, low, is_articulation, bridges, timer);
            low[u] = low[u].min(low[v]);
            if low[v] > disc[u] {
                let (a, b) = if u < v { (u, v) } else { (v, u) };
                bridges.push((a, b));
            }
            if parent != usize::MAX && low[v] >= disc[u] {
                is_articulation[u] = true;
            }
        } else if v != parent {
            low[u] = low[u].min(disc[v]);
        }
    }
    if parent == usize::MAX && child_count > 1 {
        is_articulation[u] = true;
    }
}

#[cfg(test)]
mod tests {
    use super::bridges_and_articulation;

    fn undirected(edges: &[(usize, usize)], n: usize) -> Vec<Vec<usize>> {
        let mut g: Vec<Vec<usize>> = vec![Vec::new(); n];
        for &(u, v) in edges {
            g[u].push(v);
            g[v].push(u);
        }
        g
    }

    #[test]
    fn empty() {
        let (b, a) = bridges_and_articulation(&[]);
        assert!(b.is_empty());
        assert!(a.is_empty());
    }

    #[test]
    fn single_node() {
        let (b, a) = bridges_and_articulation(&[vec![]]);
        assert!(b.is_empty());
        assert!(a.is_empty());
    }

    #[test]
    fn line_graph_all_bridges() {
        // 0 - 1 - 2 - 3 ; bridges: every edge; articulation: 1, 2.
        let g = undirected(&[(0, 1), (1, 2), (2, 3)], 4);
        let (b, a) = bridges_and_articulation(&g);
        assert_eq!(b, vec![(0, 1), (1, 2), (2, 3)]);
        assert_eq!(a, vec![1, 2]);
    }

    #[test]
    fn cycle_no_bridges_no_articulation() {
        let g = undirected(&[(0, 1), (1, 2), (2, 3), (3, 0)], 4);
        let (b, a) = bridges_and_articulation(&g);
        assert!(b.is_empty());
        assert!(a.is_empty());
    }

    #[test]
    fn classic_two_triangles_through_bridge() {
        // Two triangles 0-1-2 and 3-4-5 connected by bridge 2-3.
        let edges = [(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 5), (5, 3)];
        let g = undirected(&edges, 6);
        let (b, a) = bridges_and_articulation(&g);
        assert_eq!(b, vec![(2, 3)]);
        assert_eq!(a, vec![2, 3]);
    }

    #[test]
    fn star_graph_centre_is_articulation() {
        // 0 connected to 1, 2, 3.
        let g = undirected(&[(0, 1), (0, 2), (0, 3)], 4);
        let (b, a) = bridges_and_articulation(&g);
        assert_eq!(b, vec![(0, 1), (0, 2), (0, 3)]);
        assert_eq!(a, vec![0]);
    }
}
