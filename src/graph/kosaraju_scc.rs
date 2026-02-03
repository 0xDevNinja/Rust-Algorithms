//! Kosaraju's strongly-connected components algorithm. Two DFS passes:
//! one to compute finish order on the forward graph, one on the transpose.
//! O(V + E).

/// Returns the strongly-connected components of `graph`. Each component is
/// sorted ascending; components themselves are returned in topological order
/// of the condensation (sources first).
pub fn kosaraju_scc(graph: &[Vec<usize>]) -> Vec<Vec<usize>> {
    let n = graph.len();

    // Pass 1: DFS on forward graph, record finish order.
    let mut visited = vec![false; n];
    let mut order = Vec::with_capacity(n);
    for u in 0..n {
        if !visited[u] {
            dfs_forward(graph, u, &mut visited, &mut order);
        }
    }

    // Build the transpose.
    let mut transpose: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (u, adj) in graph.iter().enumerate() {
        for &v in adj {
            transpose[v].push(u);
        }
    }

    // Pass 2: DFS on transpose in reverse finish order.
    let mut visited2 = vec![false; n];
    let mut components = Vec::new();
    for &u in order.iter().rev() {
        if !visited2[u] {
            let mut component = Vec::new();
            dfs_collect(&transpose, u, &mut visited2, &mut component);
            component.sort_unstable();
            components.push(component);
        }
    }
    components
}

fn dfs_forward(graph: &[Vec<usize>], u: usize, visited: &mut [bool], order: &mut Vec<usize>) {
    let mut stack: Vec<(usize, usize)> = vec![(u, 0)];
    visited[u] = true;
    while let Some(&mut (node, ref mut i)) = stack.last_mut() {
        if *i < graph[node].len() {
            let v = graph[node][*i];
            *i += 1;
            if !visited[v] {
                visited[v] = true;
                stack.push((v, 0));
            }
        } else {
            order.push(node);
            stack.pop();
        }
    }
}

fn dfs_collect(graph: &[Vec<usize>], u: usize, visited: &mut [bool], out: &mut Vec<usize>) {
    let mut stack = vec![u];
    visited[u] = true;
    while let Some(node) = stack.pop() {
        out.push(node);
        for &v in &graph[node] {
            if !visited[v] {
                visited[v] = true;
                stack.push(v);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::kosaraju_scc;

    fn normalise(mut comps: Vec<Vec<usize>>) -> Vec<Vec<usize>> {
        comps.iter_mut().for_each(|c| c.sort_unstable());
        comps.sort_by_key(|c| c[0]);
        comps
    }

    #[test]
    fn empty() {
        let g: Vec<Vec<usize>> = vec![];
        assert!(kosaraju_scc(&g).is_empty());
    }

    #[test]
    fn single_node() {
        let g = vec![vec![]];
        assert_eq!(kosaraju_scc(&g), vec![vec![0]]);
    }

    #[test]
    fn dag_is_all_singletons() {
        let g = vec![vec![1], vec![2], vec![]];
        let c = normalise(kosaraju_scc(&g));
        assert_eq!(c, vec![vec![0], vec![1], vec![2]]);
    }

    #[test]
    fn classic_example() {
        // Same as Tarjan's CLRS example.
        let g = vec![
            vec![1],
            vec![2, 4, 5],
            vec![3, 6],
            vec![2, 7],
            vec![0, 5],
            vec![6],
            vec![5, 7],
            vec![7],
        ];
        let c = normalise(kosaraju_scc(&g));
        assert_eq!(c, vec![vec![0, 1, 4], vec![2, 3], vec![5, 6], vec![7]]);
    }

    #[test]
    fn one_big_cycle() {
        let g = vec![vec![1], vec![2], vec![3], vec![0]];
        let c = normalise(kosaraju_scc(&g));
        assert_eq!(c, vec![vec![0, 1, 2, 3]]);
    }

    #[test]
    fn self_loop_alone() {
        let g = vec![vec![0]];
        assert_eq!(kosaraju_scc(&g), vec![vec![0]]);
    }
}
