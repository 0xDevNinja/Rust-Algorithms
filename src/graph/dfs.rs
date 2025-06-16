//! Depth-first search on an unweighted graph (adjacency lists). Iterative.

/// Returns the visit order from `start`. Iterative implementation avoids
/// stack overflow on deep graphs.
pub fn dfs(graph: &[Vec<usize>], start: usize) -> Vec<usize> {
    let n = graph.len();
    if start >= n {
        return Vec::new();
    }
    let mut visited = vec![false; n];
    let mut order = Vec::new();
    let mut stack = vec![start];
    while let Some(u) = stack.pop() {
        if visited[u] {
            continue;
        }
        visited[u] = true;
        order.push(u);
        // Reverse so that the smallest neighbour is explored first (matches
        // recursive DFS order on a sorted adjacency list).
        for &v in graph[u].iter().rev() {
            if !visited[v] {
                stack.push(v);
            }
        }
    }
    order
}

#[cfg(test)]
mod tests {
    use super::dfs;

    #[test]
    fn line_graph() {
        let g = vec![vec![1], vec![0, 2], vec![1, 3], vec![2]];
        assert_eq!(dfs(&g, 0), vec![0, 1, 2, 3]);
    }

    #[test]
    fn branching() {
        let g = vec![vec![1, 2], vec![0, 3], vec![0, 4], vec![1], vec![2]];
        let order = dfs(&g, 0);
        assert_eq!(order[0], 0);
        assert_eq!(order.len(), 5);
        assert!(order.contains(&3));
        assert!(order.contains(&4));
    }

    #[test]
    fn disconnected_component_skipped() {
        let g = vec![vec![1], vec![0], vec![]];
        assert_eq!(dfs(&g, 0), vec![0, 1]);
    }
}
