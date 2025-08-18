//! Topological sort via Kahn's algorithm (BFS on in-degrees). O(V + E).

use std::collections::VecDeque;

/// Returns a topological ordering of the DAG `graph`. If `graph` contains a
/// cycle, returns `Err` with the partial ordering.
pub fn topological_sort(graph: &[Vec<usize>]) -> Result<Vec<usize>, Vec<usize>> {
    let n = graph.len();
    let mut in_degree = vec![0_usize; n];
    for adj in graph {
        for &v in adj {
            in_degree[v] += 1;
        }
    }
    let mut queue: VecDeque<usize> = (0..n).filter(|&i| in_degree[i] == 0).collect();
    let mut order = Vec::with_capacity(n);
    while let Some(u) = queue.pop_front() {
        order.push(u);
        for &v in &graph[u] {
            in_degree[v] -= 1;
            if in_degree[v] == 0 {
                queue.push_back(v);
            }
        }
    }
    if order.len() == n {
        Ok(order)
    } else {
        Err(order)
    }
}

#[cfg(test)]
mod tests {
    use super::topological_sort;

    #[test]
    fn linear_chain() {
        // 0 -> 1 -> 2 -> 3
        let g = vec![vec![1], vec![2], vec![3], vec![]];
        assert_eq!(topological_sort(&g).unwrap(), vec![0, 1, 2, 3]);
    }

    #[test]
    fn diamond_valid_orderings() {
        // 0 -> 1, 0 -> 2, 1 -> 3, 2 -> 3
        let g = vec![vec![1, 2], vec![3], vec![3], vec![]];
        let order = topological_sort(&g).unwrap();
        let pos: std::collections::HashMap<usize, usize> =
            order.iter().enumerate().map(|(i, &v)| (v, i)).collect();
        assert!(pos[&0] < pos[&1] && pos[&0] < pos[&2]);
        assert!(pos[&1] < pos[&3] && pos[&2] < pos[&3]);
    }

    #[test]
    fn cycle_detected() {
        let g = vec![vec![1], vec![2], vec![0]];
        assert!(topological_sort(&g).is_err());
    }

    #[test]
    fn empty() {
        let g: Vec<Vec<usize>> = vec![];
        assert_eq!(topological_sort(&g).unwrap(), Vec::<usize>::new());
    }
}
