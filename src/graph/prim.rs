//! Prim's minimum spanning tree using a binary heap. O((V + E) log V).

use std::cmp::Ordering;
use std::collections::BinaryHeap;

#[derive(Copy, Clone, Eq, PartialEq)]
struct HeapItem {
    weight: i64,
    node: usize,
    parent: usize,
}

impl Ord for HeapItem {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .weight
            .cmp(&self.weight)
            .then_with(|| self.node.cmp(&other.node))
    }
}

impl PartialOrd for HeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Returns the total MST weight. `graph[u]` is a list of `(neighbour, weight)`
/// edges; the graph is treated as undirected (every edge must appear in both
/// adjacency lists).
///
/// Returns `None` if the graph is disconnected.
pub fn prim(graph: &[Vec<(usize, i64)>], start: usize) -> Option<i64> {
    let n = graph.len();
    if n == 0 || start >= n {
        return Some(0);
    }
    let mut in_tree = vec![false; n];
    let mut heap = BinaryHeap::new();
    heap.push(HeapItem {
        weight: 0,
        node: start,
        parent: start,
    });
    let mut total: i64 = 0;
    let mut visited = 0;
    while let Some(HeapItem { weight, node, .. }) = heap.pop() {
        if in_tree[node] {
            continue;
        }
        in_tree[node] = true;
        total += weight;
        visited += 1;
        for &(v, w) in &graph[node] {
            if !in_tree[v] {
                heap.push(HeapItem {
                    weight: w,
                    node: v,
                    parent: node,
                });
            }
        }
    }
    if visited == n {
        Some(total)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::prim;

    #[test]
    fn triangle() {
        // 0-1 (1), 1-2 (2), 0-2 (5)
        let g = vec![
            vec![(1, 1), (2, 5)],
            vec![(0, 1), (2, 2)],
            vec![(0, 5), (1, 2)],
        ];
        assert_eq!(prim(&g, 0), Some(3));
    }

    #[test]
    fn disconnected_returns_none() {
        let g = vec![vec![(1, 1)], vec![(0, 1)], vec![]];
        assert_eq!(prim(&g, 0), None);
    }

    #[test]
    fn single_node_zero_weight() {
        let g: Vec<Vec<(usize, i64)>> = vec![vec![]];
        assert_eq!(prim(&g, 0), Some(0));
    }
}
