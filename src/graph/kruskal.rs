//! Kruskal's minimum spanning tree using the shared union-find data
//! structure. O(E log E).

use crate::data_structures::union_find::UnionFind;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Edge {
    pub u: usize,
    pub v: usize,
    pub weight: i64,
}

/// Returns the edges of an MST and its total weight. If the graph is
/// disconnected the result is a minimum spanning forest.
pub fn kruskal(num_nodes: usize, edges: &[Edge]) -> (Vec<Edge>, i64) {
    let mut sorted: Vec<Edge> = edges.to_vec();
    sorted.sort_by_key(|e| e.weight);

    let mut dsu = UnionFind::new(num_nodes);
    let mut tree = Vec::with_capacity(num_nodes.saturating_sub(1));
    let mut total: i64 = 0;
    for e in sorted {
        if dsu.union(e.u, e.v) {
            total += e.weight;
            tree.push(e);
        }
    }
    (tree, total)
}

#[cfg(test)]
mod tests {
    use super::{kruskal, Edge};

    fn e(u: usize, v: usize, w: i64) -> Edge {
        Edge { u, v, weight: w }
    }

    #[test]
    fn small_triangle() {
        let edges = vec![e(0, 1, 1), e(1, 2, 2), e(0, 2, 5)];
        let (tree, total) = kruskal(3, &edges);
        assert_eq!(total, 3);
        assert_eq!(tree.len(), 2);
    }

    #[test]
    fn disconnected_forest() {
        let edges = vec![e(0, 1, 1), e(2, 3, 4)];
        let (tree, total) = kruskal(4, &edges);
        assert_eq!(total, 5);
        assert_eq!(tree.len(), 2);
    }

    #[test]
    fn empty() {
        let (tree, total) = kruskal(0, &[]);
        assert!(tree.is_empty());
        assert_eq!(total, 0);
    }
}
