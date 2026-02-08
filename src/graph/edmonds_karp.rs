//! Edmonds–Karp maximum flow: Ford–Fulkerson with BFS-found augmenting paths.
//! O(V · E²). Capacity matrix representation; suitable for small/medium graphs.

use std::collections::VecDeque;

/// One directed edge with capacity.
#[derive(Copy, Clone, Debug)]
pub struct Edge {
    pub from: usize,
    pub to: usize,
    pub capacity: u64,
}

/// Returns the maximum flow value from `source` to `sink`. Capacities of
/// parallel edges are summed.
pub fn edmonds_karp(num_nodes: usize, edges: &[Edge], source: usize, sink: usize) -> u64 {
    if source >= num_nodes || sink >= num_nodes || source == sink {
        return 0;
    }
    let mut capacity = vec![vec![0_u64; num_nodes]; num_nodes];
    for e in edges {
        capacity[e.from][e.to] = capacity[e.from][e.to].saturating_add(e.capacity);
    }
    let mut total_flow = 0_u64;
    loop {
        let parent = bfs(&capacity, source, sink);
        let Some(parent) = parent else { break };
        // Find bottleneck along the augmenting path.
        let mut bottleneck = u64::MAX;
        let mut v = sink;
        while v != source {
            let u = parent[v].unwrap();
            bottleneck = bottleneck.min(capacity[u][v]);
            v = u;
        }
        // Apply flow: subtract from forward residual, add to reverse.
        let mut v = sink;
        while v != source {
            let u = parent[v].unwrap();
            capacity[u][v] -= bottleneck;
            capacity[v][u] = capacity[v][u].saturating_add(bottleneck);
            v = u;
        }
        total_flow = total_flow.saturating_add(bottleneck);
    }
    total_flow
}

fn bfs(capacity: &[Vec<u64>], source: usize, sink: usize) -> Option<Vec<Option<usize>>> {
    let n = capacity.len();
    let mut parent: Vec<Option<usize>> = vec![None; n];
    parent[source] = Some(source);
    let mut queue = VecDeque::from([source]);
    while let Some(u) = queue.pop_front() {
        for v in 0..n {
            if parent[v].is_none() && capacity[u][v] > 0 {
                parent[v] = Some(u);
                if v == sink {
                    return Some(parent);
                }
                queue.push_back(v);
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::{edmonds_karp, Edge};

    fn e(from: usize, to: usize, capacity: u64) -> Edge {
        Edge { from, to, capacity }
    }

    #[test]
    fn empty() {
        assert_eq!(edmonds_karp(0, &[], 0, 0), 0);
    }

    #[test]
    fn source_equals_sink() {
        assert_eq!(edmonds_karp(2, &[e(0, 1, 5)], 0, 0), 0);
    }

    #[test]
    fn single_edge() {
        assert_eq!(edmonds_karp(2, &[e(0, 1, 7)], 0, 1), 7);
    }

    #[test]
    fn classic_clrs_example() {
        // 6-node CLRS network with max flow 23.
        let edges = vec![
            e(0, 1, 16),
            e(0, 2, 13),
            e(1, 2, 10),
            e(2, 1, 4),
            e(1, 3, 12),
            e(2, 4, 14),
            e(3, 2, 9),
            e(3, 5, 20),
            e(4, 3, 7),
            e(4, 5, 4),
        ];
        assert_eq!(edmonds_karp(6, &edges, 0, 5), 23);
    }

    #[test]
    fn unreachable_sink() {
        let edges = vec![e(0, 1, 5)];
        assert_eq!(edmonds_karp(3, &edges, 0, 2), 0);
    }

    #[test]
    fn parallel_edges_are_summed() {
        let edges = vec![e(0, 1, 3), e(0, 1, 4)];
        assert_eq!(edmonds_karp(2, &edges, 0, 1), 7);
    }
}
