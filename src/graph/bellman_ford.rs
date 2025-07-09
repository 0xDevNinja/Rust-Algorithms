//! Bellman–Ford single-source shortest paths. Handles negative edge weights.
//! Detects negative-weight cycles reachable from the source. O(V·E).

/// One directed edge with weight.
#[derive(Copy, Clone, Debug)]
pub struct Edge {
    pub from: usize,
    pub to: usize,
    pub weight: i64,
}

/// Returns shortest distances from `start`. `Err` if a negative cycle is
/// reachable from `start`. Unreachable nodes are reported as `i64::MAX`.
pub fn bellman_ford(
    num_nodes: usize,
    edges: &[Edge],
    start: usize,
) -> Result<Vec<i64>, &'static str> {
    let mut dist = vec![i64::MAX; num_nodes];
    if start >= num_nodes {
        return Ok(dist);
    }
    dist[start] = 0;
    for _ in 0..num_nodes.saturating_sub(1) {
        let mut updated = false;
        for e in edges {
            if dist[e.from] == i64::MAX {
                continue;
            }
            let candidate = dist[e.from].saturating_add(e.weight);
            if candidate < dist[e.to] {
                dist[e.to] = candidate;
                updated = true;
            }
        }
        if !updated {
            break;
        }
    }
    for e in edges {
        if dist[e.from] != i64::MAX && dist[e.from].saturating_add(e.weight) < dist[e.to] {
            return Err("negative-weight cycle reachable from start");
        }
    }
    Ok(dist)
}

#[cfg(test)]
mod tests {
    use super::{bellman_ford, Edge};

    fn edge(from: usize, to: usize, weight: i64) -> Edge {
        Edge { from, to, weight }
    }

    #[test]
    fn basic_positive() {
        let edges = vec![edge(0, 1, 1), edge(1, 2, 2), edge(0, 2, 5)];
        let d = bellman_ford(3, &edges, 0).unwrap();
        assert_eq!(d, vec![0, 1, 3]);
    }

    #[test]
    fn handles_negative_edge() {
        let edges = vec![edge(0, 1, 4), edge(0, 2, 5), edge(1, 2, -3)];
        let d = bellman_ford(3, &edges, 0).unwrap();
        assert_eq!(d, vec![0, 4, 1]);
    }

    #[test]
    fn detects_negative_cycle() {
        let edges = vec![edge(0, 1, 1), edge(1, 2, -1), edge(2, 1, -1)];
        assert!(bellman_ford(3, &edges, 0).is_err());
    }

    #[test]
    fn unreachable_stays_max() {
        let edges = vec![edge(0, 1, 7)];
        let d = bellman_ford(3, &edges, 0).unwrap();
        assert_eq!(d[2], i64::MAX);
    }
}
