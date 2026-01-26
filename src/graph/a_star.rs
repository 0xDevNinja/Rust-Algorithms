//! A* heuristic shortest path on a weighted graph with non-negative edge weights.
//! O((V + E) log V) with a binary-heap open set, when the heuristic is
//! admissible (never overestimates) the result is an optimal path.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

#[derive(Copy, Clone, Eq, PartialEq)]
struct State {
    estimate: u64, // g + h
    cost: u64,     // g
    node: usize,
}

impl Ord for State {
    fn cmp(&self, other: &Self) -> Ordering {
        // Min-heap on `estimate`, ties broken by lower cost then node id.
        other
            .estimate
            .cmp(&self.estimate)
            .then_with(|| other.cost.cmp(&self.cost))
            .then_with(|| self.node.cmp(&other.node))
    }
}

impl PartialOrd for State {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Returns `(path, cost)` from `start` to `goal`, or `None` if unreachable.
///
/// `graph[u]` is a list of `(neighbour, weight)`. `heuristic(u)` should
/// return an admissible estimate of the remaining cost from `u` to `goal`.
/// Setting `heuristic` to a function that always returns 0 reduces this
/// algorithm to Dijkstra's.
pub fn a_star<H>(
    graph: &[Vec<(usize, u64)>],
    start: usize,
    goal: usize,
    heuristic: H,
) -> Option<(Vec<usize>, u64)>
where
    H: Fn(usize) -> u64,
{
    let n = graph.len();
    if start >= n || goal >= n {
        return None;
    }
    let mut g_score: HashMap<usize, u64> = HashMap::new();
    let mut came_from: HashMap<usize, usize> = HashMap::new();
    let mut open = BinaryHeap::new();
    g_score.insert(start, 0);
    open.push(State {
        estimate: heuristic(start),
        cost: 0,
        node: start,
    });
    while let Some(State { cost, node, .. }) = open.pop() {
        if node == goal {
            return Some((reconstruct_path(&came_from, start, goal), cost));
        }
        if cost > *g_score.get(&node).unwrap_or(&u64::MAX) {
            continue;
        }
        for &(neighbour, weight) in &graph[node] {
            let tentative = cost.saturating_add(weight);
            if tentative < *g_score.get(&neighbour).unwrap_or(&u64::MAX) {
                g_score.insert(neighbour, tentative);
                came_from.insert(neighbour, node);
                open.push(State {
                    estimate: tentative.saturating_add(heuristic(neighbour)),
                    cost: tentative,
                    node: neighbour,
                });
            }
        }
    }
    None
}

fn reconstruct_path(came_from: &HashMap<usize, usize>, start: usize, goal: usize) -> Vec<usize> {
    let mut path = vec![goal];
    let mut current = goal;
    while current != start {
        match came_from.get(&current) {
            Some(&prev) => {
                current = prev;
                path.push(current);
            }
            None => break,
        }
    }
    path.reverse();
    path
}

#[cfg(test)]
mod tests {
    use super::a_star;

    #[test]
    fn empty_graph_invalid_indices() {
        let g: Vec<Vec<(usize, u64)>> = vec![];
        assert!(a_star(&g, 0, 0, |_| 0).is_none());
    }

    #[test]
    fn start_equals_goal() {
        let g = vec![vec![]];
        let (path, cost) = a_star(&g, 0, 0, |_| 0).unwrap();
        assert_eq!(path, vec![0]);
        assert_eq!(cost, 0);
    }

    #[test]
    fn simple_chain() {
        // 0 -1-> 1 -1-> 2 -1-> 3
        let g = vec![vec![(1, 1)], vec![(2, 1)], vec![(3, 1)], vec![]];
        let (path, cost) = a_star(&g, 0, 3, |_| 0).unwrap();
        assert_eq!(path, vec![0, 1, 2, 3]);
        assert_eq!(cost, 3);
    }

    #[test]
    fn admissible_heuristic_yields_optimal_path() {
        // Layout (positions for heuristic):
        // 0(0,0) -- 1(1,0) -- 2(2,0)
        //   \                   /
        //    3(0,2) -- 4(1,2) -
        let positions = [(0_i32, 0), (1, 0), (2, 0), (0, 2), (1, 2)];
        let g = vec![
            vec![(1, 1), (3, 2)],
            vec![(0, 1), (2, 1)],
            vec![(1, 1), (4, 1)],
            vec![(0, 2), (4, 1)],
            vec![(3, 1), (2, 1)],
        ];
        let manhattan = |u: usize| {
            let (x, y) = positions[u];
            let (gx, gy) = positions[2];
            (i32::abs(x - gx) + i32::abs(y - gy)) as u64
        };
        let (path, cost) = a_star(&g, 0, 2, manhattan).unwrap();
        assert_eq!(cost, 2);
        assert_eq!(path.first(), Some(&0));
        assert_eq!(path.last(), Some(&2));
    }

    #[test]
    fn unreachable_goal() {
        let g = vec![vec![(1, 1)], vec![], vec![]];
        assert!(a_star(&g, 0, 2, |_| 0).is_none());
    }

    #[test]
    fn zero_heuristic_matches_dijkstra() {
        // Branching graph; A* with h ≡ 0 should still find shortest path.
        let g = vec![
            vec![(1, 4), (2, 1)],
            vec![(3, 1)],
            vec![(1, 2), (3, 5)],
            vec![],
        ];
        let (_path, cost) = a_star(&g, 0, 3, |_| 0).unwrap();
        assert_eq!(cost, 4); // 0 -> 2 -> 1 -> 3 = 1 + 2 + 1
    }
}
