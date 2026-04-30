//! 0-1 BFS: single-source shortest paths on a graph whose edge weights are
//! restricted to `{0, 1}`. Uses a double-ended queue (deque) instead of a
//! priority queue: 0-weight edges relax the neighbour at the same distance
//! and are pushed to the front, 1-weight edges increase the distance by one
//! and are pushed to the back. The deque therefore always holds at most two
//! consecutive distance values, mirroring the layered behaviour of plain BFS
//! and yielding O(V + E) time and O(V) extra space.
//!
//! Precondition: every edge weight must be 0 or 1. Other weights are not
//! validated at runtime; passing them produces undefined (algorithmically
//! meaningless) distances. Use [`dijkstra`](super::dijkstra::dijkstra) for
//! general non-negative weights.

use std::collections::VecDeque;

/// Returns the shortest distance from `src` to every node in `graph`, where
/// `graph[u]` is a list of `(neighbour, weight)` pairs and each `weight` is
/// expected to be `0` or `1`. The result `dist` has `dist[src] = Some(0)`,
/// `dist[v] = Some(d)` for every reachable `v`, and `None` for unreachable
/// vertices.
///
/// Complexity: O(V + E) time, O(V) auxiliary space.
///
/// # Panics
/// Panics if `src` is out of bounds for `graph`.
///
/// # Undefined behaviour
/// Passing edge weights other than 0 or 1 violates the precondition; the
/// returned distances are not meaningful in that case. The function does not
/// validate weights at runtime.
pub fn zero_one_bfs(graph: &[Vec<(usize, u32)>], src: usize) -> Vec<Option<u64>> {
    let n = graph.len();
    if n == 0 {
        return Vec::new();
    }
    assert!(
        src < n,
        "zero_one_bfs: src {src} is out of bounds for graph of length {n}"
    );

    let mut dist: Vec<Option<u64>> = vec![None; n];
    dist[src] = Some(0);
    let mut deque: VecDeque<usize> = VecDeque::new();
    deque.push_back(src);

    while let Some(u) = deque.pop_front() {
        // `dist[u]` is always Some when `u` is on the deque.
        let du = dist[u].expect("node on deque must have a distance");
        for &(v, w) in &graph[u] {
            let candidate = du + u64::from(w);
            let improves = dist[v].is_none_or(|dv| candidate < dv);
            if improves {
                dist[v] = Some(candidate);
                if w == 0 {
                    deque.push_front(v);
                } else {
                    deque.push_back(v);
                }
            }
        }
    }

    dist
}

#[cfg(test)]
mod tests {
    use super::zero_one_bfs;
    use quickcheck_macros::quickcheck;
    use std::cmp::Ordering;
    use std::collections::BinaryHeap;

    #[test]
    fn empty_graph() {
        let g: Vec<Vec<(usize, u32)>> = vec![];
        assert!(zero_one_bfs(&g, 0).is_empty());
    }

    #[test]
    fn single_node() {
        let g: Vec<Vec<(usize, u32)>> = vec![vec![]];
        assert_eq!(zero_one_bfs(&g, 0), vec![Some(0)]);
    }

    #[test]
    fn simple_zero_path() {
        // 0 --0--> 1 --0--> 2: every node reachable at distance 0.
        let g = vec![vec![(1, 0)], vec![(2, 0)], vec![]];
        assert_eq!(zero_one_bfs(&g, 0), vec![Some(0), Some(0), Some(0)]);
    }

    #[test]
    fn simple_one_path() {
        // 0 --1--> 1 --1--> 2 --1--> 3.
        let g = vec![vec![(1, 1)], vec![(2, 1)], vec![(3, 1)], vec![]];
        assert_eq!(
            zero_one_bfs(&g, 0),
            vec![Some(0), Some(1), Some(2), Some(3)]
        );
    }

    #[test]
    fn mixed_weights_pick_shortcut() {
        // Long 1-1-1 chain 0 -> 1 -> 2 -> 3 plus a 0-weight shortcut 0 -> 3.
        // Best distances: 0,1,2,0.
        let g = vec![vec![(1, 1), (3, 0)], vec![(2, 1)], vec![(3, 1)], vec![]];
        assert_eq!(
            zero_one_bfs(&g, 0),
            vec![Some(0), Some(1), Some(2), Some(0)]
        );
    }

    #[test]
    fn mixed_weights_non_trivial() {
        // 0 --1--> 1 --0--> 2 --1--> 3
        // 0 --1--> 3 (direct)
        // Best 0->3: via 1->2->3 costs 1+0+1 = 2; direct costs 1. Min is 1.
        let g = vec![vec![(1, 1), (3, 1)], vec![(2, 0)], vec![(3, 1)], vec![]];
        assert_eq!(
            zero_one_bfs(&g, 0),
            vec![Some(0), Some(1), Some(1), Some(1)]
        );
    }

    #[test]
    fn unreachable_node() {
        // 0 -> 1 with weight 1; node 2 is isolated.
        let g = vec![vec![(1, 1)], vec![], vec![]];
        assert_eq!(zero_one_bfs(&g, 0), vec![Some(0), Some(1), None]);
    }

    #[test]
    fn parallel_edges_keep_min() {
        // 0 -> 1 with parallel edges of weight 1 and 0; 0 wins.
        let g = vec![vec![(1, 1), (1, 0)], vec![]];
        assert_eq!(zero_one_bfs(&g, 0), vec![Some(0), Some(0)]);
    }

    #[test]
    fn src_not_zero() {
        // 0 --1--> 1 --1--> 2; start at 1, so 0 is unreachable in directed graph.
        let g = vec![vec![(1, 1)], vec![(2, 1)], vec![]];
        assert_eq!(zero_one_bfs(&g, 1), vec![None, Some(0), Some(1)]);
    }

    #[test]
    fn self_loop_ignored() {
        // 0 --0--> 0 self-loop must not break termination or worsen distance.
        let g = vec![vec![(0, 0), (1, 1)], vec![]];
        assert_eq!(zero_one_bfs(&g, 0), vec![Some(0), Some(1)]);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn src_out_of_bounds_panics() {
        let g = vec![vec![(1, 1)], vec![]];
        let _ = zero_one_bfs(&g, 5);
    }

    // ---- property test against Dijkstra reference ----

    /// Standard Dijkstra reference using a binary heap. Returns the same
    /// `Option<u64>` shape as the function under test for direct comparison.
    fn dijkstra_reference(graph: &[Vec<(usize, u32)>], src: usize) -> Vec<Option<u64>> {
        #[derive(Copy, Clone, Eq, PartialEq)]
        struct State {
            cost: u64,
            node: usize,
        }
        impl Ord for State {
            fn cmp(&self, other: &Self) -> Ordering {
                other
                    .cost
                    .cmp(&self.cost)
                    .then_with(|| self.node.cmp(&other.node))
            }
        }
        impl PartialOrd for State {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                Some(self.cmp(other))
            }
        }

        let n = graph.len();
        let mut dist: Vec<Option<u64>> = vec![None; n];
        if n == 0 {
            return dist;
        }
        dist[src] = Some(0);
        let mut heap = BinaryHeap::new();
        heap.push(State { cost: 0, node: src });
        while let Some(State { cost, node }) = heap.pop() {
            if dist[node].is_some_and(|d| cost > d) {
                continue;
            }
            for &(v, w) in &graph[node] {
                let next = cost + u64::from(w);
                let improves = dist[v].is_none_or(|dv| next < dv);
                if improves {
                    dist[v] = Some(next);
                    heap.push(State {
                        cost: next,
                        node: v,
                    });
                }
            }
        }
        dist
    }

    /// Build a random directed graph with `n` nodes (1..=8) and 0/1 weights.
    /// `mask` bits decide which (i, j) edges (i != j) exist; `weight_seed`
    /// drives a tiny LCG that picks each weight as 0 or 1.
    fn random_zero_one_graph(n: usize, mask: u64, weight_seed: u64) -> Vec<Vec<(usize, u32)>> {
        let mut g: Vec<Vec<(usize, u32)>> = vec![vec![]; n];
        let mut bit = 0u32;
        let mut wseed = weight_seed;
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                let present = (mask >> (bit % 64)) & 1 == 1;
                bit += 1;
                if present {
                    wseed = wseed
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1);
                    let w = ((wseed >> 33) & 1) as u32;
                    g[i].push((j, w));
                }
            }
        }
        g
    }

    #[allow(clippy::needless_pass_by_value)]
    #[quickcheck]
    fn quickcheck_matches_dijkstra(n: u8, mask: u64, weight_seed: u64) -> bool {
        let n = ((n as usize) % 8) + 1; // 1..=8
        let g = random_zero_one_graph(n, mask, weight_seed);
        let src = (weight_seed as usize) % n;
        zero_one_bfs(&g, src) == dijkstra_reference(&g, src)
    }
}
