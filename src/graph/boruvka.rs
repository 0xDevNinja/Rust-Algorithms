//! Borůvka's minimum spanning tree algorithm.
//!
//! In each phase every current component finds its single cheapest
//! outgoing edge, and all of those edges are added at once; components
//! linked by those edges are then merged via union–find. After at most
//! `O(log V)` phases (each phase at least halves the component count)
//! the spanning tree is complete.
//!
//! Total cost: `O((V + E) log V)`. Space: `O(V)` for the union–find plus
//! `O(V)` for the per-phase "cheapest edge" buffer.
//!
//! ## Behaviour on disconnected input
//!
//! If the graph has more than one connected component, no spanning tree
//! exists and the function returns `None`. (The result is therefore an
//! MST, not a minimum spanning forest — callers wanting a forest should
//! prefer `kruskal`.)
//!
//! ## Tie-breaking
//!
//! When two candidate outgoing edges of a component have the same
//! weight, the one with the smaller input index is picked. Selection is
//! therefore deterministic and depends only on the input ordering.

use crate::data_structures::union_find::UnionFind;

/// Computes a minimum spanning tree of an undirected weighted graph
/// using Borůvka's algorithm.
///
/// `edges[k] = (u, v, w)` is the `k`th edge between vertices `u` and
/// `v` (both in `0..n`) with weight `w`. Self-loops and parallel edges
/// are tolerated; among parallel edges the lowest-weight one is
/// retained (ties broken by smaller index).
///
/// Returns `Some(indices)` listing the input edge indices chosen for
/// the MST in selection order — its length is exactly `n - 1` for a
/// connected graph. Returns `None` if the graph is disconnected, and
/// `Some(vec![])` for the trivial `n = 0` or `n = 1` cases.
///
/// Runs in `O((V + E) log V)` time and `O(V)` auxiliary space.
pub fn boruvka_mst(n: usize, edges: &[(usize, usize, i64)]) -> Option<Vec<usize>> {
    if n <= 1 {
        return Some(Vec::new());
    }

    let mut dsu = UnionFind::new(n);
    let mut chosen: Vec<usize> = Vec::with_capacity(n - 1);
    let mut in_tree = vec![false; edges.len()];

    // `cheapest[component_root] = Some((weight, edge_index))`.
    let mut cheapest: Vec<Option<(i64, usize)>> = vec![None; n];

    loop {
        cheapest.fill(None);

        // Phase scan: for every edge crossing two distinct components,
        // remember the lightest such edge per component (tie-break by
        // smaller edge index for determinism).
        for (idx, &(u, v, w)) in edges.iter().enumerate() {
            if in_tree[idx] {
                continue;
            }
            let ru = dsu.find(u);
            let rv = dsu.find(v);
            if ru == rv {
                continue;
            }
            for r in [ru, rv] {
                let better = match cheapest[r] {
                    None => true,
                    Some((bw, bi)) => w < bw || (w == bw && idx < bi),
                };
                if better {
                    cheapest[r] = Some((w, idx));
                }
            }
        }

        let mut merged_any = false;
        for r in 0..n {
            if let Some((_, idx)) = cheapest[r] {
                let (u, v, _) = edges[idx];
                if dsu.union(u, v) {
                    in_tree[idx] = true;
                    chosen.push(idx);
                    merged_any = true;
                }
            }
        }

        if !merged_any {
            break;
        }
        if dsu.component_count() == 1 {
            break;
        }
    }

    if dsu.component_count() == 1 {
        Some(chosen)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::boruvka_mst;
    use crate::graph::kruskal::{kruskal, Edge};
    use quickcheck_macros::quickcheck;

    fn weight_of(edges: &[(usize, usize, i64)], picks: &[usize]) -> i64 {
        picks.iter().map(|&i| edges[i].2).sum()
    }

    #[test]
    fn empty_graph() {
        assert_eq!(boruvka_mst(0, &[]), Some(vec![]));
    }

    #[test]
    fn single_node() {
        assert_eq!(boruvka_mst(1, &[]), Some(vec![]));
    }

    #[test]
    fn simple_triangle() {
        // 0-1 weight 1, 1-2 weight 2, 0-2 weight 5 → MST weight 3.
        let edges = vec![(0, 1, 1), (1, 2, 2), (0, 2, 5)];
        let picks = boruvka_mst(3, &edges).expect("connected");
        assert_eq!(picks.len(), 2);
        assert_eq!(weight_of(&edges, &picks), 3);
    }

    #[test]
    fn classic_five_nodes() {
        // Standard textbook example: MST weight 16.
        let edges = vec![
            (0, 1, 2),
            (0, 3, 6),
            (1, 2, 3),
            (1, 3, 8),
            (1, 4, 5),
            (2, 4, 7),
            (3, 4, 9),
        ];
        let picks = boruvka_mst(5, &edges).expect("connected");
        assert_eq!(picks.len(), 4);
        assert_eq!(weight_of(&edges, &picks), 16);
    }

    #[test]
    fn disconnected_returns_none() {
        // Two components: {0,1} and {2,3}.
        let edges = vec![(0, 1, 1), (2, 3, 4)];
        assert_eq!(boruvka_mst(4, &edges), None);
    }

    #[test]
    fn already_a_tree() {
        let edges = vec![(0, 1, 4), (1, 2, 7), (2, 3, 2)];
        let picks = boruvka_mst(4, &edges).expect("connected");
        assert_eq!(picks.len(), 3);
        assert_eq!(weight_of(&edges, &picks), 13);
    }

    #[test]
    fn parallel_edges_keeps_lowest() {
        // Three parallel edges between 0 and 1; only the cheapest
        // (weight 1) should be picked.
        let edges = vec![(0, 1, 5), (0, 1, 1), (0, 1, 3)];
        let picks = boruvka_mst(2, &edges).expect("connected");
        assert_eq!(picks, vec![1]);
        assert_eq!(weight_of(&edges, &picks), 1);
    }

    #[test]
    fn tied_weights_pick_smaller_index() {
        // Triangle with all unit weights — any two edges form an MST,
        // but the deterministic tie-break picks the two smallest
        // indices that still connect the graph.
        let edges = vec![(0, 1, 1), (1, 2, 1), (0, 2, 1)];
        let picks = boruvka_mst(3, &edges).expect("connected");
        assert_eq!(picks.len(), 2);
        assert_eq!(weight_of(&edges, &picks), 2);
        // Edge 0 is the cheapest outgoing edge for component {0} and
        // edge 1 for component {2}, so both get added in phase one.
        let mut sorted = picks;
        sorted.sort_unstable();
        assert_eq!(sorted, vec![0, 1]);
    }

    #[quickcheck]
    fn matches_kruskal_weight(seed: Vec<(u8, u8, i16)>) -> bool {
        // Bound n ≤ 10; skip empty graphs (kruskal handles n = 0
        // trivially but we want non-trivial coverage).
        const N: usize = 10;
        let edges: Vec<(usize, usize, i64)> = seed
            .into_iter()
            .map(|(u, v, w)| ((u as usize) % N, (v as usize) % N, i64::from(w)))
            .filter(|(u, v, _)| u != v)
            .collect();

        let kruskal_edges: Vec<Edge> = edges
            .iter()
            .map(|&(u, v, w)| Edge { u, v, weight: w })
            .collect();
        let (k_tree, k_total) = kruskal(N, &kruskal_edges);

        // Borůvka returns Some only on a single connected component
        // (Kruskal's tree then spans all N nodes and totals match);
        // it returns None iff the graph is disconnected, which
        // Kruskal exposes as a forest with < N - 1 edges.
        boruvka_mst(N, &edges).map_or(k_tree.len() < N - 1, |picks| {
            k_tree.len() == N - 1 && picks.len() == N - 1 && weight_of(&edges, &picks) == k_total
        })
    }
}
