//! Prüfer sequence: a bijection between labeled trees on `n` vertices
//! `[0..n)` and integer sequences in `[0..n)^{n-2}`. Combined with
//! Cayley's formula `n^{n-2}` it counts labeled trees and supports
//! uniform random-tree sampling.
//!
//! Complexity:
//! - `tree_to_prufer`: `O(n log n)` using a min-heap of current leaves.
//! - `prufer_to_tree`: `O(n log n)` using a min-heap of current leaves.
//! - `count_labeled_trees`: `O(log n)` via fast exponentiation.

use std::collections::BinaryHeap;

/// Encodes a labeled tree on vertices `[0..n)` as its Prüfer sequence.
///
/// The input `edges` must describe a valid tree: exactly `n - 1` edges,
/// connected, no self-loops, no parallel edges, all endpoints in
/// `[0..n)`. The resulting sequence has length `n - 2` (empty for
/// `n <= 2`). Validity is enforced via `debug_assert!` so release
/// builds skip the linear-time checks.
pub fn tree_to_prufer(edges: &[(usize, usize)], n: usize) -> Vec<usize> {
    if n <= 2 {
        debug_assert!(
            edges.len() + 1 == n.max(1) || (n == 0 && edges.is_empty()),
            "tree must have exactly n-1 edges"
        );
        return Vec::new();
    }
    debug_assert_eq!(edges.len(), n - 1, "tree must have exactly n-1 edges");
    debug_assert!(is_tree(edges, n), "input edges must form a tree");

    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut degree: Vec<usize> = vec![0; n];
    for &(u, v) in edges {
        adj[u].push(v);
        adj[v].push(u);
        degree[u] += 1;
        degree[v] += 1;
    }

    // Min-heap of current leaves (vertices with degree 1). `BinaryHeap`
    // is a max-heap, so we negate via `std::cmp::Reverse`.
    let mut leaves: BinaryHeap<std::cmp::Reverse<usize>> = BinaryHeap::new();
    for (v, &d) in degree.iter().enumerate() {
        if d == 1 {
            leaves.push(std::cmp::Reverse(v));
        }
    }

    let mut seq = Vec::with_capacity(n - 2);
    for _ in 0..(n - 2) {
        let leaf = leaves.pop().expect("non-empty leaf set in a tree").0;
        // The leaf has exactly one remaining neighbour with degree > 0.
        let neighbour = *adj[leaf]
            .iter()
            .find(|&&v| degree[v] > 0 && v != leaf)
            .expect("leaf must still have a neighbour");
        seq.push(neighbour);
        degree[leaf] = 0;
        degree[neighbour] -= 1;
        if degree[neighbour] == 1 {
            leaves.push(std::cmp::Reverse(neighbour));
        }
    }
    seq
}

/// Decodes a Prüfer sequence back into the unique labeled tree it
/// represents.
///
/// Requires `n >= 2` and `seq.len() == n - 2`. Each entry of `seq`
/// must lie in `[0..n)`. Returns the `n - 1` edges of the resulting
/// tree.
pub fn prufer_to_tree(seq: &[usize], n: usize) -> Vec<(usize, usize)> {
    assert!(n >= 2, "prufer_to_tree requires n >= 2");
    assert_eq!(seq.len(), n - 2, "Prüfer sequence must have length n - 2");
    debug_assert!(
        seq.iter().all(|&v| v < n),
        "Prüfer entries must be in [0..n)"
    );

    let mut degree: Vec<usize> = vec![1; n];
    for &v in seq {
        degree[v] += 1;
    }

    let mut leaves: BinaryHeap<std::cmp::Reverse<usize>> = BinaryHeap::new();
    for (v, &d) in degree.iter().enumerate() {
        if d == 1 {
            leaves.push(std::cmp::Reverse(v));
        }
    }

    let mut edges = Vec::with_capacity(n - 1);
    for &x in seq {
        let leaf = leaves.pop().expect("non-empty leaf set during decoding").0;
        edges.push((leaf, x));
        degree[leaf] -= 1;
        degree[x] -= 1;
        if degree[x] == 1 {
            leaves.push(std::cmp::Reverse(x));
        }
    }

    // Two vertices of degree 1 remain; they form the final edge.
    let u = leaves.pop().expect("two leaves remain").0;
    let v = leaves.pop().expect("two leaves remain").0;
    edges.push((u, v));
    edges
}

/// Cayley's formula: the number of labeled trees on `n` vertices is
/// `n^{n - 2}` for `n >= 2`. Returns `1` for `n <= 2` to cover the
/// boundary cases (`n = 0` and `n = 1` have a single empty/trivial
/// tree; `n = 2` has the single edge `0-1`).
pub const fn count_labeled_trees(n: u64) -> u64 {
    if n <= 2 {
        return 1;
    }
    let exp = n - 2;
    let mut result: u64 = 1;
    let mut base: u64 = n;
    let mut e = exp;
    while e > 0 {
        if e & 1 == 1 {
            result = result.wrapping_mul(base);
        }
        e >>= 1;
        if e > 0 {
            base = base.wrapping_mul(base);
        }
    }
    result
}

/// Internal: union-find `find` with path compression. Used by `is_tree`.
fn dsu_find(parent: &mut [usize], x: usize) -> usize {
    let mut root = x;
    while parent[root] != root {
        root = parent[root];
    }
    let mut cur = x;
    while parent[cur] != root {
        let next = parent[cur];
        parent[cur] = root;
        cur = next;
    }
    root
}

/// Internal: confirms that `edges` describes a tree on `n` vertices via
/// Union-Find. Used only in `debug_assert!`.
fn is_tree(edges: &[(usize, usize)], n: usize) -> bool {
    if n == 0 {
        return edges.is_empty();
    }
    if edges.len() != n - 1 {
        return false;
    }
    let mut parent: Vec<usize> = (0..n).collect();
    let mut rank: Vec<u32> = vec![0; n];

    for &(u, v) in edges {
        if u >= n || v >= n || u == v {
            return false;
        }
        let ru = dsu_find(&mut parent, u);
        let rv = dsu_find(&mut parent, v);
        if ru == rv {
            return false;
        }
        match rank[ru].cmp(&rank[rv]) {
            std::cmp::Ordering::Less => parent[ru] = rv,
            std::cmp::Ordering::Greater => parent[rv] = ru,
            std::cmp::Ordering::Equal => {
                parent[rv] = ru;
                rank[ru] += 1;
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::{count_labeled_trees, is_tree, prufer_to_tree, tree_to_prufer};

    #[test]
    fn cayley_small_values() {
        assert_eq!(count_labeled_trees(1), 1);
        assert_eq!(count_labeled_trees(2), 1);
        assert_eq!(count_labeled_trees(3), 3);
        assert_eq!(count_labeled_trees(4), 16);
        assert_eq!(count_labeled_trees(5), 125);
        assert_eq!(count_labeled_trees(6), 1296);
    }

    #[test]
    fn cayley_zero() {
        // Convention: empty tree has the empty Prüfer sequence.
        assert_eq!(count_labeled_trees(0), 1);
    }

    #[test]
    fn path_prufer_sequence() {
        // Path 0-1-2-3 should give Prüfer sequence [1, 2].
        let edges = vec![(0, 1), (1, 2), (2, 3)];
        assert_eq!(tree_to_prufer(&edges, 4), vec![1, 2]);
    }

    #[test]
    fn star_prufer_sequence() {
        // Star with centre 0, leaves 1, 2, 3 should give [0, 0].
        let edges = vec![(0, 1), (0, 2), (0, 3)];
        assert_eq!(tree_to_prufer(&edges, 4), vec![0, 0]);
    }

    #[test]
    fn prufer_to_tree_path_inverse() {
        let edges = prufer_to_tree(&[1, 2], 4);
        assert_eq!(edges.len(), 3);
        assert!(is_tree(&edges, 4));
        // And it round-trips.
        assert_eq!(tree_to_prufer(&edges, 4), vec![1, 2]);
    }

    #[test]
    fn prufer_n_equals_two() {
        // Empty sequence yields the single edge 0-1.
        let edges = prufer_to_tree(&[], 2);
        assert_eq!(edges.len(), 1);
        let (u, v) = edges[0];
        assert!((u, v) == (0, 1) || (u, v) == (1, 0));
        assert_eq!(tree_to_prufer(&edges, 2), Vec::<usize>::new());
    }

    /// Recursive helper: enumerate every length-`len` sequence with
    /// entries in `[0..n)` and call `visit` on each.
    fn enumerate_sequences(
        n: usize,
        len: usize,
        buf: &mut Vec<usize>,
        visit: &mut impl FnMut(&[usize]),
    ) {
        if buf.len() == len {
            visit(buf);
            return;
        }
        for x in 0..n {
            buf.push(x);
            enumerate_sequences(n, len, buf, visit);
            buf.pop();
        }
    }

    #[test]
    fn round_trip_all_trees_up_to_six() {
        for n in 2usize..=6 {
            let len = n.saturating_sub(2);
            let mut count = 0u64;
            let mut buf = Vec::with_capacity(len);
            enumerate_sequences(n, len, &mut buf, &mut |seq| {
                let edges = prufer_to_tree(seq, n);
                assert_eq!(edges.len(), n - 1, "n = {n}, seq = {seq:?}");
                assert!(is_tree(&edges, n), "n = {n}, seq = {seq:?}");
                let recovered = tree_to_prufer(&edges, n);
                assert_eq!(recovered, seq, "round-trip failed for n = {n}");
                count += 1;
            });
            assert_eq!(count, count_labeled_trees(n as u64));
        }
    }
}
