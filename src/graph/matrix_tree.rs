//! Kirchhoff's matrix-tree theorem: count spanning trees of an undirected
//! graph.
//!
//! For a connected undirected graph on `n` vertices the number of
//! spanning trees equals any cofactor of the Laplacian
//! `L = D - A`, where `D` is the diagonal degree matrix and `A` the
//! (multi-)adjacency matrix. Equivalently, the count is the determinant
//! of the `(n-1) × (n-1)` minor obtained by deleting any one row and
//! the corresponding column from `L`. We always delete row 0 and
//! column 0.
//!
//! ## Why Bareiss
//!
//! The minor of `L` is an integer matrix, and the determinant is by
//! the theorem a non-negative integer. To keep the answer exact we use
//! the **Bareiss fraction-free algorithm**: it performs Gaussian
//! elimination over the integers, using the previous pivot as the
//! divisor at each step, and the divisions are guaranteed to be exact.
//! No floating point, no rationals — just `i128`.
//!
//! ## Conventions
//!
//! * Self-loops are ignored (a self-loop on vertex `v` adds `2` to its
//!   degree but also `2` to `A[v][v]`, so it cancels in `L` and never
//!   contributes to a spanning tree anyway).
//! * Parallel edges are honoured: each parallel copy increments both
//!   the degree and the adjacency entry, and contributes
//!   multiplicatively to the spanning-tree count (e.g. two parallel
//!   edges between two vertices give `2`, since each edge is its own
//!   spanning tree).
//! * `n == 0` returns `0`. `n == 1` returns `0` as well — the singleton
//!   `K_1` traditionally has one spanning tree (itself), but the
//!   theorem reduces to a `0 × 0` minor, and we follow the convention
//!   used elsewhere in this crate of treating "no spanning tree
//!   possible" inputs uniformly. Document this if the calling code
//!   cares about `K_1`.
//! * A disconnected graph has no spanning tree, so the determinant of
//!   the minor is `0`.
//!
//! ## Complexity
//!
//! Bareiss elimination on the `(n-1) × (n-1)` minor is `O(n³)` time
//! and `O(n²)` space. The intermediate values are integers but can
//! grow; `i128` is sufficient for the small graphs the tests exercise
//! and for any graph whose final spanning-tree count fits in `i128`,
//! but very large or dense graphs may require a big-integer backend.

/// Count the spanning trees of an undirected graph on `n` vertices
/// with the given edge multiset.
///
/// Edges are pairs `(u, v)` with `0 <= u, v < n`. Self-loops
/// (`u == v`) are ignored; parallel edges are honoured. The
/// computation runs in `O(n³)` via Bareiss fraction-free elimination
/// on the `(n-1) × (n-1)` Laplacian minor and returns the exact
/// integer count.
///
/// Returns `0` for `n < 2` and for disconnected graphs. Panics if any
/// endpoint is out of range.
pub fn spanning_tree_count(n: usize, edges: &[(usize, usize)]) -> i128 {
    if n < 2 {
        return 0;
    }

    // Build Laplacian L = D - A as an i128 matrix.
    let mut laplacian = vec![vec![0i128; n]; n];
    for &(u, v) in edges {
        assert!(u < n && v < n, "edge endpoint out of range");
        if u == v {
            // Self-loop: ignored.
            continue;
        }
        laplacian[u][u] += 1;
        laplacian[v][v] += 1;
        laplacian[u][v] -= 1;
        laplacian[v][u] -= 1;
    }

    // Delete row 0 and column 0 to obtain the (n-1) × (n-1) minor.
    let m = n - 1;
    let mut minor = vec![vec![0i128; m]; m];
    for i in 0..m {
        for j in 0..m {
            minor[i][j] = laplacian[i + 1][j + 1];
        }
    }

    bareiss_determinant(minor).abs()
}

/// Bareiss fraction-free determinant for an integer matrix.
///
/// Returns `0` if the matrix is singular. The algorithm maintains the
/// invariant that after step `k` every entry of the working submatrix
/// is exactly divisible by the previous pivot, so all divisions are
/// integral. Sign tracking handles row swaps. Time `O(n³)`.
fn bareiss_determinant(mut a: Vec<Vec<i128>>) -> i128 {
    let n = a.len();
    if n == 0 {
        // Empty determinant is conventionally 1, but the matrix-tree
        // theorem only invokes this for n >= 1, so this branch is
        // defensive.
        return 1;
    }
    let mut sign: i128 = 1;
    let mut prev: i128 = 1;
    for k in 0..n {
        // Find a non-zero pivot in column k at or below row k.
        if a[k][k] == 0 {
            let mut pivot_row = None;
            for r in (k + 1)..n {
                if a[r][k] != 0 {
                    pivot_row = Some(r);
                    break;
                }
            }
            match pivot_row {
                Some(r) => {
                    a.swap(k, r);
                    sign = -sign;
                }
                None => return 0,
            }
        }
        for i in (k + 1)..n {
            for j in (k + 1)..n {
                let num = a[i][j] * a[k][k] - a[i][k] * a[k][j];
                // Bareiss invariant: prev divides num exactly.
                a[i][j] = num / prev;
            }
            a[i][k] = 0;
        }
        prev = a[k][k];
    }
    sign * a[n - 1][n - 1]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn complete_graph_edges(n: usize) -> Vec<(usize, usize)> {
        let mut edges = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                edges.push((i, j));
            }
        }
        edges
    }

    #[test]
    fn k2_has_one_spanning_tree() {
        assert_eq!(spanning_tree_count(2, &[(0, 1)]), 1);
    }

    #[test]
    fn triangle_k3_has_three_spanning_trees() {
        // Cayley: K_n has n^(n-2) spanning trees, so K_3 has 3^1 = 3.
        assert_eq!(spanning_tree_count(3, &complete_graph_edges(3)), 3);
    }

    #[test]
    fn k4_has_sixteen_spanning_trees() {
        // 4^(4-2) = 16.
        assert_eq!(spanning_tree_count(4, &complete_graph_edges(4)), 16);
    }

    #[test]
    fn k5_matches_cayley() {
        // 5^(5-2) = 125.
        assert_eq!(spanning_tree_count(5, &complete_graph_edges(5)), 125);
    }

    #[test]
    fn star_with_three_leaves_has_one_spanning_tree() {
        // The star itself is the only spanning tree.
        let edges = [(0, 1), (0, 2), (0, 3)];
        assert_eq!(spanning_tree_count(4, &edges), 1);
    }

    #[test]
    fn complete_bipartite_k_2_3_has_twelve_spanning_trees() {
        // K_{m,n} has m^(n-1) * n^(m-1) spanning trees, so
        // K_{2,3} = 2^2 * 3^1 = 12.
        let edges = [(0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4)];
        assert_eq!(spanning_tree_count(5, &edges), 12);
    }

    #[test]
    fn disconnected_graph_has_zero_spanning_trees() {
        // Two components: {0,1} and {2,3}.
        let edges = [(0, 1), (2, 3)];
        assert_eq!(spanning_tree_count(4, &edges), 0);
    }

    #[test]
    fn parallel_edges_count_multiplicatively() {
        // K_2 with two parallel edges — each edge is its own spanning
        // tree, so the count is 2.
        let edges = [(0, 1), (0, 1)];
        assert_eq!(spanning_tree_count(2, &edges), 2);
    }

    #[test]
    fn three_parallel_edges_give_three() {
        let edges = [(0, 1), (0, 1), (0, 1)];
        assert_eq!(spanning_tree_count(2, &edges), 3);
    }

    #[test]
    fn tree_input_returns_one() {
        // A path 0 - 1 - 2 - 3 - 4 is itself a tree with exactly one
        // spanning tree.
        let edges = [(0, 1), (1, 2), (2, 3), (3, 4)];
        assert_eq!(spanning_tree_count(5, &edges), 1);
    }

    #[test]
    fn self_loops_are_ignored() {
        // Adding self-loops to K_3 must not change the count.
        let mut edges = complete_graph_edges(3);
        edges.push((0, 0));
        edges.push((1, 1));
        edges.push((2, 2));
        assert_eq!(spanning_tree_count(3, &edges), 3);
    }

    #[test]
    fn zero_or_one_vertex_returns_zero() {
        assert_eq!(spanning_tree_count(0, &[]), 0);
        assert_eq!(spanning_tree_count(1, &[]), 0);
    }

    #[test]
    fn cycle_c4_has_four_spanning_trees() {
        // Removing any one of the 4 edges of C_4 leaves a spanning
        // tree — so 4 spanning trees.
        let edges = [(0, 1), (1, 2), (2, 3), (3, 0)];
        assert_eq!(spanning_tree_count(4, &edges), 4);
    }

    // ----- Brute-force property test -----

    /// Brute force: enumerate every subset of `n - 1` edges and count
    /// those that span the graph (form a tree on all `n` vertices).
    fn brute_force_spanning_tree_count(n: usize, edges: &[(usize, usize)]) -> i128 {
        if n < 2 {
            return 0;
        }
        let need = n - 1;
        let m = edges.len();
        if m < need {
            return 0;
        }
        let mut count: i128 = 0;
        // Iterate over all combinations of edge indices of size n-1.
        let mut idx: Vec<usize> = (0..need).collect();
        loop {
            if is_spanning_tree(n, edges, &idx) {
                count += 1;
            }
            // Next combination in lexicographic order.
            let mut i = need;
            while i > 0 {
                i -= 1;
                if idx[i] != i + m - need {
                    break;
                }
                if i == 0 {
                    return count;
                }
            }
            if idx[i] == i + m - need {
                return count;
            }
            idx[i] += 1;
            for j in (i + 1)..need {
                idx[j] = idx[j - 1] + 1;
            }
        }
    }

    fn find(parent: &mut [usize], x: usize) -> usize {
        if parent[x] == x {
            x
        } else {
            let r = find(parent, parent[x]);
            parent[x] = r;
            r
        }
    }

    fn is_spanning_tree(n: usize, edges: &[(usize, usize)], picked: &[usize]) -> bool {
        // Use union-find to detect cycles and confirm connectivity on
        // exactly n - 1 edges.
        let mut parent: Vec<usize> = (0..n).collect();
        for &ei in picked {
            let (u, v) = edges[ei];
            if u == v {
                return false;
            }
            let ru = find(&mut parent, u);
            let rv = find(&mut parent, v);
            if ru == rv {
                return false;
            }
            parent[ru] = rv;
        }
        // Count distinct roots — must be 1 for a spanning tree.
        let root0 = find(&mut parent, 0);
        for v in 1..n {
            if find(&mut parent, v) != root0 {
                return false;
            }
        }
        true
    }

    /// Tiny deterministic LCG for the property test — avoids pulling
    /// in `rand`.
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            self.0
        }
        fn range(&mut self, hi: u64) -> u64 {
            self.next() % hi
        }
    }

    #[test]
    fn brute_force_agrees_for_small_random_graphs() {
        let mut rng = Lcg(0x5EED_5EED_5EED_5EED);
        for trial in 0..200 {
            let n = 2 + (rng.range(4) as usize); // 2..=5
                                                 // Random edge count up to n*(n-1)/2 + 2 (allow a couple of
                                                 // parallel edges).
            let max_e = n * (n - 1) / 2 + 2;
            let e = rng.range((max_e as u64) + 1) as usize;
            let mut edges = Vec::with_capacity(e);
            for _ in 0..e {
                let u = rng.range(n as u64) as usize;
                let v = rng.range(n as u64) as usize;
                edges.push((u, v));
            }
            let theorem = spanning_tree_count(n, &edges);
            let brute = brute_force_spanning_tree_count(n, &edges);
            assert_eq!(
                theorem, brute,
                "mismatch on trial {trial}: n={n}, edges={edges:?}"
            );
        }
    }
}
