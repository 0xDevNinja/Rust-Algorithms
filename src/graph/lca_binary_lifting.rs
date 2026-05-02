//! Lowest Common Ancestor (LCA) on a rooted tree via **binary lifting**.
//!
//! For each node `v` and each `k`, we precompute `up[k][v]`, the `2^k`-th
//! ancestor of `v`. A query lifts the deeper of the two nodes up to the
//! depth of the shallower one, then jumps both pointers up together by the
//! largest power of two that does not equate them. Their parent is the LCA.
//!
//! # Complexity
//! - Preprocessing: O(N log N) time and space.
//! - Query:         O(log N) time.
//!
//! # Preconditions
//! The input must be a **rooted tree**: connected, acyclic, undirected
//! adjacency list with exactly N − 1 edges for N nodes. The chosen `root`
//! must be a valid index in `0..N`. By convention `up[k][root] == root` for
//! all `k`, so lifting from the root stays at the root.
//!
//! Out-of-precondition behaviour:
//! - **Out-of-range query nodes** — panic via index-out-of-bounds.
//! - **Disconnected forest** — only the component containing `root` has a
//!   meaningful depth/parent assignment; querying nodes outside that
//!   component is undefined.
//! - **Cyclic input** — the BFS terminates because of the visited check, but
//!   the resulting structure is not a tree and queries are meaningless.

use std::collections::VecDeque;

/// Precomputed binary-lifting table for LCA queries on a rooted tree.
///
/// Build once with [`Lca::new`] in O(N log N), then answer arbitrary
/// [`Lca::query`] calls in O(log N).
pub struct Lca {
    /// `up[k][v]` is the `2^k`-th ancestor of `v`. The root's ancestors all
    /// point back to the root, which keeps the lifting loop branch-free.
    up: Vec<Vec<usize>>,
    /// Depth of each node from the root (root has depth 0).
    depth: Vec<u32>,
    /// `ceil(log2(max(n, 2)))` — the number of binary-lifting levels.
    log: u32,
    /// Stored root index; useful for callers that want to introspect.
    root: usize,
}

impl Lca {
    /// Builds the binary-lifting table for the tree given by `adj`, rooted
    /// at `root`.
    ///
    /// `adj[i]` holds the neighbours of node `i` in the undirected tree.
    /// Runs a BFS from `root` to compute depths and immediate parents, then
    /// fills `up[k][v] = up[k - 1][up[k - 1][v]]` for k = 1..log.
    ///
    /// # Panics
    /// Panics if `root >= adj.len()` (unless `adj` is empty, which yields an
    /// empty `Lca`).
    #[must_use]
    pub fn new(adj: &[Vec<usize>], root: usize) -> Self {
        let n = adj.len();
        if n == 0 {
            return Self {
                up: Vec::new(),
                depth: Vec::new(),
                log: 1,
                root: 0,
            };
        }
        assert!(root < n, "root index {root} out of bounds for n = {n}");

        // ceil(log2(n)), with a floor of 1 so `up` always has at least one row.
        let log = (usize::BITS - (n - 1).max(1).leading_zeros()).max(1);

        let mut depth = vec![0u32; n];
        let mut parent = vec![usize::MAX; n];
        // Treat the root as its own parent — keeps lifting idempotent at the
        // root and avoids special-casing inside `query`.
        parent[root] = root;

        // BFS from root to fill depth + immediate parent. Iterative on
        // purpose: avoids stack overflow on long chains.
        let mut visited = vec![false; n];
        visited[root] = true;
        let mut queue = VecDeque::from([root]);
        while let Some(u) = queue.pop_front() {
            for &v in &adj[u] {
                if !visited[v] {
                    visited[v] = true;
                    parent[v] = u;
                    depth[v] = depth[u] + 1;
                    queue.push_back(v);
                }
            }
        }

        // Build the lifting table. Row 0 is the immediate parent.
        let mut up = vec![vec![0usize; n]; log as usize];
        up[0].clone_from(&parent);
        for k in 1..log as usize {
            for v in 0..n {
                let mid = up[k - 1][v];
                up[k][v] = if mid == usize::MAX {
                    usize::MAX
                } else {
                    up[k - 1][mid]
                };
            }
        }

        Self {
            up,
            depth,
            log,
            root,
        }
    }

    /// Returns the lowest common ancestor of nodes `u` and `v`.
    ///
    /// If `u == v`, returns that node. If one is an ancestor of the other,
    /// returns the ancestor.
    ///
    /// # Panics
    /// Panics if `u` or `v` is out of range for the tree this `Lca` was
    /// built from.
    #[must_use]
    pub fn query(&self, mut u: usize, mut v: usize) -> usize {
        let n = self.depth.len();
        assert!(u < n && v < n, "query node out of bounds");

        // Step 1: lift the deeper node up to the depth of the shallower.
        if self.depth[u] < self.depth[v] {
            std::mem::swap(&mut u, &mut v);
        }
        let diff = self.depth[u] - self.depth[v];
        for k in 0..self.log {
            if (diff >> k) & 1 == 1 {
                u = self.up[k as usize][u];
            }
        }

        if u == v {
            return u;
        }

        // Step 2: jump both pointers up together by the largest power of two
        // that does not make them equal. After the loop their parent is LCA.
        for k in (0..self.log).rev() {
            let k = k as usize;
            if self.up[k][u] != self.up[k][v] {
                u = self.up[k][u];
                v = self.up[k][v];
            }
        }
        self.up[0][u]
    }

    /// Returns the depth of `v` from the root (root has depth 0).
    ///
    /// # Panics
    /// Panics if `v` is out of range.
    #[must_use]
    pub fn depth(&self, v: usize) -> u32 {
        self.depth[v]
    }

    /// Returns the root the table was built from.
    #[must_use]
    pub const fn root(&self) -> usize {
        self.root
    }
}

#[cfg(test)]
mod tests {
    use super::Lca;
    use quickcheck_macros::quickcheck;

    // Build an undirected adjacency list from an edge list of size `n`.
    fn build(n: usize, edges: &[(usize, usize)]) -> Vec<Vec<usize>> {
        let mut g = vec![vec![]; n];
        for &(u, v) in edges {
            g[u].push(v);
            g[v].push(u);
        }
        g
    }

    // Brute-force LCA: walk the deeper node up to match depths, then walk
    // both up one step at a time until they coincide.
    fn brute_force_lca(adj: &[Vec<usize>], root: usize, u: usize, v: usize) -> usize {
        let n = adj.len();
        let mut depth = vec![0i64; n];
        let mut parent = vec![root; n];
        let mut visited = vec![false; n];
        visited[root] = true;
        let mut queue = std::collections::VecDeque::from([root]);
        while let Some(x) = queue.pop_front() {
            for &y in &adj[x] {
                if !visited[y] {
                    visited[y] = true;
                    parent[y] = x;
                    depth[y] = depth[x] + 1;
                    queue.push_back(y);
                }
            }
        }
        let (mut a, mut b) = (u, v);
        while depth[a] > depth[b] {
            a = parent[a];
        }
        while depth[b] > depth[a] {
            b = parent[b];
        }
        while a != b {
            a = parent[a];
            b = parent[b];
        }
        a
    }

    // Deterministic XorShift-based random tree on `n` nodes rooted at 0.
    fn random_tree(n: usize, seed: u64) -> Vec<Vec<usize>> {
        if n == 0 {
            return vec![];
        }
        let mut state = seed ^ 0x9e37_79b9_7f4a_7c15;
        let mut xorshift = move || -> u64 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let mut g = vec![vec![]; n];
        for i in 1..n {
            let parent = (xorshift() as usize) % i;
            g[i].push(parent);
            g[parent].push(i);
        }
        g
    }

    #[test]
    fn single_node_tree() {
        // One node — its LCA with itself is itself.
        let g = build(1, &[]);
        let lca = Lca::new(&g, 0);
        assert_eq!(lca.query(0, 0), 0);
        assert_eq!(lca.root(), 0);
        assert_eq!(lca.depth(0), 0);
    }

    #[test]
    fn two_node_tree_lca_is_root() {
        // 0 -- 1, rooted at 0.
        let g = build(2, &[(0, 1)]);
        let lca = Lca::new(&g, 0);
        assert_eq!(lca.query(0, 1), 0);
        assert_eq!(lca.query(1, 0), 0);
        assert_eq!(lca.query(1, 1), 1);
        assert_eq!(lca.depth(1), 1);
    }

    #[test]
    fn chain_lca_of_leaf_and_root() {
        // 0 -- 1 -- 2 -- 3 -- 4, rooted at 0. LCA(4, 0) = 0; LCA(2, 4) = 2.
        let g = build(5, &[(0, 1), (1, 2), (2, 3), (3, 4)]);
        let lca = Lca::new(&g, 0);
        assert_eq!(lca.query(4, 0), 0);
        assert_eq!(lca.query(0, 4), 0);
        assert_eq!(lca.query(2, 4), 2);
        assert_eq!(lca.query(3, 1), 1);
        assert_eq!(lca.depth(4), 4);
    }

    #[test]
    fn star_lca_of_two_leaves_is_centre() {
        // Centre 0, leaves 1..=4.
        let g = build(5, &[(0, 1), (0, 2), (0, 3), (0, 4)]);
        let lca = Lca::new(&g, 0);
        for a in 1..=4 {
            for b in 1..=4 {
                if a == b {
                    assert_eq!(lca.query(a, b), a);
                } else {
                    assert_eq!(lca.query(a, b), 0, "leaves {a} and {b}");
                }
            }
        }
    }

    #[test]
    fn balanced_binary_tree_known_lcas() {
        // Tree:
        //            0
        //          /   \
        //         1     2
        //        / \   / \
        //       3   4 5   6
        //      / \
        //     7   8
        let g = build(
            9,
            &[
                (0, 1),
                (0, 2),
                (1, 3),
                (1, 4),
                (2, 5),
                (2, 6),
                (3, 7),
                (3, 8),
            ],
        );
        let lca = Lca::new(&g, 0);
        assert_eq!(lca.query(7, 8), 3);
        assert_eq!(lca.query(7, 4), 1);
        assert_eq!(lca.query(7, 5), 0);
        assert_eq!(lca.query(4, 6), 0);
        assert_eq!(lca.query(5, 6), 2);
        assert_eq!(lca.query(8, 1), 1); // ancestor case
        assert_eq!(lca.query(3, 3), 3); // identical
    }

    #[test]
    fn non_zero_root() {
        // Same chain, rooted at node 2 instead of 0:
        // 0 -- 1 -- 2 -- 3 -- 4, root = 2.
        let g = build(5, &[(0, 1), (1, 2), (2, 3), (3, 4)]);
        let lca = Lca::new(&g, 2);
        // Path from 0 to 2 in this rooted tree: 0 -> 1 -> 2.
        assert_eq!(lca.query(0, 4), 2);
        assert_eq!(lca.query(0, 1), 1);
        assert_eq!(lca.query(3, 4), 3);
        assert_eq!(lca.depth(0), 2);
    }

    #[test]
    fn random_trees_match_brute_force() {
        // Deterministic sweep: 200 seeds, n in 1..=30, all (u, v) pairs.
        for seed in 0u64..200 {
            let n = ((seed % 30) + 1) as usize;
            let g = random_tree(n, seed);
            let lca = Lca::new(&g, 0);
            for u in 0..n {
                for v in 0..n {
                    let got = lca.query(u, v);
                    let want = brute_force_lca(&g, 0, u, v);
                    assert_eq!(got, want, "seed={seed} n={n} u={u} v={v}");
                }
            }
        }
    }

    #[allow(clippy::needless_pass_by_value)]
    #[quickcheck]
    fn quickcheck_random_tree_matches_brute_force(n: u8, seed: u64, ui: u8, vi: u8) -> bool {
        let n = ((n as usize) % 30) + 1;
        let g = random_tree(n, seed);
        let u = (ui as usize) % n;
        let v = (vi as usize) % n;
        let lca = Lca::new(&g, 0);
        lca.query(u, v) == brute_force_lca(&g, 0, u, v)
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn out_of_range_query_panics() {
        let g = build(3, &[(0, 1), (1, 2)]);
        let lca = Lca::new(&g, 0);
        let _ = lca.query(0, 99);
    }
}
