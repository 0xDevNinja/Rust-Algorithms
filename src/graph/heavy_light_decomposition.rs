//! **Heavy-Light Decomposition (HLD)** of a rooted tree.
//!
//! For each non-leaf node we mark the edge to its largest subtree as the
//! *heavy* edge; the remaining edges are *light*. Following heavy edges from
//! the root partitions the tree into a set of vertex-disjoint **chains**.
//! The chains are then laid out in DFS order so that every chain occupies a
//! contiguous interval — i.e. each path from `u` to the root crosses at most
//! `O(log N)` chain boundaries (a light edge halves the subtree size, so we
//! cannot take more than `log N` of them).
//!
//! Path queries on the tree therefore become a small number of range queries
//! over the linearised positions. Combined with a segment tree (or any other
//! 1-D range structure) HLD answers path-update / path-query in
//! `O(log² N)` per operation:
//!
//! - one `O(log N)` factor from the chain hops, and
//! - one `O(log N)` factor from the underlying range structure.
//!
//! Typical use cases: path sum / max / min on weighted trees, path XOR,
//! "set all edges on the path to v" with lazy propagation, and lowest common
//! ancestor in `O(log N)` (the same chain-climbing trick).
//!
//! # Complexity
//! - Preprocessing: O(N) time and space (two DFS passes, both iterative).
//! - [`HeavyLightDecomposition::lca`]:           O(log N).
//! - [`HeavyLightDecomposition::path_segments`]: O(log N) intervals returned.
//!   The caller then runs whatever range structure they want over each
//!   interval, giving overall `O(log² N)` for typical path queries.
//!
//! # Preconditions
//! The input must be a **rooted tree**: connected, acyclic, undirected
//! adjacency list with exactly `N − 1` edges for `N` nodes, and `root` must
//! be a valid index in `0..N`. With a disconnected forest only the component
//! containing `root` is meaningful; with a cyclic graph the result is
//! undefined (the iterative DFS terminates because of the visited check, but
//! the resulting chain layout is not a valid HLD).
//!
//! Out-of-precondition behaviour:
//! - Out-of-range query nodes — panic via index-out-of-bounds.
//! - `root >= N` — panic with a descriptive message.

/// Heavy-light decomposition of a rooted tree.
///
/// Build once with [`HeavyLightDecomposition::new`] in `O(N)`, then answer
/// arbitrary [`HeavyLightDecomposition::lca`] queries in `O(log N)` and
/// [`HeavyLightDecomposition::path_segments`] queries in `O(log N)` returned
/// intervals.
pub struct HeavyLightDecomposition {
    /// Immediate parent in the rooted tree. `parent[root] == root`, which
    /// keeps the chain-climbing loop branch-free at the top.
    parent: Vec<usize>,
    /// Depth of each node from the root (root has depth 0).
    depth: Vec<u32>,
    /// `heavy[v]` is the child of `v` with the largest subtree, or `None` if
    /// `v` is a leaf. The edge to this child is the heavy edge of `v`.
    heavy: Vec<Option<usize>>,
    /// Top of the chain `v` belongs to. All nodes on the same chain share the
    /// same `head`. The chain is the maximal heavy-edge path containing `v`.
    head: Vec<usize>,
    /// DFS order index of each node. The decomposition guarantees that every
    /// chain occupies a contiguous interval `[position[head], position[tail]]`
    /// of these indices.
    position: Vec<usize>,
}

impl HeavyLightDecomposition {
    /// Builds the heavy-light decomposition for the tree given by `adj`,
    /// rooted at `root`.
    ///
    /// `adj[i]` holds the neighbours of node `i` in the undirected tree.
    /// Two iterative DFS passes (no recursion, so deep trees are safe):
    ///
    /// 1. compute `parent`, `depth`, subtree size and the heavy child of
    ///    every node;
    /// 2. visit children with the heavy child first to lay out each chain in
    ///    a contiguous block of DFS-order positions, recording `head[v]`.
    ///
    /// # Panics
    /// Panics if `root >= adj.len()` (unless `adj` is empty, which yields an
    /// empty decomposition).
    #[must_use]
    pub fn new(adj: &[Vec<usize>], root: usize) -> Self {
        let n = adj.len();
        if n == 0 {
            return Self {
                parent: Vec::new(),
                depth: Vec::new(),
                heavy: Vec::new(),
                head: Vec::new(),
                position: Vec::new(),
            };
        }
        assert!(root < n, "root index {root} out of bounds for n = {n}");

        let mut parent = vec![0usize; n];
        let mut depth = vec![0u32; n];
        let mut heavy = vec![None; n];
        let mut size = vec![1usize; n];
        // Treat the root as its own parent — keeps chain-climbing idempotent
        // at the top and avoids special-casing inside `lca`.
        parent[root] = root;

        // ---- Pass 1: iterative DFS to fill parent / depth / size / heavy.
        // Stack entries are (node, iterator-index into adj[node]). On the
        // way down we set parent + depth; on the way up we accumulate size
        // and pick the heavy child.
        let mut order = Vec::with_capacity(n);
        let mut stack: Vec<(usize, usize)> = Vec::with_capacity(n);
        let mut visited = vec![false; n];
        visited[root] = true;
        stack.push((root, 0));
        while let Some(&(u, i)) = stack.last() {
            if i < adj[u].len() {
                let v = adj[u][i];
                stack.last_mut().unwrap().1 += 1;
                if !visited[v] {
                    visited[v] = true;
                    parent[v] = u;
                    depth[v] = depth[u] + 1;
                    stack.push((v, 0));
                }
            } else {
                order.push(u);
                stack.pop();
            }
        }
        // `order` is post-order: every child is visited before its parent.
        for &u in &order {
            if u == parent[u] {
                continue;
            }
            let p = parent[u];
            size[p] += size[u];
            match heavy[p] {
                None => heavy[p] = Some(u),
                Some(h) if size[u] > size[h] => heavy[p] = Some(u),
                _ => {}
            }
        }

        // ---- Pass 2: iterative DFS that assigns chain heads + DFS-order
        // positions, always descending the heavy edge first so that each
        // chain occupies a contiguous block of positions.
        //
        // Each stack frame is `(node, next_adj_index)`. On the first entry
        // (`next_adj_index == 0`) we stamp the position and, if `node` has
        // a heavy child, push it immediately (with `head` carried over).
        // After that we walk `adj[node]` and push every remaining child as
        // a new chain head (light edge).
        let mut head = vec![0usize; n];
        let mut position = vec![0usize; n];
        let mut timer = 0usize;
        let mut stack2: Vec<(usize, usize)> = Vec::with_capacity(n);
        head[root] = root;
        stack2.push((root, usize::MAX));
        while let Some(&(u, next_idx)) = stack2.last() {
            if next_idx == usize::MAX {
                // First visit: stamp position, then descend the heavy edge.
                position[u] = timer;
                timer += 1;
                stack2.last_mut().unwrap().1 = 0;
                if let Some(h) = heavy[u] {
                    head[h] = head[u];
                    stack2.push((h, usize::MAX));
                }
                continue;
            }
            // Subsequent visits: walk light children one by one.
            let adj_u = &adj[u];
            let mut idx = next_idx;
            let heavy_child = heavy[u];
            let is_root = u == parent[u];
            let mut pushed = false;
            while idx < adj_u.len() {
                let v = adj_u[idx];
                idx += 1;
                if !is_root && v == parent[u] {
                    continue;
                }
                if Some(v) == heavy_child {
                    continue;
                }
                head[v] = v;
                stack2.last_mut().unwrap().1 = idx;
                stack2.push((v, usize::MAX));
                pushed = true;
                break;
            }
            if !pushed {
                stack2.pop();
            }
        }

        Self {
            parent,
            depth,
            heavy,
            head,
            position,
        }
    }

    /// Returns the lowest common ancestor of `u` and `v`.
    ///
    /// Repeatedly lifts whichever of `head[u]`, `head[v]` is deeper to its
    /// parent, jumping a whole chain at a time. When both nodes share a
    /// chain the shallower one is the LCA. `O(log N)` because every chain
    /// hop crosses a light edge and there are at most `log N` light edges
    /// on any root-to-leaf path.
    ///
    /// # Panics
    /// Panics if `u` or `v` is out of range for the tree this HLD was built
    /// from.
    #[must_use]
    pub fn lca(&self, mut u: usize, mut v: usize) -> usize {
        let n = self.depth.len();
        assert!(u < n && v < n, "query node out of bounds");
        while self.head[u] != self.head[v] {
            // Move the deeper-headed node up — symmetry guarantees progress.
            if self.depth[self.head[u]] < self.depth[self.head[v]] {
                std::mem::swap(&mut u, &mut v);
            }
            u = self.parent[self.head[u]];
        }
        if self.depth[u] < self.depth[v] {
            u
        } else {
            v
        }
    }

    /// Returns the disjoint contiguous DFS-order intervals `[l, r]`
    /// (inclusive, with `l <= r`) whose union covers exactly the nodes on
    /// the path from `u` to `v`.
    ///
    /// Each interval lives entirely within one heavy chain, so the caller
    /// can run a 1-D range structure (segment tree, BIT, sparse table, …)
    /// over [`HeavyLightDecomposition::position`]-indexed values to answer
    /// arbitrary path queries. There are at most `O(log N)` intervals.
    ///
    /// The returned vector is unordered; callers that need a left-to-right
    /// sweep should sort by the first coordinate.
    ///
    /// # Panics
    /// Panics if `u` or `v` is out of range for the tree this HLD was built
    /// from.
    #[must_use]
    pub fn path_segments(&self, mut u: usize, mut v: usize) -> Vec<(usize, usize)> {
        let n = self.depth.len();
        assert!(u < n && v < n, "query node out of bounds");
        let mut out = Vec::new();
        while self.head[u] != self.head[v] {
            if self.depth[self.head[u]] < self.depth[self.head[v]] {
                std::mem::swap(&mut u, &mut v);
            }
            // `u` is in the deeper chain: take the segment from its head
            // down to itself, then jump above the chain.
            let head_u = self.head[u];
            out.push((self.position[head_u], self.position[u]));
            u = self.parent[head_u];
        }
        // Same chain: emit the inclusive interval between the two nodes.
        let (a, b) = if self.position[u] <= self.position[v] {
            (self.position[u], self.position[v])
        } else {
            (self.position[v], self.position[u])
        };
        out.push((a, b));
        out
    }

    /// Returns the depth of `v` from the root (root has depth 0).
    ///
    /// # Panics
    /// Panics if `v` is out of range.
    #[must_use]
    pub fn depth(&self, v: usize) -> u32 {
        self.depth[v]
    }

    /// Returns the DFS-order position assigned to `v`. Each chain occupies a
    /// contiguous block of positions in `0..N`.
    ///
    /// # Panics
    /// Panics if `v` is out of range.
    #[must_use]
    pub fn position(&self, v: usize) -> usize {
        self.position[v]
    }

    /// Returns the head (top-most node) of the chain containing `v`.
    ///
    /// # Panics
    /// Panics if `v` is out of range.
    #[must_use]
    pub fn head(&self, v: usize) -> usize {
        self.head[v]
    }

    /// Returns the heavy child of `v`, or `None` if `v` is a leaf.
    ///
    /// # Panics
    /// Panics if `v` is out of range.
    #[must_use]
    pub fn heavy_child(&self, v: usize) -> Option<usize> {
        self.heavy[v]
    }

    /// Returns the parent of `v` (root maps to itself).
    ///
    /// # Panics
    /// Panics if `v` is out of range.
    #[must_use]
    pub fn parent(&self, v: usize) -> usize {
        self.parent[v]
    }
}

#[cfg(test)]
mod tests {
    use super::HeavyLightDecomposition;
    use quickcheck_macros::quickcheck;
    use std::collections::{HashSet, VecDeque};

    fn build(n: usize, edges: &[(usize, usize)]) -> Vec<Vec<usize>> {
        let mut g = vec![vec![]; n];
        for &(u, v) in edges {
            g[u].push(v);
            g[v].push(u);
        }
        g
    }

    // Brute-force LCA: BFS to fill depth + parent, equalise depths, walk up.
    fn brute_force_lca(adj: &[Vec<usize>], root: usize, u: usize, v: usize) -> usize {
        let n = adj.len();
        let mut depth = vec![0i64; n];
        let mut parent = vec![root; n];
        let mut visited = vec![false; n];
        visited[root] = true;
        let mut queue = VecDeque::from([root]);
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

    // Brute-force path enumeration: explicit BFS parent then walk both
    // endpoints up to their LCA, returning the set of nodes on the path.
    fn brute_force_path_nodes(
        adj: &[Vec<usize>],
        root: usize,
        u: usize,
        v: usize,
    ) -> HashSet<usize> {
        let n = adj.len();
        let mut depth = vec![0i64; n];
        let mut parent = vec![root; n];
        let mut visited = vec![false; n];
        visited[root] = true;
        let mut queue = VecDeque::from([root]);
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
        let lca = brute_force_lca(adj, root, u, v);
        let mut nodes = HashSet::new();
        let mut a = u;
        loop {
            nodes.insert(a);
            if a == lca {
                break;
            }
            a = parent[a];
        }
        let mut b = v;
        loop {
            nodes.insert(b);
            if b == lca {
                break;
            }
            b = parent[b];
        }
        nodes
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

    // Verify path_segments output:
    //   - every interval is non-empty and within `0..n`,
    //   - intervals are pairwise disjoint,
    //   - the union, mapped back through position, equals brute-force path.
    fn check_path_segments(adj: &[Vec<usize>], hld: &HeavyLightDecomposition, u: usize, v: usize) {
        let n = adj.len();
        let segs = hld.path_segments(u, v);
        // Reconstruct visited DFS-order positions.
        let mut covered = HashSet::new();
        for &(l, r) in &segs {
            assert!(l <= r, "segment l > r: ({l}, {r})");
            assert!(r < n, "segment past end: ({l}, {r}) for n = {n}");
            for p in l..=r {
                assert!(covered.insert(p), "duplicate position {p} across segments");
            }
        }
        // Map positions back to nodes.
        let mut node_of_position = vec![usize::MAX; n];
        for node in 0..n {
            node_of_position[hld.position(node)] = node;
        }
        let got: HashSet<usize> = covered.into_iter().map(|p| node_of_position[p]).collect();
        let want = brute_force_path_nodes(adj, 0, u, v);
        assert_eq!(got, want, "path mismatch for ({u}, {v})");
    }

    #[test]
    fn single_node_tree() {
        let g = build(1, &[]);
        let hld = HeavyLightDecomposition::new(&g, 0);
        assert_eq!(hld.lca(0, 0), 0);
        assert_eq!(hld.depth(0), 0);
        assert_eq!(hld.position(0), 0);
        assert_eq!(hld.head(0), 0);
        assert_eq!(hld.heavy_child(0), None);
        assert_eq!(hld.path_segments(0, 0), vec![(0, 0)]);
    }

    #[test]
    fn path_graph_is_one_chain() {
        // 0 -- 1 -- 2 -- 3 -- 4 rooted at 0. All edges are heavy (each parent
        // has exactly one child), so every node shares head 0 and the chain
        // covers positions 0..5 contiguously in depth order.
        let n = 5;
        let g = build(n, &[(0, 1), (1, 2), (2, 3), (3, 4)]);
        let hld = HeavyLightDecomposition::new(&g, 0);
        for v in 0..n {
            assert_eq!(hld.head(v), 0, "all nodes share the root chain");
        }
        // Positions are a permutation of 0..n that respects depth.
        let mut positions: Vec<usize> = (0..n).map(|v| hld.position(v)).collect();
        positions.sort_unstable();
        assert_eq!(positions, (0..n).collect::<Vec<_>>());
        for v in 1..n {
            assert!(hld.position(v) > hld.position(hld.parent(v)));
        }
        // Every (u, v) pair: lca matches brute force, path_segments covers
        // the brute-force path exactly with one interval (single chain).
        for u in 0..n {
            for v in 0..n {
                assert_eq!(hld.lca(u, v), brute_force_lca(&g, 0, u, v));
                let segs = hld.path_segments(u, v);
                assert_eq!(segs.len(), 1, "path graph yields one segment");
                check_path_segments(&g, &hld, u, v);
            }
        }
    }

    #[test]
    fn star_has_root_plus_one_singleton_per_leaf() {
        // Root 0, leaves 1..=4. The first child becomes the heavy child of
        // 0 (all four subtrees have size 1, ties broken by adjacency order),
        // so there are three light edges that each form their own chain.
        let n = 5;
        let g = build(n, &[(0, 1), (0, 2), (0, 3), (0, 4)]);
        let hld = HeavyLightDecomposition::new(&g, 0);
        // Root chain has head 0 and the heavy child shares it.
        let heavy = hld.heavy_child(0).expect("non-leaf root");
        assert_eq!(hld.head(heavy), 0);
        // The other three leaves are heads of their own singleton chains.
        let other_leaves: Vec<usize> = (1..=4).filter(|&v| v != heavy).collect();
        assert_eq!(other_leaves.len(), 3);
        for &leaf in &other_leaves {
            assert_eq!(hld.head(leaf), leaf);
            assert_eq!(hld.heavy_child(leaf), None);
        }
        for u in 0..n {
            for v in 0..n {
                assert_eq!(hld.lca(u, v), brute_force_lca(&g, 0, u, v));
                check_path_segments(&g, &hld, u, v);
            }
        }
    }

    #[test]
    fn balanced_binary_tree_known_lcas_and_paths() {
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
        let hld = HeavyLightDecomposition::new(&g, 0);
        // Spot-check a handful of well-known LCAs.
        assert_eq!(hld.lca(7, 8), 3);
        assert_eq!(hld.lca(7, 4), 1);
        assert_eq!(hld.lca(7, 5), 0);
        assert_eq!(hld.lca(4, 6), 0);
        assert_eq!(hld.lca(5, 6), 2);
        assert_eq!(hld.lca(8, 1), 1); // ancestor case
        assert_eq!(hld.lca(3, 3), 3); // identical
                                      // Full sweep: HLD agrees with brute force, segments cover paths.
        for u in 0..9 {
            for v in 0..9 {
                assert_eq!(hld.lca(u, v), brute_force_lca(&g, 0, u, v));
                check_path_segments(&g, &hld, u, v);
            }
        }
    }

    #[test]
    fn classic_small_example() {
        // Spider-leg tree: a long heavy chain 0-1-2-3 plus a light branch 4-5
        // hanging off 1, plus a single light leaf 6 hanging off 2.
        //
        //   0 — 1 — 2 — 3
        //       |   |
        //       4   6
        //       |
        //       5
        let n = 7;
        let g = build(n, &[(0, 1), (1, 2), (2, 3), (1, 4), (4, 5), (2, 6)]);
        let hld = HeavyLightDecomposition::new(&g, 0);
        // Subtree sizes: size(1) = 6, size(2) = 3, size(4) = 2 — so 1's
        // heavy child is 2 (size 3 vs 4's 2), 2's heavy child is 3 vs 6, etc.
        assert_eq!(hld.heavy_child(0), Some(1));
        assert_eq!(hld.heavy_child(1), Some(2));
        assert_eq!(hld.heavy_child(4), Some(5));
        // Heads must match the chain structure.
        assert_eq!(hld.head(0), 0);
        assert_eq!(hld.head(1), 0);
        assert_eq!(hld.head(2), 0);
        // Either 3 or 6 takes the heavy edge from 2 (both have size 1;
        // adjacency order resolves the tie). Whichever wins shares head 0.
        let heavy_of_2 = hld.heavy_child(2).unwrap();
        assert_eq!(hld.head(heavy_of_2), 0);
        // Full agreement.
        for u in 0..n {
            for v in 0..n {
                assert_eq!(hld.lca(u, v), brute_force_lca(&g, 0, u, v));
                check_path_segments(&g, &hld, u, v);
            }
        }
    }

    #[test]
    fn random_trees_match_brute_force() {
        // 200 deterministic seeds across n in 1..=30, every (u, v) pair.
        for seed in 0u64..200 {
            let n = ((seed % 30) + 1) as usize;
            let g = random_tree(n, seed);
            let hld = HeavyLightDecomposition::new(&g, 0);
            for u in 0..n {
                for v in 0..n {
                    assert_eq!(
                        hld.lca(u, v),
                        brute_force_lca(&g, 0, u, v),
                        "seed={seed} n={n} u={u} v={v}"
                    );
                    check_path_segments(&g, &hld, u, v);
                }
            }
        }
    }

    #[allow(clippy::needless_pass_by_value)]
    #[quickcheck]
    fn quickcheck_lca_matches_brute_force(n: u8, seed: u64, ui: u8, vi: u8) -> bool {
        let n = ((n as usize) % 30) + 1;
        let g = random_tree(n, seed);
        let u = (ui as usize) % n;
        let v = (vi as usize) % n;
        let hld = HeavyLightDecomposition::new(&g, 0);
        hld.lca(u, v) == brute_force_lca(&g, 0, u, v)
    }

    #[allow(clippy::needless_pass_by_value)]
    #[quickcheck]
    fn quickcheck_path_segments_cover_path(n: u8, seed: u64, ui: u8, vi: u8) -> bool {
        let n = ((n as usize) % 30) + 1;
        let g = random_tree(n, seed);
        let u = (ui as usize) % n;
        let v = (vi as usize) % n;
        let hld = HeavyLightDecomposition::new(&g, 0);
        let segs = hld.path_segments(u, v);
        // Build the node-set covered by `segs` via the inverse-position map.
        let mut node_of_position = vec![usize::MAX; n];
        for node in 0..n {
            node_of_position[hld.position(node)] = node;
        }
        let mut got = HashSet::new();
        for (l, r) in segs {
            if l > r || r >= n {
                return false;
            }
            for p in l..=r {
                if !got.insert(node_of_position[p]) {
                    return false;
                }
            }
        }
        got == brute_force_path_nodes(&g, 0, u, v)
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn out_of_range_lca_panics() {
        let g = build(3, &[(0, 1), (1, 2)]);
        let hld = HeavyLightDecomposition::new(&g, 0);
        let _ = hld.lca(0, 99);
    }
}
