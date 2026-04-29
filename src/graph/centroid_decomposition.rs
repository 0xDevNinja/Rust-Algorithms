//! **Centroid decomposition** of an unrooted tree.
//!
//! A *centroid* of a tree on `k` nodes is a vertex whose removal leaves every
//! remaining piece with at most `⌈k / 2⌉` nodes. Such a vertex always exists.
//! Recursing on each piece yields the **centroid tree**: a rooted hierarchy in
//! which every node's parent is the centroid of the smallest subtree (in the
//! decomposition sense) that contains it. The centroid tree has depth
//! `O(log N)` because each level halves the working size.
//!
//! Centroid decomposition is the standard divide-and-conquer-on-trees tool:
//! every simple path in the original tree passes through the lowest common
//! ancestor of its endpoints in the centroid tree, so problems such as
//! "count pairs at distance ≤ K", "k-th nearest marked vertex", and offline
//! tree-path counting reduce to per-centroid sweeps that together cost
//! `O(N log N)`.
//!
//! # Complexity
//! - Preprocessing: `O(N log N)` time, `O(N)` extra space (recursion depth is
//!   `O(log N)` because the centroid halves the active subtree each level).
//! - [`CentroidDecomposition::parent`][]: `O(1)`.
//!
//! # Preconditions
//! `adj` must describe an **unrooted tree** — connected, acyclic, undirected
//! adjacency list with exactly `N − 1` edges for `N` nodes. The empty graph is
//! accepted and produces an empty decomposition.
//!
//! For a disconnected forest only the component containing node `0` is
//! decomposed; the other components keep `parent = None` and contribute no
//! depth. Callers that need every component should run the decomposition
//! per component themselves.
//!
//! Out-of-precondition behaviour: a graph with cycles will not satisfy the
//! invariant `subtree size halves each level`, but the implementation still
//! halts (visited-set guards every traversal); the resulting `parent` array
//! is meaningless in that case.

/// Centroid decomposition of an unrooted tree.
///
/// Build once with [`CentroidDecomposition::new`] in `O(N log N)`, then read
/// back the centroid-tree parent of any node in `O(1)` via
/// [`CentroidDecomposition::parent`].
pub struct CentroidDecomposition {
    /// Parent of each node in the centroid tree, or `None` for the root of
    /// the centroid tree (and for nodes that were never visited because they
    /// live in a different component than node `0`).
    centroid_parent: Vec<Option<usize>>,
    /// Root of the centroid tree (the centroid of the whole input tree), or
    /// `None` for an empty graph.
    root: Option<usize>,
    /// Depth of each node in the centroid tree (root has depth 0). Nodes in
    /// other components keep depth 0 — callers should consult
    /// [`CentroidDecomposition::parent`] to tell the root from an unvisited
    /// node.
    depth: Vec<u32>,
}

impl CentroidDecomposition {
    /// Builds the centroid decomposition of the tree given by `adj`.
    ///
    /// One iterative DFS per recursion level (no recursion on node indices,
    /// so deep trees are safe — the only recursion is on the centroid tree
    /// itself, whose depth is `O(log N)`):
    ///
    /// 1. DFS from any seed of the active piece to record both
    ///    post-order and DFS-tree parent, then accumulate subtree sizes
    ///    bottom-up.
    /// 2. Walk down from the seed, always stepping into the heaviest
    ///    neighbour whose live subtree exceeds `total / 2`; the first node
    ///    with no such neighbour is the centroid.
    ///
    /// The centroid is then marked decomposed and the algorithm recurses on
    /// each surviving neighbour, recording the centroid-tree parent / depth.
    #[must_use]
    pub fn new(adj: &[Vec<usize>]) -> Self {
        let n = adj.len();
        if n == 0 {
            return Self {
                centroid_parent: Vec::new(),
                root: None,
                depth: Vec::new(),
            };
        }

        let mut centroid_parent: Vec<Option<usize>> = vec![None; n];
        let mut depth = vec![0u32; n];
        let mut decomposed = vec![false; n];
        // Reused scratch — `parent_in_piece[v] == usize::MAX` means "not yet
        // touched by the current piece's DFS" and doubles as the visited
        // flag. `size[v]` is only meaningful while `v` is in the current
        // piece; we re-zero it at the end of each iteration.
        let mut parent_in_piece = vec![usize::MAX; n];
        let mut size = vec![0usize; n];
        let mut order: Vec<usize> = Vec::with_capacity(n);
        let mut stack: Vec<(usize, usize)> = Vec::with_capacity(n);

        // Stack frames for the centroid recursion. `seed` is any node still
        // alive in the piece we're decomposing; `cp` is the centroid that
        // handed us this piece (or `None` for the very first call, which
        // produces the centroid-tree root).
        let mut work: Vec<(usize, Option<usize>)> = vec![(0, None)];
        let mut root = None;

        while let Some((seed, cp)) = work.pop() {
            // ---- Pass 1: iterative DFS from `seed` over the current live
            // piece. Records `parent_in_piece` (DFS-tree parent — the seed
            // is its own parent) and `order` (post-order = children before
            // parent), with `size[v] = 1` stamped on first entry.
            order.clear();
            stack.clear();
            stack.push((seed, 0));
            parent_in_piece[seed] = seed;
            size[seed] = 1;
            while let Some(&(u, i)) = stack.last() {
                if i < adj[u].len() {
                    let v = adj[u][i];
                    stack.last_mut().unwrap().1 += 1;
                    if !decomposed[v] && parent_in_piece[v] == usize::MAX {
                        parent_in_piece[v] = u;
                        size[v] = 1;
                        stack.push((v, 0));
                    }
                } else {
                    order.push(u);
                    stack.pop();
                }
            }
            // Accumulate subtree sizes bottom-up.
            for &u in &order {
                if parent_in_piece[u] != u {
                    let p = parent_in_piece[u];
                    size[p] += size[u];
                }
            }
            let total = size[seed];

            // ---- Pass 2: walk down toward the centroid. From the current
            // candidate `c`, step into the heaviest live neighbour whose
            // subtree (as seen from `c`) exceeds `total / 2`; otherwise `c`
            // is the centroid. The walk visits each edge at most twice, so
            // it terminates in `O(piece-size)`.
            let mut centroid = seed;
            let mut prev = usize::MAX;
            loop {
                let mut next = None;
                for &v in &adj[centroid] {
                    if decomposed[v] || v == prev {
                        continue;
                    }
                    // Size of `v`'s subtree *rooted away from* `centroid`:
                    // - if `v` is `centroid`'s child in the piece-DFS, it's
                    //   just `size[v]`;
                    // - otherwise `v` is `centroid`'s ancestor, and the
                    //   subtree "above" `centroid` has `total - size[centroid]`
                    //   live nodes.
                    let sv = if parent_in_piece[v] == centroid {
                        size[v]
                    } else {
                        total - size[centroid]
                    };
                    if sv * 2 > total {
                        next = Some(v);
                        break;
                    }
                }
                match next {
                    Some(v) => {
                        prev = centroid;
                        centroid = v;
                    }
                    None => break,
                }
            }

            // Stitch the centroid into the centroid tree.
            centroid_parent[centroid] = cp;
            depth[centroid] = cp.map_or(0, |p| depth[p] + 1);
            if cp.is_none() && root.is_none() {
                root = Some(centroid);
            }
            decomposed[centroid] = true;

            // Reset scratch for live nodes in this piece so the next
            // iteration sees the sentinels again, then schedule each
            // surviving neighbour as the seed of its piece.
            for &u in &order {
                parent_in_piece[u] = usize::MAX;
                size[u] = 0;
            }
            for &v in &adj[centroid] {
                if !decomposed[v] {
                    work.push((v, Some(centroid)));
                }
            }
        }

        Self {
            centroid_parent,
            root,
            depth,
        }
    }

    /// Returns the parent of `v` in the centroid tree, or `None` if `v` is
    /// the root of the centroid tree (or lives in a component the
    /// decomposition never visited — see the module-level note on
    /// disconnected input).
    ///
    /// # Panics
    /// Panics if `v` is out of range for the tree this decomposition was
    /// built from.
    #[must_use]
    pub fn parent(&self, v: usize) -> Option<usize> {
        self.centroid_parent[v]
    }

    /// Returns the root of the centroid tree, or `None` for an empty graph.
    #[must_use]
    pub const fn root(&self) -> Option<usize> {
        self.root
    }

    /// Returns the depth of `v` in the centroid tree (root has depth 0).
    ///
    /// For nodes the decomposition never visited (other components of a
    /// disconnected input) this returns 0; consult
    /// [`CentroidDecomposition::parent`] to distinguish those from the root.
    ///
    /// # Panics
    /// Panics if `v` is out of range.
    #[must_use]
    pub fn depth(&self, v: usize) -> u32 {
        self.depth[v]
    }
}

#[cfg(test)]
mod tests {
    use super::CentroidDecomposition;
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

    /// For each centroid `c`, removing `c` from the *piece it was the
    /// centroid of* must leave every remaining component with at most
    /// `⌈piece_size / 2⌉` nodes. The piece of `c` is exactly the set of
    /// centroid-tree descendants of `c` (inclusive).
    fn assert_centroid_property(adj: &[Vec<usize>], cd: &CentroidDecomposition) {
        let n = adj.len();
        let mut children: Vec<Vec<usize>> = vec![vec![]; n];
        for v in 0..n {
            if let Some(p) = cd.parent(v) {
                children[p].push(v);
            }
        }
        for c in 0..n {
            // Skip nodes that weren't part of the decomposition (other
            // components of a disconnected input).
            if cd.parent(c).is_none() && Some(c) != cd.root() {
                continue;
            }
            // Descendant set in the centroid tree (inclusive) = the piece
            // `c` was the centroid of.
            let mut piece: HashSet<usize> = HashSet::new();
            let mut stk = vec![c];
            while let Some(u) = stk.pop() {
                if piece.insert(u) {
                    for &w in &children[u] {
                        stk.push(w);
                    }
                }
            }
            let k = piece.len();
            let cap = k.div_ceil(2);
            // BFS the original tree restricted to `piece \ {c}`. Each
            // connected component must satisfy `size ≤ cap`.
            let mut visited: HashSet<usize> = HashSet::new();
            visited.insert(c);
            for &start in &adj[c] {
                if !piece.contains(&start) || visited.contains(&start) {
                    continue;
                }
                let mut sz = 0usize;
                let mut q = VecDeque::from([start]);
                visited.insert(start);
                while let Some(u) = q.pop_front() {
                    sz += 1;
                    for &v in &adj[u] {
                        if piece.contains(&v) && !visited.contains(&v) {
                            visited.insert(v);
                            q.push_back(v);
                        }
                    }
                }
                assert!(
                    sz <= cap,
                    "centroid {c}: piece size {k}, removed component {sz} > cap {cap}"
                );
            }
        }
    }

    /// Deterministic XorShift-based random tree on `n` nodes.
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
            let p = (xorshift() as usize) % i;
            g[i].push(p);
            g[p].push(i);
        }
        g
    }

    /// `⌈log2 n⌉`, i.e. the smallest `k` with `2^k ≥ n`. Centroid-tree depth
    /// is bounded by this (each level halves the live piece).
    const fn ceil_log2(n: usize) -> u32 {
        if n <= 1 {
            0
        } else {
            usize::BITS - (n - 1).leading_zeros()
        }
    }

    #[test]
    fn empty_graph() {
        let g: Vec<Vec<usize>> = vec![];
        let cd = CentroidDecomposition::new(&g);
        assert_eq!(cd.root(), None);
    }

    #[test]
    fn single_node() {
        let g = build(1, &[]);
        let cd = CentroidDecomposition::new(&g);
        assert_eq!(cd.root(), Some(0));
        assert_eq!(cd.parent(0), None);
        assert_eq!(cd.depth(0), 0);
        assert_centroid_property(&g, &cd);
    }

    #[test]
    fn two_node_path() {
        // Either node is a valid centroid; both leave a single piece of
        // size 1 = ⌈2/2⌉. The seed-driven walk picks node 0 because no
        // neighbour exceeds total/2 = 1.
        let g = build(2, &[(0, 1)]);
        let cd = CentroidDecomposition::new(&g);
        let root = cd.root().expect("non-empty graph");
        let other = usize::from(root == 0);
        assert_eq!(cd.parent(other), Some(root));
        assert_eq!(cd.depth(other), 1);
        assert_centroid_property(&g, &cd);
    }

    #[test]
    fn path_centroid_is_middle() {
        // 0 — 1 — 2 — 3 — 4: the centroid of a 5-path is the middle node 2
        // (every other node leaves a piece of size 4 > ⌈5/2⌉ = 3).
        let g = build(5, &[(0, 1), (1, 2), (2, 3), (3, 4)]);
        let cd = CentroidDecomposition::new(&g);
        assert_eq!(cd.root(), Some(2));
        assert_eq!(cd.parent(2), None);
        assert_centroid_property(&g, &cd);
        for v in 0..5 {
            assert!(cd.depth(v) <= ceil_log2(5));
        }
    }

    #[test]
    fn star_root_is_centroid() {
        // Star with 4 leaves around node 0: removing 0 leaves 4 singletons
        // (≤ ⌈5/2⌉ = 3); removing any leaf leaves a 4-node star (> 3). So
        // the centroid is uniquely the hub, and every leaf is its child in
        // the centroid tree.
        let g = build(5, &[(0, 1), (0, 2), (0, 3), (0, 4)]);
        let cd = CentroidDecomposition::new(&g);
        assert_eq!(cd.root(), Some(0));
        assert_eq!(cd.parent(0), None);
        for v in 1..5 {
            assert_eq!(cd.parent(v), Some(0));
            assert_eq!(cd.depth(v), 1);
        }
        assert_centroid_property(&g, &cd);
    }

    #[test]
    fn balanced_binary_tree() {
        //            0
        //          /   \
        //         1     2
        //        / \   / \
        //       3   4 5   6
        //
        // Removing 0 leaves two 3-trees, both ≤ ⌈7/2⌉ = 4 → 0 is a
        // centroid. The seed-driven walk picks it.
        let g = build(7, &[(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)]);
        let cd = CentroidDecomposition::new(&g);
        assert_eq!(cd.root(), Some(0));
        for v in 0..7 {
            assert!(cd.depth(v) <= ceil_log2(7));
        }
        assert_centroid_property(&g, &cd);
    }

    #[test]
    fn classic_small_example() {
        // Spider tree (also used by the HLD tests):
        //
        //   0 — 1 — 2 — 3
        //       |   |
        //       4   6
        //       |
        //       5
        //
        // Removing 1 leaves pieces {0}, {4, 5}, {2, 3, 6} → max 3 ≤ ⌈7/2⌉ = 4.
        // Removing 2 leaves pieces {3}, {6}, {0, 1, 4, 5} → max 4 ≤ 4.
        // Both 1 and 2 are valid centroids.
        let n = 7;
        let g = build(n, &[(0, 1), (1, 2), (2, 3), (1, 4), (4, 5), (2, 6)]);
        let cd = CentroidDecomposition::new(&g);
        let root = cd.root().expect("non-empty graph");
        assert!(matches!(root, 1 | 2));
        for v in 0..n {
            assert!(cd.depth(v) <= ceil_log2(n));
        }
        assert_centroid_property(&g, &cd);
    }

    #[test]
    fn random_trees_satisfy_centroid_property() {
        // 200 deterministic seeds across n in 1..=30. Each centroid's
        // removal must leave every remaining piece bounded by ⌈k/2⌉,
        // and the centroid tree must have depth ≤ ⌈log2 n⌉.
        for seed in 0u64..200 {
            let n = ((seed % 30) + 1) as usize;
            let g = random_tree(n, seed);
            let cd = CentroidDecomposition::new(&g);
            assert_centroid_property(&g, &cd);
            let cap = ceil_log2(n);
            for v in 0..n {
                assert!(
                    cd.depth(v) <= cap,
                    "seed={seed} n={n} v={v} depth={} > cap {cap}",
                    cd.depth(v)
                );
            }
        }
    }

    #[allow(clippy::needless_pass_by_value)]
    #[quickcheck]
    fn quickcheck_centroid_tree_depth_logarithmic(n: u8, seed: u64) -> bool {
        let n = ((n as usize) % 30) + 1;
        let g = random_tree(n, seed);
        let cd = CentroidDecomposition::new(&g);
        let cap = ceil_log2(n);
        (0..n).all(|v| cd.depth(v) <= cap)
    }

    #[allow(clippy::needless_pass_by_value)]
    #[quickcheck]
    fn quickcheck_centroid_property_holds(n: u8, seed: u64) -> bool {
        let n = ((n as usize) % 30) + 1;
        let g = random_tree(n, seed);
        let cd = CentroidDecomposition::new(&g);
        // Re-run the full property check; assertion failures inside
        // become panics that quickcheck reports as a shrinkable failure.
        assert_centroid_property(&g, &cd);
        true
    }
}
