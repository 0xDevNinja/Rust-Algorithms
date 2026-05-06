//! Link-cut tree (Sleator & Tarjan, 1983).
//!
//! A **link-cut tree** represents a forest of dynamic trees, supporting
//! the following operations all in **amortised O(log n)** time:
//!
//! | Operation      | Description                                         |
//! |----------------|-----------------------------------------------------|
//! | `link`         | Add an edge between two nodes in different trees    |
//! | `cut`          | Remove the edge from a node to its parent           |
//! | `find_root`    | Return the root of the tree containing a node       |
//! | `connected`    | Test whether two nodes are in the same tree         |
//! | `path_max`     | Maximum node value on the path between two nodes    |
//! | `update`       | Change the value stored at a node                   |
//!
//! # Algorithm overview
//!
//! The forest is partitioned into **preferred paths** (chains).  Each chain is
//! stored as an auxiliary splay tree keyed by depth.  The crucial invariant is
//! that every splay-tree node additionally holds a `path_parent` pointer that
//! skips over the gap to the topmost node of the path above — allowing `O(log n)`
//! amortised `access` operations that re-arrange preferred paths as needed.
//!
//! Nodes are stored in a plain `Vec` arena so all pointers are `usize` indices
//! and no `unsafe` is required.
//!
//! # Complexity
//! - Time: O(log n) amortised per operation (splay tree potential argument).
//! - Space: O(n).
//!
//! # Preconditions
//! - Node indices passed to all public methods must be in `0..self.len()`.
//! - `link(child, parent)`: `child` must be the root of its represented tree.
//!   Calling `link` when `child` and `parent` are already connected is a no-op.

/// Sentinel index meaning "no node".
const NIL: usize = usize::MAX;

/// One arena node.  The auxiliary splay tree is indexed by depth on the
/// represented path; left = shallower, right = deeper.
#[derive(Clone, Debug)]
struct Node {
    /// Parent in the auxiliary splay tree (NIL if this is the splay root).
    parent: usize,
    /// Left child in the auxiliary splay tree (shallower direction).
    left: usize,
    /// Right child in the auxiliary splay tree (deeper direction).
    right: usize,
    /// Path-parent pointer: points to the splay root of the path immediately
    /// above this chain.  Only valid when `parent == NIL` (i.e. this node is
    /// the root of its auxiliary splay tree).
    path_parent: usize,
    /// The value stored at this node in the represented forest.
    value: i64,
    /// Maximum `value` in the subtree rooted here in the auxiliary splay tree.
    max_in_subtree: i64,
    /// Lazy "flip" flag: when true, the left/right children of this subtree
    /// need to be swapped to implement `make_root` (path reversal).
    flip: bool,
}

impl Node {
    const fn new(value: i64) -> Self {
        Self {
            parent: NIL,
            left: NIL,
            right: NIL,
            path_parent: NIL,
            value,
            max_in_subtree: value,
            flip: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Public struct
// ---------------------------------------------------------------------------

/// Sleator-Tarjan link-cut tree over a forest of `n` nodes.
///
/// Node values are `i64`; path aggregation supports `path_max`.
pub struct LinkCutTree {
    nodes: Vec<Node>,
}

impl LinkCutTree {
    /// Creates `n` isolated nodes with values taken from `values`.
    ///
    /// # Panics
    /// Never panics.
    #[must_use]
    pub fn new(values: Vec<i64>) -> Self {
        Self {
            nodes: values.into_iter().map(Node::new).collect(),
        }
    }

    /// Returns the number of nodes.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Returns `true` if the forest has no nodes.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.nodes.len() == 0
    }

    // -----------------------------------------------------------------------
    // Public operations
    // -----------------------------------------------------------------------

    /// Links `child` (which must be a root of its represented tree) as a child
    /// of `parent`.  No-op if they are already in the same tree.
    pub fn link(&mut self, child: usize, parent: usize) {
        if self.connected(child, parent) {
            return;
        }
        // After make_root(child) the splay tree for child's chain contains only
        // child itself (child is the root of its represented tree, so it has no
        // nodes shallower than it on its preferred path).
        self.make_root(child);
        // Set path_parent: child's auxiliary-splay-root now points upward to parent's chain.
        // We also access(parent) so parent is the root of its splay tree.
        self.access(parent);
        // Now attach child's chain below parent by setting the path_parent pointer.
        // child's splay root is child itself (it has no splay-tree parent after make_root).
        self.nodes[child].path_parent = parent;
    }

    /// Cuts the edge from `child` to its parent in the represented tree.
    /// No-op if `child` is already a root.
    pub fn cut(&mut self, child: usize) {
        // access(child) makes `child` the splay root of its auxiliary tree.
        // The left subtree of child in the splay tree represents all nodes on
        // the root-to-child path that are shallower than child — i.e. child's
        // ancestors including its direct parent.  Detaching that left subtree
        // completely severs child from all its ancestors, making child a root.
        self.access(child);
        let left = self.nodes[child].left;
        if left != NIL {
            self.nodes[left].parent = NIL;
            self.nodes[child].left = NIL;
            self.pull_up(child);
        }
    }

    /// Returns the root of the represented tree containing `v`.
    pub fn find_root(&mut self, v: usize) -> usize {
        self.access(v);
        // The root of the represented tree is the leftmost node of the splay tree
        // (shallowest = smallest depth = the actual root).
        let mut x = v;
        self.push_down(x);
        while self.nodes[x].left != NIL {
            x = self.nodes[x].left;
            self.push_down(x);
        }
        // Splay the found root to the splay-tree root for future O(log n) amortisation.
        self.splay(x);
        x
    }

    /// Returns `true` iff `u` and `v` are in the same represented tree.
    pub fn connected(&mut self, u: usize, v: usize) -> bool {
        self.find_root(u) == self.find_root(v)
    }

    /// Returns the maximum node value on the path from `u` to `v`, or
    /// `i64::MIN` if `u` and `v` are not in the same tree.
    ///
    /// This operation does not permanently change the tree's root.
    pub fn path_max(&mut self, u: usize, v: usize) -> i64 {
        if !self.connected(u, v) {
            return i64::MIN;
        }
        // Save the current root so we can restore it afterwards.
        let original_root = self.find_root(u);
        // Temporarily re-root at u so the path u→v lies in the splay tree
        // for v's chain after access(v).
        self.make_root(u);
        self.access(v);
        let result = self.nodes[v].max_in_subtree;
        // Restore the original root so the tree root is unchanged from the
        // caller's perspective.
        self.make_root(original_root);
        result
    }

    /// Updates the value stored at node `v`.
    pub fn update(&mut self, v: usize, value: i64) {
        // Splay v to its chain's root so pull_up reaches it conveniently.
        self.access(v);
        self.nodes[v].value = value;
        self.pull_up(v);
    }

    // -----------------------------------------------------------------------
    // Core splay-tree helpers
    // -----------------------------------------------------------------------

    /// Returns `true` iff `x` is the root of its auxiliary splay tree.
    /// A node is a splay root either if `parent == NIL` OR if its parent's
    /// left/right child is not `x` (i.e. the parent-pointer is a path-parent,
    /// not a splay-tree edge).
    fn is_splay_root(&self, x: usize) -> bool {
        let p = self.nodes[x].parent;
        if p == NIL {
            return true;
        }
        self.nodes[p].left != x && self.nodes[p].right != x
    }

    /// Recalculates `max_in_subtree` for node `x` from its children.
    fn pull_up(&mut self, x: usize) {
        let mut m = self.nodes[x].value;
        let l = self.nodes[x].left;
        let r = self.nodes[x].right;
        if l != NIL {
            m = m.max(self.nodes[l].max_in_subtree);
        }
        if r != NIL {
            m = m.max(self.nodes[r].max_in_subtree);
        }
        self.nodes[x].max_in_subtree = m;
    }

    /// Pushes the lazy flip flag down one level.
    fn push_down(&mut self, x: usize) {
        if self.nodes[x].flip {
            let l = self.nodes[x].left;
            let r = self.nodes[x].right;
            // Swap left and right children.
            self.nodes[x].left = r;
            self.nodes[x].right = l;
            // Propagate flip to children.
            if l != NIL {
                self.nodes[l].flip ^= true;
            }
            if r != NIL {
                self.nodes[r].flip ^= true;
            }
            self.nodes[x].flip = false;
        }
    }

    /// Pushes all lazy flags from the splay-tree root down to `x`.
    /// We collect ancestors on a stack then push top-down.
    fn push_all(&mut self, x: usize) {
        // Collect the chain from x up to the splay root.
        let mut stack = Vec::new();
        let mut cur = x;
        while !self.is_splay_root(cur) {
            stack.push(cur);
            cur = self.nodes[cur].parent;
        }
        stack.push(cur);
        // Push down from the splay root toward x.
        while let Some(node) = stack.pop() {
            self.push_down(node);
        }
    }

    /// Rotates `x` up one level in the splay tree.
    ///
    /// If `x` is a left child this is a right rotation of its parent; if `x`
    /// is a right child this is a left rotation.  Path-parent pointers are
    /// maintained so the splay root's `path_parent` is transferred correctly.
    fn rotate(&mut self, x: usize) {
        let p = self.nodes[x].parent;
        let g = self.nodes[p].parent;
        let p_is_splay_root = self.is_splay_root(p);

        if self.nodes[p].left == x {
            // x is left child — right rotation at p.
            let xr = self.nodes[x].right;
            // x's right child becomes p's left child.
            self.nodes[p].left = xr;
            if xr != NIL {
                self.nodes[xr].parent = p;
            }
            // p becomes x's right child.
            self.nodes[x].right = p;
        } else {
            // x is right child — left rotation at p.
            let xl = self.nodes[x].left;
            self.nodes[p].right = xl;
            if xl != NIL {
                self.nodes[xl].parent = p;
            }
            self.nodes[x].left = p;
        }

        self.nodes[p].parent = x;
        self.nodes[x].parent = g;

        // Transfer path-parent: if p was a splay root, x inherits path_parent.
        if p_is_splay_root {
            let pp = self.nodes[p].path_parent;
            self.nodes[x].path_parent = pp;
            self.nodes[p].path_parent = NIL;
        }

        // Fix grandparent's child pointer if p was a non-root splay-tree node.
        if g != NIL && !p_is_splay_root {
            if self.nodes[g].left == p {
                self.nodes[g].left = x;
            } else if self.nodes[g].right == p {
                self.nodes[g].right = x;
            }
        }

        self.pull_up(p);
        self.pull_up(x);
    }

    /// Splays `x` to the root of its auxiliary splay tree.
    ///
    /// Uses zig / zig-zig / zig-zag cases.  Before rotating, all lazy flags
    /// are pushed from the current splay root down to `x`.
    fn splay(&mut self, x: usize) {
        // Push pending flips down to x before we start rotating.
        self.push_all(x);

        while !self.is_splay_root(x) {
            let p = self.nodes[x].parent;
            if !self.is_splay_root(p) {
                let g = self.nodes[p].parent;
                // Zig-zig: x and p are both left or both right children.
                let x_is_left_of_p = self.nodes[p].left == x;
                let p_is_left_of_g = self.nodes[g].left == p;
                if x_is_left_of_p == p_is_left_of_g {
                    self.rotate(p); // rotate parent first (zig-zig)
                } else {
                    self.rotate(x); // rotate x first (zig-zag)
                }
            }
            self.rotate(x);
        }
    }

    /// `access(v)`: the central operation.
    ///
    /// After `access(v)`, the auxiliary splay tree rooted at `v` contains
    /// exactly the nodes on the path from the represented-tree root down to `v`.
    /// `v` is the splay root; it has no right child.
    fn access(&mut self, v: usize) {
        let mut last = NIL;
        let mut x = v;

        loop {
            self.splay(x);

            // Detach x's current preferred child (right subtree) and attach
            // `last` (the chain we have been building) as the new right child.
            let old_right = self.nodes[x].right;
            if old_right != NIL {
                // The detached subtree becomes an independent chain; its splay
                // root (old_right) gets path_parent pointing to x.
                self.nodes[old_right].parent = NIL;
                self.nodes[old_right].path_parent = x;
            }

            if last != NIL {
                self.nodes[last].parent = x;
                self.nodes[last].path_parent = NIL;
            }
            self.nodes[x].right = last;
            self.pull_up(x);

            last = x;

            // Follow path-parent to the next chain above.
            let pp = self.nodes[x].path_parent;
            if pp == NIL {
                break;
            }
            x = pp;
        }

        // v is now the deepest node; splay it to the root of the full chain.
        self.splay(v);
    }

    /// `make_root(v)`: re-roots the represented tree at `v`.
    ///
    /// After this call, `v` is the root of its represented tree.
    /// Implemented by `access(v)` followed by reversing the path (lazy flip).
    fn make_root(&mut self, v: usize) {
        self.access(v);
        // v is now the splay root with the full root-to-v path in the splay tree.
        // Reversing the path makes v the shallowest node (the new root).
        self.nodes[v].flip ^= true;
        // Immediately push down so subsequent operations see correct structure.
        self.push_down(v);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::LinkCutTree;
    use quickcheck::TestResult;
    use quickcheck_macros::quickcheck;

    // -----------------------------------------------------------------------
    // Unit tests
    // -----------------------------------------------------------------------

    #[test]
    fn empty_tree() {
        let lct = LinkCutTree::new(vec![]);
        assert!(lct.is_empty());
        assert_eq!(lct.len(), 0);
    }

    #[test]
    fn single_node() {
        let mut lct = LinkCutTree::new(vec![42]);
        assert_eq!(lct.len(), 1);
        assert!(!lct.is_empty());
        assert_eq!(lct.find_root(0), 0);
        assert!(lct.connected(0, 0));
        assert_eq!(lct.path_max(0, 0), 42);
    }

    #[test]
    fn path_0_to_4_find_root_and_path_max() {
        // Build the path 0 – 1 – 2 – 3 – 4 (each node i+1 linked under i).
        let mut lct = LinkCutTree::new(vec![10, 20, 5, 30, 15]);
        lct.link(1, 0);
        lct.link(2, 1);
        lct.link(3, 2);
        lct.link(4, 3);

        // All nodes should be in the same tree with root = 0.
        for i in 0..5 {
            assert_eq!(lct.find_root(i), 0, "node {i} should have root 0");
        }

        // path_max(0, 4) should be max(10,20,5,30,15) = 30.
        assert_eq!(lct.path_max(0, 4), 30);
        // path_max(1, 3) should be max(20,5,30) = 30.
        assert_eq!(lct.path_max(1, 3), 30);
        // path_max(0, 2) should be max(10,20,5) = 20.
        assert_eq!(lct.path_max(0, 2), 20);
    }

    #[test]
    fn connectivity_after_cut() {
        let mut lct = LinkCutTree::new(vec![1, 2, 3]);
        lct.link(1, 0);
        lct.link(2, 1);

        assert!(lct.connected(0, 2));

        // Cut the edge between node 1 and its parent (0).
        lct.cut(1);

        // 0 is now isolated; 1 and 2 are still connected.
        assert!(!lct.connected(0, 1));
        assert!(!lct.connected(0, 2));
        assert!(lct.connected(1, 2));
        assert_eq!(lct.find_root(0), 0);
    }

    #[test]
    fn relink_after_cut_creates_different_topology() {
        // Star: 1,2,3 all linked to 0.
        let mut lct = LinkCutTree::new(vec![1; 4]);
        lct.link(1, 0);
        lct.link(2, 0);
        lct.link(3, 0);

        // Detach 1 and reattach it under 2.
        lct.cut(1);
        assert!(!lct.connected(0, 1));
        lct.link(1, 2);
        assert!(lct.connected(0, 1));

        // Root is still 0.
        assert_eq!(lct.find_root(1), 0);
        assert_eq!(lct.find_root(3), 0);
    }

    #[test]
    fn update_changes_path_max() {
        let mut lct = LinkCutTree::new(vec![1, 2, 3]);
        lct.link(1, 0);
        lct.link(2, 1);

        assert_eq!(lct.path_max(0, 2), 3);
        lct.update(2, 100);
        assert_eq!(lct.path_max(0, 2), 100);
        lct.update(2, 1);
        assert_eq!(lct.path_max(0, 2), 2);
    }

    #[test]
    fn path_max_returns_min_when_not_connected() {
        let mut lct = LinkCutTree::new(vec![5, 10]);
        // 0 and 1 are not linked.
        assert_eq!(lct.path_max(0, 1), i64::MIN);
    }

    // -----------------------------------------------------------------------
    // Naive oracle for property-based testing
    // -----------------------------------------------------------------------

    /// Simple parent-array forest: parent[v] = Some(p) means v's parent is p.
    struct NaiveForest {
        parent: Vec<Option<usize>>,
        value: Vec<i64>,
    }

    impl NaiveForest {
        fn new(values: Vec<i64>) -> Self {
            let n = values.len();
            Self {
                parent: vec![None; n],
                value: values,
            }
        }

        fn root(&self, mut v: usize) -> usize {
            while let Some(p) = self.parent[v] {
                v = p;
            }
            v
        }

        fn connected(&self, u: usize, v: usize) -> bool {
            self.root(u) == self.root(v)
        }

        /// True iff `v` has no parent (is a root).
        fn is_root(&self, v: usize) -> bool {
            self.parent[v].is_none()
        }

        fn link(&mut self, child: usize, parent: usize) {
            if self.connected(child, parent) {
                return;
            }
            // child must be a root in the naive oracle too.
            if !self.is_root(child) {
                return;
            }
            self.parent[child] = Some(parent);
        }

        fn cut(&mut self, child: usize) {
            self.parent[child] = None;
        }

        fn path_max(&self, u: usize, v: usize) -> i64 {
            if !self.connected(u, v) {
                return i64::MIN;
            }
            // Collect the path from u to LCA and v to LCA.
            let path_u = self.path_to_root(u);
            let path_v = self.path_to_root(v);
            // Find LCA: last common ancestor.
            let lca = *path_u
                .iter()
                .rev()
                .zip(path_v.iter().rev())
                .take_while(|(a, b)| a == b)
                .map(|(a, _)| a)
                .last()
                .expect("connected nodes must share a root");
            // Collect nodes on path u→lca and v→lca.
            let mut nodes = Vec::new();
            for &n in &path_u {
                nodes.push(n);
                if n == lca {
                    break;
                }
            }
            for &n in &path_v {
                nodes.push(n);
                if n == lca {
                    break;
                }
            }
            nodes
                .iter()
                .map(|&n| self.value[n])
                .max()
                .unwrap_or(i64::MIN)
        }

        fn path_to_root(&self, mut v: usize) -> Vec<usize> {
            let mut path = Vec::new();
            loop {
                path.push(v);
                match self.parent[v] {
                    Some(p) => v = p,
                    None => break,
                }
            }
            path
        }

        fn update(&mut self, v: usize, val: i64) {
            self.value[v] = val;
        }
    }

    // -----------------------------------------------------------------------
    // quickcheck property
    // -----------------------------------------------------------------------

    #[derive(Clone, Debug)]
    enum Op {
        Link(usize, usize),
        Cut(usize),
        Connected(usize, usize),
        PathMax(usize, usize),
        Update(usize, i64),
    }

    impl quickcheck::Arbitrary for Op {
        fn arbitrary(g: &mut quickcheck::Gen) -> Self {
            let n: usize = 6; // small forest to get interesting topologies
            let u = usize::arbitrary(g) % n;
            let v = usize::arbitrary(g) % n;
            let val = i64::arbitrary(g) % 100;
            match u8::arbitrary(g) % 5 {
                0 => Self::Link(u, v),
                1 => Self::Cut(u),
                2 => Self::Connected(u, v),
                3 => Self::PathMax(u, v),
                _ => Self::Update(u, val),
            }
        }
    }

    #[quickcheck]
    #[allow(clippy::needless_pass_by_value)]
    fn prop_matches_naive(ops: Vec<Op>) -> TestResult {
        if ops.len() > 150 {
            return TestResult::discard();
        }

        let n = 6_usize;
        let values: Vec<i64> = (0..n as i64).collect();

        let mut lct = LinkCutTree::new(values.clone());
        let mut naive = NaiveForest::new(values);

        for op in &ops {
            match *op {
                Op::Link(child, parent) => {
                    if child == parent {
                        continue;
                    }
                    // Only link if child is a root in the naive model.
                    if !naive.is_root(child) {
                        continue;
                    }
                    naive.link(child, parent);
                    lct.link(child, parent);
                }
                Op::Cut(child) => {
                    naive.cut(child);
                    lct.cut(child);
                }
                Op::Connected(u, v) => {
                    let expected = naive.connected(u, v);
                    let got = lct.connected(u, v);
                    if expected != got {
                        return TestResult::failed();
                    }
                }
                Op::PathMax(u, v) => {
                    let expected = naive.path_max(u, v);
                    let got = lct.path_max(u, v);
                    if expected != got {
                        return TestResult::failed();
                    }
                }
                Op::Update(v, val) => {
                    naive.update(v, val);
                    lct.update(v, val);
                }
            }
        }

        TestResult::passed()
    }
}
