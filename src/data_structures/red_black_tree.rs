//! Red-black tree (Bayer 1972; Guibas-Sedgewick 1978).
//!
//! A **red-black tree** is a self-balancing binary search tree that maintains
//! five invariants (CLRS, 4th ed., §13):
//!
//! 1. Every node is either **red** or **black**.
//! 2. The root is **black**.
//! 3. Every leaf (NIL sentinel) is **black**.
//! 4. If a node is **red**, both its children are **black**.
//! 5. For each node, all simple paths from the node to descendant leaves
//!    contain the same number of **black** nodes (*black-height*).
//!
//! These invariants guarantee that the tree height is at most `2 log₂(n+1)`,
//! giving **O(log n)** worst-case time for `insert`, `remove`, and `contains`.
//!
//! # Design: arena-indexed slab
//! Nodes live in a `Vec<Node<K>>` (the "arena" or "slab").  Links between
//! nodes are `usize` indices rather than pointers or `Rc<RefCell<…>>`.
//! Index **0** is a permanent NIL sentinel with `color = Black` and
//! `key = None`; all leaf edges and unused parent pointers resolve to 0.
//! This avoids the ownership headaches of parent pointers while staying
//! entirely in **safe Rust** with no `Rc`/`RefCell` overhead.
//!
//! # Complexity
//! | Operation   | Time       | Space |
//! |-------------|------------|-------|
//! | `insert`    | O(log n)   | O(1) extra (amortised arena growth) |
//! | `remove`    | O(log n)   | O(1) extra |
//! | `contains`  | O(log n)   | O(1) extra |
//! | `min`/`max` | O(log n)   | O(1) extra |
//! | `iter_inorder` | O(n)   | O(log n) stack |
//! Space overall: **O(n)**.
//!
//! # Set semantics
//! Duplicate keys are rejected: `insert` returns `false` without modifying
//! the tree.

/// Node colour.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Color {
    Red,
    Black,
}

/// A single node in the arena.  Index 0 is the permanent NIL sentinel whose
/// `key` is `None` and whose `color` is always `Black`.
struct Node<K> {
    key: Option<K>,
    color: Color,
    parent: usize,
    left: usize,
    right: usize,
}

/// A red-black tree implementing an ordered set.
///
/// Generic over `K: Ord`.  Duplicate keys are silently rejected (`insert`
/// returns `false`).  The NIL sentinel lives at index 0 of the internal arena.
pub struct RedBlackTree<K> {
    nodes: Vec<Node<K>>,
    root: usize,
    len: usize,
}

// ── Construction / sentinel helpers ─────────────────────────────────────────

impl<K: Ord> RedBlackTree<K> {
    /// Creates an empty red-black tree.
    ///
    /// The NIL sentinel (index 0) is inserted automatically.
    #[must_use]
    pub fn new() -> Self {
        let nil = Node {
            key: None,
            color: Color::Black,
            parent: 0,
            left: 0,
            right: 0,
        };
        Self {
            nodes: vec![nil],
            root: 0,
            len: 0,
        }
    }

    /// The canonical index of the NIL sentinel.
    const NIL: usize = 0;

    /// Returns `true` if `idx` is the NIL sentinel.
    const fn is_nil(idx: usize) -> bool {
        idx == Self::NIL
    }

    /// Allocates a new red node in the arena and returns its index.
    fn alloc(&mut self, key: K) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(Node {
            key: Some(key),
            color: Color::Red,
            parent: Self::NIL,
            left: Self::NIL,
            right: Self::NIL,
        });
        idx
    }

    // ── Field accessors (avoid borrowing the whole `nodes` Vec) ─────────────

    fn color(&self, idx: usize) -> Color {
        self.nodes[idx].color
    }

    fn set_color(&mut self, idx: usize, c: Color) {
        self.nodes[idx].color = c;
    }

    fn parent(&self, idx: usize) -> usize {
        self.nodes[idx].parent
    }

    fn set_parent(&mut self, idx: usize, p: usize) {
        self.nodes[idx].parent = p;
    }

    fn left(&self, idx: usize) -> usize {
        self.nodes[idx].left
    }

    fn set_left(&mut self, idx: usize, l: usize) {
        self.nodes[idx].left = l;
    }

    fn right(&self, idx: usize) -> usize {
        self.nodes[idx].right
    }

    fn set_right(&mut self, idx: usize, r: usize) {
        self.nodes[idx].right = r;
    }

    fn key(&self, idx: usize) -> Option<&K> {
        self.nodes[idx].key.as_ref()
    }

    // ── Public API ───────────────────────────────────────────────────────────

    /// Returns the number of keys in the tree.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the tree contains no keys.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns `true` if `key` is present in the tree.
    ///
    /// Time: O(log n).
    #[must_use]
    pub fn contains(&self, key: &K) -> bool {
        self.find(key) != Self::NIL
    }

    /// Returns a reference to the smallest key, or `None` if empty.
    ///
    /// Time: O(log n).
    #[must_use]
    pub fn min(&self) -> Option<&K> {
        if self.root == Self::NIL {
            return None;
        }
        self.key(self.subtree_min(self.root))
    }

    /// Returns a reference to the largest key, or `None` if empty.
    ///
    /// Time: O(log n).
    #[must_use]
    pub fn max(&self) -> Option<&K> {
        if self.root == Self::NIL {
            return None;
        }
        self.key(self.subtree_max(self.root))
    }

    /// Inserts `key` into the tree.
    ///
    /// Returns `true` if the key was newly inserted, `false` if it was already
    /// present (set semantics — duplicates are ignored).
    ///
    /// Time: O(log n).
    pub fn insert(&mut self, key: K) -> bool {
        // Standard BST insert, tracking the parent.
        let mut parent = Self::NIL;
        let mut cur = self.root;
        let mut go_left = false;

        while !Self::is_nil(cur) {
            parent = cur;
            match key.cmp(self.nodes[cur].key.as_ref().expect("non-NIL node has key")) {
                std::cmp::Ordering::Less => {
                    go_left = true;
                    cur = self.left(cur);
                }
                std::cmp::Ordering::Greater => {
                    go_left = false;
                    cur = self.right(cur);
                }
                std::cmp::Ordering::Equal => return false, // duplicate
            }
        }

        let z = self.alloc(key);
        self.set_parent(z, parent);

        if Self::is_nil(parent) {
            // Tree was empty.
            self.root = z;
        } else if go_left {
            self.set_left(parent, z);
        } else {
            self.set_right(parent, z);
        }

        // z's children already point to NIL; z is Red.
        self.insert_fixup(z);
        self.len += 1;
        true
    }

    /// Removes `key` from the tree.
    ///
    /// Returns `true` if the key was present and removed, `false` otherwise.
    ///
    /// Time: O(log n).
    pub fn remove(&mut self, key: &K) -> bool {
        let z = self.find(key);
        if Self::is_nil(z) {
            return false;
        }
        self.rb_delete(z);
        self.len -= 1;
        true
    }

    /// Returns an iterator that yields references to keys in ascending order.
    ///
    /// Time: O(n) total.  Space: O(log n) for the traversal stack.
    pub fn iter_inorder(&self) -> impl Iterator<Item = &K> {
        // Collect indices into a Vec via iterative in-order traversal.
        let mut stack: Vec<usize> = Vec::new();
        let mut keys: Vec<&K> = Vec::with_capacity(self.len);
        let mut cur = self.root;
        loop {
            while !Self::is_nil(cur) {
                stack.push(cur);
                cur = self.left(cur);
            }
            match stack.pop() {
                None => break,
                Some(idx) => {
                    if let Some(k) = self.key(idx) {
                        keys.push(k);
                    }
                    cur = self.right(idx);
                }
            }
        }
        keys.into_iter()
    }

    // ── Internal helpers ─────────────────────────────────────────────────────

    /// BST search: returns the index of the node with `key`, or `NIL`.
    fn find(&self, key: &K) -> usize {
        let mut cur = self.root;
        while !Self::is_nil(cur) {
            match key.cmp(self.nodes[cur].key.as_ref().expect("non-NIL node has key")) {
                std::cmp::Ordering::Less => cur = self.left(cur),
                std::cmp::Ordering::Greater => cur = self.right(cur),
                std::cmp::Ordering::Equal => return cur,
            }
        }
        Self::NIL
    }

    /// Returns the index of the minimum-key node in the subtree rooted at `x`.
    fn subtree_min(&self, mut x: usize) -> usize {
        while !Self::is_nil(self.left(x)) {
            x = self.left(x);
        }
        x
    }

    /// Returns the index of the maximum-key node in the subtree rooted at `x`.
    fn subtree_max(&self, mut x: usize) -> usize {
        while !Self::is_nil(self.right(x)) {
            x = self.right(x);
        }
        x
    }

    // ── CLRS rotations (§13.2) ───────────────────────────────────────────────

    /// Left-rotate around `x`.  `right(x)` must not be NIL.
    ///
    /// ```text
    ///     x                   y
    ///    / \       =>        / \
    ///   a   y              x   c
    ///      / \            / \
    ///     b   c          a   b
    /// ```
    fn rotate_left(&mut self, x: usize) {
        let y = self.right(x);
        // Turn y's left subtree into x's right subtree.
        let b = self.left(y);
        self.set_right(x, b);
        if !Self::is_nil(b) {
            self.set_parent(b, x);
        }
        // Link x's parent to y.
        let px = self.parent(x);
        self.set_parent(y, px);
        if Self::is_nil(px) {
            self.root = y;
        } else if x == self.left(px) {
            self.set_left(px, y);
        } else {
            self.set_right(px, y);
        }
        // Put x on y's left.
        self.set_left(y, x);
        self.set_parent(x, y);
    }

    /// Right-rotate around `y`.  `left(y)` must not be NIL.
    fn rotate_right(&mut self, y: usize) {
        let x = self.left(y);
        let b = self.right(x);
        self.set_left(y, b);
        if !Self::is_nil(b) {
            self.set_parent(b, y);
        }
        let py = self.parent(y);
        self.set_parent(x, py);
        if Self::is_nil(py) {
            self.root = x;
        } else if y == self.left(py) {
            self.set_left(py, x);
        } else {
            self.set_right(py, x);
        }
        self.set_right(x, y);
        self.set_parent(y, x);
    }

    // ── CLRS insert fixup (§13.3) ────────────────────────────────────────────

    /// Restores RB invariants after inserting the red node `z`.
    fn insert_fixup(&mut self, mut z: usize) {
        while self.color(self.parent(z)) == Color::Red {
            let pz = self.parent(z);
            let gpz = self.parent(pz);
            if pz == self.left(gpz) {
                // Parent is a left child.
                let uncle = self.right(gpz);
                if self.color(uncle) == Color::Red {
                    // Case 1: uncle is red — recolour and move up.
                    self.set_color(pz, Color::Black);
                    self.set_color(uncle, Color::Black);
                    self.set_color(gpz, Color::Red);
                    z = gpz;
                } else {
                    if z == self.right(pz) {
                        // Case 2: uncle is black, z is a right child — rotate left.
                        z = pz;
                        self.rotate_left(z);
                    }
                    // Case 3: uncle is black, z is a left child — rotate right.
                    let pz2 = self.parent(z);
                    let gpz2 = self.parent(pz2);
                    self.set_color(pz2, Color::Black);
                    self.set_color(gpz2, Color::Red);
                    self.rotate_right(gpz2);
                }
            } else {
                // Symmetric: parent is a right child.
                let uncle = self.left(gpz);
                if self.color(uncle) == Color::Red {
                    // Case 1 (mirror).
                    self.set_color(pz, Color::Black);
                    self.set_color(uncle, Color::Black);
                    self.set_color(gpz, Color::Red);
                    z = gpz;
                } else {
                    if z == self.left(pz) {
                        // Case 2 (mirror).
                        z = pz;
                        self.rotate_right(z);
                    }
                    // Case 3 (mirror).
                    let pz2 = self.parent(z);
                    let gpz2 = self.parent(pz2);
                    self.set_color(pz2, Color::Black);
                    self.set_color(gpz2, Color::Red);
                    self.rotate_left(gpz2);
                }
            }
        }
        self.set_color(self.root, Color::Black);
    }

    // ── CLRS transplant helper (§13.4) ───────────────────────────────────────

    /// Replaces the subtree rooted at `u` with the subtree rooted at `v`.
    fn transplant(&mut self, u: usize, v: usize) {
        let pu = self.parent(u);
        if Self::is_nil(pu) {
            self.root = v;
        } else if u == self.left(pu) {
            self.set_left(pu, v);
        } else {
            self.set_right(pu, v);
        }
        // Update v's parent unconditionally (NIL sentinel's parent can be
        // written freely; it is never read for NIL's structural role).
        self.set_parent(v, pu);
    }

    // ── CLRS delete (§13.4) ──────────────────────────────────────────────────

    /// Deletes node `z` from the tree and restores RB invariants.
    fn rb_delete(&mut self, z: usize) {
        let mut y = z;
        let mut y_original_color = self.color(y);
        let x;

        if Self::is_nil(self.left(z)) {
            // Case A: no left child — splice out z.
            x = self.right(z);
            self.transplant(z, self.right(z));
        } else if Self::is_nil(self.right(z)) {
            // Case B: no right child — splice out z.
            x = self.left(z);
            self.transplant(z, self.left(z));
        } else {
            // Case C: two children — replace z with its in-order successor y.
            y = self.subtree_min(self.right(z));
            y_original_color = self.color(y);
            x = self.right(y);

            if self.parent(y) == z {
                // x's parent will be y after transplant.
                self.set_parent(x, y);
            } else {
                self.transplant(y, self.right(y));
                let rz = self.right(z);
                self.set_right(y, rz);
                self.set_parent(rz, y);
            }

            self.transplant(z, y);
            let lz = self.left(z);
            self.set_left(y, lz);
            self.set_parent(lz, y);
            self.set_color(y, self.color(z));
        }

        if y_original_color == Color::Black {
            self.delete_fixup(x);
        }
    }

    // ── CLRS delete fixup (§13.4) ────────────────────────────────────────────

    /// Restores RB invariants after removing a black node.  `x` is the node
    /// that moved into the deleted node's position (may be NIL).
    fn delete_fixup(&mut self, mut x: usize) {
        while x != self.root && self.color(x) == Color::Black {
            let px = self.parent(x);
            if x == self.left(px) {
                let mut w = self.right(px);

                if self.color(w) == Color::Red {
                    // Case 1: sibling is red.
                    self.set_color(w, Color::Black);
                    self.set_color(px, Color::Red);
                    self.rotate_left(px);
                    w = self.right(self.parent(x));
                }

                let pw = self.parent(x); // may have changed after rotation
                let w_left = self.left(w);
                let w_right = self.right(w);

                if self.color(w_left) == Color::Black && self.color(w_right) == Color::Black {
                    // Case 2: sibling's children are both black.
                    self.set_color(w, Color::Red);
                    x = pw;
                } else {
                    if self.color(w_right) == Color::Black {
                        // Case 3: sibling's right child is black.
                        self.set_color(w_left, Color::Black);
                        self.set_color(w, Color::Red);
                        self.rotate_right(w);
                        w = self.right(self.parent(x));
                    }
                    // Case 4: sibling's right child is red.
                    let px2 = self.parent(x);
                    self.set_color(w, self.color(px2));
                    self.set_color(px2, Color::Black);
                    self.set_color(self.right(w), Color::Black);
                    self.rotate_left(px2);
                    x = self.root;
                }
            } else {
                // Symmetric: x is a right child.
                let mut w = self.left(px);

                if self.color(w) == Color::Red {
                    // Case 1 (mirror).
                    self.set_color(w, Color::Black);
                    self.set_color(px, Color::Red);
                    self.rotate_right(px);
                    w = self.left(self.parent(x));
                }

                let pw = self.parent(x);
                let w_left = self.left(w);
                let w_right = self.right(w);

                if self.color(w_right) == Color::Black && self.color(w_left) == Color::Black {
                    // Case 2 (mirror).
                    self.set_color(w, Color::Red);
                    x = pw;
                } else {
                    if self.color(w_left) == Color::Black {
                        // Case 3 (mirror).
                        self.set_color(w_right, Color::Black);
                        self.set_color(w, Color::Red);
                        self.rotate_left(w);
                        w = self.left(self.parent(x));
                    }
                    // Case 4 (mirror).
                    let px2 = self.parent(x);
                    self.set_color(w, self.color(px2));
                    self.set_color(px2, Color::Black);
                    self.set_color(self.left(w), Color::Black);
                    self.rotate_right(px2);
                    x = self.root;
                }
            }
        }
        self.set_color(x, Color::Black);
    }

    // ── Invariant checker (debug / test helper) ──────────────────────────────

    /// Verifies all five RB invariants.
    ///
    /// Returns `Ok(black_height)` on success, `Err(message)` on first failure.
    /// Exported so property tests and inline tests can call it.
    pub fn verify_rb_invariants(&self) -> Result<(), &'static str> {
        // Invariant 2: root must be black.
        if !Self::is_nil(self.root) && self.color(self.root) != Color::Black {
            return Err("invariant 2: root is not black");
        }
        // Recursive check starting at root.  Returns the black-height of the
        // subtree (not counting NIL leaves themselves).
        self.check_node(self.root).map(|_| ())
    }

    /// Recursive RB checker.  Returns the black-height of the subtree rooted
    /// at `x` (number of black nodes on every simple path from `x` down to a
    /// NIL, excluding `x` itself when it is NIL).
    ///
    /// `usize::MAX` is used as a sentinel meaning "not yet known".
    fn check_node(&self, x: usize) -> Result<usize, &'static str> {
        if Self::is_nil(x) {
            // Invariant 3: NIL leaves are black (always true by construction).
            return Ok(0);
        }

        // Invariant 4: a red node must have black children.
        if self.color(x) == Color::Red {
            if self.color(self.left(x)) == Color::Red {
                return Err("invariant 4: red node has red left child");
            }
            if self.color(self.right(x)) == Color::Red {
                return Err("invariant 4: red node has red right child");
            }
        }

        let lh = self.check_node(self.left(x))?;
        let rh = self.check_node(self.right(x))?;

        // Invariant 5: equal black-height on both sides.
        if lh != rh {
            return Err("invariant 5: unequal black-height");
        }

        let own = usize::from(self.color(x) == Color::Black);
        Ok(lh + own)
    }
}

impl<K: Ord> Default for RedBlackTree<K> {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::RedBlackTree;
    use quickcheck::TestResult;
    use quickcheck_macros::quickcheck;
    use std::collections::BTreeSet;

    // ── helpers ──────────────────────────────────────────────────────────────

    fn assert_invariants<K: Ord>(t: &RedBlackTree<K>) {
        t.verify_rb_invariants().expect("RB invariants violated");
    }

    // ── basic structural tests ────────────────────────────────────────────────

    #[test]
    fn empty_tree() {
        let t: RedBlackTree<i32> = RedBlackTree::new();
        assert!(t.is_empty());
        assert_eq!(t.len(), 0);
        assert!(!t.contains(&0));
        assert_eq!(t.min(), None);
        assert_eq!(t.max(), None);
        assert_eq!(t.iter_inorder().count(), 0);
        assert_invariants(&t);
    }

    #[test]
    fn single_insert_and_contains() {
        let mut t = RedBlackTree::new();
        assert!(t.insert(42_i32));
        assert_eq!(t.len(), 1);
        assert!(!t.is_empty());
        assert!(t.contains(&42));
        assert!(!t.contains(&0));
        assert_eq!(t.min(), Some(&42));
        assert_eq!(t.max(), Some(&42));
        assert_invariants(&t);
    }

    #[test]
    fn duplicate_returns_false() {
        let mut t = RedBlackTree::new();
        assert!(t.insert(7_i32));
        assert!(!t.insert(7_i32));
        assert_eq!(t.len(), 1);
        assert_invariants(&t);
    }

    #[test]
    fn inorder_is_sorted() {
        let mut t = RedBlackTree::new();
        for &k in &[5, 3, 8, 1, 4, 9, 2, 7, 6, 0_i32] {
            t.insert(k);
        }
        let keys: Vec<_> = t.iter_inorder().copied().collect();
        let mut sorted = keys.clone();
        sorted.sort_unstable();
        assert_eq!(keys, sorted);
        assert_invariants(&t);
    }

    #[test]
    fn min_and_max_after_inserts() {
        let mut t = RedBlackTree::new();
        for k in [10_i32, 5, 15, 3, 7] {
            t.insert(k);
        }
        assert_eq!(t.min(), Some(&3));
        assert_eq!(t.max(), Some(&15));
        assert_invariants(&t);
    }

    // ── sequential insertions ─────────────────────────────────────────────────

    #[test]
    fn sequential_insert_1_to_200() {
        let mut t = RedBlackTree::new();
        for k in 1..=200_i32 {
            assert!(t.insert(k));
            assert_invariants(&t);
        }
        assert_eq!(t.len(), 200);
        let keys: Vec<i32> = t.iter_inorder().copied().collect();
        assert_eq!(keys, (1..=200).collect::<Vec<_>>());
    }

    #[test]
    fn reverse_sequential_insert_200_to_1() {
        let mut t = RedBlackTree::new();
        for k in (1..=200_i32).rev() {
            assert!(t.insert(k));
            assert_invariants(&t);
        }
        assert_eq!(t.len(), 200);
        let keys: Vec<i32> = t.iter_inorder().copied().collect();
        assert_eq!(keys, (1..=200).collect::<Vec<_>>());
    }

    // ── remove tests ─────────────────────────────────────────────────────────

    #[test]
    fn remove_absent_returns_false() {
        let mut t: RedBlackTree<i32> = RedBlackTree::new();
        assert!(!t.remove(&99));
        t.insert(1);
        assert!(!t.remove(&99));
        assert_invariants(&t);
    }

    #[test]
    fn remove_leaf() {
        let mut t = RedBlackTree::new();
        for k in [5_i32, 3, 7] {
            t.insert(k);
        }
        assert!(t.remove(&3));
        assert_eq!(t.len(), 2);
        assert!(!t.contains(&3));
        assert_invariants(&t);
    }

    #[test]
    fn remove_root() {
        let mut t = RedBlackTree::new();
        t.insert(10_i32);
        assert!(t.remove(&10));
        assert!(t.is_empty());
        assert_eq!(t.min(), None);
        assert_eq!(t.max(), None);
        assert_invariants(&t);
    }

    #[test]
    fn remove_root_two_children() {
        let mut t = RedBlackTree::new();
        for k in [10_i32, 5, 15, 3, 7, 12, 20] {
            t.insert(k);
        }
        // Root after sequential inserts into a balanced tree is 10.
        assert!(t.remove(&10));
        assert_eq!(t.len(), 6);
        assert!(!t.contains(&10));
        assert_invariants(&t);
    }

    #[test]
    fn remove_internal_node_two_children() {
        let mut t = RedBlackTree::new();
        for k in [20_i32, 10, 30, 5, 15, 25, 35] {
            t.insert(k);
        }
        // Remove 10 which has two children.
        assert!(t.remove(&10));
        assert_eq!(t.len(), 6);
        assert!(!t.contains(&10));
        let keys: Vec<_> = t.iter_inorder().copied().collect();
        assert_eq!(keys, vec![5, 15, 20, 25, 30, 35]);
        assert_invariants(&t);
    }

    #[test]
    fn mixed_insert_remove_with_invariant_check() {
        let mut t = RedBlackTree::new();
        let ops: &[(bool, i32)] = &[
            (true, 50),
            (true, 25),
            (true, 75),
            (true, 10),
            (true, 40),
            (true, 60),
            (true, 90),
            (false, 25),
            (false, 75),
            (true, 5),
            (true, 30),
            (false, 50),
            (true, 55),
            (false, 10),
            (false, 90),
        ];
        for &(insert, key) in ops {
            if insert {
                t.insert(key);
            } else {
                t.remove(&key);
            }
            assert_invariants(&t);
        }
    }

    #[test]
    fn full_insert_then_full_remove() {
        let mut t = RedBlackTree::new();
        let n = 100_i32;
        for k in 1..=n {
            t.insert(k);
        }
        assert_invariants(&t);
        for k in 1..=n {
            assert!(t.remove(&k), "failed to remove {k}");
            assert_invariants(&t);
        }
        assert!(t.is_empty());
    }

    // ── property test against BTreeSet oracle ────────────────────────────────

    #[derive(Clone, Debug)]
    enum Op {
        Insert(i32),
        Remove(i32),
        Contains(i32),
    }

    impl quickcheck::Arbitrary for Op {
        fn arbitrary(g: &mut quickcheck::Gen) -> Self {
            let key = i32::arbitrary(g) % 50; // keep keys small so collisions occur
            match u8::arbitrary(g) % 3 {
                0 => Self::Insert(key),
                1 => Self::Remove(key),
                _ => Self::Contains(key),
            }
        }
    }

    #[allow(clippy::needless_pass_by_value)]
    #[quickcheck]
    fn prop_matches_btreeset(ops: Vec<Op>) -> TestResult {
        if ops.len() > 200 {
            return TestResult::discard();
        }

        let mut rbt: RedBlackTree<i32> = RedBlackTree::new();
        let mut oracle: BTreeSet<i32> = BTreeSet::new();

        for op in &ops {
            match *op {
                Op::Insert(k) => {
                    let r = rbt.insert(k);
                    let o = oracle.insert(k);
                    if r != o {
                        return TestResult::failed();
                    }
                }
                Op::Remove(k) => {
                    let r = rbt.remove(&k);
                    let o = oracle.remove(&k);
                    if r != o {
                        return TestResult::failed();
                    }
                }
                Op::Contains(k) => {
                    if rbt.contains(&k) != oracle.contains(&k) {
                        return TestResult::failed();
                    }
                }
            }

            // After every operation: invariants must hold.
            if rbt.verify_rb_invariants().is_err() {
                return TestResult::failed();
            }

            // In-order output must match sorted BTreeSet output.
            let rbt_keys: Vec<i32> = rbt.iter_inorder().copied().collect();
            let oracle_keys: Vec<i32> = oracle.iter().copied().collect();
            if rbt_keys != oracle_keys {
                return TestResult::failed();
            }

            if rbt.len() != oracle.len() {
                return TestResult::failed();
            }
        }

        TestResult::passed()
    }
}
