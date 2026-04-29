//! Splay tree (Sleator & Tarjan, 1985).
//!
//! A **splay tree** is a self-adjusting binary search tree that performs a
//! *splay* operation on every access, rotating the accessed node (or the
//! nearest predecessor/successor on a miss) to the root.  No per-node
//! balance metadata is stored.
//!
//! # Complexity
//! All operations — insert, remove, contains — run in **amortised O(log n)**
//! time.  The amortised analysis uses a potential function equal to the sum
//! of log(subtree sizes); individual operations may touch O(n) nodes but the
//! total cost over any sequence of m operations on an n-node tree is
//! O((m + n) log n).
//!
//! # Working-set theorem
//! Keys accessed frequently migrate toward the root so repeated access to a
//! hot key costs O(1) amortised time — the primary practical advantage over
//! other balanced BSTs on temporally-local workloads.
//!
//! # Implementation variant
//! This module uses a **recursive bottom-up splay** (zig / zig-zig / zig-zag).
//! The splay function walks down the tree taking ownership of each node via
//! `Option<Box<Node>>::take()`, then rebuilds the rotated tree on the way
//! back up the recursion.  No parent pointers, no `unsafe`, no `RefCell`.
//!
//! # Preconditions
//! None.  Set semantics: duplicate keys are rejected; `insert` returns
//! `false` for a duplicate.

/// Internal singly-owned tree node.
struct Node<K> {
    key: K,
    left: Option<Box<Self>>,
    right: Option<Box<Self>>,
}

impl<K> Node<K> {
    #[allow(clippy::unnecessary_box_returns)]
    fn leaf(key: K) -> Box<Self> {
        Box::new(Self {
            key,
            left: None,
            right: None,
        })
    }
}

// ---------------------------------------------------------------------------
// Rotation helpers (take-ownership style, no unsafe)
// ---------------------------------------------------------------------------

/// Single right rotation:
///
/// ```text
///     y           x
///    / \         / \
///   x   C  ->  A   y
///  / \             / \
/// A   B           B   C
/// ```
///
/// Consumes `y` (which must have a left child) and returns the new root `x`.
#[allow(clippy::unnecessary_box_returns)]
fn rotate_right<K>(mut y: Box<Node<K>>) -> Box<Node<K>> {
    let mut x = y.left.take().expect("rotate_right requires a left child");
    y.left = x.right.take();
    x.right = Some(y);
    x
}

/// Single left rotation (mirror of `rotate_right`).
#[allow(clippy::unnecessary_box_returns)]
fn rotate_left<K>(mut x: Box<Node<K>>) -> Box<Node<K>> {
    let mut y = x.right.take().expect("rotate_left requires a right child");
    x.right = y.left.take();
    y.left = Some(x);
    y
}

// ---------------------------------------------------------------------------
// Top-level splay
// ---------------------------------------------------------------------------
//
// `splay(node, key)` takes ownership of the sub-tree rooted at `node` and
// returns a new sub-tree root such that:
//   - If `key` is present, the root's key equals `key`.
//   - If `key` is absent, the root is the in-order predecessor or successor
//     of `key` (whichever was last visited).
//
// The splay uses three cases per level:
//   Zig       – target is a child of the current root; single rotation.
//   Zig-zig   – target and its parent are both left (or both right) children;
//               rotate at grandparent first, then at parent.
//   Zig-zag   – target is a left child of a right child (or vice-versa);
//               two rotations at the parent level.
//
// Because we recurse, "zig" is the base case that fires when we reach depth 1
// (or when we overshoot on a miss).

#[allow(clippy::unnecessary_box_returns)]
fn splay<K: Ord>(mut root: Box<Node<K>>, key: &K) -> Box<Node<K>> {
    match key.cmp(&root.key) {
        std::cmp::Ordering::Equal => root, // already at root

        std::cmp::Ordering::Less => {
            // Key is in the left subtree (or absent).
            let Some(left) = root.left.take() else {
                return root; // miss; root is closest successor
            };

            match key.cmp(&left.key) {
                std::cmp::Ordering::Equal => {
                    // Zig: target is root's left child — single right rotation.
                    root.left = Some(left);
                    rotate_right(root)
                }

                std::cmp::Ordering::Less => {
                    // Zig-zig (left-left): splay in left.left, then two right
                    // rotations.
                    let mut l = left;
                    let Some(ll) = l.left.take() else {
                        // No left-left; l is the closest node.
                        root.left = Some(l);
                        return rotate_right(root);
                    };
                    // Splay in the left-left subtree.
                    l.left = Some(splay(ll, key));
                    // Rotate at grandparent (root), then at parent (l).
                    root.left = Some(l);
                    let root = rotate_right(root); // l becomes root
                    rotate_right(root) // l.left becomes root
                }

                std::cmp::Ordering::Greater => {
                    // Zig-zag (left-right): splay in left.right, then left
                    // rotation at l, then right rotation at root.
                    let mut l = left;
                    let Some(lr) = l.right.take() else {
                        // No left-right; l is the closest node.
                        root.left = Some(l);
                        return rotate_right(root);
                    };
                    l.right = Some(splay(lr, key));
                    // Rotate left at l (so l.right rises), then right at root.
                    let new_left = rotate_left(l);
                    root.left = Some(new_left);
                    rotate_right(root)
                }
            }
        }

        std::cmp::Ordering::Greater => {
            // Mirror: key is in the right subtree.
            let Some(right) = root.right.take() else {
                return root;
            };

            match key.cmp(&right.key) {
                std::cmp::Ordering::Equal => {
                    root.right = Some(right);
                    rotate_left(root)
                }

                std::cmp::Ordering::Greater => {
                    // Zig-zig (right-right).
                    let mut r = right;
                    let Some(rr) = r.right.take() else {
                        root.right = Some(r);
                        return rotate_left(root);
                    };
                    r.right = Some(splay(rr, key));
                    root.right = Some(r);
                    let root = rotate_left(root);
                    rotate_left(root)
                }

                std::cmp::Ordering::Less => {
                    // Zig-zag (right-left).
                    let mut r = right;
                    let Some(rl) = r.left.take() else {
                        root.right = Some(r);
                        return rotate_left(root);
                    };
                    r.left = Some(splay(rl, key));
                    let new_right = rotate_right(r);
                    root.right = Some(new_right);
                    rotate_left(root)
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Public struct
// ---------------------------------------------------------------------------

/// A self-adjusting binary search tree (splay tree, Sleator & Tarjan 1985).
///
/// Every lookup, insertion, or removal splays a node to the root.  Because
/// splay mutates the internal structure, [`contains`] takes `&mut self`.
///
/// Set semantics: duplicate keys are rejected; `insert` returns `false` for
/// a key already present.
///
/// [`contains`]: SplayTree::contains
pub struct SplayTree<K: Ord> {
    root: Option<Box<Node<K>>>,
    len: usize,
}

impl<K: Ord> Default for SplayTree<K> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: Ord> SplayTree<K> {
    /// Creates an empty splay tree.
    #[must_use]
    pub const fn new() -> Self {
        Self { root: None, len: 0 }
    }

    /// Returns the number of elements in the tree.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the tree contains no elements.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Inserts `key` into the tree.
    ///
    /// Returns `true` if the key was newly inserted, `false` if it was
    /// already present (duplicates are rejected).
    ///
    /// After a successful insert the new key is the root.  After a rejected
    /// duplicate the duplicate is the root.
    pub fn insert(&mut self, key: K) -> bool {
        let root = match self.root.take() {
            None => {
                self.root = Some(Node::leaf(key));
                self.len += 1;
                return true;
            }
            Some(r) => splay(r, &key),
        };

        match key.cmp(&root.key) {
            std::cmp::Ordering::Equal => {
                // Duplicate; restore root.
                self.root = Some(root);
                false
            }

            std::cmp::Ordering::Less => {
                // New root: key < splayed root.
                // new_node.left  = root.left  (keys < key come from root's left)
                // new_node.right = root        (root has all keys >= root.key > key)
                let mut new_node = Node::leaf(key);
                let mut old_root = root;
                new_node.left = old_root.left.take();
                new_node.right = Some(old_root);
                self.root = Some(new_node);
                self.len += 1;
                true
            }

            std::cmp::Ordering::Greater => {
                // Mirror.
                let mut new_node = Node::leaf(key);
                let mut old_root = root;
                new_node.right = old_root.right.take();
                new_node.left = Some(old_root);
                self.root = Some(new_node);
                self.len += 1;
                true
            }
        }
    }

    /// Returns `true` if `key` is present in the tree.
    ///
    /// Splays the found node (or the nearest key on a miss) to the root.
    pub fn contains(&mut self, key: &K) -> bool {
        let Some(r) = self.root.take() else {
            return false;
        };
        let root = splay(r, key);
        let found = &root.key == key;
        self.root = Some(root);
        found
    }

    /// Removes `key` from the tree.
    ///
    /// Returns `true` if the key was present, `false` otherwise.
    pub fn remove(&mut self, key: &K) -> bool {
        let Some(r) = self.root.take() else {
            return false;
        };
        let root = splay(r, key);

        if &root.key != key {
            // Key not found.
            self.root = Some(root);
            return false;
        }

        // Remove root: join left and right subtrees.
        self.len -= 1;
        self.root = Self::join(root.left, root.right);
        true
    }

    /// Merges two sub-trees where every key in `left` < every key in `right`.
    #[allow(clippy::unnecessary_box_returns)]
    fn join(left: Option<Box<Node<K>>>, right: Option<Box<Node<K>>>) -> Option<Box<Node<K>>> {
        match (left, right) {
            (None, r) => r,
            (l, None) => l,
            (Some(l), Some(r)) => {
                // Splay the maximum of the left sub-tree to the left root.
                // The minimum key of `r` is larger than every key in `l`, so
                // splaying `r.key` in `l` brings the maximum of `l` to the
                // root (since `r.key` > max(l), the splay ends at the
                // rightmost node of `l`).
                let mut new_left = splay(l, &r.key);
                // The splayed left root has no right child
                // (it is the maximum, so nothing is to its right).
                new_left.right = Some(r);
                Some(new_left)
            }
        }
    }

    /// Returns an iterator over the keys in ascending order.
    ///
    /// This traversal does **not** splay; it is a plain in-order walk.
    pub fn iter_inorder(&self) -> impl Iterator<Item = &K> {
        InorderIter::new(self.root.as_deref())
    }
}

// ---------------------------------------------------------------------------
// In-order iterator (non-mutating, explicit stack)
// ---------------------------------------------------------------------------

struct InorderIter<'a, K> {
    stack: Vec<&'a Node<K>>,
}

impl<'a, K: Ord> InorderIter<'a, K> {
    fn new(root: Option<&'a Node<K>>) -> Self {
        let mut it = Self { stack: Vec::new() };
        it.push_left(root);
        it
    }

    fn push_left(&mut self, mut node: Option<&'a Node<K>>) {
        while let Some(n) = node {
            self.stack.push(n);
            node = n.left.as_deref();
        }
    }
}

impl<'a, K: Ord> Iterator for InorderIter<'a, K> {
    type Item = &'a K;

    fn next(&mut self) -> Option<Self::Item> {
        let node = self.stack.pop()?;
        self.push_left(node.right.as_deref());
        Some(&node.key)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::SplayTree;
    use quickcheck::TestResult;
    use quickcheck_macros::quickcheck;
    use std::collections::BTreeSet;

    // -----------------------------------------------------------------------
    // Unit tests
    // -----------------------------------------------------------------------

    #[test]
    fn empty_tree_behaves_correctly() {
        let mut t: SplayTree<i32> = SplayTree::new();
        assert!(t.is_empty());
        assert_eq!(t.len(), 0);
        assert!(!t.contains(&0));
        assert!(!t.remove(&0));
        assert!(t.iter_inorder().next().is_none());
    }

    #[test]
    fn single_insert_and_contains_makes_root() {
        let mut t = SplayTree::new();
        assert!(t.insert(7_i32));
        assert_eq!(t.len(), 1);
        assert!(t.contains(&7));
        // After contains the accessed key must be the root.
        assert_eq!(t.root.as_ref().map(|r| r.key), Some(7));
    }

    #[test]
    fn duplicate_insert_returns_false() {
        let mut t = SplayTree::new();
        assert!(t.insert(5_i32));
        assert!(!t.insert(5_i32));
        assert_eq!(t.len(), 1);
    }

    #[test]
    fn inorder_iteration_is_sorted() {
        let mut t = SplayTree::new();
        for k in [3_i32, 1, 4, 1, 5, 9, 2, 6] {
            t.insert(k);
        }
        let result: Vec<i32> = t.iter_inorder().copied().collect();
        let expected: Vec<i32> = {
            let mut s: Vec<i32> = [3, 1, 4, 5, 9, 2, 6].into();
            s.sort_unstable();
            s.dedup();
            s
        };
        assert_eq!(result, expected);
    }

    #[test]
    fn insert_1_to_100_all_found_and_sorted() {
        let mut t = SplayTree::new();
        for i in 1..=100_i32 {
            assert!(t.insert(i));
        }
        for i in 1..=100_i32 {
            assert!(t.contains(&i));
        }
        let result: Vec<i32> = t.iter_inorder().copied().collect();
        let expected: Vec<i32> = (1..=100).collect();
        assert_eq!(result, expected);
    }

    #[test]
    fn splay_root_effect_after_contains_50() {
        let mut t = SplayTree::new();
        for i in 1..=100_i32 {
            t.insert(i);
        }
        assert!(t.contains(&50));
        assert_eq!(t.root.as_ref().map(|r| r.key), Some(50));
    }

    #[test]
    fn removal_preserves_sortedness() {
        let mut t = SplayTree::new();
        for i in 1..=20_i32 {
            t.insert(i);
        }
        // Remove all odd keys.
        for i in (1..=20_i32).step_by(2) {
            assert!(t.remove(&i), "failed to remove {i}");
        }
        let result: Vec<i32> = t.iter_inorder().copied().collect();
        let expected: Vec<i32> = (2..=20).step_by(2).collect();
        assert_eq!(result, expected);
        assert_eq!(t.len(), 10);
    }

    #[test]
    fn remove_absent_key_returns_false() {
        let mut t = SplayTree::new();
        t.insert(1_i32);
        t.insert(2_i32);
        assert!(!t.remove(&99));
        assert_eq!(t.len(), 2);
    }

    // -----------------------------------------------------------------------
    // Property test: model-checked against BTreeSet<i32>
    // -----------------------------------------------------------------------

    #[derive(Clone, Debug)]
    enum Op {
        Insert(i32),
        Remove(i32),
        Contains(i32),
    }

    impl quickcheck::Arbitrary for Op {
        fn arbitrary(g: &mut quickcheck::Gen) -> Self {
            // Restrict key range to increase collision rate.
            let key = i32::arbitrary(g) % 50;
            match u8::arbitrary(g) % 3 {
                0 => Self::Insert(key),
                1 => Self::Remove(key),
                _ => Self::Contains(key),
            }
        }
    }

    #[quickcheck]
    #[allow(clippy::needless_pass_by_value)]
    fn prop_matches_btreeset(ops: Vec<Op>) -> TestResult {
        if ops.len() > 200 {
            return TestResult::discard();
        }
        let mut splay: SplayTree<i32> = SplayTree::new();
        let mut model: BTreeSet<i32> = BTreeSet::new();

        for op in &ops {
            match *op {
                Op::Insert(k) => {
                    let sr = splay.insert(k);
                    let mr = model.insert(k);
                    if sr != mr {
                        return TestResult::failed();
                    }
                }
                Op::Remove(k) => {
                    let sr = splay.remove(&k);
                    let mr = model.remove(&k);
                    if sr != mr {
                        return TestResult::failed();
                    }
                }
                Op::Contains(k) => {
                    let sc = splay.contains(&k);
                    let mc = model.contains(&k);
                    if sc != mc {
                        return TestResult::failed();
                    }
                }
            }

            // After every op, sorted iteration must match BTreeSet.
            let splay_keys: Vec<i32> = splay.iter_inorder().copied().collect();
            let model_keys: Vec<i32> = model.iter().copied().collect();
            if splay_keys != model_keys {
                return TestResult::failed();
            }
            if splay.len() != model.len() {
                return TestResult::failed();
            }
        }

        TestResult::passed()
    }
}
