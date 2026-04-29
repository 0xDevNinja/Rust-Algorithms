//! B-tree (Bayer & `McCreight` 1972) — ordered set with O(log n) insert,
//! lookup, and delete.
//!
//! # Algorithm
//! This implementation follows the CLRS Chapter 18 presentation of B-trees.
//! The *minimum degree* `t` (runtime parameter, default 3) governs node
//! capacity: every non-root node holds between `t-1` and `2t-1` keys, and
//! internal nodes have exactly `keys.len() + 1` children.
//!
//! - **Insert**: split-on-the-way-down (proactive). A full root is split
//!   immediately before descent; any full child is split before recursing
//!   into it, so the descent never needs to backtrack.
//! - **Delete**: borrow-or-merge-on-the-way-down (proactive). Before
//!   descending into a child, ensure it has at least `t` keys by borrowing
//!   from a sibling or merging with a sibling plus separator. The three
//!   CLRS cases: leaf delete, internal-node delete via predecessor/successor
//!   swap, and key-absent descent through children.
//!
//! # Complexity
//! | Operation  | Time                | Space      |
//! |------------|---------------------|------------|
//! | `insert`   | O(t · `log_t` n)   | O(1) extra |
//! | `contains` | O(t · `log_t` n)   | O(1) extra |
//! | `remove`   | O(t · `log_t` n)   | O(1) extra |
//! | height     | ≤ `log_t(n)`        | O(n) total |
//!
//! The high fan-out (up to `2t-1` keys per node) means fewer levels than a
//! binary BST, making B-trees well-suited to disk-resident data where each
//! node maps to a page.
//!
//! # Preconditions
//! - `t >= 2` (asserted in `with_min_degree`).
//! - Duplicate keys are rejected: `insert` returns `false` on a duplicate.

/// Internal B-tree node.
// The issue spec mandates Vec<Box<Node<K>>> for child ownership.
// clippy::use_self does not apply to struct definitions — the field type must
// name the struct explicitly in the recursive position.
#[allow(clippy::vec_box, clippy::use_self)]
struct Node<K> {
    keys: Vec<K>,
    children: Vec<Box<Node<K>>>,
    leaf: bool,
}

#[allow(clippy::vec_box)]
impl<K: Ord + Clone> Node<K> {
    const fn new_leaf() -> Self {
        Self {
            keys: Vec::new(),
            children: Vec::new(),
            leaf: true,
        }
    }

    const fn new_internal() -> Self {
        Self {
            keys: Vec::new(),
            children: Vec::new(),
            leaf: false,
        }
    }

    const fn is_full(&self, t: usize) -> bool {
        self.keys.len() == 2 * t - 1
    }

    /// Binary search inside the node. Returns `Ok(i)` if `keys[i] == key`,
    /// or `Err(i)` where `i` is the insertion point (first index `> key`).
    fn find_pos(&self, key: &K) -> Result<usize, usize> {
        self.keys.binary_search(key)
    }

    // ---- search ----------------------------------------------------------

    fn contains(&self, key: &K) -> bool {
        match self.find_pos(key) {
            Ok(_) => true,
            Err(i) => !self.leaf && self.children[i].contains(key),
        }
    }

    // ---- split -----------------------------------------------------------

    /// Split the full `i`-th child of `self`, promoting its median key into
    /// `self`. `self` must not be full.
    fn split_child(&mut self, i: usize, t: usize) {
        // After split_off(t) the child holds keys[0..t-1] (t keys).
        // The median is the last one (index t-1).
        let right = {
            let child = &mut self.children[i];
            let mut r = if child.leaf {
                Self::new_leaf()
            } else {
                Self::new_internal()
            };
            r.keys = child.keys.split_off(t); // upper half: [t..2t-1]
            if !child.leaf {
                r.children = child.children.split_off(t); // upper t children
            }
            r
        };

        // The median is now `child.keys.last()` (index t-1).
        let median = self.children[i]
            .keys
            .pop()
            .expect("child has t keys after split");

        self.keys.insert(i, median);
        self.children.insert(i + 1, Box::new(right));
    }

    // ---- insert ----------------------------------------------------------

    /// Insert into a non-full node, splitting full children proactively.
    fn insert_nonfull(&mut self, key: K, t: usize) -> bool {
        match self.find_pos(&key) {
            Ok(_) => false, // duplicate
            Err(mut i) => {
                if self.leaf {
                    self.keys.insert(i, key);
                    true
                } else {
                    if self.children[i].is_full(t) {
                        self.split_child(i, t);
                        // The median moved up; decide which half to descend into.
                        match key.cmp(&self.keys[i]) {
                            std::cmp::Ordering::Equal => return false,
                            std::cmp::Ordering::Greater => i += 1,
                            std::cmp::Ordering::Less => {}
                        }
                    }
                    self.children[i].insert_nonfull(key, t)
                }
            }
        }
    }

    // ---- predecessor / successor ----------------------------------------

    /// Largest key in the subtree rooted at `self`.
    fn max_key(&self) -> &K {
        if self.leaf {
            self.keys.last().expect("non-empty node")
        } else {
            self.children
                .last()
                .expect("internal node has children")
                .max_key()
        }
    }

    /// Smallest key in the subtree rooted at `self`.
    fn min_key(&self) -> &K {
        if self.leaf {
            self.keys.first().expect("non-empty node")
        } else {
            self.children
                .first()
                .expect("internal node has children")
                .min_key()
        }
    }

    // ---- delete helpers -------------------------------------------------

    /// Ensure `children[i]` has at least `t` keys by borrowing from a sibling
    /// or merging. Returns the (possibly adjusted) child index to descend into.
    fn fix_child(&mut self, i: usize, t: usize) -> usize {
        if self.children[i].keys.len() >= t {
            return i;
        }

        let has_left = i > 0;
        let has_right = i + 1 < self.children.len();
        let left_rich = has_left && self.children[i - 1].keys.len() >= t;
        let right_rich = has_right && self.children[i + 1].keys.len() >= t;

        if left_rich {
            // Rotate right: left sibling's last key → self.keys[i-1]; old
            // self.keys[i-1] prepended to children[i].
            let sep = self.keys[i - 1].clone();
            let (borrow_key, borrow_child) = {
                let left = &mut self.children[i - 1];
                let k = left.keys.pop().expect("left sibling non-empty");
                let c = if left.leaf {
                    None
                } else {
                    Some(left.children.pop().expect("left sibling has child"))
                };
                (k, c)
            };
            self.keys[i - 1] = borrow_key;
            let child = &mut self.children[i];
            child.keys.insert(0, sep);
            if let Some(c) = borrow_child {
                child.children.insert(0, c);
            }
            i
        } else if right_rich {
            // Rotate left: right sibling's first key → self.keys[i]; old
            // self.keys[i] appended to children[i].
            let sep = self.keys[i].clone();
            let (borrow_key, borrow_child) = {
                let right = &mut self.children[i + 1];
                let k = right.keys.remove(0);
                let c = if right.leaf {
                    None
                } else {
                    Some(right.children.remove(0))
                };
                (k, c)
            };
            self.keys[i] = borrow_key;
            let child = &mut self.children[i];
            child.keys.push(sep);
            if let Some(c) = borrow_child {
                child.children.push(c);
            }
            i
        } else if has_right {
            // Merge children[i] and children[i+1] around self.keys[i].
            self.merge_children(i);
            i
        } else {
            // Merge children[i-1] and children[i] around self.keys[i-1].
            self.merge_children(i - 1);
            i - 1
        }
    }

    /// Merge `children[i+1]` into `children[i]` using `keys[i]` as separator;
    /// remove `keys[i]` and `children[i+1]` from `self`.
    fn merge_children(&mut self, i: usize) {
        let sep = self.keys.remove(i);
        let mut right = *self.children.remove(i + 1);
        let left = &mut self.children[i];
        left.keys.push(sep);
        left.keys.append(&mut right.keys);
        left.children.append(&mut right.children);
    }

    /// Delete `key` from the subtree rooted at `self` (CLRS Ch. 18.3).
    fn delete(&mut self, key: &K, t: usize) -> bool {
        match self.find_pos(key) {
            Ok(i) => {
                // Key is in this node.
                if self.leaf {
                    // Case 1: direct removal from leaf.
                    self.keys.remove(i);
                    true
                } else if self.children[i].keys.len() >= t {
                    // Case 2a: left child (children[i]) has >= t keys.
                    // Replace keys[i] with the in-order predecessor, then
                    // delete the predecessor from the left subtree. The
                    // recursive call handles any underflows on descent.
                    let pred = self.children[i].max_key().clone();
                    self.keys[i] = pred.clone();
                    self.children[i].delete(&pred, t)
                } else if self.children[i + 1].keys.len() >= t {
                    // Case 2b: right child has >= t keys; use successor.
                    let succ = self.children[i + 1].min_key().clone();
                    self.keys[i] = succ.clone();
                    self.children[i + 1].delete(&succ, t)
                } else {
                    // Case 2c: both adjacent children have t-1 keys.
                    // Merge them; the original key descends into the merged node.
                    self.merge_children(i);
                    self.children[i].delete(key, t)
                }
            }
            Err(i) => {
                // Key not in this node.
                if self.leaf {
                    return false;
                }
                // Case 3: fix children[i] to have >= t keys, then recurse.
                let ci = self.fix_child(i, t);
                self.children[ci].delete(key, t)
            }
        }
    }
}

// ---- In-order iterator --------------------------------------------------

/// State machine per stack frame for [`BTreeIter`].
enum FrameState {
    /// Yield `keys[idx]` next; for internal nodes also descend into
    /// `children[idx + 1]` before advancing to the next key.
    Key(usize),
}

/// Iterator that yields all keys of a B-tree in ascending order.
///
/// Produced by [`BTree::iter_inorder`].
#[allow(clippy::vec_box)]
pub struct BTreeIter<'a, K> {
    stack: Vec<(&'a Node<K>, FrameState)>,
}

#[allow(clippy::vec_box)]
impl<'a, K: Ord + Clone> BTreeIter<'a, K> {
    fn new(root: &'a Node<K>) -> Self {
        let mut iter = Self { stack: Vec::new() };
        iter.descend_left(root);
        iter
    }

    /// Push `node` and its entire left spine onto the stack so the first
    /// call to `next()` yields the globally smallest key.
    fn descend_left(&mut self, mut node: &'a Node<K>) {
        loop {
            if node.keys.is_empty() {
                break;
            }
            self.stack.push((node, FrameState::Key(0)));
            if node.leaf {
                break;
            }
            node = &node.children[0];
        }
    }
}

#[allow(clippy::vec_box)]
impl<'a, K: Ord + Clone> Iterator for BTreeIter<'a, K> {
    type Item = &'a K;

    fn next(&mut self) -> Option<Self::Item> {
        // Peek at the top frame, extract what we need, then drop the borrow
        // before mutating the stack.
        let (is_leaf, key_idx, key_count, child_count) = {
            let (node, state) = self.stack.last()?;
            let FrameState::Key(i) = *state;
            (node.leaf, i, node.keys.len(), node.children.len())
        };

        // Retrieve the key pointer and (for internal nodes) the right child
        // pointer while the immutable borrow is still alive.
        let (key, right_child): (&'a K, Option<&'a Node<K>>) = {
            let (node, _) = self.stack.last().expect("checked above");
            let k = &node.keys[key_idx];
            let rc = if !is_leaf && key_idx + 1 < child_count {
                Some(&*node.children[key_idx + 1])
            } else {
                None
            };
            (k, rc)
        };

        // Now mutate: advance or pop the top frame.
        if key_idx + 1 < key_count {
            let (_, state) = self.stack.last_mut().expect("checked above");
            *state = FrameState::Key(key_idx + 1);
        } else {
            self.stack.pop();
        }

        // For internal nodes, push the left spine of the right child so
        // the next call returns the in-order successor key.
        if let Some(rc) = right_child {
            self.descend_left(rc);
        }

        Some(key)
    }
}

// ---- Public API ---------------------------------------------------------

/// B-tree ordered set (Bayer & `McCreight` 1972).
///
/// Set semantics: duplicate keys are rejected. All operations run in
/// O(t · `log_t` n) time. The height of the tree is at most `log_t(n)`.
///
/// The *minimum degree* `t` (default 3) controls the node capacity:
/// every non-root node holds between `t-1` and `2t-1` keys. Higher `t`
/// reduces tree height at the cost of more key comparisons per node —
/// ideal for disk-resident data with high fan-out requirements.
pub struct BTree<K: Ord + Clone> {
    root: Box<Node<K>>,
    t: usize,
    len: usize,
}

#[allow(clippy::vec_box)]
impl<K: Ord + Clone> BTree<K> {
    /// Creates an empty B-tree with minimum degree `t`.
    ///
    /// # Panics
    /// Panics if `t < 2`.
    pub fn with_min_degree(t: usize) -> Self {
        assert!(t >= 2, "minimum degree t must be >= 2");
        Self {
            root: Box::new(Node::new_leaf()),
            t,
            len: 0,
        }
    }

    /// Creates an empty B-tree with the default minimum degree `t = 3`.
    pub fn new() -> Self {
        Self::with_min_degree(3)
    }

    /// Number of keys stored in the tree.
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the tree contains no keys.
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns `true` if `key` is present in the tree.
    pub fn contains(&self, key: &K) -> bool {
        self.root.contains(key)
    }

    /// Inserts `key` into the tree.
    ///
    /// Returns `true` if the key was newly inserted, `false` if it was already
    /// present (the tree is unchanged in that case).
    pub fn insert(&mut self, key: K) -> bool {
        let t = self.t;
        if self.root.is_full(t) {
            // Split the root: create a new internal root whose only child is
            // the old root, then split that child.
            let old_root = std::mem::replace(&mut self.root, Box::new(Node::new_internal()));
            self.root.children.push(old_root);
            self.root.split_child(0, t);
        }
        let inserted = self.root.insert_nonfull(key, t);
        if inserted {
            self.len += 1;
        }
        inserted
    }

    /// Removes `key` from the tree.
    ///
    /// Returns `true` if the key was present and removed, `false` if absent.
    pub fn remove(&mut self, key: &K) -> bool {
        let t = self.t;
        let removed = self.root.delete(key, t);
        if removed {
            self.len -= 1;
        }
        // Shrink tree height if root became empty due to a merge.
        // This can happen even on a miss: fix_child may merge siblings while
        // descending, pulling the root's only key down and leaving the root
        // with no keys but one remaining child.
        if self.root.keys.is_empty() && !self.root.children.is_empty() {
            self.root = self.root.children.remove(0);
        }
        removed
    }

    /// Returns an iterator that yields all keys in ascending order.
    pub fn iter_inorder(&self) -> BTreeIter<'_, K> {
        BTreeIter::new(&self.root)
    }

    /// Verifies all B-tree structural invariants.
    ///
    /// Returns `Ok(())` if the tree is well-formed, `Err(msg)` otherwise.
    /// Intended for testing; not part of the normal API.
    pub fn verify_btree_invariants(&self) -> Result<(), &'static str> {
        self.check_node(&self.root, None, None, true)?;
        Ok(())
    }

    fn check_node(
        &self,
        node: &Node<K>,
        lower: Option<&K>,
        upper: Option<&K>,
        is_root: bool,
    ) -> Result<usize, &'static str> {
        let t = self.t;

        // Key-count bounds.
        if is_root {
            if node.keys.len() > 2 * t - 1 {
                return Err("root has too many keys");
            }
        } else {
            if node.keys.len() < t - 1 {
                return Err("non-root node has too few keys");
            }
            if node.keys.len() > 2 * t - 1 {
                return Err("non-root node has too many keys");
            }
        }

        // Keys must be strictly sorted within the node.
        for w in node.keys.windows(2) {
            if w[0] >= w[1] {
                return Err("keys within node are not strictly sorted");
            }
        }

        // Keys must respect the inherited BST bounds.
        if let Some(lo) = lower {
            if node.keys.first().is_some_and(|k| k <= lo) {
                return Err("key violates lower bound");
            }
        }
        if let Some(hi) = upper {
            if node.keys.last().is_some_and(|k| k >= hi) {
                return Err("key violates upper bound");
            }
        }

        if node.leaf {
            if !node.children.is_empty() {
                return Err("leaf node has children");
            }
            return Ok(0);
        }

        // Internal node: child count must be exactly keys.len() + 1.
        if node.children.len() != node.keys.len() + 1 {
            return Err("internal node child count mismatch");
        }

        // Recurse with tightened bounds; all subtrees must have equal leaf depth.
        let mut leaf_depth: Option<usize> = None;
        for (ci, child) in node.children.iter().enumerate() {
            let lo = if ci == 0 {
                lower
            } else {
                Some(&node.keys[ci - 1])
            };
            let hi = if ci == node.keys.len() {
                upper
            } else {
                Some(&node.keys[ci])
            };
            let d = self.check_node(child, lo, hi, false)?;
            match leaf_depth {
                None => leaf_depth = Some(d),
                Some(prev) if prev != d => return Err("leaves are at different depths"),
                _ => {}
            }
        }
        Ok(leaf_depth.unwrap_or(0) + 1)
    }
}

impl<K: Ord + Clone> Default for BTree<K> {
    fn default() -> Self {
        Self::new()
    }
}

// ---- Tests --------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::BTree;
    use quickcheck_macros::quickcheck;
    use std::collections::BTreeSet;

    // ---- unit tests -------------------------------------------------------

    #[test]
    fn empty_tree() {
        let bt: BTree<i32> = BTree::new();
        assert!(bt.is_empty());
        assert_eq!(bt.len(), 0);
        assert!(!bt.contains(&0));
        assert!(!bt.contains(&99));
        assert_eq!(bt.iter_inorder().count(), 0);
        bt.verify_btree_invariants().unwrap();
    }

    #[test]
    fn remove_absent_returns_false() {
        let mut bt: BTree<i32> = BTree::new();
        assert!(!bt.remove(&42));
        bt.verify_btree_invariants().unwrap();
    }

    #[test]
    fn single_insert_and_contains() {
        let mut bt = BTree::new();
        assert!(bt.insert(7));
        assert!(bt.contains(&7));
        assert!(!bt.contains(&0));
        assert_eq!(bt.len(), 1);
        bt.verify_btree_invariants().unwrap();
    }

    #[test]
    fn duplicate_insert_returns_false() {
        let mut bt = BTree::new();
        assert!(bt.insert(5));
        assert!(!bt.insert(5));
        assert_eq!(bt.len(), 1);
        bt.verify_btree_invariants().unwrap();
    }

    #[test]
    fn sequential_insert_1_to_200() {
        let mut bt = BTree::new();
        for i in 1..=200_i32 {
            assert!(bt.insert(i), "insert({i}) should succeed");
        }
        assert_eq!(bt.len(), 200);
        bt.verify_btree_invariants().unwrap();

        for i in 1..=200_i32 {
            assert!(bt.contains(&i), "should contain {i}");
        }
        let keys: Vec<i32> = bt.iter_inorder().copied().collect();
        assert_eq!(keys, (1..=200).collect::<Vec<_>>());
    }

    #[test]
    fn reverse_insert_200_to_1() {
        let mut bt = BTree::new();
        for i in (1..=200_i32).rev() {
            assert!(bt.insert(i));
        }
        assert_eq!(bt.len(), 200);
        bt.verify_btree_invariants().unwrap();
        let keys: Vec<i32> = bt.iter_inorder().copied().collect();
        assert_eq!(keys, (1..=200).collect::<Vec<_>>());
    }

    #[test]
    fn remove_all_sequential() {
        let mut bt = BTree::new();
        for i in 1..=100_i32 {
            bt.insert(i);
        }
        for i in 1..=100_i32 {
            assert!(bt.remove(&i), "remove({i}) should succeed");
            bt.verify_btree_invariants()
                .unwrap_or_else(|e| panic!("invariant violated after remove({i}): {e}"));
        }
        assert!(bt.is_empty());
    }

    #[test]
    fn insert_remove_random_with_invariant_check() {
        // Deterministic pseudo-random ops via a simple LCG.
        let mut state: u64 = 0xDEAD_BEEF_CAFE_1234;
        let lcg = |s: &mut u64| -> i32 {
            *s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((*s >> 33) as i32).abs()
        };

        let mut bt: BTree<i32> = BTree::new();
        let mut reference: BTreeSet<i32> = BTreeSet::new();

        for _ in 0..300 {
            let key = lcg(&mut state) % 50;
            if reference.contains(&key) {
                assert_eq!(
                    reference.remove(&key),
                    bt.remove(&key),
                    "remove({key}) mismatch"
                );
            } else {
                assert_eq!(
                    reference.insert(key),
                    bt.insert(key),
                    "insert({key}) mismatch"
                );
            }
            bt.verify_btree_invariants()
                .unwrap_or_else(|e| panic!("invariant violated after op on {key}: {e}"));
            assert_eq!(bt.len(), reference.len());
        }
    }

    #[test]
    fn varying_t_values() {
        for t in [2_usize, 3, 5, 10] {
            let mut bt: BTree<i32> = BTree::with_min_degree(t);
            for i in 0..100_i32 {
                bt.insert(i);
            }
            bt.verify_btree_invariants()
                .unwrap_or_else(|e| panic!("invariant violated for t={t}: {e}"));
            for i in (0..100_i32).rev() {
                bt.remove(&i);
                bt.verify_btree_invariants()
                    .unwrap_or_else(|e| panic!("t={t} invariant after remove({i}): {e}"));
            }
            assert!(bt.is_empty());
        }
    }

    // ---- quickcheck property test ----------------------------------------

    #[derive(Clone, Debug)]
    enum Op {
        Insert(i32),
        Remove(i32),
        Contains(i32),
    }

    impl quickcheck::Arbitrary for Op {
        fn arbitrary(g: &mut quickcheck::Gen) -> Self {
            let key = i32::arbitrary(g) % 50;
            match u8::arbitrary(g) % 3 {
                0 => Self::Insert(key),
                1 => Self::Remove(key),
                _ => Self::Contains(key),
            }
        }
    }

    #[allow(clippy::needless_pass_by_value)]
    #[quickcheck]
    fn model_check_against_btreeset(ops: Vec<Op>) -> bool {
        let mut bt: BTree<i32> = BTree::new(); // t = 3
        let mut model: BTreeSet<i32> = BTreeSet::new();

        for op in ops {
            match op {
                Op::Insert(k) => {
                    if bt.insert(k) != model.insert(k) {
                        return false;
                    }
                }
                Op::Remove(k) => {
                    if bt.remove(&k) != model.remove(&k) {
                        return false;
                    }
                }
                Op::Contains(k) => {
                    if bt.contains(&k) != model.contains(&k) {
                        return false;
                    }
                }
            }

            if bt.verify_btree_invariants().is_err() {
                return false;
            }
            if bt.len() != model.len() {
                return false;
            }
            let bt_keys: Vec<i32> = bt.iter_inorder().copied().collect();
            let model_keys: Vec<i32> = model.iter().copied().collect();
            if bt_keys != model_keys {
                return false;
            }
        }
        true
    }
}
