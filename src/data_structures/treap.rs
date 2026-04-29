//! Treap — Randomized Binary Search Tree (Aragon & Seidel, 1989).
//!
//! A treap combines a **BST** ordered by key with a **max-heap** ordered by a
//! randomly assigned priority. Because priorities are chosen uniformly at
//! random, the expected tree height is O(log n), giving expected O(log n) time
//! for all single-key operations.
//!
//! # Invariants
//! - BST: for every node, all keys in the left subtree are `< key` and all
//!   keys in the right subtree are `> key`.
//! - Max-heap on priority: a node's priority is ≥ both children's priorities.
//!
//! # Complexity
//! | Operation | Expected | Worst case |
//! |-----------|----------|------------|
//! | `insert`  | O(log n) | O(n)       |
//! | `get`     | O(log n) | O(n)       |
//! | `remove`  | O(log n) | O(n)       |
//! | `iter`    | O(n)     | O(n)       |
//!
//! Space: O(n) expected.
//!
//! # Preconditions
//! Keys must implement [`Ord`]. There are no constraints on values.
//!
//! # Duplicate keys
//! Inserting a key that already exists **updates** the value in place and
//! returns the previous value as `Some(old)`.

use std::cmp::Ordering::{Equal, Greater, Less};

/// Internal tree node.
struct Node<K, V> {
    key: K,
    value: V,
    priority: u64,
    left: Option<Box<Self>>,
    right: Option<Box<Self>>,
}

impl<K, V> Node<K, V> {
    const fn new(key: K, value: V, priority: u64) -> Self {
        Self {
            key,
            value,
            priority,
            left: None,
            right: None,
        }
    }
}

/// A general-purpose ordered map backed by a randomized BST (treap).
///
/// Keys are unique; inserting a duplicate key updates the value.
///
/// See the [module documentation](self) for complexity guarantees.
pub struct Treap<K: Ord, V> {
    root: Option<Box<Node<K, V>>>,
    len: usize,
    /// Linear-congruential generator state for priority generation.
    rng: u64,
}

// ── LCG constants (same as Numerical Recipes) ──────────────────────────────
const LCG_A: u64 = 6_364_136_223_846_793_005;
const LCG_C: u64 = 1_442_695_040_888_963_407;

impl<K: Ord, V> Treap<K, V> {
    /// Creates an empty treap with a default seed.
    #[must_use]
    pub const fn new() -> Self {
        Self::with_seed(0xDEAD_BEEF_CAFE_BABE)
    }

    /// Creates an empty treap whose random priorities are seeded with `seed`.
    ///
    /// Two treaps built from the same sequence of operations using the same
    /// seed will have identical structure — useful for deterministic tests.
    #[must_use]
    pub const fn with_seed(seed: u64) -> Self {
        Self {
            root: None,
            len: 0,
            rng: seed,
        }
    }

    /// Advances the LCG and returns the next pseudo-random `u64`.
    const fn next_priority(&mut self) -> u64 {
        self.rng = self.rng.wrapping_mul(LCG_A).wrapping_add(LCG_C);
        self.rng
    }

    /// Inserts `key -> value` into the treap.
    ///
    /// Returns `Some(old_value)` if the key was already present (the value is
    /// replaced), or `None` if this is a fresh insertion.
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        let priority = self.next_priority();
        let mut old_value: Option<V> = None;
        self.root = Some(Self::insert_node(
            self.root.take(),
            key,
            value,
            priority,
            &mut old_value,
        ));
        if old_value.is_none() {
            self.len += 1;
        }
        old_value
    }

    /// Recursive BST insert followed by rotate-up to restore the heap property.
    // Returning Box<Node> is intentional: the caller always stores the result
    // in an Option<Box<Node>> field, so unboxing here would just re-box it.
    #[allow(clippy::unnecessary_box_returns)]
    fn insert_node(
        node: Option<Box<Node<K, V>>>,
        key: K,
        value: V,
        priority: u64,
        old_value: &mut Option<V>,
    ) -> Box<Node<K, V>> {
        match node {
            None => Box::new(Node::new(key, value, priority)),
            Some(mut n) => {
                match key.cmp(&n.key) {
                    Equal => {
                        // Key exists: update in place, no structural change needed.
                        *old_value = Some(std::mem::replace(&mut n.value, value));
                        // Keep existing priority — no rotation required.
                        n
                    }
                    Less => {
                        n.left = Some(Self::insert_node(
                            n.left.take(),
                            key,
                            value,
                            priority,
                            old_value,
                        ));
                        // Restore max-heap on priority: if left child has higher
                        // priority, rotate right.
                        if n.left.as_ref().map_or(0, |c| c.priority) > n.priority {
                            n = Self::rotate_right(n);
                        }
                        n
                    }
                    Greater => {
                        n.right = Some(Self::insert_node(
                            n.right.take(),
                            key,
                            value,
                            priority,
                            old_value,
                        ));
                        // Restore max-heap: if right child has higher priority,
                        // rotate left.
                        if n.right.as_ref().map_or(0, |c| c.priority) > n.priority {
                            n = Self::rotate_left(n);
                        }
                        n
                    }
                }
            }
        }
    }

    /// Returns a reference to the value associated with `key`, or `None`.
    #[must_use]
    pub fn get(&self, key: &K) -> Option<&V> {
        let mut cur = self.root.as_deref();
        while let Some(n) = cur {
            match key.cmp(&n.key) {
                Equal => return Some(&n.value),
                Less => cur = n.left.as_deref(),
                Greater => cur = n.right.as_deref(),
            }
        }
        None
    }

    /// Returns `true` if the treap contains `key`.
    #[must_use]
    pub fn contains_key(&self, key: &K) -> bool {
        self.get(key).is_some()
    }

    /// Removes `key` from the treap and returns its value, or `None` if absent.
    ///
    /// The target node is rotated down (toward the child with higher priority)
    /// until it becomes a leaf, then detached.
    pub fn remove(&mut self, key: &K) -> Option<V> {
        let mut removed: Option<V> = None;
        self.root = Self::remove_node(self.root.take(), key, &mut removed);
        if removed.is_some() {
            self.len -= 1;
        }
        removed
    }

    /// Recursive helper: walks the BST path to `key`, rotates target down,
    /// then drops the leaf.
    fn remove_node(
        node: Option<Box<Node<K, V>>>,
        key: &K,
        removed: &mut Option<V>,
    ) -> Option<Box<Node<K, V>>> {
        let mut n = node?;
        match key.cmp(&n.key) {
            Less => {
                n.left = Self::remove_node(n.left.take(), key, removed);
                Some(n)
            }
            Greater => {
                n.right = Self::remove_node(n.right.take(), key, removed);
                Some(n)
            }
            Equal => {
                // Found the target. Rotate it down until it's a leaf, then drop.
                match (&n.left, &n.right) {
                    (None, None) => {
                        // Leaf — capture value and detach.
                        *removed = Some(n.value);
                        None
                    }
                    (Some(_), None) => {
                        // Only left child: rotate right, recurse.
                        let mut rotated = Self::rotate_right(n);
                        rotated.right = Self::remove_node(rotated.right.take(), key, removed);
                        Some(rotated)
                    }
                    (None, Some(_)) => {
                        // Only right child: rotate left, recurse.
                        let mut rotated = Self::rotate_left(n);
                        rotated.left = Self::remove_node(rotated.left.take(), key, removed);
                        Some(rotated)
                    }
                    (Some(l), Some(r)) => {
                        // Both children present: rotate toward the higher-priority child.
                        if l.priority >= r.priority {
                            let mut rotated = Self::rotate_right(n);
                            rotated.right = Self::remove_node(rotated.right.take(), key, removed);
                            Some(rotated)
                        } else {
                            let mut rotated = Self::rotate_left(n);
                            rotated.left = Self::remove_node(rotated.left.take(), key, removed);
                            Some(rotated)
                        }
                    }
                }
            }
        }
    }

    /// Returns the number of key-value pairs in the treap.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the treap contains no entries.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns an iterator over `(&key, &value)` pairs in ascending key order.
    ///
    /// The iterator performs an in-order traversal using an explicit stack,
    /// visiting each node exactly once.
    pub fn iter(&self) -> impl Iterator<Item = (&K, &V)> {
        InOrderIter::new(self.root.as_deref())
    }

    // ── Rotation helpers ────────────────────────────────────────────────────

    /// Right-rotation: `n` becomes the right child of its current left child.
    ///
    /// ```text
    ///     n              L
    ///    / \            / \
    ///   L   R   →    LL    n
    ///  / \                / \
    /// LL  LR            LR   R
    /// ```
    // Box is threaded through tree links; returning the inner type would require
    // re-boxing at every call site.
    #[allow(clippy::unnecessary_box_returns)]
    fn rotate_right(mut n: Box<Node<K, V>>) -> Box<Node<K, V>> {
        let mut l = n
            .left
            .take()
            .expect("rotate_right called without left child");
        n.left = l.right.take();
        l.right = Some(n);
        l
    }

    /// Left-rotation: `n` becomes the left child of its current right child.
    ///
    /// ```text
    ///   n                R
    ///  / \              / \
    /// L   R    →       n   RR
    ///    / \          / \
    ///   RL  RR       L   RL
    /// ```
    #[allow(clippy::unnecessary_box_returns)]
    fn rotate_left(mut n: Box<Node<K, V>>) -> Box<Node<K, V>> {
        let mut r = n
            .right
            .take()
            .expect("rotate_left called without right child");
        n.right = r.left.take();
        r.left = Some(n);
        r
    }
}

impl<K: Ord, V> Default for Treap<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

// ── In-order iterator ────────────────────────────────────────────────────────

/// Iterates over `(&K, &V)` pairs in ascending key order via an explicit stack.
struct InOrderIter<'a, K, V> {
    stack: Vec<&'a Node<K, V>>,
}

impl<'a, K, V> InOrderIter<'a, K, V> {
    fn new(root: Option<&'a Node<K, V>>) -> Self {
        let mut iter = Self { stack: Vec::new() };
        iter.push_left_spine(root);
        iter
    }

    /// Pushes all left-spine nodes starting at `node` onto the stack.
    fn push_left_spine(&mut self, mut node: Option<&'a Node<K, V>>) {
        while let Some(n) = node {
            self.stack.push(n);
            node = n.left.as_deref();
        }
    }
}

impl<'a, K, V> Iterator for InOrderIter<'a, K, V> {
    type Item = (&'a K, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        let n = self.stack.pop()?;
        // After visiting `n`, push the left spine of its right subtree.
        self.push_left_spine(n.right.as_deref());
        Some((&n.key, &n.value))
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::Treap;
    use quickcheck_macros::quickcheck;
    use std::collections::BTreeMap;

    // ── unit tests ───────────────────────────────────────────────────────────

    #[test]
    fn empty_treap() {
        let t: Treap<i32, i32> = Treap::new();
        assert!(t.is_empty());
        assert_eq!(t.len(), 0);
        assert_eq!(t.get(&1), None);
        assert!(!t.contains_key(&1));
    }

    #[test]
    fn insert_and_get() {
        let mut t = Treap::new();
        assert_eq!(t.insert(10, "ten"), None);
        assert_eq!(t.insert(5, "five"), None);
        assert_eq!(t.insert(20, "twenty"), None);
        assert_eq!(t.len(), 3);
        assert_eq!(t.get(&10), Some(&"ten"));
        assert_eq!(t.get(&5), Some(&"five"));
        assert_eq!(t.get(&20), Some(&"twenty"));
        assert_eq!(t.get(&99), None);
        assert!(t.contains_key(&10));
        assert!(!t.contains_key(&99));
    }

    #[test]
    fn insert_duplicate_returns_old_value() {
        let mut t = Treap::new();
        assert_eq!(t.insert(1, 100), None);
        assert_eq!(t.insert(1, 200), Some(100));
        assert_eq!(t.insert(1, 300), Some(200));
        // len stays 1 — no duplicate nodes.
        assert_eq!(t.len(), 1);
        assert_eq!(t.get(&1), Some(&300));
    }

    #[test]
    fn remove_returns_value_and_absent_key_returns_none() {
        let mut t = Treap::new();
        t.insert(3, "c");
        t.insert(1, "a");
        t.insert(2, "b");
        // Remove present key.
        assert_eq!(t.remove(&2), Some("b"));
        assert_eq!(t.len(), 2);
        assert!(!t.contains_key(&2));
        // Remove same key again -> None.
        assert_eq!(t.remove(&2), None);
        assert_eq!(t.len(), 2);
        // Remove absent key.
        assert_eq!(t.remove(&99), None);
        assert_eq!(t.len(), 2);
    }

    #[test]
    fn in_order_iteration_is_sorted() {
        let keys = vec![5, 3, 8, 1, 4, 7, 9, 2, 6];
        let mut t = Treap::new();
        for k in &keys {
            t.insert(*k, k * 10);
        }
        let collected: Vec<i32> = t.iter().map(|(&k, _)| k).collect();
        let mut sorted = keys;
        sorted.sort_unstable();
        assert_eq!(collected, sorted);
    }

    #[test]
    fn deterministic_build_expected_depth() {
        // Build a 100-key treap with a fixed seed and verify the depth is
        // well within the generous O(log n) bound of 30.

        // Measure actual tree depth by walking the Node tree.
        // Defined before first use to satisfy items_after_statements.
        fn depth<K, V>(node: Option<&super::Node<K, V>>) -> usize {
            node.map_or(0, |n| {
                1 + depth(n.left.as_deref()).max(depth(n.right.as_deref()))
            })
        }

        let mut t: Treap<i32, i32> = Treap::with_seed(7);
        for k in 1..=100 {
            t.insert(k, k);
        }
        assert_eq!(t.len(), 100);

        let d = depth(t.root.as_deref());
        // log2(100) ~= 6.6; allowing <= 30 is very generous to avoid flake.
        assert!(d <= 30, "tree depth {d} exceeds generous bound of 30");
    }

    // ── property test ────────────────────────────────────────────────────────

    /// Operations replayed on both the treap and a `BTreeMap` oracle.
    #[derive(Clone, Debug)]
    enum Op {
        Insert(i32, i32),
        Remove(i32),
        Get(i32),
    }

    impl quickcheck::Arbitrary for Op {
        fn arbitrary(g: &mut quickcheck::Gen) -> Self {
            let tag = u8::arbitrary(g) % 3;
            // small key space to generate collisions
            let k = i32::arbitrary(g) % 20;
            match tag {
                0 => Self::Insert(k, i32::arbitrary(g)),
                1 => Self::Remove(k),
                _ => Self::Get(k),
            }
        }
    }

    #[allow(clippy::needless_pass_by_value)]
    #[quickcheck]
    fn model_check_against_btreemap(ops: Vec<Op>) -> bool {
        let mut treap: Treap<i32, i32> = Treap::new();
        let mut model: BTreeMap<i32, i32> = BTreeMap::new();

        for op in ops {
            match op {
                Op::Insert(k, v) => {
                    let t_old = treap.insert(k, v);
                    let m_old = model.insert(k, v);
                    if t_old != m_old {
                        return false;
                    }
                }
                Op::Remove(k) => {
                    let t_val = treap.remove(&k);
                    let m_val = model.remove(&k);
                    if t_val != m_val {
                        return false;
                    }
                }
                Op::Get(k) => {
                    if treap.get(&k) != model.get(&k) {
                        return false;
                    }
                }
            }

            // After every operation: len must match.
            if treap.len() != model.len() {
                return false;
            }
        }

        // Final iter must exactly match BTreeMap::iter in key-value order.
        let treap_pairs: Vec<(i32, i32)> = treap.iter().map(|(&k, &v)| (k, v)).collect();
        let model_pairs: Vec<(i32, i32)> = model.iter().map(|(&k, &v)| (k, v)).collect();
        treap_pairs == model_pairs
    }
}
