//! AVL Tree (Adelson-Velsky & Landis, 1962): a self-balancing binary search tree.
//!
//! Every node stores an explicit `height`. After each `insert` or `remove`,
//! the code retraces the path back to the root and rebalances every node whose
//! left/right height difference exceeds 1 via the four standard rotation cases:
//! LL, LR, RR, RL.
//!
//! # Complexity
//! - `insert`, `remove`, `contains`: **O(log n)** worst-case.
//! - `min`, `max`, `height`, `len`, `is_empty`: **O(log n)** / **O(1)**.
//! - `iter_inorder`: **O(n)** (yields keys in strictly ascending order).
//! - Space: **O(n)**.
//!
//! # Balance invariant
//! At every node: `|height(left) - height(right)| <= 1`.
//!
//! # Semantics
//! Set semantics: duplicate inserts are rejected and return `false`.

/// An AVL-balanced binary search tree storing unique keys.
pub struct AvlTree<K: Ord> {
    root: Option<Box<Node<K>>>,
    len: usize,
}

#[allow(clippy::use_self)] // `Node<K>` in field types cannot use `Self` (no impl block)
struct Node<K> {
    key: K,
    height: i32,
    left: Option<Box<Node<K>>>,
    right: Option<Box<Node<K>>>,
}

// ---------------------------------------------------------------------------
// Free helper functions
// ---------------------------------------------------------------------------

/// Returns the stored height of a node, or 0 for `None`.
#[inline]
#[allow(clippy::ref_option)] // Taking `&Option<Box<Node<K>>>` is idiomatic here
fn height<K>(opt: &Option<Box<Node<K>>>) -> i32 {
    opt.as_ref().map_or(0, |n| n.height)
}

/// Re-computes and caches the height of a node from its children.
#[inline]
fn update_height<K>(node: &mut Node<K>) {
    node.height = 1 + height(&node.left).max(height(&node.right));
}

/// `left_height - right_height` at `node`.
#[inline]
fn balance_factor<K>(node: &Node<K>) -> i32 {
    height(&node.left) - height(&node.right)
}

// ---------------------------------------------------------------------------
// Rotations
// ---------------------------------------------------------------------------

/// Right-rotation around `node` (fixes a left-heavy subtree).
///
/// ```text
///     node               pivot
///     /  \               /   \
///  pivot   C    =>      A    node
///  /   \                     /  \
/// A     B                   B    C
/// ```
#[allow(clippy::unnecessary_box_returns)] // Box is load-bearing: it's the tree node allocation
fn rotate_right<K: Ord>(mut node: Box<Node<K>>) -> Box<Node<K>> {
    let mut pivot = node
        .left
        .take()
        .expect("rotate_right: left child must exist");
    node.left = pivot.right.take();
    update_height(&mut node);
    pivot.right = Some(node);
    update_height(&mut pivot);
    pivot
}

/// Left-rotation around `node` (fixes a right-heavy subtree).
///
/// ```text
///  node                  pivot
///  /  \                  /   \
/// A   pivot    =>      node   C
///     /   \            /  \
///    B     C          A    B
/// ```
#[allow(clippy::unnecessary_box_returns)]
fn rotate_left<K: Ord>(mut node: Box<Node<K>>) -> Box<Node<K>> {
    let mut pivot = node
        .right
        .take()
        .expect("rotate_left: right child must exist");
    node.right = pivot.left.take();
    update_height(&mut node);
    pivot.left = Some(node);
    update_height(&mut pivot);
    pivot
}

// ---------------------------------------------------------------------------
// Rebalancing
// ---------------------------------------------------------------------------

/// Examines `balance_factor` and applies the appropriate rotation(s).
///
/// | Case | Condition                    | Fix                              |
/// |------|------------------------------|----------------------------------|
/// | LL   | bf > 1, left bf >= 0         | single right rotation            |
/// | LR   | bf > 1, left bf < 0          | left-rotate left child, then right |
/// | RR   | bf < -1, right bf <= 0       | single left rotation             |
/// | RL   | bf < -1, right bf > 0        | right-rotate right child, then left |
#[allow(clippy::unnecessary_box_returns)]
fn rebalance<K: Ord>(mut node: Box<Node<K>>) -> Box<Node<K>> {
    update_height(&mut node);
    let bf = balance_factor(&node);

    if bf > 1 {
        // Left-heavy
        if balance_factor(node.left.as_ref().expect("left must exist")) < 0 {
            // LR case: left child is right-heavy
            node.left = Some(rotate_left(node.left.take().expect("left must exist")));
        }
        // LL case (or LR after pre-rotation)
        return rotate_right(node);
    }

    if bf < -1 {
        // Right-heavy
        if balance_factor(node.right.as_ref().expect("right must exist")) > 0 {
            // RL case: right child is left-heavy
            node.right = Some(rotate_right(node.right.take().expect("right must exist")));
        }
        // RR case (or RL after pre-rotation)
        return rotate_left(node);
    }

    node
}

// ---------------------------------------------------------------------------
// Recursive insert / remove helpers (return ownership of subtree root)
// ---------------------------------------------------------------------------

/// Inserts `key` into the subtree rooted at `opt`, returning the (possibly
/// rotated) new root and `true` if a new node was created.
fn insert_into<K: Ord>(opt: Option<Box<Node<K>>>, key: K) -> (Option<Box<Node<K>>>, bool) {
    match opt {
        None => {
            let node = Box::new(Node {
                key,
                height: 1,
                left: None,
                right: None,
            });
            (Some(node), true)
        }
        Some(mut node) => {
            let inserted = match key.cmp(&node.key) {
                std::cmp::Ordering::Less => {
                    let (new_left, ins) = insert_into(node.left.take(), key);
                    node.left = new_left;
                    ins
                }
                std::cmp::Ordering::Greater => {
                    let (new_right, ins) = insert_into(node.right.take(), key);
                    node.right = new_right;
                    ins
                }
                std::cmp::Ordering::Equal => return (Some(node), false), // duplicate
            };
            (Some(rebalance(node)), inserted)
        }
    }
}

/// Removes and returns the minimum key in the subtree rooted at `node`
/// (guaranteed non-None), returning `(new_subtree_root, min_key)`.
fn remove_min<K: Ord>(mut node: Box<Node<K>>) -> (Option<Box<Node<K>>>, K) {
    match node.left.take() {
        None => {
            // This node is the minimum; splice it out.
            (node.right.take(), node.key)
        }
        Some(left) => {
            let (new_left, min_key) = remove_min(left);
            node.left = new_left;
            (Some(rebalance(node)), min_key)
        }
    }
}

/// Removes `key` from the subtree rooted at `opt`, returning the (possibly
/// rotated) new root and `true` if a node was removed.
#[allow(clippy::option_if_let_else)] // map_or_else here harms readability of the match arms
fn remove_from<K: Ord>(opt: Option<Box<Node<K>>>, key: &K) -> (Option<Box<Node<K>>>, bool) {
    match opt {
        None => (None, false),
        Some(mut node) => match key.cmp(&node.key) {
            std::cmp::Ordering::Less => {
                let (new_left, removed) = remove_from(node.left.take(), key);
                node.left = new_left;
                (Some(rebalance(node)), removed)
            }
            std::cmp::Ordering::Greater => {
                let (new_right, removed) = remove_from(node.right.take(), key);
                node.right = new_right;
                (Some(rebalance(node)), removed)
            }
            std::cmp::Ordering::Equal => {
                // Found the target node.
                let new_root = match (node.left.take(), node.right.take()) {
                    (None, right) => right,
                    (left, None) => left,
                    (left, Some(right)) => {
                        // Two children: replace with in-order successor (min of right),
                        // then remove the successor from the right subtree.
                        let (new_right, successor_key) = remove_min(right);
                        node.key = successor_key;
                        node.left = left;
                        node.right = new_right;
                        Some(rebalance(node))
                    }
                };
                (new_root, true)
            }
        },
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

impl<K: Ord> AvlTree<K> {
    /// Creates an empty AVL tree.
    pub const fn new() -> Self {
        Self { root: None, len: 0 }
    }

    /// Inserts `key` into the tree.
    ///
    /// Returns `true` if the key was newly inserted, `false` if it was already
    /// present (the tree is unchanged in the latter case).
    pub fn insert(&mut self, key: K) -> bool {
        let (new_root, inserted) = insert_into(self.root.take(), key);
        self.root = new_root;
        if inserted {
            self.len += 1;
        }
        inserted
    }

    /// Returns `true` if `key` is present in the tree.
    pub fn contains(&self, key: &K) -> bool {
        let mut cur = &self.root;
        while let Some(node) = cur {
            match key.cmp(&node.key) {
                std::cmp::Ordering::Less => cur = &node.left,
                std::cmp::Ordering::Greater => cur = &node.right,
                std::cmp::Ordering::Equal => return true,
            }
        }
        false
    }

    /// Removes `key` from the tree.
    ///
    /// Returns `true` if the key was found and removed, `false` if absent.
    pub fn remove(&mut self, key: &K) -> bool {
        let (new_root, removed) = remove_from(self.root.take(), key);
        self.root = new_root;
        if removed {
            self.len -= 1;
        }
        removed
    }

    /// Returns the number of keys stored in the tree.
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the tree contains no keys.
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns a reference to the smallest key, or `None` if the tree is empty.
    pub fn min(&self) -> Option<&K> {
        let mut cur = self.root.as_ref()?;
        while let Some(left) = cur.left.as_ref() {
            cur = left;
        }
        Some(&cur.key)
    }

    /// Returns a reference to the largest key, or `None` if the tree is empty.
    pub fn max(&self) -> Option<&K> {
        let mut cur = self.root.as_ref()?;
        while let Some(right) = cur.right.as_ref() {
            cur = right;
        }
        Some(&cur.key)
    }

    /// Returns the height of the tree (0 for an empty tree).
    pub fn height(&self) -> i32 {
        height(&self.root)
    }

    /// Returns an iterator that yields all keys in strictly ascending order.
    ///
    /// Complexity: O(n) time, O(h) stack space where h is the tree height.
    pub fn iter_inorder(&self) -> impl Iterator<Item = &K> {
        InorderIter::new(self.root.as_deref())
    }
}

impl<K: Ord> Default for AvlTree<K> {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// In-order iterator (iterative, uses an explicit stack)
// ---------------------------------------------------------------------------

struct InorderIter<'a, K> {
    stack: Vec<&'a Node<K>>,
}

impl<'a, K> InorderIter<'a, K> {
    fn new(mut node: Option<&'a Node<K>>) -> Self {
        let mut stack = Vec::new();
        while let Some(n) = node {
            stack.push(n);
            node = n.left.as_deref();
        }
        Self { stack }
    }
}

impl<'a, K> Iterator for InorderIter<'a, K> {
    type Item = &'a K;

    fn next(&mut self) -> Option<Self::Item> {
        let node = self.stack.pop()?;
        // Push the right subtree's leftmost spine.
        let mut cur = node.right.as_deref();
        while let Some(n) = cur {
            self.stack.push(n);
            cur = n.left.as_deref();
        }
        Some(&node.key)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::AvlTree;
    use quickcheck_macros::quickcheck;
    use std::collections::BTreeSet;

    // ---- debug helper ----

    /// Returns `true` iff every node in the subtree satisfies the AVL invariant.
    #[allow(clippy::ref_option)]
    fn is_avl_balanced_node<K>(opt: &Option<Box<super::Node<K>>>) -> bool {
        match opt {
            None => true,
            Some(node) => {
                let bf = super::balance_factor(node);
                if bf.abs() > 1 {
                    return false;
                }
                // Also verify the cached height is correct.
                let expected_h = 1 + super::height(&node.left).max(super::height(&node.right));
                if node.height != expected_h {
                    return false;
                }
                is_avl_balanced_node(&node.left) && is_avl_balanced_node(&node.right)
            }
        }
    }

    impl<K: Ord> AvlTree<K> {
        fn is_avl_balanced(&self) -> bool {
            is_avl_balanced_node(&self.root)
        }
    }

    // ---- unit tests ----

    #[test]
    fn empty_tree() {
        let t: AvlTree<i32> = AvlTree::new();
        assert!(t.is_empty());
        assert_eq!(t.len(), 0);
        assert!(!t.contains(&0));
        assert_eq!(t.min(), None);
        assert_eq!(t.max(), None);
        assert_eq!(t.height(), 0);
    }

    #[test]
    fn single_insert_and_contains() {
        let mut t = AvlTree::new();
        assert!(t.insert(42));
        assert_eq!(t.len(), 1);
        assert!(t.contains(&42));
        assert!(!t.contains(&0));
        assert_eq!(t.min(), Some(&42));
        assert_eq!(t.max(), Some(&42));
        assert_eq!(t.height(), 1);
    }

    #[test]
    fn duplicate_insert_returns_false() {
        let mut t = AvlTree::new();
        assert!(t.insert(7));
        assert!(!t.insert(7));
        assert_eq!(t.len(), 1);
    }

    #[test]
    fn remove_absent_returns_false() {
        let mut t: AvlTree<i32> = AvlTree::new();
        assert!(!t.remove(&99));
        t.insert(1);
        assert!(!t.remove(&99));
        assert_eq!(t.len(), 1);
    }

    #[test]
    fn remove_only_element() {
        let mut t = AvlTree::new();
        t.insert(5);
        assert!(t.remove(&5));
        assert!(t.is_empty());
        assert_eq!(t.min(), None);
        assert_eq!(t.max(), None);
    }

    #[test]
    fn inorder_traversal_yields_sorted_output() {
        let mut t = AvlTree::new();
        for k in [5, 3, 7, 1, 4, 6, 8] {
            t.insert(k);
        }
        let keys: Vec<i32> = t.iter_inorder().copied().collect();
        assert_eq!(keys, [1, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn sequential_insert_height_bound() {
        let mut t = AvlTree::new();
        for k in 1..=100_i32 {
            t.insert(k);
            assert!(t.is_avl_balanced());
        }
        assert_eq!(t.len(), 100);
        // For n=100, floor(log2(100)) = 6, so 2*log2(100) ~ 13.3; bound at 14.
        assert!(t.height() <= 14, "height {} exceeds bound 14", t.height());
    }

    #[test]
    fn reverse_sequential_insert_height_bound() {
        let mut t = AvlTree::new();
        for k in (1..=100_i32).rev() {
            t.insert(k);
            assert!(t.is_avl_balanced());
        }
        assert_eq!(t.len(), 100);
        assert!(t.height() <= 14, "height {} exceeds bound 14", t.height());
    }

    #[test]
    fn random_insert_and_remove_avl_invariant() {
        // Deterministic pseudo-random sequence using a simple LCG.
        let mut state: u64 = 0xDEAD_BEEF_CAFE_1337;
        let lcg = |s: &mut u64| -> i32 {
            *s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((*s >> 33) as i32).abs()
        };

        let mut t = AvlTree::new();
        let mut oracle = BTreeSet::new();

        for _ in 0..500 {
            let k = lcg(&mut state) % 80;
            if lcg(&mut state) % 3 == 0 {
                // remove ~1/3 of the time
                let removed = t.remove(&k);
                assert_eq!(removed, oracle.remove(&k));
            } else {
                let inserted = t.insert(k);
                assert_eq!(inserted, oracle.insert(k));
            }
            assert!(t.is_avl_balanced());
            assert_eq!(t.len(), oracle.len());
        }
    }

    #[test]
    fn min_max_after_operations() {
        let mut t = AvlTree::new();
        for k in [10, 3, 15, 1, 7, 20] {
            t.insert(k);
        }
        assert_eq!(t.min(), Some(&1));
        assert_eq!(t.max(), Some(&20));
        t.remove(&1);
        assert_eq!(t.min(), Some(&3));
        t.remove(&20);
        assert_eq!(t.max(), Some(&15));
    }

    #[test]
    fn two_children_removal_maintains_bst_order() {
        let mut t = AvlTree::new();
        for k in [8, 4, 12, 2, 6, 10, 14] {
            t.insert(k);
        }
        // Remove a node with two children (the root, 8).
        assert!(t.remove(&8));
        let keys: Vec<i32> = t.iter_inorder().copied().collect();
        assert_eq!(keys, [2, 4, 6, 10, 12, 14]);
    }

    // ---- property-based test ----

    #[derive(Clone, Debug)]
    enum Op {
        Insert(i32),
        Remove(i32),
        Contains(i32),
    }

    impl quickcheck::Arbitrary for Op {
        fn arbitrary(g: &mut quickcheck::Gen) -> Self {
            let key = i32::arbitrary(g) % 50; // small domain for collisions
            match u8::arbitrary(g) % 3 {
                0 => Self::Insert(key),
                1 => Self::Remove(key),
                _ => Self::Contains(key),
            }
        }
    }

    #[allow(clippy::needless_pass_by_value)]
    #[quickcheck]
    fn model_checked_against_btreeset(ops: Vec<Op>) -> bool {
        let mut tree = AvlTree::new();
        let mut model: BTreeSet<i32> = BTreeSet::new();

        for op in &ops {
            match *op {
                Op::Insert(k) => {
                    let tree_res = tree.insert(k);
                    let model_res = model.insert(k);
                    if tree_res != model_res {
                        return false;
                    }
                }
                Op::Remove(k) => {
                    let tree_res = tree.remove(&k);
                    let model_res = model.remove(&k);
                    if tree_res != model_res {
                        return false;
                    }
                }
                Op::Contains(k) => {
                    if tree.contains(&k) != model.contains(&k) {
                        return false;
                    }
                }
            }
            if !tree.is_avl_balanced() {
                return false;
            }
            if tree.len() != model.len() {
                return false;
            }
        }

        // Final in-order traversal must match the sorted model.
        let tree_keys: Vec<i32> = tree.iter_inorder().copied().collect();
        let model_keys: Vec<i32> = model.iter().copied().collect();
        tree_keys == model_keys
    }
}
