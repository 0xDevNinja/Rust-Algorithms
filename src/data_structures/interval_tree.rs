//! Augmented BST interval tree (CLRS §14.3) with treap-based balancing.
//!
//! An **interval tree** stores a set of closed intervals `[low, high]`, each
//! tagged with an associated value, and supports two key queries:
//! - **Point query**: find every interval that contains a given point `p`
//!   (i.e. `low <= p <= high`).
//! - **Overlap query**: find every interval that overlaps a given query
//!   interval `[lo, hi]` (two closed intervals overlap iff
//!   `a.low <= b.high && b.low <= a.high`).
//!
//! # Augmentation
//! Each internal node stores `max_high`: the maximum `high` endpoint in the
//! subtree rooted at that node. This allows early pruning during search:
//! - Descend into the **left** child only if `left.max_high >= query.low`
//!   (otherwise no interval in that subtree can overlap the query).
//! - Descend into the **right** child only if `node.low <= query.high`
//!   (otherwise all intervals in the right subtree start after the query ends).
//!
//! # Balancing: treap
//! Each node receives a random `u64` priority at insertion time. The BST is
//! maintained as a **treap**: a simultaneous BST on `(low, high)` and a
//! max-heap on `priority`. Standard split/merge rotations keep the expected
//! height at `O(log n)` without the implementation complexity of red-black
//! trees. The random priorities ensure O(log n) expected height with high
//! probability for any insertion sequence.
//!
//! # Complexity
//! - **Space**: O(n).
//! - **Insert**: O(log n) expected.
//! - **Remove**: O(log n) expected.
//! - **Query (point or overlap)**: O(log n + k) expected, where k is the
//!   number of intervals in the result.
//!
//! # Preconditions
//! `low <= high` is asserted on every insert.

// Clippy: the Node struct fields must name the type explicitly; Self is only
// available inside impl blocks, so suppress the lint at struct level.
#[allow(clippy::use_self)]
/// A node in the treap.
struct Node<T, V> {
    /// The closed interval stored at this node.
    interval: (T, T),
    /// User-supplied value associated with the interval.
    value: V,
    /// `max(high, left.max_high, right.max_high)` for pruning.
    max_high: T,
    /// Random priority; the treap maintains max-heap order on this field.
    priority: u64,
    /// Number of nodes in this subtree (including self).
    size: usize,
    left: Option<Box<Node<T, V>>>,
    right: Option<Box<Node<T, V>>>,
}

/// Convenience alias to reduce repetition in free function signatures.
type Link<T, V> = Option<Box<Node<T, V>>>;

impl<T: Ord + Copy, V: Clone> Node<T, V> {
    const fn new(low: T, high: T, value: V, priority: u64) -> Self {
        Self {
            interval: (low, high),
            value,
            max_high: high,
            priority,
            size: 1,
            left: None,
            right: None,
        }
    }

    /// Recomputes `max_high` and `size` from the children. Must be called
    /// after any structural change to this node's children.
    fn pull_up(&mut self) {
        let left_max = self.left.as_ref().map(|n| n.max_high);
        let right_max = self.right.as_ref().map(|n| n.max_high);
        self.max_high = self.interval.1;
        if let Some(m) = left_max {
            if m > self.max_high {
                self.max_high = m;
            }
        }
        if let Some(m) = right_max {
            if m > self.max_high {
                self.max_high = m;
            }
        }
        self.size = 1
            + self.left.as_ref().map_or(0, |n| n.size)
            + self.right.as_ref().map_or(0, |n| n.size);
    }
}

// ---------------------------------------------------------------------------
// Treap helpers (all operating on Link<T,V> = Option<Box<Node<T,V>>>)
// ---------------------------------------------------------------------------

/// Split `tree` into two trees:
/// - left result: all nodes whose `(low, high)` key is `<= (low_key, high_key)`.
/// - right result: all nodes whose `(low, high)` key is `> (low_key, high_key)`.
#[allow(clippy::option_if_let_else)]
fn split<T: Ord + Copy, V: Clone>(
    tree: Link<T, V>,
    low_key: T,
    high_key: T,
) -> (Link<T, V>, Link<T, V>) {
    match tree {
        None => (None, None),
        Some(mut node) => {
            let cmp_low = node.interval.0.cmp(&low_key);
            let cmp_high = node.interval.1.cmp(&high_key);
            // BST key: primary on low, tie-break on high.
            let goes_left = cmp_low < std::cmp::Ordering::Equal
                || (cmp_low == std::cmp::Ordering::Equal
                    && cmp_high <= std::cmp::Ordering::Equal);
            if goes_left {
                // This node and everything to its left go into the left half;
                // recurse on the right child.
                let (rl, rr) = split(node.right.take(), low_key, high_key);
                node.right = rl;
                node.pull_up();
                (Some(node), rr)
            } else {
                // This node and everything to its right go into the right half;
                // recurse on the left child.
                let (ll, lr) = split(node.left.take(), low_key, high_key);
                node.left = lr;
                node.pull_up();
                (ll, Some(node))
            }
        }
    }
}

/// Merge two treap trees where every key in `left` is `<=` every key in
/// `right`. Maintains max-heap order on `priority`.
fn merge<T: Ord + Copy, V: Clone>(left: Link<T, V>, right: Link<T, V>) -> Link<T, V> {
    match (left, right) {
        (None, r) => r,
        (l, None) => l,
        (Some(mut l), Some(mut r)) => {
            if l.priority >= r.priority {
                // l is the root; merge l.right with r.
                l.right = merge(l.right.take(), Some(r));
                l.pull_up();
                Some(l)
            } else {
                // r is the root; merge l with r.left.
                r.left = merge(Some(l), r.left.take());
                r.pull_up();
                Some(r)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Public struct
// ---------------------------------------------------------------------------

/// Augmented BST interval tree backed by a randomised treap.
///
/// Generic over endpoint type `T` (requires `Ord + Copy`) and associated
/// value `V` (requires `Clone`). Multiple identical intervals may be stored;
/// they are treated as distinct entries. The BST sort key is `(low, high)`.
pub struct IntervalTree<T: Ord + Copy, V: Clone> {
    root: Link<T, V>,
    /// xorshift64 state for priority generation.
    rng: u64,
}

impl<T: Ord + Copy, V: Clone> IntervalTree<T, V> {
    /// Creates an empty interval tree.
    pub const fn new() -> Self {
        Self {
            root: None,
            // A fixed seed gives deterministic behaviour within a single
            // execution; the priorities are purely internal.
            rng: 0x517c_c1b7_2722_0a95,
        }
    }

    /// Returns the number of intervals stored in the tree.
    pub fn len(&self) -> usize {
        self.root.as_ref().map_or(0, |n| n.size)
    }

    /// Returns `true` if the tree holds no intervals.
    pub const fn is_empty(&self) -> bool {
        self.root.is_none()
    }

    /// Next pseudo-random priority (xorshift64).
    const fn next_priority(&mut self) -> u64 {
        let mut x = self.rng;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng = x;
        x
    }

    /// Inserts the interval `[low, high]` with associated `value`.
    ///
    /// # Panics
    /// Panics if `low > high`.
    pub fn insert(&mut self, low: T, high: T, value: V) {
        assert!(low <= high, "interval low must be <= high");
        let priority = self.next_priority();
        let new_node = Box::new(Node::new(low, high, value, priority));
        // Split at (low, high): everything <= key goes left, rest goes right.
        let (left, right) = split(self.root.take(), low, high);
        // Insert: merge left + new_node + right.
        self.root = merge(merge(left, Some(new_node)), right);
    }

    /// Removes the first node whose interval exactly equals `(low, high)` and
    /// returns its value. Returns `None` if no such interval is present.
    pub fn remove_first_match(&mut self, low: T, high: T) -> Option<V> {
        remove_first(&mut self.root, low, high)
    }

    /// Returns references to the values of all intervals that contain `p`
    /// (i.e. `low <= p <= high` for each returned interval).
    ///
    /// The result is unsorted; its order is an implementation detail.
    pub fn query_point(&self, p: T) -> Vec<&V> {
        let mut out = Vec::new();
        // A point p is contained in [lo, hi] iff lo <= p <= hi,
        // which is exactly the overlap condition for the degenerate interval [p, p].
        collect_overlap(self.root.as_deref(), p, p, &mut out);
        out
    }

    /// Returns references to the values of all intervals that overlap
    /// `[low, high]` (closed-interval semantics: two intervals overlap iff
    /// `a.low <= b.high && b.low <= a.high`).
    ///
    /// The result is unsorted; its order is an implementation detail.
    pub fn query_overlap(&self, low: T, high: T) -> Vec<&V> {
        let mut out = Vec::new();
        collect_overlap(self.root.as_deref(), low, high, &mut out);
        out
    }
}

impl<T: Ord + Copy, V: Clone> Default for IntervalTree<T, V> {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Internal recursive helpers
// ---------------------------------------------------------------------------

/// Collect all values in the subtree rooted at `node` whose interval overlaps
/// `[qlo, qhi]`.
fn collect_overlap<'a, T: Ord + Copy, V: Clone>(
    node: Option<&'a Node<T, V>>,
    qlo: T,
    qhi: T,
    out: &mut Vec<&'a V>,
) {
    let Some(n) = node else { return };

    // Pruning: if the maximum high in this entire subtree is less than the
    // query's low endpoint, no interval here can overlap.
    if n.max_high < qlo {
        return;
    }

    // Descend left child if it could contain overlapping intervals.
    if let Some(left) = n.left.as_deref() {
        if left.max_high >= qlo {
            collect_overlap(Some(left), qlo, qhi, out);
        }
    }

    // Check this node.
    let (nlo, nhi) = n.interval;
    if nlo <= qhi && qlo <= nhi {
        out.push(&n.value);
    }

    // Descend right child only if this node's low is within the query range
    // (all right-subtree nodes have low >= nlo; if nlo > qhi they are too far
    // right to overlap).
    if nlo <= qhi {
        if let Some(right) = n.right.as_deref() {
            collect_overlap(Some(right), qlo, qhi, out);
        }
    }
}

/// Remove the first node in the subtree with interval exactly `(low, high)`.
/// Returns the removed value, or `None` if not found.
fn remove_first<T: Ord + Copy, V: Clone>(
    link: &mut Link<T, V>,
    low: T,
    high: T,
) -> Option<V> {
    let node = link.as_mut()?;

    let cmp_low = low.cmp(&node.interval.0);
    let cmp_high = high.cmp(&node.interval.1);

    let result = match (cmp_low, cmp_high) {
        // Exact match — remove this node by merging its children.
        (std::cmp::Ordering::Equal, std::cmp::Ordering::Equal) => {
            let found = link.take().expect("we just checked Some");
            *link = merge(found.left, found.right);
            return Some(found.value);
        }
        // The target key is smaller: go left.
        (std::cmp::Ordering::Less, _) => remove_first(&mut link.as_mut()?.left, low, high),
        // Same low, target high is smaller: go left (BST orders by (low, high)
        // lexicographically).
        (std::cmp::Ordering::Equal, std::cmp::Ordering::Less) => {
            remove_first(&mut link.as_mut()?.left, low, high)
        }
        // The target key is larger: go right.
        _ => remove_first(&mut link.as_mut()?.right, low, high),
    };

    if result.is_some() {
        link.as_mut()?.pull_up();
    }
    result
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::IntervalTree;
    use quickcheck::TestResult;
    use quickcheck_macros::quickcheck;

    // ---- empty tree --------------------------------------------------------

    #[test]
    fn empty_point_query_returns_empty() {
        let t: IntervalTree<i32, u32> = IntervalTree::new();
        assert!(t.is_empty());
        assert_eq!(t.len(), 0);
        assert!(t.query_point(0).is_empty());
    }

    #[test]
    fn empty_overlap_query_returns_empty() {
        let t: IntervalTree<i32, u32> = IntervalTree::new();
        assert!(t.query_overlap(0, 10).is_empty());
    }

    // ---- single interval ---------------------------------------------------

    #[test]
    fn single_interval_basic() {
        let mut t: IntervalTree<i32, u32> = IntervalTree::new();
        t.insert(3, 7, 42);
        assert_eq!(t.len(), 1);
        assert!(!t.is_empty());
    }

    #[test]
    fn point_query_left_endpoint() {
        let mut t: IntervalTree<i32, u32> = IntervalTree::new();
        t.insert(3, 7, 42);
        let r = t.query_point(3);
        assert_eq!(r.len(), 1);
        assert_eq!(*r[0], 42);
    }

    #[test]
    fn point_query_right_endpoint() {
        let mut t: IntervalTree<i32, u32> = IntervalTree::new();
        t.insert(3, 7, 42);
        let r = t.query_point(7);
        assert_eq!(r.len(), 1);
        assert_eq!(*r[0], 42);
    }

    #[test]
    fn point_query_strictly_inside() {
        let mut t: IntervalTree<i32, u32> = IntervalTree::new();
        t.insert(3, 7, 42);
        let r = t.query_point(5);
        assert_eq!(r.len(), 1);
        assert_eq!(*r[0], 42);
    }

    #[test]
    fn point_query_strictly_outside_left() {
        let mut t: IntervalTree<i32, u32> = IntervalTree::new();
        t.insert(3, 7, 42);
        assert!(t.query_point(2).is_empty());
    }

    #[test]
    fn point_query_strictly_outside_right() {
        let mut t: IntervalTree<i32, u32> = IntervalTree::new();
        t.insert(3, 7, 42);
        assert!(t.query_point(8).is_empty());
    }

    // ---- overlap semantics (closed intervals) ------------------------------

    #[test]
    fn overlap_touching_left_endpoint() {
        // Query [1, 3] touches [3, 7] at 3 — should overlap.
        let mut t: IntervalTree<i32, u32> = IntervalTree::new();
        t.insert(3, 7, 1);
        let r = t.query_overlap(1, 3);
        assert_eq!(r.len(), 1);
    }

    #[test]
    fn overlap_touching_right_endpoint() {
        // Query [7, 10] touches [3, 7] at 7 — should overlap.
        let mut t: IntervalTree<i32, u32> = IntervalTree::new();
        t.insert(3, 7, 1);
        let r = t.query_overlap(7, 10);
        assert_eq!(r.len(), 1);
    }

    #[test]
    fn overlap_query_no_overlap() {
        let mut t: IntervalTree<i32, u32> = IntervalTree::new();
        t.insert(3, 7, 1);
        assert!(t.query_overlap(8, 10).is_empty());
        assert!(t.query_overlap(0, 2).is_empty());
    }

    // ---- nested intervals --------------------------------------------------

    #[test]
    fn nested_intervals() {
        let mut t: IntervalTree<i32, u32> = IntervalTree::new();
        t.insert(1, 10, 1); // outer
        t.insert(3, 6, 2); // inner
        t.insert(5, 8, 3); // overlapping
        assert_eq!(t.len(), 3);

        // Point 5 hits all three.
        let mut r: Vec<u32> = t.query_point(5).iter().map(|v| **v).collect();
        r.sort_unstable();
        assert_eq!(r, vec![1, 2, 3]);

        // Point 2 hits only the outer.
        let r2: Vec<u32> = t.query_point(2).iter().map(|v| **v).collect();
        assert_eq!(r2, vec![1]);

        // Point 7 hits outer and overlapping.
        let mut r3: Vec<u32> = t.query_point(7).iter().map(|v| **v).collect();
        r3.sort_unstable();
        assert_eq!(r3, vec![1, 3]);
    }

    // ---- identical intervals stored multiple times -------------------------

    #[test]
    fn identical_intervals_are_distinct_entries() {
        let mut t: IntervalTree<i32, u32> = IntervalTree::new();
        t.insert(2, 5, 10);
        t.insert(2, 5, 20);
        t.insert(2, 5, 30);
        assert_eq!(t.len(), 3);

        let mut r: Vec<u32> = t.query_point(3).iter().map(|v| **v).collect();
        r.sort_unstable();
        assert_eq!(r, vec![10, 20, 30]);
    }

    // ---- remove_first_match ------------------------------------------------

    #[test]
    fn remove_first_match_decreases_len() {
        let mut t: IntervalTree<i32, u32> = IntervalTree::new();
        t.insert(1, 5, 99);
        t.insert(2, 8, 77);
        assert_eq!(t.len(), 2);

        let v = t.remove_first_match(1, 5);
        assert_eq!(v, Some(99));
        assert_eq!(t.len(), 1);

        // The remaining interval still works.
        assert_eq!(t.query_point(3).len(), 1);
        assert_eq!(*t.query_point(3)[0], 77);
    }

    #[test]
    fn remove_first_match_absent_returns_none() {
        let mut t: IntervalTree<i32, u32> = IntervalTree::new();
        t.insert(1, 5, 42);
        assert_eq!(t.remove_first_match(2, 5), None);
        assert_eq!(t.remove_first_match(1, 6), None);
        assert_eq!(t.len(), 1);
    }

    #[test]
    fn remove_first_match_one_of_duplicates() {
        let mut t: IntervalTree<i32, u32> = IntervalTree::new();
        t.insert(3, 7, 1);
        t.insert(3, 7, 2);
        assert_eq!(t.len(), 2);
        let v = t.remove_first_match(3, 7);
        assert!(v == Some(1) || v == Some(2));
        assert_eq!(t.len(), 1);
    }

    #[test]
    fn remove_all_nodes_leaves_empty_tree() {
        let mut t: IntervalTree<i32, u32> = IntervalTree::new();
        for i in 0..5_i32 {
            t.insert(i, i + 2, i as u32);
        }
        for i in 0..5_i32 {
            assert!(t.remove_first_match(i, i + 2).is_some());
        }
        assert!(t.is_empty());
        assert!(t.query_point(1).is_empty());
    }

    // ---- deterministic build via fixed insertion order ---------------------

    #[test]
    fn deterministic_build_fixed_order() {
        // Same insertion sequence must yield the same query results every run.
        let mut t: IntervalTree<i32, u32> = IntervalTree::new();
        let intervals: Vec<(i32, i32, u32)> =
            vec![(1, 4, 10), (2, 6, 20), (3, 9, 30), (5, 5, 40), (7, 10, 50)];
        for (lo, hi, v) in &intervals {
            t.insert(*lo, *hi, *v);
        }
        assert_eq!(t.len(), 5);

        // Overlap [3, 5]: hits (1,4), (2,6), (3,9), (5,5) — NOT (7,10).
        let mut r: Vec<u32> = t.query_overlap(3, 5).iter().map(|v| **v).collect();
        r.sort_unstable();
        assert_eq!(r, vec![10, 20, 30, 40]);

        // Point 8: hits (3,9) and (7,10).
        let mut r2: Vec<u32> = t.query_point(8).iter().map(|v| **v).collect();
        r2.sort_unstable();
        assert_eq!(r2, vec![30, 50]);
    }

    // ---- quickcheck property: tree matches brute-force reference -----------

    /// Brute-force reference: a flat Vec of `((low, high), value)`.
    fn brute_overlap(store: &[((i32, i32), u32)], qlo: i32, qhi: i32) -> Vec<u32> {
        store
            .iter()
            .filter(|((lo, hi), _)| *lo <= qhi && qlo <= *hi)
            .map(|(_, v)| *v)
            .collect()
    }

    #[derive(Clone, Debug)]
    enum Op {
        Insert(i32, i32, u32),
        Remove(i32, i32),
        QueryOverlap(i32, i32),
    }

    impl quickcheck::Arbitrary for Op {
        fn arbitrary(g: &mut quickcheck::Gen) -> Self {
            let kind: u8 = u8::arbitrary(g) % 3;
            let a = i32::arbitrary(g) % 20;
            let b = i32::arbitrary(g) % 20;
            let lo = a.min(b);
            let hi = a.max(b);
            let v = u32::arbitrary(g) % 100;
            match kind {
                0 => Self::Insert(lo, hi, v),
                1 => Self::Remove(lo, hi),
                _ => Self::QueryOverlap(lo, hi),
            }
        }
    }

    #[allow(clippy::needless_pass_by_value)]
    #[quickcheck]
    fn prop_tree_matches_brute_force(raw_ops: Vec<Op>) -> TestResult {
        if raw_ops.len() > 60 {
            return TestResult::discard();
        }
        let mut tree: IntervalTree<i32, u32> = IntervalTree::new();
        let mut store: Vec<((i32, i32), u32)> = Vec::new();

        for op in raw_ops {
            match op {
                Op::Insert(lo, hi, v) => {
                    tree.insert(lo, hi, v);
                    store.push(((lo, hi), v));
                }
                Op::Remove(lo, hi) => {
                    let tree_val = tree.remove_first_match(lo, hi);
                    // Find and remove the first matching entry from the reference.
                    let pos = store.iter().position(|((l, h), _)| *l == lo && *h == hi);
                    let ref_val = pos.map(|i| store.remove(i).1);
                    if tree_val != ref_val {
                        return TestResult::failed();
                    }
                }
                Op::QueryOverlap(qlo, qhi) => {
                    let mut tree_vals: Vec<u32> =
                        tree.query_overlap(qlo, qhi).iter().map(|v| **v).collect();
                    let mut ref_vals = brute_overlap(&store, qlo, qhi);
                    tree_vals.sort_unstable();
                    ref_vals.sort_unstable();
                    if tree_vals != ref_vals {
                        return TestResult::failed();
                    }
                }
            }
        }
        TestResult::passed()
    }
}
