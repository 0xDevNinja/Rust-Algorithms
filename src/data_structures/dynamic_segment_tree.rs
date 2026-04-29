//! Dynamic (implicit) segment tree over a virtual integer range `[range_lo, range_hi)`.
//!
//! Unlike a static segment tree, nodes are allocated lazily on demand. Only the
//! paths touched by update operations exist in memory, making this structure ideal
//! for sparse coordinate spaces where coordinate compression is unavailable (e.g.
//! online problems where query bounds arrive one at a time).
//!
//! # Complexity
//! - Update (`point_add`, `point_set`): O(log R) time, O(log R) new nodes
//! - Query (`range_sum`): O(log R) time
//! - Memory: O(updates · log R) total nodes
//!
//! where `R = range_hi - range_lo`.
//!
//! # Specialisation
//! This implementation uses `i64` sum aggregation. The identity element is `0`.

/// A single node in the arena-allocated tree.
struct Node {
    left: Option<usize>,
    right: Option<usize>,
    sum: i64,
}

impl Node {
    const fn new() -> Self {
        Self {
            left: None,
            right: None,
            sum: 0,
        }
    }
}

/// Dynamic (implicit) segment tree supporting point updates and range-sum queries
/// over a half-open integer range `[range_lo, range_hi)`.
///
/// Nodes are stored in a flat `Vec<Node>` arena and indexed by `usize` so that
/// no `unsafe` code is required. Root is always at index `0`.
pub struct DynamicSegmentTree {
    nodes: Vec<Node>,
    range_lo: i64,
    range_hi: i64,
}

impl DynamicSegmentTree {
    /// Creates a new empty tree covering the half-open range `[range_lo, range_hi)`.
    ///
    /// # Panics
    /// Panics if `range_lo >= range_hi`.
    pub fn new(range_lo: i64, range_hi: i64) -> Self {
        assert!(
            range_lo < range_hi,
            "range_lo must be strictly less than range_hi"
        );
        // Pre-allocate the root node (index 0).
        Self {
            nodes: vec![Node::new()],
            range_lo,
            range_hi,
        }
    }

    /// Returns the number of allocated nodes. Each update allocates at most
    /// O(log R) new nodes, so this is a proxy for memory usage.
    pub const fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Returns `true` if no nodes beyond the root have been allocated (no updates
    /// have been applied).
    pub fn is_empty(&self) -> bool {
        // The root always exists; "empty" from the user perspective means sum == 0
        // and no children allocated, i.e. only the root node exists.
        self.nodes.len() == 1 && self.nodes[0].sum == 0
    }

    /// Allocates a new node in the arena and returns its index.
    fn alloc(&mut self) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(Node::new());
        idx
    }

    /// Ensures the left child of `node_idx` exists and returns its index.
    fn ensure_left(&mut self, node_idx: usize) -> usize {
        if let Some(idx) = self.nodes[node_idx].left {
            idx
        } else {
            let idx = self.alloc();
            self.nodes[node_idx].left = Some(idx);
            idx
        }
    }

    /// Ensures the right child of `node_idx` exists and returns its index.
    fn ensure_right(&mut self, node_idx: usize) -> usize {
        if let Some(idx) = self.nodes[node_idx].right {
            idx
        } else {
            let idx = self.alloc();
            self.nodes[node_idx].right = Some(idx);
            idx
        }
    }

    /// Adds `delta` to the element at `index`.
    ///
    /// # Panics
    /// Panics if `index` is outside `[range_lo, range_hi)`.
    pub fn point_add(&mut self, index: i64, delta: i64) {
        assert!(
            index >= self.range_lo && index < self.range_hi,
            "index {index} is outside [{}, {})",
            self.range_lo,
            self.range_hi
        );
        self.add_inner(0, self.range_lo, self.range_hi, index, delta);
    }

    fn add_inner(&mut self, node_idx: usize, lo: i64, hi: i64, index: i64, delta: i64) {
        self.nodes[node_idx].sum += delta;
        if hi - lo <= 1 {
            // Leaf node — nothing further to descend into.
            return;
        }
        let mid = lo + (hi - lo) / 2;
        if index < mid {
            let child = self.ensure_left(node_idx);
            self.add_inner(child, lo, mid, index, delta);
        } else {
            let child = self.ensure_right(node_idx);
            self.add_inner(child, mid, hi, index, delta);
        }
    }

    /// Returns the sum of all elements in the half-open range `[lo, hi)`.
    ///
    /// Query bounds are silently clamped to `[range_lo, range_hi)`, so callers
    /// do not need to guard against out-of-range queries.
    pub fn range_sum(&self, lo: i64, hi: i64) -> i64 {
        let lo = lo.max(self.range_lo);
        let hi = hi.min(self.range_hi);
        if lo >= hi {
            return 0;
        }
        self.query_inner(0, self.range_lo, self.range_hi, lo, hi)
    }

    fn query_inner(&self, node_idx: usize, node_lo: i64, node_hi: i64, lo: i64, hi: i64) -> i64 {
        // Query range does not overlap this node's range.
        if lo >= node_hi || hi <= node_lo {
            return 0;
        }
        // Query range completely covers this node's range.
        if lo <= node_lo && node_hi <= hi {
            return self.nodes[node_idx].sum;
        }
        let mid = node_lo + (node_hi - node_lo) / 2;
        let left_sum = self.nodes[node_idx]
            .left
            .map_or(0, |child| self.query_inner(child, node_lo, mid, lo, hi));
        let right_sum = self.nodes[node_idx]
            .right
            .map_or(0, |child| self.query_inner(child, mid, node_hi, lo, hi));
        left_sum + right_sum
    }

    /// Reads the current value stored at a single point `index`.
    ///
    /// This is a pure traversal — it does not allocate any nodes.
    fn point_get(&self, index: i64) -> i64 {
        let mut node_idx = 0usize;
        let mut lo = self.range_lo;
        let mut hi = self.range_hi;
        loop {
            if hi - lo <= 1 {
                return self.nodes[node_idx].sum;
            }
            let mid = lo + (hi - lo) / 2;
            if index < mid {
                match self.nodes[node_idx].left {
                    Some(child) => {
                        node_idx = child;
                        hi = mid;
                    }
                    None => return 0,
                }
            } else {
                match self.nodes[node_idx].right {
                    Some(child) => {
                        node_idx = child;
                        lo = mid;
                    }
                    None => return 0,
                }
            }
        }
    }

    /// Sets the element at `index` to `value`.
    ///
    /// Computes `delta = value - current_value` and delegates to `point_add`.
    ///
    /// # Panics
    /// Panics if `index` is outside `[range_lo, range_hi)`.
    pub fn point_set(&mut self, index: i64, value: i64) {
        assert!(
            index >= self.range_lo && index < self.range_hi,
            "index {index} is outside [{}, {})",
            self.range_lo,
            self.range_hi
        );
        let current = self.point_get(index);
        let delta = value - current;
        if delta != 0 {
            self.point_add(index, delta);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::DynamicSegmentTree;
    use quickcheck_macros::quickcheck;
    use std::collections::BTreeMap;

    // ---- unit tests --------------------------------------------------------

    #[test]
    fn empty_tree_all_zero() {
        let t = DynamicSegmentTree::new(0, 1_000_000_000);
        assert_eq!(t.range_sum(0, 1_000_000_000), 0);
        assert_eq!(t.range_sum(42, 100), 0);
        assert_eq!(t.range_sum(0, 1), 0);
    }

    #[test]
    fn single_point_add_then_range_sum() {
        let mut t = DynamicSegmentTree::new(0, 100);
        t.point_add(50, 7);
        assert_eq!(t.range_sum(0, 100), 7);
        assert_eq!(t.range_sum(50, 51), 7);
        assert_eq!(t.range_sum(0, 50), 0);
        assert_eq!(t.range_sum(51, 100), 0);
    }

    #[test]
    fn range_sum_fully_outside_updates_is_zero() {
        let mut t = DynamicSegmentTree::new(0, 1000);
        t.point_add(500, 10);
        t.point_add(501, 20);
        // Range entirely below the updates.
        assert_eq!(t.range_sum(0, 500), 0);
        // Range entirely above the updates.
        assert_eq!(t.range_sum(502, 1000), 0);
    }

    #[test]
    fn range_sum_partial_overlap() {
        let mut t = DynamicSegmentTree::new(0, 1000);
        t.point_add(100, 1);
        t.point_add(200, 2);
        t.point_add(300, 3);
        // Covers only positions 200 and 300.
        assert_eq!(t.range_sum(150, 350), 5);
        // Covers only position 100.
        assert_eq!(t.range_sum(0, 150), 1);
        // Covers all three.
        assert_eq!(t.range_sum(0, 1000), 6);
    }

    #[test]
    fn large_range_sparse_updates_small_footprint() {
        // Range [0, 1e9). log2(1e9) ≈ 30; each update adds ≤ 30 nodes.
        // 100 updates × 30 nodes = 3000 worst case; we verify it stays clearly
        // sub-linear compared to the full range size (1_000_000_000 nodes).
        let mut t = DynamicSegmentTree::new(0, 1_000_000_000);
        for i in 0..100_i64 {
            t.point_add(i * 9_999_983, i + 1); // pseudo-random spread
        }
        // O(updates * log R) growth: at most ~3100 nodes total for 100 updates
        // over a range of 1e9. This is vastly smaller than the 1e9 nodes a
        // static tree would require.
        assert!(
            t.len() < 3200,
            "Expected < 3200 nodes for 100 updates, got {}",
            t.len()
        );
        // Basic correctness: sum of 1..=100.
        assert_eq!(t.range_sum(0, 1_000_000_000), 5050);
    }

    #[test]
    fn point_set_twice_leaves_latest_value() {
        let mut t = DynamicSegmentTree::new(0, 100);
        t.point_set(42, 100);
        assert_eq!(t.range_sum(42, 43), 100);
        t.point_set(42, 7);
        assert_eq!(t.range_sum(42, 43), 7);
        // Total should only reflect the final value.
        assert_eq!(t.range_sum(0, 100), 7);
    }

    #[test]
    fn range_sum_clamps_outside_tree_bounds() {
        let mut t = DynamicSegmentTree::new(10, 90);
        t.point_add(50, 99);
        // lo below range_lo and hi above range_hi — should clamp silently.
        assert_eq!(t.range_sum(-1_000, 200_000), 99);
        // Partially outside on the low side.
        assert_eq!(t.range_sum(0, 60), 99);
        // Partially outside on the high side.
        assert_eq!(t.range_sum(40, 10_000), 99);
    }

    #[test]
    fn multiple_adds_same_point() {
        let mut t = DynamicSegmentTree::new(0, 50);
        t.point_add(10, 3);
        t.point_add(10, 4);
        t.point_add(10, -2);
        assert_eq!(t.range_sum(10, 11), 5);
    }

    // ---- quickcheck property test ------------------------------------------

    /// Model: a `BTreeMap<i64, i64>` tracking the current point values.
    /// A random mix of `point_add` and `range_sum` operations over `[0, 10000)`
    /// must yield identical results between the dynamic tree and the model.
    #[quickcheck]
    fn matches_btreemap_model(ops: Vec<(bool, i64, i64)>) -> bool {
        const RANGE: i64 = 10_000;
        let mut tree = DynamicSegmentTree::new(0, RANGE);
        let mut model: BTreeMap<i64, i64> = BTreeMap::new();

        for (is_add, a, b) in ops {
            // rem_euclid maps any i64 (including MIN) into [0, RANGE) safely.
            let idx = a.rem_euclid(RANGE);
            let val = b % 1_000_000; // keep values bounded to avoid i64 overflow

            if is_add {
                // point_add
                tree.point_add(idx, val);
                *model.entry(idx).or_insert(0) += val;
            } else {
                // range_sum over [lo, hi)
                let lo = idx;
                let hi = (b.rem_euclid(RANGE) + 1).max(lo + 1).min(RANGE);
                let tree_sum = tree.range_sum(lo, hi);
                let model_sum: i64 = model.range(lo..hi).map(|(_, v)| v).sum();
                if tree_sum != model_sum {
                    return false;
                }
            }
        }
        true
    }
}
