//! Persistent (functional) segment tree over `i64` sums.
//!
//! Each [`PersistentSegmentTree::update`] creates a **new root** that shares
//! all unmodified subtrees with the previous version (path copying). Old roots
//! remain fully valid and queryable. This gives a complete history of all
//! point-set operations at the cost of one extra tree path per update.
//!
//! # Complexity
//! - Build: `O(n)` time, `O(n)` nodes.
//! - Update: `O(log n)` time, `O(log n)` new nodes per call.
//! - Query: `O(log n)` time.
//! - Space: `O(n + q log n)` for `q` updates.
//!
//! # Preconditions
//! - `values` may be empty; all operations on an empty tree return `0`.
//! - `range_sum` uses a **half-open** interval `[l, r)` so `l == r` is valid
//!   (returns `0`).
//! - `idx` passed to `update` must be in `[0, n)` where `n` is the length of
//!   the slice passed to `new`.

/// A single node in the arena.
///
/// `left` and `right` are indices into `PersistentSegmentTree::nodes`.
/// A value of `usize::MAX` means "no child" (only true for leaves).
#[derive(Clone, Copy)]
struct Node {
    sum: i64,
    left: usize,
    right: usize,
}

impl Node {
    /// Leaf node with no children.
    const fn leaf(sum: i64) -> Self {
        Self {
            sum,
            left: usize::MAX,
            right: usize::MAX,
        }
    }

    /// Internal node combining two child indices.
    const fn internal(sum: i64, left: usize, right: usize) -> Self {
        Self { sum, left, right }
    }
}

/// Persistent segment tree over `i64` sums supporting point-set updates and
/// half-open range-sum queries.
///
/// All versions share unchanged subtree nodes in a single arena (`Vec<Node>`),
/// so historical roots stay valid after further updates.
pub struct PersistentSegmentTree {
    nodes: Vec<Node>,
    /// Number of elements in the logical array (fixed after construction).
    n: usize,
}

impl PersistentSegmentTree {
    /// Builds the initial tree from `values` and returns `(tree, root)`.
    ///
    /// The returned `root` is the index of the root node for version 0.
    /// Pass it to [`Self::update`] or [`Self::range_sum`].
    ///
    /// Empty `values` is allowed; the returned root is `usize::MAX` (sentinel)
    /// and all queries on it return `0`.
    pub fn new(values: &[i64]) -> (Self, usize) {
        let n = values.len();
        if n == 0 {
            return (
                Self {
                    nodes: Vec::new(),
                    n: 0,
                },
                usize::MAX,
            );
        }
        // Upper bound on nodes: a perfect binary tree over n leaves needs at
        // most 4*n nodes (standard safe upper bound). We pre-allocate to avoid
        // repeated reallocation during the recursive build.
        let mut tree = Self {
            nodes: Vec::with_capacity(4 * n),
            n,
        };
        let root = tree.build(values, 0, n - 1);
        (tree, root)
    }

    // -------------------------------------------------------------------------
    // Public API
    // -------------------------------------------------------------------------

    /// Returns the number of allocated nodes (arena size). Useful for testing
    /// the space complexity guarantee.
    pub const fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Returns `true` if the tree was built from an empty slice.
    pub const fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Sets position `idx` to `value`, creating a new persistent root.
    ///
    /// Returns the index of the **new** root; the old `root` is unchanged.
    ///
    /// # Panics
    /// Panics if `idx >= n` (where `n` is the length passed to [`Self::new`])
    /// or if the tree was built from an empty slice.
    pub fn update(&mut self, root: usize, idx: usize, value: i64) -> usize {
        assert!(
            self.n > 0,
            "PersistentSegmentTree::update: tree is empty (built from empty slice)"
        );
        assert!(
            idx < self.n,
            "PersistentSegmentTree::update: index {idx} out of bounds for n={}",
            self.n
        );
        self.update_inner(root, 0, self.n - 1, idx, value)
    }

    /// Returns the sum of `values[l..r]` (half-open interval) under `root`.
    ///
    /// Returns `0` for an empty range or if the tree was built from an empty
    /// slice. Silently clamps `r` to `n` if `r > n`.
    pub fn range_sum(&self, root: usize, l: usize, r: usize) -> i64 {
        if self.n == 0 || root == usize::MAX || l >= r {
            return 0;
        }
        let r_clamped = r.min(self.n);
        if l >= r_clamped {
            return 0;
        }
        self.query_inner(root, 0, self.n - 1, l, r_clamped - 1)
    }

    // -------------------------------------------------------------------------
    // Private helpers
    // -------------------------------------------------------------------------

    /// Recursively builds the initial tree; returns the index of the root node.
    fn build(&mut self, values: &[i64], lo: usize, hi: usize) -> usize {
        if lo == hi {
            let node = Node::leaf(values[lo]);
            return self.push(node);
        }
        let mid = lo.midpoint(hi);
        let left = self.build(values, lo, mid);
        let right = self.build(values, mid + 1, hi);
        let sum = self.nodes[left].sum + self.nodes[right].sum;
        let node = Node::internal(sum, left, right);
        self.push(node)
    }

    /// Creates a new version of the path from `node` down to leaf `idx`,
    /// setting that leaf to `value`. All other children are reused (shared).
    /// Returns the index of the new root node.
    fn update_inner(&mut self, node: usize, lo: usize, hi: usize, idx: usize, value: i64) -> usize {
        if lo == hi {
            // Leaf: create a brand-new leaf with the new value.
            let new_leaf = Node::leaf(value);
            return self.push(new_leaf);
        }
        let mid = lo.midpoint(hi);
        let (new_left, new_right) = if idx <= mid {
            let updated = self.update_inner(self.nodes[node].left, lo, mid, idx, value);
            (updated, self.nodes[node].right)
        } else {
            let updated = self.update_inner(self.nodes[node].right, mid + 1, hi, idx, value);
            (self.nodes[node].left, updated)
        };
        let sum = self.nodes[new_left].sum + self.nodes[new_right].sum;
        let new_node = Node::internal(sum, new_left, new_right);
        self.push(new_node)
    }

    /// Half-open `[l, r]` (inclusive) query under `node` covering `[lo, hi]`.
    fn query_inner(&self, node: usize, lo: usize, hi: usize, l: usize, r: usize) -> i64 {
        if l > hi || r < lo {
            return 0;
        }
        if l <= lo && hi <= r {
            return self.nodes[node].sum;
        }
        let mid = lo.midpoint(hi);
        self.query_inner(self.nodes[node].left, lo, mid, l, r)
            + self.query_inner(self.nodes[node].right, mid + 1, hi, l, r)
    }

    /// Appends a node to the arena and returns its index.
    fn push(&mut self, node: Node) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(node);
        idx
    }
}

#[cfg(test)]
mod tests {
    use super::PersistentSegmentTree;
    use quickcheck::TestResult;
    use quickcheck_macros::quickcheck;

    // -------------------------------------------------------------------------
    // Unit tests
    // -------------------------------------------------------------------------

    #[test]
    fn empty_input() {
        let (tree, root) = PersistentSegmentTree::new(&[]);
        assert!(tree.is_empty());
        assert_eq!(tree.len(), 0);
        // Any query on the empty tree must return 0.
        assert_eq!(tree.range_sum(root, 0, 0), 0);
        assert_eq!(tree.range_sum(root, 0, 100), 0);
    }

    #[test]
    fn single_element_query() {
        let (tree, root) = PersistentSegmentTree::new(&[42]);
        assert_eq!(tree.range_sum(root, 0, 1), 42);
        assert_eq!(tree.range_sum(root, 0, 0), 0); // [0,0) is empty
    }

    #[test]
    fn update_creates_new_root_old_unchanged() {
        let values = vec![1_i64, 2, 3, 4, 5];
        let (mut tree, root0) = PersistentSegmentTree::new(&values);

        // Update index 2 from 3 → 99, producing root1.
        let root1 = tree.update(root0, 2, 99);

        // root0 still reflects original values.
        assert_eq!(tree.range_sum(root0, 0, 5), 15);
        assert_eq!(tree.range_sum(root0, 2, 3), 3);

        // root1 reflects updated value.
        assert_eq!(tree.range_sum(root1, 0, 5), 15 - 3 + 99); // 111
        assert_eq!(tree.range_sum(root1, 2, 3), 99);
    }

    #[test]
    fn multiple_historical_roots_independently_queryable() {
        // Build v0 = [10, 20, 30, 40]
        // v1: set index 0 to 1
        // v2: set index 3 to 1
        // v3: set index 1 to 1 (from v0, not v2)
        let (mut tree, v0) = PersistentSegmentTree::new(&[10_i64, 20, 30, 40]);
        let v1 = tree.update(v0, 0, 1);
        let v2 = tree.update(v1, 3, 1);
        let v3 = tree.update(v0, 1, 1); // branch from v0, independent of v1/v2

        assert_eq!(tree.range_sum(v0, 0, 4), 100); // 10+20+30+40
        assert_eq!(tree.range_sum(v1, 0, 4), 91); //  1+20+30+40
        assert_eq!(tree.range_sum(v2, 0, 4), 52); //  1+20+30+ 1
        assert_eq!(tree.range_sum(v3, 0, 4), 81); // 10+ 1+30+40

        // Spot-check individual positions via single-element half-open queries.
        assert_eq!(tree.range_sum(v0, 0, 1), 10);
        assert_eq!(tree.range_sum(v1, 0, 1), 1);
        assert_eq!(tree.range_sum(v2, 3, 4), 1);
        assert_eq!(tree.range_sum(v3, 1, 2), 1);
    }

    #[test]
    fn range_sum_half_open_semantics() {
        // Verify [l, r) contract: [2, 5) covers indices 2,3,4.
        let (tree, root) = PersistentSegmentTree::new(&[1_i64, 2, 3, 4, 5, 6]);
        assert_eq!(tree.range_sum(root, 2, 5), 3 + 4 + 5); // 12
        assert_eq!(tree.range_sum(root, 0, 6), 21);
        assert_eq!(tree.range_sum(root, 3, 3), 0); // empty interval
    }

    #[test]
    fn successive_updates_same_index() {
        let (mut tree, root) = PersistentSegmentTree::new(&[0_i64; 4]);
        let r1 = tree.update(root, 1, 10);
        let r2 = tree.update(r1, 1, 20);
        let r3 = tree.update(r2, 1, 30);

        assert_eq!(tree.range_sum(root, 0, 4), 0);
        assert_eq!(tree.range_sum(r1, 1, 2), 10);
        assert_eq!(tree.range_sum(r2, 1, 2), 20);
        assert_eq!(tree.range_sum(r3, 1, 2), 30);
    }

    // -------------------------------------------------------------------------
    // Quickcheck property test
    // -------------------------------------------------------------------------

    /// Bounded value to keep cumulative sums well within i64.
    const BOUND: i64 = 1_000_000;

    /// Simulates a random sequence of point-set updates (branching from the
    /// previous version) and range-sum queries, validating every query result
    /// against a brute-force `Vec<i64>`.
    ///
    /// `ops` is a list of `(is_query, a, b, val)`.
    /// - `is_query == true`: query `range_sum(current_root, l, r)`.
    /// - `is_query == false`: set `arr[idx] = bounded(val)` and create a new
    ///   root that extends the linear history.
    #[quickcheck]
    #[allow(clippy::needless_pass_by_value)]
    fn prop_matches_brute_force(initial: Vec<i64>, ops: Vec<(bool, u8, u8, i64)>) -> TestResult {
        if initial.is_empty() || initial.len() > 64 {
            return TestResult::discard();
        }
        if ops.len() > 200 {
            return TestResult::discard();
        }

        let bounded_init: Vec<i64> = initial.iter().map(|v| v % BOUND).collect();
        let n = bounded_init.len();

        let mut reference = bounded_init.clone();
        let (mut tree, root0) = PersistentSegmentTree::new(&bounded_init);
        let mut current_root = root0;

        for &(is_query, a, b, val) in &ops {
            if is_query {
                let lo = (a as usize) % n;
                let hi_raw = (b as usize) % n;
                // Ensure [lo, hi) is a valid non-empty half-open interval.
                let (l, r) = match lo.cmp(&hi_raw) {
                    std::cmp::Ordering::Less => (lo, hi_raw),
                    std::cmp::Ordering::Greater => (hi_raw, lo),
                    std::cmp::Ordering::Equal => continue, // empty interval; skip
                };
                let expected: i64 = reference[l..r].iter().sum();
                let got = tree.range_sum(current_root, l, r);
                if got != expected {
                    return TestResult::failed();
                }
            } else {
                let idx = (a as usize) % n;
                let new_val = val % BOUND;
                reference[idx] = new_val;
                current_root = tree.update(current_root, idx, new_val);
            }
        }
        TestResult::passed()
    }
}
