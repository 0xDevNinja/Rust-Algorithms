//! K-D tree (Bentley 1975) specialised to 2-D nearest-neighbour queries.
//!
//! A **K-D tree** is a binary space-partitioning tree where each node splits
//! the point set along one axis at the median, alternating between the x-axis
//! (depth 0, 2, 4, …) and the y-axis (depth 1, 3, 5, …).
//!
//! This implementation is specialised to 2-D (`f64` coordinates) for clarity.
//! Generalising to k dimensions would replace the axis selection with
//! `depth % k` and the bounding-box with a k-length interval array.
//!
//! ## Complexity
//!
//! | Operation        | Time                         | Space  |
//! |------------------|------------------------------|--------|
//! | `build`          | O(n log n) †                 | O(n)   |
//! | `nearest`        | O(log n) avg, O(n) worst     | O(n)   |
//! | `k_nearest`      | O((k + log n) · log n) avg   | O(n)   |
//! | `within_radius`  | O(√n + m) avg, O(n) worst    | O(n)   |
//!
//! † Build uses `select_nth_unstable_by` for O(n) median selection per level,
//! giving O(n log n) total over the recursion tree. A naïve `sort_by` at each
//! level would be O(n log²n).
//!
//! ## Preconditions
//!
//! - Points may contain any finite `f64` values. NaN coordinates produce
//!   unspecified (but not unsafe) results.
//! - Duplicate points are allowed; all copies are stored and may all be
//!   returned by `within_radius` or `k_nearest`.

use std::collections::BinaryHeap;

/// A node inside the flat arena used by [`KdTree2D`].
///
/// Leaf nodes have `left == right == usize::MAX`.
#[derive(Clone)]
struct Node {
    /// Index into the original `points` slice stored in [`KdTree2D`].
    point_idx: usize,
    /// Split axis: `0` = x, `1` = y.
    axis: u8,
    /// Split value (coordinate of this node's point along `axis`).
    split: f64,
    /// Index into the arena of the left child, or `usize::MAX` if absent.
    left: usize,
    /// Index into the arena of the right child, or `usize::MAX` if absent.
    right: usize,
}

/// 2-D K-D tree for nearest-neighbour queries over `(f64, f64)` points.
///
/// Built via median-split on alternating axes (x then y). Internally stores
/// a flat `Vec<Node>` arena so no heap-allocated boxes appear in the tree
/// structure. Points are stored in a companion `Vec`; the tree holds only
/// indices into that `Vec`.
///
/// See the [module documentation](self) for complexity details.
pub struct KdTree2D {
    /// Stored points in their original insertion order.
    points: Vec<(f64, f64)>,
    /// Flat arena of tree nodes.
    nodes: Vec<Node>,
    /// Index of the root in `nodes`, or `usize::MAX` for an empty tree.
    root: usize,
}

// ---------------------------------------------------------------------------
// Helper: squared Euclidean distance (no sqrt, used for all comparisons).
// ---------------------------------------------------------------------------
#[inline]
fn dist2(a: (f64, f64), b: (f64, f64)) -> f64 {
    let dx = a.0 - b.0;
    let dy = a.1 - b.1;
    dx.mul_add(dx, dy * dy)
}

impl KdTree2D {
    // -----------------------------------------------------------------------
    // Build
    // -----------------------------------------------------------------------

    /// Builds a K-D tree from the given points in O(n log n) time.
    ///
    /// Points are cloned into the tree; the caller may drop the original `Vec`
    /// after this returns. An empty `Vec` produces an empty tree on which
    /// `nearest`, `k_nearest`, and `within_radius` each return `None`/empty.
    pub fn build(points: Vec<(f64, f64)>) -> Self {
        let n = points.len();
        // Pre-allocate the arena: a complete binary tree has at most 2n-1 nodes.
        let mut nodes: Vec<Node> = Vec::with_capacity(n);
        // Working index array partitioned in-place during build.
        let mut indices: Vec<usize> = (0..n).collect();

        let root = if n == 0 {
            usize::MAX
        } else {
            Self::build_recursive(&points, &mut indices, 0, n, 0, &mut nodes)
        };

        Self {
            points,
            nodes,
            root,
        }
    }

    /// Recursively partitions `indices[lo..hi]` and appends nodes to `arena`.
    /// Returns the arena index of the subtree root.
    fn build_recursive(
        points: &[(f64, f64)],
        indices: &mut [usize],
        lo: usize,
        hi: usize,
        depth: usize,
        arena: &mut Vec<Node>,
    ) -> usize {
        debug_assert!(lo < hi);
        let axis = (depth % 2) as u8;
        let slice = &mut indices[lo..hi];
        let len = slice.len();
        let mid = len / 2;

        // O(n) median selection: partial sort so that slice[mid] holds the
        // true median and elements before/after satisfy the partition property.
        slice.select_nth_unstable_by(mid, |&a, &b| {
            let va = if axis == 0 { points[a].0 } else { points[a].1 };
            let vb = if axis == 0 { points[b].0 } else { points[b].1 };
            va.total_cmp(&vb)
        });

        let point_idx = indices[lo + mid];
        let split = if axis == 0 {
            points[point_idx].0
        } else {
            points[point_idx].1
        };

        // Reserve a slot in the arena before recursing (avoids borrow issues).
        let node_idx = arena.len();
        arena.push(Node {
            point_idx,
            axis,
            split,
            left: usize::MAX,
            right: usize::MAX,
        });

        if mid > 0 {
            let left = Self::build_recursive(points, indices, lo, lo + mid, depth + 1, arena);
            arena[node_idx].left = left;
        }
        if mid + 1 < len {
            let right = Self::build_recursive(points, indices, lo + mid + 1, hi, depth + 1, arena);
            arena[node_idx].right = right;
        }

        node_idx
    }

    // -----------------------------------------------------------------------
    // Public queries
    // -----------------------------------------------------------------------

    /// Returns the closest stored point and its Euclidean distance from
    /// `query`, or `None` if the tree is empty.
    ///
    /// Ties are broken arbitrarily (whichever subtree is visited last).
    pub fn nearest(&self, query: (f64, f64)) -> Option<((f64, f64), f64)> {
        if self.root == usize::MAX {
            return None;
        }
        let mut best_sq = f64::INFINITY;
        let mut best_idx = 0;
        self.nearest_recursive(self.root, query, &mut best_sq, &mut best_idx);
        Some((self.points[best_idx], best_sq.sqrt()))
    }

    /// Returns the `k` closest stored points sorted by ascending Euclidean
    /// distance. Returns fewer than `k` entries if the tree contains fewer
    /// than `k` points.
    ///
    /// Uses a max-heap of size `k` internally; entries with larger distances
    /// are pruned from the heap as tighter candidates are found.
    pub fn k_nearest(&self, query: (f64, f64), k: usize) -> Vec<((f64, f64), f64)> {
        if self.root == usize::MAX || k == 0 {
            return Vec::new();
        }
        // Max-heap keyed by squared distance (f64 wrapped for Ord).
        // Each entry: (OrdF64(sq_dist), point_idx).
        let mut heap: BinaryHeap<(OrdF64, usize)> = BinaryHeap::with_capacity(k + 1);
        self.k_nearest_recursive(self.root, query, k, &mut heap);

        let mut result: Vec<((f64, f64), f64)> = heap
            .into_iter()
            .map(|(OrdF64(sq), idx)| (self.points[idx], sq.sqrt()))
            .collect();
        // Heap gives items in max-first order; reverse to ascending.
        result.sort_by(|a, b| a.1.total_cmp(&b.1));
        result
    }

    /// Returns every stored point whose Euclidean distance from `query` is
    /// ≤ `radius`, in unspecified order. For `radius = 0` only points whose
    /// distance rounds to exactly zero (i.e. identical coordinates) are
    /// returned.
    pub fn within_radius(&self, query: (f64, f64), radius: f64) -> Vec<((f64, f64), f64)> {
        if self.root == usize::MAX || radius < 0.0 {
            return Vec::new();
        }
        let radius_sq = radius * radius;
        let mut result = Vec::new();
        self.radius_recursive(self.root, query, radius_sq, &mut result);
        result
    }

    /// Number of points stored in the tree.
    pub const fn len(&self) -> usize {
        self.points.len()
    }

    /// Returns `true` if the tree contains no points.
    pub const fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    // -----------------------------------------------------------------------
    // Private recursive helpers
    // -----------------------------------------------------------------------

    fn nearest_recursive(
        &self,
        node_idx: usize,
        query: (f64, f64),
        best_sq: &mut f64,
        best_point_idx: &mut usize,
    ) {
        let node = &self.nodes[node_idx];
        let pt = self.points[node.point_idx];
        let d2 = dist2(pt, query);
        if d2 < *best_sq {
            *best_sq = d2;
            *best_point_idx = node.point_idx;
        }

        // Determine which side of the splitting plane the query falls on.
        let query_coord = if node.axis == 0 { query.0 } else { query.1 };
        let (near, far) = if query_coord <= node.split {
            (node.left, node.right)
        } else {
            (node.right, node.left)
        };

        // Descend into the near side first.
        if near != usize::MAX {
            self.nearest_recursive(near, query, best_sq, best_point_idx);
        }

        // Only visit the far side if the splitting plane is closer than the
        // current best (hyperrectangle pruning).
        let plane_dist_sq = (query_coord - node.split) * (query_coord - node.split);
        if far != usize::MAX && plane_dist_sq < *best_sq {
            self.nearest_recursive(far, query, best_sq, best_point_idx);
        }
    }

    fn k_nearest_recursive(
        &self,
        node_idx: usize,
        query: (f64, f64),
        k: usize,
        heap: &mut BinaryHeap<(OrdF64, usize)>,
    ) {
        let node = &self.nodes[node_idx];
        let pt = self.points[node.point_idx];
        let d2 = dist2(pt, query);

        // Add to heap; if over capacity, pop the worst (largest distance).
        heap.push((OrdF64(d2), node.point_idx));
        if heap.len() > k {
            heap.pop();
        }

        let query_coord = if node.axis == 0 { query.0 } else { query.1 };
        let (near, far) = if query_coord <= node.split {
            (node.left, node.right)
        } else {
            (node.right, node.left)
        };

        if near != usize::MAX {
            self.k_nearest_recursive(near, query, k, heap);
        }

        // Visit far side only if it could contain a closer point than the
        // worst currently in the heap.
        let plane_dist_sq = (query_coord - node.split) * (query_coord - node.split);
        let worst_sq = heap.peek().map_or(f64::INFINITY, |e| e.0 .0);
        if far != usize::MAX && (heap.len() < k || plane_dist_sq < worst_sq) {
            self.k_nearest_recursive(far, query, k, heap);
        }
    }

    fn radius_recursive(
        &self,
        node_idx: usize,
        query: (f64, f64),
        radius_sq: f64,
        result: &mut Vec<((f64, f64), f64)>,
    ) {
        let node = &self.nodes[node_idx];
        let pt = self.points[node.point_idx];
        let d2 = dist2(pt, query);
        if d2 <= radius_sq {
            result.push((pt, d2.sqrt()));
        }

        let query_coord = if node.axis == 0 { query.0 } else { query.1 };
        let plane_dist_sq = (query_coord - node.split) * (query_coord - node.split);

        if node.left != usize::MAX && (query_coord <= node.split || plane_dist_sq <= radius_sq) {
            self.radius_recursive(node.left, query, radius_sq, result);
        }
        if node.right != usize::MAX && (query_coord > node.split || plane_dist_sq <= radius_sq) {
            self.radius_recursive(node.right, query, radius_sq, result);
        }
    }
}

// ---------------------------------------------------------------------------
// OrdF64: a total-order wrapper around f64 for use in BinaryHeap.
// NaN is treated as greater than any finite value (worst possible distance).
// ---------------------------------------------------------------------------
#[derive(Clone, Copy, PartialEq)]
struct OrdF64(f64);

impl Eq for OrdF64 {}

impl PartialOrd for OrdF64 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrdF64 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::KdTree2D;
    use quickcheck::TestResult;
    use quickcheck_macros::quickcheck;

    const EPSILON: f64 = 1e-10;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < EPSILON
    }

    // -----------------------------------------------------------------------
    // Empty tree
    // -----------------------------------------------------------------------

    #[test]
    fn empty_nearest_is_none() {
        let tree = KdTree2D::build(vec![]);
        assert!(tree.nearest((0.0, 0.0)).is_none());
    }

    #[test]
    fn empty_k_nearest_is_empty() {
        let tree = KdTree2D::build(vec![]);
        assert!(tree.k_nearest((0.0, 0.0), 3).is_empty());
    }

    #[test]
    fn empty_within_radius_is_empty() {
        let tree = KdTree2D::build(vec![]);
        assert!(tree.within_radius((0.0, 0.0), 1.0).is_empty());
    }

    // -----------------------------------------------------------------------
    // Single point
    // -----------------------------------------------------------------------

    #[test]
    fn single_point_nearest() {
        let tree = KdTree2D::build(vec![(3.0, 4.0)]);
        let (pt, d) = tree.nearest((0.0, 0.0)).unwrap();
        assert_eq!(pt, (3.0, 4.0));
        assert!(approx_eq(d, 5.0)); // sqrt(9+16)
    }

    #[test]
    fn single_point_k_nearest() {
        let tree = KdTree2D::build(vec![(1.0, 1.0)]);
        let res = tree.k_nearest((0.0, 0.0), 5);
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].0, (1.0, 1.0));
    }

    #[test]
    fn single_point_within_radius() {
        let tree = KdTree2D::build(vec![(3.0, 4.0)]);
        // Exactly on the boundary (distance == 5).
        let res = tree.within_radius((0.0, 0.0), 5.0);
        assert_eq!(res.len(), 1);
        // Outside.
        let res2 = tree.within_radius((0.0, 0.0), 4.9);
        assert!(res2.is_empty());
    }

    // -----------------------------------------------------------------------
    // Two collinear points: nearest is correct when query is between them.
    // -----------------------------------------------------------------------

    #[test]
    fn two_collinear_nearest() {
        // Points at x = -1 and x = 3; query at x = 1 (closer to x = -1 by 2,
        // vs x = 3 by 2 — equidistant). Move query to x = 0: dist to -1 is 1,
        // dist to 3 is 3; nearest should be (-1, 0).
        let tree = KdTree2D::build(vec![(-1.0, 0.0), (3.0, 0.0)]);
        let (pt, d) = tree.nearest((0.0, 0.0)).unwrap();
        assert_eq!(pt, (-1.0, 0.0));
        assert!(approx_eq(d, 1.0));
    }

    // -----------------------------------------------------------------------
    // Grid of points
    // -----------------------------------------------------------------------

    #[test]
    fn grid_nearest_matches_brute_force() {
        // 5x5 integer grid: points (i, j) for i,j in 0..5.
        let pts: Vec<(f64, f64)> = (0..5_i32)
            .flat_map(|i| (0..5_i32).map(move |j| (f64::from(i), f64::from(j))))
            .collect();
        let tree = KdTree2D::build(pts.clone());
        // Queries that each have a unique nearest neighbour (no ties).
        let queries = [(1.3, 2.7), (0.1, 0.1), (4.9, 4.9), (-0.5, -0.5)];
        for q in queries {
            let (tree_pt, tree_d) = tree.nearest(q).unwrap();
            let brute_min_d2 = pts
                .iter()
                .copied()
                .map(|p| {
                    let dx = p.0 - q.0;
                    let dy = p.1 - q.1;
                    dx.mul_add(dx, dy * dy)
                })
                .min_by(f64::total_cmp)
                .unwrap();
            // The returned point must actually achieve the minimum distance.
            let tree_d2 = {
                let dx = tree_pt.0 - q.0;
                let dy = tree_pt.1 - q.1;
                dx.mul_add(dx, dy * dy)
            };
            assert!(
                approx_eq(tree_d2, brute_min_d2),
                "nearest distance mismatch for query {q:?}: tree_d2={tree_d2} brute_min_d2={brute_min_d2}"
            );
            assert!(
                approx_eq(tree_d, brute_min_d2.sqrt()),
                "reported distance mismatch for query {q:?}: tree_d={tree_d} brute={}",
                brute_min_d2.sqrt()
            );
        }
    }

    // -----------------------------------------------------------------------
    // k_nearest with k == len() returns all points sorted by distance.
    // -----------------------------------------------------------------------

    #[test]
    fn k_nearest_all_points_sorted() {
        let pts: Vec<(f64, f64)> = vec![(1.0, 0.0), (2.0, 0.0), (0.5, 0.0), (3.0, 0.0)];
        let tree = KdTree2D::build(pts.clone());
        let result = tree.k_nearest((0.0, 0.0), pts.len());
        assert_eq!(result.len(), pts.len());
        // Distances must be non-decreasing.
        for w in result.windows(2) {
            assert!(
                w[0].1 <= w[1].1 + EPSILON,
                "k_nearest result not sorted: {result:?}"
            );
        }
        // Distances must match brute-force.
        let mut brute: Vec<f64> = pts.iter().map(|&p| p.0.hypot(p.1)).collect();
        brute.sort_by(f64::total_cmp);
        for (r, b) in result.iter().zip(brute.iter()) {
            assert!(approx_eq(r.1, *b));
        }
    }

    // -----------------------------------------------------------------------
    // within_radius with radius = 0 returns only exact matches.
    // -----------------------------------------------------------------------

    #[test]
    fn within_radius_zero_exact_matches_only() {
        let pts = vec![(1.0, 1.0), (2.0, 2.0), (1.0, 1.0)];
        let tree = KdTree2D::build(pts);
        let res = tree.within_radius((1.0, 1.0), 0.0);
        // Both copies of (1,1) should be returned; (2,2) must not.
        assert_eq!(res.len(), 2, "expected 2 exact matches, got {res:?}");
        for (pt, d) in &res {
            assert_eq!(*pt, (1.0, 1.0));
            assert!(approx_eq(*d, 0.0));
        }
    }

    // -----------------------------------------------------------------------
    // within_radius with very large radius returns all points.
    // -----------------------------------------------------------------------

    #[test]
    fn within_radius_large_returns_all() {
        let pts: Vec<(f64, f64)> = (0..20).map(|i| (f64::from(i), f64::from(i))).collect();
        let n = pts.len();
        let tree = KdTree2D::build(pts);
        let res = tree.within_radius((0.0, 0.0), 1e9);
        assert_eq!(res.len(), n);
    }

    // -----------------------------------------------------------------------
    // Query coincident with a stored point returns distance 0.
    // -----------------------------------------------------------------------

    #[test]
    fn query_coincident_returns_zero_distance() {
        let pts = vec![(1.5, 2.5), (3.0, 4.0), (-1.0, -1.0)];
        let tree = KdTree2D::build(pts);
        let (pt, d) = tree.nearest((3.0, 4.0)).unwrap();
        assert_eq!(pt, (3.0, 4.0));
        assert!(approx_eq(d, 0.0));
    }

    // -----------------------------------------------------------------------
    // len / is_empty
    // -----------------------------------------------------------------------

    #[test]
    fn len_and_is_empty() {
        let empty = KdTree2D::build(vec![]);
        assert!(empty.is_empty());
        assert_eq!(empty.len(), 0);

        let tree = KdTree2D::build(vec![(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)]);
        assert!(!tree.is_empty());
        assert_eq!(tree.len(), 3);
    }

    // -----------------------------------------------------------------------
    // Quickcheck property: nearest matches brute-force on random inputs.
    // -----------------------------------------------------------------------

    /// Brute-force nearest-neighbour for reference.
    fn brute_nearest(pts: &[(f64, f64)], q: (f64, f64)) -> (f64, f64) {
        pts.iter()
            .copied()
            .min_by(|&a, &b| {
                let da = (a.0 - q.0).mul_add(a.0 - q.0, (a.1 - q.1) * (a.1 - q.1));
                let db = (b.0 - q.0).mul_add(b.0 - q.0, (b.1 - q.1) * (b.1 - q.1));
                da.total_cmp(&db)
            })
            .unwrap()
    }

    #[quickcheck]
    #[allow(clippy::needless_pass_by_value)]
    fn prop_nearest_matches_brute_force(raw_pts: Vec<(i16, i16)>, raw_q: (i16, i16)) -> TestResult {
        // Discard degenerate inputs.
        if raw_pts.is_empty() || raw_pts.len() > 100 {
            return TestResult::discard();
        }
        // Convert i16 coordinates to f64 so they are always finite.
        let pts: Vec<(f64, f64)> = raw_pts
            .iter()
            .map(|&(x, y)| (f64::from(x), f64::from(y)))
            .collect();
        let q = (f64::from(raw_q.0), f64::from(raw_q.1));

        let tree = KdTree2D::build(pts.clone());
        let Some((tree_pt, _)) = tree.nearest(q) else {
            return TestResult::failed();
        };
        let brute_pt = brute_nearest(&pts, q);

        // It is possible that multiple points share the minimum distance;
        // compare squared distances rather than coordinates.
        let tree_d2 =
            (tree_pt.0 - q.0).mul_add(tree_pt.0 - q.0, (tree_pt.1 - q.1) * (tree_pt.1 - q.1));
        let brute_d2 =
            (brute_pt.0 - q.0).mul_add(brute_pt.0 - q.0, (brute_pt.1 - q.1) * (brute_pt.1 - q.1));

        TestResult::from_bool(tree_d2.total_cmp(&brute_d2).is_eq())
    }
}
