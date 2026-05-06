//! Wavelet tree over an `i64` slice.
//!
//! A wavelet tree partitions the value alphabet recursively by the midpoint of
//! the current value range, mirroring a segment tree over values rather than
//! positions. Each internal node stores a dense bit-vector that records, for
//! every position, whether its value was routed to the left (smaller) child.
//! Prefix sums on that bit-vector allow `O(log σ)` range queries.
//!
//! - **Build**: `O(n log σ)` time and space, where `σ` is the number of
//!   distinct values (bounded by `n` after coordinate compression).
//! - **Queries** ([`WaveletTree::kth_smallest`], [`WaveletTree::rank`],
//!   [`WaveletTree::count_less`]): each `O(log σ)`.
//! - **Preconditions**: all queries require `l < r <= len()` and, for
//!   `kth_smallest`, `k < r - l`.
//!
//! The alphabet is split at the numerical midpoint of the live value range, so
//! tree depth is `O(log(max_value - min_value))` in the worst case. For skewed
//! distributions this may exceed `log n`; if that is a concern, coordinate-
//! compress the input first.

/// Dense bit-vector with prefix-sum cache for a single wavelet tree level.
///
/// `prefix[i]` stores the count of positions in `0..i` that were routed to the
/// **left** child (value ≤ mid) at this level.
struct BitLevel {
    prefix: Vec<usize>,
}

impl BitLevel {
    fn new(n: usize) -> Self {
        Self {
            prefix: vec![0; n + 1],
        }
    }

    /// Record that position `i` goes left (`true`) or right (`false`).
    fn set(&mut self, i: usize, goes_left: bool) {
        self.prefix[i + 1] = self.prefix[i] + usize::from(goes_left);
    }

    /// Number of left-going elements in `[0, i)`.
    fn count_left(&self, i: usize) -> usize {
        self.prefix[i]
    }

    /// Number of right-going elements in `[0, i)`.
    fn count_right(&self, i: usize) -> usize {
        i - self.prefix[i]
    }
}

/// A node in the wavelet tree.
///
/// Leaf nodes (lo == hi) carry no bit-level data.
enum Node {
    Leaf,
    Internal {
        mid: i64,
        level: BitLevel,
        left: Box<Self>,
        right: Box<Self>,
    },
}

impl Node {
    /// Recursively build the wavelet tree over `values` with alphabet `[lo, hi]`.
    fn build(values: &[i64], lo: i64, hi: i64) -> Self {
        if lo == hi || values.is_empty() {
            return Self::Leaf;
        }

        // Midpoint split: lo + (hi - lo) / 2  — avoids signed overflow.
        let mid = lo + (hi - lo) / 2;

        let n = values.len();
        let mut level = BitLevel::new(n);
        let mut left_vals: Vec<i64> = Vec::with_capacity(n);
        let mut right_vals: Vec<i64> = Vec::with_capacity(n);

        for (i, &v) in values.iter().enumerate() {
            let goes_left = v <= mid;
            level.set(i, goes_left);
            if goes_left {
                left_vals.push(v);
            } else {
                right_vals.push(v);
            }
        }

        let left = Box::new(Self::build(&left_vals, lo, mid));
        let right = Box::new(Self::build(&right_vals, mid + 1, hi));

        Self::Internal {
            mid,
            level,
            left,
            right,
        }
    }

    /// Returns the `k`-th smallest value (0-indexed) in position range `[l, r)`.
    fn kth_smallest(&self, l: usize, r: usize, mut k: usize, lo: i64, hi: i64) -> i64 {
        if lo == hi {
            return lo;
        }

        let Self::Internal {
            mid,
            level,
            left,
            right,
        } = self
        else {
            panic!("WaveletTree: internal inconsistency — leaf with lo != hi");
        };

        let left_count = level.count_left(r) - level.count_left(l);

        if k < left_count {
            let new_l = level.count_left(l);
            let new_r = level.count_left(r);
            left.kth_smallest(new_l, new_r, k, lo, *mid)
        } else {
            k -= left_count;
            let new_l = level.count_right(l);
            let new_r = level.count_right(r);
            right.kth_smallest(new_l, new_r, k, *mid + 1, hi)
        }
    }

    /// Counts how many values in position range `[l, r)` equal `value`.
    fn rank(&self, l: usize, r: usize, value: i64, lo: i64, hi: i64) -> usize {
        if lo == hi {
            return r - l;
        }

        let Self::Internal {
            mid,
            level,
            left,
            right,
        } = self
        else {
            return 0;
        };

        if value <= *mid {
            let new_l = level.count_left(l);
            let new_r = level.count_left(r);
            left.rank(new_l, new_r, value, lo, *mid)
        } else {
            let new_l = level.count_right(l);
            let new_r = level.count_right(r);
            right.rank(new_l, new_r, value, *mid + 1, hi)
        }
    }

    /// Counts how many values in position range `[l, r)` are strictly less than `value`.
    fn count_less(&self, l: usize, r: usize, value: i64, lo: i64, hi: i64) -> usize {
        if l == r {
            return 0;
        }
        // All elements in this node are >= lo; none qualify.
        if value <= lo {
            return 0;
        }
        // All elements in this node are <= hi < value; all qualify.
        if value > hi {
            return r - l;
        }
        // At this point lo < value <= hi, so lo != hi, meaning this is internal.
        let Self::Internal {
            mid,
            level,
            left,
            right,
        } = self
        else {
            return 0;
        };

        // Elements routed left have values in [lo, mid].
        let left_count = level.count_left(r) - level.count_left(l);

        if value <= *mid + 1 {
            // All sought elements are within the left subtree.
            let new_l = level.count_left(l);
            let new_r = level.count_left(r);
            left.count_less(new_l, new_r, value, lo, *mid)
        } else {
            // Everything routed left is < mid+1 <= value, so add left_count
            // then recurse into the right subtree for the rest.
            let new_l = level.count_right(l);
            let new_r = level.count_right(r);
            left_count + right.count_less(new_l, new_r, value, *mid + 1, hi)
        }
    }
}

/// Wavelet tree supporting range `kth_smallest`, `rank`, and `count_less`
/// queries in `O(log σ)` per query after `O(n log σ)` construction.
///
/// `σ` is bounded by the span of distinct values in the input (or `n` after
/// coordinate compression). All position ranges are **half-open** `[l, r)`.
pub struct WaveletTree {
    n: usize,
    root: Node,
    min_val: i64,
    max_val: i64,
}

impl WaveletTree {
    /// Builds a wavelet tree from `values` in `O(n log σ)` time and space.
    ///
    /// An empty slice is accepted; subsequent indexed queries will panic.
    pub fn new(values: &[i64]) -> Self {
        let n = values.len();
        if n == 0 {
            return Self {
                n: 0,
                root: Node::Leaf,
                min_val: 0,
                max_val: 0,
            };
        }
        let min_val = *values.iter().min().expect("non-empty");
        let max_val = *values.iter().max().expect("non-empty");
        let root = Node::build(values, min_val, max_val);
        Self {
            n,
            root,
            min_val,
            max_val,
        }
    }

    /// Number of elements in the tree.
    pub const fn len(&self) -> usize {
        self.n
    }

    /// Returns `true` if the tree was built from an empty slice.
    pub const fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Returns the `k`-th smallest value (0-indexed) in the half-open
    /// position range `[l, r)` in `O(log σ)`.
    ///
    /// # Panics
    /// Panics if the tree is empty, if `l >= r`, if `r > len()`, or if
    /// `k >= r - l`.
    pub fn kth_smallest(&self, l: usize, r: usize, k: usize) -> i64 {
        assert!(!self.is_empty(), "WaveletTree::kth_smallest: empty tree");
        assert!(l < r, "WaveletTree::kth_smallest: empty range [{l}, {r})");
        assert!(
            r <= self.n,
            "WaveletTree::kth_smallest: range [{l}, {r}) out of bounds for len {}",
            self.n
        );
        assert!(
            k < r - l,
            "WaveletTree::kth_smallest: k={k} out of range for window size {}",
            r - l
        );
        self.root.kth_smallest(l, r, k, self.min_val, self.max_val)
    }

    /// Returns the number of occurrences of `value` in the half-open position
    /// range `[l, r)` in `O(log σ)`.
    ///
    /// Returns 0 if `value` was not present in the original input.
    ///
    /// # Panics
    /// Panics if `l > r` or `r > len()`.
    pub fn rank(&self, l: usize, r: usize, value: i64) -> usize {
        assert!(l <= r, "WaveletTree::rank: invalid range [{l}, {r})");
        assert!(
            r <= self.n,
            "WaveletTree::rank: range [{l}, {r}) out of bounds for len {}",
            self.n
        );
        if l == r || self.is_empty() {
            return 0;
        }
        if value < self.min_val || value > self.max_val {
            return 0;
        }
        self.root.rank(l, r, value, self.min_val, self.max_val)
    }

    /// Returns the count of values in the half-open position range `[l, r)`
    /// that are strictly less than `value`, in `O(log σ)`.
    ///
    /// # Panics
    /// Panics if `l > r` or `r > len()`.
    pub fn count_less(&self, l: usize, r: usize, value: i64) -> usize {
        assert!(l <= r, "WaveletTree::count_less: invalid range [{l}, {r})");
        assert!(
            r <= self.n,
            "WaveletTree::count_less: range [{l}, {r}) out of bounds for len {}",
            self.n
        );
        if l == r || self.is_empty() {
            return 0;
        }
        self.root
            .count_less(l, r, value, self.min_val, self.max_val)
    }
}

#[cfg(test)]
mod tests {
    use super::WaveletTree;
    use quickcheck::TestResult;
    use quickcheck_macros::quickcheck;

    // ── helpers ──────────────────────────────────────────────────────────────

    fn brute_kth(values: &[i64], l: usize, r: usize, k: usize) -> i64 {
        let mut window: Vec<i64> = values[l..r].to_vec();
        window.sort_unstable();
        window[k]
    }

    fn brute_rank(values: &[i64], l: usize, r: usize, value: i64) -> usize {
        values[l..r].iter().filter(|&&v| v == value).count()
    }

    fn brute_count_less(values: &[i64], l: usize, r: usize, value: i64) -> usize {
        values[l..r].iter().filter(|&&v| v < value).count()
    }

    // ── structural ───────────────────────────────────────────────────────────

    #[test]
    fn empty_tree_reports_empty() {
        let wt = WaveletTree::new(&[]);
        assert!(wt.is_empty());
        assert_eq!(wt.len(), 0);
    }

    #[test]
    fn empty_rank_returns_zero() {
        let wt = WaveletTree::new(&[]);
        assert_eq!(wt.rank(0, 0, 42), 0);
    }

    #[test]
    fn empty_count_less_returns_zero() {
        let wt = WaveletTree::new(&[]);
        assert_eq!(wt.count_less(0, 0, 42), 0);
    }

    #[test]
    #[should_panic(expected = "empty tree")]
    fn kth_on_empty_panics() {
        let wt = WaveletTree::new(&[]);
        let _ = wt.kth_smallest(0, 1, 0);
    }

    // ── single element ───────────────────────────────────────────────────────

    #[test]
    fn single_element_kth() {
        let wt = WaveletTree::new(&[7]);
        assert_eq!(wt.kth_smallest(0, 1, 0), 7);
    }

    #[test]
    fn single_element_rank() {
        let wt = WaveletTree::new(&[7]);
        assert_eq!(wt.rank(0, 1, 7), 1);
        assert_eq!(wt.rank(0, 1, 5), 0);
    }

    #[test]
    fn single_element_count_less() {
        let wt = WaveletTree::new(&[7]);
        assert_eq!(wt.count_less(0, 1, 7), 0);
        assert_eq!(wt.count_less(0, 1, 8), 1);
        assert_eq!(wt.count_less(0, 1, 0), 0);
    }

    // ── known small array ────────────────────────────────────────────────────

    #[test]
    fn kth_smallest_known_array() {
        //          0  1  2  3  4  5  6  7
        let arr = [3, 1, 4, 1, 5, 9, 2, 6];
        let wt = WaveletTree::new(&arr);

        // full range sorted: [1, 1, 2, 3, 4, 5, 6, 9]
        assert_eq!(wt.kth_smallest(0, 8, 0), 1);
        assert_eq!(wt.kth_smallest(0, 8, 3), 3);
        assert_eq!(wt.kth_smallest(0, 8, 7), 9);

        // sub-range [2, 6) = [4, 1, 5, 9] -> sorted [1, 4, 5, 9]
        assert_eq!(wt.kth_smallest(2, 6, 0), 1);
        assert_eq!(wt.kth_smallest(2, 6, 1), 4);
        assert_eq!(wt.kth_smallest(2, 6, 2), 5);
        assert_eq!(wt.kth_smallest(2, 6, 3), 9);
    }

    #[test]
    fn rank_known_array() {
        let arr = [3_i64, 1, 4, 1, 5, 9, 2, 6];
        let wt = WaveletTree::new(&arr);

        assert_eq!(wt.rank(0, 8, 1), 2);
        assert_eq!(wt.rank(0, 8, 9), 1);
        assert_eq!(wt.rank(0, 8, 7), 0);
        assert_eq!(wt.rank(2, 5, 1), 1); // [4, 1, 5]
        assert_eq!(wt.rank(2, 5, 5), 1);
    }

    #[test]
    fn count_less_known_array() {
        let arr = [3_i64, 1, 4, 1, 5, 9, 2, 6];
        let wt = WaveletTree::new(&arr);

        // full range: values < 5 are [3,1,4,1,2] => 5 elements
        assert_eq!(wt.count_less(0, 8, 5), 5);
        // full range: values < 1 => 0
        assert_eq!(wt.count_less(0, 8, 1), 0);
        // full range: values < 10 => all 8
        assert_eq!(wt.count_less(0, 8, 10), 8);
        // [2,6) = [4,1,5,9]: values < 5 => [4,1] => 2
        assert_eq!(wt.count_less(2, 6, 5), 2);
    }

    // ── exhaustive brute-force comparison ────────────────────────────────────

    #[test]
    fn exhaustive_small_array() {
        let arr: Vec<i64> = vec![5, 3, 8, 1, 7, 2, 9, 4, 6, 0];
        let wt = WaveletTree::new(&arr);
        let n = arr.len();

        for l in 0..n {
            for r in (l + 1)..=n {
                for k in 0..(r - l) {
                    assert_eq!(
                        wt.kth_smallest(l, r, k),
                        brute_kth(&arr, l, r, k),
                        "kth_smallest({l},{r},{k})"
                    );
                }
                for &v in &arr {
                    assert_eq!(
                        wt.rank(l, r, v),
                        brute_rank(&arr, l, r, v),
                        "rank({l},{r},{v})"
                    );
                    assert_eq!(
                        wt.count_less(l, r, v),
                        brute_count_less(&arr, l, r, v),
                        "count_less({l},{r},{v})"
                    );
                }
            }
        }
    }

    // ── quickcheck properties ─────────────────────────────────────────────────

    #[quickcheck]
    #[allow(clippy::needless_pass_by_value)]
    fn prop_kth_matches_sorted(values: Vec<i32>, l: u8, r: u8, k: u8) -> TestResult {
        if values.is_empty() || values.len() > 64 {
            return TestResult::discard();
        }
        let n = values.len();
        let lo = (l as usize) % n;
        let hi = (r as usize) % n;
        let (ql, qr) = if lo < hi {
            (lo, hi)
        } else if lo > hi {
            (hi, lo)
        } else if lo < n {
            (lo, lo + 1)
        } else {
            return TestResult::discard();
        };
        let window = qr - ql;
        if window == 0 {
            return TestResult::discard();
        }
        let qk = (k as usize) % window;
        let vals: Vec<i64> = values.iter().map(|&x| i64::from(x)).collect();
        let wt = WaveletTree::new(&vals);
        let got = wt.kth_smallest(ql, qr, qk);
        let expected = brute_kth(&vals, ql, qr, qk);
        TestResult::from_bool(got == expected)
    }

    #[quickcheck]
    #[allow(clippy::needless_pass_by_value)]
    fn prop_rank_matches_brute(values: Vec<i32>, l: u8, r: u8) -> TestResult {
        if values.is_empty() || values.len() > 64 {
            return TestResult::discard();
        }
        let n = values.len();
        let lo = (l as usize) % n;
        let hi = ((r as usize) % n) + 1;
        let (ql, qr) = if lo < hi { (lo, hi) } else { (hi - 1, lo + 1) };
        let qr = qr.min(n);
        if ql >= qr {
            return TestResult::discard();
        }
        let vals: Vec<i64> = values.iter().map(|&x| i64::from(x)).collect();
        let wt = WaveletTree::new(&vals);
        for &v in &vals {
            if wt.rank(ql, qr, v) != brute_rank(&vals, ql, qr, v) {
                return TestResult::failed();
            }
        }
        TestResult::passed()
    }

    #[quickcheck]
    #[allow(clippy::needless_pass_by_value)]
    fn prop_count_less_matches_brute(values: Vec<i32>, l: u8, r: u8) -> TestResult {
        if values.is_empty() || values.len() > 64 {
            return TestResult::discard();
        }
        let n = values.len();
        let lo = (l as usize) % n;
        let hi = ((r as usize) % n) + 1;
        let (ql, qr) = if lo < hi { (lo, hi) } else { (hi - 1, lo + 1) };
        let qr = qr.min(n);
        if ql >= qr {
            return TestResult::discard();
        }
        let vals: Vec<i64> = values.iter().map(|&x| i64::from(x)).collect();
        let wt = WaveletTree::new(&vals);
        for &v in &vals {
            if wt.count_less(ql, qr, v) != brute_count_less(&vals, ql, qr, v) {
                return TestResult::failed();
            }
        }
        TestResult::passed()
    }
}
