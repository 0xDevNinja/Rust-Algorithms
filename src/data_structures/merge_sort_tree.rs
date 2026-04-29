//! Merge-sort tree — a segment tree in which each node stores the **sorted**
//! contents of its covered index range.
//!
//! # Complexity
//! - **Build**: O(n log n) time, O(n log n) space.
//! - **Query** (`count_less_than`, `count_in_range`): O(log² n) — O(log n)
//!   nodes visited during segment-tree descent, each costing O(log n) for
//!   `partition_point` on a sorted inner vector.
//!
//! # Preconditions
//! - The structure is **static**: it is built once from a slice and does not
//!   support point updates (those require a more advanced persistent or
//!   wavelet-tree structure).
//!
//! # Classical use cases
//! - Count elements in `arr[l..=r]` that are strictly less than a value `k`.
//! - Count elements in `arr[l..=r]` in a half-open value range `[lo, hi)`.
//! - Find the kth-smallest element in a range by binary searching over the
//!   answer space and calling `count_less_than` as a predicate.

/// Merge-sort tree over a static array of type `T`.
///
/// Build with [`MergeSortTree::build`]; query with
/// [`count_less_than`](MergeSortTree::count_less_than) or
/// [`count_in_range`](MergeSortTree::count_in_range).
pub struct MergeSortTree<T: Ord + Clone> {
    /// Sorted sub-arrays stored at each segment-tree node (1-indexed; node 0
    /// is unused so that left = 2*i and right = 2*i+1 arithmetic works).
    tree: Vec<Vec<T>>,
    /// Original array length.
    n: usize,
}

impl<T: Ord + Clone> MergeSortTree<T> {
    // ------------------------------------------------------------------
    // Build
    // ------------------------------------------------------------------

    /// Builds a merge-sort tree from `values` in O(n log n) time and space.
    ///
    /// An empty slice produces an empty tree; all query methods will return 0.
    pub fn build(values: &[T]) -> Self {
        let n = values.len();
        if n == 0 {
            return Self {
                tree: Vec::new(),
                n: 0,
            };
        }
        // Allocate 4*n inner vecs (the standard over-estimate for a segment
        // tree on n leaves).
        let capacity = 4 * n;
        let mut tree: Vec<Vec<T>> = (0..capacity).map(|_| Vec::new()).collect();
        Self::build_rec(&mut tree, 1, 0, n - 1, values);
        Self { tree, n }
    }

    fn build_rec(tree: &mut Vec<Vec<T>>, node: usize, lo: usize, hi: usize, values: &[T]) {
        if lo == hi {
            tree[node] = vec![values[lo].clone()];
            return;
        }
        let mid = lo + (hi - lo) / 2;
        let left = 2 * node;
        let right = 2 * node + 1;
        Self::build_rec(tree, left, lo, mid, values);
        Self::build_rec(tree, right, mid + 1, hi, values);
        // Merge the two sorted children into this node's sorted vector.
        tree[node] = Self::merge(&tree[left], &tree[right]);
    }

    /// Standard two-pointer merge of two sorted slices.
    fn merge(a: &[T], b: &[T]) -> Vec<T> {
        let mut out = Vec::with_capacity(a.len() + b.len());
        let (mut i, mut j) = (0, 0);
        while i < a.len() && j < b.len() {
            if a[i] <= b[j] {
                out.push(a[i].clone());
                i += 1;
            } else {
                out.push(b[j].clone());
                j += 1;
            }
        }
        out.extend_from_slice(&a[i..]);
        out.extend_from_slice(&b[j..]);
        out
    }

    // ------------------------------------------------------------------
    // Queries
    // ------------------------------------------------------------------

    /// Returns the original array length that the tree was built from.
    pub const fn len(&self) -> usize {
        self.n
    }

    /// Returns `true` if the tree was built from an empty slice.
    pub const fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Counts elements `arr[i]` with `l <= i <= r` and `arr[i] < value`.
    ///
    /// Returns 0 when `l > r` or the range is out of bounds.
    pub fn count_less_than(&self, l: usize, r: usize, value: &T) -> usize {
        if self.n == 0 || l > r || r >= self.n {
            return 0;
        }
        self.query_less(1, 0, self.n - 1, l, r, value)
    }

    fn query_less(
        &self,
        node: usize,
        lo: usize,
        hi: usize,
        ql: usize,
        qr: usize,
        value: &T,
    ) -> usize {
        if qr < lo || hi < ql {
            return 0;
        }
        if ql <= lo && hi <= qr {
            // The whole node is covered — binary-search in the sorted vec.
            return self.tree[node].partition_point(|x| x < value);
        }
        let mid = lo + (hi - lo) / 2;
        self.query_less(2 * node, lo, mid, ql, qr, value)
            + self.query_less(2 * node + 1, mid + 1, hi, ql, qr, value)
    }

    /// Counts elements `arr[i]` with `l <= i <= r` and `lo <= arr[i] < hi`.
    ///
    /// Returns 0 when the value range is empty (`lo >= hi`), when `l > r`,
    /// or when the index range is out of bounds.
    pub fn count_in_range(&self, l: usize, r: usize, lo: &T, hi: &T) -> usize {
        if self.n == 0 || l > r || r >= self.n || lo >= hi {
            return 0;
        }
        self.query_range(1, 0, self.n - 1, l, r, lo, hi)
    }

    #[allow(clippy::too_many_arguments)]
    fn query_range(
        &self,
        node: usize,
        lo: usize,
        hi: usize,
        ql: usize,
        qr: usize,
        val_lo: &T,
        val_hi: &T,
    ) -> usize {
        if qr < lo || hi < ql {
            return 0;
        }
        if ql <= lo && hi <= qr {
            // Elements in [val_lo, val_hi): those < val_hi minus those < val_lo.
            let upper = self.tree[node].partition_point(|x| x < val_hi);
            let lower = self.tree[node].partition_point(|x| x < val_lo);
            return upper - lower;
        }
        let mid = lo + (hi - lo) / 2;
        self.query_range(2 * node, lo, mid, ql, qr, val_lo, val_hi)
            + self.query_range(2 * node + 1, mid + 1, hi, ql, qr, val_lo, val_hi)
    }
}

// ------------------------------------------------------------------
// Tests
// ------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::MergeSortTree;
    use quickcheck::TestResult;
    use quickcheck_macros::quickcheck;

    // ---- unit tests ----

    #[test]
    fn empty_array() {
        let t: MergeSortTree<i32> = MergeSortTree::build(&[]);
        assert_eq!(t.len(), 0);
        assert!(t.is_empty());
        assert_eq!(t.count_less_than(0, 0, &5), 0);
        assert_eq!(t.count_in_range(0, 0, &1, &5), 0);
    }

    #[test]
    fn single_element_less_than() {
        let t = MergeSortTree::build(&[7_i32]);
        assert_eq!(t.len(), 1);
        // value > element
        assert_eq!(t.count_less_than(0, 0, &10), 1);
        // value == element  → strict less-than, so 0
        assert_eq!(t.count_less_than(0, 0, &7), 0);
        // value < element
        assert_eq!(t.count_less_than(0, 0, &3), 0);
    }

    #[test]
    fn single_element_count_in_range() {
        let t = MergeSortTree::build(&[7_i32]);
        // element inside [5, 10)
        assert_eq!(t.count_in_range(0, 0, &5, &10), 1);
        // element outside
        assert_eq!(t.count_in_range(0, 0, &8, &20), 0);
        // lo == hi → empty value range
        assert_eq!(t.count_in_range(0, 0, &7, &7), 0);
    }

    #[test]
    fn sorted_array_every_range() {
        let values: Vec<i32> = (1..=8).collect(); // [1,2,3,4,5,6,7,8]
        let t = MergeSortTree::build(&values);
        // count_less_than(0, 7, 5) → elements < 5 in full range = {1,2,3,4} = 4
        assert_eq!(t.count_less_than(0, 7, &5), 4);
        // count_less_than(2, 5, 4) → arr[2..=5]=[3,4,5,6], elements < 4 = {3} = 1
        assert_eq!(t.count_less_than(2, 5, &4), 1);
        // count_in_range(0, 7, 3, 7) → {3,4,5,6} = 4
        assert_eq!(t.count_in_range(0, 7, &3, &7), 4);
    }

    #[test]
    fn reverse_sorted_array() {
        let values = vec![8_i32, 7, 6, 5, 4, 3, 2, 1];
        let t = MergeSortTree::build(&values);
        // full range, value larger than all → r - l + 1
        assert_eq!(t.count_less_than(0, 7, &100), 8);
        // full range, value smaller than all → 0
        assert_eq!(t.count_less_than(0, 7, &0), 0);
        // sub-range [1..=4] = [7,6,5,4], elements < 6 = {5,4} = 2
        assert_eq!(t.count_less_than(1, 4, &6), 2);
    }

    #[test]
    fn value_smaller_than_all_returns_zero() {
        let values = vec![10_i32, 20, 30, 40, 50];
        let t = MergeSortTree::build(&values);
        assert_eq!(t.count_less_than(0, 4, &5), 0);
    }

    #[test]
    fn value_larger_than_all_returns_range_size() {
        let values = vec![10_i32, 20, 30, 40, 50];
        let t = MergeSortTree::build(&values);
        // [l, r] = [1, 3] → size 3
        assert_eq!(t.count_less_than(1, 3, &100), 3);
    }

    #[test]
    fn full_range_matches_linear_scan() {
        let values = vec![3_i32, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5];
        let t = MergeSortTree::build(&values);
        let value = 5_i32;
        let expected = values.iter().filter(|&&x| x < value).count();
        assert_eq!(t.count_less_than(0, values.len() - 1, &value), expected);
    }

    #[test]
    fn count_in_range_lo_equals_hi_returns_zero() {
        let values = vec![1_i32, 2, 3, 4, 5];
        let t = MergeSortTree::build(&values);
        assert_eq!(t.count_in_range(0, 4, &3, &3), 0);
    }

    #[test]
    fn count_in_range_hand_computed() {
        // arr = [5, 2, 8, 1, 9, 3, 7, 4, 6, 0]
        let values = vec![5_i32, 2, 8, 1, 9, 3, 7, 4, 6, 0];
        let t = MergeSortTree::build(&values);
        // arr[2..=6] = [8,1,9,3,7]; elements in [2, 8) = {3,7} → wait:
        //   2<=3<8 yes, 2<=7<8 yes, 2<=1<8 yes (1>=2? no), 8>=8 no
        //   Actually: {3, 7, 1}? 1 < 2 so no. → {3, 7} = 2
        // Let's recount: [8,1,9,3,7] in [2,8): 8 no, 1 no, 9 no, 3 yes, 7 yes → 2
        assert_eq!(t.count_in_range(2, 6, &2, &8), 2);
        // full range [0..=9], elements in [3,7): {5,3,4,6} = 4
        // arr = [5,2,8,1,9,3,7,4,6,0]; values in [3,7): 5 yes, 2 no, 8 no, 1 no,
        //   9 no, 3 yes, 7 no, 4 yes, 6 yes, 0 no → 4
        assert_eq!(t.count_in_range(0, 9, &3, &7), 4);
    }

    #[test]
    fn random_sequence_hand_verified() {
        // arr = [4, 7, 2, 9, 1, 5, 3, 8, 6, 0]
        let values = vec![4_i32, 7, 2, 9, 1, 5, 3, 8, 6, 0];
        let t = MergeSortTree::build(&values);
        // arr[0..=3]=[4,7,2,9], elements < 5: {4,2} = 2
        assert_eq!(t.count_less_than(0, 3, &5), 2);
        // arr[3..=7]=[9,1,5,3,8], elements < 6: {1,5,3} = 3
        assert_eq!(t.count_less_than(3, 7, &6), 3);
        // arr[1..=5]=[7,2,9,1,5], elements in [2,7): {2,5} = 2
        // 7 no (not < 7), 2 yes, 9 no, 1 no (<2), 5 yes → 2
        assert_eq!(t.count_in_range(1, 5, &2, &7), 2);
    }

    // ---- property-based test ----

    #[quickcheck]
    #[allow(clippy::needless_pass_by_value)]
    fn prop_count_less_than_matches_brute_force(
        values: Vec<i32>,
        l: u8,
        r: u8,
        value: i32,
    ) -> TestResult {
        if values.is_empty() || values.len() > 100 {
            return TestResult::discard();
        }
        let n = values.len();
        let l = (l as usize) % n;
        let r = (r as usize) % n;
        let (l, r) = if l <= r { (l, r) } else { (r, l) };

        let t = MergeSortTree::build(&values);
        let expected = values[l..=r].iter().filter(|&&x| x < value).count();
        let got = t.count_less_than(l, r, &value);
        TestResult::from_bool(got == expected)
    }
}
