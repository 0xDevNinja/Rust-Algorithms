//! Cartesian tree of a sequence.
//!
//! A **Cartesian tree** of a sequence `a[0..n]` is a binary tree with two
//! defining properties:
//!
//! - **Heap-order on values.** This implementation builds the *min-heap*
//!   variant: every node's value is less than or equal to the values of its
//!   children.
//! - **In-order on positions.** An in-order (left, root, right) traversal of
//!   the tree visits node indices in the order `0, 1, ..., n - 1`, which
//!   reproduces the original input sequence.
//!
//! Together these properties make the tree unique whenever all values are
//! distinct. With ties present, a tie-breaking rule is required: this
//! implementation uses a **strict** monotonic stack — when the value at the
//! top of the stack equals the new value, the new node becomes the right
//! child of that equal-valued node already in the stack. Equivalently, equal
//! values lean right, so an all-equal input degenerates into a right-spine.
//!
//! The build uses a single left-to-right pass with a monotonic stack and runs
//! in `O(n)` time and `O(n)` space. The tree is stored as a `Vec`-based slab:
//! node `i` corresponds to the input position `i`, and `left[i]`, `right[i]`
//! hold child indices into the same slab. Empty input yields a tree with
//! `root() == None` and empty backing vectors.
//!
//! Cartesian trees pair naturally with range-minimum queries via the Eulerian
//! tour + LCA reduction.

/// Min-heap Cartesian tree of a sequence, stored as a flat slab indexed by
/// the original positions of the input.
///
/// - Build: `O(n)` time and `O(n)` space via a monotonic stack
///   ([`Self::build`]).
/// - Heap order: every node's value is `<=` its children's values.
/// - In-order: an in-order traversal yields the input sequence.
/// - Tie-break: equal values lean right (the newer index becomes the right
///   child of the equal-valued node already on the stack).
///
/// Indices returned by [`Self::root`], [`Self::left`], and [`Self::right`]
/// are positions in the original input slice (0-based).
pub struct CartesianTree<T> {
    values: Vec<T>,
    left: Vec<Option<usize>>,
    right: Vec<Option<usize>>,
    root: Option<usize>,
}

impl<T: Ord + Clone> CartesianTree<T> {
    /// Builds the min-heap Cartesian tree of `values` in `O(n)` time using a
    /// monotonic stack.
    ///
    /// Empty input produces a tree with [`Self::root`] equal to `None` and no
    /// nodes. With ties, equal values lean right (see module docs).
    #[must_use]
    pub fn build(values: &[T]) -> Self {
        let n = values.len();
        let mut left: Vec<Option<usize>> = vec![None; n];
        let mut right: Vec<Option<usize>> = vec![None; n];

        // Monotonic stack of indices with strictly increasing values from
        // bottom to top. When a new value is strictly less than the top, the
        // top is popped and becomes the new node's left child. The new node
        // then becomes the right child of whatever remains on the stack.
        let mut stack: Vec<usize> = Vec::with_capacity(n);
        for i in 0..n {
            let mut last_popped: Option<usize> = None;
            while let Some(&top) = stack.last() {
                if values[top] > values[i] {
                    last_popped = stack.pop();
                } else {
                    break;
                }
            }
            // Anything popped becomes our left subtree (it is the previous
            // right-spine of the smallest element on the stack that was still
            // greater than us).
            left[i] = last_popped;
            // We become the right child of whatever is still on top of the
            // stack. Because the comparison above is strict (`>`), equal
            // values stay on the stack and we attach to the right of the
            // last equal value — equal-leans-right tie-breaking.
            if let Some(&top) = stack.last() {
                right[top] = Some(i);
            }
            stack.push(i);
        }

        // The bottom of the stack (if any) is the root: it is the minimum of
        // the input, since nothing smaller could have unseated it.
        let root = stack.first().copied();

        Self {
            values: values.to_vec(),
            left,
            right,
            root,
        }
    }

    /// Index of the root node, or `None` if the tree is empty.
    ///
    /// For a non-empty min-heap Cartesian tree the root is the position of
    /// the minimum value in the input (leftmost minimum under the
    /// equal-leans-right tie-break used here).
    #[must_use]
    pub const fn root(&self) -> Option<usize> {
        self.root
    }

    /// Index of the left child of node `i`, or `None` if absent.
    ///
    /// # Panics
    /// Panics if `i` is out of bounds for the underlying input length.
    #[must_use]
    pub fn left(&self, i: usize) -> Option<usize> {
        self.left[i]
    }

    /// Index of the right child of node `i`, or `None` if absent.
    ///
    /// # Panics
    /// Panics if `i` is out of bounds for the underlying input length.
    #[must_use]
    pub fn right(&self, i: usize) -> Option<usize> {
        self.right[i]
    }

    /// Reference to the value stored at node `i`, which equals the original
    /// input value at position `i`.
    ///
    /// # Panics
    /// Panics if `i` is out of bounds for the underlying input length.
    #[must_use]
    pub fn value(&self, i: usize) -> &T {
        &self.values[i]
    }

    /// Number of nodes in the tree (equivalently, the length of the input).
    #[must_use]
    pub const fn len(&self) -> usize {
        self.values.len()
    }

    /// True if the tree was built from an empty slice.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::CartesianTree;
    use quickcheck::TestResult;
    use quickcheck_macros::quickcheck;

    /// Iterative in-order traversal collecting node indices.
    fn in_order<T: Ord + Clone>(tree: &CartesianTree<T>) -> Vec<usize> {
        let mut out = Vec::with_capacity(tree.len());
        let mut stack: Vec<usize> = Vec::new();
        let mut cur = tree.root();
        loop {
            while let Some(i) = cur {
                stack.push(i);
                cur = tree.left(i);
            }
            let Some(i) = stack.pop() else {
                break;
            };
            out.push(i);
            cur = tree.right(i);
        }
        out
    }

    /// Asserts the min-heap property by walking from each node to its
    /// children and checking value relations.
    fn assert_min_heap<T: Ord + Clone + std::fmt::Debug>(tree: &CartesianTree<T>) {
        for i in 0..tree.len() {
            if let Some(l) = tree.left(i) {
                assert!(
                    tree.value(i) <= tree.value(l),
                    "min-heap violation at {i} -> left {l}"
                );
            }
            if let Some(r) = tree.right(i) {
                assert!(
                    tree.value(i) <= tree.value(r),
                    "min-heap violation at {i} -> right {r}"
                );
            }
        }
    }

    #[test]
    fn empty_input_has_no_root() {
        let tree: CartesianTree<i32> = CartesianTree::build(&[]);
        assert!(tree.is_empty());
        assert_eq!(tree.len(), 0);
        assert_eq!(tree.root(), None);
    }

    #[test]
    fn single_element_is_root() {
        let tree = CartesianTree::build(&[42_i32]);
        assert_eq!(tree.len(), 1);
        assert_eq!(tree.root(), Some(0));
        assert_eq!(tree.left(0), None);
        assert_eq!(tree.right(0), None);
        assert_eq!(tree.value(0), &42);
    }

    #[test]
    fn two_ascending_form_right_spine() {
        // [1, 2]: 1 is root, 2 is right child.
        let tree = CartesianTree::build(&[1_i32, 2]);
        assert_eq!(tree.root(), Some(0));
        assert_eq!(tree.left(0), None);
        assert_eq!(tree.right(0), Some(1));
        assert_eq!(tree.left(1), None);
        assert_eq!(tree.right(1), None);
        assert_eq!(in_order(&tree), vec![0, 1]);
    }

    #[test]
    fn two_descending_form_left_spine() {
        // [2, 1]: 1 is root, 2 is left child.
        let tree = CartesianTree::build(&[2_i32, 1]);
        assert_eq!(tree.root(), Some(1));
        assert_eq!(tree.left(1), Some(0));
        assert_eq!(tree.right(1), None);
        assert_eq!(tree.left(0), None);
        assert_eq!(tree.right(0), None);
        assert_eq!(in_order(&tree), vec![0, 1]);
    }

    #[test]
    fn classic_example_3_2_6_1_9_5() {
        // Expected min-heap Cartesian tree:
        //
        //                 1 (idx 3)
        //                / \
        //         2 (idx 1)  5 (idx 5)
        //          /  \       /
        //   3 (idx 0)  6     9 (idx 4)
        //              (idx 2)
        let values = [3_i32, 2, 6, 1, 9, 5];
        let tree = CartesianTree::build(&values);

        assert_eq!(tree.root(), Some(3));

        // Root = idx 3 (value 1).
        assert_eq!(tree.left(3), Some(1));
        assert_eq!(tree.right(3), Some(5));

        // idx 1 (value 2): left = idx 0 (value 3), right = idx 2 (value 6).
        assert_eq!(tree.left(1), Some(0));
        assert_eq!(tree.right(1), Some(2));

        // idx 0 (value 3): leaf.
        assert_eq!(tree.left(0), None);
        assert_eq!(tree.right(0), None);

        // idx 2 (value 6): leaf.
        assert_eq!(tree.left(2), None);
        assert_eq!(tree.right(2), None);

        // idx 5 (value 5): left = idx 4 (value 9), no right.
        assert_eq!(tree.left(5), Some(4));
        assert_eq!(tree.right(5), None);

        // idx 4 (value 9): leaf.
        assert_eq!(tree.left(4), None);
        assert_eq!(tree.right(4), None);

        assert_min_heap(&tree);
        assert_eq!(in_order(&tree), vec![0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn all_equal_leans_right() {
        // Equal-leans-right: each new equal element becomes the right child
        // of the previous one, producing a pure right-spine.
        let values = vec![7_i32; 5];
        let tree = CartesianTree::build(&values);
        assert_eq!(tree.root(), Some(0));
        for i in 0..4 {
            assert_eq!(tree.left(i), None, "left at {i}");
            assert_eq!(tree.right(i), Some(i + 1), "right at {i}");
        }
        assert_eq!(tree.left(4), None);
        assert_eq!(tree.right(4), None);
        assert_min_heap(&tree);
        assert_eq!(in_order(&tree), vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn in_order_recovers_input_sequence() {
        let values = vec![5_i32, 3, 8, 1, 4, 9, 2, 7, 6, 0];
        let tree = CartesianTree::build(&values);
        assert_min_heap(&tree);
        let order = in_order(&tree);
        assert_eq!(order, (0..values.len()).collect::<Vec<_>>());
    }

    #[test]
    fn root_is_position_of_minimum_for_distinct_values() {
        let values = vec![5_i32, 3, 8, 1, 4, 9, 2, 7, 6, 0];
        let tree = CartesianTree::build(&values);
        // Minimum 0 is at index 9.
        assert_eq!(tree.root(), Some(9));
    }

    #[test]
    fn accessors_report_consistent_values() {
        let values = vec![10_i32, 20, 30];
        let tree = CartesianTree::build(&values);
        for (i, v) in values.iter().enumerate() {
            assert_eq!(tree.value(i), v);
        }
    }

    #[quickcheck]
    #[allow(clippy::needless_pass_by_value)]
    fn prop_in_order_traversal_equals_input(values: Vec<i32>) -> TestResult {
        if values.len() > 50 {
            return TestResult::discard();
        }
        let tree = CartesianTree::build(&values);
        let order = in_order(&tree);
        let expected: Vec<usize> = (0..values.len()).collect();
        if order != expected {
            return TestResult::failed();
        }
        // Also assert min-heap property holds.
        for i in 0..tree.len() {
            if let Some(l) = tree.left(i) {
                if tree.value(i) > tree.value(l) {
                    return TestResult::failed();
                }
            }
            if let Some(r) = tree.right(i) {
                if tree.value(i) > tree.value(r) {
                    return TestResult::failed();
                }
            }
        }
        TestResult::passed()
    }
}
