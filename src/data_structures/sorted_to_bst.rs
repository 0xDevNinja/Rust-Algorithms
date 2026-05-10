//! Sorted-array to balanced binary-search-tree construction.
//!
//! Given a slice that is already sorted in non-decreasing order, this module
//! builds a height-balanced binary search tree by recursively choosing the
//! middle element as the root of each subtree. Each element is visited
//! exactly once and one [`Node`] is allocated per element, so the build
//! runs in `O(n)` time and `O(n)` space, with recursion depth bounded by
//! `O(log n)`.
//!
//! The resulting tree has the minimum possible height for `n` nodes,
//! `ceil(log2(n + 1))`, and an in-order traversal of the result reproduces
//! the input slice (round-trip property).
//!
//! # Precondition
//!
//! Callers must ensure the input slice is sorted in non-decreasing order.
//! No validation is performed: passing an unsorted slice silently produces
//! a tree that violates the BST invariant.
//!
//! # Example
//!
//! ```
//! use rust_algorithms::data_structures::sorted_to_bst::{
//!     build_balanced_bst, height,
//! };
//!
//! let tree = build_balanced_bst(&[1, 2, 3, 4, 5, 6, 7]);
//! assert_eq!(height(tree.as_deref()), 3);
//! ```
//!
//! [`Node`]: Node

/// A node in the balanced binary search tree built by [`build_balanced_bst`].
pub struct Node<T> {
    /// Value stored at this node.
    pub value: T,
    /// Left subtree (values that compare not greater than [`value`](Self::value)).
    pub left: Option<Box<Self>>,
    /// Right subtree (values that compare not less than [`value`](Self::value)).
    pub right: Option<Box<Self>>,
}

/// Build a height-balanced binary search tree from a sorted slice.
///
/// At each recursive step the middle element of the current sub-slice is
/// chosen as the root, and the left/right halves are recursively converted
/// into the left/right subtrees. The slice is **assumed** to be sorted in
/// non-decreasing order; this function performs no validation.
///
/// Returns `None` for an empty input. Runs in `O(n)` time, allocating one
/// node per element.
pub fn build_balanced_bst<T: Clone>(sorted: &[T]) -> Option<Box<Node<T>>> {
    if sorted.is_empty() {
        return None;
    }
    let mid = sorted.len() / 2;
    Some(Box::new(Node {
        value: sorted[mid].clone(),
        left: build_balanced_bst(&sorted[..mid]),
        right: build_balanced_bst(&sorted[mid + 1..]),
    }))
}

/// Return the height of the tree rooted at `root`.
///
/// The height of an empty tree is `0`; the height of a single leaf is `1`.
/// More generally, the height equals the number of nodes on the longest
/// root-to-leaf path. Runs in `O(n)` time.
pub fn height<T>(root: Option<&Node<T>>) -> usize {
    root.map_or(0, |node| {
        let lh = height(node.left.as_deref());
        let rh = height(node.right.as_deref());
        1 + lh.max(rh)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn in_order<T: Clone>(root: Option<&Node<T>>, out: &mut Vec<T>) {
        if let Some(node) = root {
            in_order(node.left.as_deref(), out);
            out.push(node.value.clone());
            in_order(node.right.as_deref(), out);
        }
    }

    fn expected_height(n: usize) -> usize {
        // ceil(log2(n + 1))
        let mut h = 0usize;
        let mut cap = 1usize; // 2^h - 1 capacity at height h is (1<<h) - 1
        while cap < n + 1 {
            h += 1;
            cap = cap.saturating_mul(2);
        }
        h
    }

    #[test]
    fn empty_input_yields_none() {
        let tree: Option<Box<Node<i32>>> = build_balanced_bst::<i32>(&[]);
        assert!(tree.is_none());
        assert_eq!(height::<i32>(None), 0);
    }

    #[test]
    fn single_element_is_leaf() {
        let tree = build_balanced_bst(&[42]).expect("non-empty");
        assert_eq!(tree.value, 42);
        assert!(tree.left.is_none());
        assert!(tree.right.is_none());
        assert_eq!(height(Some(tree.as_ref())), 1);
    }

    #[test]
    fn three_elements_balanced_height_two() {
        let tree = build_balanced_bst(&[1, 2, 3]).expect("non-empty");
        assert_eq!(height(Some(tree.as_ref())), 2);
        let mut out = Vec::new();
        in_order(Some(tree.as_ref()), &mut out);
        assert_eq!(out, vec![1, 2, 3]);
    }

    #[test]
    fn seven_elements_height_three() {
        let input: Vec<i32> = (1..=7).collect();
        let tree = build_balanced_bst(&input).expect("non-empty");
        assert_eq!(height(Some(tree.as_ref())), 3);
        let mut out = Vec::new();
        in_order(Some(tree.as_ref()), &mut out);
        assert_eq!(out, input);
    }

    #[test]
    fn fifteen_elements_height_four() {
        let input: Vec<i32> = (1..=15).collect();
        let tree = build_balanced_bst(&input).expect("non-empty");
        assert_eq!(height(Some(tree.as_ref())), 4);
        let mut out = Vec::new();
        in_order(Some(tree.as_ref()), &mut out);
        assert_eq!(out, input);
    }

    #[test]
    fn in_order_round_trip_property() {
        for n in 0i32..=32 {
            let input: Vec<i32> = (0..n).collect();
            let tree = build_balanced_bst(&input);
            let mut out = Vec::new();
            in_order(tree.as_deref(), &mut out);
            assert_eq!(out, input, "round-trip failed for n = {n}");
        }
    }

    #[test]
    fn height_matches_ceil_log2_n_plus_one() {
        for n in 0i32..=64 {
            let input: Vec<i32> = (0..n).collect();
            let tree = build_balanced_bst(&input);
            let h = height(tree.as_deref());
            assert_eq!(
                h,
                expected_height(n as usize),
                "height mismatch for n = {n}"
            );
        }
    }

    #[test]
    fn duplicates_preserved_in_order() {
        let input = vec![1, 1, 2, 2, 3, 3, 3];
        let tree = build_balanced_bst(&input).expect("non-empty");
        let mut out = Vec::new();
        in_order(Some(tree.as_ref()), &mut out);
        assert_eq!(out, input);
    }
}
