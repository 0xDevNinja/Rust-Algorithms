//! Count the number of unival (uni-value) subtrees in a binary tree.
//!
//! A *unival subtree* is a subtree in which every node carries the same value.
//! Every leaf of a non-empty tree is trivially a unival subtree of size one,
//! and an internal node `v` rooted subtree is unival iff:
//!
//! * its left subtree is empty or is a unival subtree whose root value equals
//!   `v.value`, and
//! * its right subtree is empty or is a unival subtree whose root value equals
//!   `v.value`.
//!
//! The implementation is a single post-order DFS that, for every node, returns
//! a pair `(is_unival, count)` describing the subtree rooted at that node.
//! Because each node is visited exactly once and does `O(1)` work, the total
//! running time is `O(n)` in the number of nodes; auxiliary space is `O(d)`
//! where `d` is the depth of the tree (recursion stack).
//!
//! # Examples
//!
//! ```
//! use rust_algorithms::data_structures::unival_subtrees::{count_unival, Node};
//!
//! //     5
//! //    / \
//! //   5   5
//! // All seven (well, three) subtrees are unival → 3.
//! let root = Node {
//!     value: 5,
//!     left: Some(Box::new(Node { value: 5, left: None, right: None })),
//!     right: Some(Box::new(Node { value: 5, left: None, right: None })),
//! };
//! assert_eq!(count_unival(Some(&root)), 3);
//! ```

/// A node in a binary tree.
pub struct Node<T> {
    /// Value stored at this node.
    pub value: T,
    /// Left subtree, if any.
    pub left: Option<Box<Self>>,
    /// Right subtree, if any.
    pub right: Option<Box<Self>>,
}

/// Returns the number of unival subtrees of the binary tree rooted at `root`.
///
/// An empty tree contains zero unival subtrees. Runs in `O(n)` time and `O(d)`
/// auxiliary space where `n` is the number of nodes and `d` is the depth of
/// the tree.
pub fn count_unival<T: Eq>(root: Option<&Node<T>>) -> usize {
    fn dfs<T: Eq>(node: &Node<T>) -> (bool, usize) {
        let mut is_unival = true;
        let mut count = 0;

        if let Some(left) = node.left.as_deref() {
            let (left_unival, left_count) = dfs(left);
            count += left_count;
            if !left_unival || left.value != node.value {
                is_unival = false;
            }
        }

        if let Some(right) = node.right.as_deref() {
            let (right_unival, right_count) = dfs(right);
            count += right_count;
            if !right_unival || right.value != node.value {
                is_unival = false;
            }
        }

        if is_unival {
            count += 1;
        }
        (is_unival, count)
    }

    root.map_or(0, |n| dfs(n).1)
}

#[cfg(test)]
#[allow(clippy::unnecessary_wraps)]
mod tests {
    use super::*;

    fn leaf<T>(value: T) -> Option<Box<Node<T>>> {
        Some(Box::new(Node {
            value,
            left: None,
            right: None,
        }))
    }

    fn branch<T>(
        value: T,
        left: Option<Box<Node<T>>>,
        right: Option<Box<Node<T>>>,
    ) -> Option<Box<Node<T>>> {
        Some(Box::new(Node { value, left, right }))
    }

    #[test]
    fn empty_tree_has_zero_unival_subtrees() {
        let root: Option<&Node<i32>> = None;
        assert_eq!(count_unival(root), 0);
    }

    #[test]
    fn single_node_is_one_unival_subtree() {
        let root = Node {
            value: 7,
            left: None,
            right: None,
        };
        assert_eq!(count_unival(Some(&root)), 1);
    }

    #[test]
    fn full_seven_node_all_same_values() {
        //         1
        //       /   \
        //      1     1
        //     / \   / \
        //    1   1 1   1
        // Every subtree is unival → 7.
        let root = branch(1, branch(1, leaf(1), leaf(1)), branch(1, leaf(1), leaf(1))).unwrap();
        assert_eq!(count_unival(Some(&root)), 7);
    }

    #[test]
    fn mixed_tree_counts_correctly() {
        //         5
        //       /   \
        //      1     5
        //     / \     \
        //    5   5     5
        // Unival subtrees: leaf(5) [left of 1], leaf(5) [right of 1],
        //                   leaf(5) [grandchild of root via right.right],
        //                   subtree rooted at right child of root
        //                       (value 5, only-right-child of value 5) — unival,
        // Total = 4.
        let root = branch(
            5,
            branch(1, leaf(5), leaf(5)),
            Some(Box::new(Node {
                value: 5,
                left: None,
                right: leaf(5),
            })),
        )
        .unwrap();
        assert_eq!(count_unival(Some(&root)), 4);
    }

    #[test]
    fn two_level_mismatched_single_child() {
        //   1
        //    \
        //     2
        // Only the leaf `2` is a unival subtree (root differs from its only
        // child) → 1.
        let root = Node {
            value: 1,
            left: None,
            right: leaf(2),
        };
        assert_eq!(count_unival(Some(&root)), 1);
    }

    #[test]
    fn left_skewed_chain_all_equal() {
        // A left-skewed chain of equal values: every node is the root of a
        // unival subtree → count == number of nodes.
        let mut root: Option<Box<Node<i32>>> = None;
        for _ in 0..50 {
            root = Some(Box::new(Node {
                value: 9,
                left: root,
                right: None,
            }));
        }
        assert_eq!(count_unival(root.as_deref()), 50);
    }
}
