//! Binary search tree (BST) validation.
//!
//! A binary search tree is *valid* iff for every node `n` in the tree,
//! every value in `n.left` is strictly less than `n.value` and every value
//! in `n.right` is strictly greater than `n.value`. This implementation is
//! **strict**: duplicate values are not allowed and cause validation to fail.
//!
//! # Approach
//!
//! We use the min/max bound recursion. Each recursive call carries an
//! exclusive lower bound and an exclusive upper bound that the current
//! node's value must lie strictly between. Descending left tightens the
//! upper bound to the parent value; descending right tightens the lower
//! bound to the parent value. This catches the classic "tricky" case where
//! every local parent/child relationship looks valid but a deep node
//! violates an ancestor's bound (for example, a node in the left subtree
//! with a value larger than the root).
//!
//! An alternative is to perform an inorder traversal and verify the
//! visited sequence is strictly ascending; both run in `O(n)` time, but
//! the bounds approach avoids materialising the traversal and short-
//! circuits on the first violation.
//!
//! - Time: `O(n)` where `n` is the number of nodes (each node visited once).
//! - Space: `O(h)` recursion stack, where `h` is the height of the tree
//!   (`O(log n)` if balanced, `O(n)` worst case).
//!
//! # Example
//!
//! ```
//! use rust_algorithms::data_structures::bst_validation::{is_valid_bst, Node};
//!
//! let root = Node {
//!     value: 2,
//!     left: Some(Box::new(Node { value: 1, left: None, right: None })),
//!     right: Some(Box::new(Node { value: 3, left: None, right: None })),
//! };
//! assert!(is_valid_bst(Some(&root)));
//! ```

/// A node in a binary tree, generic over the stored value type.
///
/// Used as input to [`is_valid_bst`]. Children are owned via `Box` to
/// keep the recursive type sized.
pub struct Node<T> {
    /// Value stored at this node.
    pub value: T,
    /// Left subtree (values strictly less than `value` in a valid BST).
    pub left: Option<Box<Self>>,
    /// Right subtree (values strictly greater than `value` in a valid BST).
    pub right: Option<Box<Self>>,
}

impl<T> Node<T> {
    /// Convenience constructor for a leaf node.
    pub const fn leaf(value: T) -> Self {
        Self {
            value,
            left: None,
            right: None,
        }
    }
}

/// Returns `true` iff the tree rooted at `root` is a valid binary search
/// tree under strict ordering (no duplicate values allowed).
///
/// An empty tree (`None`) is considered valid, as is a single-node tree.
///
/// - Time: `O(n)`.
/// - Space: `O(h)` where `h` is the height of the tree.
pub fn is_valid_bst<T: Ord>(root: Option<&Node<T>>) -> bool {
    check(root, None, None)
}

/// Recursive helper enforcing exclusive `(lo, hi)` bounds on every value
/// in the subtree rooted at `node`.
fn check<T: Ord>(node: Option<&Node<T>>, lo: Option<&T>, hi: Option<&T>) -> bool {
    let Some(n) = node else {
        return true;
    };
    if let Some(l) = lo {
        if &n.value <= l {
            return false;
        }
    }
    if let Some(h) = hi {
        if &n.value >= h {
            return false;
        }
    }
    check(n.left.as_deref(), lo, Some(&n.value)) && check(n.right.as_deref(), Some(&n.value), hi)
}

#[cfg(test)]
mod tests {
    use super::{is_valid_bst, Node};

    #[test]
    fn empty_tree_is_valid() {
        let root: Option<&Node<i32>> = None;
        assert!(is_valid_bst(root));
    }

    #[test]
    fn single_node_is_valid() {
        let root = Node::leaf(42);
        assert!(is_valid_bst(Some(&root)));
    }

    #[test]
    fn valid_three_node_tree() {
        // root 2, left 1, right 3
        let root = Node {
            value: 2,
            left: Some(Box::new(Node::leaf(1))),
            right: Some(Box::new(Node::leaf(3))),
        };
        assert!(is_valid_bst(Some(&root)));
    }

    #[test]
    fn invalid_left_child_greater_than_root() {
        // root 5, left 10
        let root = Node {
            value: 5,
            left: Some(Box::new(Node::leaf(10))),
            right: None,
        };
        assert!(!is_valid_bst(Some(&root)));
    }

    #[test]
    fn invalid_right_child_smaller_than_root() {
        // root 10, right 5 — local subtree (just root + right) violates
        // because right child must be > root.
        let root = Node {
            value: 10,
            left: None,
            right: Some(Box::new(Node::leaf(5))),
        };
        assert!(!is_valid_bst(Some(&root)));
    }

    #[test]
    fn tricky_global_violation() {
        // Each parent/child pair looks fine locally, but node `6` lives in
        // the left subtree of `5` and so violates the root's upper bound.
        //
        //        5
        //       / \
        //      3   7
        //     / \
        //    2   6   <- 6 > 5, invalid globally
        let root = Node {
            value: 5,
            left: Some(Box::new(Node {
                value: 3,
                left: Some(Box::new(Node::leaf(2))),
                right: Some(Box::new(Node::leaf(6))),
            })),
            right: Some(Box::new(Node::leaf(7))),
        };
        assert!(!is_valid_bst(Some(&root)));
    }

    #[test]
    fn duplicate_values_invalid() {
        // root 2, left 2 — strict ordering forbids equal values.
        let root = Node {
            value: 2,
            left: Some(Box::new(Node::leaf(2))),
            right: None,
        };
        assert!(!is_valid_bst(Some(&root)));
    }

    #[test]
    fn duplicate_in_right_subtree_invalid() {
        // root 2, right 2.
        let root = Node {
            value: 2,
            left: None,
            right: Some(Box::new(Node::leaf(2))),
        };
        assert!(!is_valid_bst(Some(&root)));
    }

    #[test]
    fn deeper_balanced_valid() {
        //        4
        //       / \
        //      2   6
        //     / \ / \
        //    1  3 5  7
        let root = Node {
            value: 4,
            left: Some(Box::new(Node {
                value: 2,
                left: Some(Box::new(Node::leaf(1))),
                right: Some(Box::new(Node::leaf(3))),
            })),
            right: Some(Box::new(Node {
                value: 6,
                left: Some(Box::new(Node::leaf(5))),
                right: Some(Box::new(Node::leaf(7))),
            })),
        };
        assert!(is_valid_bst(Some(&root)));
    }

    #[test]
    fn works_with_string_keys() {
        let root = Node {
            value: "m".to_string(),
            left: Some(Box::new(Node::leaf("a".to_string()))),
            right: Some(Box::new(Node::leaf("z".to_string()))),
        };
        assert!(is_valid_bst(Some(&root)));
    }
}
