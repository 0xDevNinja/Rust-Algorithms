//! Root-to-leaf path-sum search in a binary tree.
//!
//! Provides two utilities over a simple owned binary tree:
//!
//! * [`has_path_sum`]: returns `true` iff there exists a root-to-leaf path
//!   whose node values sum to a target.
//! * [`all_path_sums`]: returns every root-to-leaf path as a `Vec<i64>` of
//!   the values along the path.
//!
//! Both traversals are implemented iteratively with an explicit stack to
//! avoid recursion-depth blow-up on skewed trees.
//!
//! # Complexity
//!
//! For a tree with `n` nodes and height `h`:
//!
//! * [`has_path_sum`]: `O(n)` time, `O(h)` auxiliary space.
//! * [`all_path_sums`]: `O(n * h)` time and space in the worst case (each of
//!   up to `n/2` leaves can produce a path of length `h`).
//!
//! A leaf is a node with no children. By convention, an empty tree (`None`
//! root) contains no root-to-leaf paths, so [`has_path_sum`] returns `false`
//! and [`all_path_sums`] returns an empty `Vec`.
//!
//! # Examples
//!
//! ```
//! use rust_algorithms::data_structures::path_sum::{all_path_sums, has_path_sum, Node};
//!
//! let root = Node {
//!     value: 1,
//!     left: Some(Box::new(Node { value: 2, left: None, right: None })),
//!     right: Some(Box::new(Node { value: 3, left: None, right: None })),
//! };
//!
//! assert!(has_path_sum(Some(&root), 3));
//! assert!(has_path_sum(Some(&root), 4));
//! assert!(!has_path_sum(Some(&root), 5));
//! assert_eq!(all_path_sums(Some(&root)), vec![vec![1, 2], vec![1, 3]]);
//! ```

/// A node in an owned binary tree carrying an `i64` value.
pub struct Node {
    pub value: i64,
    pub left: Option<Box<Self>>,
    pub right: Option<Box<Self>>,
}

/// Returns `true` iff there is a root-to-leaf path whose node values sum to
/// `target`.
///
/// Returns `false` when `root` is `None` (an empty tree has no leaves).
///
/// Uses an iterative DFS with an explicit stack to avoid recursion overflow.
pub fn has_path_sum(root: Option<&Node>, target: i64) -> bool {
    let Some(start) = root else {
        return false;
    };

    let mut stack: Vec<(&Node, i64)> = Vec::new();
    stack.push((start, start.value));

    while let Some((node, running)) = stack.pop() {
        let left = node.left.as_deref();
        let right = node.right.as_deref();

        if left.is_none() && right.is_none() && running == target {
            return true;
        }

        if let Some(child) = right {
            stack.push((child, running + child.value));
        }
        if let Some(child) = left {
            stack.push((child, running + child.value));
        }
    }

    false
}

/// Returns every root-to-leaf path as a `Vec` of node values, ordered as a
/// pre-order traversal (left subtree before right subtree).
///
/// Returns an empty `Vec` when `root` is `None`.
///
/// Uses an iterative DFS with an explicit stack to avoid recursion overflow.
pub fn all_path_sums(root: Option<&Node>) -> Vec<Vec<i64>> {
    let mut paths: Vec<Vec<i64>> = Vec::new();
    let Some(start) = root else {
        return paths;
    };

    // Each stack frame holds the node and the path of values from the root
    // down to (and including) that node.
    let mut stack: Vec<(&Node, Vec<i64>)> = Vec::new();
    stack.push((start, vec![start.value]));

    while let Some((node, path)) = stack.pop() {
        let left = node.left.as_deref();
        let right = node.right.as_deref();

        if left.is_none() && right.is_none() {
            paths.push(path);
            continue;
        }

        // Push right first so left is processed first (pre-order).
        if let Some(child) = right {
            let mut next = path.clone();
            next.push(child.value);
            stack.push((child, next));
        }
        if let Some(child) = left {
            let mut next = path.clone();
            next.push(child.value);
            stack.push((child, next));
        }
    }

    paths
}

#[cfg(test)]
mod tests {
    use super::*;

    fn leaf(value: i64) -> Node {
        Node {
            value,
            left: None,
            right: None,
        }
    }

    #[test]
    fn empty_tree_has_no_path() {
        assert!(!has_path_sum(None, 0));
        assert!(!has_path_sum(None, 42));
        assert_eq!(all_path_sums(None), Vec::<Vec<i64>>::new());
    }

    #[test]
    fn single_node_target_match() {
        let root = Node {
            value: 7,
            left: None,
            right: None,
        };
        assert!(has_path_sum(Some(&root), 7));
        assert_eq!(all_path_sums(Some(&root)), vec![vec![7]]);
    }

    #[test]
    fn single_node_target_mismatch() {
        let root = Node {
            value: 7,
            left: None,
            right: None,
        };
        assert!(!has_path_sum(Some(&root), 0));
        assert!(!has_path_sum(Some(&root), 8));
    }

    #[test]
    fn small_tree_target_left_branch() {
        // Tree: 1 with left=2, right=3.
        let root = Node {
            value: 1,
            left: Some(Box::new(leaf(2))),
            right: Some(Box::new(leaf(3))),
        };
        // 1 + 2 = 3
        assert!(has_path_sum(Some(&root), 3));
    }

    #[test]
    fn small_tree_target_right_branch() {
        let root = Node {
            value: 1,
            left: Some(Box::new(leaf(2))),
            right: Some(Box::new(leaf(3))),
        };
        // 1 + 3 = 4
        assert!(has_path_sum(Some(&root), 4));
    }

    #[test]
    fn small_tree_target_unreachable() {
        let root = Node {
            value: 1,
            left: Some(Box::new(leaf(2))),
            right: Some(Box::new(leaf(3))),
        };
        assert!(!has_path_sum(Some(&root), 5));
        // The root itself is not a leaf, so its value alone is not a path.
        assert!(!has_path_sum(Some(&root), 1));
    }

    #[test]
    fn small_tree_all_paths_listed() {
        let root = Node {
            value: 1,
            left: Some(Box::new(leaf(2))),
            right: Some(Box::new(leaf(3))),
        };
        assert_eq!(all_path_sums(Some(&root)), vec![vec![1, 2], vec![1, 3]]);
    }

    #[test]
    fn deeper_tree_all_paths_preorder() {
        // Tree:
        //         5
        //        / \
        //       4   8
        //      /   / \
        //     11  13  4
        //    /  \      \
        //   7    2      1
        let root = Node {
            value: 5,
            left: Some(Box::new(Node {
                value: 4,
                left: Some(Box::new(Node {
                    value: 11,
                    left: Some(Box::new(leaf(7))),
                    right: Some(Box::new(leaf(2))),
                })),
                right: None,
            })),
            right: Some(Box::new(Node {
                value: 8,
                left: Some(Box::new(leaf(13))),
                right: Some(Box::new(Node {
                    value: 4,
                    left: None,
                    right: Some(Box::new(leaf(1))),
                })),
            })),
        };

        assert!(has_path_sum(Some(&root), 22)); // 5+4+11+2
        assert!(has_path_sum(Some(&root), 27)); // 5+4+11+7
        assert!(has_path_sum(Some(&root), 26)); // 5+8+13
        assert!(has_path_sum(Some(&root), 18)); // 5+8+4+1
        assert!(!has_path_sum(Some(&root), 100));

        assert_eq!(
            all_path_sums(Some(&root)),
            vec![
                vec![5, 4, 11, 7],
                vec![5, 4, 11, 2],
                vec![5, 8, 13],
                vec![5, 8, 4, 1],
            ]
        );
    }

    #[test]
    fn negative_values_are_supported() {
        // Tree:
        //       -2
        //       / \
        //      3  -4
        //          \
        //           1
        let root = Node {
            value: -2,
            left: Some(Box::new(leaf(3))),
            right: Some(Box::new(Node {
                value: -4,
                left: None,
                right: Some(Box::new(leaf(1))),
            })),
        };

        assert!(has_path_sum(Some(&root), 1)); // -2 + 3
        assert!(has_path_sum(Some(&root), -5)); // -2 + -4 + 1
        assert!(!has_path_sum(Some(&root), 0));

        assert_eq!(
            all_path_sums(Some(&root)),
            vec![vec![-2, 3], vec![-2, -4, 1]]
        );
    }

    #[test]
    fn left_skewed_chain_iterative_safe() {
        // Build a deep left-skewed chain to exercise the explicit-stack DFS.
        let depth: i64 = 5_000;
        let mut node: Option<Box<Node>> = None;
        let mut expected: Vec<i64> = Vec::with_capacity(depth as usize);
        for v in (1..=depth).rev() {
            node = Some(Box::new(Node {
                value: v,
                left: node,
                right: None,
            }));
            expected.push(v);
        }
        expected.reverse();

        let root = node.expect("chain should have at least one node");
        let total: i64 = (1..=depth).sum();

        assert!(has_path_sum(Some(&root), total));
        assert!(!has_path_sum(Some(&root), total + 1));
        assert_eq!(all_path_sums(Some(&root)), vec![expected]);
    }
}
