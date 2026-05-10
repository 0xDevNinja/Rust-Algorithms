//! Owned, recursive binary tree with iterative traversals.
//!
//! A `BinaryTree<T>` owns its nodes through `Box`-allocated children. Every
//! node carries a value of type `T` and at most two children (`left`, `right`).
//! The tree itself is just an `Option<Box<Node<T>>>` root, so the empty tree
//! is the natural default.
//!
//! Although the structure is recursive, all traversal methods are implemented
//! **iteratively** with explicit stacks (or a `VecDeque` for level-order) to
//! avoid stack overflow on deep or skewed trees. Every traversal returns a
//! `Vec<&T>` of references in visit order.
//!
//! # Complexity
//!
//! For a tree with `n` nodes:
//!
//! - [`BinaryTree::inorder`], [`BinaryTree::preorder`],
//!   [`BinaryTree::postorder`], [`BinaryTree::level_order`]: `O(n)` time and
//!   `O(h)` auxiliary space (or `O(w)` for level-order, where `h` is the
//!   height and `w` is the maximum level width).
//! - [`BinaryTree::height`]: `O(n)` time, `O(h)` stack-equivalent space.
//! - [`BinaryTree::size`]: `O(n)` time, `O(h)` stack-equivalent space.
//! - Builders [`BinaryTree::new`], [`BinaryTree::leaf`], [`BinaryTree::node`]:
//!   `O(1)`.

use std::collections::VecDeque;

/// A node in a [`BinaryTree`], owning its children through `Box`.
pub struct Node<T> {
    /// The value stored at this node.
    pub value: T,
    /// The left subtree, if any.
    pub left: Option<Box<Self>>,
    /// The right subtree, if any.
    pub right: Option<Box<Self>>,
}

/// An owned binary tree with iterative traversals.
///
/// The tree is empty when `root` is `None`. Otherwise the root is a
/// `Box<Node<T>>` and the rest of the tree is reachable through its
/// children.
pub struct BinaryTree<T> {
    /// The root node, or `None` for an empty tree.
    pub root: Option<Box<Node<T>>>,
}

impl<T> Default for BinaryTree<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> BinaryTree<T> {
    /// Creates an empty tree.
    pub const fn new() -> Self {
        Self { root: None }
    }

    /// Creates a tree consisting of a single leaf node holding `value`.
    pub fn leaf(value: T) -> Self {
        Self {
            root: Some(Box::new(Node {
                value,
                left: None,
                right: None,
            })),
        }
    }

    /// Creates a tree whose root holds `value` with the given `left` and
    /// `right` subtrees.
    pub fn node(value: T, left: Self, right: Self) -> Self {
        Self {
            root: Some(Box::new(Node {
                value,
                left: left.root,
                right: right.root,
            })),
        }
    }

    /// Returns the in-order traversal (left, root, right) as a vector of
    /// references in visit order.
    ///
    /// Iterative implementation using an explicit `Vec` stack; runs in `O(n)`
    /// time and uses `O(h)` auxiliary space for a tree of height `h`.
    pub fn inorder(&self) -> Vec<&T> {
        let mut out = Vec::new();
        let mut stack: Vec<&Node<T>> = Vec::new();
        let mut current = self.root.as_deref();
        while current.is_some() || !stack.is_empty() {
            while let Some(node) = current {
                stack.push(node);
                current = node.left.as_deref();
            }
            let node = stack.pop().expect("stack non-empty by loop guard");
            out.push(&node.value);
            current = node.right.as_deref();
        }
        out
    }

    /// Returns the pre-order traversal (root, left, right) as a vector of
    /// references in visit order.
    ///
    /// Iterative implementation using an explicit `Vec` stack; runs in `O(n)`
    /// time and uses `O(h)` auxiliary space for a tree of height `h`.
    pub fn preorder(&self) -> Vec<&T> {
        let mut out = Vec::new();
        let mut stack: Vec<&Node<T>> = Vec::new();
        if let Some(root) = self.root.as_deref() {
            stack.push(root);
        }
        while let Some(node) = stack.pop() {
            out.push(&node.value);
            // Push right first so left is processed next (LIFO order).
            if let Some(right) = node.right.as_deref() {
                stack.push(right);
            }
            if let Some(left) = node.left.as_deref() {
                stack.push(left);
            }
        }
        out
    }

    /// Returns the post-order traversal (left, right, root) as a vector of
    /// references in visit order.
    ///
    /// Iterative implementation using an explicit `Vec` stack; runs in `O(n)`
    /// time and uses `O(h)` auxiliary space for a tree of height `h`.
    pub fn postorder(&self) -> Vec<&T> {
        let mut out = Vec::new();
        let mut stack: Vec<(&Node<T>, bool)> = Vec::new();
        if let Some(root) = self.root.as_deref() {
            stack.push((root, false));
        }
        while let Some((node, visited)) = stack.pop() {
            if visited {
                out.push(&node.value);
            } else {
                stack.push((node, true));
                if let Some(right) = node.right.as_deref() {
                    stack.push((right, false));
                }
                if let Some(left) = node.left.as_deref() {
                    stack.push((left, false));
                }
            }
        }
        out
    }

    /// Returns the level-order (breadth-first) traversal as a vector of
    /// references in visit order.
    ///
    /// Uses a `VecDeque` queue and runs in `O(n)` time with `O(w)` auxiliary
    /// space, where `w` is the maximum width of any level.
    pub fn level_order(&self) -> Vec<&T> {
        let mut out = Vec::new();
        let mut queue: VecDeque<&Node<T>> = VecDeque::new();
        if let Some(root) = self.root.as_deref() {
            queue.push_back(root);
        }
        while let Some(node) = queue.pop_front() {
            out.push(&node.value);
            if let Some(left) = node.left.as_deref() {
                queue.push_back(left);
            }
            if let Some(right) = node.right.as_deref() {
                queue.push_back(right);
            }
        }
        out
    }

    /// Returns the height of the tree: the number of edges on the longest
    /// root-to-leaf path. The empty tree has height `0`; a single-node tree
    /// also has height `0`.
    ///
    /// Iterative implementation using an explicit stack of
    /// `(node, depth)` pairs; runs in `O(n)` time with `O(h)` auxiliary
    /// space.
    pub fn height(&self) -> usize {
        let mut max_depth = 0usize;
        let mut stack: Vec<(&Node<T>, usize)> = Vec::new();
        if let Some(root) = self.root.as_deref() {
            stack.push((root, 0));
        }
        while let Some((node, depth)) = stack.pop() {
            if depth > max_depth {
                max_depth = depth;
            }
            if let Some(left) = node.left.as_deref() {
                stack.push((left, depth + 1));
            }
            if let Some(right) = node.right.as_deref() {
                stack.push((right, depth + 1));
            }
        }
        max_depth
    }

    /// Returns the number of nodes in the tree.
    ///
    /// Iterative implementation using an explicit stack; runs in `O(n)` time
    /// with `O(h)` auxiliary space.
    pub fn size(&self) -> usize {
        let mut count = 0usize;
        let mut stack: Vec<&Node<T>> = Vec::new();
        if let Some(root) = self.root.as_deref() {
            stack.push(root);
        }
        while let Some(node) = stack.pop() {
            count += 1;
            if let Some(left) = node.left.as_deref() {
                stack.push(left);
            }
            if let Some(right) = node.right.as_deref() {
                stack.push(right);
            }
        }
        count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds the classic 7-node tree:
    ///
    /// ```text
    ///         1
    ///        / \
    ///       2   3
    ///      / \ / \
    ///     4  5 6  7
    /// ```
    fn classic_tree() -> BinaryTree<i32> {
        BinaryTree::node(
            1,
            BinaryTree::node(2, BinaryTree::leaf(4), BinaryTree::leaf(5)),
            BinaryTree::node(3, BinaryTree::leaf(6), BinaryTree::leaf(7)),
        )
    }

    #[test]
    fn empty_tree_traversals_are_empty() {
        let tree: BinaryTree<i32> = BinaryTree::new();
        assert!(tree.inorder().is_empty());
        assert!(tree.preorder().is_empty());
        assert!(tree.postorder().is_empty());
        assert!(tree.level_order().is_empty());
        assert_eq!(tree.height(), 0);
        assert_eq!(tree.size(), 0);
    }

    #[test]
    fn default_is_empty() {
        let tree: BinaryTree<i32> = BinaryTree::default();
        assert_eq!(tree.size(), 0);
        assert!(tree.root.is_none());
    }

    #[test]
    fn single_node() {
        let tree = BinaryTree::leaf(42);
        assert_eq!(tree.inorder(), vec![&42]);
        assert_eq!(tree.preorder(), vec![&42]);
        assert_eq!(tree.postorder(), vec![&42]);
        assert_eq!(tree.level_order(), vec![&42]);
        assert_eq!(tree.height(), 0);
        assert_eq!(tree.size(), 1);
    }

    #[test]
    fn classic_seven_node_inorder() {
        let tree = classic_tree();
        let got: Vec<i32> = tree.inorder().into_iter().copied().collect();
        assert_eq!(got, vec![4, 2, 5, 1, 6, 3, 7]);
    }

    #[test]
    fn classic_seven_node_preorder() {
        let tree = classic_tree();
        let got: Vec<i32> = tree.preorder().into_iter().copied().collect();
        assert_eq!(got, vec![1, 2, 4, 5, 3, 6, 7]);
    }

    #[test]
    fn classic_seven_node_postorder() {
        let tree = classic_tree();
        let got: Vec<i32> = tree.postorder().into_iter().copied().collect();
        assert_eq!(got, vec![4, 5, 2, 6, 7, 3, 1]);
    }

    #[test]
    fn classic_seven_node_level_order() {
        let tree = classic_tree();
        let got: Vec<i32> = tree.level_order().into_iter().copied().collect();
        assert_eq!(got, vec![1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn classic_seven_node_height_and_size() {
        let tree = classic_tree();
        assert_eq!(tree.height(), 2);
        assert_eq!(tree.size(), 7);
    }

    #[test]
    fn left_skewed_tree_height_and_size() {
        // A tree with only left children: 1 -> 2 -> 3 -> 4.
        let tree = BinaryTree::node(
            1,
            BinaryTree::node(
                2,
                BinaryTree::node(3, BinaryTree::leaf(4), BinaryTree::new()),
                BinaryTree::new(),
            ),
            BinaryTree::new(),
        );
        assert_eq!(tree.size(), 4);
        assert_eq!(tree.height(), 3);
        let got: Vec<i32> = tree.inorder().into_iter().copied().collect();
        assert_eq!(got, vec![4, 3, 2, 1]);
        let got: Vec<i32> = tree.preorder().into_iter().copied().collect();
        assert_eq!(got, vec![1, 2, 3, 4]);
        let got: Vec<i32> = tree.postorder().into_iter().copied().collect();
        assert_eq!(got, vec![4, 3, 2, 1]);
        let got: Vec<i32> = tree.level_order().into_iter().copied().collect();
        assert_eq!(got, vec![1, 2, 3, 4]);
    }
}
