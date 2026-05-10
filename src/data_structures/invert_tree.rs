//! Invert (mirror) a binary tree.
//!
//! Given the root of a binary tree, swap the left and right children at every
//! node. The mirrored tree has the property that an in-order traversal yields
//! the reverse of the original in-order traversal.
//!
//! Two implementations are provided:
//!
//! * [`invert`] — iterative, depth-first using an explicit stack. This is the
//!   default because it does not consume the call stack and therefore handles
//!   pathologically deep (e.g. degenerate / linked-list-shaped) trees without
//!   risking stack overflow.
//! * [`invert_recursive`] — direct recursive implementation, kept for clarity.
//!   Note that on a tree of depth `d` it uses `O(d)` call-stack frames; for a
//!   skewed tree of `n` nodes that is `O(n)` and may overflow on large inputs.
//!
//! Both run in `O(n)` time and `O(d)` auxiliary space where `d` is the depth
//! of the tree.
//!
//! Inverting twice is the identity transformation.
//!
//! # Examples
//!
//! ```
//! use rust_algorithms::data_structures::invert_tree::{invert, Node};
//!
//! //   1            1
//! //  / \    =>    / \
//! // 2   3        3   2
//! let root = Some(Box::new(Node {
//!     value: 1,
//!     left: Some(Box::new(Node { value: 2, left: None, right: None })),
//!     right: Some(Box::new(Node { value: 3, left: None, right: None })),
//! }));
//! let inverted = invert(root);
//! let r = inverted.unwrap();
//! assert_eq!(r.value, 1);
//! assert_eq!(r.left.as_ref().unwrap().value, 3);
//! assert_eq!(r.right.as_ref().unwrap().value, 2);
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

/// Inverts (mirrors) the binary tree rooted at `root` in place, swapping the
/// left and right child of every node.
///
/// Iterative implementation using an explicit stack: safe for arbitrarily deep
/// trees. Runs in `O(n)` time where `n` is the number of nodes.
pub fn invert<T>(root: Option<Box<Node<T>>>) -> Option<Box<Node<T>>> {
    let mut root = root;
    if let Some(boxed) = root.as_mut() {
        let mut stack: Vec<&mut Node<T>> = Vec::new();
        stack.push(boxed.as_mut());
        while let Some(node) = stack.pop() {
            std::mem::swap(&mut node.left, &mut node.right);
            if let Some(l) = node.left.as_mut() {
                stack.push(l.as_mut());
            }
            if let Some(r) = node.right.as_mut() {
                stack.push(r.as_mut());
            }
        }
    }
    root
}

/// Recursive variant of [`invert`].
///
/// Clearer to read but uses one call-stack frame per level of the tree, so it
/// can overflow on very deep / skewed inputs. Prefer [`invert`] in production
/// code; this is provided for pedagogical contrast.
#[allow(clippy::single_option_map)]
pub fn invert_recursive<T>(root: Option<Box<Node<T>>>) -> Option<Box<Node<T>>> {
    root.map(|mut node| {
        let left = invert_recursive(node.left.take());
        let right = invert_recursive(node.right.take());
        node.left = right;
        node.right = left;
        node
    })
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

    /// Collect values in level-order (BFS) for stable structural comparison,
    /// using a sentinel for missing children.
    fn level_order<T: Clone>(root: Option<&Node<T>>) -> Vec<Option<T>> {
        let mut out = Vec::new();
        let mut queue: std::collections::VecDeque<Option<&Node<T>>> =
            std::collections::VecDeque::new();
        queue.push_back(root);
        while let Some(slot) = queue.pop_front() {
            match slot {
                None => out.push(None),
                Some(n) => {
                    out.push(Some(n.value.clone()));
                    queue.push_back(n.left.as_deref());
                    queue.push_back(n.right.as_deref());
                }
            }
        }
        out
    }

    #[test]
    fn empty_tree_returns_none() {
        let r: Option<Box<Node<i32>>> = invert(None);
        assert!(r.is_none());
        let r: Option<Box<Node<i32>>> = invert_recursive(None);
        assert!(r.is_none());
    }

    #[test]
    fn single_node_is_identity() {
        let r = invert(leaf(42));
        let r = r.unwrap();
        assert_eq!(r.value, 42);
        assert!(r.left.is_none());
        assert!(r.right.is_none());
    }

    #[test]
    fn three_node_mirror() {
        //   1           1
        //  / \    =>   / \
        // 2   3       3   2
        let root = branch(1, leaf(2), leaf(3));
        let inverted = invert(root).unwrap();
        assert_eq!(inverted.value, 1);
        assert_eq!(inverted.left.as_ref().unwrap().value, 3);
        assert_eq!(inverted.right.as_ref().unwrap().value, 2);
    }

    fn full_seven_node() -> Option<Box<Node<i32>>> {
        //         1
        //       /   \
        //      2     3
        //     / \   / \
        //    4   5 6   7
        branch(1, branch(2, leaf(4), leaf(5)), branch(3, leaf(6), leaf(7)))
    }

    #[test]
    fn full_seven_node_mirror() {
        let inverted = invert(full_seven_node());
        // Expected level order:
        //         1
        //       /   \
        //      3     2
        //     / \   / \
        //    7   6 5   4
        let expected = vec![
            Some(1),
            Some(3),
            Some(2),
            Some(7),
            Some(6),
            Some(5),
            Some(4),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ];
        assert_eq!(level_order(inverted.as_deref()), expected);
    }

    #[test]
    fn recursive_matches_iterative() {
        let a = invert(full_seven_node());
        let b = invert_recursive(full_seven_node());
        assert_eq!(level_order(a.as_deref()), level_order(b.as_deref()));
    }

    #[test]
    fn double_invert_is_identity_iterative() {
        let original = full_seven_node();
        let snapshot = level_order(original.as_deref());
        let twice = invert(invert(original));
        assert_eq!(level_order(twice.as_deref()), snapshot);
    }

    #[test]
    fn double_invert_is_identity_recursive() {
        let original = full_seven_node();
        let snapshot = level_order(original.as_deref());
        let twice = invert_recursive(invert_recursive(original));
        assert_eq!(level_order(twice.as_deref()), snapshot);
    }

    /// Tiny deterministic LCG; avoids pulling in `rand` as a new dep.
    struct Lcg(u64);
    impl Lcg {
        fn next_u32(&mut self) -> u32 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (self.0 >> 32) as u32
        }
    }

    fn random_tree(rng: &mut Lcg, depth: u32, counter: &mut i32) -> Option<Box<Node<i32>>> {
        if depth == 0 || rng.next_u32().is_multiple_of(4) {
            return None;
        }
        *counter += 1;
        let v = *counter;
        let left = random_tree(rng, depth - 1, counter);
        let right = random_tree(rng, depth - 1, counter);
        Some(Box::new(Node {
            value: v,
            left,
            right,
        }))
    }

    #[test]
    fn double_invert_property_random_trees() {
        let mut rng = Lcg(0x00C0_FFEE);
        for _ in 0..32 {
            let mut counter = 0;
            let tree = random_tree(&mut rng, 6, &mut counter);
            let snapshot = level_order(tree.as_deref());
            let twice = invert(invert(tree));
            assert_eq!(level_order(twice.as_deref()), snapshot);
        }
    }

    #[test]
    fn iterative_handles_deep_skewed_tree() {
        // A left-skewed chain of 20_000 nodes would blow a typical recursive
        // implementation's stack; the iterative version must cope.
        let mut root: Option<Box<Node<i32>>> = None;
        for i in 0..20_000 {
            root = Some(Box::new(Node {
                value: i,
                left: root,
                right: None,
            }));
        }
        let inverted = invert(root).expect("non-empty");
        // After inversion the chain hangs off `right` only.
        let mut cur = Some(inverted);
        let mut count = 0;
        while let Some(n) = cur {
            assert!(n.left.is_none());
            count += 1;
            cur = n.right;
        }
        assert_eq!(count, 20_000);
    }
}
