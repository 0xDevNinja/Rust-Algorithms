//! Binary tree reconstruction from preorder + inorder traversals.
//!
//! Given a preorder traversal `[root, left_subtree..., right_subtree...]` and
//! an inorder traversal `[left_subtree..., root, right_subtree...]` of the
//! same binary tree (with distinct values), the tree is uniquely determined.
//!
//! Algorithm: the first element of `preorder` is always the current root.
//! Locate it in `inorder`; everything to its left forms the left subtree's
//! inorder sequence, everything to its right forms the right subtree's. The
//! corresponding preorder slices have the same lengths. Recurse.
//!
//! A `HashMap<value, index_in_inorder>` precomputed once turns the lookup
//! into `O(1)`, giving overall `O(n)` time and `O(n)` extra space.
//!
//! Returns `None` for empty inputs, mismatched lengths, or when the two
//! traversals do not describe the same tree (e.g. value not found in
//! inorder, or different multisets).

use std::collections::HashMap;
use std::hash::Hash;

/// A node of an owned binary tree.
pub struct Node<T> {
    pub value: T,
    pub left: Option<Box<Self>>,
    pub right: Option<Box<Self>>,
}

/// Reconstruct a binary tree from its preorder and inorder traversals.
///
/// Returns `None` if the inputs are empty, have different lengths, or are
/// otherwise inconsistent (a preorder root cannot be found in the inorder
/// slice during recursion).
pub fn build_tree<T: Eq + Hash + Clone>(preorder: &[T], inorder: &[T]) -> Option<Box<Node<T>>> {
    if preorder.is_empty() || inorder.is_empty() || preorder.len() != inorder.len() {
        return None;
    }
    let index: HashMap<T, usize> = inorder
        .iter()
        .enumerate()
        .map(|(i, v)| (v.clone(), i))
        .collect();
    // Distinct-value requirement: if the map collapsed duplicates, the two
    // sequences cannot describe a tree with distinct labels unambiguously.
    if index.len() != inorder.len() {
        return None;
    }
    let mut pre_idx = 0usize;
    build(preorder, &index, &mut pre_idx, 0, inorder.len())
}

fn build<T: Eq + Hash + Clone>(
    preorder: &[T],
    index: &HashMap<T, usize>,
    pre_idx: &mut usize,
    in_lo: usize,
    in_hi: usize,
) -> Option<Box<Node<T>>> {
    if in_lo >= in_hi || *pre_idx >= preorder.len() {
        return None;
    }
    let root_val = preorder[*pre_idx].clone();
    *pre_idx += 1;
    let root_in = *index.get(&root_val)?;
    if root_in < in_lo || root_in >= in_hi {
        // Root is not in the current inorder window — inconsistent input.
        return None;
    }
    let left = build(preorder, index, pre_idx, in_lo, root_in);
    let right = build(preorder, index, pre_idx, root_in + 1, in_hi);
    Some(Box::new(Node {
        value: root_val,
        left,
        right,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn preorder_collect<T: Clone>(node: &Option<Box<Node<T>>>, out: &mut Vec<T>) {
        if let Some(n) = node {
            out.push(n.value.clone());
            preorder_collect(&n.left, out);
            preorder_collect(&n.right, out);
        }
    }

    fn inorder_collect<T: Clone>(node: &Option<Box<Node<T>>>, out: &mut Vec<T>) {
        if let Some(n) = node {
            inorder_collect(&n.left, out);
            out.push(n.value.clone());
            inorder_collect(&n.right, out);
        }
    }

    #[test]
    fn empty_inputs_return_none() {
        let empty: [i32; 0] = [];
        assert!(build_tree::<i32>(&empty, &empty).is_none());
    }

    #[test]
    fn single_node_is_leaf() {
        let tree = build_tree(&[1], &[1]).expect("non-empty result");
        assert_eq!(tree.value, 1);
        assert!(tree.left.is_none());
        assert!(tree.right.is_none());
    }

    #[test]
    fn classic_example_round_trips() {
        let pre = [3, 9, 20, 15, 7];
        let ino = [9, 3, 15, 20, 7];
        let tree = build_tree(&pre, &ino).expect("valid tree");
        // Root and shape checks.
        assert_eq!(tree.value, 3);
        let left = tree.left.as_ref().expect("left child exists");
        assert_eq!(left.value, 9);
        assert!(left.left.is_none() && left.right.is_none());
        let right = tree.right.as_ref().expect("right child exists");
        assert_eq!(right.value, 20);
        assert_eq!(right.left.as_ref().unwrap().value, 15);
        assert_eq!(right.right.as_ref().unwrap().value, 7);
        // Traversal round-trip.
        let mut got_pre = Vec::new();
        let mut got_in = Vec::new();
        preorder_collect(
            &Some(Box::new(Node {
                value: tree.value,
                left: tree.left,
                right: tree.right,
            })),
            &mut got_pre,
        );
        // Rebuild for inorder check (previous tree was moved).
        let tree2 = build_tree(&pre, &ino).unwrap();
        inorder_collect(&Some(tree2), &mut got_in);
        assert_eq!(got_pre, pre);
        assert_eq!(got_in, ino);
    }

    #[test]
    fn mismatched_lengths_return_none() {
        assert!(build_tree(&[1, 2, 3], &[1, 2]).is_none());
    }

    #[test]
    fn value_not_found_returns_none() {
        // Same length, but inorder contains a value not in preorder (and vice
        // versa), so the recursive lookup fails.
        assert!(build_tree(&[1, 2, 3], &[4, 5, 6]).is_none());
    }
}
