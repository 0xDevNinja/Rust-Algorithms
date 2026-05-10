//! Binary tree serialization and deserialization.
//!
//! Encodes a binary tree as a comma-separated pre-order traversal, using `#`
//! as a sentinel for missing children. The format is symmetric — calling
//! [`deserialize`] on the output of [`serialize`] reproduces the original
//! tree (and vice versa for any well-formed input).
//!
//! # Format
//!
//! Pre-order traversal: visit the root, then the left subtree, then the right
//! subtree. Each visit writes either the node's [`i64`] value or `#` for a
//! missing node. Tokens are separated by commas.
//!
//! Example: a tree with root `1`, left child `2` (leaf), and right child `3`
//! (leaf) serializes to `"1,2,#,#,3,#,#"`. The empty tree serializes to `"#"`.
//!
//! # Complexity
//!
//! Both [`serialize`] and [`deserialize`] run in `O(n)` time and use `O(n)`
//! auxiliary space, where `n` is the number of nodes (the recursion depth is
//! `O(h)` for a tree of height `h`, plus the output/input string of size
//! `O(n)`).

/// A node in a binary tree. Values are [`i64`] so that any integer (including
/// negatives and the platform-independent extremes) round-trips through the
/// textual format.
pub struct Node {
    pub value: i64,
    pub left: Option<Box<Self>>,
    pub right: Option<Box<Self>>,
}

impl Node {
    /// Construct a leaf node with the given value.
    pub const fn leaf(value: i64) -> Self {
        Self {
            value,
            left: None,
            right: None,
        }
    }

    /// Construct a node with the given value and children.
    pub const fn new(value: i64, left: Option<Box<Self>>, right: Option<Box<Self>>) -> Self {
        Self { value, left, right }
    }
}

/// Serialize a binary tree using pre-order traversal with `#` for null
/// children.
///
/// Pass `None` for the empty tree (which yields `"#"`). For a non-empty tree,
/// pass `Some(&*root)`.
pub fn serialize(root: Option<&Node>) -> String {
    let mut out = String::new();
    write_preorder(root, &mut out);
    out
}

fn write_preorder(node: Option<&Node>, out: &mut String) {
    if !out.is_empty() {
        out.push(',');
    }
    match node {
        None => out.push('#'),
        Some(n) => {
            // i64 values may be up to 20 chars including sign — push directly.
            out.push_str(&n.value.to_string());
            write_preorder(n.left.as_deref(), out);
            write_preorder(n.right.as_deref(), out);
        }
    }
}

/// Deserialize the textual pre-order representation back into a tree.
///
/// Returns `None` for the empty tree (`"#"`) and also for any malformed
/// input — extra trailing tokens, missing tokens, or non-integer non-`#`
/// tokens. To distinguish "valid empty tree" from "malformed input", call
/// [`serialize`] on the result and compare against the original string, or
/// validate the input first.
pub fn deserialize(s: &str) -> Option<Box<Node>> {
    let mut iter = s.split(',');
    let root = parse_preorder(&mut iter).ok()?;
    // Reject any trailing tokens.
    if iter.next().is_some() {
        return None;
    }
    root
}

/// Parse one sub-tree token from `iter`. Returns `Ok(None)` for a `#`
/// terminator, `Ok(Some(node))` for a parsed sub-tree, and `Err(())` if the
/// stream is exhausted early or contains a non-integer non-`#` token.
fn parse_preorder<'a, I>(iter: &mut I) -> Result<Option<Box<Node>>, ()>
where
    I: Iterator<Item = &'a str>,
{
    let token = iter.next().ok_or(())?;
    if token == "#" {
        return Ok(None);
    }
    let value: i64 = token.parse().map_err(|_| ())?;
    let left = parse_preorder(iter)?;
    let right = parse_preorder(iter)?;
    Ok(Some(Box::new(Node { value, left, right })))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trip(root: Option<&Node>) -> String {
        let serialized = serialize(root);
        let parsed = deserialize(&serialized);
        // Re-serialize the parsed tree and compare against the original
        // serialization — structural equality without writing a custom
        // comparator.
        let reserialized = serialize(parsed.as_deref());
        assert_eq!(serialized, reserialized);
        serialized
    }

    fn boxed(node: Node) -> Box<Node> {
        Box::new(node)
    }

    #[test]
    fn empty_tree_round_trip() {
        let s = serialize(None);
        assert_eq!(s, "#");
        assert!(deserialize(&s).is_none());
        // Re-serializing the deserialized empty tree must still be "#".
        assert_eq!(serialize(deserialize(&s).as_deref()), "#");
    }

    #[test]
    fn single_node_round_trip() {
        let tree = Node::leaf(42);
        let s = round_trip(Some(&tree));
        assert_eq!(s, "42,#,#");
    }

    #[test]
    fn complex_five_node_round_trip() {
        // Five-node tree:
        //       1
        //      / \
        //     2   3
        //    / \
        //   4   5
        let tree = Node::new(
            1,
            Some(boxed(Node::new(
                2,
                Some(boxed(Node::leaf(4))),
                Some(boxed(Node::leaf(5))),
            ))),
            Some(boxed(Node::leaf(3))),
        );
        let s = round_trip(Some(&tree));
        assert_eq!(s, "1,2,4,#,#,5,#,#,3,#,#");
    }

    #[test]
    fn doc_example_round_trip() {
        let tree = Node::new(1, Some(boxed(Node::leaf(2))), Some(boxed(Node::leaf(3))));
        let s = round_trip(Some(&tree));
        assert_eq!(s, "1,2,#,#,3,#,#");
    }

    #[test]
    fn negative_values_round_trip() {
        let tree = Node::new(
            -7,
            Some(boxed(Node::leaf(-100))),
            Some(boxed(Node::leaf(0))),
        );
        let s = round_trip(Some(&tree));
        assert_eq!(s, "-7,-100,#,#,0,#,#");
    }

    #[test]
    fn extreme_i64_values_round_trip() {
        let tree = Node::new(
            i64::MAX,
            Some(boxed(Node::leaf(i64::MIN))),
            Some(boxed(Node::leaf(i64::MAX - 1))),
        );
        let s = serialize(Some(&tree));
        let parsed = deserialize(&s).expect("must parse");
        assert_eq!(parsed.value, i64::MAX);
        assert_eq!(parsed.left.as_ref().unwrap().value, i64::MIN);
        assert_eq!(parsed.right.as_ref().unwrap().value, i64::MAX - 1);
        assert_eq!(serialize(Some(&parsed)), s);
    }

    #[test]
    fn malformed_empty_string_is_none() {
        // The empty string contains a single empty token, which is neither
        // "#" nor a valid integer — should be rejected.
        assert!(deserialize("").is_none());
    }

    #[test]
    fn malformed_truncated_is_none() {
        // Missing the right-subtree terminators.
        assert!(deserialize("1,2,#").is_none());
    }

    #[test]
    fn malformed_trailing_tokens_is_none() {
        // Extra tokens after a complete tree.
        assert!(deserialize("1,#,#,99").is_none());
    }

    #[test]
    fn malformed_non_integer_token_is_none() {
        assert!(deserialize("1,foo,#,#,#").is_none());
    }

    #[test]
    fn malformed_lone_separator_is_none() {
        assert!(deserialize(",").is_none());
    }
}
