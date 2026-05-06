//! Ukkonen's suffix tree: online O(n) construction over byte strings.
//!
//! A **suffix tree** is a compressed trie of all suffixes of a string. Every
//! internal node has at least two children, and every leaf represents exactly
//! one suffix. This module implements Ukkonen's classic online construction
//! (1995), which processes the input left-to-right in amortised O(1) work per
//! character, giving an O(n) total build time.
//!
//! # Sentinel
//! A virtual sentinel byte `256` (represented as `u16`) is appended during
//! construction so that every suffix ends at a distinct leaf (turning the
//! *implicit* suffix tree into an *explicit* one). The sentinel is never stored
//! in `text` and is invisible to the public API.
//!
//! # Complexity
//! - Build:   O(n) time, O(n) space (at most 2n−1 nodes for input of length n).
//! - [`SuffixTree::contains`][]: O(|pattern|).
//! - [`SuffixTree::distinct_substring_count`][]: O(n).
//!
//! # Preconditions
//! None. The algorithm handles arbitrary `&[u8]` including empty input and
//! byte sequences with repeated characters.

use std::collections::BTreeMap;

// ---------------------------------------------------------------------------
// Node arena
// ---------------------------------------------------------------------------

/// Sentinel value appended internally so every suffix has a unique leaf.
/// Chosen outside the `u8` range so it never clashes with real bytes.
const SENTINEL: u16 = 256;

/// Index type for edges and nodes.
type NodeId = usize;

/// A node in the suffix tree (internal node or leaf).
///
/// Leaf nodes have `end == None`, meaning their `end` moves with the global
/// "current end" pointer during construction (Ukkonen's open-ended leaves
/// trick). Internal nodes store a concrete `end` value.
#[derive(Debug, Clone)]
struct Node {
    /// Byte index in the augmented text where this node's edge-label starts.
    start: usize,
    /// Exclusive end of this node's edge-label, or `None` for a leaf (open
    /// end pointer — grows with each new character during construction).
    end: Option<usize>,
    /// Suffix link used only during construction. Points to another internal
    /// node; `0` means "not set yet".
    suffix_link: NodeId,
    /// Children keyed by the first byte of their edge-label (u16 to include
    /// the sentinel 256).
    children: BTreeMap<u16, NodeId>,
}

impl Node {
    /// Create a new node. `end = None` indicates a leaf (open end).
    const fn new(start: usize, end: Option<usize>) -> Self {
        Self {
            start,
            end,
            suffix_link: 0,
            children: BTreeMap::new(),
        }
    }

    /// The number of characters on the edge leading into this node, given the
    /// current global end pointer `global_end` (used only for leaves).
    fn edge_len(&self, global_end: usize) -> usize {
        self.end.unwrap_or(global_end) - self.start
    }
}

// ---------------------------------------------------------------------------
// SuffixTree
// ---------------------------------------------------------------------------

/// Suffix tree built with Ukkonen's online O(n) algorithm.
///
/// Constructed via [`SuffixTree::new`]; supports [`SuffixTree::contains`] and
/// [`SuffixTree::distinct_substring_count`].
#[derive(Debug, Clone)]
pub struct SuffixTree {
    /// Augmented text: original bytes followed by the sentinel (256).
    text: Vec<u16>,
    /// Node arena. Index 0 is the root. Nodes are never removed.
    nodes: Vec<Node>,
    /// The global "leaf end" pointer — equal to the length of text processed
    /// so far (exclusive). Open-ended leaves implicitly use this as their end.
    global_end: usize,
}

impl SuffixTree {
    // ----- construction ----

    /// Builds the suffix tree for `text` using Ukkonen's algorithm.
    ///
    /// Runs in O(n) time and O(n) space where n = `text.len()`.
    #[must_use]
    pub fn new(text: &[u8]) -> Self {
        // Convert to u16 and append sentinel.
        let mut augmented: Vec<u16> = text.iter().map(|&b| u16::from(b)).collect();
        augmented.push(SENTINEL);

        let n = augmented.len();

        // Preallocate arena. Ukkonen's tree has at most 2n − 1 nodes.
        let mut nodes: Vec<Node> = Vec::with_capacity(2 * n + 2);

        // Index 0 = root.
        // The root has a dummy edge-label of length 0 (start = 0, end = Some(0)).
        nodes.push(Node::new(0, Some(0)));

        let mut tree = Self {
            text: augmented,
            nodes,
            global_end: 0,
        };

        // Active point: (active_node, active_edge_char, active_len).
        // `active_edge` stores the text index of the first char of the active
        // edge — not the character value itself — so we can look it up cheaply.
        let mut active_node: NodeId = 0; // root
        let mut active_edge: usize = 0; // index into text of the active edge's first char
        let mut active_len: usize = 0;
        let mut remainder: usize = 0;

        for i in 0..n {
            tree.global_end = i + 1;
            remainder += 1;
            let c = tree.text[i];

            let mut last_new_internal: Option<NodeId> = None;

            while remainder > 0 {
                // Canonicalise: if active_len is 0, the active edge starts at c.
                if active_len == 0 {
                    active_edge = i;
                }

                let edge_char = tree.text[active_edge];

                if tree.nodes[active_node].children.contains_key(&edge_char) {
                    // There is an edge starting with edge_char from active_node.
                    let next = tree.nodes[active_node].children[&edge_char];
                    let next_edge_len = tree.nodes[next].edge_len(tree.global_end);

                    // Walk-down: if active_len >= edge length, skip over node.
                    if active_len >= next_edge_len {
                        active_edge += next_edge_len;
                        active_len -= next_edge_len;
                        active_node = next;
                        // Do NOT consume remainder here; retry with new active point.
                        continue;
                    }

                    // Check if the next character on the active edge matches c.
                    let next_char_pos = tree.nodes[next].start + active_len;
                    if tree.text[next_char_pos] == c {
                        // Rule 3 (character already in tree): extend active length
                        // and stop (implicit suffix — no new node needed).
                        active_len += 1;
                        if let Some(prev) = last_new_internal {
                            tree.nodes[prev].suffix_link = active_node;
                        }
                        break;
                    }

                    // Rule 2 (mismatch mid-edge): split the edge.
                    // New internal node goes at the split point.
                    let split_end = tree.nodes[next].start + active_len;
                    let split = tree.alloc_node(tree.nodes[next].start, Some(split_end));

                    // Redirect parent's edge to the split node.
                    tree.nodes[active_node].children.insert(edge_char, split);

                    // Old child: fix its start to split_end.
                    tree.nodes[next].start = split_end;
                    // Add old child under split, keyed by the char at split_end.
                    let old_first_char = tree.text[split_end];
                    tree.nodes[split].children.insert(old_first_char, next);

                    // New leaf for the current character.
                    let leaf = tree.alloc_node(i, None);
                    tree.nodes[split].children.insert(c, leaf);

                    // Suffix link bookkeeping.
                    if let Some(prev) = last_new_internal {
                        tree.nodes[prev].suffix_link = split;
                    }
                    last_new_internal = Some(split);
                } else {
                    // Rule 2 (no matching edge from active node): create leaf.
                    let leaf = tree.alloc_node(i, None);
                    tree.nodes[active_node].children.insert(edge_char, leaf);

                    // Set suffix link on previously created internal node.
                    if let Some(prev) = last_new_internal {
                        tree.nodes[prev].suffix_link = active_node;
                        last_new_internal = None;
                    }
                }

                // One suffix has been implanted; decrement remainder.
                remainder -= 1;

                // Follow suffix link or walk back to root.
                if active_node == 0 {
                    // Active node is root: shorten active edge/len.
                    if active_len > 0 {
                        active_len -= 1;
                        active_edge = i - remainder + 1;
                    }
                } else {
                    let link = tree.nodes[active_node].suffix_link;
                    active_node = if link != 0 { link } else { 0 };
                }
            }
        }

        tree
    }

    /// Allocate a new node in the arena; returns its index.
    fn alloc_node(&mut self, start: usize, end: Option<usize>) -> NodeId {
        let id = self.nodes.len();
        self.nodes.push(Node::new(start, end));
        id
    }

    // ----- public queries --------------------------------------------------

    /// Returns `true` if `pattern` occurs as a contiguous substring of the
    /// text passed to [`SuffixTree::new`].
    ///
    /// The empty pattern is always a substring. Runs in O(|pattern|) time.
    #[must_use]
    pub fn contains(&self, pattern: &[u8]) -> bool {
        if pattern.is_empty() {
            return true;
        }
        let mut node = 0_usize; // start at root
        let mut pi = 0_usize; // position in pattern

        loop {
            if pi == pattern.len() {
                return true;
            }
            let c = u16::from(pattern[pi]);

            let Some(&child) = self.nodes[node].children.get(&c) else {
                return false;
            };

            // Match characters along the edge.
            let edge_start = self.nodes[child].start;
            let edge_end = self.nodes[child].end.unwrap_or(self.global_end);

            for ti in edge_start..edge_end {
                if pi == pattern.len() {
                    return true;
                }
                if self.text[ti] != u16::from(pattern[pi]) {
                    return false;
                }
                pi += 1;
            }

            node = child;
        }
    }

    /// Returns the number of distinct non-empty substrings of the original
    /// text.
    ///
    /// Each edge in the suffix tree contributes `edge_len` distinct substrings
    /// (one for each prefix length of the edge). Summing over all edges (but
    /// excluding the sentinel character, which is not a real byte) gives the
    /// total count. Runs in O(n) time.
    #[must_use]
    pub fn distinct_substring_count(&self) -> u64 {
        // The number of distinct (non-empty) substrings equals the sum of
        // edge-label lengths across all edges, minus edges that contain or
        // consist solely of the sentinel.
        //
        // Simpler formulation: DFS over the tree; for every node (except root)
        // add the length of its edge-label, clamped to exclude sentinel chars.
        let n = self.text.len(); // augmented length (includes sentinel)
        let real_len = n - 1; // length of original text

        let mut total: u64 = 0;
        // Use an explicit stack to avoid recursion-stack issues on large inputs.
        let mut stack: Vec<NodeId> = self.nodes[0].children.values().copied().collect();

        while let Some(id) = stack.pop() {
            let node = &self.nodes[id];
            let start = node.start;
            let end = node.end.unwrap_or(self.global_end);

            // The edge-label is text[start..end]. Clamp the end so we do not
            // count the sentinel (always at text[real_len]).
            // If start >= real_len the entire edge is sentinel — skip.
            if start >= real_len {
                continue;
            }
            let clamped_end = end.min(real_len);
            let contribution = (clamped_end - start) as u64;
            total += contribution;

            for &child in node.children.values() {
                stack.push(child);
            }
        }

        total
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::SuffixTree;
    use std::collections::HashSet;

    // ------------------------------------------------------------------
    // Brute-force helpers
    // ------------------------------------------------------------------

    fn brute_contains(text: &[u8], pattern: &[u8]) -> bool {
        if pattern.is_empty() {
            return true;
        }
        if pattern.len() > text.len() {
            return false;
        }
        text.windows(pattern.len()).any(|w| w == pattern)
    }

    fn brute_distinct_substrings(text: &[u8]) -> u64 {
        let mut set: HashSet<Vec<u8>> = HashSet::new();
        for i in 0..text.len() {
            for j in (i + 1)..=text.len() {
                set.insert(text[i..j].to_vec());
            }
        }
        set.len() as u64
    }

    // ------------------------------------------------------------------
    // Basic structural tests
    // ------------------------------------------------------------------

    #[test]
    fn empty_text() {
        let st = SuffixTree::new(b"");
        assert!(st.contains(b""), "empty pattern must match");
        assert!(
            !st.contains(b"a"),
            "non-empty pattern must not match empty text"
        );
        assert_eq!(st.distinct_substring_count(), 0);
    }

    #[test]
    fn single_char() {
        let st = SuffixTree::new(b"a");
        assert!(st.contains(b""));
        assert!(st.contains(b"a"));
        assert!(!st.contains(b"b"));
        assert!(!st.contains(b"aa"));
        assert_eq!(st.distinct_substring_count(), 1);
    }

    #[test]
    fn two_chars_ab() {
        let st = SuffixTree::new(b"ab");
        assert!(st.contains(b"a"));
        assert!(st.contains(b"b"));
        assert!(st.contains(b"ab"));
        assert!(!st.contains(b"ba"));
        assert!(!st.contains(b"abc"));
        assert_eq!(st.distinct_substring_count(), 3); // a, b, ab
    }

    #[test]
    fn banana_contains() {
        let st = SuffixTree::new(b"banana");
        assert!(st.contains(b"ban"));
        assert!(st.contains(b"ana"));
        assert!(st.contains(b"nan"));
        assert!(st.contains(b"na"));
        assert!(!st.contains(b"nb"));
        assert!(!st.contains(b"bana_")); // underscore not in "banana"
        assert!(st.contains(b"banana"));
        assert!(st.contains(b""));
    }

    #[test]
    fn banana_distinct_count_matches_brute_force() {
        let text = b"banana";
        let st = SuffixTree::new(text);
        assert_eq!(
            st.distinct_substring_count(),
            brute_distinct_substrings(text),
            "distinct substring count mismatch for \"banana\""
        );
    }

    #[test]
    fn mississippi_distinct_count() {
        let text = b"mississippi";
        let st = SuffixTree::new(text);
        assert_eq!(
            st.distinct_substring_count(),
            brute_distinct_substrings(text),
        );
    }

    #[test]
    fn all_same_chars() {
        for n in 1_usize..=8 {
            let text = vec![b'a'; n];
            let st = SuffixTree::new(&text);
            assert_eq!(
                st.distinct_substring_count(),
                n as u64,
                "all-same text of length {n}"
            );
            assert!(st.contains(&text));
            assert!(st.contains(b"a"));
        }
    }

    #[test]
    fn abracadabra() {
        let text = b"abracadabra";
        let st = SuffixTree::new(text);
        assert_eq!(
            st.distinct_substring_count(),
            brute_distinct_substrings(text),
        );
    }

    // ------------------------------------------------------------------
    // Exhaustive brute-force check for short strings
    // ------------------------------------------------------------------

    #[test]
    fn exhaustive_short_strings() {
        let short_texts: &[&[u8]] = &[
            b"a", b"ab", b"aa", b"abc", b"aab", b"aba", b"abab", b"abba", b"banana", b"abcde",
            b"aaaaaa",
        ];
        for &text in short_texts {
            let st = SuffixTree::new(text);
            // Distinct count.
            assert_eq!(
                st.distinct_substring_count(),
                brute_distinct_substrings(text),
                "distinct count mismatch for {text:?}"
            );
            // contains() agrees with brute force for every substring and some
            // non-substrings.
            for i in 0..text.len() {
                for j in (i + 1)..=text.len() {
                    let sub = &text[i..j];
                    assert!(
                        st.contains(sub),
                        "contains({sub:?}) should be true for text {text:?}"
                    );
                }
            }
            // Non-occurrences.
            for &bad in &[b"xyz".as_slice(), b"\xff\xfe"] {
                assert_eq!(
                    st.contains(bad),
                    brute_contains(text, bad),
                    "contains({bad:?}) mismatch for {text:?}"
                );
            }
        }
    }

    // ------------------------------------------------------------------
    // QuickCheck property tests
    // ------------------------------------------------------------------

    #[allow(clippy::needless_pass_by_value)]
    #[quickcheck_macros::quickcheck]
    fn quickcheck_contains_matches_brute_force(text: Vec<u8>, pattern: Vec<u8>) -> bool {
        // Keep sizes reasonable to avoid very slow brute-force on large inputs.
        let text: Vec<u8> = text.into_iter().map(|b| b % 5).take(40).collect();
        let pattern: Vec<u8> = pattern.into_iter().map(|b| b % 5).take(10).collect();
        let st = SuffixTree::new(&text);
        st.contains(&pattern) == brute_contains(&text, &pattern)
    }

    #[allow(clippy::needless_pass_by_value)]
    #[quickcheck_macros::quickcheck]
    fn quickcheck_distinct_count_matches_brute_force(text: Vec<u8>) -> bool {
        let text: Vec<u8> = text.into_iter().map(|b| b % 5).take(15).collect();
        let st = SuffixTree::new(&text);
        st.distinct_substring_count() == brute_distinct_substrings(&text)
    }

    #[allow(clippy::needless_pass_by_value)]
    #[quickcheck_macros::quickcheck]
    fn quickcheck_all_substrings_found(text: Vec<u8>) -> bool {
        let text: Vec<u8> = text.into_iter().map(|b| b % 5).take(20).collect();
        let st = SuffixTree::new(&text);
        for i in 0..text.len() {
            for j in (i + 1)..=text.len() {
                if !st.contains(&text[i..j]) {
                    return false;
                }
            }
        }
        true
    }
}
