//! Eertree (palindromic tree): an online data structure that maintains every
//! distinct palindromic substring of a byte string built up one character at
//! a time.
//!
//! The structure was introduced by Mikhail Rubinchik (2014) and has the
//! striking property that the total number of distinct palindromic substrings
//! of any string of length `n` is at most `n + 1` — so an Eertree built over
//! such a string has at most `n + 2` nodes (the two roots plus one node per
//! distinct palindrome).
//!
//! Each node represents a unique palindrome and stores
//!
//! - its length,
//! - a *suffix link* to the longest proper palindromic suffix of the node,
//! - transitions keyed by the byte that is added on both ends to produce a
//!   longer palindrome,
//! - an occurrence count (number of times this palindrome appears as a
//!   suffix of a prefix during construction; summing along suffix links gives
//!   total occurrences).
//!
//! Two roots are always present:
//!
//! - the *imaginary* root with length `-1` (index 0): every single character
//!   added to it produces a length-1 palindrome,
//! - the *empty* root with length `0` (index 1): every pair of identical
//!   surrounding characters produces a length-2 palindrome.
//!
//! # Applications
//!
//! - Counting distinct palindromic substrings in `O(n)` total time.
//! - Counting total palindromic substring occurrences (with suffix-link
//!   aggregation).
//! - Finding the longest palindromic suffix of every prefix online.
//! - Building palindromic factorizations.
//!
//! # Complexity
//!
//! - Construction: `O(n)` amortized for an alphabet of constant size, or
//!   `O(n log Σ)` / `O(n)` expected with a `HashMap` for transitions over an
//!   arbitrary byte alphabet (this implementation uses `HashMap<u8, usize>`).
//! - Space: `O(n)` — at most `n + 2` nodes for an input of length `n`.

use std::collections::HashMap;

/// A single node of the palindromic tree.
///
/// Every node corresponds to a distinct palindromic substring of the source
/// (with two synthetic roots, see the module docs).
#[derive(Debug, Clone)]
pub struct EertreeNode {
    /// Length of the palindrome this node represents. The imaginary root has
    /// length `-1`; the empty root has length `0`; every other node has
    /// length `>= 1`.
    pub length: i32,
    /// Index of the node representing the longest proper palindromic suffix
    /// of this palindrome. The two roots both link to the imaginary root.
    pub suffix_link: usize,
    /// Outgoing edges keyed by the byte added on both ends.
    pub transitions: HashMap<u8, usize>,
    /// Number of times this palindrome appears as the longest palindromic
    /// suffix of some prefix during construction. Summing along suffix links
    /// yields the total occurrence count of the palindrome in the source.
    pub count: u64,
}

impl EertreeNode {
    fn new(length: i32, suffix_link: usize) -> Self {
        Self {
            length,
            suffix_link,
            transitions: HashMap::new(),
            count: 0,
        }
    }
}

/// Online palindromic tree (Eertree) over a byte string.
///
/// Build by repeatedly calling [`Eertree::extend`] (or [`Eertree::extend_str`])
/// to append bytes; the structure incrementally tracks every distinct
/// palindromic substring of the bytes seen so far.
#[derive(Debug, Clone)]
pub struct Eertree {
    /// All nodes; index 0 is the imaginary root (length `-1`) and index 1 is
    /// the empty root (length `0`).
    nodes: Vec<EertreeNode>,
    /// Index of the node representing the longest palindromic suffix of the
    /// source so far.
    last: usize,
    /// All bytes appended via [`Eertree::extend`], in order.
    source: Vec<u8>,
}

impl Default for Eertree {
    fn default() -> Self {
        Self::new()
    }
}

impl Eertree {
    /// Creates an empty palindromic tree with the two synthetic roots in
    /// place.
    #[must_use]
    pub fn new() -> Self {
        // Imaginary root (length -1) at index 0, empty root (length 0) at
        // index 1. Both suffix links point at the imaginary root: from the
        // empty root, that is the conventional choice; from the imaginary
        // root, the link is never followed (length -1 is the fallback case
        // in `extend`).
        let imaginary = EertreeNode::new(-1, 0);
        let empty = EertreeNode::new(0, 0);
        Self {
            nodes: vec![imaginary, empty],
            last: 1,
            source: Vec::new(),
        }
    }

    /// Returns the number of *distinct* palindromic substrings of the source
    /// seen so far.
    ///
    /// This is simply the node count minus the two synthetic roots.
    #[must_use]
    pub const fn distinct_palindrome_count(&self) -> usize {
        self.nodes.len() - 2
    }

    /// Returns the source bytes that have been appended so far.
    #[must_use]
    pub fn source(&self) -> &[u8] {
        &self.source
    }

    /// Returns the underlying nodes. Index 0 is the imaginary root and index
    /// 1 is the empty root; everything from index 2 onward represents a
    /// distinct non-empty palindromic substring.
    #[must_use]
    pub fn nodes(&self) -> &[EertreeNode] {
        &self.nodes
    }

    /// Walks the suffix-link chain from `start` until it finds a node `v`
    /// such that the character at position `pos - len(v) - 1` in the source
    /// equals `c`, i.e. extending `v` by `c` on both ends stays inside the
    /// already-built source. Returns the index of that node.
    ///
    /// `pos` is the index in `self.source` of the byte just appended (so
    /// `self.source.len() - 1` at the call site).
    fn find_extendable_link(&self, start: usize, pos: usize, c: u8) -> usize {
        let mut cur = start;
        loop {
            let len = self.nodes[cur].length;
            // Position to compare: pos - len - 1. With len possibly -1, this
            // works out via signed arithmetic.
            let idx = pos as i32 - len - 1;
            if idx >= 0 && self.source[idx as usize] == c {
                return cur;
            }
            cur = self.nodes[cur].suffix_link;
        }
    }

    /// Appends a single byte to the source and updates the tree online.
    pub fn extend(&mut self, c: u8) {
        self.source.push(c);
        let pos = self.source.len() - 1;

        // Find the largest palindromic suffix that can be extended by `c`.
        let parent = self.find_extendable_link(self.last, pos, c);

        // If we already have an edge for `c` from `parent`, reuse it.
        if let Some(&existing) = self.nodes[parent].transitions.get(&c) {
            self.last = existing;
            self.nodes[existing].count += 1;
            return;
        }

        // Otherwise create a new node for the new palindrome.
        let new_length = self.nodes[parent].length + 2;
        let new_index = self.nodes.len();

        // Suffix link of the new node:
        // - if the new palindrome has length 1, it is a single character
        //   whose only proper palindromic suffix is the empty string;
        // - otherwise, walk the parent's suffix chain to find another
        //   extendable node, and the suffix link is its `c` transition.
        let suffix_link = if new_length == 1 {
            1
        } else {
            let grand_parent = self.find_extendable_link(self.nodes[parent].suffix_link, pos, c);
            // By the Eertree invariant this transition must exist.
            self.nodes[grand_parent].transitions[&c]
        };

        let mut node = EertreeNode::new(new_length, suffix_link);
        node.count = 1;
        self.nodes.push(node);
        self.nodes[parent].transitions.insert(c, new_index);
        self.last = new_index;
    }

    /// Appends every byte of `s` in order. Equivalent to looping
    /// [`Eertree::extend`].
    pub fn extend_str(&mut self, s: &[u8]) {
        for &c in s {
            self.extend(c);
        }
    }

    /// Builds an Eertree over `s` in one shot.
    #[must_use]
    pub fn from_bytes(s: &[u8]) -> Self {
        let mut t = Self::new();
        t.extend_str(s);
        t
    }
}

#[cfg(test)]
mod tests {
    use super::Eertree;
    use quickcheck_macros::quickcheck;
    use std::collections::HashSet;

    /// `O(n^3)` reference: enumerate every substring and keep palindromic
    /// ones, deduplicated.
    fn brute_force_distinct_palindromes(s: &[u8]) -> usize {
        let n = s.len();
        let mut seen: HashSet<&[u8]> = HashSet::new();
        for i in 0..n {
            for j in i + 1..=n {
                let sub = &s[i..j];
                let rev: Vec<u8> = sub.iter().rev().copied().collect();
                if rev.as_slice() == sub {
                    seen.insert(sub);
                }
            }
        }
        seen.len()
    }

    #[test]
    fn empty_input() {
        let t = Eertree::new();
        assert_eq!(t.distinct_palindrome_count(), 0);
        // Two roots are always present.
        assert_eq!(t.nodes().len(), 2);
        assert_eq!(t.nodes()[0].length, -1);
        assert_eq!(t.nodes()[1].length, 0);
    }

    #[test]
    fn single_char() {
        let t = Eertree::from_bytes(b"a");
        // "a" is the only distinct palindrome.
        assert_eq!(t.distinct_palindrome_count(), 1);
    }

    #[test]
    fn aa_two_palindromes() {
        // Distinct palindromes of "aa": "a", "aa" → 2.
        let t = Eertree::from_bytes(b"aa");
        assert_eq!(t.distinct_palindrome_count(), 2);
    }

    #[test]
    fn abba_four_palindromes() {
        // Distinct palindromes of "abba": "a", "b", "bb", "abba" → 4.
        let t = Eertree::from_bytes(b"abba");
        assert_eq!(t.distinct_palindrome_count(), 4);
    }

    #[test]
    fn aabaa_four_palindromes() {
        // Distinct palindromes of "aabaa": "a", "aa", "aba", "aabaa" → 4.
        // Note: "b" is also a palindrome, so the actual count is 5.
        let t = Eertree::from_bytes(b"aabaa");
        assert_eq!(
            t.distinct_palindrome_count(),
            brute_force_distinct_palindromes(b"aabaa")
        );
    }

    #[test]
    fn aabaa_matches_brute_force() {
        // Sanity-check against the brute-force enumerator: { a, b, aa, aba,
        // aabaa } → 5.
        assert_eq!(brute_force_distinct_palindromes(b"aabaa"), 5);
    }

    #[test]
    fn racecar() {
        // { r, a, c, e, cec, aceca, racecar } → 7
        let t = Eertree::from_bytes(b"racecar");
        assert_eq!(
            t.distinct_palindrome_count(),
            brute_force_distinct_palindromes(b"racecar")
        );
        assert_eq!(t.distinct_palindrome_count(), 7);
    }

    #[test]
    fn all_same_chars_run() {
        // "aaaa" → { a, aa, aaa, aaaa } → 4.
        let t = Eertree::from_bytes(b"aaaa");
        assert_eq!(t.distinct_palindrome_count(), 4);
    }

    #[test]
    fn node_count_bound() {
        // The Eertree of any string of length n has at most n + 2 nodes.
        for s in [
            &b""[..],
            b"a",
            b"ab",
            b"aabaa",
            b"abcabcabc",
            b"forgeeksskeegfor",
        ] {
            let t = Eertree::from_bytes(s);
            assert!(
                t.nodes().len() <= s.len() + 2,
                "node count {} exceeded n+2 for input of length {}",
                t.nodes().len(),
                s.len()
            );
        }
    }

    #[test]
    fn extend_matches_extend_str() {
        let mut a = Eertree::new();
        for &c in b"mississippi" {
            a.extend(c);
        }
        let b = Eertree::from_bytes(b"mississippi");
        assert_eq!(a.distinct_palindrome_count(), b.distinct_palindrome_count());
    }

    #[allow(clippy::needless_pass_by_value)]
    #[quickcheck]
    fn quickcheck_matches_brute_force(bytes: Vec<u8>) -> bool {
        // Cap length to keep the O(n^3) reference fast.
        let bytes: Vec<u8> = bytes.into_iter().take(20).collect();
        let t = Eertree::from_bytes(&bytes);
        t.distinct_palindrome_count() == brute_force_distinct_palindromes(&bytes)
    }

    #[allow(clippy::needless_pass_by_value)]
    #[quickcheck]
    fn quickcheck_node_count_bound(bytes: Vec<u8>) -> bool {
        let bytes: Vec<u8> = bytes.into_iter().take(20).collect();
        let t = Eertree::from_bytes(&bytes);
        t.nodes().len() <= bytes.len() + 2
    }
}
