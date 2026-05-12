//! Trie-based autocomplete with optional top-k frequency ranking.
//!
//! Stores a multiset of words in a trie keyed by Unicode `char`s. Each
//! terminal node remembers the cumulative insertion frequency for the word
//! that ends there, so [`TrieAutocomplete::insert`] is additive — calling it
//! repeatedly with the same word increases that word's score.
//!
//! [`TrieAutocomplete::complete`] navigates the trie to the node matching
//! the query prefix (returning an empty vector if no such path exists),
//! then performs a DFS over the subtree to gather every completion together
//! with its frequency. The collected list is sorted by frequency descending,
//! breaking ties lexicographically ascending, and the top `k` words are
//! returned.
//!
//! Children are stored in a [`BTreeMap`], which gives stable, sorted-by-char
//! iteration. That keeps the DFS output reproducible and provides a natural
//! lex order for tie-breaking before the final sort.
//!
//! # Complexity
//!
//! Let `L` be the length (in `char`s) of an inserted word and `P` the length
//! of a query prefix.
//!
//! - `insert`:  `O(L * log A)` time, where `A` is the per-node alphabet size
//!   (the `BTreeMap` lookup cost). Memory is `O(L)` worst case for a brand
//!   new path.
//! - `complete(prefix, k)`:  `O(P * log A)` to locate the prefix node, plus
//!   `O(C)` to enumerate the `C` completions in the subtree, plus
//!   `O(C log C)` to sort them. The final `take(k)` is `O(k)`.
//!
//! # Examples
//!
//! ```
//! use rust_algorithms::string::trie_autocomplete::TrieAutocomplete;
//!
//! let mut ac = TrieAutocomplete::new();
//! ac.insert("car", 5);
//! ac.insert("cart", 2);
//! ac.insert("care", 7);
//! ac.insert("dog", 3);
//!
//! // Top 2 completions of "car" — by frequency descending.
//! assert_eq!(ac.complete("car", 2), vec!["care".to_string(), "car".to_string()]);
//!
//! // Prefix that matches nothing returns empty.
//! assert!(ac.complete("zzz", 5).is_empty());
//! ```

use std::collections::BTreeMap;

/// One node in the autocomplete trie.
///
/// `children` is keyed on `char` so iteration is sorted and deterministic,
/// which gives us a free lex tie-break before the final sort.
struct TrieNode {
    children: BTreeMap<char, Self>,
    /// Cumulative frequency of the word that ends exactly here, if any.
    /// `0` (the default) means no word terminates at this node.
    count: u64,
}

impl TrieNode {
    const fn new() -> Self {
        Self {
            children: BTreeMap::new(),
            count: 0,
        }
    }
}

/// A trie that supports prefix-based autocomplete with frequency ranking.
///
/// Build with [`TrieAutocomplete::new`] + [`TrieAutocomplete::insert`].
/// Query with [`TrieAutocomplete::complete`].
pub struct TrieAutocomplete {
    root: TrieNode,
}

impl TrieAutocomplete {
    /// Creates an empty autocomplete index.
    pub const fn new() -> Self {
        Self {
            root: TrieNode::new(),
        }
    }

    /// Adds `count` to the frequency of `word` (inserting the path if
    /// necessary).
    ///
    /// Calling `insert` repeatedly with the same word accumulates: e.g.
    /// `insert("foo", 2)` followed by `insert("foo", 3)` leaves "foo" with
    /// total frequency `5`. Calling with `count == 0` still ensures the
    /// path exists but, because the terminal counter stays at `0`, the
    /// word will not be reported as a completion until a non-zero insert
    /// arrives.
    pub fn insert(&mut self, word: &str, count: u64) {
        let mut node = &mut self.root;
        for ch in word.chars() {
            node = node.children.entry(ch).or_insert_with(TrieNode::new);
        }
        node.count = node.count.saturating_add(count);
    }

    /// Returns the top-`k` completions of `prefix`, ordered by frequency
    /// descending and breaking ties by lexicographic ascending order.
    ///
    /// A word qualifies as a completion of `prefix` iff `prefix` is a prefix
    /// of the word (the word itself counts when it equals the prefix exactly).
    /// Words inserted with cumulative frequency `0` are skipped.
    ///
    /// `k == 0` returns an empty vector. `k` larger than the number of
    /// available completions returns all of them.
    pub fn complete(&self, prefix: &str, k: usize) -> Vec<String> {
        if k == 0 {
            return Vec::new();
        }

        // Walk to the node representing `prefix`. Bail out if any character
        // along the way is missing.
        let mut node = &self.root;
        for ch in prefix.chars() {
            match node.children.get(&ch) {
                Some(child) => node = child,
                None => return Vec::new(),
            }
        }

        // DFS the subtree, accumulating (word, frequency) pairs.
        let mut buf: String = prefix.to_string();
        let mut out: Vec<(String, u64)> = Vec::new();
        Self::collect(node, &mut buf, &mut out);

        // Frequency descending, then lex ascending on ties.
        out.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        out.into_iter().take(k).map(|(w, _)| w).collect()
    }

    /// Recursive DFS helper. `buf` holds the word formed by the path from
    /// the root to `node` and is mutated in place — characters are pushed
    /// before recursing and popped on the way back out, so the same buffer
    /// is reused for every completion we report.
    fn collect(node: &TrieNode, buf: &mut String, out: &mut Vec<(String, u64)>) {
        if node.count > 0 {
            out.push((buf.clone(), node.count));
        }
        for (&ch, child) in &node.children {
            buf.push(ch);
            Self::collect(child, buf, out);
            buf.pop();
        }
    }
}

impl Default for TrieAutocomplete {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_trie_returns_empty() {
        let ac = TrieAutocomplete::new();
        assert!(ac.complete("", 5).is_empty());
        assert!(ac.complete("anything", 5).is_empty());
    }

    #[test]
    fn single_word_completion() {
        let mut ac = TrieAutocomplete::new();
        ac.insert("hello", 1);
        assert_eq!(ac.complete("he", 5), vec!["hello".to_string()]);
        assert_eq!(ac.complete("hello", 5), vec!["hello".to_string()]);
        assert_eq!(ac.complete("", 5), vec!["hello".to_string()]);
    }

    #[test]
    fn missing_prefix_returns_empty() {
        let mut ac = TrieAutocomplete::new();
        ac.insert("hello", 1);
        assert!(ac.complete("hz", 5).is_empty());
        assert!(ac.complete("helloo", 5).is_empty());
    }

    #[test]
    fn multiple_words_sharing_prefix() {
        let mut ac = TrieAutocomplete::new();
        ac.insert("car", 1);
        ac.insert("cart", 1);
        ac.insert("care", 1);
        ac.insert("dog", 1);

        let mut hits = ac.complete("car", 10);
        hits.sort(); // equal frequencies — order is just lex
        assert_eq!(
            hits,
            vec!["car".to_string(), "care".to_string(), "cart".to_string()]
        );

        // Prefix "do" should only see "dog".
        assert_eq!(ac.complete("do", 10), vec!["dog".to_string()]);
    }

    #[test]
    fn k_zero_returns_empty() {
        let mut ac = TrieAutocomplete::new();
        ac.insert("a", 1);
        ac.insert("ab", 2);
        assert!(ac.complete("a", 0).is_empty());
    }

    #[test]
    fn k_larger_than_completions_returns_all() {
        let mut ac = TrieAutocomplete::new();
        ac.insert("apple", 3);
        ac.insert("apply", 2);
        let hits = ac.complete("app", 100);
        assert_eq!(hits.len(), 2);
        assert!(hits.contains(&"apple".to_string()));
        assert!(hits.contains(&"apply".to_string()));
    }

    #[test]
    fn frequency_ranking_descending() {
        let mut ac = TrieAutocomplete::new();
        ac.insert("car", 5);
        ac.insert("cart", 2);
        ac.insert("care", 7);
        ac.insert("cared", 1);

        // Highest frequency first.
        assert_eq!(
            ac.complete("car", 4),
            vec![
                "care".to_string(),
                "car".to_string(),
                "cart".to_string(),
                "cared".to_string(),
            ]
        );

        // Top-2 cuts off the tail.
        assert_eq!(
            ac.complete("car", 2),
            vec!["care".to_string(), "car".to_string()]
        );
    }

    #[test]
    fn tie_break_lex_ascending() {
        let mut ac = TrieAutocomplete::new();
        // Same frequency — must be returned in lex ascending order.
        ac.insert("banana", 3);
        ac.insert("apple", 3);
        ac.insert("cherry", 3);

        assert_eq!(
            ac.complete("", 3),
            vec![
                "apple".to_string(),
                "banana".to_string(),
                "cherry".to_string(),
            ]
        );
    }

    #[test]
    fn insert_accumulates_frequency() {
        let mut ac = TrieAutocomplete::new();
        ac.insert("foo", 2);
        ac.insert("foo", 3);
        ac.insert("bar", 4);

        // "foo" should now beat "bar" (5 vs 4).
        assert_eq!(
            ac.complete("", 2),
            vec!["foo".to_string(), "bar".to_string()]
        );
    }

    #[test]
    fn zero_count_path_is_not_a_completion() {
        let mut ac = TrieAutocomplete::new();
        // Path exists but no word terminates with positive frequency.
        ac.insert("xyz", 0);
        assert!(ac.complete("x", 5).is_empty());
        assert!(ac.complete("xyz", 5).is_empty());

        // Now make it a real word and confirm it shows up.
        ac.insert("xyz", 4);
        assert_eq!(ac.complete("x", 5), vec!["xyz".to_string()]);
    }

    #[test]
    fn unicode_prefix_handled() {
        let mut ac = TrieAutocomplete::new();
        ac.insert("café", 2);
        ac.insert("cafeteria", 1);
        let hits = ac.complete("caf", 5);
        assert!(hits.contains(&"café".to_string()));
        assert!(hits.contains(&"cafeteria".to_string()));
        // "café" has higher frequency, so it ranks first.
        assert_eq!(hits[0], "café".to_string());
    }
}
