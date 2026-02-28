//! Trie (prefix tree) over `&str`. `insert` / `contains` / `starts_with` in
//! O(L) per operation, where L is the key length.

use std::collections::HashMap;

/// A simple trie storing string keys.
pub struct Trie {
    root: TrieNode,
}

#[derive(Default)]
struct TrieNode {
    children: HashMap<char, Self>,
    is_end: bool,
}

impl Trie {
    /// Creates an empty trie.
    pub fn new() -> Self {
        Self {
            root: TrieNode::default(),
        }
    }

    /// Inserts `word` into the trie. Idempotent.
    pub fn insert(&mut self, word: &str) {
        let mut node = &mut self.root;
        for ch in word.chars() {
            node = node.children.entry(ch).or_default();
        }
        node.is_end = true;
    }

    /// Returns `true` if `word` was inserted previously.
    pub fn contains(&self, word: &str) -> bool {
        self.find_node(word).is_some_and(|n| n.is_end)
    }

    /// Returns `true` if any inserted word starts with `prefix`.
    pub fn starts_with(&self, prefix: &str) -> bool {
        self.find_node(prefix).is_some()
    }

    fn find_node(&self, key: &str) -> Option<&TrieNode> {
        let mut node = &self.root;
        for ch in key.chars() {
            node = node.children.get(&ch)?;
        }
        Some(node)
    }
}

impl Default for Trie {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::Trie;

    #[test]
    fn empty_trie() {
        let t = Trie::new();
        assert!(!t.contains("anything"));
        assert!(!t.starts_with("any"));
        assert!(t.starts_with("")); // every trie starts_with the empty prefix
    }

    #[test]
    fn insert_and_contains() {
        let mut t = Trie::new();
        t.insert("apple");
        assert!(t.contains("apple"));
        assert!(!t.contains("app"));
        assert!(t.starts_with("app"));
    }

    #[test]
    fn shared_prefix() {
        let mut t = Trie::new();
        t.insert("car");
        t.insert("cart");
        t.insert("cargo");
        assert!(t.contains("car"));
        assert!(t.contains("cart"));
        assert!(t.contains("cargo"));
        assert!(t.starts_with("ca"));
        assert!(!t.contains("carp"));
    }

    #[test]
    fn unicode_keys() {
        let mut t = Trie::new();
        t.insert("café");
        assert!(t.contains("café"));
        assert!(t.starts_with("caf"));
        assert!(!t.contains("cafe"));
    }

    #[test]
    fn duplicate_insert_is_idempotent() {
        let mut t = Trie::new();
        t.insert("a");
        t.insert("a");
        assert!(t.contains("a"));
    }

    #[test]
    fn empty_string_is_member_after_insert() {
        let mut t = Trie::new();
        assert!(!t.contains(""));
        t.insert("");
        assert!(t.contains(""));
    }
}
