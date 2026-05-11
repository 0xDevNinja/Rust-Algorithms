//! Trie-based spell checker with bounded Levenshtein search.
//!
//! Stores a dictionary in a trie keyed by Unicode `char`s, then enumerates
//! every dictionary word within edit distance `max_dist` of a query by
//! walking the trie and carrying one row of the standard Levenshtein DP
//! table down with the traversal. Whenever the minimum value in the active
//! row exceeds `max_dist`, the entire subtree is pruned — so the search
//! visits only nodes whose prefix is still a viable candidate.
//!
//! Building the trie is `O(N)` in the total number of dictionary characters.
//! A single search is worst-case `O(T * Q)`, where `T` is the trie size and
//! `Q = query.chars().count() + 1` is the row width, but pruning makes it
//! dramatically faster on realistic inputs and small `max_dist`. Memory is
//! `O(N)` for the trie plus `O(D * Q)` per active recursion stack of depth
//! `D` for the DP rows.
//!
//! The structure is `char`-oriented, so it handles arbitrary UTF-8 input
//! correctly (edit distance is measured in codepoints, not bytes).
//!
//! # Examples
//!
//! ```
//! use rust_algorithms::string::trie_spell_checker::SpellChecker;
//!
//! let sc = SpellChecker::from_words(&["cat", "car", "cart", "dog"]);
//! let hits = sc.search("cat", 1);
//! assert!(hits.iter().any(|(w, d)| w == "cat" && *d == 0));
//! assert!(hits.iter().any(|(w, d)| w == "car" && *d == 1));
//! assert!(hits.iter().any(|(w, d)| w == "cart" && *d == 1));
//! ```

use std::collections::BTreeMap;

/// One node in the spell-checker trie.
///
/// `children` is keyed on `char` so iteration order is deterministic, which
/// keeps search output stable for ties in edit distance before the final sort.
struct TrieNode {
    children: BTreeMap<char, Self>,
    /// `Some(word)` if a dictionary word terminates exactly at this node,
    /// otherwise `None`. Storing the full word avoids reconstructing it from
    /// the traversal stack on every match.
    word: Option<String>,
}

impl TrieNode {
    const fn new() -> Self {
        Self {
            children: BTreeMap::new(),
            word: None,
        }
    }
}

/// Trie-backed dictionary supporting bounded edit-distance lookups.
///
/// Build with [`SpellChecker::new`] + [`SpellChecker::insert`] or in one
/// shot via [`SpellChecker::from_words`]. Query with
/// [`SpellChecker::search`].
pub struct SpellChecker {
    root: TrieNode,
}

impl SpellChecker {
    /// Creates an empty spell checker.
    pub const fn new() -> Self {
        Self {
            root: TrieNode::new(),
        }
    }

    /// Inserts `word` into the dictionary.
    ///
    /// Empty strings are accepted and become a match at the root with
    /// distance equal to the query length. Re-inserting the same word is a
    /// no-op beyond walking the existing path.
    pub fn insert(&mut self, word: &str) {
        let mut node = &mut self.root;
        for ch in word.chars() {
            node = node.children.entry(ch).or_insert_with(TrieNode::new);
        }
        if node.word.is_none() {
            node.word = Some(word.to_string());
        }
    }

    /// Convenience constructor building a checker from a slice of words.
    pub fn from_words(words: &[&str]) -> Self {
        let mut sc = Self::new();
        for w in words {
            sc.insert(w);
        }
        sc
    }

    /// Returns every dictionary word within edit distance `max_dist` of
    /// `query`, paired with its Levenshtein distance.
    ///
    /// Results are sorted by ascending distance, then lexicographically by
    /// word. The search uses the classic row-by-row trie + DP technique:
    /// the row for the empty prefix is `[0, 1, 2, ..., Q-1]`, and each
    /// child node extends it by one column based on the inserted character.
    /// Subtrees whose row minimum already exceeds `max_dist` are pruned.
    pub fn search(&self, query: &str, max_dist: usize) -> Vec<(String, usize)> {
        let q_chars: Vec<char> = query.chars().collect();
        let width = q_chars.len() + 1;

        // Initial row: cost of transforming the empty prefix into each
        // prefix of the query is just the prefix length.
        let initial_row: Vec<usize> = (0..width).collect();

        let mut results: Vec<(String, usize)> = Vec::new();

        // Root itself represents the empty string. If the empty word was
        // inserted, it's a candidate too.
        if let Some(w) = &self.root.word {
            let dist = q_chars.len();
            if dist <= max_dist {
                results.push((w.clone(), dist));
            }
        }

        for (&ch, child) in &self.root.children {
            Self::walk(child, ch, &q_chars, &initial_row, max_dist, &mut results);
        }

        results.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        results
    }

    /// Recursive trie walk that maintains the current Levenshtein DP row.
    ///
    /// `prev_row` is the DP row for the parent node; `node_char` is the
    /// character on the edge from the parent to `node`. The new row is
    /// computed in `O(Q)` and then handed to each child.
    fn walk(
        node: &TrieNode,
        node_char: char,
        query: &[char],
        prev_row: &[usize],
        max_dist: usize,
        results: &mut Vec<(String, usize)>,
    ) {
        let width = prev_row.len();
        let mut current_row: Vec<usize> = Vec::with_capacity(width);
        // First column: deleting `depth` characters from the trie prefix
        // to match the empty query prefix — that's prev_row[0] + 1.
        current_row.push(prev_row[0] + 1);

        for col in 1..width {
            let insert_cost = current_row[col - 1] + 1;
            let delete_cost = prev_row[col] + 1;
            let replace_cost = if query[col - 1] == node_char {
                prev_row[col - 1]
            } else {
                prev_row[col - 1] + 1
            };
            current_row.push(insert_cost.min(delete_cost).min(replace_cost));
        }

        // Record a hit if a word terminates here and its distance fits.
        if let Some(w) = &node.word {
            let dist = current_row[width - 1];
            if dist <= max_dist {
                results.push((w.clone(), dist));
            }
        }

        // Prune: if every cell in this row already exceeds max_dist, no
        // descendant can improve, so abandon the subtree.
        let row_min = *current_row.iter().min().unwrap();
        if row_min > max_dist {
            return;
        }

        for (&ch, child) in &node.children {
            Self::walk(child, ch, query, &current_row, max_dist, results);
        }
    }
}

impl Default for SpellChecker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn words_only(hits: &[(String, usize)]) -> Vec<String> {
        hits.iter().map(|(w, _)| w.clone()).collect()
    }

    #[test]
    fn empty_trie_returns_empty() {
        let sc = SpellChecker::new();
        assert!(sc.search("anything", 3).is_empty());
        assert!(sc.search("", 0).is_empty());
    }

    #[test]
    fn exact_match_distance_zero() {
        let sc = SpellChecker::from_words(&["cat", "car", "cart", "dog"]);
        let hits = sc.search("cat", 0);
        assert_eq!(hits, vec![("cat".to_string(), 0)]);
    }

    #[test]
    fn one_substitution() {
        let sc = SpellChecker::from_words(&["cat", "car", "cart", "bat", "dog"]);
        let hits = sc.search("cat", 1);
        let names: Vec<String> = words_only(&hits);
        assert!(names.contains(&"cat".to_string()));
        assert!(names.contains(&"car".to_string())); // substitute t -> r
        assert!(names.contains(&"bat".to_string())); // substitute c -> b
        assert!(names.contains(&"cart".to_string())); // single insertion, still dist 1
        assert!(!names.contains(&"dog".to_string()));
        // distance for cat itself must be zero
        assert_eq!(hits[0], ("cat".to_string(), 0));
    }

    #[test]
    fn one_insertion() {
        // Query is missing a char relative to dictionary entry.
        let sc = SpellChecker::from_words(&["cart", "carts", "card"]);
        let hits = sc.search("car", 1);
        let names = words_only(&hits);
        assert!(names.contains(&"cart".to_string()));
        assert!(names.contains(&"card".to_string()));
        assert!(!names.contains(&"carts".to_string())); // distance 2
        for (_, d) in &hits {
            assert!(*d <= 1);
        }
    }

    #[test]
    fn one_deletion() {
        // Query has an extra char relative to dictionary entry.
        let sc = SpellChecker::from_words(&["cat", "cab", "dog"]);
        let hits = sc.search("cats", 1);
        let names = words_only(&hits);
        assert!(names.contains(&"cat".to_string()));
        assert!(!names.contains(&"cab".to_string())); // distance 2
        assert!(!names.contains(&"dog".to_string()));
    }

    #[test]
    fn max_dist_zero_only_exact() {
        let sc = SpellChecker::from_words(&["hello", "help", "hell", "yellow"]);
        assert_eq!(sc.search("hello", 0), vec![("hello".to_string(), 0)]);
        assert!(sc.search("helloo", 0).is_empty());
        assert!(sc.search("hxllo", 0).is_empty());
    }

    #[test]
    fn results_sorted_by_distance_then_word() {
        let sc = SpellChecker::from_words(&["bat", "cat", "rat", "cab", "car"]);
        let hits = sc.search("cat", 1);
        // Distance 0 first, then alphabetical within distance 1.
        assert_eq!(hits[0], ("cat".to_string(), 0));
        let rest: Vec<&String> = hits[1..].iter().map(|(w, _)| w).collect();
        let mut sorted = rest.clone();
        sorted.sort();
        assert_eq!(rest, sorted);
        for (_, d) in &hits[1..] {
            assert_eq!(*d, 1);
        }
    }

    #[test]
    fn large_dictionary_with_query() {
        let dict = [
            "algorithm",
            "altruism",
            "alpha",
            "beta",
            "gamma",
            "delta",
            "epsilon",
            "zeta",
            "eta",
            "theta",
            "iota",
            "kappa",
            "lambda",
            "mu",
            "nu",
            "xi",
            "omicron",
            "pi",
            "rho",
            "sigma",
            "tau",
            "upsilon",
            "phi",
            "chi",
            "psi",
            "omega",
            "rust",
            "trust",
            "crust",
            "trie",
            "tree",
            "three",
            "thread",
            "threat",
        ];
        let sc = SpellChecker::from_words(&dict);

        // Exact hit + close neighbours.
        let hits = sc.search("trie", 1);
        let names = words_only(&hits);
        assert!(names.contains(&"trie".to_string()));
        assert!(names.contains(&"tree".to_string())); // i -> e

        // Distance 2 expands the candidate set.
        let hits2 = sc.search("trie", 2);
        let names2 = words_only(&hits2);
        assert!(names2.contains(&"three".to_string()));

        // Sanity: every reported distance is within the bound.
        for (_, d) in &hits2 {
            assert!(*d <= 2);
        }

        // Garbage query far from anything should still terminate quickly
        // and respect the bound.
        let hits3 = sc.search("zzzzzzzz", 1);
        assert!(hits3.is_empty());
    }

    #[test]
    fn unicode_codepoints_handled() {
        let sc = SpellChecker::from_words(&["café", "cafe", "cake"]);
        let hits = sc.search("café", 1);
        let names = words_only(&hits);
        assert!(names.contains(&"café".to_string()));
        // "cafe" differs from "café" by one codepoint substitution.
        assert!(names.contains(&"cafe".to_string()));
    }

    #[test]
    fn duplicate_inserts_dont_duplicate_results() {
        let mut sc = SpellChecker::new();
        sc.insert("hello");
        sc.insert("hello");
        let hits = sc.search("hello", 0);
        assert_eq!(hits, vec![("hello".to_string(), 0)]);
    }
}
