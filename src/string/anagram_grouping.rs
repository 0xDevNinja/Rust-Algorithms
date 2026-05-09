//! Group strings by anagram class (signature multimap).
//!
//! Two strings are anagrams iff they share the same multiset of `char`s,
//! i.e. the same canonical *signature* (the input's `chars()` collected
//! and sorted in ascending Unicode-scalar order). [`group_anagrams`]
//! partitions a slice of words into groups, where each group contains
//! exactly the words that share a signature.
//!
//! ```text
//! ["eat", "tea", "tan", "ate", "nat", "bat"]
//!     -> [["eat", "tea", "ate"], ["tan", "nat"], ["bat"]]
//! ```
//!
//! # Order
//!
//! Both the order of groups and the order of words within each group are
//! deterministic and follow the input:
//!
//! - Groups appear in the order their *first* member appears in the input.
//! - Within a group, words appear in their original input order.
//!
//! # Complexity
//!
//! Let `N` be the number of input words and `L_i` the number of Unicode
//! scalars in word `i`. [`group_anagrams`] runs in
//! `O(sum_i L_i log L_i)` time — each word is sorted once to compute its
//! signature — and `O(sum_i L_i)` extra space for the signature map plus
//! the output groups.
//!
//! # Normalization policy
//!
//! No normalization is performed. The signature is built from the raw
//! `chars()` of each input, exactly as in [`super::anagram`]. Concretely:
//!
//! - **case-sensitive**: `"Eat"` and `"tea"` are *not* grouped together,
//! - **whitespace counts**: `"a b"` and `"ab"` land in different groups,
//! - **Unicode-aware** at the scalar level: `"résumé"` and `"éumésr"` are
//!   grouped together.
//!
//! Callers wanting case-insensitive or whitespace-stripped semantics
//! should preprocess inputs before calling.
//!
//! See [`super::anagram`] for the pairwise-comparison variant.

use std::collections::hash_map::Entry;
use std::collections::HashMap;

/// Compute the canonical anagram signature of `s` — its `chars()`
/// collected into a `String` and sorted in ascending Unicode-scalar
/// order. Kept private; this matches the policy of [`super::anagram`].
fn signature(s: &str) -> String {
    let mut chars: Vec<char> = s.chars().collect();
    chars.sort_unstable();
    chars.into_iter().collect()
}

/// Group `words` by anagram class, returning one `Vec<String>` per class.
///
/// Two words land in the same group iff they share the same canonical
/// signature (sorted-`char` form). The grouping is deterministic:
///
/// - Groups are emitted in the order their first member appears in
///   `words` (insertion order).
/// - Within each group, words appear in their original input order.
///
/// The empty input returns an empty `Vec`.
///
/// # Complexity
///
/// `O(sum_i L_i log L_i)` time, `O(sum_i L_i)` extra space, where `L_i`
/// is the number of `chars()` in the `i`-th word.
///
/// # Examples
///
/// ```
/// use rust_algorithms::string::anagram_grouping::group_anagrams;
///
/// let words: Vec<String> =
///     ["eat", "tea", "tan", "ate", "nat", "bat"]
///         .iter()
///         .map(|s| s.to_string())
///         .collect();
/// let groups = group_anagrams(&words);
/// assert_eq!(
///     groups,
///     vec![
///         vec!["eat".to_string(), "tea".to_string(), "ate".to_string()],
///         vec!["tan".to_string(), "nat".to_string()],
///         vec!["bat".to_string()],
///     ]
/// );
/// ```
pub fn group_anagrams(words: &[String]) -> Vec<Vec<String>> {
    // `order` records the insertion order of signatures so that the final
    // output respects "first occurrence wins". `index_of` maps a signature
    // to its slot in `groups`, giving O(1) append per word.
    let mut groups: Vec<Vec<String>> = Vec::new();
    let mut index_of: HashMap<String, usize> = HashMap::new();

    for word in words {
        let sig = signature(word);
        match index_of.entry(sig) {
            Entry::Occupied(slot) => {
                groups[*slot.get()].push(word.clone());
            }
            Entry::Vacant(slot) => {
                slot.insert(groups.len());
                groups.push(vec![word.clone()]);
            }
        }
    }

    groups
}

#[cfg(test)]
mod tests {
    use super::*;
    use quickcheck_macros::quickcheck;
    use std::collections::HashSet;

    fn to_strings<const N: usize>(arr: [&str; N]) -> Vec<String> {
        arr.iter().map(|s| (*s).to_string()).collect()
    }

    #[test]
    fn empty_input_yields_empty_output() {
        let words: Vec<String> = Vec::new();
        assert!(group_anagrams(&words).is_empty());
    }

    #[test]
    fn single_word_yields_single_group() {
        let words = to_strings(["hello"]);
        assert_eq!(group_anagrams(&words), vec![vec!["hello".to_string()]]);
    }

    #[test]
    fn canonical_example() {
        let words = to_strings(["eat", "tea", "tan", "ate", "nat", "bat"]);
        let expected: Vec<Vec<String>> = vec![
            to_strings(["eat", "tea", "ate"]),
            to_strings(["tan", "nat"]),
            to_strings(["bat"]),
        ];
        assert_eq!(group_anagrams(&words), expected);
    }

    #[test]
    fn duplicates_stay_in_same_group_in_order() {
        // Identical inputs are trivially anagrams of each other.
        let words = to_strings(["ab", "ba", "ab"]);
        assert_eq!(group_anagrams(&words), vec![to_strings(["ab", "ba", "ab"])]);
    }

    #[test]
    fn empty_string_is_its_own_class() {
        // The empty string has the empty signature; it should not collide
        // with any non-empty word.
        let words = to_strings(["", "a", ""]);
        assert_eq!(
            group_anagrams(&words),
            vec![to_strings(["", ""]), to_strings(["a"])]
        );
    }

    #[test]
    fn case_sensitive() {
        // Documented policy: raw chars, so 'E' != 'e'. "Eat" and "tea"
        // therefore land in different groups.
        let words = to_strings(["Eat", "tea", "ate"]);
        assert_eq!(
            group_anagrams(&words),
            vec![to_strings(["Eat"]), to_strings(["tea", "ate"])]
        );
    }

    #[test]
    fn unicode_strings_group_correctly() {
        let words = to_strings(["résumé", "éumésr", "naïve", "vïane", "résumés"]);
        let groups = group_anagrams(&words);
        assert_eq!(
            groups,
            vec![
                to_strings(["résumé", "éumésr"]),
                to_strings(["naïve", "vïane"]),
                to_strings(["résumés"]),
            ]
        );
    }

    #[test]
    fn whitespace_counts_in_signature() {
        // " ab" and "ab" differ by a space; they must not be grouped.
        let words = to_strings([" ab", "ab", "ba ", "b a"]);
        assert_eq!(
            group_anagrams(&words),
            vec![to_strings([" ab", "ba ", "b a"]), to_strings(["ab"])]
        );
    }

    #[test]
    fn group_order_follows_first_occurrence() {
        // "bat" appears before any 'eat'-class word, so its group must
        // come first in the output.
        let words = to_strings(["bat", "tan", "eat", "nat", "tea"]);
        assert_eq!(
            group_anagrams(&words),
            vec![
                to_strings(["bat"]),
                to_strings(["tan", "nat"]),
                to_strings(["eat", "tea"]),
            ]
        );
    }

    // ---- property tests ----

    #[quickcheck]
    fn every_word_appears_in_exactly_one_group(mut words: Vec<String>) -> bool {
        let groups = group_anagrams(&words);

        // Total word count is preserved.
        let total: usize = groups.iter().map(Vec::len).sum();
        if total != words.len() {
            return false;
        }

        // Each input word appears in exactly one group, with multiplicity
        // matching the input. Compare multisets via sorted clones, then
        // drain the original to consume `words` (silences
        // `needless_pass_by_value`, matching the pattern in `anagram.rs`).
        let mut flat: Vec<String> = groups.into_iter().flatten().collect();
        let mut original: Vec<String> = words.clone();
        words.clear();
        flat.sort_unstable();
        original.sort_unstable();
        flat == original
    }

    #[quickcheck]
    fn groups_are_anagram_classes(mut words: Vec<String>) -> bool {
        let groups = group_anagrams(&words);
        words.clear();

        // Every word inside a group shares the group's signature, and no
        // two groups share a signature.
        let mut seen_sigs: HashSet<String> = HashSet::new();
        for group in &groups {
            if group.is_empty() {
                return false;
            }
            let sig = signature(&group[0]);
            if !seen_sigs.insert(sig.clone()) {
                return false;
            }
            if !group.iter().all(|w| signature(w) == sig) {
                return false;
            }
        }
        true
    }

    #[quickcheck]
    fn within_group_order_matches_input(mut words: Vec<String>) -> bool {
        let groups = group_anagrams(&words);

        // For each group, the sequence of words must be a subsequence of
        // `words` in input order. Walk both pointers in lockstep.
        for group in &groups {
            let mut it = words.iter();
            for w in group {
                if !it.any(|candidate| candidate == w) {
                    return false;
                }
            }
        }
        words.clear();
        true
    }
}
