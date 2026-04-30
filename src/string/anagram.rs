//! Anagram detection via canonical signatures.
//!
//! Two strings are anagrams iff they contain the same multiset of
//! characters. The approach taken here is the *signature* method: map
//! each input to a canonical form by sorting its Unicode scalars, then
//! compare signatures. Two strings are anagrams iff their signatures are
//! equal.
//!
//! ```text
//! signature("listen") == signature("silent") == "eilnst"
//! ```
//!
//! # Complexity
//!
//! [`anagram_signature`] runs in `O(n log n)` time and `O(n)` space,
//! where `n` is the number of Unicode scalar values in the input — the
//! cost is dominated by the sort. [`are_anagrams`] is two signature
//! computations plus a `String` equality check, so it is also
//! `O(n log n)`.
//!
//! # Normalization policy
//!
//! No normalization is performed: the signature is built from the raw
//! `chars()` of the input. Concretely this means
//!
//! - **case-sensitive**: `"ABC"` and `"cba"` are *not* anagrams,
//! - **whitespace counts**: `"a b"` and `"ab"` are *not* anagrams,
//! - **Unicode-aware** at the scalar level: `"résumé"` and `"éumésr"`
//!   are anagrams because they contain the same multiset of `char`s.
//!
//! Callers that want case-insensitive or whitespace-stripped semantics
//! should preprocess their inputs before calling these functions.

/// Return the canonical anagram signature of `s`: its `chars()` collected
/// into a `String`, sorted in ascending Unicode-scalar order.
///
/// Two inputs are anagrams iff their signatures are equal. See the module
/// docs for the (deliberately minimal) normalization policy.
///
/// # Complexity
///
/// `O(n log n)` time, `O(n)` space, where `n = s.chars().count()`.
pub fn anagram_signature(s: &str) -> String {
    let mut chars: Vec<char> = s.chars().collect();
    chars.sort_unstable();
    chars.into_iter().collect()
}

/// Return `true` iff `a` and `b` are anagrams of each other under the
/// raw-`chars()` policy documented at the module level.
///
/// # Complexity
///
/// `O(n log n)` time, `O(n)` space.
pub fn are_anagrams(a: &str, b: &str) -> bool {
    anagram_signature(a) == anagram_signature(b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use quickcheck_macros::quickcheck;

    // ---- anagram_signature ----

    #[test]
    fn signature_empty() {
        assert_eq!(anagram_signature(""), "");
    }

    #[test]
    fn signature_single_char() {
        assert_eq!(anagram_signature("a"), "a");
    }

    #[test]
    fn signature_sorts_chars() {
        assert_eq!(anagram_signature("listen"), "eilnst");
        assert_eq!(anagram_signature("silent"), "eilnst");
    }

    #[test]
    fn signature_is_idempotent() {
        let sig = anagram_signature("banana");
        assert_eq!(anagram_signature(&sig), sig);
    }

    // ---- are_anagrams ----

    #[test]
    fn anagrams_empty_pair() {
        assert!(are_anagrams("", ""));
    }

    #[test]
    fn anagrams_single_char_match() {
        assert!(are_anagrams("a", "a"));
    }

    #[test]
    fn anagrams_single_char_mismatch() {
        assert!(!are_anagrams("a", "b"));
    }

    #[test]
    fn anagrams_simple_pair() {
        assert!(are_anagrams("listen", "silent"));
    }

    #[test]
    fn anagrams_case_sensitive() {
        // The policy is "raw chars, no normalization", so case differs.
        assert!(!are_anagrams("ABC", "cba"));
    }

    #[test]
    fn anagrams_unicode() {
        assert!(are_anagrams("résumé", "éumésr"));
    }

    #[test]
    fn anagrams_different_lengths() {
        assert!(!are_anagrams("abc", "abcd"));
    }

    #[test]
    fn anagrams_spaces_count() {
        // Whitespace is part of the multiset, so the signatures differ.
        assert!(!are_anagrams("a b", "ab"));
        // ...but two strings with the same whitespace multiset do match.
        assert!(are_anagrams("a b c", "c b a"));
    }

    #[test]
    fn anagrams_self() {
        assert!(are_anagrams("banana", "banana"));
    }

    #[test]
    fn anagrams_repeated_chars_count() {
        // Same letters but different multiplicities → not anagrams.
        assert!(!are_anagrams("aab", "abb"));
    }

    // ---- property tests ----

    #[quickcheck]
    fn signature_invariant_under_reverse(mut s: String) -> bool {
        // Take the original signature, then mutate `s` in place to its
        // reverse and compare. Mutating-in-place satisfies clippy's
        // `needless_pass_by_value` lint, matching the pattern already used
        // by the RLE round-trip property test.
        let original_sig = anagram_signature(&s);
        let reversed: String = s.chars().rev().collect();
        s.clear();
        s.push_str(&reversed);
        anagram_signature(&s) == original_sig
    }

    #[quickcheck]
    fn reverse_is_always_an_anagram(mut s: String) -> bool {
        let reversed: String = s.chars().rev().collect();
        let copy = s.clone();
        s.clear();
        s.push_str(&reversed);
        are_anagrams(&copy, &s)
    }

    #[quickcheck]
    fn signature_is_a_permutation_of_input(mut s: String) -> bool {
        // The signature must contain exactly the same multiset of chars
        // as the input — only the order changes.
        let sig = anagram_signature(&s);
        let mut input_chars: Vec<char> = s.chars().collect();
        let mut sig_chars: Vec<char> = sig.chars().collect();
        s.clear();
        input_chars.sort_unstable();
        sig_chars.sort_unstable();
        input_chars == sig_chars
    }
}
