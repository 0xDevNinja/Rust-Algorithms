//! Longest repeated substring via suffix array + LCP.
//!
//! Returns the longest substring that occurs at least twice in the input.
//! The classical observation: any substring that repeats is a common prefix
//! of two suffixes, so it appears as a prefix of two adjacent suffixes in
//! suffix-array order. Therefore the answer length is `max(lcp)`, and the
//! substring itself is the corresponding prefix of either of those suffixes.
//!
//! # Tie-break
//! When several substrings of the same maximum length repeat, this function
//! returns the **lexicographically smallest** one. Because the suffix array
//! sorts suffixes lexicographically, scanning the LCP array left-to-right
//! and keeping the **first** index that attains the maximum yields the
//! smallest prefix of that length.
//!
//! # Complexity
//! - Time:  O(n log² n) — dominated by the suffix-array construction; the
//!   Kasai LCP pass and the final argmax scan are O(n).
//! - Space: O(n) for the suffix array, the LCP array, and the result.
//!
//! Reuses [`crate::string::suffix_array::suffix_array`] and
//! [`crate::string::lcp_kasai::lcp_kasai`] — see those modules for details.

use crate::string::lcp_kasai::lcp_kasai;
use crate::string::suffix_array::suffix_array;

/// Returns the longest substring of `s` that appears at least twice, or the
/// empty string when no character repeats.
///
/// When several distinct substrings tie for the maximum length, the
/// lexicographically smallest one is returned. The result is always a valid
/// UTF-8 slice when the input is valid UTF-8 **and** the maximum-LCP boundary
/// falls on a character boundary; for arbitrary `&str` input that is not
/// guaranteed in general, so this function operates on byte indices and
/// returns a byte-wise copy via `String::from_utf8_lossy` would lose data —
/// instead we slice the original `&str` and rely on the caller passing ASCII
/// or UTF-8 inputs whose repeats fall on character boundaries (which is the
/// case for every all-ASCII input, including every test in this crate).
pub fn longest_repeated_substring(s: &str) -> String {
    let bytes = s.as_bytes();
    let n = bytes.len();
    if n < 2 {
        return String::new();
    }

    let sa = suffix_array(bytes);
    let lcp = lcp_kasai(bytes, &sa);

    // Walk the LCP array; remember the first index that attains the max.
    // Because `sa` is sorted lexicographically, the earliest index with the
    // maximum LCP corresponds to the lex-smallest substring of that length.
    let mut best_len: usize = 0;
    let mut best_start: usize = 0;
    for i in 1..n {
        if lcp[i] > best_len {
            best_len = lcp[i];
            best_start = sa[i];
        }
    }

    if best_len == 0 {
        return String::new();
    }

    // Safe slice: `best_start + best_len <= n` because `best_len = lcp[i]`
    // is the LCP of the suffix starting at `sa[i] = best_start` with another
    // suffix, so at least `best_len` bytes follow `best_start` in the input.
    String::from_utf8_lossy(&bytes[best_start..best_start + best_len]).into_owned()
}

#[cfg(test)]
mod tests {
    use super::longest_repeated_substring;
    use quickcheck_macros::quickcheck;

    /// Brute-force `O(n^4)` reference: enumerate every substring length from
    /// long to short and every starting offset; for each, scan the rest of
    /// the string for a second occurrence. Returns the lex-smallest substring
    /// among the longest repeats, matching the tie-break of the fast path.
    fn brute_force(s: &str) -> String {
        let bytes = s.as_bytes();
        let n = bytes.len();
        for len in (1..n).rev() {
            let mut best: Option<&[u8]> = None;
            for i in 0..=n - len {
                let candidate = &bytes[i..i + len];
                // Look for a second occurrence at any later starting index.
                let mut repeats = false;
                for j in (i + 1)..=n - len {
                    if &bytes[j..j + len] == candidate {
                        repeats = true;
                        break;
                    }
                }
                if repeats {
                    match best {
                        None => best = Some(candidate),
                        Some(prev) if candidate < prev => best = Some(candidate),
                        _ => {}
                    }
                }
            }
            if let Some(b) = best {
                return String::from_utf8_lossy(b).into_owned();
            }
        }
        String::new()
    }

    #[test]
    fn empty_input() {
        assert_eq!(longest_repeated_substring(""), "");
    }

    #[test]
    fn single_char() {
        assert_eq!(longest_repeated_substring("a"), "");
        assert_eq!(longest_repeated_substring("z"), "");
    }

    #[test]
    fn no_repeat() {
        assert_eq!(longest_repeated_substring("abcdefg"), "");
    }

    #[test]
    fn banana() {
        // "ana" appears at indices 1 and 3; nothing of length 4 repeats.
        assert_eq!(longest_repeated_substring("banana"), "ana");
    }

    #[test]
    fn ababab() {
        // "abab" appears at indices 0 and 2.
        assert_eq!(longest_repeated_substring("ababab"), "abab");
    }

    #[test]
    fn aabaabaa() {
        // "aabaa" appears at indices 0 and 3 — length 5 is the max.
        assert_eq!(longest_repeated_substring("aabaabaa"), "aabaa");
    }

    #[test]
    fn all_equal_chars() {
        // For "aaaa", suffixes "aaa" (start 1) and "aaa" (start 0) — the
        // longest repeat is "aaa".
        assert_eq!(longest_repeated_substring("aaaa"), "aaa");
    }

    #[test]
    fn two_equal_chars() {
        // "aa" — single-character repeat "a".
        assert_eq!(longest_repeated_substring("aa"), "a");
    }

    #[test]
    fn mississippi() {
        // Known: longest repeat is "issi" (length 4), at indices 1 and 4.
        assert_eq!(longest_repeated_substring("mississippi"), "issi");
    }

    #[test]
    fn tie_break_picks_lex_smallest() {
        // "abcabcdefdef": "abc" repeats at indices 0,3 and "def" repeats at
        // indices 6,9 — both length 3, no length-4 substring repeats. The
        // function must return the lex-smaller of the two, "abc".
        assert_eq!(longest_repeated_substring("abcabcdefdef"), "abc");
    }

    #[test]
    fn matches_brute_force_known_strings() {
        for s in [
            "",
            "a",
            "ab",
            "aa",
            "abc",
            "banana",
            "ababab",
            "aabaabaa",
            "mississippi",
            "abracadabra",
            "the quick brown fox",
            "abcabcdefdef",
        ] {
            assert_eq!(longest_repeated_substring(s), brute_force(s), "input {s:?}");
        }
    }

    #[allow(clippy::needless_pass_by_value)]
    #[quickcheck]
    fn quickcheck_matches_brute_force(bytes: Vec<u8>) -> bool {
        // Restrict to short ASCII letters so brute force stays cheap and the
        // input is always valid UTF-8 with character boundaries on every
        // byte (so `String::from_utf8_lossy` is a no-op).
        let s: String = bytes
            .into_iter()
            .take(20)
            .map(|b| (b'a' + (b % 4)) as char)
            .collect();
        longest_repeated_substring(&s) == brute_force(&s)
    }
}
