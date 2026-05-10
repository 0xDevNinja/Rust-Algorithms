//! First non-repeating character.
//!
//! Given a string slice, return the first Unicode character (`char`) that
//! occurs exactly once when scanning the string left-to-right, or `None` if
//! every character repeats (or the string is empty).
//!
//! # Algorithm
//!
//! Two linear passes over the input:
//!
//! 1. Walk the string and record the number of occurrences of each `char` in
//!    a `HashMap<char, usize>`.
//! 2. Walk the string again and return the first `char` whose recorded count
//!    is `1`.
//!
//! # Complexity
//!
//! Let `n` be the number of `char`s in the input.
//!
//! - Time:  `O(n)` average (hash-map operations are amortised `O(1)`).
//! - Space: `O(k)` where `k` is the number of distinct characters
//!   (`k <= n`).
//!
//! Operating on `char`s (Unicode scalar values) means multi-byte UTF-8
//! sequences are handled correctly — we never split a code point.
//!
//! # Examples
//!
//! ```
//! use rust_algorithms::string::first_non_repeating::first_non_repeating;
//!
//! assert_eq!(first_non_repeating("leetcode"),     Some('l'));
//! assert_eq!(first_non_repeating("loveleetcode"), Some('v'));
//! assert_eq!(first_non_repeating("aabb"),         None);
//! assert_eq!(first_non_repeating(""),             None);
//! ```

use std::collections::HashMap;

/// Returns the first non-repeating Unicode `char` in `s`, or `None` if no
/// such character exists.
///
/// The scan is performed in `char` order (left-to-right), so for inputs
/// containing multi-byte UTF-8 sequences each Unicode scalar value is
/// considered as a single unit.
///
/// # Examples
///
/// ```
/// use rust_algorithms::string::first_non_repeating::first_non_repeating;
///
/// assert_eq!(first_non_repeating("swiss"), Some('w'));
/// assert_eq!(first_non_repeating("aabb"),  None);
/// ```
pub fn first_non_repeating(s: &str) -> Option<char> {
    let mut counts: HashMap<char, usize> = HashMap::new();
    for c in s.chars() {
        *counts.entry(c).or_insert(0) += 1;
    }
    s.chars().find(|c| counts.get(c) == Some(&1))
}

#[cfg(test)]
mod tests {
    use super::first_non_repeating;

    #[test]
    fn empty_string_returns_none() {
        assert_eq!(first_non_repeating(""), None);
    }

    #[test]
    fn leetcode_returns_l() {
        assert_eq!(first_non_repeating("leetcode"), Some('l'));
    }

    #[test]
    fn loveleetcode_returns_v() {
        assert_eq!(first_non_repeating("loveleetcode"), Some('v'));
    }

    #[test]
    fn all_pairs_returns_none() {
        assert_eq!(first_non_repeating("aabb"), None);
    }

    #[test]
    fn single_char_returns_that_char() {
        assert_eq!(first_non_repeating("z"), Some('z'));
    }

    #[test]
    fn all_same_chars_returns_none() {
        assert_eq!(first_non_repeating("aaaa"), None);
        assert_eq!(first_non_repeating("aa"), None);
    }

    #[test]
    fn picks_first_unique_when_multiple() {
        // 'w' is the first non-repeating char in "swiss" (s,s,s repeat;
        // i is also unique but appears later than w).
        assert_eq!(first_non_repeating("swiss"), Some('w'));
    }

    #[test]
    fn handles_unicode_chars() {
        // 'é' repeats; 'ñ' is the first unique scalar value.
        assert_eq!(first_non_repeating("éñé"), Some('ñ'));
        // Emoji are a single `char` only when they are a single scalar
        // value; pick a simple one to stay correct.
        assert_eq!(first_non_repeating("☃☃★"), Some('★'));
    }

    #[test]
    fn whitespace_and_punctuation_are_chars() {
        assert_eq!(first_non_repeating("  a  "), Some('a'));
        assert_eq!(first_non_repeating("!!?"), Some('?'));
    }

    #[test]
    fn case_sensitive() {
        // 'A' and 'a' are distinct chars.
        assert_eq!(first_non_repeating("aAbBcC"), Some('a'));
    }
}
