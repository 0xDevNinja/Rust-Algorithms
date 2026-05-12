//! Word break: decide whether a string can be segmented into a sequence of
//! dictionary words, and optionally enumerate all such segmentations.
//!
//! Recurrence: `can[i]` is true iff there exists `j < i` with `can[j]` true
//! and the substring `s[j..i]` appears in the dictionary, with `can[0] = true`.
//! Filling the table requires O(n^2) substring lookups, each O(L) on the
//! substring length, giving an overall O(n^3) worst case (O(n^2) hash probes
//! ignoring hashing cost on the substring).
//!
//! Enumeration of all segmentations is exponential in the worst case, since
//! a single string may admit exponentially many decompositions.

use std::collections::HashSet;
use std::hash::BuildHasher;

/// Returns true iff `s` can be segmented into a space-separated sequence of
/// one or more dictionary words.
///
/// An empty string with an empty (or any) dictionary returns `true` — the
/// empty segmentation is always valid.
pub fn word_break<S: BuildHasher>(s: &str, dict: &HashSet<String, S>) -> bool {
    let n = s.len();
    let mut can = vec![false; n + 1];
    can[0] = true;
    for i in 1..=n {
        for j in 0..i {
            if can[j] && dict.contains(&s[j..i]) {
                can[i] = true;
                break;
            }
        }
    }
    can[n]
}

/// Returns every segmentation of `s` into dictionary words, each rendered as a
/// single space-separated string.
///
/// The result order is not specified. For an empty input the function returns
/// a single empty segmentation (`vec![String::new()]`) when the dictionary is
/// also empty or arbitrary — the empty string is trivially segmentable.
pub fn word_break_all<S: BuildHasher>(s: &str, dict: &HashSet<String, S>) -> Vec<String> {
    let n = s.len();
    // memo[i] holds every way to segment s[i..].
    let mut memo: Vec<Option<Vec<String>>> = vec![None; n + 1];
    collect(s, 0, dict, &mut memo);
    memo[0].take().unwrap_or_default()
}

fn collect<S: BuildHasher>(
    s: &str,
    start: usize,
    dict: &HashSet<String, S>,
    memo: &mut Vec<Option<Vec<String>>>,
) -> Vec<String> {
    if let Some(cached) = &memo[start] {
        return cached.clone();
    }
    let mut out: Vec<String> = Vec::new();
    if start == s.len() {
        out.push(String::new());
    } else {
        for end in (start + 1)..=s.len() {
            let word = &s[start..end];
            if dict.contains(word) {
                let suffixes = collect(s, end, dict, memo);
                for suffix in suffixes {
                    if suffix.is_empty() {
                        out.push(word.to_string());
                    } else {
                        out.push(format!("{word} {suffix}"));
                    }
                }
            }
        }
    }
    memo[start] = Some(out.clone());
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dict<I, S>(words: I) -> HashSet<String>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        words.into_iter().map(Into::into).collect()
    }

    #[test]
    fn empty_string_empty_dict() {
        let d: HashSet<String> = HashSet::new();
        assert!(word_break("", &d));
    }

    #[test]
    fn leetcode_example() {
        let d = dict(["leet", "code"]);
        assert!(word_break("leetcode", &d));
    }

    #[test]
    fn applepenapple() {
        let d = dict(["apple", "pen"]);
        assert!(word_break("applepenapple", &d));
    }

    #[test]
    fn catsandog_no_segmentation() {
        let d = dict(["cats", "sand", "dog", "cat", "and"]);
        assert!(!word_break("catsandog", &d));
    }

    #[test]
    fn single_char_match() {
        let d = dict(["a"]);
        assert!(word_break("a", &d));
    }

    #[test]
    fn single_char_mismatch() {
        let d = dict(["b"]);
        assert!(!word_break("a", &d));
    }

    #[test]
    fn all_segmentations_pineapplepen() {
        let d = dict(["apple", "pen", "applepen", "pine", "pineapple"]);
        let mut out = word_break_all("pineapplepenapple", &d);
        out.sort();
        let mut expected = vec![
            "pine apple pen apple".to_string(),
            "pine applepen apple".to_string(),
            "pineapple pen apple".to_string(),
        ];
        expected.sort();
        assert_eq!(out, expected);
    }

    #[test]
    fn all_segmentations_none() {
        let d = dict(["cats", "sand", "dog", "cat", "and"]);
        let out = word_break_all("catsandog", &d);
        assert!(out.is_empty());
    }

    #[test]
    fn all_segmentations_empty_string() {
        let d: HashSet<String> = HashSet::new();
        let out = word_break_all("", &d);
        assert_eq!(out, vec![String::new()]);
    }
}
