//! LCP array construction via Kasai's algorithm.
//!
//! Given a string `s` and its suffix array `SA`, the longest-common-prefix
//! (LCP) array stores, for each adjacent pair of suffixes in `SA` order, the
//! length of their longest common prefix. Concretely `lcp[i]` is the LCP of
//! `s[SA[i-1]..]` and `s[SA[i]..]` for `i >= 1`. By convention `lcp[0] = 0`,
//! since there is no preceding suffix to compare against.
//!
//! Kasai's algorithm computes the LCP array in `O(n)` time using the
//! **rank array** trick. The key observation is: walking the original string
//! in index order (rather than SA order) and reusing the previous LCP value
//! `h`, the LCP of `s[i..]` with the suffix preceding it in `SA` is at least
//! `h - 1`. So `h` decreases by at most one per step and increases by
//! arbitrary amounts, giving `O(n)` total work amortised.
//!
//! # Complexity
//! - Time:  O(n) — each character of `s` is visited a constant number of
//!   times across the entire run.
//! - Space: O(n) for the rank array and the returned LCP vector.
//!
//! # Preconditions
//! `sa` MUST be the suffix array of `s` — a permutation of `0..s.len()`
//! ordering the suffixes lexicographically. Passing a non-permutation or a
//! mismatched array yields nonsense (the function does not validate).
//!
//! # Use cases
//! - Longest repeated substring: `max(lcp)` plus the corresponding SA entry.
//! - Number of distinct substrings: `n*(n+1)/2 - sum(lcp)`.
//! - Longest common substring of multiple strings (with a separator trick).
//! - Building blocks for full-text indices and bioinformatics pipelines.
//!
//! Operates on `&[u8]`. Callers with a `&str` can pass `s.as_bytes()`.

/// Returns the LCP array of `s` given its suffix array `sa`, computed by
/// Kasai's algorithm in `O(n)` time.
///
/// `lcp[i]` is the length of the longest common prefix of `s[sa[i-1]..]` and
/// `s[sa[i]..]` for `i >= 1`. By convention `lcp[0] = 0`. The returned vector
/// has the same length as `s`. For empty input the result is empty.
///
/// `sa` MUST be the suffix array of `s` (a permutation of `0..s.len()` sorting
/// the suffixes lexicographically); the function does not validate this.
pub fn lcp_kasai(s: &[u8], sa: &[usize]) -> Vec<usize> {
    let n = s.len();
    if n == 0 {
        return Vec::new();
    }

    // `rank[i]` is the position of suffix `s[i..]` in the suffix array, i.e.
    // the inverse permutation of `sa`. This lets us, given a starting index
    // `i` in the original string, find the suffix that precedes `s[i..]` in
    // SA order in O(1).
    let mut rank: Vec<usize> = vec![0; n];
    for (i, &sa_i) in sa.iter().enumerate() {
        rank[sa_i] = i;
    }

    let mut lcp: Vec<usize> = vec![0; n];
    let mut h: usize = 0;

    // Walk the original string in index order. The amortisation argument:
    // `h` drops by at most one per outer iteration but can grow by many,
    // so the total number of byte comparisons is bounded by `2n`.
    for i in 0..n {
        let r = rank[i];
        if r == 0 {
            // Smallest suffix in SA order: no predecessor to compare with.
            h = 0;
            continue;
        }
        let j = sa[r - 1];
        while i + h < n && j + h < n && s[i + h] == s[j + h] {
            h += 1;
        }
        lcp[r] = h;
        // The suffix starting at `i + 1` shares all but the first character
        // with the current one, so its LCP with its SA-predecessor is at
        // least `h - 1` — never start the next match from scratch.
        h = h.saturating_sub(1);
    }

    lcp
}

#[cfg(test)]
mod tests {
    use super::lcp_kasai;
    use crate::string::suffix_array::suffix_array;
    use quickcheck_macros::quickcheck;

    /// Brute-force reference: byte-by-byte LCP of every adjacent pair in
    /// SA order, with `lcp[0] = 0` by convention.
    fn brute_force(s: &[u8], sa: &[usize]) -> Vec<usize> {
        let n = s.len();
        let mut lcp = vec![0_usize; n];
        for i in 1..n {
            let a = &s[sa[i - 1]..];
            let b = &s[sa[i]..];
            let mut k = 0;
            while k < a.len() && k < b.len() && a[k] == b[k] {
                k += 1;
            }
            lcp[i] = k;
        }
        lcp
    }

    #[test]
    fn empty_input() {
        assert_eq!(lcp_kasai(b"", &[]), Vec::<usize>::new());
    }

    #[test]
    fn single_char() {
        // Only one suffix: lcp[0] = 0 by convention.
        assert_eq!(lcp_kasai(b"a", &[0]), vec![0]);
        assert_eq!(lcp_kasai(b"z", &[0]), vec![0]);
    }

    #[test]
    fn two_chars_distinct() {
        // "ab" -> SA = [0, 1]; suffixes "ab" and "b" share no prefix.
        assert_eq!(lcp_kasai(b"ab", &[0, 1]), vec![0, 0]);
    }

    #[test]
    fn two_chars_equal() {
        // "aa" -> SA = [1, 0]; "a" vs "aa" share "a" -> 1.
        assert_eq!(lcp_kasai(b"aa", &[1, 0]), vec![0, 1]);
    }

    #[test]
    fn banana_classic() {
        // Textbook example: SA = [5, 3, 1, 0, 4, 2] for "banana".
        // Suffixes in SA order:
        //   a, ana, anana, banana, na, nana
        // Adjacent LCPs: -, 1, 3, 0, 0, 2  -> stored with leading 0.
        let s = b"banana";
        let sa = suffix_array(s);
        assert_eq!(sa, vec![5, 3, 1, 0, 4, 2]);
        assert_eq!(lcp_kasai(s, &sa), vec![0, 1, 3, 0, 0, 2]);
    }

    #[test]
    fn all_equal_chars() {
        // "aaaa" -> SA = [3, 2, 1, 0]; suffixes "a","aa","aaa","aaaa".
        // Adjacent LCPs: -, 1, 2, 3.
        let s = b"aaaa";
        let sa = suffix_array(s);
        assert_eq!(lcp_kasai(s, &sa), vec![0, 1, 2, 3]);
    }

    #[test]
    fn mississippi() {
        let s = b"mississippi";
        let sa = suffix_array(s);
        assert_eq!(lcp_kasai(s, &sa), brute_force(s, &sa));
    }

    #[test]
    fn abracadabra() {
        let s = b"abracadabra";
        let sa = suffix_array(s);
        assert_eq!(lcp_kasai(s, &sa), brute_force(s, &sa));
    }

    #[test]
    fn matches_brute_force_known_strings() {
        for s in [
            b"banana".as_slice(),
            b"abracadabra".as_slice(),
            b"mississippi".as_slice(),
            b"aabaaabaaa".as_slice(),
            b"the quick brown fox".as_slice(),
            b"\x00\x01\x00\x02\x00".as_slice(),
        ] {
            let sa = suffix_array(s);
            assert_eq!(lcp_kasai(s, &sa), brute_force(s, &sa), "input {s:?}");
        }
    }

    #[test]
    fn lcp_zero_at_index_zero() {
        // The convention lcp[0] = 0 holds for every non-empty input.
        for s in [
            b"a".as_slice(),
            b"ab".as_slice(),
            b"banana".as_slice(),
            b"aaaa".as_slice(),
        ] {
            let sa = suffix_array(s);
            let lcp = lcp_kasai(s, &sa);
            assert_eq!(lcp[0], 0, "input {s:?}");
        }
    }

    #[test]
    fn distinct_substring_count_banana() {
        // Number of distinct non-empty substrings of `s` equals
        // n*(n+1)/2 - sum(lcp). For "banana" that is 21 - 6 = 15.
        let s = b"banana";
        let sa = suffix_array(s);
        let lcp = lcp_kasai(s, &sa);
        let n = s.len();
        let total = n * (n + 1) / 2;
        let shared: usize = lcp.iter().sum();
        assert_eq!(total - shared, 15);
    }

    #[allow(clippy::needless_pass_by_value)]
    #[quickcheck]
    fn quickcheck_matches_brute_force(bytes: Vec<u8>) -> bool {
        let bytes: Vec<u8> = bytes.into_iter().take(30).collect();
        let sa = suffix_array(&bytes);
        lcp_kasai(&bytes, &sa) == brute_force(&bytes, &sa)
    }

    #[allow(clippy::needless_pass_by_value)]
    #[quickcheck]
    fn quickcheck_lcp_length_matches_input(bytes: Vec<u8>) -> bool {
        let bytes: Vec<u8> = bytes.into_iter().take(30).collect();
        let sa = suffix_array(&bytes);
        lcp_kasai(&bytes, &sa).len() == bytes.len()
    }

    #[allow(clippy::needless_pass_by_value)]
    #[quickcheck]
    fn quickcheck_first_entry_is_zero(bytes: Vec<u8>) -> bool {
        let bytes: Vec<u8> = bytes.into_iter().take(30).collect();
        if bytes.is_empty() {
            return true;
        }
        let sa = suffix_array(&bytes);
        lcp_kasai(&bytes, &sa)[0] == 0
    }
}
