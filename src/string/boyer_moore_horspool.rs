//! Boyer–Moore–Horspool substring search.
//!
//! A simplification of the full Boyer–Moore algorithm that uses only the
//! bad-character shift table. Preprocessing is O(m + σ) where σ is the
//! alphabet size (256 for bytes); matching is O(n) on average and O(n · m)
//! in the worst case.
//!
//! BMH is fundamentally byte-oriented: the bad-character table is indexed by
//! a single byte value, so this module operates on `&[u8]` rather than
//! `&str`. Callers that need to search Unicode text can pass
//! `s.as_bytes()` — matches will be at byte offsets within the haystack.

const ALPHABET: usize = 256;

/// Returns the byte offset of the first occurrence of `needle` in
/// `haystack`, or `None` if `needle` does not occur.
///
/// An empty `needle` matches at index `0` (consistent with `str::find`).
#[must_use]
pub fn bmh_search(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() {
        return Some(0);
    }
    let n = haystack.len();
    let m = needle.len();
    if m > n {
        return None;
    }

    let shift = bad_char_shift(needle);
    let last = m - 1;
    let mut i = 0;
    while i <= n - m {
        let mut j = last;
        while haystack[i + j] == needle[j] {
            if j == 0 {
                return Some(i);
            }
            j -= 1;
        }
        i += shift[haystack[i + last] as usize];
    }
    None
}

/// Returns the byte offsets of every non-overlapping occurrence of `needle`
/// in `haystack`, in increasing order.
///
/// After a match at index `i`, the search resumes at `i + needle.len()`.
/// An empty `needle` returns an empty vector to avoid an infinite stream of
/// zero-width matches.
#[must_use]
pub fn bmh_search_all(haystack: &[u8], needle: &[u8]) -> Vec<usize> {
    let mut matches = Vec::new();
    if needle.is_empty() {
        return matches;
    }
    let n = haystack.len();
    let m = needle.len();
    if m > n {
        return matches;
    }

    let shift = bad_char_shift(needle);
    let last = m - 1;
    let mut i = 0;
    while i <= n - m {
        let mut j = last;
        let mut matched = true;
        loop {
            if haystack[i + j] != needle[j] {
                matched = false;
                break;
            }
            if j == 0 {
                break;
            }
            j -= 1;
        }
        if matched {
            matches.push(i);
            i += m;
        } else {
            i += shift[haystack[i + last] as usize];
        }
    }
    matches
}

/// Builds the bad-character shift table for `needle`.
///
/// `shift[c] = needle.len() - 1 - last_index_of(c)` for each byte `c` that
/// appears in `needle[..needle.len() - 1]`; all other entries default to
/// `needle.len()`. The last byte of the needle is intentionally excluded so
/// that a mismatch on the rightmost position still produces a positive
/// shift.
fn bad_char_shift(needle: &[u8]) -> [usize; ALPHABET] {
    let m = needle.len();
    let mut shift = [m; ALPHABET];
    for (i, &b) in needle.iter().enumerate().take(m - 1) {
        shift[b as usize] = m - 1 - i;
    }
    shift
}

#[cfg(test)]
mod tests {
    use super::{bmh_search, bmh_search_all};

    #[test]
    fn empty_needle_matches_at_zero() {
        assert_eq!(bmh_search(b"abc", b""), Some(0));
        assert_eq!(bmh_search(b"", b""), Some(0));
    }

    #[test]
    fn empty_needle_search_all_returns_empty() {
        assert_eq!(bmh_search_all(b"abc", b""), Vec::<usize>::new());
    }

    #[test]
    fn empty_haystack_nonempty_needle() {
        assert_eq!(bmh_search(b"", b"a"), None);
        assert_eq!(bmh_search_all(b"", b"a"), Vec::<usize>::new());
    }

    #[test]
    fn no_match() {
        assert_eq!(bmh_search(b"abcdef", b"xyz"), None);
        assert_eq!(bmh_search_all(b"abcdef", b"xyz"), Vec::<usize>::new());
    }

    #[test]
    fn match_at_start() {
        assert_eq!(bmh_search(b"hello world", b"hello"), Some(0));
        assert_eq!(bmh_search_all(b"hello world", b"hello"), vec![0]);
    }

    #[test]
    fn match_in_middle() {
        assert_eq!(bmh_search(b"hello world", b"lo wo"), Some(3));
        assert_eq!(bmh_search_all(b"hello world", b"lo wo"), vec![3]);
    }

    #[test]
    fn match_at_end() {
        assert_eq!(bmh_search(b"hello world", b"world"), Some(6));
        assert_eq!(bmh_search_all(b"hello world", b"world"), vec![6]);
    }

    #[test]
    fn multiple_non_overlapping_matches() {
        // "ab" appears at 0, 2, 4.
        assert_eq!(bmh_search_all(b"ababab", b"ab"), vec![0, 2, 4]);
    }

    #[test]
    fn overlapping_pattern_is_skipped() {
        // "aa" in "aaaa": after a match at 0, search resumes at 2 → [0, 2].
        assert_eq!(bmh_search_all(b"aaaa", b"aa"), vec![0, 2]);
    }

    #[test]
    fn first_match_returned_only() {
        assert_eq!(bmh_search(b"ababab", b"ab"), Some(0));
    }

    #[test]
    fn needle_longer_than_haystack() {
        assert_eq!(bmh_search(b"ab", b"abc"), None);
        assert_eq!(bmh_search_all(b"ab", b"abc"), Vec::<usize>::new());
    }

    #[test]
    fn needle_equals_haystack() {
        assert_eq!(bmh_search(b"abc", b"abc"), Some(0));
        assert_eq!(bmh_search_all(b"abc", b"abc"), vec![0]);
    }

    #[test]
    fn classic_example() {
        let text = b"ABABDABACDABABCABAB";
        let pat = b"ABABCABAB";
        assert_eq!(bmh_search(text, pat), Some(10));
        assert_eq!(bmh_search_all(text, pat), vec![10]);
    }

    #[test]
    fn non_ascii_bytes() {
        // High-bit bytes including a multi-byte UTF-8 sequence for "é" (0xC3 0xA9).
        let haystack: &[u8] = &[0x00, 0xC3, 0xA9, 0xFF, 0xC3, 0xA9, 0x10];
        let needle: &[u8] = &[0xC3, 0xA9];
        assert_eq!(bmh_search(haystack, needle), Some(1));
        assert_eq!(bmh_search_all(haystack, needle), vec![1, 4]);
    }

    #[test]
    fn unicode_via_as_bytes() {
        // "é" is two bytes in UTF-8 (0xC3 0xA9), so "café" is 5 bytes long
        // and the second occurrence sits at byte offset 15.
        let text = "café au lait, café noir";
        assert_eq!(
            bmh_search_all(text.as_bytes(), "café".as_bytes()),
            vec![0, 15]
        );
    }

    /// Trivial reference search used to cross-check the BMH implementation.
    fn naive_search_all(haystack: &[u8], needle: &[u8]) -> Vec<usize> {
        let mut out = Vec::new();
        if needle.is_empty() || needle.len() > haystack.len() {
            return out;
        }
        let mut i = 0;
        while i + needle.len() <= haystack.len() {
            if &haystack[i..i + needle.len()] == needle {
                out.push(i);
                i += needle.len();
            } else {
                i += 1;
            }
        }
        out
    }

    fn naive_search_first(haystack: &[u8], needle: &[u8]) -> Option<usize> {
        if needle.is_empty() {
            return Some(0);
        }
        if needle.len() > haystack.len() {
            return None;
        }
        (0..=haystack.len() - needle.len()).find(|&i| &haystack[i..i + needle.len()] == needle)
    }

    #[quickcheck_macros::quickcheck]
    #[allow(clippy::needless_pass_by_value)]
    fn matches_naive_first(haystack: Vec<u8>, needle: Vec<u8>) -> bool {
        // Cap lengths to keep the test fast.
        let haystack: &[u8] = &haystack[..haystack.len().min(50)];
        let needle: &[u8] = &needle[..needle.len().min(50)];
        bmh_search(haystack, needle) == naive_search_first(haystack, needle)
    }

    #[quickcheck_macros::quickcheck]
    #[allow(clippy::needless_pass_by_value)]
    fn matches_naive_all(haystack: Vec<u8>, needle: Vec<u8>) -> bool {
        let haystack: &[u8] = &haystack[..haystack.len().min(50)];
        let needle: &[u8] = &needle[..needle.len().min(50)];
        bmh_search_all(haystack, needle) == naive_search_all(haystack, needle)
    }
}
