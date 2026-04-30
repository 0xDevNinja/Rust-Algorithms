//! Boyer–Moore substring search (full algorithm).
//!
//! The classic Boyer–Moore algorithm combines two independent shift
//! heuristics computed during preprocessing:
//!
//! * **Bad-character rule** — on a mismatch at position `j` of the needle,
//!   shift so that the offending haystack byte aligns with its rightmost
//!   occurrence in the needle (or past it if the byte does not occur).
//! * **Good-suffix rule** — when a non-empty suffix of the needle has
//!   already matched, shift so that either an earlier occurrence of that
//!   suffix lines up, or — failing that — the longest needle prefix that is
//!   also a suffix of the matched portion lines up.
//!
//! On every mismatch the algorithm takes the maximum of the two suggested
//! shifts. Preprocessing is O(m + σ) (σ = 256 for bytes) and matching runs
//! in O(n + m) on average and O(n · m) in the worst case, but typically
//! sublinear on natural-language text.
//!
//! This is the **full** Boyer–Moore. The crate also ships
//! [`boyer_moore_horspool`](super::boyer_moore_horspool) which is a
//! simplification using only the bad-character table; the good-suffix
//! tables here are what give Boyer–Moore its better worst-case behaviour
//! on inputs that defeat BMH.
//!
//! Like BMH the algorithm is byte-oriented (the bad-character table is
//! indexed by a single byte), so this module operates on `&[u8]`. Pass
//! `s.as_bytes()` to search Unicode text — matches will be byte offsets.

const ALPHABET: usize = 256;

/// Returns the byte offset of the first occurrence of `needle` in
/// `haystack`, or `None` if `needle` does not occur.
///
/// An empty `needle` matches at index `0` (consistent with `str::find`).
///
/// Runs in O(n + m) time on average and O(n · m) in the worst case, with
/// O(m + σ) preprocessing space.
#[must_use]
pub fn bm_search(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() {
        return Some(0);
    }
    let n = haystack.len();
    let m = needle.len();
    if m > n {
        return None;
    }

    let bad_char = bad_char_table(needle);
    let good_suffix = good_suffix_table(needle);

    let mut s: usize = 0;
    while s <= n - m {
        let mut j = (m - 1) as isize;
        while j >= 0 && needle[j as usize] == haystack[s + j as usize] {
            j -= 1;
        }
        if j < 0 {
            return Some(s);
        }
        let bc_shift = bad_char_shift(&bad_char, haystack[s + j as usize], j as usize);
        let gs_shift = good_suffix[j as usize];
        s += bc_shift.max(gs_shift);
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
pub fn bm_search_all(haystack: &[u8], needle: &[u8]) -> Vec<usize> {
    let mut matches = Vec::new();
    if needle.is_empty() {
        return matches;
    }
    let n = haystack.len();
    let m = needle.len();
    if m > n {
        return matches;
    }

    let bad_char = bad_char_table(needle);
    let good_suffix = good_suffix_table(needle);

    let mut s: usize = 0;
    while s <= n - m {
        let mut j = (m - 1) as isize;
        while j >= 0 && needle[j as usize] == haystack[s + j as usize] {
            j -= 1;
        }
        if j < 0 {
            matches.push(s);
            s += m;
        } else {
            let bc_shift = bad_char_shift(&bad_char, haystack[s + j as usize], j as usize);
            let gs_shift = good_suffix[j as usize];
            s += bc_shift.max(gs_shift);
        }
    }
    matches
}

/// Builds the bad-character table: `table[c]` is the index of the
/// rightmost occurrence of byte `c` in `needle`, or `-1` if absent.
fn bad_char_table(needle: &[u8]) -> [isize; ALPHABET] {
    let mut table = [-1isize; ALPHABET];
    for (i, &b) in needle.iter().enumerate() {
        table[b as usize] = i as isize;
    }
    table
}

/// Bad-character shift on a mismatch at position `j` against haystack byte
/// `c`. Always returns at least 1 to guarantee forward progress.
const fn bad_char_shift(table: &[isize; ALPHABET], c: u8, j: usize) -> usize {
    let last = table[c as usize];
    let shift = j as isize - last;
    if shift < 1 {
        1
    } else {
        shift as usize
    }
}

/// Builds the good-suffix shift table.
///
/// `shift[j]` is the amount to advance the alignment when a mismatch
/// occurs at needle position `j` (after `needle[j+1..]` has matched). The
/// table is built in two passes following the classic Knuth/Morris/Pratt
/// adaptation described in Crochemore & Rytter:
///
/// 1. Compute the `suff` array, where `suff[i]` is the length of the
///    longest suffix of `needle[..=i]` that is also a suffix of `needle`.
/// 2. First pass — case 2: borders of `needle` cover the matched suffix.
/// 3. Second pass — case 1: an earlier occurrence of the matched suffix
///    sits inside the needle.
fn good_suffix_table(needle: &[u8]) -> Vec<usize> {
    let m = needle.len();
    let mut shift = vec![0usize; m];
    let suff = suffixes(needle);

    // Case 2: every position defaults to the shift derived from the
    // longest border of the needle that is a suffix of the needle.
    let mut j = 0usize;
    for i in (0..m).rev() {
        if suff[i] == i + 1 {
            while j < m - 1 - i {
                if shift[j] == 0 {
                    shift[j] = m - 1 - i;
                }
                j += 1;
            }
        }
    }

    // Case 1: an earlier internal occurrence of the matched suffix.
    for i in 0..m - 1 {
        shift[m - 1 - suff[i]] = m - 1 - i;
    }

    shift
}

/// Computes the `suff` array: `suff[i]` is the length of the longest
/// substring of `needle` ending at index `i` that is also a suffix of
/// `needle`.
///
/// Uses the straightforward O(m^2) definition; the implementation cost is
/// dominated by matching, not preprocessing, so the simpler quadratic
/// scan is preferred over Crochemore's clever linear variant.
fn suffixes(needle: &[u8]) -> Vec<usize> {
    let m = needle.len();
    let mut suff = vec![0usize; m];
    suff[m - 1] = m;
    for i in (0..m - 1).rev() {
        let mut k = 0usize;
        while k <= i && needle[i - k] == needle[m - 1 - k] {
            k += 1;
        }
        suff[i] = k;
    }
    suff
}

#[cfg(test)]
mod tests {
    use super::{bm_search, bm_search_all};

    #[test]
    fn empty_needle_matches_at_zero() {
        assert_eq!(bm_search(b"abc", b""), Some(0));
        assert_eq!(bm_search(b"", b""), Some(0));
    }

    #[test]
    fn empty_needle_search_all_returns_empty() {
        assert_eq!(bm_search_all(b"abc", b""), Vec::<usize>::new());
    }

    #[test]
    fn empty_haystack_nonempty_needle() {
        assert_eq!(bm_search(b"", b"a"), None);
        assert_eq!(bm_search_all(b"", b"a"), Vec::<usize>::new());
    }

    #[test]
    fn no_match() {
        assert_eq!(bm_search(b"abcdef", b"xyz"), None);
        assert_eq!(bm_search_all(b"abcdef", b"xyz"), Vec::<usize>::new());
    }

    #[test]
    fn match_at_start() {
        assert_eq!(bm_search(b"hello world", b"hello"), Some(0));
        assert_eq!(bm_search_all(b"hello world", b"hello"), vec![0]);
    }

    #[test]
    fn match_in_middle() {
        assert_eq!(bm_search(b"hello world", b"lo wo"), Some(3));
        assert_eq!(bm_search_all(b"hello world", b"lo wo"), vec![3]);
    }

    #[test]
    fn match_at_end() {
        assert_eq!(bm_search(b"hello world", b"world"), Some(6));
        assert_eq!(bm_search_all(b"hello world", b"world"), vec![6]);
    }

    #[test]
    fn multiple_non_overlapping_matches() {
        assert_eq!(bm_search_all(b"ababab", b"ab"), vec![0, 2, 4]);
    }

    #[test]
    fn overlapping_pattern_is_skipped() {
        // "aa" in "aaaa": after a match at 0, search resumes at 2 → [0, 2].
        assert_eq!(bm_search_all(b"aaaa", b"aa"), vec![0, 2]);
    }

    #[test]
    fn first_match_returned_only() {
        assert_eq!(bm_search(b"ababab", b"ab"), Some(0));
    }

    #[test]
    fn needle_longer_than_haystack() {
        assert_eq!(bm_search(b"ab", b"abc"), None);
        assert_eq!(bm_search_all(b"ab", b"abc"), Vec::<usize>::new());
    }

    #[test]
    fn needle_equals_haystack() {
        assert_eq!(bm_search(b"abc", b"abc"), Some(0));
        assert_eq!(bm_search_all(b"abc", b"abc"), vec![0]);
    }

    #[test]
    fn classic_example() {
        // The textbook Boyer–Moore example.
        let text = b"ABABDABACDABABCABAB";
        let pat = b"ABABCABAB";
        assert_eq!(bm_search(text, pat), Some(10));
        assert_eq!(bm_search_all(text, pat), vec![10]);
    }

    #[test]
    fn good_suffix_dominates_bad_char() {
        // Pattern designed so the good-suffix shift is the larger one:
        // searching "GCAGAGAG" — the rightmost mismatch can be skipped
        // further by the good-suffix rule than by the bad-char rule.
        let text = b"GCATCGCAGAGAGTATACAGTACG";
        let pat = b"GCAGAGAG";
        assert_eq!(bm_search(text, pat), Some(5));
        assert_eq!(bm_search_all(text, pat), vec![5]);
    }

    #[test]
    fn non_ascii_bytes() {
        let haystack: &[u8] = &[0x00, 0xC3, 0xA9, 0xFF, 0xC3, 0xA9, 0x10];
        let needle: &[u8] = &[0xC3, 0xA9];
        assert_eq!(bm_search(haystack, needle), Some(1));
        assert_eq!(bm_search_all(haystack, needle), vec![1, 4]);
    }

    #[test]
    fn unicode_via_as_bytes() {
        let text = "café au lait, café noir";
        assert_eq!(
            bm_search_all(text.as_bytes(), "café".as_bytes()),
            vec![0, 15]
        );
    }

    /// Trivial reference search used to cross-check the BM implementation.
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
        let haystack: &[u8] = &haystack[..haystack.len().min(50)];
        let needle: &[u8] = &needle[..needle.len().min(50)];
        bm_search(haystack, needle) == naive_search_first(haystack, needle)
    }

    #[quickcheck_macros::quickcheck]
    #[allow(clippy::needless_pass_by_value)]
    fn matches_naive_all(haystack: Vec<u8>, needle: Vec<u8>) -> bool {
        let haystack: &[u8] = &haystack[..haystack.len().min(50)];
        let needle: &[u8] = &needle[..needle.len().min(50)];
        bm_search_all(haystack, needle) == naive_search_all(haystack, needle)
    }
}
