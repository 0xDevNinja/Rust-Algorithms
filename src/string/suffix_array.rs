//! Suffix array construction via prefix doubling.
//!
//! The suffix array `SA` of a string `s` of length `n` is the permutation of
//! `0..n` that lists every suffix `s[i..]` in ascending lexicographic order.
//! That is, `SA[i]` is the start index of the `i`-th smallest suffix.
//!
//! This module implements the prefix-doubling construction: at round `k` each
//! suffix is ranked by its first `2^k` characters using the ranks computed in
//! the previous round, then the suffixes are sorted by `(rank[i], rank[i+2^k])`
//! pairs. After `O(log n)` rounds every suffix has a unique rank and the
//! permutation that achieved it is the suffix array.
//!
//! # Complexity
//! - Time:  O(n log² n) — `O(log n)` doubling rounds, each calling Rust's
//!   stable `sort_by` (`O(n log n)` comparisons).
//! - Space: O(n) for the ranks and the working permutation.
//!
//! # Use cases
//! - Substring search in `O(m log n)` via binary search on the suffix array.
//! - Building blocks for the Burrows–Wheeler transform, longest-common-substring
//!   queries (paired with the LCP array), and full-text indices used by
//!   bioinformatics pipelines and compression schemes.
//!
//! Operates on `&[u8]`. Callers with a `&str` can pass `s.as_bytes()`; the
//! resulting order matches `<` on `&str` for ASCII inputs and is well-defined
//! byte-wise lexicographic order for arbitrary UTF-8.

/// Returns the suffix array of `s`.
///
/// `SA[i]` is the start index (in `s`) of the `i`-th smallest suffix in
/// byte-wise lexicographic order. The returned vector is a permutation of
/// `0..s.len()`. For the empty input the result is an empty vector.
pub fn suffix_array(s: &[u8]) -> Vec<usize> {
    let n = s.len();
    if n == 0 {
        return Vec::new();
    }

    // Initial permutation sorted by the single byte at each position.
    let mut sa: Vec<usize> = (0..n).collect();
    sa.sort_by_key(|&i| s[i]);

    // Initial ranks: equal bytes share a rank, breaking ties later by `k`.
    let mut rank: Vec<usize> = vec![0; n];
    for i in 1..n {
        rank[sa[i]] = rank[sa[i - 1]] + usize::from(s[sa[i]] != s[sa[i - 1]]);
    }

    let mut k = 1_usize;
    while k < n {
        // Sort by the pair (rank[i], rank[i + k]); use `None` when `i + k >= n`
        // so shorter suffixes (which compare smaller at equal prefixes) come
        // first — `Option::None` is `Ord`-less than `Some(_)` in Rust.
        let key = |&i: &usize| -> (usize, Option<usize>) {
            (rank[i], if i + k < n { Some(rank[i + k]) } else { None })
        };
        sa.sort_by_key(key);

        // Recompute ranks based on the new ordering.
        let mut new_rank: Vec<usize> = vec![0; n];
        for i in 1..n {
            new_rank[sa[i]] = new_rank[sa[i - 1]] + usize::from(key(&sa[i]) != key(&sa[i - 1]));
        }
        rank = new_rank;

        // Early exit once every suffix has a distinct rank.
        if rank[sa[n - 1]] == n - 1 {
            break;
        }
        k *= 2;
    }

    sa
}

#[cfg(test)]
mod tests {
    use super::suffix_array;
    use quickcheck_macros::quickcheck;

    /// Brute-force reference: collect every suffix and sort lexicographically.
    fn brute_force(s: &[u8]) -> Vec<usize> {
        let n = s.len();
        let mut idx: Vec<usize> = (0..n).collect();
        idx.sort_by(|&a, &b| s[a..].cmp(&s[b..]));
        idx
    }

    #[test]
    fn empty_input() {
        assert_eq!(suffix_array(b""), Vec::<usize>::new());
    }

    #[test]
    fn single_char() {
        assert_eq!(suffix_array(b"a"), vec![0]);
        assert_eq!(suffix_array(b"z"), vec![0]);
    }

    #[test]
    fn two_chars() {
        // "ba" — suffix "a" (index 1) < "ba" (index 0).
        assert_eq!(suffix_array(b"ba"), vec![1, 0]);
        // "ab" — suffix "ab" (index 0) < "b" (index 1).
        assert_eq!(suffix_array(b"ab"), vec![0, 1]);
    }

    #[test]
    fn banana_classic() {
        // The textbook suffix array for "banana".
        assert_eq!(suffix_array(b"banana"), vec![5, 3, 1, 0, 4, 2]);
    }

    #[test]
    fn abracadabra() {
        let s = b"abracadabra";
        assert_eq!(suffix_array(s), brute_force(s));
    }

    #[test]
    fn all_equal_chars() {
        // For "aaaa", shorter suffixes come first under the empty-tail rule.
        assert_eq!(suffix_array(b"aaaa"), vec![3, 2, 1, 0]);
        assert_eq!(suffix_array(b"aaaaaaa"), vec![6, 5, 4, 3, 2, 1, 0]);
    }

    #[test]
    fn mississippi() {
        let s = b"mississippi";
        assert_eq!(suffix_array(s), brute_force(s));
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
            assert_eq!(suffix_array(s), brute_force(s), "input {s:?}");
        }
    }

    #[test]
    fn result_is_a_permutation() {
        let s = b"abracadabra";
        let mut sa = suffix_array(s);
        sa.sort_unstable();
        assert_eq!(sa, (0..s.len()).collect::<Vec<_>>());
    }

    #[allow(clippy::needless_pass_by_value)]
    #[quickcheck]
    fn quickcheck_matches_brute_force(bytes: Vec<u8>) -> bool {
        let bytes: Vec<u8> = bytes.into_iter().take(50).collect();
        suffix_array(&bytes) == brute_force(&bytes)
    }

    #[allow(clippy::needless_pass_by_value)]
    #[quickcheck]
    fn quickcheck_result_is_permutation(bytes: Vec<u8>) -> bool {
        let bytes: Vec<u8> = bytes.into_iter().take(50).collect();
        let mut sa = suffix_array(&bytes);
        sa.sort_unstable();
        sa == (0..bytes.len()).collect::<Vec<_>>()
    }
}
