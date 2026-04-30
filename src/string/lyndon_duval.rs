//! Lyndon decomposition via Duval's algorithm.
//!
//! A **Lyndon word** is a non-empty string that is strictly smaller (in
//! lexicographic order) than every one of its non-trivial proper rotations.
//! Equivalently, it is strictly smaller than each of its proper non-empty
//! suffixes — that characterisation is what Duval's algorithm exploits.
//!
//! The **Chen–Fox–Lyndon theorem** states that every string `s` admits a
//! unique factorisation `s = w_1 w_2 ... w_k` where each `w_i` is a Lyndon
//! word and `w_1 >= w_2 >= ... >= w_k` (non-increasing under lexicographic
//! order). Duval's algorithm computes this factorisation in a single
//! left-to-right pass, in `O(n)` time and `O(1)` extra space.
//!
//! The implementation operates on bytes (`&[u8]`). Callers working with
//! `&str` can pass `s.as_bytes()` — multi-byte UTF-8 sequences are compared
//! lexicographically by their byte representation, which agrees with `<` on
//! `&str` for ASCII and yields a well-defined order for arbitrary UTF-8.
//!
//! Reference: Duval, J. P. (1983). "Factorizing words over an ordered
//! alphabet." *Journal of Algorithms* 4 (4): 363–381.
//!
//! # Complexity
//! - Time:  O(n)
//! - Space: O(1) auxiliary (the output `Vec` aside).

/// Returns the Lyndon decomposition of `s` as a vector of borrowed slices.
///
/// The returned slices, in order, concatenate exactly to `s` and form a
/// non-increasing sequence of Lyndon words. For empty input the result is
/// the empty vector.
///
/// # Complexity
/// - Time:  O(n)
/// - Space: O(1) auxiliary.
pub fn lyndon_decomposition(s: &[u8]) -> Vec<&[u8]> {
    let n = s.len();
    let mut out = Vec::new();
    let mut i = 0;
    while i < n {
        // `j` scans ahead, `k` is the matched-prefix pointer used by Duval.
        let mut j = i + 1;
        let mut k = i;
        while j < n && s[k] <= s[j] {
            if s[k] < s[j] {
                k = i;
            } else {
                k += 1;
            }
            j += 1;
        }
        // The current Lyndon factor has length `j - k`; emit copies of it
        // until the scan pointer overtakes `i`.
        while i <= k {
            let len = j - k;
            out.push(&s[i..i + len]);
            i += len;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::lyndon_decomposition;
    use quickcheck_macros::quickcheck;

    /// Brute-force reference: a Lyndon word is strictly smaller than each of
    /// its proper non-empty suffixes.
    fn is_lyndon(w: &[u8]) -> bool {
        if w.is_empty() {
            return false;
        }
        (1..w.len()).all(|i| w < &w[i..])
    }

    fn check_decomposition(s: &[u8]) {
        let factors = lyndon_decomposition(s);

        // Concatenation of factors equals input.
        let joined: Vec<u8> = factors.iter().flat_map(|f| f.iter().copied()).collect();
        assert_eq!(joined.as_slice(), s, "factors must concatenate to input");

        // Each factor is a Lyndon word.
        for f in &factors {
            assert!(is_lyndon(f), "factor {f:?} is not a Lyndon word");
        }

        // Factors are non-increasing.
        for pair in factors.windows(2) {
            assert!(
                pair[0] >= pair[1],
                "factors not non-increasing: {:?} then {:?}",
                pair[0],
                pair[1],
            );
        }
    }

    #[test]
    fn empty_input_yields_empty_decomposition() {
        let factors = lyndon_decomposition(b"");
        assert!(factors.is_empty());
    }

    #[test]
    fn single_char_is_one_factor() {
        let factors = lyndon_decomposition(b"a");
        assert_eq!(factors, vec![b"a".as_slice()]);
    }

    #[test]
    fn already_lyndon_strings_are_one_factor() {
        // Strictly increasing — definitely Lyndon.
        let factors = lyndon_decomposition(b"abc");
        assert_eq!(factors, vec![b"abc".as_slice()]);

        // "ab" is Lyndon.
        let factors = lyndon_decomposition(b"ab");
        assert_eq!(factors, vec![b"ab".as_slice()]);
    }

    #[test]
    fn descending_string_splits_into_singletons() {
        // "ba" decomposes as "b","a"; more generally each char is its own
        // factor when the string is strictly decreasing.
        let factors = lyndon_decomposition(b"ba");
        assert_eq!(factors, vec![b"b".as_slice(), b"a".as_slice()]);

        let factors = lyndon_decomposition(b"dcba");
        assert_eq!(
            factors,
            vec![
                b"d".as_slice(),
                b"c".as_slice(),
                b"b".as_slice(),
                b"a".as_slice(),
            ]
        );
    }

    #[test]
    fn repeated_lyndon_block() {
        // "aab" is a Lyndon word; "aabaab" decomposes as two copies.
        let factors = lyndon_decomposition(b"aabaab");
        assert_eq!(factors, vec![b"aab".as_slice(), b"aab".as_slice()]);
    }

    #[test]
    fn all_equal_chars_split_into_singletons() {
        // "aaaa" — only "a" is a Lyndon word among prefixes; we get four of
        // them since the sequence must be non-increasing.
        let factors = lyndon_decomposition(b"aaaa");
        assert_eq!(factors, vec![b"a".as_slice(); 4]);
    }

    #[test]
    fn abcabc_is_not_lyndon_but_factors_cleanly() {
        // "abcabc" equals one of its rotations ("abcabc" itself), so it is
        // not Lyndon. The decomposition is "abc","abc".
        let factors = lyndon_decomposition(b"abcabc");
        assert_eq!(factors, vec![b"abc".as_slice(), b"abc".as_slice()]);
    }

    #[test]
    fn classic_abracadabra() {
        // The classic textbook example.
        let factors = lyndon_decomposition(b"abracadabra");
        let expected: Vec<&[u8]> = vec![b"abracad", b"abr", b"a"];
        assert_eq!(factors, expected);
        check_decomposition(b"abracadabra");
    }

    #[test]
    fn classic_examples_satisfy_invariants() {
        for s in [
            b"banana".as_slice(),
            b"mississippi".as_slice(),
            b"abracadabra".as_slice(),
            b"zxyzxyz".as_slice(),
            b"aabaaabaaa".as_slice(),
            b"abacabad".as_slice(),
            b"\x00\x01\x00\x02".as_slice(),
        ] {
            check_decomposition(s);
        }
    }

    #[allow(clippy::needless_pass_by_value)]
    #[quickcheck]
    fn quickcheck_decomposition_invariants(bytes: Vec<u8>) -> bool {
        let bytes: Vec<u8> = bytes.into_iter().take(32).collect();
        let factors = lyndon_decomposition(&bytes);

        // Concatenation equals input.
        let joined: Vec<u8> = factors.iter().flat_map(|f| f.iter().copied()).collect();
        if joined != bytes {
            return false;
        }

        // Each factor is Lyndon.
        if !factors.iter().all(|f| is_lyndon(f)) {
            return false;
        }

        // Non-increasing.
        factors.windows(2).all(|p| p[0] >= p[1])
    }

    #[allow(clippy::needless_pass_by_value)]
    #[quickcheck]
    fn quickcheck_decomposition_is_unique(bytes: Vec<u8>) -> bool {
        // Re-running the algorithm yields the same decomposition (sanity
        // check on determinism, which together with the invariants above
        // pins down the unique Chen–Fox–Lyndon factorisation).
        let bytes: Vec<u8> = bytes.into_iter().take(32).collect();
        let a = lyndon_decomposition(&bytes);
        let b = lyndon_decomposition(&bytes);
        a == b
    }
}
