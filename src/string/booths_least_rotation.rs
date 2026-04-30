//! Booth's algorithm for the lexicographically least rotation of a string.
//!
//! Given a sequence of bytes `s`, returns the starting index `k` such that the
//! rotation `s[k..] ++ s[..k]` is the lexicographically smallest among all `n`
//! rotations of `s`. Runs in `O(n)` time and `O(n)` extra space using the
//! classic failure-function-on-the-doubled-string trick.
//!
//! The implementation operates on bytes (`&[u8]`). Callers working with `&str`
//! can pass `s.as_bytes()` — multi-byte UTF-8 sequences are compared
//! lexicographically by their byte representation, which agrees with `<` on
//! `&str` for ASCII and yields a well-defined ordering for arbitrary UTF-8.
//!
//! Reference: Booth, K. S. (1980). "Lexicographically least circular
//! substrings." *Information Processing Letters* 10 (4–5): 240–242.
//!
//! # Complexity
//! - Time:  O(n)
//! - Space: O(n) for the failure-function table over the doubled string.

/// Returns the starting index of the lexicographically least rotation of `s`.
///
/// For empty input the returned index is `0`. For non-empty input the result
/// `k` lies in `0..s.len()` and `s[k..] ++ s[..k]` is the minimum rotation
/// under byte-wise lexicographic order. Ties (equal rotations) are broken by
/// the standard Booth tie-breaking rule, which selects the smallest such `k`
/// the algorithm discovers during its single linear scan.
pub fn booths_least_rotation(s: &[u8]) -> usize {
    let n = s.len();
    if n == 0 {
        return 0;
    }

    // Failure function over the doubled string s ++ s.
    let mut f: Vec<isize> = vec![-1; 2 * n];
    // Running candidate index of the least rotation found so far.
    let mut k: usize = 0;

    for i in 1..2 * n {
        let mut j = f[i - k - 1];
        // Walk back along the failure links while there is a mismatch.
        while j != -1 && s[i % n] != s[(k + (j as usize) + 1) % n] {
            if s[i % n] < s[(k + (j as usize) + 1) % n] {
                k = i - (j as usize) - 1;
            }
            j = f[j as usize];
        }
        if j == -1 && s[i % n] != s[(k + (j.wrapping_add(1) as usize)) % n] {
            // Mismatch at the root of the failure chain.
            if s[i % n] < s[(k + (j.wrapping_add(1) as usize)) % n] {
                k = i;
            }
            f[i - k] = -1;
        } else {
            f[i - k] = j + 1;
        }
    }

    k
}

#[cfg(test)]
mod tests {
    use super::booths_least_rotation;
    use quickcheck_macros::quickcheck;

    /// Brute-force reference: try every rotation, return the index of the min.
    fn brute_force(s: &[u8]) -> usize {
        let n = s.len();
        if n == 0 {
            return 0;
        }
        let mut best = 0_usize;
        for k in 1..n {
            // Compare rotation starting at k against rotation starting at best.
            let mut less = false;
            let mut greater = false;
            for i in 0..n {
                let a = s[(k + i) % n];
                let b = s[(best + i) % n];
                if a < b {
                    less = true;
                    break;
                }
                if a > b {
                    greater = true;
                    break;
                }
            }
            if less && !greater {
                best = k;
            }
        }
        best
    }

    #[test]
    fn empty_returns_zero() {
        assert_eq!(booths_least_rotation(b""), 0);
    }

    #[test]
    fn single_char_returns_zero() {
        assert_eq!(booths_least_rotation(b"a"), 0);
        assert_eq!(booths_least_rotation(b"z"), 0);
    }

    #[test]
    fn already_minimal() {
        assert_eq!(booths_least_rotation(b"abcd"), 0);
    }

    #[test]
    fn rotated_once() {
        // "dabc" rotated by 1 yields "abcd".
        assert_eq!(booths_least_rotation(b"dabc"), 1);
    }

    #[test]
    fn rotation_index_two() {
        // "bca" rotated by 2 yields "abc".
        assert_eq!(booths_least_rotation(b"bca"), 2);
    }

    #[test]
    fn aab_minimal_at_zero() {
        assert_eq!(booths_least_rotation(b"aab"), 0);
    }

    #[test]
    fn all_equal_chars() {
        // Every rotation is identical; Booth picks the smallest valid index.
        assert_eq!(booths_least_rotation(b"aaaa"), 0);
        assert_eq!(booths_least_rotation(b"aaaaaaa"), 0);
    }

    #[test]
    fn classic_examples() {
        // "cabcab" — minimum rotation is "abcabc" starting at index 1.
        assert_eq!(booths_least_rotation(b"cabcab"), 1);
        // "bbaaccaadd" — verify against brute force.
        let s = b"bbaaccaadd";
        assert_eq!(booths_least_rotation(s), brute_force(s));
    }

    #[test]
    fn matches_brute_force_known_strings() {
        for s in [
            b"banana".as_slice(),
            b"mississippi".as_slice(),
            b"abracadabra".as_slice(),
            b"zxyzxyz".as_slice(),
            b"aabaaabaaa".as_slice(),
            b"\x00\x01\x00\x02".as_slice(),
        ] {
            assert_eq!(booths_least_rotation(s), brute_force(s), "input {s:?}");
        }
    }

    #[allow(clippy::needless_pass_by_value)]
    #[quickcheck]
    fn quickcheck_matches_brute_force(bytes: Vec<u8>) -> bool {
        let bytes: Vec<u8> = bytes.into_iter().take(16).collect();
        let k = booths_least_rotation(&bytes);
        let bf = brute_force(&bytes);
        if bytes.is_empty() {
            return k == 0 && bf == 0;
        }
        // The two indices may differ when ties exist, but the rotations they
        // produce must be byte-equal — that's the actual invariant.
        let n = bytes.len();
        (0..n).all(|i| bytes[(k + i) % n] == bytes[(bf + i) % n])
    }

    #[allow(clippy::needless_pass_by_value)]
    #[quickcheck]
    fn quickcheck_result_is_minimum_rotation(bytes: Vec<u8>) -> bool {
        let bytes: Vec<u8> = bytes.into_iter().take(16).collect();
        let n = bytes.len();
        if n == 0 {
            return booths_least_rotation(&bytes) == 0;
        }
        let k = booths_least_rotation(&bytes);
        let rot_k: Vec<u8> = (0..n).map(|i| bytes[(k + i) % n]).collect();
        // No other rotation is strictly smaller.
        for j in 0..n {
            let rot_j: Vec<u8> = (0..n).map(|i| bytes[(j + i) % n]).collect();
            if rot_j < rot_k {
                return false;
            }
        }
        true
    }
}
