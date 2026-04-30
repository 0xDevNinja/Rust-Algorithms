//! Manacher's algorithm for the longest palindromic substring in `O(n)`.
//!
//! The classic transformation interleaves a sentinel byte (`#`) between every
//! character and at both ends, e.g. `abba` becomes `#a#b#b#a#`. Every
//! palindrome of the original — odd or even length — corresponds to an
//! odd-length palindrome around some center of the transformed string, which
//! lets a single linear pass with a "rightmost reach" trick compute every
//! palindromic radius without ever rewinding.
//!
//! The implementation is byte-oriented (`&[u8]`). This keeps it Unicode-safe
//! in the sense that it never splits or panics on multi-byte sequences:
//! callers pass `s.as_bytes()` and receive a sub-slice of those bytes back.
//! The returned slice is always a valid sub-slice of the input bytes, but is
//! not guaranteed to be valid UTF-8 if the input contains multi-byte
//! characters whose bytes happen to form a palindrome that splits a code
//! point — for purely ASCII input this concern does not apply.
//!
//! # Complexity
//! - Time:  `O(n)`
//! - Space: `O(n)` for the radii array over the transformed string.

/// Sentinel byte used between characters in the transformed string.
///
/// The value must not appear in the input for the standard correctness
/// argument; in practice `b'#'` is fine because expansion stops at sentinel
/// vs non-sentinel mismatches anyway (sentinels only ever match sentinels and
/// non-sentinel bytes only ever match non-sentinel bytes), so the algorithm
/// is robust to inputs that contain `#` as well.
const SEP: u8 = b'#';

/// Returns the palindrome radii at each center of the transformed string
/// `# s[0] # s[1] # ... # s[n-1] #`.
///
/// The transformed string has length `2n + 1`. For each center `i`, the
/// returned value `p[i]` is the largest `r` such that the substring
/// `t[i - r ..= i + r]` is a palindrome (where `t` is the transformed
/// string). Even-length palindromes of the original are centered on `SEP`
/// positions; odd-length palindromes are centered on original-character
/// positions.
#[must_use]
pub fn palindrome_radii(s: &[u8]) -> Vec<usize> {
    if s.is_empty() {
        // Transformed string is just "#", radius 0.
        return vec![0];
    }

    // Build the transformed byte array: SEP between every char and at ends.
    let n = s.len();
    let m = 2 * n + 1;
    let mut t: Vec<u8> = Vec::with_capacity(m);
    t.push(SEP);
    for &b in s {
        t.push(b);
        t.push(SEP);
    }

    let mut p = vec![0_usize; m];
    let (mut center, mut right) = (0_usize, 0_usize);

    for i in 0..m {
        // Mirror of i around center.
        if i < right {
            let mirror = 2 * center - i;
            p[i] = p[mirror].min(right - i);
        }

        // Attempt to expand around i.
        let mut a = i + p[i] + 1;
        let mut b_idx = i.wrapping_sub(p[i] + 1);
        // b_idx underflows when the left side falls off the start; that
        // wraps to usize::MAX, which the bounds check catches via `b_idx < m`
        // being false in that case (we explicitly guard with i >= p[i] + 1).
        while a < m && i > p[i] && t[a] == t[b_idx] {
            p[i] += 1;
            a = i + p[i] + 1;
            b_idx = i.wrapping_sub(p[i] + 1);
        }

        if i + p[i] > right {
            center = i;
            right = i + p[i];
        }
    }

    p
}

/// Returns the longest palindromic substring of `s` as a byte sub-slice.
///
/// On ties the leftmost longest palindrome is returned. Empty input returns
/// an empty slice.
#[must_use]
pub fn longest_palindromic_substring(s: &[u8]) -> &[u8] {
    if s.is_empty() {
        return s;
    }

    let p = palindrome_radii(s);
    // Find the center with the largest radius; ties go to the leftmost.
    let (mut best_center, mut best_radius) = (0_usize, 0_usize);
    for (i, &r) in p.iter().enumerate() {
        if r > best_radius {
            best_radius = r;
            best_center = i;
        }
    }

    // In the transformed string, a center i with radius r corresponds to the
    // original substring of length r starting at (i - r) / 2.
    let start = (best_center - best_radius) / 2;
    &s[start..start + best_radius]
}

#[cfg(test)]
mod tests {
    use super::{longest_palindromic_substring, palindrome_radii};
    use quickcheck_macros::quickcheck;

    /// `O(n^2)` reference: expand around every possible center (odd + even).
    fn brute_force_longest(s: &[u8]) -> &[u8] {
        let n = s.len();
        if n == 0 {
            return s;
        }
        let (mut best_start, mut best_len) = (0_usize, 1_usize);

        let try_expand =
            |left: isize, right: isize, best_start: &mut usize, best_len: &mut usize| {
                let (mut l, mut r) = (left, right);
                while l >= 0 && (r as usize) < n && s[l as usize] == s[r as usize] {
                    let len = (r - l + 1) as usize;
                    if len > *best_len {
                        *best_len = len;
                        *best_start = l as usize;
                    }
                    l -= 1;
                    r += 1;
                }
            };

        for i in 0..n {
            // Odd-length palindrome centered at i.
            try_expand(i as isize, i as isize, &mut best_start, &mut best_len);
            // Even-length palindrome centered between i and i+1.
            if i + 1 < n {
                try_expand(i as isize, (i + 1) as isize, &mut best_start, &mut best_len);
            }
        }

        &s[best_start..best_start + best_len]
    }

    #[test]
    fn empty_input() {
        assert_eq!(longest_palindromic_substring(b""), b"");
        assert_eq!(palindrome_radii(b""), vec![0]);
    }

    #[test]
    fn single_char() {
        assert_eq!(longest_palindromic_substring(b"a"), b"a");
    }

    #[test]
    fn all_same_chars() {
        assert_eq!(longest_palindromic_substring(b"aaaa"), b"aaaa");
        assert_eq!(longest_palindromic_substring(b"aaaaaaa"), b"aaaaaaa");
    }

    #[test]
    fn no_palindrome_longer_than_one() {
        // Any single character is a palindrome of length 1; pick the leftmost.
        assert_eq!(longest_palindromic_substring(b"abcde"), b"a");
    }

    #[test]
    fn classic_babad() {
        // "babad" → "bab" or "aba"; we return the leftmost longest, "bab".
        assert_eq!(longest_palindromic_substring(b"babad"), b"bab");
    }

    #[test]
    fn racecar() {
        assert_eq!(longest_palindromic_substring(b"racecar"), b"racecar");
    }

    #[test]
    fn even_length_palindrome() {
        assert_eq!(longest_palindromic_substring(b"abba"), b"abba");
        assert_eq!(longest_palindromic_substring(b"cbbd"), b"bb");
    }

    #[test]
    fn embedded_palindrome() {
        // The longest palindrome inside "forgeeksskeegfor" is "geeksskeeg".
        assert_eq!(
            longest_palindromic_substring(b"forgeeksskeegfor"),
            b"geeksskeeg"
        );
    }

    #[test]
    fn unicode_via_as_bytes() {
        // Multi-byte characters are compared at the byte level. The reverse
        // of "ana" is itself, so it's a palindrome regardless of the
        // surrounding bytes from "café".
        let s = "café ana éfac";
        let bytes = s.as_bytes();
        let result = longest_palindromic_substring(bytes);
        // Reversing the bytes of the result must equal the bytes themselves.
        let reversed: Vec<u8> = result.iter().rev().copied().collect();
        assert_eq!(reversed, result.to_vec());
        // And the result must match what brute force finds on the same bytes.
        assert_eq!(result, brute_force_longest(bytes));
    }

    #[test]
    fn radii_sanity_for_aba() {
        // Transformed: # a # b # a #  (length 7)
        // Radii:       0 1 0 3 0 1 0
        assert_eq!(palindrome_radii(b"aba"), vec![0, 1, 0, 3, 0, 1, 0]);
    }

    #[test]
    fn radii_sanity_for_abba() {
        // Transformed: # a # b # b # a #  (length 9)
        // Radii:       0 1 0 1 4 1 0 1 0
        assert_eq!(palindrome_radii(b"abba"), vec![0, 1, 0, 1, 4, 1, 0, 1, 0]);
    }

    #[allow(clippy::needless_pass_by_value)]
    #[quickcheck]
    fn quickcheck_matches_brute_force(bytes: Vec<u8>) -> bool {
        // Cap length to keep the O(n^2) reference fast.
        let bytes: Vec<u8> = bytes.into_iter().take(30).collect();
        let manacher = longest_palindromic_substring(&bytes);
        let brute = brute_force_longest(&bytes);
        // Lengths must match. The actual byte content must also be a
        // palindrome and must equal the brute-force pick (we tie-break to
        // leftmost in both implementations).
        manacher.len() == brute.len() && manacher == brute
    }

    #[allow(clippy::needless_pass_by_value)]
    #[quickcheck]
    fn quickcheck_result_is_palindrome(bytes: Vec<u8>) -> bool {
        let bytes: Vec<u8> = bytes.into_iter().take(30).collect();
        let result = longest_palindromic_substring(&bytes);
        let reversed: Vec<u8> = result.iter().rev().copied().collect();
        reversed.as_slice() == result
    }
}
