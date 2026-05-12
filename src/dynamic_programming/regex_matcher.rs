//! Regular expression matcher with `.` and `*`.
//!
//! Supports a LeetCode-style minimal regex grammar:
//! - `.` matches any single character.
//! - `*` matches zero or more occurrences of the preceding element
//!   (which is either a literal character or `.`).
//!
//! The match must cover the entire input string (anchored).
//!
//! # Algorithm
//!
//! Let `m = s.len()` and `n = p.len()`. Define a boolean DP table
//! `dp[i][j]` = "does `s[..i]` match `p[..j]`?". Transitions:
//!
//! - If `p[j-1]` is a normal char or `.`:
//!   `dp[i][j] = dp[i-1][j-1] && (p[j-1] == '.' || s[i-1] == p[j-1])`
//! - If `p[j-1]` is `*` (paired with `p[j-2]`):
//!   - Zero copies: `dp[i][j] = dp[i][j-2]`
//!   - One or more copies (when `p[j-2]` matches `s[i-1]`):
//!     `dp[i][j] |= dp[i-1][j]`
//!
//! Base case: `dp[0][0] = true`. The first row is seeded so that patterns
//! like `a*b*c*` can match the empty string.
//!
//! # Complexity
//!
//! - Time: `O(m * n)`
//! - Space: `O(m * n)`

/// Returns whether `s` matches the regular expression `p` using the
/// `.`/`*` grammar described in the module docs.
pub fn is_match(s: &str, p: &str) -> bool {
    let s: Vec<char> = s.chars().collect();
    let p: Vec<char> = p.chars().collect();
    let m = s.len();
    let n = p.len();

    let mut dp = vec![vec![false; n + 1]; m + 1];
    dp[0][0] = true;

    // Empty string against patterns like a*, a*b*, a*b*c*.
    for j in 1..=n {
        if p[j - 1] == '*' && j >= 2 {
            dp[0][j] = dp[0][j - 2];
        }
    }

    for i in 1..=m {
        for j in 1..=n {
            let pc = p[j - 1];
            if pc == '*' {
                // `*` must follow a real element; treat malformed patterns
                // as non-matching by leaving dp[i][j] = false when j < 2.
                if j < 2 {
                    continue;
                }
                let prev = p[j - 2];
                // Zero occurrences of the preceding element.
                dp[i][j] = dp[i][j - 2];
                // One or more, if the preceding element matches s[i-1].
                if prev == '.' || prev == s[i - 1] {
                    dp[i][j] = dp[i][j] || dp[i - 1][j];
                }
            } else if pc == '.' || pc == s[i - 1] {
                dp[i][j] = dp[i - 1][j - 1];
            }
        }
    }

    dp[m][n]
}

#[cfg(test)]
mod tests {
    use super::is_match;

    #[test]
    fn empty_matches_empty() {
        assert!(is_match("", ""));
    }

    #[test]
    fn single_literal() {
        assert!(is_match("a", "a"));
    }

    #[test]
    fn literal_too_short() {
        assert!(!is_match("aa", "a"));
    }

    #[test]
    fn star_repeats_previous() {
        assert!(is_match("aa", "a*"));
    }

    #[test]
    fn dot_star_matches_anything() {
        assert!(is_match("ab", ".*"));
    }

    #[test]
    fn mixed_star_groups() {
        assert!(is_match("aab", "c*a*b"));
    }

    #[test]
    fn mississippi_negative() {
        assert!(!is_match("mississippi", "mis*is*p*."));
    }

    #[test]
    fn star_can_consume_zero() {
        assert!(is_match("a", "ab*"));
    }

    #[test]
    fn empty_string_against_starred_pattern() {
        assert!(is_match("", "a*b*c*"));
        assert!(is_match("", ".*"));
        assert!(!is_match("", "a"));
        assert!(!is_match("", "."));
    }

    #[test]
    fn dot_matches_single_char() {
        assert!(is_match("z", "."));
        assert!(!is_match("zz", "."));
        assert!(is_match("abc", "..."));
    }

    #[test]
    fn dot_star_then_literal() {
        assert!(is_match("aaab", ".*b"));
        assert!(!is_match("aaab", ".*c"));
    }

    #[test]
    fn star_followed_by_literal() {
        assert!(is_match("aaa", "a*a"));
        assert!(is_match("aaaa", "a*aa"));
    }

    #[test]
    fn complex_pattern() {
        assert!(is_match("mississippi", "mis*is*ip*."));
        assert!(is_match("aaa", "ab*a*c*a"));
    }

    #[test]
    fn nonmatching_literal_against_starred() {
        assert!(!is_match("ab", ".*c"));
    }
}
