//! Longest palindromic subsequence. DP over (i, j) windows. O(n²) time and
//! space.

/// Returns the length of the longest palindromic subsequence of `s`.
pub fn lps_length(s: &str) -> usize {
    let chars: Vec<char> = s.chars().collect();
    let n = chars.len();
    if n == 0 {
        return 0;
    }
    let mut dp = vec![vec![0_usize; n]; n];
    for i in 0..n {
        dp[i][i] = 1;
    }
    for len in 2..=n {
        for i in 0..=n - len {
            let j = i + len - 1;
            dp[i][j] = if chars[i] == chars[j] {
                if len == 2 {
                    2
                } else {
                    dp[i + 1][j - 1] + 2
                }
            } else {
                dp[i + 1][j].max(dp[i][j - 1])
            };
        }
    }
    dp[0][n - 1]
}

/// Returns one longest palindromic subsequence of `s` (as a `String`).
pub fn lps_string(s: &str) -> String {
    let chars: Vec<char> = s.chars().collect();
    let n = chars.len();
    if n == 0 {
        return String::new();
    }
    let mut dp = vec![vec![0_usize; n]; n];
    for i in 0..n {
        dp[i][i] = 1;
    }
    for len in 2..=n {
        for i in 0..=n - len {
            let j = i + len - 1;
            dp[i][j] = if chars[i] == chars[j] {
                if len == 2 {
                    2
                } else {
                    dp[i + 1][j - 1] + 2
                }
            } else {
                dp[i + 1][j].max(dp[i][j - 1])
            };
        }
    }
    let mut left = Vec::new();
    let mut right = Vec::new();
    let (mut i, mut j) = (0_usize, n - 1);
    while i < j {
        if chars[i] == chars[j] {
            left.push(chars[i]);
            right.push(chars[j]);
            i += 1;
            j -= 1;
        } else if dp[i + 1][j] >= dp[i][j - 1] {
            i += 1;
        } else {
            j -= 1;
        }
    }
    if i == j {
        left.push(chars[i]);
    }
    let mut out: String = left.into_iter().collect();
    out.extend(right.into_iter().rev());
    out
}

#[cfg(test)]
mod tests {
    use super::{lps_length, lps_string};

    fn is_subseq(sub: &str, full: &str) -> bool {
        let mut iter = full.chars();
        sub.chars().all(|c| iter.any(|x| x == c))
    }

    fn is_palindrome(s: &str) -> bool {
        let v: Vec<char> = s.chars().collect();
        let n = v.len();
        (0..n / 2).all(|i| v[i] == v[n - 1 - i])
    }

    #[test]
    fn empty() {
        assert_eq!(lps_length(""), 0);
        assert_eq!(lps_string(""), "");
    }

    #[test]
    fn single_char() {
        assert_eq!(lps_length("a"), 1);
        assert_eq!(lps_string("a"), "a");
    }

    #[test]
    fn classic_bbabcbcab() {
        let s = "bbabcbcab";
        assert_eq!(lps_length(s), 7);
        let lps = lps_string(s);
        assert_eq!(lps.chars().count(), 7);
        assert!(is_palindrome(&lps));
        assert!(is_subseq(&lps, s));
    }

    #[test]
    fn already_palindrome() {
        let s = "racecar";
        assert_eq!(lps_length(s), 7);
        assert_eq!(lps_string(s), "racecar");
    }

    #[test]
    fn no_repeats() {
        // Each char is its own LPS (length 1).
        assert_eq!(lps_length("abcde"), 1);
        assert_eq!(lps_string("abcde").chars().count(), 1);
    }

    #[test]
    fn all_same() {
        assert_eq!(lps_length("aaaaa"), 5);
        assert_eq!(lps_string("aaaaa"), "aaaaa");
    }
}
