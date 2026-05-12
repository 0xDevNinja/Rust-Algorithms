//! Decode ways: count the number of ways to decode a digit string where the
//! mapping is `1 = A`, `2 = B`, ..., `26 = Z`. Empty string decodes one way
//! (the empty decoding); any string starting with `'0'` decodes zero ways.
//!
//! Recurrence (with `dp[0] = 1`):
//!   `dp[i] = (single-digit s[i-1] in 1..=9 ? dp[i-1] : 0)`
//!         `+ (two-digit  s[i-2..i] in 10..=26 ? dp[i-2] : 0)`.
//!
//! Runs in O(n) time and O(1) auxiliary space (rolling window).
//!
//! # Panics
//! Panics if `s` contains a non-ASCII-digit byte. Callers must validate
//! their input; only `'0'..='9'` characters are accepted.

/// Returns the number of distinct decodings of the digit string `s`.
///
/// # Panics
/// Panics if `s` contains any character outside `'0'..='9'`.
pub fn num_decodings(s: &str) -> u64 {
    let bytes = s.as_bytes();
    for &b in bytes {
        assert!(
            b.is_ascii_digit(),
            "num_decodings: non-digit input byte {b:#x}"
        );
    }
    let n = bytes.len();
    if n == 0 {
        return 1;
    }
    if bytes[0] == b'0' {
        return 0;
    }

    // Rolling window: prev2 = dp[i-2], prev1 = dp[i-1].
    let mut prev2: u64 = 1; // dp[0]
    let mut prev1: u64 = 1; // dp[1] (since s[0] != '0')
    for i in 2..=n {
        let mut cur: u64 = 0;
        let one = bytes[i - 1];
        if one != b'0' {
            cur += prev1;
        }
        let tens = bytes[i - 2] - b'0';
        let ones = bytes[i - 1] - b'0';
        let two = (tens as u32) * 10 + ones as u32;
        if (10..=26).contains(&two) {
            cur += prev2;
        }
        prev2 = prev1;
        prev1 = cur;
    }
    prev1
}

#[cfg(test)]
mod tests {
    use super::num_decodings;

    #[test]
    fn empty_string_is_one() {
        assert_eq!(num_decodings(""), 1);
    }

    #[test]
    fn twelve_two_ways() {
        assert_eq!(num_decodings("12"), 2);
    }

    #[test]
    fn two_two_six_three_ways() {
        assert_eq!(num_decodings("226"), 3);
    }

    #[test]
    fn leading_zero_is_zero() {
        assert_eq!(num_decodings("0"), 0);
    }

    #[test]
    fn ten_one_way() {
        assert_eq!(num_decodings("10"), 1);
    }

    #[test]
    fn twenty_seven_one_way() {
        assert_eq!(num_decodings("27"), 1);
    }

    #[test]
    fn one_hundred_zero_ways() {
        assert_eq!(num_decodings("100"), 0);
    }

    #[test]
    fn zero_six_zero_ways() {
        assert_eq!(num_decodings("06"), 0);
    }

    #[test]
    fn eleven_one_zero_six_two_ways() {
        assert_eq!(num_decodings("11106"), 2);
    }

    #[test]
    #[should_panic(expected = "non-digit input byte")]
    fn non_digit_panics() {
        let _ = num_decodings("12a");
    }
}
