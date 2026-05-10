//! LeetCode-style `atoi`: parse a signed 32-bit integer from a string slice.
//!
//! The parser mimics the classic C `atoi` semantics with explicit clamping:
//!
//! 1. Skip leading ASCII whitespace.
//! 2. Read an optional `+` or `-` sign (at most one).
//! 3. Read consecutive ASCII digits (`0`..=`9`) and accumulate them into an
//!    `i32`, clamping to the `i32` range on overflow.
//! 4. Stop at the first non-digit character (or end of input).
//!
//! If no digits are read after the optional sign, the result is `0`.
//!
//! # Complexity
//!
//! Runs in `O(n)` time and `O(1)` extra space, where `n = s.len()`. Each byte
//! of `s` is examined at most once.
//!
//! # Examples
//!
//! ```
//! use rust_algorithms::string::atoi::my_atoi;
//!
//! assert_eq!(my_atoi("42"), 42);
//! assert_eq!(my_atoi("   -42"), -42);
//! assert_eq!(my_atoi("4193 with words"), 4193);
//! assert_eq!(my_atoi("words and 987"), 0);
//! assert_eq!(my_atoi("-91283472332"), i32::MIN);
//! ```
//!
//! # References
//!
//! - `LeetCode` 8. *String to Integer (atoi)*.

/// Parse a signed 32-bit integer from `s` using LeetCode-style `atoi` rules.
///
/// Skips leading ASCII whitespace, accepts an optional single `+` or `-`
/// sign, then reads ASCII digits until a non-digit byte (or end of input)
/// is encountered. The accumulated value is clamped to the `i32` range on
/// overflow. Returns `0` when no digits are present after the optional
/// sign.
///
/// # Complexity
///
/// `O(n)` time, `O(1)` space, where `n = s.len()`.
pub fn my_atoi(s: &str) -> i32 {
    let bytes = s.as_bytes();
    let mut i = 0usize;
    let n = bytes.len();

    // 1) Skip leading ASCII whitespace.
    while i < n && bytes[i].is_ascii_whitespace() {
        i += 1;
    }

    // 2) Optional single sign.
    let mut negative = false;
    if i < n {
        match bytes[i] {
            b'+' => {
                i += 1;
            }
            b'-' => {
                negative = true;
                i += 1;
            }
            _ => {}
        }
    }

    // 3) Accumulate digits with saturation.
    let mut result: i32 = 0;
    let mut any_digit = false;
    while i < n {
        let c = bytes[i];
        if !c.is_ascii_digit() {
            break;
        }
        any_digit = true;
        let digit = (c - b'0') as i32;

        // result = result * 10 +/- digit, with saturation.
        if negative {
            match result.checked_mul(10).and_then(|v| v.checked_sub(digit)) {
                Some(v) => result = v,
                None => return i32::MIN,
            }
        } else {
            match result.checked_mul(10).and_then(|v| v.checked_add(digit)) {
                Some(v) => result = v,
                None => return i32::MAX,
            }
        }
        i += 1;
    }

    if any_digit {
        result
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_plain_number() {
        assert_eq!(my_atoi("42"), 42);
    }

    #[test]
    fn parses_negative_with_leading_whitespace() {
        assert_eq!(my_atoi("   -42"), -42);
    }

    #[test]
    fn stops_at_first_non_digit() {
        assert_eq!(my_atoi("4193 with words"), 4193);
    }

    #[test]
    fn returns_zero_when_no_leading_digits() {
        assert_eq!(my_atoi("words and 987"), 0);
    }

    #[test]
    fn clamps_to_i32_min_on_negative_overflow() {
        assert_eq!(my_atoi("-91283472332"), i32::MIN);
    }

    #[test]
    fn clamps_to_i32_max_on_positive_overflow() {
        assert_eq!(my_atoi("91283472332"), i32::MAX);
    }

    #[test]
    fn empty_string_returns_zero() {
        assert_eq!(my_atoi(""), 0);
    }

    #[test]
    fn explicit_plus_sign() {
        assert_eq!(my_atoi("+1"), 1);
    }

    #[test]
    fn multiple_signs_return_zero() {
        // After consuming '+', the '-' is not a digit, so no digits are read.
        assert_eq!(my_atoi("  +-12"), 0);
    }

    #[test]
    fn leading_zeros_then_non_digit_non_sign() {
        // "00000" parses to 0; the '-' terminates the digit run.
        assert_eq!(my_atoi("00000-42a1234"), 0);
    }

    #[test]
    fn whitespace_only_returns_zero() {
        assert_eq!(my_atoi("   \t\n"), 0);
    }

    #[test]
    fn sign_only_returns_zero() {
        assert_eq!(my_atoi("+"), 0);
        assert_eq!(my_atoi("-"), 0);
    }

    #[test]
    fn boundary_values_parse_exactly() {
        assert_eq!(my_atoi("2147483647"), i32::MAX);
        assert_eq!(my_atoi("-2147483648"), i32::MIN);
    }

    #[test]
    fn just_above_max_clamps() {
        assert_eq!(my_atoi("2147483648"), i32::MAX);
        assert_eq!(my_atoi("-2147483649"), i32::MIN);
    }

    #[test]
    fn leading_zeros_then_digits() {
        assert_eq!(my_atoi("0000123"), 123);
        assert_eq!(my_atoi("-0000123"), -123);
    }

    #[test]
    fn plus_then_digits_after_whitespace() {
        assert_eq!(my_atoi("   +0 123"), 0);
    }

    #[test]
    fn sign_after_whitespace_then_text_returns_zero() {
        assert_eq!(my_atoi("  -word"), 0);
    }
}
