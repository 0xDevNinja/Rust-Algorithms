//! Run-length encoding (RLE) over Unicode strings.
//!
//! Each maximal run of identical characters is encoded as `<count><char>`
//! where `<count>` is a decimal integer (one or more ASCII digits) and
//! `<char>` is a single Unicode scalar value. For example:
//!
//! ```text
//! "aaabbc"  -> "3a2b1c"
//! "ééé"     -> "3é"
//! ""        -> ""
//! ```
//!
//! # Complexity
//!
//! Both [`rle_encode`] and [`rle_decode`] run in `O(n)` time and `O(n)`
//! space, where `n` is the number of Unicode scalar values in the input
//! (encoding) or the number of bytes (decoding). Iteration is over
//! `chars()`, so the encoding is Unicode-aware and never splits a
//! multi-byte scalar.
//!
//! # Format restriction (digits)
//!
//! Because counts are decimal and the format is unframed, the alphabet of
//! data characters cannot contain ASCII digits (`'0'..='9'`); otherwise
//! the encoding would be ambiguous (e.g. `"11"` could be a run of two
//! `'1'` or eleven of `'\0'`). [`rle_encode`] therefore **panics** if its
//! input contains an ASCII digit. [`rle_decode`] does not need a special
//! check: it parses leading digits as a count, so a digit appearing where
//! a data character is expected simply starts the next run's count.
//!
//! # Errors (decoding)
//!
//! [`rle_decode`] returns `None` for malformed input:
//! - leading data character with no count (e.g. `"a"`),
//! - trailing count with no data character (e.g. `"1"`, `"12"`),
//! - a zero count (e.g. `"0a"`), which would be redundant.

/// Encode `s` as run-length pairs `<count><char>`.
///
/// # Panics
///
/// Panics if `s` contains an ASCII digit, since the format reserves digits
/// for run counts and would otherwise produce ambiguous output.
pub fn rle_encode(s: &str) -> String {
    assert!(
        !s.chars().any(|c| c.is_ascii_digit()),
        "rle_encode: input must not contain ASCII digits (the format reserves them for counts)"
    );

    let mut out = String::new();
    let mut chars = s.chars();
    let Some(mut current) = chars.next() else {
        return out;
    };
    let mut count: usize = 1;
    for c in chars {
        if c == current {
            count += 1;
        } else {
            out.push_str(&count.to_string());
            out.push(current);
            current = c;
            count = 1;
        }
    }
    out.push_str(&count.to_string());
    out.push(current);
    out
}

/// Decode an RLE-encoded string. Returns `None` on malformed input.
pub fn rle_decode(s: &str) -> Option<String> {
    let mut out = String::new();
    let mut chars = s.chars().peekable();

    while let Some(&c) = chars.peek() {
        // Parse one or more decimal digits as the count.
        if !c.is_ascii_digit() {
            // Data character with no preceding count: malformed.
            return None;
        }
        let mut count: usize = 0;
        while let Some(&d) = chars.peek() {
            if let Some(digit) = d.to_digit(10) {
                count = count.checked_mul(10)?.checked_add(digit as usize)?;
                chars.next();
            } else {
                break;
            }
        }
        if count == 0 {
            // Zero-count runs are forbidden (would be redundant / ambiguous).
            return None;
        }
        // The next char (if any) is the data character.
        let data = chars.next()?;
        for _ in 0..count {
            out.push(data);
        }
    }

    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use quickcheck_macros::quickcheck;

    // ---- encode ----

    #[test]
    fn encode_empty() {
        assert_eq!(rle_encode(""), "");
    }

    #[test]
    fn encode_single() {
        assert_eq!(rle_encode("a"), "1a");
    }

    #[test]
    fn encode_pair() {
        assert_eq!(rle_encode("aa"), "2a");
    }

    #[test]
    fn encode_all_distinct() {
        assert_eq!(rle_encode("abc"), "1a1b1c");
    }

    #[test]
    fn encode_mixed() {
        assert_eq!(rle_encode("aaabbc"), "3a2b1c");
    }

    #[test]
    fn encode_multi_digit_count() {
        let s = "x".repeat(12);
        assert_eq!(rle_encode(&s), "12x");
    }

    #[test]
    fn encode_unicode() {
        assert_eq!(rle_encode("ééé"), "3é");
    }

    #[test]
    #[should_panic(expected = "must not contain ASCII digits")]
    fn encode_panics_on_digits() {
        let _ = rle_encode("a1b");
    }

    // ---- decode ----

    #[test]
    fn decode_empty() {
        assert_eq!(rle_decode("").as_deref(), Some(""));
    }

    #[test]
    fn decode_single() {
        assert_eq!(rle_decode("1a").as_deref(), Some("a"));
    }

    #[test]
    fn decode_mixed() {
        assert_eq!(rle_decode("3a2b1c").as_deref(), Some("aaabbc"));
    }

    #[test]
    fn decode_multi_digit_count() {
        let expected = "x".repeat(12);
        assert_eq!(rle_decode("12x").as_deref(), Some(expected.as_str()));
    }

    #[test]
    fn decode_unicode() {
        assert_eq!(rle_decode("3é").as_deref(), Some("ééé"));
    }

    #[test]
    fn decode_rejects_no_count() {
        assert_eq!(rle_decode("a"), None);
    }

    #[test]
    fn decode_rejects_zero_count() {
        assert_eq!(rle_decode("0a"), None);
    }

    #[test]
    fn decode_rejects_trailing_count_single_digit() {
        assert_eq!(rle_decode("1"), None);
    }

    #[test]
    fn decode_rejects_trailing_count_multi_digit() {
        assert_eq!(rle_decode("2"), None);
    }

    // ---- roundtrip property ----

    #[quickcheck]
    fn roundtrip_encode_then_decode(mut s: String) -> bool {
        // Strip ASCII digits so the input is always valid for `rle_encode`
        // (the format reserves digits for run counts). `retain` consumes the
        // input by mutating it in place, which satisfies clippy's
        // `needless_pass_by_value` lint without an extra allocation.
        s.retain(|c| !c.is_ascii_digit());
        rle_decode(&rle_encode(&s)).as_deref() == Some(s.as_str())
    }
}
