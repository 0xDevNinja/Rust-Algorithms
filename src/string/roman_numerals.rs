//! Bidirectional Roman numeral conversion in the standard range `[1, 3999]`.
//!
//! # Grammar (canonical / standard form)
//!
//! ```text
//! roman      := thousands hundreds tens units
//! thousands  := "M"{0,3}
//! hundreds   := "CM" | "CD" | "D" "C"{0,3} | "C"{0,3}
//! tens       := "XC" | "XL" | "L" "X"{0,3} | "X"{0,3}
//! units      := "IX" | "IV" | "V" "I"{0,3} | "I"{0,3}
//! ```
//!
//! Subtractive pairs are limited to `IV`, `IX`, `XL`, `XC`, `CD`, `CM`. Forms
//! such as `IIII`, `VV`, or `IC` are non-canonical and rejected by
//! [`from_roman`]. Lower-case input is also rejected.
//!
//! # Complexity
//!
//! Both [`to_roman`] and [`from_roman`] run in `O(1)` time and space: the
//! lookup table has 13 entries and the output length is bounded by 15
//! characters (the longest standard form, `MMMDCCCLXXXVIII = 3888`).
//!
//! # Preconditions
//!
//! - [`to_roman`] accepts `n` in `1..=3999`. Other values return `None`.
//! - [`from_roman`] accepts only the uppercase canonical form.
//!
//! [`to_roman`] and [`from_roman`] roundtrip for every `n` in `1..=3999`.
//!
//! # References
//!
//! - <https://en.wikipedia.org/wiki/Roman_numerals>

/// Greedy lookup table, ordered from largest to smallest. The subtractive
/// pairs sit between their adjacent powers of ten so that greedy left-to-right
/// matching always picks the longest valid token first.
const PAIRS: &[(u32, &str)] = &[
    (1000, "M"),
    (900, "CM"),
    (500, "D"),
    (400, "CD"),
    (100, "C"),
    (90, "XC"),
    (50, "L"),
    (40, "XL"),
    (10, "X"),
    (9, "IX"),
    (5, "V"),
    (4, "IV"),
    (1, "I"),
];

/// Converts an integer to its standard Roman-numeral string.
///
/// Returns `Some(roman)` for `n` in `1..=3999` and `None` otherwise (Roman
/// numerals have no standard representation for `0` or for values `>= 4000`).
///
/// # Examples
///
/// ```
/// use rust_algorithms::string::roman_numerals::to_roman;
/// assert_eq!(to_roman(1994).as_deref(), Some("MCMXCIV"));
/// assert_eq!(to_roman(0), None);
/// assert_eq!(to_roman(4000), None);
/// ```
#[must_use]
pub fn to_roman(n: u32) -> Option<String> {
    if !(1..=3999).contains(&n) {
        return None;
    }
    let mut remaining = n;
    let mut out = String::with_capacity(15);
    for &(value, symbol) in PAIRS {
        while remaining >= value {
            out.push_str(symbol);
            remaining -= value;
        }
    }
    Some(out)
}

/// Parses a standard (canonical) uppercase Roman-numeral string.
///
/// Returns `None` for the empty string, for any input containing characters
/// outside `M D C L X V I`, for invalid subtractive pairs (e.g. `IC`), and
/// for non-canonical forms that don't roundtrip through [`to_roman`] (e.g.
/// `IIII`, `VV`).
///
/// # Examples
///
/// ```
/// use rust_algorithms::string::roman_numerals::from_roman;
/// assert_eq!(from_roman("MCMXCIV"), Some(1994));
/// assert_eq!(from_roman("IIII"), None);
/// assert_eq!(from_roman("iv"), None);
/// ```
#[must_use]
pub fn from_roman(s: &str) -> Option<u32> {
    if s.is_empty() {
        return None;
    }
    // Reject any non-uppercase-Roman character up-front.
    if !s
        .bytes()
        .all(|b| matches!(b, b'M' | b'D' | b'C' | b'L' | b'X' | b'V' | b'I'))
    {
        return None;
    }

    let mut total: u32 = 0;
    let mut rest = s;
    while !rest.is_empty() {
        let mut matched = false;
        for &(value, symbol) in PAIRS {
            if let Some(stripped) = rest.strip_prefix(symbol) {
                total = total.checked_add(value)?;
                rest = stripped;
                matched = true;
                break;
            }
        }
        if !matched {
            return None;
        }
    }

    // Canonical-form check: only accept inputs that match the greedy encoding.
    // This rejects "IIII", "VV", "IC", "CMM", etc.
    if to_roman(total).as_deref() == Some(s) {
        Some(total)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::{from_roman, to_roman};
    use quickcheck_macros::quickcheck;

    #[test]
    fn to_roman_zero_is_none() {
        assert_eq!(to_roman(0), None);
    }

    #[test]
    fn to_roman_four_thousand_is_none() {
        assert_eq!(to_roman(4000), None);
    }

    #[test]
    fn to_roman_known_values() {
        let cases = [
            (1, "I"),
            (4, "IV"),
            (9, "IX"),
            (40, "XL"),
            (49, "XLIX"),
            (90, "XC"),
            (400, "CD"),
            (900, "CM"),
            (944, "CMXLIV"),
            (1994, "MCMXCIV"),
            (3999, "MMMCMXCIX"),
        ];
        for (n, expected) in cases {
            assert_eq!(to_roman(n).as_deref(), Some(expected), "to_roman({n})");
        }
    }

    #[test]
    fn from_roman_known_values() {
        let cases = [
            ("I", 1),
            ("IV", 4),
            ("IX", 9),
            ("XL", 40),
            ("XLIX", 49),
            ("XC", 90),
            ("CD", 400),
            ("CM", 900),
            ("CMXLIV", 944),
            ("MCMXCIV", 1994),
            ("MMMCMXCIX", 3999),
        ];
        for (s, expected) in cases {
            assert_eq!(from_roman(s), Some(expected), "from_roman({s:?})");
        }
    }

    #[test]
    fn from_roman_empty_is_none() {
        assert_eq!(from_roman(""), None);
    }

    #[test]
    fn from_roman_rejects_non_canonical_iiii() {
        assert_eq!(from_roman("IIII"), None);
    }

    #[test]
    fn from_roman_rejects_repeated_five() {
        assert_eq!(from_roman("VV"), None);
    }

    #[test]
    fn from_roman_rejects_lowercase() {
        assert_eq!(from_roman("iv"), None);
    }

    #[test]
    fn from_roman_rejects_invalid_character() {
        assert_eq!(from_roman("Z"), None);
        assert_eq!(from_roman("XIIZ"), None);
    }

    #[test]
    fn from_roman_rejects_invalid_subtractive_pair() {
        // IC is not a valid subtractive pair (only IV and IX are valid for I).
        assert_eq!(from_roman("IC"), None);
        assert_eq!(from_roman("IL"), None);
        assert_eq!(from_roman("XD"), None);
        assert_eq!(from_roman("XM"), None);
    }

    #[test]
    fn from_roman_rejects_too_many_thousands() {
        // 4000+ would parse arithmetically but isn't canonical.
        assert_eq!(from_roman("CMM"), None);
        assert_eq!(from_roman("MMMM"), None);
    }

    #[quickcheck]
    fn roundtrip_to_from(n: u32) -> bool {
        // Restrict to the supported range; outside it `to_roman` is None and
        // there's nothing to roundtrip.
        let n = (n % 3999) + 1;
        let roman = to_roman(n).expect("in-range value should encode");
        from_roman(&roman) == Some(n)
    }

    #[test]
    fn exhaustive_roundtrip() {
        for n in 1..=3999_u32 {
            let roman = to_roman(n).unwrap();
            assert_eq!(from_roman(&roman), Some(n), "roundtrip failed for {n}");
        }
    }
}
