//! Telephone keypad letter combinations (`LeetCode` 17).
//!
//! Given a string containing digits from `'2'` through `'9'`, return all
//! possible letter combinations the digits could represent on a classic
//! phone keypad. Mapping:
//!
//! | digit | letters |
//! |------:|:--------|
//! | 2     | abc     |
//! | 3     | def     |
//! | 4     | ghi     |
//! | 5     | jkl     |
//! | 6     | mno     |
//! | 7     | pqrs    |
//! | 8     | tuv     |
//! | 9     | wxyz    |
//!
//! # Algorithm
//! Backtracking. Walk the digits left to right; at each position append
//! every letter mapped to that digit, recurse, then pop. The traversal
//! visits children in left-to-right alphabetical order, so the output is
//! ordered lexicographically by digit position (the alphabetical letters
//! at digit `0` form the outer loop, then digit `1`, etc.).
//!
//! # Empty input
//! `letter_combinations("")` returns `vec![]` (empty vector). This matches
//! `LeetCode` 17's expected behaviour, where an empty digit string yields
//! no combinations rather than the single empty string.
//!
//! # Complexity
//! Let `n = digits.len()` and let `k_i` be the number of letters mapped to
//! digit `i` (`3` or `4`). Let `P = ∏ k_i` be the number of outputs.
//! - **Time**: `O(n · P)` — `P` outputs, each materialised as a `String`
//!   of length `n`.
//! - **Space**: `O(n)` auxiliary (recursion + scratch buffer), excluding
//!   the `O(n · P)` output.

/// Returns all letter combinations for a phone-keypad digit string.
///
/// `digits` must contain only ASCII digits in `'2'..='9'`. Any other
/// character (including `'0'`, `'1'`, or non-digits) is silently skipped:
/// it contributes no letters to the Cartesian product, so the function
/// returns `vec![]` if every character is invalid and `digits` is
/// non-empty.
///
/// # Output order
/// Lexicographic by digit position: the letters at digit `0` vary slowest
/// (outermost), the letters at the last digit vary fastest (innermost).
/// For example, `"23"` returns
/// `["ad","ae","af","bd","be","bf","cd","ce","cf"]`.
///
/// # Empty input
/// Returns `vec![]` (empty vector), matching `LeetCode` 17.
pub fn letter_combinations(digits: &str) -> Vec<String> {
    if digits.is_empty() {
        return Vec::new();
    }

    // Collect the letter set per input digit. Invalid digits map to an
    // empty slice, which would zero out the Cartesian product.
    let mapped: Vec<&'static [u8]> = digits.bytes().map(letters_for).collect();
    if mapped.iter().any(|s| s.is_empty()) {
        return Vec::new();
    }

    let total: usize = mapped.iter().map(|s| s.len()).product();
    let mut out: Vec<String> = Vec::with_capacity(total);
    let mut buf: Vec<u8> = Vec::with_capacity(mapped.len());
    backtrack(&mapped, 0, &mut buf, &mut out);
    out
}

/// Returns the letters mapped to a phone-keypad digit byte, or an empty
/// slice if the byte is not a digit in `'2'..='9'`.
const fn letters_for(d: u8) -> &'static [u8] {
    match d {
        b'2' => b"abc",
        b'3' => b"def",
        b'4' => b"ghi",
        b'5' => b"jkl",
        b'6' => b"mno",
        b'7' => b"pqrs",
        b'8' => b"tuv",
        b'9' => b"wxyz",
        _ => b"",
    }
}

/// Recursive helper. At depth `i`, append each letter mapped to digit `i`
/// to `buf`, recurse on `i + 1`, then pop. When `i == mapped.len()`, the
/// current `buf` state is materialised as a `String` into `out`.
fn backtrack(mapped: &[&'static [u8]], i: usize, buf: &mut Vec<u8>, out: &mut Vec<String>) {
    if i == mapped.len() {
        // All bytes are ASCII letters by construction, so this is valid UTF-8.
        out.push(String::from_utf8(buf.clone()).expect("ASCII letters only"));
        return;
    }
    for &b in mapped[i] {
        buf.push(b);
        backtrack(mapped, i + 1, buf, out);
        buf.pop();
    }
}

#[cfg(test)]
mod tests {
    use super::letter_combinations;
    use std::collections::HashSet;

    #[test]
    fn empty_input_returns_empty_vec() {
        let out = letter_combinations("");
        assert!(out.is_empty());
    }

    #[test]
    fn single_digit_two_returns_abc() {
        let out = letter_combinations("2");
        assert_eq!(out, vec!["a", "b", "c"]);
    }

    #[test]
    fn two_three_returns_lexicographic_by_digit_position() {
        let out = letter_combinations("23");
        assert_eq!(
            out,
            vec!["ad", "ae", "af", "bd", "be", "bf", "cd", "ce", "cf"]
        );
    }

    #[test]
    fn single_digit_seven_has_four_letters() {
        let out = letter_combinations("7");
        assert_eq!(out, vec!["p", "q", "r", "s"]);
    }

    #[test]
    fn seven_nine_has_sixteen_unique_combinations() {
        let out = letter_combinations("79");
        assert_eq!(out.len(), 16);
        // Every combination must be exactly 2 letters: one of pqrs then
        // one of wxyz.
        let pqrs: HashSet<u8> = b"pqrs".iter().copied().collect();
        let wxyz: HashSet<u8> = b"wxyz".iter().copied().collect();
        for combo in &out {
            let bytes = combo.as_bytes();
            assert_eq!(bytes.len(), 2);
            assert!(pqrs.contains(&bytes[0]));
            assert!(wxyz.contains(&bytes[1]));
        }
        // All 16 outputs are pairwise distinct.
        let unique: HashSet<&String> = out.iter().collect();
        assert_eq!(unique.len(), 16);
    }

    #[test]
    fn seven_nine_first_and_last_match_lex_order() {
        let out = letter_combinations("79");
        // Outer digit '7' (pqrs) varies slowest; inner digit '9' (wxyz)
        // varies fastest.
        assert_eq!(out.first().unwrap(), "pw");
        assert_eq!(out.last().unwrap(), "sz");
    }

    #[test]
    fn invalid_digit_zero_returns_empty() {
        // '0' is not a valid keypad digit and zeros out the product.
        let out = letter_combinations("20");
        assert!(out.is_empty());
    }

    #[test]
    fn invalid_digit_one_returns_empty() {
        // '1' is not a valid keypad digit either.
        let out = letter_combinations("1");
        assert!(out.is_empty());
    }

    #[test]
    fn three_digit_count_is_product_of_letters_per_digit() {
        // 2 -> abc (3), 3 -> def (3), 4 -> ghi (3) => 27.
        let out = letter_combinations("234");
        assert_eq!(out.len(), 27);
        let unique: HashSet<&String> = out.iter().collect();
        assert_eq!(unique.len(), 27);
        for combo in &out {
            assert_eq!(combo.len(), 3);
        }
        // First and last by lex-by-digit-position.
        assert_eq!(out.first().unwrap(), "adg");
        assert_eq!(out.last().unwrap(), "cfi");
    }

    #[test]
    fn includes_seven_and_nine_quad_letter_digits() {
        // 7 -> pqrs (4), 8 -> tuv (3), 9 -> wxyz (4) => 48.
        let out = letter_combinations("789");
        assert_eq!(out.len(), 4 * 3 * 4);
        let unique: HashSet<&String> = out.iter().collect();
        assert_eq!(unique.len(), 48);
    }
}
