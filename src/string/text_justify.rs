//! Full text justification (greedy line packing).
//!
//! Given a list of words and a target line width `max_width`, pack the words
//! into lines greedily and then justify each line so that it has exactly
//! `max_width` characters. This is the classic `LeetCode` 68 problem.
//!
//! # Algorithm
//!
//! 1. **Pack** words greedily: keep adding the next word to the current line
//!    while it still fits. The line fits if the sum of word lengths plus one
//!    mandatory space between every adjacent pair is `<= max_width`.
//! 2. **Justify** each completed line by distributing the leftover spaces
//!    `extra = max_width - sum_of_word_lengths` evenly into the gaps between
//!    words. If `extra` does not divide evenly into the number of gaps, the
//!    leftmost gaps each receive one additional space.
//! 3. **Last line** and any line containing a single word are left-justified:
//!    words separated by a single space, with the remaining width padded by
//!    trailing spaces.
//!
//! # Complexity
//!
//! Let `n` be the number of words and `W = max_width`. Packing visits each
//! word once, so it runs in `O(n)`. Building each output line writes at most
//! `W` characters, so emitting all lines costs `O(n + total_output_chars)`,
//! i.e. linear in the size of the produced output.
//!
//! # Panics
//!
//! Panics if any word is longer than `max_width`, since such a word can
//! never be placed on any line.
//!
//! # Examples
//!
//! ```
//! use rust_algorithms::string::text_justify::full_justify;
//!
//! let words: Vec<String> = ["This", "is", "an", "example", "of", "text", "justification."]
//!     .iter()
//!     .map(|s| s.to_string())
//!     .collect();
//! let out = full_justify(&words, 16);
//! assert_eq!(
//!     out,
//!     vec![
//!         "This    is    an".to_string(),
//!         "example  of text".to_string(),
//!         "justification.  ".to_string(),
//!     ]
//! );
//! ```

/// Greedy full-justification of `words` into lines of width `max_width`.
///
/// Returns a `Vec<String>`, one entry per output line, each of length exactly
/// `max_width`. The last line (and any single-word line) is left-justified
/// and padded with trailing spaces; every other line distributes extra
/// spaces evenly between words, with leftover space pushed into the leftmost
/// gaps.
///
/// # Panics
///
/// Panics if any word's length exceeds `max_width`.
pub fn full_justify(words: &[String], max_width: usize) -> Vec<String> {
    if words.is_empty() {
        return Vec::new();
    }

    // Sanity: every word must fit on a line by itself.
    for w in words {
        assert!(
            w.len() <= max_width,
            "word longer than max_width cannot be justified"
        );
    }

    let mut out: Vec<String> = Vec::new();
    let mut i = 0;
    let n = words.len();

    while i < n {
        // Greedy pack: find the largest j such that words[i..j] fit on one
        // line. `len` tracks `sum(word.len() for word in line) + (count - 1)`
        // i.e. the minimum width needed if every gap is a single space.
        let mut j = i;
        let mut len = 0usize;
        while j < n {
            // Adding words[j] needs words[j].len() chars, plus a single
            // separating space if it is not the first word on the line.
            let add = words[j].len() + usize::from(j != i);
            if len + add > max_width {
                break;
            }
            len += add;
            j += 1;
        }

        let line_words = &words[i..j];
        let is_last_line = j == n;
        out.push(build_line(line_words, max_width, is_last_line));
        i = j;
    }

    out
}

/// Build a single justified line from `line_words`.
///
/// If `is_last_line` is true, or there is only one word, the line is
/// left-justified: words are joined by single spaces and the remainder is
/// padded with trailing spaces. Otherwise, the slack
/// `max_width - sum_of_word_lengths` is distributed across the
/// `line_words.len() - 1` gaps, with the leftmost `slack % gaps` gaps
/// receiving one extra space.
fn build_line(line_words: &[String], max_width: usize, is_last_line: bool) -> String {
    let mut line = String::with_capacity(max_width);
    let count = line_words.len();
    debug_assert!(count >= 1, "packing should never produce an empty line");

    if is_last_line || count == 1 {
        // Left-justify: single spaces between words, pad the rest with
        // trailing spaces.
        for (k, w) in line_words.iter().enumerate() {
            if k > 0 {
                line.push(' ');
            }
            line.push_str(w);
        }
        while line.len() < max_width {
            line.push(' ');
        }
        return line;
    }

    let total_word_len: usize = line_words.iter().map(String::len).sum();
    let total_spaces = max_width - total_word_len;
    let gaps = count - 1;
    let base = total_spaces / gaps;
    let extra = total_spaces % gaps;

    for (k, w) in line_words.iter().enumerate() {
        line.push_str(w);
        if k < gaps {
            // Leftmost `extra` gaps receive one additional space.
            let spaces = base + usize::from(k < extra);
            for _ in 0..spaces {
                line.push(' ');
            }
        }
    }

    debug_assert_eq!(line.len(), max_width);
    line
}

#[cfg(test)]
mod tests {
    use super::full_justify;

    fn s(strs: &[&str]) -> Vec<String> {
        strs.iter().map(ToString::to_string).collect()
    }

    #[test]
    fn empty_input_yields_empty_output() {
        let words: Vec<String> = Vec::new();
        assert!(full_justify(&words, 10).is_empty());
    }

    #[test]
    fn leetcode_example_width_16() {
        let words = s(&[
            "This",
            "is",
            "an",
            "example",
            "of",
            "text",
            "justification.",
        ]);
        let got = full_justify(&words, 16);
        let want = vec![
            "This    is    an".to_string(),
            "example  of text".to_string(),
            "justification.  ".to_string(),
        ];
        assert_eq!(got, want);
    }

    #[test]
    fn leftmost_gaps_get_extra_space() {
        // "What must be acknowledgment shall be" / max_width = 16
        // expected: "What   must   be" then "acknowledgment  " then "shall be        "
        let words = s(&["What", "must", "be", "acknowledgment", "shall", "be"]);
        let got = full_justify(&words, 16);
        let want = vec![
            "What   must   be".to_string(),
            "acknowledgment  ".to_string(),
            "shall be        ".to_string(),
        ];
        assert_eq!(got, want);
    }

    #[test]
    fn every_line_has_exact_width() {
        let words = s(&[
            "Science",
            "is",
            "what",
            "we",
            "understand",
            "well",
            "enough",
            "to",
            "explain",
            "to",
            "a",
            "computer.",
            "Art",
            "is",
            "everything",
            "else",
            "we",
            "do.",
        ]);
        let max_width = 20;
        let got = full_justify(&words, max_width);
        for line in &got {
            assert_eq!(line.len(), max_width, "line {line:?} has wrong width");
        }
    }

    #[test]
    fn single_word_per_line_when_width_tight() {
        // Each word is too long to share a line at width 7, so each lands on
        // its own line and is left-justified with trailing spaces.
        let words = s(&["abcdef", "ghijkl", "mnopqr"]);
        let got = full_justify(&words, 7);
        assert_eq!(
            got,
            vec![
                "abcdef ".to_string(),
                "ghijkl ".to_string(),
                "mnopqr ".to_string(),
            ]
        );
    }

    #[test]
    fn single_word_line_inside_paragraph_is_left_justified() {
        // The middle line "acknowledgment" has only one word and must be
        // left-justified with trailing pad even though it is not the last
        // line of output.
        let words = s(&["What", "must", "be", "acknowledgment", "shall", "be"]);
        let got = full_justify(&words, 16);
        assert_eq!(got[1], "acknowledgment  ");
    }

    #[test]
    fn last_line_is_left_justified_with_trailing_pad() {
        let words = s(&[
            "This",
            "is",
            "an",
            "example",
            "of",
            "text",
            "justification.",
        ]);
        let got = full_justify(&words, 16);
        let last = got.last().unwrap();
        assert_eq!(last, "justification.  ");
        assert_eq!(last.len(), 16);
    }

    #[test]
    fn single_word_input() {
        let words = s(&["hello"]);
        let got = full_justify(&words, 10);
        assert_eq!(got, vec!["hello     ".to_string()]);
    }

    #[test]
    fn word_exactly_fills_line() {
        let words = s(&["abcdef"]);
        let got = full_justify(&words, 6);
        assert_eq!(got, vec!["abcdef".to_string()]);
    }

    #[test]
    #[should_panic(expected = "word longer than max_width")]
    fn panics_when_word_exceeds_max_width() {
        let words = s(&["short", "waytoolongword"]);
        let _ = full_justify(&words, 6);
    }
}
