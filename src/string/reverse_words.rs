//! Reverse the order of words in an ASCII byte sentence, in place.
//!
//! Given a mutable byte buffer representing a sentence whose words are
//! separated by a single ASCII space (`b' '`), this routine reverses the
//! order of the words without allocating any auxiliary buffer.
//!
//! # Algorithm
//!
//! 1. Reverse the entire buffer at the byte level.
//! 2. Walk the reversed buffer and reverse each maximal run of non-space
//!    bytes (a word) in place.
//!
//! After step 1 the words are in the desired order but each word is itself
//! reversed; step 2 restores the original spelling of every word.
//!
//! # Complexity
//!
//! Each byte is touched a constant number of times, so the overall running
//! time is `O(n)` where `n = s.len()`. No heap allocation is performed.
//!
//! # Input assumptions
//!
//! The input is assumed to be a single line of ASCII text whose words are
//! separated by exactly one space, with no leading or trailing whitespace
//! and no runs of consecutive spaces. The function does not normalise
//! whitespace; if the caller violates these assumptions the result will
//! still be a valid in-place reversal of "tokens between spaces", but
//! empty tokens produced by adjacent or edge spaces will be preserved.
//!
//! # Examples
//!
//! ```
//! use rust_algorithms::string::reverse_words::reverse_words_in_place;
//!
//! let mut buf = b"the sky is blue".to_vec();
//! reverse_words_in_place(&mut buf);
//! assert_eq!(&buf, b"blue is sky the");
//! ```
//!
//! # See also
//!
//! - [`crate::string::booths_least_rotation`] for cyclic rotation of a
//!   string viewed as a sequence of bytes.

/// Reverse the order of space-separated words in `s` in place.
///
/// Operates on raw ASCII bytes: the buffer is mutated directly without
/// any allocation. Words are runs of bytes that are not the ASCII space
/// `b' '`; a single space is treated as the only delimiter.
///
/// # Examples
///
/// ```
/// use rust_algorithms::string::reverse_words::reverse_words_in_place;
///
/// let mut buf = b"hello".to_vec();
/// reverse_words_in_place(&mut buf);
/// assert_eq!(&buf, b"hello");
///
/// let mut buf = b"the sky is blue".to_vec();
/// reverse_words_in_place(&mut buf);
/// assert_eq!(&buf, b"blue is sky the");
/// ```
pub fn reverse_words_in_place(s: &mut [u8]) {
    // Step 1: reverse the whole buffer.
    s.reverse();

    // Step 2: reverse each maximal run of non-space bytes.
    let n = s.len();
    let mut i = 0;
    while i < n {
        // Skip any run of spaces (under the documented assumptions there
        // is at most one, but handling >=1 keeps this routine robust).
        while i < n && s[i] == b' ' {
            i += 1;
        }
        let start = i;
        while i < n && s[i] != b' ' {
            i += 1;
        }
        if start < i {
            s[start..i].reverse();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::reverse_words_in_place;

    fn run(input: &[u8]) -> Vec<u8> {
        let mut buf = input.to_vec();
        reverse_words_in_place(&mut buf);
        buf
    }

    #[test]
    fn empty_buffer() {
        assert_eq!(run(b""), b"");
    }

    #[test]
    fn single_word_unchanged() {
        assert_eq!(run(b"hello"), b"hello");
    }

    #[test]
    fn single_character() {
        assert_eq!(run(b"a"), b"a");
    }

    #[test]
    fn two_words() {
        assert_eq!(run(b"hello world"), b"world hello");
    }

    #[test]
    fn classic_example() {
        assert_eq!(run(b"the sky is blue"), b"blue is sky the");
    }

    #[test]
    fn reversing_twice_is_identity() {
        let original: &[u8] = b"reverse the words in this sentence";
        let mut buf = original.to_vec();
        reverse_words_in_place(&mut buf);
        reverse_words_in_place(&mut buf);
        assert_eq!(buf, original);
    }

    #[test]
    fn words_of_varying_length() {
        assert_eq!(run(b"a bb ccc dddd"), b"dddd ccc bb a");
    }

    #[test]
    fn palindromic_word_set() {
        // Each word is itself a palindrome, so only word order changes.
        assert_eq!(run(b"aba cdc efe"), b"efe cdc aba");
    }

    #[test]
    fn buffer_is_mutated_in_place() {
        // Confirm that the function works on an existing &mut [u8] without
        // requiring a Vec, matching the documented signature contract.
        let mut buf: [u8; 11] = *b"foo bar baz";
        reverse_words_in_place(&mut buf);
        assert_eq!(&buf, b"baz bar foo");
    }
}
