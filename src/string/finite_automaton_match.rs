//! Finite-automaton string matching.
//!
//! Builds a deterministic finite automaton (DFA) from the pattern, then scans
//! the haystack in a single linear pass. Each haystack byte triggers exactly
//! one transition, so matching is O(n) regardless of the pattern.
//!
//! The automaton has `m + 1` states (one per prefix of the pattern, including
//! the empty prefix and the full-pattern accept state). Construction uses the
//! KMP failure function to compute transitions in O(m · 256) time and the
//! same memory: `m + 1` rows × 256 columns of `usize`. This is the classic
//! "extended KMP" / Aho–Corasick-for-one-pattern trade-off — heavier table,
//! branch-free hot loop, and identical asymptotic match cost to KMP.
//!
//! Like the other byte-oriented searchers in this module, the API operates on
//! `&[u8]`. Callers searching Unicode text should pass `s.as_bytes()`; match
//! indices are byte offsets into the haystack.

const ALPHABET: usize = 256;

/// Deterministic finite automaton compiled from a fixed byte pattern.
///
/// `transitions[state][byte]` is the next state after reading `byte` while in
/// `state`. `accept_state` equals the pattern length and signals a match.
pub struct PatternAutomaton {
    transitions: Vec<[usize; ALPHABET]>,
    accept_state: usize,
}

impl PatternAutomaton {
    /// Builds the DFA for `pattern`.
    ///
    /// Construction is O(m · 256) time and memory. The automaton has
    /// `m + 1` states; state `m` is the accept state.
    ///
    /// An empty pattern yields a single-state automaton whose only state is
    /// already accepting — see [`find_first`](Self::find_first) and
    /// [`find_all`](Self::find_all) for how that case is handled.
    #[must_use]
    pub fn new(pattern: &[u8]) -> Self {
        let m = pattern.len();
        let mut transitions = vec![[0usize; ALPHABET]; m + 1];

        // Empty pattern: the single state has every byte as a self-loop and
        // is itself the accept state.
        if m == 0 {
            return Self {
                transitions,
                accept_state: 0,
            };
        }

        // Compute KMP failure function over the pattern. fail[i] is the
        // length of the longest proper prefix of pattern[..i] that is also
        // a suffix.
        let mut fail = vec![0usize; m + 1];
        let mut k = 0usize;
        for i in 1..m {
            while k > 0 && pattern[k] != pattern[i] {
                k = fail[k];
            }
            if pattern[k] == pattern[i] {
                k += 1;
            }
            fail[i + 1] = k;
        }

        // Derive the DFA from the failure function. The forward transition
        // from state s on the next pattern byte advances to s + 1; every
        // other byte falls back to the transition from fail[s].
        for s in 0..=m {
            for c in 0..ALPHABET {
                if s < m && pattern[s] as usize == c {
                    transitions[s][c] = s + 1;
                } else if s == 0 {
                    transitions[s][c] = 0;
                } else {
                    transitions[s][c] = transitions[fail[s]][c];
                }
            }
        }

        Self {
            transitions,
            accept_state: m,
        }
    }

    /// Returns the byte offset of the first occurrence of the pattern in
    /// `haystack`, or `None` if it does not occur.
    ///
    /// An empty pattern matches at index `0` (consistent with `str::find`).
    /// Runs in O(n) time and O(1) extra space beyond the automaton itself.
    #[must_use]
    pub fn find_first(&self, haystack: &[u8]) -> Option<usize> {
        if self.accept_state == 0 {
            return Some(0);
        }
        let mut state = 0usize;
        for (i, &b) in haystack.iter().enumerate() {
            state = self.transitions[state][b as usize];
            if state == self.accept_state {
                return Some(i + 1 - self.accept_state);
            }
        }
        None
    }

    /// Returns the byte offsets of every occurrence of the pattern in
    /// `haystack`, in increasing order.
    ///
    /// Matches may overlap: `find_all` of `b"aa"` in `b"aaaa"` returns
    /// `[0, 1, 2]`. An empty pattern returns an empty vector to avoid an
    /// infinite stream of zero-width matches (matching the BMH-family
    /// convention used elsewhere in this module).
    #[must_use]
    pub fn find_all(&self, haystack: &[u8]) -> Vec<usize> {
        let mut matches = Vec::new();
        if self.accept_state == 0 {
            return matches;
        }
        let mut state = 0usize;
        for (i, &b) in haystack.iter().enumerate() {
            state = self.transitions[state][b as usize];
            if state == self.accept_state {
                matches.push(i + 1 - self.accept_state);
            }
        }
        matches
    }
}

#[cfg(test)]
mod tests {
    use super::PatternAutomaton;

    #[test]
    fn empty_pattern_matches_at_zero() {
        let dfa = PatternAutomaton::new(b"");
        assert_eq!(dfa.find_first(b"abc"), Some(0));
        assert_eq!(dfa.find_first(b""), Some(0));
    }

    #[test]
    fn empty_pattern_find_all_is_empty() {
        let dfa = PatternAutomaton::new(b"");
        assert_eq!(dfa.find_all(b"abc"), Vec::<usize>::new());
    }

    #[test]
    fn empty_haystack_nonempty_pattern() {
        let dfa = PatternAutomaton::new(b"a");
        assert_eq!(dfa.find_first(b""), None);
        assert_eq!(dfa.find_all(b""), Vec::<usize>::new());
    }

    #[test]
    fn no_match() {
        let dfa = PatternAutomaton::new(b"xyz");
        assert_eq!(dfa.find_first(b"abcdef"), None);
        assert_eq!(dfa.find_all(b"abcdef"), Vec::<usize>::new());
    }

    #[test]
    fn match_at_start() {
        let dfa = PatternAutomaton::new(b"hello");
        assert_eq!(dfa.find_first(b"hello world"), Some(0));
        assert_eq!(dfa.find_all(b"hello world"), vec![0]);
    }

    #[test]
    fn match_in_middle() {
        let dfa = PatternAutomaton::new(b"lo wo");
        assert_eq!(dfa.find_first(b"hello world"), Some(3));
        assert_eq!(dfa.find_all(b"hello world"), vec![3]);
    }

    #[test]
    fn match_at_end() {
        let dfa = PatternAutomaton::new(b"world");
        assert_eq!(dfa.find_first(b"hello world"), Some(6));
        assert_eq!(dfa.find_all(b"hello world"), vec![6]);
    }

    #[test]
    fn multiple_matches() {
        let dfa = PatternAutomaton::new(b"ab");
        assert_eq!(dfa.find_all(b"ababab"), vec![0, 2, 4]);
    }

    #[test]
    fn overlapping_matches() {
        // DFA matching reports every occurrence, including overlaps.
        let dfa = PatternAutomaton::new(b"aa");
        assert_eq!(dfa.find_all(b"aaaa"), vec![0, 1, 2]);
    }

    #[test]
    fn first_match_returned_only() {
        let dfa = PatternAutomaton::new(b"ab");
        assert_eq!(dfa.find_first(b"ababab"), Some(0));
    }

    #[test]
    fn pattern_longer_than_haystack() {
        let dfa = PatternAutomaton::new(b"abc");
        assert_eq!(dfa.find_first(b"ab"), None);
        assert_eq!(dfa.find_all(b"ab"), Vec::<usize>::new());
    }

    #[test]
    fn pattern_equals_haystack() {
        let dfa = PatternAutomaton::new(b"abc");
        assert_eq!(dfa.find_first(b"abc"), Some(0));
        assert_eq!(dfa.find_all(b"abc"), vec![0]);
    }

    #[test]
    fn classic_kmp_example() {
        // The pattern's own failure structure means the DFA must rewind
        // correctly after the false start at index 10.
        let text = b"ABABDABACDABABCABAB";
        let pat = b"ABABCABAB";
        let dfa = PatternAutomaton::new(pat);
        assert_eq!(dfa.find_first(text), Some(10));
        assert_eq!(dfa.find_all(text), vec![10]);
    }

    #[test]
    fn non_ascii_bytes() {
        // High-bit bytes including a multi-byte UTF-8 sequence for "é".
        let haystack: &[u8] = &[0x00, 0xC3, 0xA9, 0xFF, 0xC3, 0xA9, 0x10];
        let pat: &[u8] = &[0xC3, 0xA9];
        let dfa = PatternAutomaton::new(pat);
        assert_eq!(dfa.find_first(haystack), Some(1));
        assert_eq!(dfa.find_all(haystack), vec![1, 4]);
    }

    #[test]
    fn unicode_via_as_bytes() {
        let text = "café au lait, café noir";
        let dfa = PatternAutomaton::new("café".as_bytes());
        assert_eq!(dfa.find_all(text.as_bytes()), vec![0, 15]);
    }

    /// Naive reference: report every occurrence, including overlaps.
    fn naive_search_all(haystack: &[u8], pattern: &[u8]) -> Vec<usize> {
        let mut out = Vec::new();
        if pattern.is_empty() || pattern.len() > haystack.len() {
            return out;
        }
        for i in 0..=haystack.len() - pattern.len() {
            if &haystack[i..i + pattern.len()] == pattern {
                out.push(i);
            }
        }
        out
    }

    fn naive_search_first(haystack: &[u8], pattern: &[u8]) -> Option<usize> {
        if pattern.is_empty() {
            return Some(0);
        }
        if pattern.len() > haystack.len() {
            return None;
        }
        (0..=haystack.len() - pattern.len()).find(|&i| &haystack[i..i + pattern.len()] == pattern)
    }

    #[quickcheck_macros::quickcheck]
    #[allow(clippy::needless_pass_by_value)]
    fn matches_naive_first(haystack: Vec<u8>, pattern: Vec<u8>) -> bool {
        let haystack: &[u8] = &haystack[..haystack.len().min(50)];
        let pattern: &[u8] = &pattern[..pattern.len().min(50)];
        let dfa = PatternAutomaton::new(pattern);
        dfa.find_first(haystack) == naive_search_first(haystack, pattern)
    }

    #[quickcheck_macros::quickcheck]
    #[allow(clippy::needless_pass_by_value)]
    fn matches_naive_all(haystack: Vec<u8>, pattern: Vec<u8>) -> bool {
        let haystack: &[u8] = &haystack[..haystack.len().min(50)];
        let pattern: &[u8] = &pattern[..pattern.len().min(50)];
        let dfa = PatternAutomaton::new(pattern);
        dfa.find_all(haystack) == naive_search_all(haystack, pattern)
    }
}
