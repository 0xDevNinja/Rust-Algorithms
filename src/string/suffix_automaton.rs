//! Suffix automaton over the byte alphabet.
//!
//! A suffix automaton (SAM) for a string `s` is the smallest deterministic
//! finite automaton that accepts exactly the set of substrings of `s`. Each
//! state corresponds to an *equivalence class* of right-extensions: two
//! substrings collapse into the same state when they occur at the same set of
//! end positions in `s`. The construction implemented here is Blumer's online
//! algorithm — append one character at a time, optionally cloning a state when
//! a transition's target has a strictly larger `length` than the new state's
//! candidate parent.
//!
//! # Size and complexity
//! - States: at most `2n - 1` for `n >= 2` (and exactly 1 for `n = 0`).
//! - Transitions: at most `3n - 4` for `n >= 3`.
//! - Construction: `O(n)` amortized over a fixed alphabet, `O(n log σ)` with
//!   `BTreeMap` transitions where `σ` is the alphabet size.
//! - Substring containment query: `O(m log σ)` for a pattern of length `m`.
//!
//! # Applications
//! - Substring containment in `O(m)` per query after `O(n)` preprocessing.
//! - Counting the number of distinct substrings of `s` as
//!   `Σ (len(v) − len(link(v)))` over all non-initial states `v`.
//! - Longest common substring of two strings, building blocks for streaming
//!   pattern matching, and indexed full-text search.
//!
//! Operates on `&[u8]` for predictable byte-wise semantics on arbitrary
//! inputs (ASCII, raw bytes, pre-encoded UTF-8). Transitions are stored in a
//! `BTreeMap<u8, usize>` so iteration order — and therefore the state numbering
//! produced by `new` — is deterministic across runs and platforms.

use std::collections::BTreeMap;

/// A single state of the suffix automaton.
///
/// `length` is the longest substring landing in this state. `link` is the
/// suffix link (the state representing the longest proper suffix of every
/// string ending here that lives in a different equivalence class). The
/// initial state is the only one with `link == None`.
#[derive(Debug, Clone)]
pub struct State {
    /// Length of the longest substring whose end-positions equal this state's
    /// equivalence class.
    pub length: usize,
    /// Suffix link, or `None` for the initial state.
    pub link: Option<usize>,
    /// Outgoing transitions keyed by the next byte.
    pub transitions: BTreeMap<u8, usize>,
}

impl State {
    const fn new(length: usize, link: Option<usize>) -> Self {
        Self {
            length,
            link,
            transitions: BTreeMap::new(),
        }
    }
}

/// A suffix automaton over `&[u8]` built with Blumer's online construction.
///
/// State `0` is always the initial state. After [`SuffixAutomaton::new`] the
/// automaton recognizes exactly the set of substrings of the input.
#[derive(Debug, Clone)]
pub struct SuffixAutomaton {
    /// All states; index `0` is the initial state.
    states: Vec<State>,
    /// Index of the state representing the full input (the most recently
    /// extended state). Suffixes of `s` are exactly the states reachable from
    /// `last` via suffix links.
    last: usize,
}

impl SuffixAutomaton {
    /// Builds the suffix automaton for `s` using Blumer's online algorithm.
    ///
    /// Runs in `O(n log σ)` time and `O(n)` space, where `σ` is the alphabet
    /// size (256 for `u8`). For empty input the result contains only the
    /// initial state.
    #[must_use]
    pub fn new(s: &[u8]) -> Self {
        let mut sa = Self {
            states: vec![State::new(0, None)],
            last: 0,
        };
        for &c in s {
            sa.extend(c);
        }
        sa
    }

    /// Appends one byte to the automaton, maintaining the invariants of the
    /// suffix-automaton construction (Blumer et al., 1985).
    fn extend(&mut self, c: u8) {
        let cur = self.states.len();
        self.states
            .push(State::new(self.states[self.last].length + 1, Some(0)));

        // Walk the suffix-link chain from `last`, adding the new transition
        // wherever it is missing.
        let mut p = Some(self.last);
        while let Some(pp) = p {
            if self.states[pp].transitions.contains_key(&c) {
                break;
            }
            self.states[pp].transitions.insert(c, cur);
            p = self.states[pp].link;
        }

        if let Some(pp) = p {
            let q = self.states[pp].transitions[&c];
            if self.states[pp].length + 1 == self.states[q].length {
                // `q` is already a continuous extension of `p` — reuse it.
                self.states[cur].link = Some(q);
            } else {
                // Otherwise clone `q` into `clone`, redirect ancestors that
                // pointed at `q` via `c`, and point both `cur` and `q` at the
                // clone via suffix links.
                let clone = self.states.len();
                let mut cloned = self.states[q].clone();
                cloned.length = self.states[pp].length + 1;
                self.states.push(cloned);

                let mut walker = Some(pp);
                while let Some(w) = walker {
                    if self.states[w].transitions.get(&c) == Some(&q) {
                        self.states[w].transitions.insert(c, clone);
                        walker = self.states[w].link;
                    } else {
                        break;
                    }
                }
                self.states[q].link = Some(clone);
                self.states[cur].link = Some(clone);
            }
        }
        // If `p` is `None`, the suffix link of `cur` stays at the initial
        // state (set above when the state was created).
        self.last = cur;
    }

    /// Returns `true` if `pattern` occurs as a (contiguous) substring of the
    /// original input. The empty pattern is always a substring.
    ///
    /// Walks the deterministic transition function from the initial state in
    /// `O(m log σ)` time for a pattern of length `m`.
    #[must_use]
    pub fn contains(&self, pattern: &[u8]) -> bool {
        let mut node = 0_usize;
        for &c in pattern {
            match self.states[node].transitions.get(&c) {
                Some(&next) => node = next,
                None => return false,
            }
        }
        true
    }

    /// Returns the number of distinct *non-empty* substrings of the original
    /// input.
    ///
    /// Computed as `Σ (len(v) − len(link(v)))` over every non-initial state
    /// `v` — each state contributes the count of substrings that land in its
    /// equivalence class.
    #[must_use]
    pub fn distinct_substrings_count(&self) -> u64 {
        let mut total: u64 = 0;
        for (i, st) in self.states.iter().enumerate() {
            if i == 0 {
                continue;
            }
            let parent_len = st.link.map_or(0, |l| self.states[l].length);
            total += (st.length - parent_len) as u64;
        }
        total
    }

    /// Number of states in the automaton (always `>= 1`; the initial state is
    /// always present).
    #[must_use]
    pub const fn num_states(&self) -> usize {
        self.states.len()
    }
}

#[cfg(test)]
mod tests {
    use super::SuffixAutomaton;
    use std::collections::HashSet;

    /// Brute-force reference: collect every distinct non-empty substring.
    fn brute_force_substrings(s: &[u8]) -> HashSet<Vec<u8>> {
        let mut set = HashSet::new();
        for i in 0..s.len() {
            for j in (i + 1)..=s.len() {
                set.insert(s[i..j].to_vec());
            }
        }
        set
    }

    #[test]
    fn empty_input_has_only_initial_state() {
        let sa = SuffixAutomaton::new(b"");
        assert_eq!(sa.num_states(), 1);
        assert!(sa.states[0].transitions.is_empty());
        assert!(sa.contains(b""));
        assert!(!sa.contains(b"a"));
        assert_eq!(sa.distinct_substrings_count(), 0);
    }

    #[test]
    fn single_char() {
        let sa = SuffixAutomaton::new(b"a");
        assert!(sa.contains(b""));
        assert!(sa.contains(b"a"));
        assert!(!sa.contains(b"b"));
        assert!(!sa.contains(b"aa"));
        assert_eq!(sa.distinct_substrings_count(), 1);
    }

    #[test]
    fn abcbc_distinct_count() {
        // Substrings of "abcbc": a, b, c, ab, bc, cb, abc, bcb, cbc,
        // abcb, bcbc, abcbc — 12 distinct non-empty substrings.
        let sa = SuffixAutomaton::new(b"abcbc");
        assert_eq!(sa.distinct_substrings_count(), 12);
    }

    #[test]
    fn banana_contains_all_substrings() {
        let s = b"banana";
        let sa = SuffixAutomaton::new(s);
        for sub in brute_force_substrings(s) {
            assert!(sa.contains(&sub), "missing substring {sub:?}");
        }
        assert!(sa.contains(b""));
        assert!(!sa.contains(b"x"));
        assert!(!sa.contains(b"banand"));
        assert_eq!(
            sa.distinct_substrings_count() as usize,
            brute_force_substrings(s).len()
        );
    }

    #[test]
    fn aaaa_distinct_count_is_n() {
        // For "aaaa" the distinct non-empty substrings are a, aa, aaa, aaaa.
        let sa = SuffixAutomaton::new(b"aaaa");
        assert_eq!(sa.distinct_substrings_count(), 4);
        assert!(sa.contains(b"aaaa"));
        assert!(!sa.contains(b"aaaaa"));
    }

    #[test]
    fn mississippi_matches_brute_force() {
        let s = b"mississippi";
        let sa = SuffixAutomaton::new(s);
        let bf = brute_force_substrings(s);
        assert_eq!(sa.distinct_substrings_count() as usize, bf.len());
        for sub in &bf {
            assert!(sa.contains(sub));
        }
    }

    #[test]
    fn state_count_bound() {
        // For n >= 2 the number of states is at most 2n - 1.
        for s in [b"abcbc".as_slice(), b"banana", b"mississippi", b"abcdef"] {
            let sa = SuffixAutomaton::new(s);
            assert!(
                sa.num_states() < 2 * s.len(),
                "state count {} exceeds 2n-1 for {s:?}",
                sa.num_states()
            );
        }
    }

    #[test]
    fn rejects_non_substrings() {
        let sa = SuffixAutomaton::new(b"hello world");
        assert!(sa.contains(b"hello"));
        assert!(sa.contains(b"o w"));
        assert!(sa.contains(b"world"));
        assert!(!sa.contains(b"helloo"));
        assert!(!sa.contains(b"word"));
        assert!(!sa.contains(b"xyz"));
    }

    #[allow(clippy::needless_pass_by_value)]
    #[quickcheck_macros::quickcheck]
    fn quickcheck_contains_matches_brute_force(s: Vec<u8>, p: Vec<u8>) -> bool {
        let s: Vec<u8> = s.into_iter().take(20).collect();
        let p: Vec<u8> = p.into_iter().take(5).collect();
        let sa = SuffixAutomaton::new(&s);
        let brute = if p.is_empty() {
            true
        } else if p.len() > s.len() {
            false
        } else {
            s.windows(p.len()).any(|w| w == p.as_slice())
        };
        sa.contains(&p) == brute
    }

    #[allow(clippy::needless_pass_by_value)]
    #[quickcheck_macros::quickcheck]
    fn quickcheck_distinct_count_matches_brute_force(s: Vec<u8>) -> bool {
        let s: Vec<u8> = s.into_iter().take(20).collect();
        let sa = SuffixAutomaton::new(&s);
        sa.distinct_substrings_count() as usize == brute_force_substrings(&s).len()
    }
}
