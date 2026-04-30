//! Aho–Corasick multi-pattern substring search.
//!
//! Builds a trie of all input patterns, then computes failure (suffix) links
//! via breadth-first traversal so the haystack can be scanned in a single
//! linear pass. Reporting every match — including overlaps and patterns that
//! end mid-string — runs in `O(N + M + Z)` time, where `N` is the haystack
//! length, `M` is the total length of all patterns, and `Z` is the number of
//! reported matches. Memory usage is `O(M)` for the trie plus the failure
//! and dictionary-suffix link tables.
//!
//! The automaton is byte-oriented: patterns and haystacks are slices of `u8`,
//! so it works uniformly on ASCII, raw bytes, and pre-encoded UTF-8 (matches
//! land on byte indices, not codepoint indices). Empty patterns are skipped
//! during construction so they never produce spurious whole-string matches.

use std::collections::{HashMap, VecDeque};

/// A compiled Aho–Corasick automaton over the byte alphabet.
///
/// Construct with [`AhoCorasick::new`] and search with
/// [`AhoCorasick::find_matches`]. The automaton is immutable after
/// construction; clone or rebuild to change the pattern set.
pub struct AhoCorasick {
    /// Child transitions for each node, keyed by the next byte.
    goto: Vec<HashMap<u8, usize>>,
    /// Failure link for each node — the longest proper suffix of the path
    /// from the root that is itself a prefix of some pattern.
    fail: Vec<usize>,
    /// Output: `(pattern_index, pattern_length)` pairs that terminate at
    /// this node directly (dictionary-suffix walks are handled at search
    /// time via [`Self::dict_link`]).
    output: Vec<Vec<(usize, usize)>>,
    /// Dictionary-suffix link: the nearest ancestor along failure links that
    /// is the end of some pattern, or 0 (root) if none exists.
    dict_link: Vec<usize>,
}

impl AhoCorasick {
    /// Builds the automaton from `patterns`.
    ///
    /// Empty patterns are skipped so they do not match at every position.
    /// Pattern indices in returned matches refer to the original `patterns`
    /// slice, including any skipped empty entries.
    pub fn new<P: AsRef<[u8]>>(patterns: &[P]) -> Self {
        let mut ac = Self {
            goto: vec![HashMap::new()],
            fail: vec![0],
            output: vec![Vec::new()],
            dict_link: vec![0],
        };

        // Phase 1: build the trie.
        for (idx, pat) in patterns.iter().enumerate() {
            let bytes = pat.as_ref();
            if bytes.is_empty() {
                continue;
            }
            let mut node = 0_usize;
            for &b in bytes {
                if let Some(&next) = ac.goto[node].get(&b) {
                    node = next;
                } else {
                    let new_node = ac.goto.len();
                    ac.goto.push(HashMap::new());
                    ac.fail.push(0);
                    ac.output.push(Vec::new());
                    ac.dict_link.push(0);
                    ac.goto[node].insert(b, new_node);
                    node = new_node;
                }
            }
            ac.output[node].push((idx, bytes.len()));
        }

        // Phase 2: BFS to wire failure and dictionary-suffix links.
        let mut queue: VecDeque<usize> = VecDeque::new();
        let root_children: Vec<usize> = ac.goto[0].values().copied().collect();
        for child in root_children {
            ac.fail[child] = 0;
            queue.push_back(child);
        }

        while let Some(u) = queue.pop_front() {
            let edges: Vec<(u8, usize)> = ac.goto[u].iter().map(|(&b, &v)| (b, v)).collect();
            for (b, v) in edges {
                // Failure link for v: walk u's failure chain until we find a
                // node with a transition on byte b (or fall back to root).
                let mut f = ac.fail[u];
                let target = loop {
                    if let Some(&next) = ac.goto[f].get(&b) {
                        if next != v {
                            break next;
                        }
                    }
                    if f == 0 {
                        break 0;
                    }
                    f = ac.fail[f];
                };
                ac.fail[v] = target;

                // Dictionary-suffix link: nearest ancestor in fail chain
                // that itself terminates a pattern.
                ac.dict_link[v] = if ac.output[target].is_empty() {
                    ac.dict_link[target]
                } else {
                    target
                };
                queue.push_back(v);
            }
        }

        ac
    }

    /// Scans `haystack` and returns every match as a
    /// `(haystack_start_index, pattern_index)` pair.
    ///
    /// Overlapping matches are included. Order is by end-position of the
    /// match in the haystack; ties (multiple patterns ending at the same
    /// position) follow the order in which the trie reports them.
    pub fn find_matches(&self, haystack: &[u8]) -> Vec<(usize, usize)> {
        let mut matches = Vec::new();
        if self.output.len() <= 1 || haystack.is_empty() {
            // Trie holds only the root, so no non-empty patterns were inserted.
            return matches;
        }

        let mut node = 0_usize;
        for (i, &b) in haystack.iter().enumerate() {
            // Follow failure links until we find a transition or hit root.
            loop {
                if let Some(&next) = self.goto[node].get(&b) {
                    node = next;
                    break;
                }
                if node == 0 {
                    break;
                }
                node = self.fail[node];
            }

            // Emit direct hits at this node, then walk the dictionary-suffix
            // chain to emit every pattern that ends here as a proper suffix.
            let mut out = node;
            while out != 0 {
                for &(pat_idx, pat_len) in &self.output[out] {
                    matches.push((i + 1 - pat_len, pat_idx));
                }
                out = self.dict_link[out];
            }
        }

        matches
    }
}

#[cfg(test)]
mod tests {
    use super::AhoCorasick;
    use quickcheck_macros::quickcheck;

    fn naive_search(patterns: &[&[u8]], haystack: &[u8]) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        for i in 0..=haystack.len() {
            for (idx, pat) in patterns.iter().enumerate() {
                if pat.is_empty() {
                    continue;
                }
                if i + pat.len() <= haystack.len() && &haystack[i..i + pat.len()] == *pat {
                    out.push((i, idx));
                }
            }
        }
        out
    }

    fn sorted(mut v: Vec<(usize, usize)>) -> Vec<(usize, usize)> {
        v.sort_unstable();
        v
    }

    #[test]
    fn empty_patterns() {
        let ac = AhoCorasick::new::<&[u8]>(&[]);
        assert_eq!(ac.find_matches(b"hello"), Vec::<(usize, usize)>::new());
    }

    #[test]
    fn empty_haystack() {
        let ac = AhoCorasick::new(&[b"abc".as_slice()]);
        assert_eq!(ac.find_matches(b""), Vec::<(usize, usize)>::new());
    }

    #[test]
    fn empty_pattern_skipped() {
        let ac = AhoCorasick::new(&[b"".as_slice(), b"a".as_slice()]);
        // The empty pattern at index 0 is skipped; only "a" produces matches.
        assert_eq!(ac.find_matches(b"aa"), vec![(0, 1), (1, 1)]);
    }

    #[test]
    fn single_pattern_matches_naive() {
        let ac = AhoCorasick::new(&[b"world".as_slice()]);
        assert_eq!(ac.find_matches(b"hello world"), vec![(6, 0)]);
    }

    #[test]
    fn classic_ushers() {
        // The textbook Aho–Corasick example: he, she, his, hers in "ushers"
        // matches she @ 1, he @ 2 (as a suffix of she), hers @ 2.
        let pats: &[&[u8]] = &[b"he", b"she", b"his", b"hers"];
        let ac = AhoCorasick::new(pats);
        let got = sorted(ac.find_matches(b"ushers"));
        let expected = sorted(vec![(1, 1), (2, 0), (2, 3)]);
        assert_eq!(got, expected);
    }

    #[test]
    fn overlapping_matches() {
        let ac = AhoCorasick::new(&[b"aa".as_slice()]);
        assert_eq!(ac.find_matches(b"aaaa"), vec![(0, 0), (1, 0), (2, 0)]);
    }

    #[test]
    fn pattern_equals_haystack() {
        let ac = AhoCorasick::new(&[b"abc".as_slice()]);
        assert_eq!(ac.find_matches(b"abc"), vec![(0, 0)]);
    }

    #[test]
    fn pattern_not_present() {
        let ac = AhoCorasick::new(&[b"xyz".as_slice()]);
        assert_eq!(ac.find_matches(b"abcdef"), Vec::<(usize, usize)>::new());
    }

    #[test]
    fn multiple_patterns_with_shared_prefix() {
        let pats: &[&[u8]] = &[b"abc", b"abcd", b"bcd"];
        let ac = AhoCorasick::new(pats);
        let got = sorted(ac.find_matches(b"abcd"));
        let expected = sorted(vec![(0, 0), (0, 1), (1, 2)]);
        assert_eq!(got, expected);
    }

    #[test]
    fn duplicate_patterns_each_reported() {
        let pats: &[&[u8]] = &[b"ab", b"ab"];
        let ac = AhoCorasick::new(pats);
        let got = sorted(ac.find_matches(b"ab"));
        let expected = sorted(vec![(0, 0), (0, 1)]);
        assert_eq!(got, expected);
    }

    #[quickcheck]
    fn matches_brute_force(patterns: Vec<Vec<u8>>, haystack: Vec<u8>) -> bool {
        // Trim down to the documented bounds: ≤5 patterns of length ≤5,
        // haystack length ≤50.
        let patterns: Vec<Vec<u8>> = patterns
            .into_iter()
            .take(5)
            .map(|p| p.into_iter().take(5).collect())
            .collect();
        let haystack: Vec<u8> = haystack.into_iter().take(50).collect();

        let pat_refs: Vec<&[u8]> = patterns.iter().map(Vec::as_slice).collect();
        let ac = AhoCorasick::new(&pat_refs);
        let got = sorted(ac.find_matches(&haystack));
        let expected = sorted(naive_search(&pat_refs, &haystack));
        got == expected
    }
}
