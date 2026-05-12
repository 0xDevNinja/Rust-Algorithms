//! Huffman coding: optimal prefix-free binary codes for a symbol-frequency
//! table.
//!
//! Given an alphabet with non-negative integer frequencies, Huffman's greedy
//! construction produces a prefix-free binary code that minimises the
//! expected encoded length `Σ freq(c) · len(code(c))`. The classic argument:
//! in any optimal tree the two least-frequent symbols are siblings at maximum
//! depth, so repeatedly merging the two lightest partial trees into a new
//! internal node never loses optimality.
//!
//! Algorithm: push every `(freq, leaf)` into a min-heap (a `BinaryHeap` with
//! `Reverse` weights), then `n - 1` times pop the two lightest trees, fuse
//! them under a fresh internal node with summed frequency, and push the
//! result back. The remaining root is walked once to read off each leaf's
//! bit-path: a left edge emits `false`, a right edge emits `true`.
//!
//! Tie-break determinism: heap entries are ordered by `(frequency, sequence)`
//! where `sequence` is a strictly increasing counter assigned in input order
//! and incremented again for each merged internal node. Two trees with equal
//! frequency therefore always pop in the order they were created. Combined
//! with the convention that the *first* pop becomes the left child and the
//! *second* pop becomes the right child, this makes [`build_codes`] return
//! the same `HashMap` for the same `(char, freq)` multiset regardless of the
//! input slice's order — modulo the `HashMap`'s own iteration randomness,
//! which never affects the codes themselves.
//!
//! Degenerate input: an alphabet of a single distinct symbol cannot form a
//! two-leaf tree, so the natural Huffman code length would be zero — useless
//! for round-tripping. [`build_codes`] therefore emits a one-bit code
//! (`[false]`) in that case so that [`encode`] / [`decode`] still work.
//!
//! Complexity: `O(n log n)` time and `O(n)` extra space, where `n` is the
//! number of distinct symbols. Encoding and decoding run in time
//! proportional to the input length plus the produced/consumed bit count.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

/// A node in the partial Huffman tree held on the heap. Leaves carry a
/// symbol; internal nodes own their two children. Boxed to keep the enum
/// size small and to allow recursive ownership.
enum Node {
    Leaf(char),
    Internal(Box<Self>, Box<Self>),
}

/// Heap entry. Ordered by `(weight, seq)` so that ties break deterministically
/// in insertion order. Wrapping with `Reverse` at push-time turns the
/// max-heap `BinaryHeap` into a min-heap.
struct HeapItem {
    weight: u64,
    seq: u64,
    node: Node,
}

impl PartialEq for HeapItem {
    fn eq(&self, other: &Self) -> bool {
        self.weight == other.weight && self.seq == other.seq
    }
}
impl Eq for HeapItem {}
impl PartialOrd for HeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for HeapItem {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.weight
            .cmp(&other.weight)
            .then_with(|| self.seq.cmp(&other.seq))
    }
}

/// Build prefix-free Huffman codes for the given `(symbol, frequency)` table.
///
/// Symbols with frequency zero are ignored. Duplicate symbols have their
/// frequencies summed. An empty table yields an empty map; a table with a
/// single distinct (positive-frequency) symbol yields a one-bit fallback
/// code so that [`encode`] / [`decode`] still round-trip.
///
/// Tie-breaking is deterministic on the multiset of `(symbol, frequency)`
/// pairs (see module docs); the produced code lengths and bit-paths are
/// stable across runs and across reorderings of the input slice.
///
/// Time: `O(n log n)`. Space: `O(n)`.
#[must_use]
pub fn build_codes(freqs: &[(char, u64)]) -> HashMap<char, Vec<bool>> {
    // Coalesce duplicate symbols and drop zero-frequency entries. Iteration
    // order over the resulting map is randomised, but we re-sort below to
    // make the heap insertion order — and therefore tie-breaks — deterministic.
    let mut totals: HashMap<char, u64> = HashMap::new();
    for &(ch, f) in freqs {
        if f == 0 {
            continue;
        }
        *totals.entry(ch).or_insert(0) += f;
    }

    if totals.is_empty() {
        return HashMap::new();
    }

    // Single distinct symbol: emit a one-bit fallback so encode/decode work.
    if totals.len() == 1 {
        let (&ch, _) = totals.iter().next().expect("len == 1");
        let mut codes = HashMap::with_capacity(1);
        codes.insert(ch, vec![false]);
        return codes;
    }

    // Sort by (char, freq) so the initial heap-push order is fully
    // deterministic, independent of the input slice's order.
    let mut leaves: Vec<(char, u64)> = totals.into_iter().collect();
    leaves.sort_unstable_by_key(|&(ch, _)| ch);

    let mut heap: BinaryHeap<Reverse<HeapItem>> = BinaryHeap::with_capacity(leaves.len());
    let mut next_seq: u64 = 0;
    for (ch, weight) in leaves {
        heap.push(Reverse(HeapItem {
            weight,
            seq: next_seq,
            node: Node::Leaf(ch),
        }));
        next_seq += 1;
    }

    while heap.len() > 1 {
        let Reverse(left) = heap.pop().expect("heap has ≥ 2 items");
        let Reverse(right) = heap.pop().expect("heap has ≥ 2 items");
        let merged_weight = left.weight + right.weight;
        let merged = HeapItem {
            weight: merged_weight,
            seq: next_seq,
            node: Node::Internal(Box::new(left.node), Box::new(right.node)),
        };
        next_seq += 1;
        heap.push(Reverse(merged));
    }

    let Reverse(root) = heap.pop().expect("heap had ≥ 1 item after merges");
    let mut codes: HashMap<char, Vec<bool>> = HashMap::new();
    let mut path: Vec<bool> = Vec::new();
    walk(&root.node, &mut path, &mut codes);
    codes
}

/// DFS that records each leaf's root-to-leaf bit-path.
fn walk(node: &Node, path: &mut Vec<bool>, out: &mut HashMap<char, Vec<bool>>) {
    match node {
        Node::Leaf(ch) => {
            out.insert(*ch, path.clone());
        }
        Node::Internal(left, right) => {
            path.push(false);
            walk(left, path, out);
            path.pop();
            path.push(true);
            walk(right, path, out);
            path.pop();
        }
    }
}

/// Encode `text` using the supplied code table. Panics if `text` contains a
/// symbol absent from `codes`; callers should derive `codes` from a frequency
/// table that covers every character they intend to encode.
//
// `implicit_hasher` is suppressed because the public API is fixed by issue #82
// to use the default `HashMap` hasher.
#[allow(clippy::implicit_hasher)]
#[must_use]
pub fn encode(text: &str, codes: &HashMap<char, Vec<bool>>) -> Vec<bool> {
    let mut out: Vec<bool> = Vec::new();
    for ch in text.chars() {
        let bits = codes
            .get(&ch)
            .unwrap_or_else(|| panic!("character {ch:?} missing from code table"));
        out.extend_from_slice(bits);
    }
    out
}

/// Decode a Huffman bitstream against the same code table used to produce it.
///
/// Returns `None` if the bitstream is malformed — i.e. it ends mid-codeword
/// or contains a prefix that does not match any code (the latter cannot
/// happen for a complete code table built by [`build_codes`] but is checked
/// defensively in case the caller hand-rolls a partial table).
//
// `implicit_hasher` is suppressed because the public API is fixed by issue #82
// to use the default `HashMap` hasher.
#[allow(clippy::implicit_hasher)]
#[must_use]
pub fn decode(bits: &[bool], codes: &HashMap<char, Vec<bool>>) -> Option<String> {
    if codes.is_empty() {
        // No symbols defined: only the empty bitstream is valid.
        return if bits.is_empty() {
            Some(String::new())
        } else {
            None
        };
    }

    // Invert the table: code-bits → symbol. Codes are prefix-free by
    // construction, so a longest-prefix scan is unambiguous.
    let mut by_code: HashMap<&[bool], char> = HashMap::with_capacity(codes.len());
    for (ch, code) in codes {
        by_code.insert(code.as_slice(), *ch);
    }
    let max_len = codes.values().map(Vec::len).max().unwrap_or(0);

    let mut out = String::new();
    let mut i = 0;
    while i < bits.len() {
        let mut matched: Option<(usize, char)> = None;
        let upper = (i + max_len).min(bits.len());
        for end in (i + 1)..=upper {
            if let Some(&ch) = by_code.get(&bits[i..end]) {
                matched = Some((end, ch));
                break;
            }
        }
        let (next_i, ch) = matched?;
        out.push(ch);
        i = next_i;
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::{build_codes, decode, encode};
    use quickcheck_macros::quickcheck;
    use std::collections::HashMap;

    /// Frequency-count helper for tests.
    fn freqs_of(s: &str) -> Vec<(char, u64)> {
        let mut m: HashMap<char, u64> = HashMap::new();
        for c in s.chars() {
            *m.entry(c).or_insert(0) += 1;
        }
        m.into_iter().collect()
    }

    #[test]
    fn empty_freqs_yields_empty_codes() {
        let codes = build_codes(&[]);
        assert!(codes.is_empty());
        assert_eq!(decode(&[], &codes).as_deref(), Some(""));
    }

    #[test]
    fn zero_frequencies_are_ignored() {
        let codes = build_codes(&[('a', 0), ('b', 0)]);
        assert!(codes.is_empty());
    }

    #[test]
    fn single_symbol_uses_one_bit_fallback() {
        let codes = build_codes(&[('a', 4)]);
        assert_eq!(codes.len(), 1);
        assert_eq!(codes.get(&'a'), Some(&vec![false]));

        let encoded = encode("aaaa", &codes);
        assert_eq!(encoded, vec![false, false, false, false]);
        assert_eq!(decode(&encoded, &codes).as_deref(), Some("aaaa"));
    }

    #[test]
    fn prefix_free_property_holds() {
        // No code may be a prefix of another. Verified directly on the table.
        let codes = build_codes(&freqs_of("the quick brown fox jumps over the lazy dog"));
        let bits: Vec<&Vec<bool>> = codes.values().collect();
        for (i, a) in bits.iter().enumerate() {
            for (j, b) in bits.iter().enumerate() {
                if i == j {
                    continue;
                }
                let shorter = a.len().min(b.len());
                let a_is_prefix_of_b = a.len() <= b.len() && a[..] == b[..shorter];
                assert!(
                    !a_is_prefix_of_b,
                    "codes are not prefix-free: {a:?} vs {b:?}"
                );
            }
        }
    }

    #[test]
    fn abracadabra_round_trip_and_length() {
        let text = "abracadabra";
        let counts = freqs_of(text);
        let codes = build_codes(&counts);
        let encoded = encode(text, &codes);
        let decoded = decode(&encoded, &codes).expect("well-formed bitstream");
        assert_eq!(decoded, text);

        // Total encoded length must equal Σ freq · code_len.
        let expected_len: usize = counts
            .iter()
            .map(|(c, f)| (*f as usize) * codes[c].len())
            .sum();
        assert_eq!(encoded.len(), expected_len);

        // For "abracadabra" (a:5, b:2, r:2, c:1, d:1) the optimal code length
        // sum is 23 bits. Sanity check we land on the optimum.
        assert_eq!(encoded.len(), 23);
    }

    #[test]
    fn deterministic_on_tiebreaks() {
        // Permuting the input slice does not change the produced codes, since
        // build_codes coalesces and re-sorts by (char, freq) before heap
        // construction.
        let a = build_codes(&[('a', 1), ('b', 1), ('c', 1), ('d', 1)]);
        let b = build_codes(&[('d', 1), ('c', 1), ('b', 1), ('a', 1)]);
        let c = build_codes(&[('b', 1), ('d', 1), ('a', 1), ('c', 1)]);
        assert_eq!(a, b);
        assert_eq!(a, c);

        // Documented tie-break shape: with four equal-weight leaves visited in
        // ('a','b','c','d') order, the first merge fuses (a,b) → seq 4, the
        // second fuses (c,d) → seq 5, and the root fuses those two internals.
        // That makes 'a','b' the left subtree and 'c','d' the right, with
        // 'a' as the leftmost leaf.
        assert_eq!(a[&'a'], vec![false, false]);
        assert_eq!(a[&'b'], vec![false, true]);
        assert_eq!(a[&'c'], vec![true, false]);
        assert_eq!(a[&'d'], vec![true, true]);
    }

    #[test]
    fn decode_rejects_truncated_bitstream() {
        let codes = build_codes(&freqs_of("abracadabra"));
        // Pick the symbol with the longest codeword and chop just *one* bit
        // off its encoding. The remaining prefix cannot itself be a complete
        // code (Huffman codes are prefix-free), so decode must fail.
        let (sym, longest) = codes
            .iter()
            .max_by_key(|(_, v)| v.len())
            .expect("non-empty codes");
        assert!(longest.len() > 1, "need a multi-bit code for this test");
        let mut bits = encode(&sym.to_string(), &codes);
        bits.pop();
        assert!(decode(&bits, &codes).is_none());

        // Same idea on a longer text: corrupt by dropping the trailing bit of
        // the last codeword if that codeword is multi-bit.
        let text = "abracadabra";
        let last = text.chars().next_back().unwrap();
        if codes[&last].len() > 1 {
            let mut encoded = encode(text, &codes);
            encoded.pop();
            assert!(decode(&encoded, &codes).is_none());
        }
    }

    #[test]
    fn decode_rejects_unrecognised_prefix() {
        // Hand-roll a partial code table missing one symbol: any bitstream
        // that would require the missing code must be rejected.
        let mut codes: HashMap<char, Vec<bool>> = HashMap::new();
        codes.insert('a', vec![false]);
        codes.insert('b', vec![true, false]);
        // No code starts with `1, 1`, so this is unrecognisable.
        let bits = vec![true, true];
        assert!(decode(&bits, &codes).is_none());
    }

    #[test]
    fn decode_extra_input_when_table_empty() {
        let codes: HashMap<char, Vec<bool>> = HashMap::new();
        assert_eq!(decode(&[], &codes).as_deref(), Some(""));
        assert!(decode(&[false], &codes).is_none());
    }

    #[allow(clippy::needless_pass_by_value)]
    #[quickcheck]
    fn random_strings_round_trip(s: String) -> bool {
        // Cap to keep the test fast; quickcheck strings can be long.
        let text: String = s.chars().take(64).collect();
        let counts = freqs_of(&text);
        let codes = build_codes(&counts);
        let encoded = encode(&text, &codes);
        decode(&encoded, &codes).map_or(text.is_empty() && !encoded.is_empty(), |decoded| {
            decoded == text
        })
    }
}
