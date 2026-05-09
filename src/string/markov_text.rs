//! Word-level k-gram (order-`k`) Markov text generator.
//!
//! Build a probabilistic model from a corpus of words by mapping every
//! length-`k` window (the *prefix*) to the multiset of words that follow
//! it in the corpus, then emit fresh text by repeatedly sampling a
//! follower for the current prefix and sliding the window. The classic
//! Bentley / Kernighan / Pike example from *Programming Pearls*.
//!
//! The model is intentionally *deterministic in its choices once the rng
//! is fixed*: the caller supplies a function `rng(m) -> [0, m)` that
//! picks an index into the follower list. Because followers are stored
//! flat with their original multiplicities, picking a uniform index is
//! the same as sampling proportional to empirical frequency. A
//! convenience entry point [`MarkovModel::generate_xorshift`] uses an
//! internal seedable `XorShift64` PRNG so callers don't need to plug in
//! their own randomness.
//!
//! Generation stops early — before reaching `max_words` — whenever the
//! current prefix has no recorded follower (i.e. it appeared only at
//! the very end of the corpus). This is the standard "dead-end"
//! behaviour and keeps the algorithm dependency-free; richer policies
//! (wrap-around, smoothing, restart) can be layered on top.
//!
//! # Complexity
//!
//! Let `n = corpus.len()` and `k` be the order. Training is `O(n * k)`
//! time and `O(n * k)` space in the worst case — every window owns its
//! own `Vec<String>` key and a follower entry. Generation of `m` words
//! is `O(m * k)` — each step clones the `k`-word prefix to look it up
//! in the map.
//!
//! # Edge cases
//!
//! - **Empty corpus** → empty model, `generate` returns `Vec::new()`.
//! - **`k == 0`** → the prefix is the empty vector, so every follower
//!   in the corpus is collapsed under a single key. Sampling is then a
//!   draw from the unigram distribution.
//! - **Corpus shorter than `k + 1`** → no full prefix-plus-follower
//!   window exists, so the model is empty and generation returns
//!   `Vec::new()`.

use std::collections::HashMap;

/// A trained word-level k-gram Markov model.
///
/// Construct via [`MarkovModel::build`]; sample via
/// [`MarkovModel::generate`] or [`MarkovModel::generate_xorshift`]. The
/// internal representation is a `HashMap<Vec<String>, Vec<String>>`
/// from each k-word prefix to the flat list of follower words observed
/// in the corpus, with multiplicities preserved so a uniform pick over
/// the list is equivalent to sampling proportional to frequency.
#[derive(Debug, Clone, Default)]
pub struct MarkovModel {
    /// Window order. The prefix of every transition has exactly `k`
    /// words (or zero when `k == 0`).
    k: usize,
    /// Map from a k-word prefix to its empirical follower distribution,
    /// stored as a flat list with multiplicities.
    transitions: HashMap<Vec<String>, Vec<String>>,
    /// The initial prefix used to seed generation: the first `k` words
    /// of the corpus. Empty when `k == 0` or the corpus is too short.
    seed_prefix: Vec<String>,
}

impl MarkovModel {
    /// Train an order-`k` word-level Markov model on `corpus`.
    ///
    /// Each contiguous window of `k + 1` words becomes one transition:
    /// the first `k` words form the prefix and the `(k+1)`-th word is
    /// recorded as a follower. Repeated occurrences of the same
    /// prefix→follower pair are stored multiple times so that uniform
    /// sampling reflects empirical frequency.
    ///
    /// # Complexity
    ///
    /// `O(n * k)` time and `O(n * k)` space, where `n = corpus.len()`.
    pub fn build(corpus: &[&str], k: usize) -> Self {
        // No transitions exist when the corpus is shorter than k+1.
        if corpus.len() <= k {
            return Self {
                k,
                transitions: HashMap::new(),
                seed_prefix: Vec::new(),
            };
        }

        let mut transitions: HashMap<Vec<String>, Vec<String>> = HashMap::new();
        for window in corpus.windows(k + 1) {
            let prefix: Vec<String> = window[..k].iter().map(|w| (*w).to_string()).collect();
            let follower = window[k].to_string();
            transitions.entry(prefix).or_default().push(follower);
        }

        let seed_prefix: Vec<String> = corpus[..k].iter().map(|w| (*w).to_string()).collect();

        Self {
            k,
            transitions,
            seed_prefix,
        }
    }

    /// The order `k` this model was built with.
    pub const fn order(&self) -> usize {
        self.k
    }

    /// Number of distinct prefixes stored in the model.
    pub fn prefix_count(&self) -> usize {
        self.transitions.len()
    }

    /// Generate up to `max_words` words by sampling forward from the
    /// initial seed prefix (the first `k` words of the training
    /// corpus). `rng(m)` must return a uniformly random integer in
    /// `[0, m)`; it is called once per emitted word.
    ///
    /// Generation halts early in two cases:
    /// 1. The model is empty (empty corpus, or corpus shorter than
    ///    `k + 1`) — returns `Vec::new()`.
    /// 2. The current prefix has no recorded follower — returns the
    ///    words emitted so far.
    ///
    /// Otherwise the returned vector has length exactly `max_words`.
    ///
    /// # Complexity
    ///
    /// `O(max_words * k)` time and `O(max_words + k)` space.
    pub fn generate(&self, max_words: usize, mut rng: impl FnMut(usize) -> usize) -> Vec<String> {
        if max_words == 0 || self.transitions.is_empty() {
            return Vec::new();
        }

        let mut out: Vec<String> = Vec::with_capacity(max_words);
        // Sliding window of the most recent k emitted (or seeded) words.
        let mut prefix: Vec<String> = self.seed_prefix.clone();

        while out.len() < max_words {
            let Some(followers) = self.transitions.get(&prefix) else {
                break;
            };
            // Defensive: an empty follower list would be a bug, but
            // bail rather than panic on a zero-mod call.
            if followers.is_empty() {
                break;
            }
            let raw = rng(followers.len());
            // Clamp pathological rng outputs into range so we never
            // index out of bounds (the contract says `[0, m)`, but a
            // misbehaving caller shouldn't be able to crash the model).
            let idx = raw % followers.len();
            let next = followers[idx].clone();
            out.push(next.clone());

            if self.k > 0 {
                // Slide the window: drop the oldest, push the newest.
                prefix.remove(0);
                prefix.push(next);
            }
            // When k == 0 the prefix is and stays the empty vector.
        }

        out
    }

    /// Convenience wrapper around [`generate`](Self::generate) that
    /// uses an internal `XorShift64` PRNG seeded from `seed`. Useful
    /// for reproducible regression tests and demos that don't want to
    /// pull in the `rand` crate.
    ///
    /// A `seed` of `0` is silently promoted to `1` because `XorShift64`
    /// has a fixed point at zero.
    pub fn generate_xorshift(&self, max_words: usize, seed: u64) -> Vec<String> {
        let mut state: u64 = if seed == 0 { 1 } else { seed };
        let rng = |m: usize| -> usize {
            // XorShift64 — period 2^64 - 1, dependency-free.
            let mut x = state;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            state = x;
            (x as usize) % m
        };
        self.generate(max_words, rng)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn empty_corpus_yields_empty_output() {
        let model = MarkovModel::build(&[], 2);
        assert_eq!(model.prefix_count(), 0);
        assert!(model.generate(10, |_| 0).is_empty());
        assert!(model.generate_xorshift(10, 42).is_empty());
    }

    #[test]
    fn corpus_shorter_than_order_yields_empty_output() {
        // 2 words, k = 3 → no full window of size k+1 exists.
        let model = MarkovModel::build(&["hello", "world"], 3);
        assert_eq!(model.prefix_count(), 0);
        assert!(model.generate(10, |_| 0).is_empty());
    }

    #[test]
    fn k_zero_collapses_to_unigram_distribution() {
        // With k = 0 the only prefix is `[]`, and every word in the
        // corpus is a follower. Sampling is therefore a draw from the
        // unigram distribution — every emitted word must come from the
        // input multiset.
        let corpus = ["a", "b", "a", "c", "a", "b"];
        let model = MarkovModel::build(&corpus, 0);
        assert_eq!(model.prefix_count(), 1);

        let unigram: HashSet<&str> = corpus.iter().copied().collect();
        let out = model.generate_xorshift(20, 12345);
        assert_eq!(out.len(), 20);
        for w in &out {
            assert!(unigram.contains(w.as_str()), "unexpected word: {w}");
        }
    }

    #[test]
    fn k_one_unigram_like_draw() {
        // Order-1 model, single-word prefixes. Pinning the rng to
        // always pick index 0 traces a deterministic walk through the
        // first-recorded follower of each prefix.
        let corpus = ["a", "b", "a", "c"];
        let model = MarkovModel::build(&corpus, 1);
        // Prefix "a" → followers ["b", "c"]; "b" → ["a"]; "c" → [].
        let out = model.generate(10, |_| 0);
        // Seed prefix is "a" → first follower "b" → its first follower
        // "a" → "b" → "a" → ...
        assert_eq!(out, vec!["b", "a", "b", "a", "b", "a", "b", "a", "b", "a"]);
    }

    #[test]
    fn bigram_followers_after_known_prefix() {
        // From the issue: bigram on this corpus, "the quick" appears
        // twice and is followed by {"brown", "lazy"}.
        let corpus = [
            "the", "quick", "brown", "fox", "the", "quick", "lazy", "dog",
        ];
        let model = MarkovModel::build(&corpus, 2);

        let key = vec!["the".to_string(), "quick".to_string()];
        let followers = model
            .transitions
            .get(&key)
            .expect("'the quick' must be a known prefix");
        let unique: HashSet<&str> = followers.iter().map(String::as_str).collect();
        assert_eq!(unique, HashSet::from(["brown", "lazy"]));
        assert_eq!(followers.len(), 2);
    }

    #[test]
    fn output_length_bounded_by_max_words() {
        let corpus = [
            "the", "quick", "brown", "fox", "the", "quick", "lazy", "dog",
        ];
        let model = MarkovModel::build(&corpus, 2);
        for max in [0usize, 1, 3, 5, 100] {
            let out = model.generate_xorshift(max, 7);
            assert!(out.len() <= max, "len {} exceeded max {}", out.len(), max);
        }
    }

    #[test]
    fn all_emitted_words_appear_in_corpus() {
        let corpus = [
            "the", "quick", "brown", "fox", "the", "quick", "lazy", "dog",
        ];
        let vocab: HashSet<&str> = corpus.iter().copied().collect();
        let model = MarkovModel::build(&corpus, 2);

        for seed in [1u64, 2, 7, 99, 12345] {
            for w in model.generate_xorshift(50, seed) {
                assert!(vocab.contains(w.as_str()), "out-of-vocab word: {w}");
            }
        }
    }

    #[test]
    fn deterministic_regression_with_fixed_seed() {
        // Same seed → same output, byte-for-byte. This guards against
        // accidental nondeterminism (HashMap iteration, etc.) creeping
        // into generation.
        let corpus = [
            "the", "quick", "brown", "fox", "the", "quick", "lazy", "dog",
        ];
        let model = MarkovModel::build(&corpus, 2);
        let a = model.generate_xorshift(20, 42);
        let b = model.generate_xorshift(20, 42);
        assert_eq!(a, b);
        // ...and a different seed should generally diverge somewhere.
        let c = model.generate_xorshift(20, 43);
        assert!(a != c || a.is_empty());
    }

    #[test]
    fn dead_end_stops_generation_early() {
        // Linear corpus: every prefix has exactly one follower except
        // the last one, which has none. Generation must stop after at
        // most `corpus.len() - k` emitted words.
        let corpus = ["a", "b", "c", "d", "e"];
        let k = 2;
        let model = MarkovModel::build(&corpus, k);
        let out = model.generate(100, |_| 0);
        assert!(out.len() <= corpus.len() - k);
        // Walk is forced: prefix "a b" → "c"; "b c" → "d"; "c d" → "e";
        // "d e" → no follower, stop.
        assert_eq!(out, vec!["c", "d", "e"]);
    }

    #[test]
    fn rng_out_of_range_does_not_panic() {
        // Defensive: a misbehaving rng must not be able to crash us.
        let corpus = ["a", "b", "a", "c"];
        let model = MarkovModel::build(&corpus, 1);
        let out = model.generate(5, |_| usize::MAX);
        assert!(out.len() <= 5);
    }

    #[test]
    fn order_accessor_round_trips() {
        let model = MarkovModel::build(&["x", "y", "z"], 2);
        assert_eq!(model.order(), 2);
    }
}
