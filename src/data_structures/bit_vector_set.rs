//! Bit-vector set: a fast set of bounded non-negative integers backed by a
//! `Vec<u64>` bitmap.
//!
//! Each element of the universe `[0, universe)` occupies a single bit. Bit
//! position `x` lives at word `x / 64` and bit `x % 64`. A cached length is
//! kept in sync with every mutation so `len` is O(1).
//!
//! # Complexities
//! Let `w = universe.div_ceil(64)` be the number of backing words.
//!
//! | Operation         | Time    | Space      |
//! |-------------------|---------|------------|
//! | `new`             | O(w)    | O(w)       |
//! | `insert`          | O(1)    | —          |
//! | `remove`          | O(1)    | —          |
//! | `contains`        | O(1)    | —          |
//! | `len`             | O(1)    | —          |
//! | `is_empty`        | O(1)    | —          |
//! | `iter`            | O(w + k) where `k` is the number of set bits |
//! | `union_with`      | O(w)    | —          |
//! | `intersect_with`  | O(w)    | —          |
//! | `difference_with` | O(w)    | —          |
//! | `clear`           | O(w)    | —          |
//!
//! # Preconditions
//! * `insert`, `remove`, and `contains` panic on `x >= universe`.
//! * `union_with`, `intersect_with`, and `difference_with` panic if the two
//!   sets do not share the same universe.

/// Number of bits packed per backing word.
const BITS_PER_WORD: usize = 64;

/// Set of integers drawn from `[0, universe)` stored as a packed bitmap.
///
/// The universe is fixed at construction; mutations stay within that range
/// and bulk operations require both operands to share the same universe.
pub struct BitVectorSet {
    words: Vec<u64>,
    universe: usize,
    len: usize,
}

impl BitVectorSet {
    /// Creates an empty set whose elements live in `[0, universe)`.
    pub fn new(universe: usize) -> Self {
        let n_words = universe.div_ceil(BITS_PER_WORD);
        Self {
            words: vec![0; n_words],
            universe,
            len: 0,
        }
    }

    /// The size of the underlying universe (`x` must satisfy `x < universe`).
    pub const fn universe(&self) -> usize {
        self.universe
    }

    /// Number of elements currently in the set. O(1) (cached).
    pub const fn len(&self) -> usize {
        self.len
    }

    /// True if the set contains no elements.
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns `true` if `x` is in the set.
    ///
    /// # Panics
    /// Panics if `x >= self.universe()`.
    pub fn contains(&self, x: usize) -> bool {
        assert!(
            x < self.universe,
            "contains: x={x} is out of bounds (universe={})",
            self.universe
        );
        let (w, b) = (x / BITS_PER_WORD, x % BITS_PER_WORD);
        (self.words[w] >> b) & 1 == 1
    }

    /// Inserts `x`; returns `true` if it was newly inserted, `false` if it
    /// was already present.
    ///
    /// # Panics
    /// Panics if `x >= self.universe()`.
    pub fn insert(&mut self, x: usize) -> bool {
        assert!(
            x < self.universe,
            "insert: x={x} is out of bounds (universe={})",
            self.universe
        );
        let (w, b) = (x / BITS_PER_WORD, x % BITS_PER_WORD);
        let mask = 1u64 << b;
        let was_present = self.words[w] & mask != 0;
        if !was_present {
            self.words[w] |= mask;
            self.len += 1;
        }
        !was_present
    }

    /// Removes `x`; returns `true` if it was present.
    ///
    /// # Panics
    /// Panics if `x >= self.universe()`.
    pub fn remove(&mut self, x: usize) -> bool {
        assert!(
            x < self.universe,
            "remove: x={x} is out of bounds (universe={})",
            self.universe
        );
        let (w, b) = (x / BITS_PER_WORD, x % BITS_PER_WORD);
        let mask = 1u64 << b;
        let was_present = self.words[w] & mask != 0;
        if was_present {
            self.words[w] &= !mask;
            self.len -= 1;
        }
        was_present
    }

    /// Removes every element. O(words).
    pub fn clear(&mut self) {
        for w in &mut self.words {
            *w = 0;
        }
        self.len = 0;
    }

    /// Yields the elements of the set in ascending order.
    ///
    /// Walks the backing words and pops set bits via `u64::trailing_zeros`,
    /// so the cost is O(w + k) for `w` words and `k` set bits.
    pub fn iter(&self) -> Iter<'_> {
        Iter {
            words: &self.words,
            word_idx: 0,
            current: self.words.first().copied().unwrap_or(0),
        }
    }

    /// In-place union: `self ← self ∪ other`. O(words).
    ///
    /// # Panics
    /// Panics if the two sets have different universes.
    pub fn union_with(&mut self, other: &Self) {
        assert!(
            self.universe == other.universe,
            "union_with: universe mismatch ({} vs {})",
            self.universe,
            other.universe
        );
        let mut len = 0usize;
        for (a, &b) in self.words.iter_mut().zip(other.words.iter()) {
            *a |= b;
            len += a.count_ones() as usize;
        }
        self.len = len;
    }

    /// In-place intersection: `self ← self ∩ other`. O(words).
    ///
    /// # Panics
    /// Panics if the two sets have different universes.
    pub fn intersect_with(&mut self, other: &Self) {
        assert!(
            self.universe == other.universe,
            "intersect_with: universe mismatch ({} vs {})",
            self.universe,
            other.universe
        );
        let mut len = 0usize;
        for (a, &b) in self.words.iter_mut().zip(other.words.iter()) {
            *a &= b;
            len += a.count_ones() as usize;
        }
        self.len = len;
    }

    /// In-place difference: `self ← self \ other`. O(words).
    ///
    /// # Panics
    /// Panics if the two sets have different universes.
    pub fn difference_with(&mut self, other: &Self) {
        assert!(
            self.universe == other.universe,
            "difference_with: universe mismatch ({} vs {})",
            self.universe,
            other.universe
        );
        let mut len = 0usize;
        for (a, &b) in self.words.iter_mut().zip(other.words.iter()) {
            *a &= !b;
            len += a.count_ones() as usize;
        }
        self.len = len;
    }
}

/// Ascending-order iterator returned by [`BitVectorSet::iter`].
pub struct Iter<'a> {
    words: &'a [u64],
    word_idx: usize,
    current: u64,
}

impl<'a> IntoIterator for &'a BitVectorSet {
    type Item = usize;
    type IntoIter = Iter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl Iterator for Iter<'_> {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        // Advance to the next non-empty word.
        while self.current == 0 {
            self.word_idx += 1;
            if self.word_idx >= self.words.len() {
                return None;
            }
            self.current = self.words[self.word_idx];
        }
        // Lowest set bit.
        let bit = self.current.trailing_zeros() as usize;
        // Clear it (LSB reset trick).
        self.current &= self.current - 1;
        Some(self.word_idx * BITS_PER_WORD + bit)
    }
}

#[cfg(test)]
mod tests {
    use super::BitVectorSet;
    use quickcheck::TestResult;
    use quickcheck_macros::quickcheck;
    use std::collections::BTreeSet;

    // -----------------------------------------------------------------------
    // Unit tests
    // -----------------------------------------------------------------------

    #[test]
    fn empty_set_basics() {
        let s = BitVectorSet::new(0);
        assert!(s.is_empty());
        assert_eq!(s.len(), 0);
        assert_eq!(s.iter().count(), 0);
        assert_eq!(s.universe(), 0);
    }

    #[test]
    fn empty_universe_iter_is_empty() {
        let s = BitVectorSet::new(0);
        assert!(s.iter().next().is_none());
    }

    #[test]
    fn insert_remove_contains_round_trip() {
        let mut s = BitVectorSet::new(200);
        assert!(s.insert(0));
        assert!(s.insert(63));
        assert!(s.insert(64));
        assert!(s.insert(199));
        assert!(!s.insert(64)); // duplicate
        assert_eq!(s.len(), 4);

        assert!(s.contains(0));
        assert!(s.contains(63));
        assert!(s.contains(64));
        assert!(s.contains(199));
        assert!(!s.contains(1));
        assert!(!s.contains(100));

        assert!(s.remove(64));
        assert!(!s.remove(64)); // already gone
        assert!(!s.contains(64));
        assert_eq!(s.len(), 3);
    }

    #[test]
    fn insert_full_universe() {
        let n = 130;
        let mut s = BitVectorSet::new(n);
        for x in 0..n {
            assert!(s.insert(x));
        }
        assert_eq!(s.len(), n);
        for x in 0..n {
            assert!(s.contains(x));
        }
    }

    #[test]
    fn iter_yields_ascending_order() {
        let mut s = BitVectorSet::new(300);
        let inserted = [299, 5, 64, 0, 128, 63, 65, 200];
        for &x in &inserted {
            s.insert(x);
        }
        let collected: Vec<usize> = s.iter().collect();
        let mut expected: Vec<usize> = inserted.to_vec();
        expected.sort_unstable();
        assert_eq!(collected, expected);
    }

    #[test]
    fn iter_count_matches_len() {
        let mut s = BitVectorSet::new(1000);
        for x in (0..1000).step_by(7) {
            s.insert(x);
        }
        assert_eq!(s.iter().count(), s.len());
    }

    #[test]
    fn clear_resets_state() {
        let mut s = BitVectorSet::new(128);
        s.insert(1);
        s.insert(64);
        s.insert(127);
        s.clear();
        assert!(s.is_empty());
        assert_eq!(s.len(), 0);
        assert_eq!(s.iter().count(), 0);
        // Reusable after clear.
        assert!(s.insert(10));
        assert!(s.contains(10));
    }

    #[test]
    fn union_with_basic() {
        let mut a = BitVectorSet::new(100);
        let mut b = BitVectorSet::new(100);
        for x in [1, 2, 3, 64] {
            a.insert(x);
        }
        for x in [3, 4, 65] {
            b.insert(x);
        }
        a.union_with(&b);
        let collected: Vec<usize> = a.iter().collect();
        assert_eq!(collected, vec![1, 2, 3, 4, 64, 65]);
        assert_eq!(a.len(), 6);
    }

    #[test]
    fn intersect_with_basic() {
        let mut a = BitVectorSet::new(100);
        let mut b = BitVectorSet::new(100);
        for x in [1, 2, 3, 64] {
            a.insert(x);
        }
        for x in [2, 3, 4, 64] {
            b.insert(x);
        }
        a.intersect_with(&b);
        let collected: Vec<usize> = a.iter().collect();
        assert_eq!(collected, vec![2, 3, 64]);
        assert_eq!(a.len(), 3);
    }

    #[test]
    fn difference_with_basic() {
        let mut a = BitVectorSet::new(100);
        let mut b = BitVectorSet::new(100);
        for x in [1, 2, 3, 64] {
            a.insert(x);
        }
        for x in [3, 64] {
            b.insert(x);
        }
        a.difference_with(&b);
        let collected: Vec<usize> = a.iter().collect();
        assert_eq!(collected, vec![1, 2]);
        assert_eq!(a.len(), 2);
    }

    #[test]
    #[should_panic(expected = "is out of bounds")]
    fn insert_out_of_bounds_panics() {
        let mut s = BitVectorSet::new(10);
        s.insert(10);
    }

    #[test]
    #[should_panic(expected = "is out of bounds")]
    fn remove_out_of_bounds_panics() {
        let mut s = BitVectorSet::new(10);
        s.remove(42);
    }

    #[test]
    #[should_panic(expected = "is out of bounds")]
    fn contains_out_of_bounds_panics() {
        let s = BitVectorSet::new(10);
        let _ = s.contains(10);
    }

    #[test]
    #[should_panic(expected = "universe mismatch")]
    fn union_universe_mismatch_panics() {
        let mut a = BitVectorSet::new(10);
        let b = BitVectorSet::new(20);
        a.union_with(&b);
    }

    #[test]
    #[should_panic(expected = "universe mismatch")]
    fn intersect_universe_mismatch_panics() {
        let mut a = BitVectorSet::new(10);
        let b = BitVectorSet::new(20);
        a.intersect_with(&b);
    }

    #[test]
    #[should_panic(expected = "universe mismatch")]
    fn difference_universe_mismatch_panics() {
        let mut a = BitVectorSet::new(10);
        let b = BitVectorSet::new(20);
        a.difference_with(&b);
    }

    // -----------------------------------------------------------------------
    // Property-based tests against std `BTreeSet` on a small universe
    // -----------------------------------------------------------------------

    const PROP_UNIVERSE: usize = 200;

    fn build_pair(items: &[u8]) -> (BitVectorSet, BTreeSet<usize>) {
        let mut bv = BitVectorSet::new(PROP_UNIVERSE);
        let mut tree = BTreeSet::new();
        for &x in items {
            let v = (x as usize) % PROP_UNIVERSE;
            bv.insert(v);
            tree.insert(v);
        }
        (bv, tree)
    }

    #[quickcheck]
    #[allow(clippy::needless_pass_by_value)]
    fn prop_iter_matches_btreeset(items: Vec<u8>) -> bool {
        let (bv, tree) = build_pair(&items);
        let bv_collected: Vec<usize> = bv.iter().collect();
        let tree_collected: Vec<usize> = tree.iter().copied().collect();
        bv_collected == tree_collected && bv.len() == tree.len()
    }

    #[quickcheck]
    #[allow(clippy::needless_pass_by_value)]
    fn prop_contains_matches_btreeset(items: Vec<u8>, queries: Vec<u8>) -> bool {
        let (bv, tree) = build_pair(&items);
        for q in queries {
            let v = (q as usize) % PROP_UNIVERSE;
            if bv.contains(v) != tree.contains(&v) {
                return false;
            }
        }
        true
    }

    #[quickcheck]
    #[allow(clippy::needless_pass_by_value)]
    fn prop_remove_matches_btreeset(items: Vec<u8>, removals: Vec<u8>) -> bool {
        let (mut bv, mut tree) = build_pair(&items);
        for r in removals {
            let v = (r as usize) % PROP_UNIVERSE;
            let bv_was = bv.remove(v);
            let tree_was = tree.remove(&v);
            if bv_was != tree_was {
                return false;
            }
        }
        let bv_collected: Vec<usize> = bv.iter().collect();
        let tree_collected: Vec<usize> = tree.iter().copied().collect();
        bv_collected == tree_collected && bv.len() == tree.len()
    }

    #[quickcheck]
    #[allow(clippy::needless_pass_by_value)]
    fn prop_union_matches_btreeset(a: Vec<u8>, b: Vec<u8>) -> bool {
        let (mut bv_a, mut tree_a) = build_pair(&a);
        let (bv_b, tree_b) = build_pair(&b);
        bv_a.union_with(&bv_b);
        tree_a.extend(tree_b.iter().copied());
        let bv_collected: Vec<usize> = bv_a.iter().collect();
        let tree_collected: Vec<usize> = tree_a.iter().copied().collect();
        bv_collected == tree_collected && bv_a.len() == tree_a.len()
    }

    #[quickcheck]
    #[allow(clippy::needless_pass_by_value)]
    fn prop_intersect_matches_btreeset(a: Vec<u8>, b: Vec<u8>) -> bool {
        let (mut bv_a, tree_a) = build_pair(&a);
        let (bv_b, tree_b) = build_pair(&b);
        bv_a.intersect_with(&bv_b);
        let expected: BTreeSet<usize> = tree_a.intersection(&tree_b).copied().collect();
        let bv_collected: Vec<usize> = bv_a.iter().collect();
        let tree_collected: Vec<usize> = expected.iter().copied().collect();
        bv_collected == tree_collected && bv_a.len() == expected.len()
    }

    #[quickcheck]
    #[allow(clippy::needless_pass_by_value)]
    fn prop_difference_matches_btreeset(a: Vec<u8>, b: Vec<u8>) -> bool {
        let (mut bv_a, tree_a) = build_pair(&a);
        let (bv_b, tree_b) = build_pair(&b);
        bv_a.difference_with(&bv_b);
        let expected: BTreeSet<usize> = tree_a.difference(&tree_b).copied().collect();
        let bv_collected: Vec<usize> = bv_a.iter().collect();
        let tree_collected: Vec<usize> = expected.iter().copied().collect();
        bv_collected == tree_collected && bv_a.len() == expected.len()
    }

    #[quickcheck]
    #[allow(clippy::needless_pass_by_value)]
    fn prop_insert_returns_newly_inserted(items: Vec<u8>) -> TestResult {
        let mut bv = BitVectorSet::new(PROP_UNIVERSE);
        let mut tree = BTreeSet::new();
        for x in items {
            let v = (x as usize) % PROP_UNIVERSE;
            let bv_new = bv.insert(v);
            let tree_new = tree.insert(v);
            if bv_new != tree_new {
                return TestResult::failed();
            }
        }
        if bv.len() != tree.len() {
            return TestResult::failed();
        }
        TestResult::passed()
    }
}
