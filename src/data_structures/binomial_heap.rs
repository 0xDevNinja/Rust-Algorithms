//! Binomial heap (Vuillemin 1978).
//!
//! A **binomial heap** is a forest of binomial trees satisfying the min-heap
//! property. A binomial tree `B_k` of order k has `2^k` nodes; `B_0` is a
//! single node, and `B_k` is formed by linking two `B_{k-1}` trees (the one
//! with the larger root becomes the left-most child of the other). A binomial
//! heap stores at most one tree of each order, making the root list analogous
//! to the binary representation of n (the total number of elements).
//!
//! # Complexities
//!
//! | Operation  | Time       | Notes                                    |
//! |------------|------------|------------------------------------------|
//! | `push`     | O(log n)   | create a size-1 heap, then merge         |
//! | `pop_min`  | O(log n)   | scan O(log n) roots, then merge children |
//! | `peek_min` | O(log n)   | linear scan over O(log n) roots          |
//! | `merge`    | O(log n)   | binary-addition over two root lists      |
//! | `len`      | O(1)       | maintained as a field                    |
//!
//! Space: O(n) total.
//!
//! # Preconditions
//!
//! None — the heap works for any `T: Ord`.

/// A node in a binomial tree.
#[derive(Debug)]
struct Node<T> {
    key: T,
    children: Vec<Self>,
    degree: usize,
}

impl<T: Ord> Node<T> {
    /// Creates a new degree-0 node (single element, no children).
    const fn new(key: T) -> Self {
        Self {
            key,
            children: Vec::new(),
            degree: 0,
        }
    }

    /// Links `other` under `self`: `self` becomes the parent (its key must be
    /// `<=` `other.key` to preserve the min-heap property). Increments
    /// `self`'s degree.
    fn link(&mut self, other: Self) {
        debug_assert!(self.key <= other.key, "link: heap order violated");
        self.children.insert(0, other);
        self.degree += 1;
    }
}

/// Binomial heap — a mergeable min-priority queue.
///
/// The heap exposes [`BinomialHeap::push`], [`BinomialHeap::pop_min`],
/// [`BinomialHeap::peek_min`], [`BinomialHeap::merge`], [`BinomialHeap::len`],
/// and [`BinomialHeap::is_empty`].
///
/// All mutating operations are O(log n); `len` and `is_empty` are O(1);
/// `peek_min` is O(log n).
#[derive(Debug, Default)]
pub struct BinomialHeap<T: Ord> {
    /// Root list in strictly increasing order of degree.
    roots: Vec<Node<T>>,
    len: usize,
}

impl<T: Ord> BinomialHeap<T> {
    /// Creates a new empty heap.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            roots: Vec::new(),
            len: 0,
        }
    }

    /// Returns the number of elements in the heap.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the heap contains no elements.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns a reference to the minimum element, or `None` if the heap is
    /// empty.
    ///
    /// Scans the O(log n) roots; O(log n) time.
    #[must_use]
    pub fn peek_min(&self) -> Option<&T> {
        self.roots.iter().map(|n| &n.key).min()
    }

    /// Inserts `value` into the heap in O(log n) time.
    pub fn push(&mut self, value: T) {
        let singleton = Self {
            roots: vec![Node::new(value)],
            len: 1,
        };
        self.merge_in(singleton);
    }

    /// Removes and returns the minimum element, or `None` if the heap is
    /// empty. O(log n) time.
    pub fn pop_min(&mut self) -> Option<T> {
        if self.roots.is_empty() {
            return None;
        }

        // Find the index of the root with the minimum key.
        let min_idx = self
            .roots
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.key.cmp(&b.key))
            .map(|(i, _)| i)?;

        // Remove that root from the list.
        let min_node = self.roots.remove(min_idx);

        // The children of the removed root form a valid binomial heap when
        // reversed (their degrees are order-1, order-2, …, 0 — reversing
        // gives strictly increasing degrees 0, 1, …, order-1).
        let mut ch = min_node.children;
        ch.reverse();
        let children_count: usize = ch.iter().map(subtree_size).sum();
        let children_heap = Self {
            roots: ch,
            len: children_count,
        };

        self.len -= 1 + children_count;
        self.merge_in(children_heap);

        Some(min_node.key)
    }

    /// Merges `other` into `self`, consuming `other`. O(log n) time.
    ///
    /// After the call `other` is empty; all elements formerly in `other` are
    /// now in `self`.
    pub fn merge(&mut self, other: Self) {
        self.merge_in(other);
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Core merge: appends the root list of `other` to `self`, then runs the
    /// binary-addition pass to restore the invariant that every degree appears
    /// at most once.
    fn merge_in(&mut self, other: Self) {
        self.len += other.len;

        // Concatenate and sort by degree so equal-degree pairs are adjacent.
        self.roots.extend(other.roots);
        self.roots.sort_by_key(|n| n.degree);

        // Walk forward; when two consecutive trees share the same degree, link
        // the one with the larger root under the one with the smaller root
        // (binary-carry). A three-way tie is handled by leaving the first
        // in place and linking the latter two.
        let mut i = 0;
        while i + 1 < self.roots.len() {
            if self.roots[i].degree == self.roots[i + 1].degree {
                if i + 2 < self.roots.len() && self.roots[i].degree == self.roots[i + 2].degree {
                    // Three-way tie: leave roots[i] in place, link [i+1] and
                    // [i+2] together.
                    let b = self.roots.remove(i + 2);
                    let b_key_smaller = b.key < self.roots[i + 1].key;
                    if b_key_smaller {
                        let mut winner = b;
                        let loser = self.roots.remove(i + 1);
                        winner.link(loser);
                        self.roots.insert(i + 1, winner);
                    } else {
                        self.roots[i + 1].link(b);
                    }
                    // roots[i] stays; roots[i+1] is now degree d+1.
                    i += 1;
                } else {
                    // Two-way tie: link them.
                    let b = self.roots.remove(i + 1);
                    let b_key_smaller = b.key < self.roots[i].key;
                    if b_key_smaller {
                        let mut winner = b;
                        let loser = self.roots.remove(i);
                        winner.link(loser);
                        self.roots.insert(i, winner);
                    } else {
                        self.roots[i].link(b);
                    }
                    // roots[i] is now degree d+1; re-examine without advancing.
                }
            } else {
                i += 1;
            }
        }
    }
}

/// Counts the total number of nodes in the subtree rooted at `node`.
fn subtree_size<T>(node: &Node<T>) -> usize {
    1 + node.children.iter().map(subtree_size).sum::<usize>()
}

#[cfg(test)]
mod tests {
    use super::BinomialHeap;
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;

    use quickcheck::TestResult;
    use quickcheck_macros::quickcheck;

    // ------------------------------------------------------------------
    // Basic / unit tests
    // ------------------------------------------------------------------

    #[test]
    fn empty_heap_peek_and_pop_return_none() {
        let mut h: BinomialHeap<i32> = BinomialHeap::new();
        assert!(h.is_empty());
        assert_eq!(h.len(), 0);
        assert_eq!(h.peek_min(), None);
        assert_eq!(h.pop_min(), None);
    }

    #[test]
    fn single_push_then_peek_and_pop() {
        let mut h = BinomialHeap::new();
        h.push(42_i32);
        assert_eq!(h.len(), 1);
        assert!(!h.is_empty());
        assert_eq!(h.peek_min(), Some(&42));
        assert_eq!(h.pop_min(), Some(42));
        assert!(h.is_empty());
        assert_eq!(h.pop_min(), None);
    }

    #[test]
    fn push_1_to_100_pop_yields_sorted_order() {
        let mut h = BinomialHeap::new();
        for i in 1..=100_i32 {
            h.push(i);
        }
        assert_eq!(h.len(), 100);
        for expected in 1..=100_i32 {
            assert_eq!(h.pop_min(), Some(expected), "expected {expected}");
        }
        assert!(h.is_empty());
    }

    #[test]
    fn push_100_to_1_pop_yields_sorted_order() {
        let mut h = BinomialHeap::new();
        for i in (1..=100_i32).rev() {
            h.push(i);
        }
        for expected in 1..=100_i32 {
            assert_eq!(h.pop_min(), Some(expected));
        }
        assert!(h.is_empty());
    }

    #[test]
    fn merge_two_heaps_yields_union_in_order() {
        let mut h1 = BinomialHeap::new();
        let mut h2 = BinomialHeap::new();
        for i in (1..=10_i32).step_by(2) {
            h1.push(i); // 1 3 5 7 9
        }
        for i in (2..=10_i32).step_by(2) {
            h2.push(i); // 2 4 6 8 10
        }
        h1.merge(h2);
        assert_eq!(h1.len(), 10);
        for expected in 1..=10_i32 {
            assert_eq!(h1.pop_min(), Some(expected));
        }
        assert!(h1.is_empty());
    }

    #[test]
    fn alternating_push_pop_preserves_correctness() {
        let mut h = BinomialHeap::new();
        let mut std_min: BinaryHeap<Reverse<i32>> = BinaryHeap::new();

        let ops: &[(bool, i32)] = &[
            (true, 5),
            (true, 3),
            (false, 0),
            (true, 8),
            (true, 1),
            (false, 0),
            (true, 7),
            (false, 0),
            (true, 2),
            (false, 0),
            (false, 0),
            (false, 0),
        ];

        for &(is_push, val) in ops {
            if is_push {
                h.push(val);
                std_min.push(Reverse(val));
            } else {
                let ours = h.pop_min();
                let theirs = std_min.pop().map(|Reverse(v)| v);
                assert_eq!(ours, theirs);
            }
        }
    }

    #[test]
    fn large_random_cross_check_against_std_binary_heap() {
        // Deterministic pseudo-random sequence via a simple LCG.
        let mut state: u64 = 0xdead_beef_cafe_babe;
        let lcg = |s: &mut u64| -> i32 {
            *s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((*s >> 33) & 0xffff) as i32
        };

        let mut our_heap: BinomialHeap<i32> = BinomialHeap::new();
        let mut std_heap: BinaryHeap<Reverse<i32>> = BinaryHeap::new();

        for _ in 0..512 {
            let v = lcg(&mut state);
            our_heap.push(v);
            std_heap.push(Reverse(v));
        }

        while !std_heap.is_empty() {
            let ours = our_heap.pop_min();
            let theirs = std_heap.pop().map(|Reverse(v)| v);
            assert_eq!(ours, theirs);
        }
        assert!(our_heap.is_empty());
    }

    // ------------------------------------------------------------------
    // QuickCheck property test
    // ------------------------------------------------------------------

    /// For a random sequence of push/pop operations, every pop from our heap
    /// must yield the same value as a pop from `BinaryHeap<Reverse<i32>>`.
    #[quickcheck]
    #[allow(clippy::needless_pass_by_value)]
    fn prop_model_checked_against_std_binary_heap(ops: Vec<(bool, i32)>) -> TestResult {
        if ops.len() > 200 {
            return TestResult::discard();
        }

        let mut our_heap: BinomialHeap<i32> = BinomialHeap::new();
        let mut std_heap: BinaryHeap<Reverse<i32>> = BinaryHeap::new();

        for (is_push, val) in ops {
            if is_push {
                our_heap.push(val);
                std_heap.push(Reverse(val));
            } else {
                let ours = our_heap.pop_min();
                let theirs = std_heap.pop().map(|Reverse(v)| v);
                if ours != theirs {
                    return TestResult::failed();
                }
            }
        }

        // Drain remaining elements and compare.
        loop {
            let ours = our_heap.pop_min();
            let theirs = std_heap.pop().map(|Reverse(v)| v);
            match (ours, theirs) {
                (None, None) => break,
                (a, b) if a == b => {}
                _ => return TestResult::failed(),
            }
        }

        TestResult::passed()
    }
}
