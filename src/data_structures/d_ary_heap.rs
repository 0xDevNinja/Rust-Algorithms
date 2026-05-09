//! d-ary heap (generalised binary heap).
//!
//! A **d-ary heap** is an array-backed implicit tree in which every internal
//! node has up to `D` children. The classical [`BinaryHeap`] is the special
//! case `D = 2`. Increasing the arity flattens the tree (height becomes
//! `log_D(n)`), which trades fewer key comparisons on `push` for more
//! comparisons per `pop` (each sift-down step inspects `D` siblings instead
//! of `2`).
//!
//! For a node at index `i`:
//!
//! * children occupy indices `D*i + 1 ..= D*i + D`,
//! * the parent (when `i > 0`) is at index `(i - 1) / D`.
//!
//! The implementation here is a **min-heap**: `peek` and `pop` return the
//! smallest element by `T`'s `Ord` impl. To obtain a max-heap, wrap values in
//! [`std::cmp::Reverse`].
//!
//! # Complexities
//!
//! Let `n = self.len()`.
//!
//! | Operation       | Time            | Notes                                        |
//! |-----------------|-----------------|----------------------------------------------|
//! | `new`           | O(1)            |                                              |
//! | `with_capacity` | O(1)            | allocates, does not initialise                |
//! | `len` / `is_empty` / `peek` | O(1) |                                              |
//! | `push`          | `O(log_D(n))`   | sift-up walks parent chain                   |
//! | `pop`           | `O(D · log_D(n))` | sift-down compares D children per level    |
//! | `from_vec`      | O(n)            | classic Floyd heapify                        |
//!
//! Space: O(n), no per-node overhead beyond `Vec<T>`'s own.
//!
//! # Preconditions
//!
//! `D` must be at least `2`. The constructors `debug_assert!` this; in release
//! builds `D == 0` or `D == 1` produce a logically degenerate (effectively
//! linked-list) structure but will not invoke undefined behaviour.

use core::cmp::Ordering;

/// A min-heap with compile-time arity `D` backed by a contiguous `Vec<T>`.
///
/// `D` is the maximum number of children per node. `D = 2` is the standard
/// binary heap; `D = 4` and `D = 8` are common choices for cache-friendly
/// priority queues.
///
/// Construct with [`DAryHeap::new`], [`DAryHeap::with_capacity`], or
/// [`DAryHeap::from_vec`]. Use [`DAryHeap::push`] / [`DAryHeap::pop`] to
/// insert / extract the minimum.
#[derive(Debug, Clone)]
pub struct DAryHeap<T: Ord, const D: usize> {
    data: Vec<T>,
}

impl<T: Ord, const D: usize> Default for DAryHeap<T, D> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Ord, const D: usize> DAryHeap<T, D> {
    /// Creates an empty heap.
    ///
    /// `D` must be at least `2` (debug-checked).
    #[must_use]
    pub fn new() -> Self {
        debug_assert!(D >= 2, "DAryHeap requires arity D >= 2");
        Self { data: Vec::new() }
    }

    /// Creates an empty heap with space pre-reserved for at least `capacity`
    /// elements.
    ///
    /// `D` must be at least `2` (debug-checked).
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        debug_assert!(D >= 2, "DAryHeap requires arity D >= 2");
        Self {
            data: Vec::with_capacity(capacity),
        }
    }

    /// Builds a heap by taking ownership of `data` and running Floyd's
    /// linear-time heapify.
    ///
    /// Runs in O(n) time, faster than `n` separate `push` calls (which would
    /// take `O(n log_D(n))`).
    ///
    /// `D` must be at least `2` (debug-checked).
    #[must_use]
    pub fn from_vec(data: Vec<T>) -> Self {
        debug_assert!(D >= 2, "DAryHeap requires arity D >= 2");
        let mut heap = Self { data };
        // Every index >= first_leaf is a leaf and trivially heap-ordered.
        // The first leaf in a d-ary heap is at index ceil((n - 1) / D); the
        // last internal node is therefore at index (n - 2) / D when n > 1.
        let n = heap.data.len();
        if n > 1 {
            // Sift down every internal node, last-to-first.
            let last_internal = (n - 2) / D;
            for i in (0..=last_internal).rev() {
                heap.sift_down(i);
            }
        }
        heap
    }

    /// Returns the number of elements in the heap.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns `true` if the heap is empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Returns a reference to the minimum element, or `None` if empty.
    #[must_use]
    pub fn peek(&self) -> Option<&T> {
        self.data.first()
    }

    /// Inserts `value` into the heap.
    ///
    /// Runs in `O(log_D(n))` time: the new element is appended to the end of
    /// the backing vector and then sifted up along its parent chain.
    pub fn push(&mut self, value: T) {
        self.data.push(value);
        let last = self.data.len() - 1;
        self.sift_up(last);
    }

    /// Removes and returns the minimum element, or `None` if empty.
    ///
    /// Runs in `O(D · log_D(n))` time.
    pub fn pop(&mut self) -> Option<T> {
        let n = self.data.len();
        if n == 0 {
            return None;
        }
        if n == 1 {
            return self.data.pop();
        }
        // Swap root with last element, pop, then sift the new root down.
        let last = n - 1;
        self.data.swap(0, last);
        let min = self.data.pop();
        self.sift_down(0);
        min
    }

    // ----------------------------------------------------------------------
    // Internal helpers
    // ----------------------------------------------------------------------

    /// Index of the parent of `i`. Caller must ensure `i > 0`.
    #[inline]
    const fn parent(i: usize) -> usize {
        (i - 1) / D
    }

    /// Index of the first child of `i`.
    #[inline]
    const fn first_child(i: usize) -> usize {
        D * i + 1
    }

    /// Restores the heap invariant by walking `idx` up its parent chain.
    fn sift_up(&mut self, mut idx: usize) {
        while idx > 0 {
            let parent = Self::parent(idx);
            if self.data[idx] < self.data[parent] {
                self.data.swap(idx, parent);
                idx = parent;
            } else {
                break;
            }
        }
    }

    /// Restores the heap invariant by sifting `idx` down toward the leaves.
    ///
    /// At each step, finds the smallest among `idx` and its (up to `D`)
    /// children; if the smallest is a child, swaps and descends, else stops.
    fn sift_down(&mut self, mut idx: usize) {
        let n = self.data.len();
        loop {
            let first = Self::first_child(idx);
            if first >= n {
                return; // leaf
            }
            // Find the smallest child in [first, first + D) ∩ [0, n).
            let end = core::cmp::min(first + D, n);
            let mut best = first;
            for c in (first + 1)..end {
                if self.data[c].cmp(&self.data[best]) == Ordering::Less {
                    best = c;
                }
            }
            // Swap if a child is strictly smaller than the current node.
            if self.data[best].cmp(&self.data[idx]) == Ordering::Less {
                self.data.swap(idx, best);
                idx = best;
            } else {
                return;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::DAryHeap;
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;

    use quickcheck::TestResult;
    use quickcheck_macros::quickcheck;

    // ------------------------------------------------------------------
    // Empty / trivial cases
    // ------------------------------------------------------------------

    #[test]
    fn empty_heap_peek_and_pop_return_none() {
        let mut h: DAryHeap<i32, 4> = DAryHeap::new();
        assert!(h.is_empty());
        assert_eq!(h.len(), 0);
        assert_eq!(h.peek(), None);
        assert_eq!(h.pop(), None);
    }

    #[test]
    fn with_capacity_does_not_change_logical_size() {
        let h: DAryHeap<i32, 3> = DAryHeap::with_capacity(64);
        assert!(h.is_empty());
        assert_eq!(h.len(), 0);
    }

    #[test]
    fn single_element_push_then_pop() {
        let mut h: DAryHeap<i32, 4> = DAryHeap::new();
        h.push(7);
        assert_eq!(h.len(), 1);
        assert_eq!(h.peek(), Some(&7));
        assert_eq!(h.pop(), Some(7));
        assert!(h.is_empty());
        assert_eq!(h.pop(), None);
    }

    // ------------------------------------------------------------------
    // Heapify (from_vec)
    // ------------------------------------------------------------------

    #[test]
    fn from_vec_then_pop_yields_ascending_order() {
        let v = vec![5, 2, 8, 1, 9, 3];
        let mut h: DAryHeap<i32, 4> = DAryHeap::from_vec(v.clone());
        let mut popped = Vec::with_capacity(v.len());
        while let Some(x) = h.pop() {
            popped.push(x);
        }
        let mut expected = v;
        expected.sort_unstable();
        assert_eq!(popped, expected);
    }

    #[test]
    fn from_vec_empty_and_singleton() {
        let mut h: DAryHeap<i32, 4> = DAryHeap::from_vec(Vec::new());
        assert_eq!(h.pop(), None);

        let mut h: DAryHeap<i32, 4> = DAryHeap::from_vec(vec![42]);
        assert_eq!(h.pop(), Some(42));
        assert_eq!(h.pop(), None);
    }

    // ------------------------------------------------------------------
    // D = 2 cross-check against std BinaryHeap<Reverse<_>>
    // ------------------------------------------------------------------

    #[test]
    fn d2_matches_std_binary_heap_with_reverse() {
        // Deterministic LCG to keep test reproducible.
        let mut state: u64 = 0x0123_4567_89ab_cdef;
        let lcg = |s: &mut u64| -> i32 {
            *s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((*s >> 33) & 0xffff) as i32
        };

        let mut ours: DAryHeap<i32, 2> = DAryHeap::new();
        let mut theirs: BinaryHeap<Reverse<i32>> = BinaryHeap::new();
        for _ in 0..500 {
            let v = lcg(&mut state);
            ours.push(v);
            theirs.push(Reverse(v));
        }
        assert_eq!(ours.len(), theirs.len());

        while !theirs.is_empty() {
            let a = ours.pop();
            let b = theirs.pop().map(|Reverse(v)| v);
            assert_eq!(a, b);
        }
        assert!(ours.is_empty());
    }

    // ------------------------------------------------------------------
    // D = 4 / D = 8: random vs slice::sort
    // ------------------------------------------------------------------

    fn drain<T: Ord, const D: usize>(mut h: DAryHeap<T, D>) -> Vec<T> {
        let mut out = Vec::with_capacity(h.len());
        while let Some(x) = h.pop() {
            out.push(x);
        }
        out
    }

    #[test]
    fn d4_random_pop_order_matches_sorted() {
        let mut state: u64 = 0xdead_beef_cafe_babe;
        let lcg = |s: &mut u64| -> i32 {
            *s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((*s >> 33) & 0xffff) as i32
        };

        let mut input = Vec::with_capacity(1024);
        let mut h: DAryHeap<i32, 4> = DAryHeap::new();
        for _ in 0..1024 {
            let v = lcg(&mut state);
            input.push(v);
            h.push(v);
        }
        let mut expected = input;
        expected.sort_unstable();
        assert_eq!(drain(h), expected);
    }

    #[test]
    fn d8_random_pop_order_matches_sorted() {
        let mut state: u64 = 0xfeed_face_dead_beef;
        let lcg = |s: &mut u64| -> i32 {
            *s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((*s >> 33) & 0xffff) as i32
        };

        let mut input = Vec::with_capacity(1024);
        let mut h: DAryHeap<i32, 8> = DAryHeap::new();
        for _ in 0..1024 {
            let v = lcg(&mut state);
            input.push(v);
            h.push(v);
        }
        let mut expected = input;
        expected.sort_unstable();
        assert_eq!(drain(h), expected);
    }

    #[test]
    fn d4_from_vec_random_matches_sorted() {
        let mut state: u64 = 0x1234_5678_9abc_def0;
        let lcg = |s: &mut u64| -> i32 {
            *s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((*s >> 33) & 0xffff) as i32
        };

        let mut input = Vec::with_capacity(777);
        for _ in 0..777 {
            input.push(lcg(&mut state));
        }
        let h: DAryHeap<i32, 4> = DAryHeap::from_vec(input.clone());
        let mut expected = input;
        expected.sort_unstable();
        assert_eq!(drain(h), expected);
    }

    // ------------------------------------------------------------------
    // QuickCheck property test (D = 4)
    // ------------------------------------------------------------------

    /// For any sequence of `(is_push, value)` ops with `D = 4`, the heap must
    /// agree element-for-element with a `BinaryHeap<Reverse<i32>>` model.
    #[quickcheck]
    #[allow(clippy::needless_pass_by_value)]
    fn prop_d4_matches_std_binary_heap(ops: Vec<(bool, i32)>) -> TestResult {
        if ops.len() > 200 {
            return TestResult::discard();
        }
        let mut ours: DAryHeap<i32, 4> = DAryHeap::new();
        let mut theirs: BinaryHeap<Reverse<i32>> = BinaryHeap::new();
        for (is_push, val) in ops {
            if is_push {
                ours.push(val);
                theirs.push(Reverse(val));
                if ours.peek() != theirs.peek().map(|Reverse(v)| v) {
                    return TestResult::failed();
                }
            } else {
                let a = ours.pop();
                let b = theirs.pop().map(|Reverse(v)| v);
                if a != b {
                    return TestResult::failed();
                }
            }
            if ours.len() != theirs.len() {
                return TestResult::failed();
            }
        }
        // Drain.
        loop {
            match (ours.pop(), theirs.pop().map(|Reverse(v)| v)) {
                (None, None) => break,
                (a, b) if a == b => {}
                _ => return TestResult::failed(),
            }
        }
        TestResult::passed()
    }
}
