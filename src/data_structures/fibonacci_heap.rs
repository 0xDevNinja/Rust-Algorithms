//! Fibonacci heap — Fredman & Tarjan, 1987.
//!
//! A **mergeable heap** that achieves better amortised complexity than both
//! binary and binomial heaps by deferring consolidation work until `pop_min`.
//!
//! # Amortised complexity
//!
//! | Operation      | Amortised cost |
//! |----------------|----------------|
//! | `push`         | **O(1)**       |
//! | `peek_min`     | **O(1)**       |
//! | `merge`        | **O(n)**¹      |
//! | `pop_min`      | **O(log n)**   |
//! | `decrease_key` | **O(1)**       |
//!
//! ¹ The canonical Fibonacci-heap `merge` is O(1) when nodes are kept as
//! pointers.  Here we **re-index** `other`'s nodes into `self.nodes` during
//! merge, which costs O(|other|) time.  That linear-time relabelling is the
//! unavoidable price of the safe-Rust arena representation (index stability
//! requires a single backing `Vec`).  For workloads that chain many merges, a
//! `Rc<RefCell<…>>` representation trades implementation ease for the O(1)
//! pointer-splicing bound.
//!
//! # Why Fibonacci heaps?
//!
//! They are the asymptotically-optimal mergeable heap: the O(1) amortised
//! `decrease_key` drives Dijkstra's shortest-path algorithm to
//! **O(E + V log V)** (vs O(E log V) with a binary heap).  The trade-off vs
//! binomial heaps is large hidden constants — cascading-cut and consolidate
//! bookkeeping make each operation much slower in practice; Fibonacci heaps
//! pay off only for large, dense graphs where `decrease_key` dominates.
//!
//! # Preconditions
//!
//! - `decrease_key` requires `new_key ≤ old_key`; it returns `Err` otherwise.
//! - A [`Handle`] stays valid until `pop_min` extracts the corresponding node;
//!   stale handles are detected via the `active` flag and cause `decrease_key`
//!   to return `Err`.
//!
//! # Implementation notes
//!
//! Nodes live in a `Vec<Node<T>>` arena.  Each node's `left`/`right` fields
//! are `usize` indices that form a **doubly-linked circular sibling list**.
//! `parent` and `child` are `Option<usize>`.  The key is stored as
//! `Option<T>` so that `pop_min` can safely *move* the minimum out without
//! `unsafe` or a `Default` bound.  After extraction the slot's key is `None`
//! and `active` is `false`; the slot is never reclaimed (arenas are simple
//! here — no free-list, because `Handle` indices must remain stable).

use std::cmp::Ordering;

// ── Internal node ─────────────────────────────────────────────────────────────

struct Node<T> {
    /// `None` only after the node has been extracted by `pop_min`.
    key: Option<T>,
    parent: Option<usize>,
    child: Option<usize>,
    left: usize,
    right: usize,
    degree: usize,
    marked: bool,
    /// `false` once extracted; used to detect stale handles.
    active: bool,
}

// ── Public handle ─────────────────────────────────────────────────────────────

/// Opaque index returned by [`FibonacciHeap::push`]; required by
/// [`FibonacciHeap::decrease_key`].
///
/// A handle becomes **stale** after the corresponding key has been removed via
/// [`FibonacciHeap::pop_min`].  Passing a stale handle to `decrease_key`
/// returns `Err`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Handle(usize);

// ── Fibonacci heap ────────────────────────────────────────────────────────────

/// Min-oriented Fibonacci heap generic over `T: Ord`.
///
/// See the [module documentation](self) for amortised complexity, design
/// rationale, and caveats.
pub struct FibonacciHeap<T> {
    nodes: Vec<Node<T>>,
    min: Option<usize>,
    len: usize,
}

impl<T: Ord> FibonacciHeap<T> {
    // ── Constructors ──────────────────────────────────────────────────────────

    /// Creates an empty heap.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            nodes: Vec::new(),
            min: None,
            len: 0,
        }
    }

    // ── Accessors ─────────────────────────────────────────────────────────────

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

    /// Returns a reference to the minimum key, or `None` if the heap is empty.
    ///
    /// # O(1)
    #[must_use]
    pub fn peek_min(&self) -> Option<&T> {
        self.min.and_then(|i| self.nodes[i].key.as_ref())
    }

    // ── Mutating operations ───────────────────────────────────────────────────

    /// Inserts `key` into the heap and returns a [`Handle`] usable with
    /// [`decrease_key`](Self::decrease_key).
    ///
    /// # Amortised O(1)
    pub fn push(&mut self, key: T) -> Handle {
        let idx = self.nodes.len();
        self.nodes.push(Node {
            key: Some(key),
            parent: None,
            child: None,
            left: idx,
            right: idx,
            degree: 0,
            marked: false,
            active: true,
        });
        // Append to root list and update min pointer.
        self.root_list_insert(idx);
        self.len += 1;
        Handle(idx)
    }

    /// Removes and returns the minimum key, or `None` if the heap is empty.
    ///
    /// # Amortised O(log n)
    pub fn pop_min(&mut self) -> Option<T> {
        let min_idx = self.min?;

        // Save the right-neighbour before modifying the ring.
        // If min_idx is the sole root its right == itself.
        let next_root = self.nodes[min_idx].right;
        let was_sole_root = next_root == min_idx;

        // 1. Remove min from the root list (adjusts neighbours; min_idx's own
        //    left/right still point at old neighbours — that's intentional).
        self.ring_detach(min_idx);

        // Update the root-list anchor so subsequent splices have a valid target.
        self.min = if was_sole_root { None } else { Some(next_root) };

        // 2. Splice min's children into the root list.
        if let Some(first_child) = self.nodes[min_idx].child {
            // Snapshot the ring before we modify any pointers.
            let children: Vec<usize> = self.ring_collect(first_child);
            for c in children {
                self.nodes[c].parent = None;
                self.nodes[c].marked = false;
                // Detach c from the (already snapshotted) child ring.
                self.ring_detach(c);
                // Make c a self-loop, then insert into root list.
                self.nodes[c].left = c;
                self.nodes[c].right = c;
                self.root_list_insert(c);
            }
            self.nodes[min_idx].child = None;
        }

        // 3. Mark as extracted.
        self.nodes[min_idx].active = false;
        self.len -= 1;

        // 4. Consolidate (no-op if root list is empty).
        if self.min.is_some() {
            self.consolidate();
        }

        self.nodes[min_idx].key.take()
    }

    /// Merges `other` into `self`, consuming `other`.
    ///
    /// All of `other`'s nodes are re-indexed (shifted by `self.nodes.len()`)
    /// and appended to `self.nodes`.  This is **O(|other|)** — see module docs.
    pub fn merge(&mut self, other: Self) {
        if other.len == 0 {
            return;
        }
        let offset = self.nodes.len();
        let other_min = other.min;
        let other_len = other.len;

        // Remap every index field in other's nodes before extending our arena.
        let remapped: Vec<Node<T>> = other
            .nodes
            .into_iter()
            .map(|mut n| {
                n.left += offset;
                n.right += offset;
                n.parent = n.parent.map(|p| p + offset);
                n.child = n.child.map(|c| c + offset);
                n
            })
            .collect();

        self.nodes.extend(remapped);
        self.len += other_len;

        let Some(other_min_idx) = other_min else {
            return; // other had no live roots (shouldn't happen given len>0, but guard anyway)
        };
        let other_min_remapped = other_min_idx + offset;

        match self.min {
            None => {
                // self was empty; adopt other's root list wholesale.
                self.min = Some(other_min_remapped);
            }
            Some(self_min) => {
                // Splice other's root ring between self_min and its left neighbour.
                //
                // Before:
                //   self  ring: … <-> self_left <-> self_min <-> …
                //   other ring: … <-> other_tail <-> other_head <-> …
                //     (other_head = other_min_remapped, its right = other_head's right)
                //
                // We insert the entire other ring between self_left and self_min.
                let self_left = self.nodes[self_min].left;
                // Save other's "tail" (the node immediately to the left of other_head)
                // BEFORE we touch any pointers.
                let other_tail = self.nodes[other_min_remapped].left;

                // Cross-link:
                self.nodes[self_left].right = other_min_remapped;
                self.nodes[other_min_remapped].left = self_left;
                self.nodes[other_tail].right = self_min;
                self.nodes[self_min].left = other_tail;

                // Update min pointer if other's min is smaller.
                let self_key = self.nodes[self_min].key.as_ref().expect("active");
                let other_key = self.nodes[other_min_remapped].key.as_ref().expect("active");
                if other_key < self_key {
                    self.min = Some(other_min_remapped);
                }
            }
        }
    }

    /// Decreases the key held by handle `h` to `new_key`.
    ///
    /// Returns `Err` if:
    /// - `h` refers to a node that has already been extracted (`pop_min`), or
    /// - `new_key > old_key` (increasing would violate the min-heap property).
    ///
    /// # Amortised O(1)
    pub fn decrease_key(&mut self, h: Handle, new_key: T) -> Result<(), &'static str> {
        let idx = h.0;
        if idx >= self.nodes.len() || !self.nodes[idx].active {
            return Err("stale handle: node has already been extracted");
        }
        {
            let old_key = self.nodes[idx].key.as_ref().expect("active node has key");
            if new_key.cmp(old_key) == Ordering::Greater {
                return Err("new_key must be less than or equal to old_key");
            }
        }
        self.nodes[idx].key = Some(new_key);

        let parent = self.nodes[idx].parent;
        if let Some(p) = parent {
            let violates = {
                let node_key = self.nodes[idx].key.as_ref().expect("just set");
                let parent_key = self.nodes[p].key.as_ref().expect("parent is active");
                node_key < parent_key
            };
            if violates {
                self.cut(idx, p);
                self.cascading_cut(p);
            }
        }

        // Update global min if this node (now a root) has a smaller key.
        if self.nodes[idx].parent.is_none() {
            match self.min {
                None => self.min = Some(idx),
                Some(m) => {
                    let cur_min_key = self.nodes[m].key.as_ref().expect("min is active");
                    let node_key = self.nodes[idx].key.as_ref().expect("active");
                    if node_key < cur_min_key {
                        self.min = Some(idx);
                    }
                }
            }
        }

        Ok(())
    }

    // ── Internal helpers ──────────────────────────────────────────────────────

    /// Splices node `idx` (already a self-loop) into the root list to the left
    /// of `self.min`.  Also updates `self.min` when the list was previously
    /// empty or when `idx`'s key is smaller.
    fn root_list_insert(&mut self, idx: usize) {
        match self.min {
            None => {
                // idx becomes the sole root; it's already a self-loop.
                self.min = Some(idx);
            }
            Some(m) => {
                // Insert idx to the left of m.
                let left_of_m = self.nodes[m].left;
                self.nodes[idx].right = m;
                self.nodes[idx].left = left_of_m;
                self.nodes[left_of_m].right = idx;
                self.nodes[m].left = idx;
                // Update min pointer.
                let m_key = self.nodes[m].key.as_ref().expect("active root");
                let new_key = self.nodes[idx].key.as_ref().expect("just inserted");
                if new_key < m_key {
                    self.min = Some(idx);
                }
            }
        }
    }

    /// Removes `idx` from its sibling ring by relinking its neighbours.
    /// Does NOT clear `idx`'s own `left`/`right` pointers.
    fn ring_detach(&mut self, idx: usize) {
        let l = self.nodes[idx].left;
        let r = self.nodes[idx].right;
        self.nodes[l].right = r;
        self.nodes[r].left = l;
    }

    /// Collects all node indices in the circular ring starting at `start`
    /// (traversing right links).  Does not mutate any pointers.
    fn ring_collect(&self, start: usize) -> Vec<usize> {
        let mut ring = Vec::new();
        let mut cur = start;
        loop {
            ring.push(cur);
            cur = self.nodes[cur].right;
            if cur == start {
                break;
            }
        }
        ring
    }

    /// Consolidates the root list so that no two roots share a degree.
    ///
    /// Algorithm: for each root x, while the degree table already holds a root
    /// y with the same degree, link the larger-key one under the smaller-key
    /// one (increasing the winner's degree by 1).  Rebuild the root list from
    /// the degree table and find the new minimum.
    ///
    /// Max degree bound: for n nodes, Fibonacci-heap max degree ≤ 1.44 log₂ n.
    /// 64 slots covers n up to ~2^44, more than sufficient.
    fn consolidate(&mut self) {
        let Some(start) = self.min else { return };

        // Snapshot the root list before we start modifying links.
        let roots: Vec<usize> = self.ring_collect(start);

        // Degree table: slot d holds the unique root of degree d found so far.
        let max_degree: usize = 64;
        let mut degree_table: Vec<Option<usize>> = vec![None; max_degree];

        for &r in &roots {
            // Isolate r as a self-loop; consolidate will re-link everything.
            self.nodes[r].left = r;
            self.nodes[r].right = r;

            let mut x = r;
            let mut d = self.nodes[x].degree;

            loop {
                if d >= max_degree {
                    // Should never happen for any realistic n.
                    break;
                }
                match degree_table[d] {
                    None => {
                        degree_table[d] = Some(x);
                        break;
                    }
                    Some(y) => {
                        // We need x to be the one with the smaller key.
                        let (winner, loser) = {
                            let xk = self.nodes[x].key.as_ref().expect("root active");
                            let yk = self.nodes[y].key.as_ref().expect("root active");
                            if xk <= yk {
                                (x, y)
                            } else {
                                (y, x)
                            }
                        };
                        // Link loser under winner.
                        self.link(loser, winner);
                        degree_table[d] = None;
                        d += 1;
                        x = winner;
                    }
                }
            }
        }

        // Rebuild root list and find new minimum.
        self.min = None;
        for slot in &degree_table {
            if let Some(&idx) = slot.as_ref() {
                // idx is already a self-loop from the isolation step above
                // (or it was the winner of a series of links, in which case
                // `link` ensured its left/right are intact within the child list
                // but we already isolated it at the start — actually we need
                // to verify: if x was involved in a link, its left/right were
                // last set during the isolation step, and link() only touches
                // the child's pointers.  So x itself stays isolated.  Good.)
                self.nodes[idx].parent = None;
                self.root_list_insert(idx);
            }
        }
    }

    /// Adds `child_idx` as a child of `root_idx`, incrementing `root_idx.degree`.
    ///
    /// Precondition: `root_idx.key ≤ child_idx.key`.
    fn link(&mut self, child_idx: usize, root_idx: usize) {
        self.nodes[child_idx].parent = Some(root_idx);
        self.nodes[child_idx].marked = false;

        match self.nodes[root_idx].child {
            None => {
                // child_idx becomes its own self-loop child ring.
                self.nodes[child_idx].left = child_idx;
                self.nodes[child_idx].right = child_idx;
                self.nodes[root_idx].child = Some(child_idx);
            }
            Some(existing) => {
                // Insert child_idx to the left of `existing` in the child ring.
                let left_of_existing = self.nodes[existing].left;
                self.nodes[child_idx].right = existing;
                self.nodes[child_idx].left = left_of_existing;
                self.nodes[left_of_existing].right = child_idx;
                self.nodes[existing].left = child_idx;
            }
        }
        self.nodes[root_idx].degree += 1;
    }

    /// Cuts `idx` from its parent `parent_idx` and moves it to the root list.
    fn cut(&mut self, idx: usize, parent_idx: usize) {
        let l = self.nodes[idx].left;
        let r = self.nodes[idx].right;

        if l == idx {
            // idx was the only child.
            self.nodes[parent_idx].child = None;
        } else {
            self.nodes[l].right = r;
            self.nodes[r].left = l;
            if self.nodes[parent_idx].child == Some(idx) {
                self.nodes[parent_idx].child = Some(r);
            }
        }
        self.nodes[parent_idx].degree -= 1;

        // Move idx to root list as a self-loop.
        self.nodes[idx].parent = None;
        self.nodes[idx].marked = false;
        self.nodes[idx].left = idx;
        self.nodes[idx].right = idx;
        self.root_list_insert(idx);
    }

    /// Iteratively walks up the parent chain from `idx`, cutting each already-
    /// marked ancestor until an unmarked one (which gets marked) or a root.
    fn cascading_cut(&mut self, idx: usize) {
        let mut cur = idx;
        while let Some(parent) = self.nodes[cur].parent {
            if self.nodes[cur].marked {
                self.cut(cur, parent);
                cur = parent;
            } else {
                self.nodes[cur].marked = true;
                break;
            }
        }
    }
}

impl<T: Ord> Default for FibonacciHeap<T> {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::{FibonacciHeap, Handle};
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;

    // ── Basic sanity ──────────────────────────────────────────────────────────

    #[test]
    fn empty_heap() {
        let h: FibonacciHeap<i32> = FibonacciHeap::new();
        assert!(h.is_empty());
        assert_eq!(h.len(), 0);
        assert_eq!(h.peek_min(), None);
    }

    #[test]
    fn pop_from_empty_returns_none() {
        let mut h: FibonacciHeap<i32> = FibonacciHeap::new();
        assert_eq!(h.pop_min(), None);
        assert_eq!(h.len(), 0);
    }

    #[test]
    fn single_push_then_pop() {
        let mut h = FibonacciHeap::new();
        h.push(42i32);
        assert_eq!(h.len(), 1);
        assert!(!h.is_empty());
        assert_eq!(h.peek_min(), Some(&42));
        assert_eq!(h.pop_min(), Some(42));
        assert!(h.is_empty());
        assert_eq!(h.peek_min(), None);
        assert_eq!(h.pop_min(), None);
    }

    // ── Ordered extraction ────────────────────────────────────────────────────

    #[test]
    fn push_1_to_100_pop_ascending() {
        let mut h = FibonacciHeap::new();
        for i in 1i32..=100 {
            h.push(i);
        }
        assert_eq!(h.len(), 100);
        for expected in 1i32..=100 {
            assert_eq!(
                h.pop_min(),
                Some(expected),
                "mismatch at expected={expected}"
            );
        }
        assert!(h.is_empty());
    }

    #[test]
    fn push_reversed_pops_ascending() {
        let mut h = FibonacciHeap::new();
        for i in (1i32..=50).rev() {
            h.push(i);
        }
        for expected in 1i32..=50 {
            assert_eq!(h.pop_min(), Some(expected));
        }
        assert!(h.is_empty());
    }

    // ── Interleaved push/pop ──────────────────────────────────────────────────

    #[test]
    fn alternating_push_pop() {
        let mut h = FibonacciHeap::new();
        h.push(5i32);
        h.push(3i32);
        h.push(7i32);
        assert_eq!(h.pop_min(), Some(3));
        assert_eq!(h.pop_min(), Some(5));
        h.push(1i32);
        h.push(4i32);
        assert_eq!(h.pop_min(), Some(1));
        assert_eq!(h.pop_min(), Some(4));
        assert_eq!(h.pop_min(), Some(7));
        assert_eq!(h.pop_min(), None);
    }

    // ── decrease_key ─────────────────────────────────────────────────────────

    #[test]
    fn decrease_key_makes_it_pop_first() {
        let mut h = FibonacciHeap::new();
        h.push(10i32);
        h.push(20i32);
        let handle = h.push(30i32);
        assert!(h.decrease_key(handle, 1).is_ok());
        assert_eq!(h.peek_min(), Some(&1));
        assert_eq!(h.pop_min(), Some(1));
        assert_eq!(h.pop_min(), Some(10));
        assert_eq!(h.pop_min(), Some(20));
    }

    #[test]
    fn decrease_key_increase_returns_err() {
        let mut h = FibonacciHeap::new();
        let handle = h.push(5i32);
        assert!(h.decrease_key(handle, 10).is_err());
    }

    #[test]
    fn decrease_key_stale_handle_returns_err() {
        let mut h = FibonacciHeap::new();
        let handle = h.push(5i32);
        h.pop_min();
        assert!(h.decrease_key(handle, 1).is_err());
    }

    #[test]
    fn decrease_key_equal_is_ok() {
        let mut h = FibonacciHeap::new();
        let handle = h.push(5i32);
        assert!(h.decrease_key(handle, 5).is_ok());
        assert_eq!(h.pop_min(), Some(5));
    }

    #[test]
    fn decrease_key_deep_child_cascading_cut() {
        // Push 16 elements and pop the minimum once to force consolidation into
        // binomial trees.  Then decrease a non-root key and verify the heap
        // remains sorted throughout.
        let mut h = FibonacciHeap::new();
        let handles: Vec<Handle> = (0i32..16).map(|i| h.push(i * 10)).collect();
        assert_eq!(h.pop_min(), Some(0)); // triggers consolidate
                                          // handles[8] → key 80; decrease to 5 (below current min 10).
        assert!(h.decrease_key(handles[8], 5).is_ok());
        assert_eq!(h.pop_min(), Some(5));
        let mut prev = 5i32;
        while let Some(v) = h.pop_min() {
            assert!(v >= prev, "out of order: {v} after {prev}");
            prev = v;
        }
    }

    // ── merge ─────────────────────────────────────────────────────────────────

    #[test]
    fn merge_two_heaps() {
        let mut a = FibonacciHeap::new();
        a.push(3i32);
        a.push(7i32);
        let mut b = FibonacciHeap::new();
        b.push(1i32);
        b.push(5i32);
        a.merge(b);
        assert_eq!(a.len(), 4);
        assert_eq!(a.pop_min(), Some(1));
        assert_eq!(a.pop_min(), Some(3));
        assert_eq!(a.pop_min(), Some(5));
        assert_eq!(a.pop_min(), Some(7));
    }

    #[test]
    fn merge_into_empty() {
        let mut a: FibonacciHeap<i32> = FibonacciHeap::new();
        let mut b = FibonacciHeap::new();
        b.push(42i32);
        a.merge(b);
        assert_eq!(a.len(), 1);
        assert_eq!(a.pop_min(), Some(42));
    }

    #[test]
    fn merge_empty_into_nonempty() {
        let mut a = FibonacciHeap::new();
        a.push(42i32);
        let b: FibonacciHeap<i32> = FibonacciHeap::new();
        a.merge(b);
        assert_eq!(a.len(), 1);
        assert_eq!(a.pop_min(), Some(42));
    }

    // ── Cross-check against BinaryHeap ────────────────────────────────────────

    #[test]
    fn large_random_sequence_vs_binary_heap() {
        let mut fib: FibonacciHeap<i32> = FibonacciHeap::new();
        let mut bh: BinaryHeap<Reverse<i32>> = BinaryHeap::new();
        // Simple deterministic pseudo-random sequence.
        let seq: Vec<i32> = (0i64..200)
            .map(|i| ((i * 1_000_003 + 7) % 997) as i32)
            .collect();
        for &v in &seq {
            fib.push(v);
            bh.push(Reverse(v));
        }
        while let (Some(fv), Some(Reverse(bv))) = (fib.pop_min(), bh.pop()) {
            assert_eq!(fv, bv, "diverged: fib gave {fv}, std BinaryHeap gave {bv}");
        }
        assert!(fib.is_empty());
        assert!(bh.is_empty());
    }

    // ── Property test ─────────────────────────────────────────────────────────

    use quickcheck_macros::quickcheck;

    /// Model-checks `FibonacciHeap` against `BinaryHeap<Reverse<i32>>` over a
    /// random mix of push and `pop_min` operations.
    #[allow(clippy::needless_pass_by_value)]
    #[quickcheck]
    fn prop_matches_binary_heap(ops: Vec<(bool, i32)>) -> bool {
        let mut fib: FibonacciHeap<i32> = FibonacciHeap::new();
        let mut oracle: BinaryHeap<Reverse<i32>> = BinaryHeap::new();

        for (is_push, val) in ops.into_iter().take(100) {
            if is_push {
                fib.push(val);
                oracle.push(Reverse(val));
            } else {
                let fib_min = fib.pop_min();
                let oracle_min = oracle.pop().map(|Reverse(v)| v);
                if fib_min != oracle_min {
                    return false;
                }
            }
        }
        // Drain both and verify the sequences match.
        loop {
            let fv = fib.pop_min();
            let bv = oracle.pop().map(|Reverse(v)| v);
            match (fv, bv) {
                (None, None) => return true,
                (a, b) if a != b => return false,
                _ => {}
            }
        }
    }
}
