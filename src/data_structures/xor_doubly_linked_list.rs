//! XOR doubly-linked list (memory-efficient deque).
//!
//! A classic XOR linked list collapses the two pointers of a doubly-linked
//! list into a single field `npx = prev ^ next`. Knowing one neighbour's
//! address lets you recover the other via XOR, so the list still walks both
//! directions but stores only one link per node.
//!
//! This implementation uses a `Vec<Node<T>>` arena instead of raw pointers,
//! so it is **fully safe** (no `unsafe`). Each node holds the XOR of its two
//! neighbours' arena *indices*. The sentinel for "null" is `usize::MAX`,
//! which behaves like a null pointer would: `idx ^ usize::MAX` flips every
//! bit so endpoints recover the lone real neighbour.
//!
//! Live nodes wrap their value in `Option<T>` so that vacated slots can hand
//! ownership of `T` back to the caller without requiring `T: Default` and
//! without `unsafe`. Vacated indices are pushed onto a free-list and reused
//! by the next allocation, keeping the arena compact under churn.
//!
//! # Layout
//! - Arena: `Vec<Node<T>>`, slots reused via a `free` stack of vacated indices.
//! - `head` / `tail`: indices of the first / last node, or `usize::MAX` if empty.
//! - `Node::npx`: `prev_idx ^ next_idx` (with `usize::MAX` for absent ends).
//!
//! # Complexity
//! - `push_front`, `push_back`, `pop_front`, `pop_back`: **O(1)** amortised.
//! - `len`, `is_empty`: **O(1)**.
//! - `iter`: **O(1)** to construct, **O(n)** to drain.
//!
//! # Caveat
//! Because slots are reused from the `free` stack, **arena indices are not
//! stable** across pops; do not rely on them externally.

/// Sentinel index meaning "no neighbour" (analogue of a null pointer).
const NIL: usize = usize::MAX;

/// Internal arena node. Stores the value and the XOR of neighbour indices.
///
/// The value lives in an `Option<T>` so that pop operations can move the
/// `T` out without an `unsafe` block and without requiring `T: Default`.
struct Node<T> {
    value: Option<T>,
    /// `prev_idx ^ next_idx`, with `NIL` for absent neighbours.
    npx: usize,
}

/// A memory-efficient doubly-linked list using XOR-encoded neighbour links.
///
/// Backed by a `Vec<Node<T>>` arena, so it is safe Rust. Operations at both
/// ends are O(1) amortised; iteration is O(n) and walks the list forward.
pub struct XorList<T> {
    arena: Vec<Node<T>>,
    free: Vec<usize>,
    head: usize,
    tail: usize,
    len: usize,
}

impl<T> XorList<T> {
    /// Creates an empty list.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            arena: Vec::new(),
            free: Vec::new(),
            head: NIL,
            tail: NIL,
            len: 0,
        }
    }

    /// Returns the number of elements currently stored. **O(1)**.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the list contains no elements. **O(1)**.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Allocates a slot, reusing a vacated index if one exists.
    fn alloc(&mut self, value: T, npx: usize) -> usize {
        let node = Node {
            value: Some(value),
            npx,
        };
        if let Some(idx) = self.free.pop() {
            self.arena[idx] = node;
            idx
        } else {
            self.arena.push(node);
            self.arena.len() - 1
        }
    }

    /// Inserts `value` at the front of the list. **O(1)** amortised.
    pub fn push_front(&mut self, value: T) {
        // New head: prev = NIL, next = old head, so npx = NIL ^ old_head.
        let new_idx = self.alloc(value, NIL ^ self.head);
        if self.head == NIL {
            // List was empty: this node is both head and tail.
            self.tail = new_idx;
        } else {
            // Old head's npx was (NIL ^ old_next); becomes (new_idx ^ old_next).
            let old_head = self.head;
            self.arena[old_head].npx ^= NIL ^ new_idx;
        }
        self.head = new_idx;
        self.len += 1;
    }

    /// Inserts `value` at the back of the list. **O(1)** amortised.
    pub fn push_back(&mut self, value: T) {
        // New tail: prev = old tail, next = NIL, so npx = old_tail ^ NIL.
        let new_idx = self.alloc(value, self.tail ^ NIL);
        if self.tail == NIL {
            self.head = new_idx;
        } else {
            // Old tail's npx was (old_prev ^ NIL); becomes (old_prev ^ new_idx).
            let old_tail = self.tail;
            self.arena[old_tail].npx ^= NIL ^ new_idx;
        }
        self.tail = new_idx;
        self.len += 1;
    }

    /// Removes and returns the front element, or `None` if empty. **O(1)**.
    pub fn pop_front(&mut self) -> Option<T> {
        if self.head == NIL {
            return None;
        }
        let head_idx = self.head;
        // head's npx = NIL ^ next, so next = npx ^ NIL.
        let next_idx = self.arena[head_idx].npx ^ NIL;
        if next_idx == NIL {
            // Only one element.
            self.head = NIL;
            self.tail = NIL;
        } else {
            // next becomes new head: its npx was (head_idx ^ after_next),
            // becomes (NIL ^ after_next) = old_npx ^ head_idx ^ NIL.
            self.arena[next_idx].npx ^= head_idx ^ NIL;
            self.head = next_idx;
        }
        self.len -= 1;
        Some(self.recycle(head_idx))
    }

    /// Removes and returns the back element, or `None` if empty. **O(1)**.
    pub fn pop_back(&mut self) -> Option<T> {
        if self.tail == NIL {
            return None;
        }
        let tail_idx = self.tail;
        // tail's npx = prev ^ NIL, so prev = npx ^ NIL.
        let prev_idx = self.arena[tail_idx].npx ^ NIL;
        if prev_idx == NIL {
            self.head = NIL;
            self.tail = NIL;
        } else {
            // prev becomes new tail: its npx was (before_prev ^ tail_idx),
            // becomes (before_prev ^ NIL) = old_npx ^ tail_idx ^ NIL.
            self.arena[prev_idx].npx ^= tail_idx ^ NIL;
            self.tail = prev_idx;
        }
        self.len -= 1;
        Some(self.recycle(tail_idx))
    }

    /// Moves the value out of the slot and pushes the index onto the free
    /// list. The husk left behind has `value: None` and is overwritten by
    /// the next [`alloc`].
    fn recycle(&mut self, idx: usize) -> T {
        let value = self.arena[idx]
            .value
            .take()
            .expect("xor list slot already vacant");
        self.free.push(idx);
        value
    }

    /// Returns a forward iterator over the list. **O(1)** to construct.
    #[must_use]
    pub fn iter(&self) -> Iter<'_, T> {
        Iter {
            arena: &self.arena,
            current: self.head,
            prev: NIL,
            remaining: self.len,
        }
    }
}

impl<T> Default for XorList<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Forward iterator over [`XorList`]. Yields each element from head to tail.
pub struct Iter<'a, T> {
    arena: &'a [Node<T>],
    current: usize,
    prev: usize,
    remaining: usize,
}

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current == NIL {
            return None;
        }
        let node = &self.arena[self.current];
        // npx ^ prev = next.
        let next_idx = node.npx ^ self.prev;
        self.prev = self.current;
        self.current = next_idx;
        self.remaining = self.remaining.saturating_sub(1);
        Some(
            node.value
                .as_ref()
                .expect("live xor list node missing value"),
        )
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<T> ExactSizeIterator for Iter<'_, T> {}

impl<'a, T> IntoIterator for &'a XorList<T> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::XorList;

    #[test]
    fn new_is_empty() {
        let list: XorList<i32> = XorList::new();
        assert!(list.is_empty());
        assert_eq!(list.len(), 0);
    }

    #[test]
    fn default_matches_new() {
        let list: XorList<i32> = XorList::default();
        assert!(list.is_empty());
    }

    #[test]
    fn pop_on_empty_returns_none() {
        let mut list: XorList<i32> = XorList::new();
        assert!(list.pop_front().is_none());
        assert!(list.pop_back().is_none());
    }

    #[test]
    fn push_back_pop_front_is_fifo() {
        let mut list = XorList::new();
        for i in 1..=5 {
            list.push_back(i);
        }
        assert_eq!(list.len(), 5);
        for expected in 1..=5 {
            assert_eq!(list.pop_front(), Some(expected));
        }
        assert!(list.is_empty());
        assert!(list.pop_front().is_none());
    }

    #[test]
    fn push_front_pop_back_is_fifo() {
        let mut list = XorList::new();
        for i in 1..=5 {
            list.push_front(i);
        }
        for expected in 1..=5 {
            assert_eq!(list.pop_back(), Some(expected));
        }
        assert!(list.is_empty());
    }

    #[test]
    fn push_back_pop_back_is_lifo() {
        let mut list = XorList::new();
        for i in 1..=5 {
            list.push_back(i);
        }
        for expected in (1..=5).rev() {
            assert_eq!(list.pop_back(), Some(expected));
        }
    }

    #[test]
    fn mixed_push_pop_sequence() {
        let mut list = XorList::new();
        list.push_back(2);
        list.push_front(1);
        list.push_back(3);
        list.push_front(0);
        // Order should be 0, 1, 2, 3.
        assert_eq!(list.len(), 4);
        let collected: Vec<i32> = list.iter().copied().collect();
        assert_eq!(collected, vec![0, 1, 2, 3]);
        assert_eq!(list.pop_front(), Some(0));
        assert_eq!(list.pop_back(), Some(3));
        assert_eq!(list.pop_front(), Some(1));
        assert_eq!(list.pop_back(), Some(2));
        assert!(list.is_empty());
    }

    #[test]
    fn iter_forward_yields_head_to_tail() {
        let mut list = XorList::new();
        for i in 0..10 {
            list.push_back(i);
        }
        let collected: Vec<i32> = list.iter().copied().collect();
        assert_eq!(collected, (0..10).collect::<Vec<_>>());
    }

    #[test]
    fn iter_size_hint_tracks_remaining() {
        let mut list = XorList::new();
        for i in 0..4 {
            list.push_back(i);
        }
        let mut it = list.iter();
        assert_eq!(it.size_hint(), (4, Some(4)));
        it.next();
        assert_eq!(it.size_hint(), (3, Some(3)));
    }

    #[test]
    fn len_consistency_across_ops() {
        let mut list = XorList::new();
        assert_eq!(list.len(), 0);
        list.push_back(1);
        assert_eq!(list.len(), 1);
        list.push_front(0);
        assert_eq!(list.len(), 2);
        list.pop_back();
        assert_eq!(list.len(), 1);
        list.pop_front();
        assert_eq!(list.len(), 0);
        // Drained list still works after being emptied.
        list.push_back(42);
        assert_eq!(list.len(), 1);
        assert_eq!(list.pop_front(), Some(42));
        assert_eq!(list.len(), 0);
    }

    #[test]
    fn slot_reuse_after_pops() {
        // Exercise the free-list path: push, drain, push again, ensure order
        // is still correct (this catches arena bookkeeping bugs).
        let mut list = XorList::new();
        for i in 0..16 {
            list.push_back(i);
        }
        for _ in 0..16 {
            list.pop_front();
        }
        assert!(list.is_empty());
        for i in 100..116 {
            list.push_back(i);
        }
        let collected: Vec<i32> = list.iter().copied().collect();
        assert_eq!(collected, (100..116).collect::<Vec<_>>());
    }

    #[test]
    fn works_with_non_copy_type() {
        let mut list: XorList<String> = XorList::new();
        list.push_back("hello".to_string());
        list.push_front("world".to_string());
        assert_eq!(list.pop_front().as_deref(), Some("world"));
        assert_eq!(list.pop_back().as_deref(), Some("hello"));
        assert!(list.is_empty());
    }
}
