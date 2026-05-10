//! Cycle detection and removal in a singly-linked list (Floyd's algorithm).
//!
//! The list is stored in an index-based arena so that cycles can be created
//! freely without using `Rc<RefCell<_>>` or `unsafe`.  Each node holds an
//! `Option<usize>` pointer into the arena's `nodes` vector.
//!
//! # Algorithm
//!
//! Floyd's tortoise-and-hare runs two pointers through the list, advancing the
//! "slow" pointer one step and the "fast" pointer two steps per iteration.
//! If the list contains a cycle, the two pointers eventually meet inside it.
//! Restarting the slow pointer at the head and advancing both at the same rate
//! makes them re-meet exactly at the cycle's entry node.
//!
//! Removal walks around the cycle once more from its entry to find the tail
//! (the node whose `next` closes the loop) and clears that link.
//!
//! # Complexity
//! - `detect_cycle_start` / `has_cycle`: O(n) time, O(1) extra space.
//! - `remove_cycle`: O(n) time, O(1) extra space.
//! - `from_vec`: O(n) time and space.

/// One arena node holding a value and an optional `next` index.
#[derive(Clone, Debug)]
pub struct Node<T> {
    /// Value stored at the node.
    pub value: T,
    /// Index of the next node in the arena, or `None` if this is the tail.
    pub next: Option<usize>,
}

/// Singly-linked list backed by an arena of nodes.
///
/// `head` is the index of the first node, or `None` if the list is empty.
#[derive(Clone, Debug, Default)]
pub struct ArenaList<T> {
    /// Backing storage for nodes.  Indices are stable for the lifetime of the
    /// list — no node is ever removed from this vector.
    pub nodes: Vec<Node<T>>,
    /// Index of the head node, or `None` for an empty list.
    pub head: Option<usize>,
}

impl<T> ArenaList<T> {
    /// Create an empty list.
    pub const fn new() -> Self {
        Self {
            nodes: Vec::new(),
            head: None,
        }
    }

    /// Build an acyclic list from `values`, preserving order.
    ///
    /// Node `i` lives at arena index `i` and points to `i + 1` (or `None` for
    /// the last element).  Returns an empty list when `values` is empty.
    pub fn from_vec(values: Vec<T>) -> Self {
        let n = values.len();
        let mut nodes = Vec::with_capacity(n);
        for (i, value) in values.into_iter().enumerate() {
            let next = if i + 1 < n { Some(i + 1) } else { None };
            nodes.push(Node { value, next });
        }
        let head = if n == 0 { None } else { Some(0) };
        Self { nodes, head }
    }

    /// Number of nodes in the arena (including any unreachable from `head`).
    pub const fn len(&self) -> usize {
        self.nodes.len()
    }

    /// `true` if the arena contains no nodes.
    pub const fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Make the current tail point to `idx`, creating a cycle.
    ///
    /// Used in tests and as a generic builder for cyclic lists.  Walks the
    /// list from `head` until it finds a node with `next == None` and rewires
    /// that node to point at `idx`.  Does nothing if the list is empty or
    /// already has no acyclic tail (e.g. it is already cyclic).  `idx` must
    /// be a valid arena index.
    pub fn link_tail_to(&mut self, idx: usize) {
        assert!(idx < self.nodes.len(), "link_tail_to: index out of bounds");
        let Some(head) = self.head else {
            return;
        };
        let mut cur = head;
        while let Some(next) = self.nodes[cur].next {
            cur = next;
        }
        self.nodes[cur].next = Some(idx);
    }

    /// Return the index where the cycle begins, or `None` if the list is
    /// acyclic.
    ///
    /// Implements Floyd's tortoise-and-hare in two phases:
    /// 1. Advance `slow` by 1 and `fast` by 2 until they meet (cycle) or
    ///    `fast` reaches the end (no cycle).
    /// 2. Reset `slow` to `head` and advance both by 1 step at a time; the
    ///    node where they meet is the cycle's entry point.
    pub fn detect_cycle_start(&self) -> Option<usize> {
        let head = self.head?;
        let mut slow = head;
        let mut fast = head;

        loop {
            // Advance fast by 2; bail out if it falls off the list.
            let f1 = self.nodes[fast].next?;
            let f2 = self.nodes[f1].next?;
            fast = f2;

            // Advance slow by 1.  `slow.next` cannot be None here because
            // fast has already traversed at least that far successfully.
            slow = self.nodes[slow].next.expect("slow has a successor");

            if slow == fast {
                break;
            }
        }

        // Phase 2: walk from head and from meeting point at the same rate.
        let mut a = head;
        let mut b = slow;
        while a != b {
            a = self.nodes[a].next.expect("phase 2: a inside cycle");
            b = self.nodes[b].next.expect("phase 2: b inside cycle");
        }
        Some(a)
    }

    /// `true` if the list contains a cycle.
    pub fn has_cycle(&self) -> bool {
        self.detect_cycle_start().is_some()
    }

    /// Remove the cycle (if any) by clearing the `next` pointer of the
    /// node that closes the loop.
    ///
    /// Returns `true` if a cycle was present and broken, `false` otherwise.
    /// After a successful call, `has_cycle` returns `false` and the list is
    /// a finite path starting from `head`.
    pub fn remove_cycle(&mut self) -> bool {
        let Some(start) = self.detect_cycle_start() else {
            return false;
        };

        // Walk from `start` until we find the node whose `next` points back
        // to `start` — that is the cycle's tail.  A self-loop is handled by
        // the loop's first iteration (`next == start`).
        let mut tail = start;
        loop {
            let next = self.nodes[tail]
                .next
                .expect("nodes inside a cycle always have a successor");
            if next == start {
                break;
            }
            tail = next;
        }
        self.nodes[tail].next = None;
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_list_has_no_cycle() {
        let list: ArenaList<i32> = ArenaList::new();
        assert_eq!(list.detect_cycle_start(), None);
        assert!(!list.has_cycle());
    }

    #[test]
    fn acyclic_list_returns_none() {
        let list = ArenaList::from_vec(vec![1, 2, 3, 4, 5]);
        assert_eq!(list.detect_cycle_start(), None);
        assert!(!list.has_cycle());
    }

    #[test]
    fn single_node_self_loop() {
        let mut list = ArenaList::from_vec(vec![42]);
        list.link_tail_to(0);
        assert_eq!(list.detect_cycle_start(), Some(0));
        assert!(list.has_cycle());
    }

    #[test]
    fn three_node_full_cycle() {
        // 0 -> 1 -> 2 -> 0
        let mut list = ArenaList::from_vec(vec![10, 20, 30]);
        list.link_tail_to(0);
        assert_eq!(list.detect_cycle_start(), Some(0));
        assert!(list.has_cycle());
    }

    #[test]
    fn cycle_in_middle() {
        // 0 -> 1 -> 2 -> 3 -> 4 -> 2  (cycle entry at index 2)
        let mut list = ArenaList::from_vec(vec![1, 2, 3, 4, 5]);
        list.link_tail_to(2);
        assert_eq!(list.detect_cycle_start(), Some(2));
        assert!(list.has_cycle());
    }

    #[test]
    fn remove_cycle_restores_acyclic_state() {
        let mut list = ArenaList::from_vec(vec![1, 2, 3, 4, 5]);
        list.link_tail_to(2);
        assert!(list.has_cycle());

        let removed = list.remove_cycle();
        assert!(removed);
        assert!(!list.has_cycle());
        assert_eq!(list.detect_cycle_start(), None);

        // The original tail (index 4) closed the loop, so its next should now
        // be None and the list should be traversable in 5 steps.
        assert_eq!(list.nodes[4].next, None);
        let mut count = 0;
        let mut cur = list.head;
        while let Some(i) = cur {
            count += 1;
            cur = list.nodes[i].next;
        }
        assert_eq!(count, 5);
    }

    #[test]
    fn remove_cycle_on_acyclic_list_is_noop() {
        let mut list = ArenaList::from_vec(vec![1, 2, 3]);
        assert!(!list.remove_cycle());
        assert!(!list.has_cycle());
    }

    #[test]
    fn remove_self_loop() {
        let mut list = ArenaList::from_vec(vec![7]);
        list.link_tail_to(0);
        assert!(list.has_cycle());
        assert!(list.remove_cycle());
        assert!(!list.has_cycle());
        assert_eq!(list.nodes[0].next, None);
    }

    #[test]
    fn has_cycle_agrees_with_detect_cycle_start() {
        // Acyclic
        let acyclic = ArenaList::from_vec(vec![1, 2, 3, 4]);
        assert_eq!(acyclic.has_cycle(), acyclic.detect_cycle_start().is_some());

        // Cycle entry at start
        let mut a = ArenaList::from_vec(vec![1, 2, 3]);
        a.link_tail_to(0);
        assert_eq!(a.has_cycle(), a.detect_cycle_start().is_some());

        // Cycle entry in the middle
        let mut b = ArenaList::from_vec(vec![1, 2, 3, 4, 5, 6]);
        b.link_tail_to(3);
        assert_eq!(b.has_cycle(), b.detect_cycle_start().is_some());

        // Empty
        let empty: ArenaList<i32> = ArenaList::new();
        assert_eq!(empty.has_cycle(), empty.detect_cycle_start().is_some());
    }
}
