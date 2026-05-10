//! Merge of two sorted singly-linked lists.
//!
//! Given two singly-linked lists whose elements are already in non-decreasing
//! order, [`merge`] consumes both and produces a single sorted list using
//! purely iterative pointer manipulation. The procedure walks each input list
//! exactly once, splicing the smaller current head onto a growing tail, and
//! never recurses, so it is safe for arbitrarily long inputs without risking
//! stack overflow.
//!
//! - Time: `O(m + n)` where `m` and `n` are the input list lengths.
//! - Space: `O(1)` auxiliary; the returned list reuses the input nodes.

/// A node in a singly-linked list owning its successor via `Box`.
pub struct Node<T> {
    /// The value stored at this node.
    pub value: T,
    /// The next node in the list, if any.
    pub next: Option<Box<Self>>,
}

/// A simple owning singly-linked list.
pub struct LinkedList<T> {
    /// The head pointer of the list.
    pub head: Option<Box<Node<T>>>,
}

impl<T> Default for LinkedList<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> LinkedList<T> {
    /// Creates an empty list.
    pub const fn new() -> Self {
        Self { head: None }
    }

    /// Returns `true` if the list contains no elements.
    pub const fn is_empty(&self) -> bool {
        self.head.is_none()
    }

    /// Pushes `value` at the front of the list in `O(1)`.
    pub fn push_front(&mut self, value: T) {
        let new_node = Box::new(Node {
            value,
            next: self.head.take(),
        });
        self.head = Some(new_node);
    }

    /// Builds a list from `values`, preserving the input order. The element
    /// at index `0` becomes the head.
    pub fn from_vec(values: Vec<T>) -> Self {
        let mut list = Self::new();
        for v in values.into_iter().rev() {
            list.push_front(v);
        }
        list
    }

    /// Consumes the list and returns its elements in head-to-tail order.
    pub fn into_vec(self) -> Vec<T> {
        let mut out = Vec::new();
        let mut cur = self.head;
        while let Some(mut node) = cur {
            cur = node.next.take();
            out.push(node.value);
        }
        out
    }
}

/// Merges two sorted singly-linked lists into one sorted list, consuming both
/// inputs.
///
/// Both `a` and `b` must already be sorted in non-decreasing order. The merge
/// is stable: when elements compare equal, those from `a` precede those from
/// `b`. Implemented iteratively by repeatedly attaching the smaller current
/// head to a growing tail; no recursion is used.
///
/// - Time: `O(m + n)`.
/// - Space: `O(1)` auxiliary.
pub fn merge<T: Ord>(a: LinkedList<T>, b: LinkedList<T>) -> LinkedList<T> {
    let mut a_head = a.head;
    let mut b_head = b.head;

    // Pick the initial head, then keep `tail` as a mutable reference to the
    // `next` slot we will splice the next node into.
    let mut result = LinkedList::new();
    let mut tail: &mut Option<Box<Node<T>>> = &mut result.head;

    while a_head.is_some() && b_head.is_some() {
        // Decide which side to take without moving the nodes yet.
        let take_a = match (a_head.as_ref(), b_head.as_ref()) {
            (Some(an), Some(bn)) => an.value <= bn.value,
            _ => unreachable!(),
        };

        let mut node = if take_a {
            let mut n = a_head.take().expect("a_head is Some");
            a_head = n.next.take();
            n
        } else {
            let mut n = b_head.take().expect("b_head is Some");
            b_head = n.next.take();
            n
        };

        node.next = None;
        *tail = Some(node);
        // Advance `tail` to the freshly-attached node's `next` slot.
        tail = &mut tail.as_mut().expect("just assigned").next;
    }

    // Attach the remainder (at most one of these is non-empty).
    *tail = if a_head.is_some() { a_head } else { b_head };

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn merged_vec(a: Vec<i32>, b: Vec<i32>) -> Vec<i32> {
        let la = LinkedList::from_vec(a);
        let lb = LinkedList::from_vec(b);
        merge(la, lb).into_vec()
    }

    #[test]
    fn empty_plus_empty() {
        let a: LinkedList<i32> = LinkedList::new();
        let b: LinkedList<i32> = LinkedList::new();
        let m = merge(a, b);
        assert!(m.is_empty());
        assert_eq!(m.into_vec(), Vec::<i32>::new());
    }

    #[test]
    fn empty_plus_non_empty_left() {
        assert_eq!(merged_vec(vec![], vec![1, 2, 3]), vec![1, 2, 3]);
    }

    #[test]
    fn empty_plus_non_empty_right() {
        assert_eq!(merged_vec(vec![1, 2, 3], vec![]), vec![1, 2, 3]);
    }

    #[test]
    fn interleaving_merge() {
        assert_eq!(
            merged_vec(vec![1, 4, 7, 10], vec![2, 3, 8, 9]),
            vec![1, 2, 3, 4, 7, 8, 9, 10]
        );
    }

    #[test]
    fn equal_elements_are_kept_stable() {
        // Equal values from `a` should appear before equal values from `b`.
        assert_eq!(merged_vec(vec![1, 1, 1], vec![1, 1]), vec![1, 1, 1, 1, 1]);
        assert_eq!(
            merged_vec(vec![1, 2, 2, 3], vec![2, 2, 4]),
            vec![1, 2, 2, 2, 2, 3, 4]
        );
    }

    #[test]
    fn full_prefix_then_suffix() {
        // Entire `a` precedes entire `b`.
        assert_eq!(
            merged_vec(vec![1, 2, 3], vec![4, 5, 6]),
            vec![1, 2, 3, 4, 5, 6]
        );
        // Entire `b` precedes entire `a`.
        assert_eq!(
            merged_vec(vec![10, 11, 12], vec![1, 2, 3]),
            vec![1, 2, 3, 10, 11, 12]
        );
    }

    #[test]
    fn from_vec_into_vec_round_trip() {
        let v = vec![5, 4, 3, 2, 1];
        let list = LinkedList::from_vec(v.clone());
        assert_eq!(list.into_vec(), v);
    }

    #[test]
    fn push_front_builds_reverse_order() {
        let mut list = LinkedList::new();
        list.push_front(1);
        list.push_front(2);
        list.push_front(3);
        assert_eq!(list.into_vec(), vec![3, 2, 1]);
    }

    #[test]
    fn property_random_small_vecs_match_sorted_concat() {
        // Deterministic xorshift32 PRNG so the test stays reproducible without
        // adding any dev-dependency.
        let mut state: u32 = 0x9E37_79B9;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };

        for _ in 0..200 {
            let len_a = (next() % 16) as usize;
            let len_b = (next() % 16) as usize;
            let mut a: Vec<i32> = (0..len_a).map(|_| (next() % 50) as i32 - 25).collect();
            let mut b: Vec<i32> = (0..len_b).map(|_| (next() % 50) as i32 - 25).collect();
            a.sort();
            b.sort();

            let mut expected: Vec<i32> = a.iter().chain(b.iter()).copied().collect();
            expected.sort();

            let got = merged_vec(a, b);
            assert_eq!(got, expected);
        }
    }
}
