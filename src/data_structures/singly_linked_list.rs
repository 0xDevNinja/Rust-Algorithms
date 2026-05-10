//! Singly-linked list with iterative reverse-in-place.
//!
//! A textbook singly-linked list built on owned `Box<Node<T>>` links. Each
//! node stores a value and an `Option<Box<Node<T>>>` pointing at the next
//! node, so the list owns its entire backbone with no reference counting and
//! no `unsafe`. A cached length keeps `len` / `is_empty` in O(1).
//!
//! # Operations
//! - `push_front`, `pop_front`, `front`, `front_mut`: **O(1)**.
//! - `len`, `is_empty`: **O(1)**.
//! - `iter`: **O(1)** to construct, **O(n)** to drain.
//! - `reverse_in_place`: **O(n)** time, **O(1)** extra space, no recursion.
//! - `into_vec`: **O(n)** time and space.
//! - `from_iter`: **O(n)** time, **O(n)** space.
//!
//! # Why iterative reverse?
//! A naive recursive reversal walks one stack frame per node and overflows
//! the thread stack for long lists. The pointer-swap loop used here keeps
//! a constant number of locals on the stack regardless of list length.
//!
//! # Drop
//! The default `Drop` for a deeply nested `Box<Node<T>>` chain is itself
//! recursive and can blow the stack on long lists. A manual `Drop` impl
//! unlinks nodes iteratively to keep destruction safe for any length.

/// Internal list node. Owns its successor through a `Box`.
struct Node<T> {
    value: T,
    next: Option<Box<Self>>,
}

/// A singly-linked list with O(1) front operations and iterative reverse.
pub struct SinglyLinkedList<T> {
    head: Option<Box<Node<T>>>,
    len: usize,
}

impl<T> SinglyLinkedList<T> {
    /// Creates an empty list.
    #[must_use]
    pub const fn new() -> Self {
        Self { head: None, len: 0 }
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

    /// Inserts `value` at the front of the list. **O(1)**.
    pub fn push_front(&mut self, value: T) {
        let new_head = Box::new(Node {
            value,
            next: self.head.take(),
        });
        self.head = Some(new_head);
        self.len += 1;
    }

    /// Removes and returns the front element, or `None` if empty. **O(1)**.
    pub fn pop_front(&mut self) -> Option<T> {
        self.head.take().map(|node| {
            let node = *node;
            self.head = node.next;
            self.len -= 1;
            node.value
        })
    }

    /// Returns a reference to the front element, or `None` if empty. **O(1)**.
    #[must_use]
    pub fn front(&self) -> Option<&T> {
        self.head.as_deref().map(|node| &node.value)
    }

    /// Returns a mutable reference to the front element, or `None` if empty.
    /// **O(1)**.
    pub fn front_mut(&mut self) -> Option<&mut T> {
        self.head.as_deref_mut().map(|node| &mut node.value)
    }

    /// Returns an iterator yielding shared references to each element from
    /// front to back. **O(1)** to construct, **O(n)** to fully consume.
    #[must_use]
    pub fn iter(&self) -> Iter<'_, T> {
        Iter {
            next: self.head.as_deref(),
        }
    }

    /// Reverses the list in place using a pointer-swap loop.
    ///
    /// Runs in **O(n)** time and **O(1)** extra space. The loop is iterative
    /// so it cannot overflow the thread stack on arbitrarily long lists.
    pub fn reverse_in_place(&mut self) {
        let mut prev: Option<Box<Node<T>>> = None;
        let mut current = self.head.take();
        while let Some(mut node) = current {
            let next = node.next.take();
            node.next = prev;
            prev = Some(node);
            current = next;
        }
        self.head = prev;
    }

    /// Consumes the list and returns its elements in front-to-back order.
    /// **O(n)**.
    #[must_use]
    pub fn into_vec(mut self) -> Vec<T> {
        let mut out = Vec::with_capacity(self.len);
        while let Some(value) = self.pop_front() {
            out.push(value);
        }
        out
    }
}

impl<T> Default for SinglyLinkedList<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Drop for SinglyLinkedList<T> {
    fn drop(&mut self) {
        // Walk the chain iteratively so dropping a million-node list does
        // not blow the recursion limit through cascading `Box` drops.
        let mut current = self.head.take();
        while let Some(mut node) = current {
            current = node.next.take();
        }
        self.len = 0;
    }
}

impl<T> FromIterator<T> for SinglyLinkedList<T> {
    /// Builds a list whose front-to-back order matches the iterator's order.
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        // Push each item onto a temporary list (which reverses order), then
        // reverse once to restore the iterator's order. Avoids an O(n) walk
        // to the tail per insertion.
        let mut list = Self::new();
        for value in iter {
            list.push_front(value);
        }
        list.reverse_in_place();
        list
    }
}

impl<'a, T> IntoIterator for &'a SinglyLinkedList<T> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Borrowed iterator over a [`SinglyLinkedList`].
pub struct Iter<'a, T> {
    next: Option<&'a Node<T>>,
}

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        self.next.map(|node| {
            self.next = node.next.as_deref();
            &node.value
        })
    }
}

#[cfg(test)]
mod tests {
    use super::SinglyLinkedList;

    #[test]
    fn new_is_empty() {
        let list: SinglyLinkedList<i32> = SinglyLinkedList::new();
        assert!(list.is_empty());
        assert_eq!(list.len(), 0);
        assert!(list.front().is_none());
    }

    #[test]
    fn default_matches_new() {
        let list: SinglyLinkedList<i32> = SinglyLinkedList::default();
        assert!(list.is_empty());
    }

    #[test]
    fn push_pop_round_trip() {
        let mut list = SinglyLinkedList::new();
        list.push_front(1);
        list.push_front(2);
        list.push_front(3);
        assert_eq!(list.len(), 3);
        assert_eq!(list.front(), Some(&3));
        assert_eq!(list.pop_front(), Some(3));
        assert_eq!(list.pop_front(), Some(2));
        assert_eq!(list.pop_front(), Some(1));
        assert_eq!(list.pop_front(), None);
        assert!(list.is_empty());
    }

    #[test]
    fn front_mut_mutates_value() {
        let mut list = SinglyLinkedList::new();
        list.push_front(10);
        if let Some(v) = list.front_mut() {
            *v = 42;
        }
        assert_eq!(list.front(), Some(&42));
    }

    #[test]
    fn len_consistency_across_ops() {
        let mut list = SinglyLinkedList::new();
        assert_eq!(list.len(), 0);
        for i in 0..5 {
            list.push_front(i);
            assert_eq!(list.len(), i + 1);
        }
        for expected in (0..5).rev() {
            assert_eq!(list.len(), expected + 1);
            list.pop_front();
        }
        assert_eq!(list.len(), 0);
    }

    #[test]
    fn iter_yields_front_to_back() {
        let list: SinglyLinkedList<i32> = [1, 2, 3, 4].into_iter().collect();
        let collected: Vec<i32> = list.iter().copied().collect();
        assert_eq!(collected, vec![1, 2, 3, 4]);
    }

    #[test]
    fn into_vec_preserves_order() {
        let list: SinglyLinkedList<i32> = [10, 20, 30].into_iter().collect();
        assert_eq!(list.into_vec(), vec![10, 20, 30]);
    }

    #[test]
    fn reverse_empty_is_noop() {
        let mut list: SinglyLinkedList<i32> = SinglyLinkedList::new();
        list.reverse_in_place();
        assert!(list.is_empty());
        assert_eq!(list.len(), 0);
    }

    #[test]
    fn reverse_single_element() {
        let mut list = SinglyLinkedList::new();
        list.push_front(7);
        list.reverse_in_place();
        assert_eq!(list.len(), 1);
        assert_eq!(list.front(), Some(&7));
    }

    #[test]
    fn reverse_many_elements() {
        let mut list: SinglyLinkedList<i32> = (1..=6).collect();
        list.reverse_in_place();
        let collected: Vec<i32> = list.iter().copied().collect();
        assert_eq!(collected, vec![6, 5, 4, 3, 2, 1]);
        assert_eq!(list.len(), 6);
    }

    #[test]
    fn double_reverse_is_identity() {
        let original: Vec<i32> = (0..50).collect();
        let mut list: SinglyLinkedList<i32> = original.iter().copied().collect();
        list.reverse_in_place();
        list.reverse_in_place();
        assert_eq!(list.into_vec(), original);
    }

    #[test]
    fn from_iter_matches_input_order() {
        let src = vec!["a", "b", "c"];
        let list: SinglyLinkedList<&str> = src.iter().copied().collect();
        assert_eq!(list.into_vec(), src);
    }

    #[test]
    fn reverse_long_list_no_stack_overflow() {
        // 100k nodes — recursion would overflow on most default stacks.
        let mut list: SinglyLinkedList<u32> = (0..100_000_u32).collect();
        list.reverse_in_place();
        assert_eq!(list.len(), 100_000);
        assert_eq!(list.front(), Some(&99_999));
    }
}
