//! Middle of a singly linked list via the slow/fast pointer technique.
//!
//! Floyd's two-pointer (also called the "tortoise and hare") trick walks two
//! cursors over the list at different speeds: a slow cursor advances one node
//! at a time while a fast cursor advances two nodes at a time. By the time the
//! fast cursor falls off the end, the slow cursor sits exactly at the middle.
//! For an even-length list of `n` nodes the slow cursor lands on the
//! upper-middle element (1-indexed position `n / 2 + 1`, i.e. 0-indexed
//! `n / 2`).
//!
//! The algorithm is a single pass over the list with `O(1)` extra memory and
//! does not require knowing the length in advance.
//!
//! This module ships a small inline `LinkedList<T>` (built from `Box`-linked
//! nodes) so the routine is self-contained and does not lean on
//! `std::collections::LinkedList` or any other crate.
//!
//! - Time: `O(n)` — each pointer visits every node at most once.
//! - Space: `O(1)` — only the two cursors are kept.

/// A node in the singly linked list.
struct Node<T> {
    value: T,
    next: Option<Box<Self>>,
}

/// A minimal singly linked list with helpers for construction and iteration.
pub struct LinkedList<T> {
    head: Option<Box<Node<T>>>,
}

impl<T> Default for LinkedList<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> LinkedList<T> {
    /// Returns an empty list.
    pub const fn new() -> Self {
        Self { head: None }
    }

    /// Builds a list whose nodes hold the elements of `items` in order.
    pub fn from_vec(items: Vec<T>) -> Self {
        let mut head: Option<Box<Node<T>>> = None;
        for value in items.into_iter().rev() {
            head = Some(Box::new(Node { value, next: head }));
        }
        Self { head }
    }

    /// Returns `true` iff the list contains no nodes.
    pub const fn is_empty(&self) -> bool {
        self.head.is_none()
    }

    /// Returns an iterator that yields shared references to each value in
    /// order from head to tail.
    pub fn iter(&self) -> Iter<'_, T> {
        Iter {
            cursor: self.head.as_deref(),
        }
    }
}

impl<'a, T> IntoIterator for &'a LinkedList<T> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Borrowing iterator over a [`LinkedList`].
pub struct Iter<'a, T> {
    cursor: Option<&'a Node<T>>,
}

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        let node = self.cursor?;
        self.cursor = node.next.as_deref();
        Some(&node.value)
    }
}

/// Returns a reference to the middle element of `list` using slow/fast
/// pointers.
///
/// For a list of length `n`:
/// - `n == 0` returns `None`.
/// - `n` odd returns the unique middle element (0-indexed `n / 2`).
/// - `n` even returns the upper-middle element (0-indexed `n / 2`).
///
/// Runs in `O(n)` time and `O(1)` extra space with a single pass over the
/// list.
pub fn middle<T>(list: &LinkedList<T>) -> Option<&T> {
    let mut slow = list.head.as_deref()?;
    let mut fast = list.head.as_deref()?;
    while let Some(next_fast) = fast.next.as_deref() {
        if let Some(after) = next_fast.next.as_deref() {
            // Fast jumps two nodes; slow advances one.
            fast = after;
            // Safe to unwrap: slow has at least one successor since fast
            // was able to advance two more nodes from its prior position.
            slow = slow.next.as_deref().expect("slow has a successor");
        } else {
            // Fast can only advance one more step => even length, and the
            // upper-middle is one past the current slow.
            slow = slow.next.as_deref().expect("slow has a successor");
            break;
        }
    }
    Some(&slow.value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_list_returns_none() {
        let list: LinkedList<i32> = LinkedList::new();
        assert!(list.is_empty());
        assert_eq!(middle(&list), None);
    }

    #[test]
    fn single_element_returns_that_element() {
        let list = LinkedList::from_vec(vec![42]);
        assert_eq!(middle(&list), Some(&42));
    }

    #[test]
    fn two_elements_returns_upper_middle() {
        let list = LinkedList::from_vec(vec![1, 2]);
        assert_eq!(middle(&list), Some(&2));
    }

    #[test]
    fn odd_length_five_returns_third() {
        let list = LinkedList::from_vec(vec![1, 2, 3, 4, 5]);
        assert_eq!(middle(&list), Some(&3));
    }

    #[test]
    fn even_length_six_returns_fourth() {
        let list = LinkedList::from_vec(vec![1, 2, 3, 4, 5, 6]);
        assert_eq!(middle(&list), Some(&4));
    }

    #[test]
    fn iter_yields_values_in_order() {
        let list = LinkedList::from_vec(vec![10, 20, 30]);
        let collected: Vec<i32> = list.iter().copied().collect();
        assert_eq!(collected, vec![10, 20, 30]);
    }

    #[test]
    fn three_elements_returns_middle() {
        let list = LinkedList::from_vec(vec![7, 8, 9]);
        assert_eq!(middle(&list), Some(&8));
    }

    #[test]
    fn four_elements_returns_upper_middle() {
        let list = LinkedList::from_vec(vec![1, 2, 3, 4]);
        assert_eq!(middle(&list), Some(&3));
    }

    #[test]
    fn large_list_thousand_returns_five_oh_one() {
        let values: Vec<i32> = (1..=1000).collect();
        let list = LinkedList::from_vec(values);
        // Even length 1000 => upper-middle is 0-indexed 500 => value 501.
        assert_eq!(middle(&list), Some(&501));
    }

    #[test]
    fn works_with_string_payload() {
        let list = LinkedList::from_vec(vec![
            String::from("a"),
            String::from("b"),
            String::from("c"),
            String::from("d"),
            String::from("e"),
        ]);
        assert_eq!(middle(&list), Some(&String::from("c")));
    }
}
