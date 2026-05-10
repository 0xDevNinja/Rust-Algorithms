//! Nth-from-last node in a singly linked list (single-pass, two-pointer).
//!
//! Given a singly linked list and a positive integer `n`, return a reference
//! to the value of the node located `n` positions from the end. The classical
//! single-pass solution uses two pointers separated by `n` steps: advance the
//! lead pointer `n` nodes ahead, then advance both pointers in lockstep until
//! the lead falls off the end. The trailing pointer then references the
//! desired node.
//!
//! - Time: `O(n)` (single traversal of the list).
//! - Space: `O(1)` auxiliary.
//!
//! Conventions used here: `n = 1` returns the last element, `n = len` returns
//! the first element. `n = 0` and `n > len` return `None`. Querying an empty
//! list returns `None` for any `n`.
//!
//! A minimal `LinkedList<T>` with `from_vec` and `iter` helpers is provided so
//! the algorithm can be exercised without pulling in `std::collections`'
//! doubly linked list (whose iterators would trivially permit a `Vec`-style
//! solution and obscure the two-pointer technique).

/// A node in the singly linked list.
struct Node<T> {
    value: T,
    next: Option<Box<Self>>,
}

/// A small singly linked list used to demonstrate the two-pointer algorithm.
pub struct LinkedList<T> {
    head: Option<Box<Node<T>>>,
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

    /// Builds a list from a vector, preserving input order.
    pub fn from_vec(items: Vec<T>) -> Self {
        let mut head: Option<Box<Node<T>>> = None;
        for value in items.into_iter().rev() {
            head = Some(Box::new(Node { value, next: head }));
        }
        Self { head }
    }

    /// Returns a forward iterator over the list's values.
    pub fn iter(&self) -> Iter<'_, T> {
        Iter {
            current: self.head.as_deref(),
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

/// Forward iterator over [`LinkedList`] values.
pub struct Iter<'a, T> {
    current: Option<&'a Node<T>>,
}

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        let node = self.current?;
        self.current = node.next.as_deref();
        Some(&node.value)
    }
}

/// Returns a reference to the value of the node located `n` positions from
/// the end of `list`, using the two-pointer technique in a single pass.
///
/// Returns `None` if `n == 0`, `n` exceeds the list length, or the list is
/// empty. With `n = 1` this returns the last element; with `n = list.len()`
/// it returns the first element.
///
/// - Time: `O(n)`.
/// - Space: `O(1)`.
pub fn nth_from_last<T>(list: &LinkedList<T>, n: usize) -> Option<&T> {
    if n == 0 {
        return None;
    }

    let mut lead = list.head.as_deref();
    // Advance the lead pointer `n` steps ahead. If the list has fewer than
    // `n` nodes the lead falls off the end, signalling `n > len`.
    for _ in 0..n {
        lead = lead?.next.as_deref();
    }

    let mut trail = list.head.as_deref();
    while let Some(node) = lead {
        lead = node.next.as_deref();
        // Safe: as long as `lead` was reachable, so is the trailing pointer.
        trail = trail.and_then(|t| t.next.as_deref());
    }

    trail.map(|node| &node.value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_list_returns_none_for_any_n() {
        let list: LinkedList<i32> = LinkedList::new();
        assert_eq!(nth_from_last(&list, 0), None);
        assert_eq!(nth_from_last(&list, 1), None);
        assert_eq!(nth_from_last(&list, 5), None);
    }

    #[test]
    fn n_equal_one_returns_last_element() {
        let list = LinkedList::from_vec(vec![1, 2, 3, 4, 5]);
        assert_eq!(nth_from_last(&list, 1), Some(&5));
    }

    #[test]
    fn n_equal_len_returns_first_element() {
        let list = LinkedList::from_vec(vec![1, 2, 3, 4, 5]);
        assert_eq!(nth_from_last(&list, 5), Some(&1));
    }

    #[test]
    fn n_greater_than_len_returns_none() {
        let list = LinkedList::from_vec(vec![1, 2, 3, 4, 5]);
        assert_eq!(nth_from_last(&list, 6), None);
        assert_eq!(nth_from_last(&list, 100), None);
    }

    #[test]
    fn n_zero_returns_none() {
        let list = LinkedList::from_vec(vec![1, 2, 3, 4, 5]);
        assert_eq!(nth_from_last(&list, 0), None);
    }

    #[test]
    fn middle_query_returns_expected_value() {
        let list = LinkedList::from_vec(vec![1, 2, 3, 4, 5]);
        assert_eq!(nth_from_last(&list, 3), Some(&3));
    }

    #[test]
    fn single_element_list() {
        let list = LinkedList::from_vec(vec![42]);
        assert_eq!(nth_from_last(&list, 1), Some(&42));
        assert_eq!(nth_from_last(&list, 2), None);
        assert_eq!(nth_from_last(&list, 0), None);
    }

    #[test]
    fn works_with_string_values() {
        let list = LinkedList::from_vec(vec![
            String::from("a"),
            String::from("b"),
            String::from("c"),
        ]);
        assert_eq!(nth_from_last(&list, 2).map(String::as_str), Some("b"));
    }

    #[test]
    fn iter_yields_values_in_order() {
        let list = LinkedList::from_vec(vec![10, 20, 30]);
        let collected: Vec<i32> = list.iter().copied().collect();
        assert_eq!(collected, vec![10, 20, 30]);
    }

    #[test]
    fn from_vec_empty_produces_empty_list() {
        let list: LinkedList<i32> = LinkedList::from_vec(vec![]);
        assert_eq!(list.iter().count(), 0);
    }
}
