//! Stack supporting `O(1)` maximum queries.
//!
//! `MaxStack` augments a regular LIFO stack with a parallel "running max"
//! stack: at each level we record the maximum value among all elements at or
//! below that level. Pushing a new value `x` records `max(x, current_max)`
//! on the auxiliary stack; popping removes the top entry from both stacks.
//! Because the auxiliary stack mirrors the main stack one-to-one, every
//! operation — `push`, `pop`, `top`, `max`, `len`, `is_empty` — runs in
//! `O(1)` worst-case time, at the cost of `O(n)` extra space.
//!
//! Ties are handled correctly: when the new value equals the running max,
//! the auxiliary stack still records the running max, so popping a duplicate
//! does not prematurely drop the maximum.

/// A LIFO stack that also exposes the maximum stored value in `O(1)`.
///
/// The element type must be `Ord + Clone` so the auxiliary stack can store
/// independent copies of the running maximum at each level.
pub struct MaxStack<T: Ord + Clone> {
    data: Vec<T>,
    maxs: Vec<T>,
}

impl<T: Ord + Clone> MaxStack<T> {
    /// Creates an empty `MaxStack`.
    pub const fn new() -> Self {
        Self {
            data: Vec::new(),
            maxs: Vec::new(),
        }
    }

    /// Pushes `x` onto the stack. `O(1)`.
    pub fn push(&mut self, x: T) {
        let new_max = match self.maxs.last() {
            Some(m) if *m > x => m.clone(),
            _ => x.clone(),
        };
        self.data.push(x);
        self.maxs.push(new_max);
    }

    /// Removes and returns the top element, or `None` if empty. `O(1)`.
    pub fn pop(&mut self) -> Option<T> {
        let v = self.data.pop()?;
        self.maxs.pop();
        Some(v)
    }

    /// Returns a reference to the top element without removing it. `O(1)`.
    pub fn top(&self) -> Option<&T> {
        self.data.last()
    }

    /// Returns a reference to the current maximum element. `O(1)`.
    pub fn max(&self) -> Option<&T> {
        self.maxs.last()
    }

    /// Number of elements on the stack. `O(1)`.
    pub const fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the stack is empty. `O(1)`.
    pub const fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

impl<T: Ord + Clone> Default for MaxStack<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::MaxStack;

    #[test]
    fn empty_returns_none() {
        let mut s: MaxStack<i32> = MaxStack::new();
        assert!(s.is_empty());
        assert_eq!(s.len(), 0);
        assert_eq!(s.top(), None);
        assert_eq!(s.max(), None);
        assert_eq!(s.pop(), None);
    }

    #[test]
    fn default_matches_new() {
        let s: MaxStack<i32> = MaxStack::default();
        assert!(s.is_empty());
    }

    #[test]
    fn push_sequence_tracks_max() {
        let mut s = MaxStack::new();
        s.push(3);
        assert_eq!(s.max(), Some(&3));
        assert_eq!(s.top(), Some(&3));
        s.push(1);
        assert_eq!(s.max(), Some(&3));
        assert_eq!(s.top(), Some(&1));
        s.push(5);
        assert_eq!(s.max(), Some(&5));
        assert_eq!(s.top(), Some(&5));
        s.push(2);
        assert_eq!(s.max(), Some(&5));
        assert_eq!(s.top(), Some(&2));
        s.push(4);
        assert_eq!(s.max(), Some(&5));
        assert_eq!(s.top(), Some(&4));
        assert_eq!(s.len(), 5);
    }

    #[test]
    fn pop_reduces_max_correctly() {
        let mut s = MaxStack::new();
        for v in [3, 1, 5, 2, 4] {
            s.push(v);
        }
        // [3, 1, 5, 2, 4] with running max [3, 3, 5, 5, 5]
        assert_eq!(s.pop(), Some(4));
        assert_eq!(s.max(), Some(&5));
        assert_eq!(s.pop(), Some(2));
        assert_eq!(s.max(), Some(&5));
        assert_eq!(s.pop(), Some(5));
        assert_eq!(s.max(), Some(&3));
        assert_eq!(s.pop(), Some(1));
        assert_eq!(s.max(), Some(&3));
        assert_eq!(s.pop(), Some(3));
        assert_eq!(s.max(), None);
        assert!(s.is_empty());
    }

    #[test]
    fn ties_handled() {
        let mut s = MaxStack::new();
        s.push(2);
        s.push(2);
        s.push(2);
        assert_eq!(s.max(), Some(&2));
        assert_eq!(s.pop(), Some(2));
        assert_eq!(s.max(), Some(&2));
        assert_eq!(s.pop(), Some(2));
        assert_eq!(s.max(), Some(&2));
        assert_eq!(s.pop(), Some(2));
        assert_eq!(s.max(), None);
    }

    #[test]
    fn ties_with_intermediate_smaller() {
        // Pushing the current max again, then a smaller value, then popping
        // the smaller should still leave max intact.
        let mut s = MaxStack::new();
        s.push(5);
        s.push(5);
        s.push(1);
        assert_eq!(s.max(), Some(&5));
        assert_eq!(s.pop(), Some(1));
        assert_eq!(s.max(), Some(&5));
        assert_eq!(s.pop(), Some(5));
        assert_eq!(s.max(), Some(&5));
        assert_eq!(s.pop(), Some(5));
        assert!(s.is_empty());
    }

    #[test]
    fn works_with_strings() {
        let mut s: MaxStack<String> = MaxStack::new();
        s.push("apple".to_string());
        s.push("banana".to_string());
        s.push("aardvark".to_string());
        assert_eq!(s.max().map(String::as_str), Some("banana"));
        assert_eq!(s.top().map(String::as_str), Some("aardvark"));
        s.pop();
        assert_eq!(s.max().map(String::as_str), Some("banana"));
        s.pop();
        assert_eq!(s.max().map(String::as_str), Some("apple"));
    }

    #[test]
    fn interleaved_push_pop() {
        let mut s = MaxStack::new();
        s.push(1);
        s.push(10);
        assert_eq!(s.max(), Some(&10));
        s.pop();
        assert_eq!(s.max(), Some(&1));
        s.push(7);
        assert_eq!(s.max(), Some(&7));
        s.push(3);
        assert_eq!(s.max(), Some(&7));
        s.pop();
        s.pop();
        assert_eq!(s.max(), Some(&1));
        assert_eq!(s.len(), 1);
    }
}
