//! Running median over a stream via two heaps.
//!
//! Maintains the median of an online sequence of `i64` values using a pair of
//! priority queues:
//!
//! * `lo` – a **max-heap** holding the lower half of the values seen so far,
//! * `hi` – a **min-heap** (encoded with [`std::cmp::Reverse`] over the
//!   standard [`BinaryHeap`]) holding the upper half.
//!
//! The size invariant kept after every insertion is
//! `lo.len() == hi.len()` or `lo.len() == hi.len() + 1`. With this invariant:
//!
//! * if `n = lo.len() + hi.len()` is **odd**, the median is `*lo.peek().unwrap()`,
//! * if `n` is **even** and non-zero**, the median is the average of
//!   `*lo.peek().unwrap()` and `hi.peek().unwrap().0`.
//!
//! # Complexities
//!
//! Let `n` be the number of elements inserted so far.
//!
//! | Operation | Time       | Notes                                |
//! |-----------|------------|--------------------------------------|
//! | `new`     | O(1)       | empty heaps                          |
//! | `add`     | O(log n)   | one push + at most one rebalance     |
//! | `median`  | O(1)       | reads heap tops only                 |
//! | `len` / `is_empty` | O(1) |                                  |
//!
//! Space: O(n).
//!
//! # Example
//!
//! ```
//! use rust_algorithms::data_structures::running_median::RunningMedian;
//!
//! let mut rm = RunningMedian::new();
//! assert_eq!(rm.median(), None);
//!
//! rm.add(1);
//! rm.add(2);
//! rm.add(3);
//! assert_eq!(rm.median(), Some(2.0));
//!
//! rm.add(4);
//! assert_eq!(rm.median(), Some(2.5));
//! ```

use std::cmp::Reverse;
use std::collections::BinaryHeap;

/// Online median tracker for an `i64` stream.
///
/// Insert values with [`RunningMedian::add`]; query the current median with
/// [`RunningMedian::median`]. Each insertion runs in `O(log n)` time.
#[derive(Debug, Default)]
pub struct RunningMedian {
    /// Max-heap of the lower half of the stream (top = largest of the lower half).
    lo: BinaryHeap<i64>,
    /// Min-heap of the upper half of the stream (top = smallest of the upper half).
    hi: BinaryHeap<Reverse<i64>>,
}

impl RunningMedian {
    /// Creates an empty `RunningMedian`.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            lo: BinaryHeap::new(),
            hi: BinaryHeap::new(),
        }
    }

    /// Returns the number of elements inserted so far.
    #[must_use]
    pub fn len(&self) -> usize {
        self.lo.len() + self.hi.len()
    }

    /// Returns `true` if no elements have been inserted yet.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.lo.is_empty() && self.hi.is_empty()
    }

    /// Inserts `value` into the stream in `O(log n)` time.
    pub fn add(&mut self, value: i64) {
        // Route the new value into the appropriate half.
        match self.lo.peek() {
            Some(&top) if value <= top => self.lo.push(value),
            _ => self.hi.push(Reverse(value)),
        }

        // Rebalance so that `lo.len()` is either equal to or one greater than
        // `hi.len()`.
        if self.lo.len() > self.hi.len() + 1 {
            // SAFETY-equivalent: lo is non-empty by the size check above.
            let moved = self.lo.pop().expect("lo is non-empty");
            self.hi.push(Reverse(moved));
        } else if self.hi.len() > self.lo.len() {
            let Reverse(moved) = self.hi.pop().expect("hi is non-empty");
            self.lo.push(moved);
        }
    }

    /// Returns the current median, or `None` if no values have been inserted.
    ///
    /// For an even count this is the average of the two middle values; for an
    /// odd count it is the single middle value.
    #[must_use]
    pub fn median(&self) -> Option<f64> {
        match (self.lo.len(), self.hi.len()) {
            (0, 0) => None,
            (a, b) if a == b => {
                let l = *self.lo.peek().expect("lo is non-empty");
                let h = self.hi.peek().expect("hi is non-empty").0;
                Some(f64::midpoint(l as f64, h as f64))
            }
            _ => Some(*self.lo.peek().expect("lo is non-empty") as f64),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_returns_none() {
        let rm = RunningMedian::new();
        assert!(rm.is_empty());
        assert_eq!(rm.len(), 0);
        assert_eq!(rm.median(), None);
    }

    #[test]
    fn single_value() {
        let mut rm = RunningMedian::new();
        rm.add(42);
        assert_eq!(rm.len(), 1);
        assert_eq!(rm.median(), Some(42.0));
    }

    #[test]
    fn stream_one_two_three() {
        let mut rm = RunningMedian::new();
        rm.add(1);
        assert_eq!(rm.median(), Some(1.0));
        rm.add(2);
        assert_eq!(rm.median(), Some(1.5));
        rm.add(3);
        assert_eq!(rm.median(), Some(2.0));
    }

    #[test]
    fn stream_one_through_four() {
        let mut rm = RunningMedian::new();
        for v in [1, 2, 3, 4] {
            rm.add(v);
        }
        assert_eq!(rm.median(), Some(2.5));
    }

    #[test]
    fn handles_duplicates_and_negatives() {
        let mut rm = RunningMedian::new();
        for v in [-5, -5, 0, 5, 5] {
            rm.add(v);
        }
        // Sorted: [-5, -5, 0, 5, 5] -> median 0
        assert_eq!(rm.median(), Some(0.0));
        rm.add(-5);
        // Sorted: [-5, -5, -5, 0, 5, 5] -> median (-5 + 0) / 2 = -2.5
        assert_eq!(rm.median(), Some(-2.5));
    }

    #[test]
    fn descending_stream() {
        let mut rm = RunningMedian::new();
        let mut buf: Vec<i64> = Vec::new();
        for v in (1..=10).rev() {
            rm.add(v);
            buf.push(v);
            buf.sort();
            let expected = oracle_median(&buf);
            assert_eq!(rm.median(), Some(expected), "after inserting {v}");
        }
    }

    #[test]
    fn large_stream_against_sorted_oracle() {
        // Deterministic pseudo-random stream using a linear congruential generator.
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut rm = RunningMedian::new();
        let mut sorted: Vec<i64> = Vec::new();

        for _ in 0..1000 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            // Map to a signed range to exercise both halves.
            let v = (state >> 32) as i32 as i64;

            rm.add(v);
            let pos = sorted.partition_point(|&x| x <= v);
            sorted.insert(pos, v);

            let expected = oracle_median(&sorted);
            assert_eq!(rm.median(), Some(expected));
        }

        assert_eq!(rm.len(), 1000);
    }

    fn oracle_median(sorted: &[i64]) -> f64 {
        let n = sorted.len();
        assert!(n > 0);
        if n % 2 == 1 {
            sorted[n / 2] as f64
        } else {
            f64::midpoint(sorted[n / 2 - 1] as f64, sorted[n / 2] as f64)
        }
    }
}
