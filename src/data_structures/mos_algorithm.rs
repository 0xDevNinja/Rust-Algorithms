//! Mo's algorithm — offline batch range-query processing.
//!
//! Given an array of `n` elements and `q` range queries `[l, r]` (inclusive),
//! answers all queries in `O((n + q) · √n · cost_add_remove)` by reordering
//! queries through block-based sorting and maintaining a sliding window via
//! caller-supplied `add` / `remove` callbacks.
//!
//! # Offline only
//! All queries must be known up front; the algorithm does not support online
//! (interleaved update/query) workloads.
//!
//! # Odd-block r-reversal optimisation
//! Queries are sorted by `(l / block, r)`, but within odd-numbered blocks the
//! `r` order is **reversed** (descending rather than ascending). When the
//! right pointer moves right through an even block then reverses leftward
//! through the next odd block, total right-pointer travel across a full block
//! pair is `O(block_size)` instead of `O(2 · block_size)`, roughly halving
//! the constant on the right-pointer movement. Reference: "Mos Algorithm with
//! Updates" and competitive-programming folklore (often called the "zigzag"
//! or "Hilbert order" trick in block form).
//!
//! # Complexity
//! - Time: `O((n + q) · √n · cost_of_add_remove)` where block size = `⌊√n⌋`.
//! - Space: `O(q)` auxiliary for the sorted query index array; the caller owns
//!   the element array and the state.

use std::cmp::Ordering;

/// Trait that callers implement to drive Mo's algorithm.
///
/// The driver calls `add` and `remove` to expand/shrink the current window,
/// then calls `answer` to capture the result for each query. The `value_index`
/// parameter is a raw index into the user's data slice; the implementor looks
/// up the actual value themselves, keeping this driver fully generic over `T`.
pub trait MosState {
    /// The answer type returned for each query.
    type Answer;

    /// Extend the current window to include the element at `value_index`.
    fn add(&mut self, value_index: usize);

    /// Shrink the current window by removing the element at `value_index`.
    fn remove(&mut self, value_index: usize);

    /// Return the answer for the current window `[cur_l, cur_r]`.
    fn answer(&self) -> Self::Answer;
}

/// Answers all `queries` using Mo's algorithm and returns results in the same
/// order as the input slice.
///
/// - `n` is the length of the caller's data array; all query indices must
///   satisfy `l <= r < n`.
/// - `queries` is a slice of inclusive `[l, r]` pairs.
/// - `state` is a mutable reference to the caller's query-state machine.
///
/// Returns a `Vec<S::Answer>` whose `i`-th element is the answer to
/// `queries[i]`.
///
/// # Panics
/// Panics with a descriptive message if any query has `l > r` or `r >= n`.
pub fn mos_algorithm<S: MosState>(
    n: usize,
    queries: &[(usize, usize)],
    state: &mut S,
) -> Vec<S::Answer> {
    if queries.is_empty() {
        return Vec::new();
    }

    // Validate all queries up front so the main loop never accesses out-of-bounds.
    for (qi, &(l, r)) in queries.iter().enumerate() {
        assert!(
            l <= r,
            "mos_algorithm: query {qi} has l={l} > r={r} (empty range)"
        );
        assert!(r < n, "mos_algorithm: query {qi} has r={r} >= n={n}");
    }

    // block_size = floor(sqrt(n)), but at least 1 to avoid division by zero.
    let block_size = ((n as f64).sqrt() as usize).max(1);

    // Build an array of query indices sorted by Mo's order.
    // Within the same block the sort is by r ascending for even blocks and
    // r descending for odd blocks (the zigzag optimisation).
    let mut order: Vec<usize> = (0..queries.len()).collect();
    order.sort_by(|&a, &b| {
        let (la, ra) = queries[a];
        let (lb, rb) = queries[b];
        let ba = la / block_size;
        let bb = lb / block_size;
        match ba.cmp(&bb) {
            Ordering::Equal => {
                // Same block: zigzag on r.
                if ba.is_multiple_of(2) {
                    ra.cmp(&rb) // ascending r for even blocks
                } else {
                    rb.cmp(&ra) // descending r for odd blocks
                }
            }
            other => other,
        }
    });

    // Allocate the answer buffer. We use an Option-filled Vec so we can write
    // answers at arbitrary original positions without initialising S::Answer.
    let mut answers: Vec<Option<S::Answer>> = (0..queries.len()).map(|_| None).collect();

    // cur_l and cur_r are maintained as i64 so the "empty window" start state
    // (cur_r = -1, meaning no elements are in the window) is representable
    // without unsafe code or Option overhead in the hot loop.
    let mut cur_l: i64 = 0;
    let mut cur_r: i64 = -1;

    for qi in order {
        let (l, r) = queries[qi];
        let l = l as i64;
        let r = r as i64;

        // Expand right boundary first (before shrinking left) to avoid
        // momentarily having an invalid window where cur_l > cur_r.
        while cur_r < r {
            cur_r += 1;
            state.add(cur_r as usize);
        }
        while cur_l > l {
            cur_l -= 1;
            state.add(cur_l as usize);
        }
        while cur_r > r {
            state.remove(cur_r as usize);
            cur_r -= 1;
        }
        while cur_l < l {
            state.remove(cur_l as usize);
            cur_l += 1;
        }

        answers[qi] = Some(state.answer());
    }

    // All slots were filled by the loop above; unwrap is safe here.
    // We use expect rather than unwrap to provide a clearer message.
    answers
        .into_iter()
        .map(|a| a.expect("mos_algorithm: answer slot unfilled (internal error)"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{mos_algorithm, MosState};
    use std::collections::HashMap;

    use quickcheck::TestResult;
    use quickcheck_macros::quickcheck;

    /// Upper bound on individual values in the quickcheck property test.
    /// Clamping to ±`BOUND` ensures sums over ≤ 80 elements stay within `i64`.
    const BOUND: i64 = 1_000_000;

    // ------------------------------------------------------------------
    // SumState: adds values held in a shared slice
    // ------------------------------------------------------------------

    struct SumState<'a> {
        values: &'a [i64],
        current_sum: i64,
    }

    impl<'a> SumState<'a> {
        fn new(values: &'a [i64]) -> Self {
            Self {
                values,
                current_sum: 0,
            }
        }
    }

    impl MosState for SumState<'_> {
        type Answer = i64;

        fn add(&mut self, idx: usize) {
            self.current_sum += self.values[idx];
        }

        fn remove(&mut self, idx: usize) {
            self.current_sum -= self.values[idx];
        }

        fn answer(&self) -> i64 {
            self.current_sum
        }
    }

    // ------------------------------------------------------------------
    // DistinctCountState: counts distinct values in [l, r]
    // ------------------------------------------------------------------

    struct DistinctCountState<'a> {
        values: &'a [i64],
        freq: HashMap<i64, usize>,
        distinct: usize,
    }

    impl<'a> DistinctCountState<'a> {
        fn new(values: &'a [i64]) -> Self {
            Self {
                values,
                freq: HashMap::new(),
                distinct: 0,
            }
        }
    }

    impl MosState for DistinctCountState<'_> {
        type Answer = usize;

        fn add(&mut self, idx: usize) {
            let v = self.values[idx];
            let cnt = self.freq.entry(v).or_insert(0);
            if *cnt == 0 {
                self.distinct += 1;
            }
            *cnt += 1;
        }

        fn remove(&mut self, idx: usize) {
            let v = self.values[idx];
            if let Some(cnt) = self.freq.get_mut(&v) {
                *cnt -= 1;
                if *cnt == 0 {
                    self.distinct -= 1;
                }
            }
        }

        fn answer(&self) -> usize {
            self.distinct
        }
    }

    // ------------------------------------------------------------------
    // Helper: brute-force range sum
    // ------------------------------------------------------------------

    fn brute_sum(values: &[i64], l: usize, r: usize) -> i64 {
        values[l..=r].iter().sum()
    }

    // ------------------------------------------------------------------
    // Tests
    // ------------------------------------------------------------------

    #[test]
    fn empty_queries_returns_empty_vec() {
        let values = vec![1_i64, 2, 3];
        let mut state = SumState::new(&values);
        let result = mos_algorithm(values.len(), &[], &mut state);
        assert!(result.is_empty());
    }

    #[test]
    fn single_full_range_query() {
        let values = vec![3_i64, 1, 4, 1, 5, 9, 2, 6];
        let queries = vec![(0, values.len() - 1)];
        let mut state = SumState::new(&values);
        let result = mos_algorithm(values.len(), &queries, &mut state);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], values.iter().sum::<i64>());
    }

    #[test]
    fn range_sum_verified_against_brute_force() {
        let values = vec![10_i64, -3, 5, 7, -2, 8, 1, 4];
        let queries = vec![(0, 3), (2, 6), (1, 7), (0, 0), (4, 4), (3, 5)];
        let mut state = SumState::new(&values);
        let result = mos_algorithm(values.len(), &queries, &mut state);
        for (i, &(l, r)) in queries.iter().enumerate() {
            assert_eq!(result[i], brute_sum(&values, l, r), "query {i}: [{l}, {r}]");
        }
    }

    #[test]
    fn range_distinct_count() {
        // Canonical Mo's use-case: count distinct elements in [l, r].
        let values = vec![1_i64, 2, 1, 3, 2, 1, 4, 2];
        let queries = vec![
            (0, 7), // {1,2,3,4} → 4
            (0, 2), // {1,2} → 2
            (2, 5), // {1,3,2} → 3
            (5, 7), // {1,4,2} → 3
            (3, 3), // {3} → 1
        ];
        let mut state = DistinctCountState::new(&values);
        let result = mos_algorithm(values.len(), &queries, &mut state);

        // Brute-force expected values.
        let expected: Vec<usize> = queries
            .iter()
            .map(|&(l, r)| {
                let mut seen = std::collections::HashSet::new();
                for &v in &values[l..=r] {
                    seen.insert(v);
                }
                seen.len()
            })
            .collect();

        assert_eq!(result, expected);
    }

    #[test]
    fn answer_order_matches_input_order() {
        // Intentionally supply queries in an order that requires output
        // reordering to verify the original-index restoration.
        let values: Vec<i64> = (0..20).collect();
        let queries: Vec<(usize, usize)> = vec![(15, 19), (0, 4), (10, 14), (5, 9)];
        let mut state = SumState::new(&values);
        let result = mos_algorithm(values.len(), &queries, &mut state);
        for (i, &(l, r)) in queries.iter().enumerate() {
            assert_eq!(result[i], brute_sum(&values, l, r), "query {i}");
        }
    }

    // ------------------------------------------------------------------
    // QuickCheck property test
    // ------------------------------------------------------------------

    /// Property: Mo's range-sum driver always matches brute-force summation
    /// for any `Vec<i64>` of length ≤ 80 and any set of in-bounds queries.
    ///
    /// Values are clamped to `±BOUND` so that sums over ≤ 80 elements stay
    /// well within `i64` range and don't overflow.
    #[quickcheck]
    #[allow(clippy::needless_pass_by_value)]
    fn prop_range_sum_matches_brute_force(
        values: Vec<i64>,
        raw_queries: Vec<(u8, u8)>,
    ) -> TestResult {
        if values.is_empty() || values.len() > 80 {
            return TestResult::discard();
        }
        if raw_queries.is_empty() || raw_queries.len() > 50 {
            return TestResult::discard();
        }

        // Clamp values so cumulative sums over ≤ 80 elements fit in i64.
        let values: Vec<i64> = values.iter().map(|v| v % BOUND).collect();
        let n = values.len();

        // Map raw u8 pairs to valid in-bounds [l, r] pairs.
        let queries: Vec<(usize, usize)> = raw_queries
            .iter()
            .map(|&(a, b)| {
                let lo = (a as usize) % n;
                let hi = (b as usize) % n;
                if lo <= hi {
                    (lo, hi)
                } else {
                    (hi, lo)
                }
            })
            .collect();

        let mut state = SumState::new(&values);
        let result = mos_algorithm(n, &queries, &mut state);

        for (i, &(l, r)) in queries.iter().enumerate() {
            let expected = brute_sum(&values, l, r);
            if result[i] != expected {
                return TestResult::failed();
            }
        }
        TestResult::passed()
    }
}
