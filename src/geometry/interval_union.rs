//! Union of closed 1D intervals via sweep line.
//!
//! Given `n` closed intervals `[start, end]` on the integer line, return
//! the merged disjoint union as a sorted list of non-overlapping intervals.
//!
//! # Algorithm
//!
//! 1. Sort the input by `start` (`O(n log n)`).
//! 2. Sweep the intervals left to right, maintaining the current merged
//!    interval `[cur_start, cur_end]`. For each next interval `[s, e]`:
//!    * if `s <= cur_end` (overlap or touch), extend `cur_end = max(cur_end, e)`;
//!    * otherwise emit `[cur_start, cur_end]` and start a fresh run at `[s, e]`.
//! 3. After the sweep, emit the final pending run.
//!
//! Intervals that merely touch at an endpoint (e.g. `[1, 3]` and `[3, 5]`)
//! are merged into a single interval (`[1, 5]`), matching the closed-interval
//! semantics.
//!
//! # Complexity
//!
//! * Time: `O(n log n)` — dominated by the initial sort.
//! * Space: `O(n)` for the output (and a copy of the input for sorting).
//!
//! # Preconditions
//!
//! Each interval must satisfy `start <= end`; violating this triggers a
//! panic.

/// Returns the union of `intervals` as sorted, pairwise non-overlapping
/// closed intervals.
///
/// Intervals that touch at a single endpoint are merged. Returns an empty
/// vector for empty input.
///
/// # Panics
///
/// Panics if any input interval has `start > end`.
pub fn interval_union(intervals: &[(i64, i64)]) -> Vec<(i64, i64)> {
    if intervals.is_empty() {
        return Vec::new();
    }

    let mut sorted: Vec<(i64, i64)> = intervals
        .iter()
        .map(|&(s, e)| {
            assert!(s <= e, "interval start must be <= end");
            (s, e)
        })
        .collect();
    sorted.sort_unstable_by_key(|&(s, _)| s);

    let mut merged: Vec<(i64, i64)> = Vec::with_capacity(sorted.len());
    let (mut cur_start, mut cur_end) = sorted[0];

    for &(s, e) in &sorted[1..] {
        if s <= cur_end {
            // overlap or touch — extend the current run
            cur_end = cur_end.max(e);
        } else {
            // disjoint — emit the run and start a new one
            merged.push((cur_start, cur_end));
            cur_start = s;
            cur_end = e;
        }
    }
    merged.push((cur_start, cur_end));

    merged
}

#[cfg(test)]
mod tests {
    use super::interval_union;
    use quickcheck_macros::quickcheck;
    use std::collections::HashSet;

    #[test]
    fn empty_input() {
        assert_eq!(interval_union(&[]), vec![]);
    }

    #[test]
    fn single_interval() {
        assert_eq!(interval_union(&[(3, 7)]), vec![(3, 7)]);
        assert_eq!(interval_union(&[(-2, -2)]), vec![(-2, -2)]);
    }

    #[test]
    fn disjoint_already_sorted() {
        assert_eq!(
            interval_union(&[(0, 1), (3, 4), (6, 8)]),
            vec![(0, 1), (3, 4), (6, 8)]
        );
    }

    #[test]
    fn disjoint_unsorted_input() {
        assert_eq!(
            interval_union(&[(6, 8), (0, 1), (3, 4)]),
            vec![(0, 1), (3, 4), (6, 8)]
        );
    }

    #[test]
    fn overlapping_pair() {
        assert_eq!(interval_union(&[(1, 5), (3, 7)]), vec![(1, 7)]);
    }

    #[test]
    fn touching_endpoints_merge() {
        // closed semantics: [1,3] and [3,5] share the point 3 → merged
        assert_eq!(interval_union(&[(1, 3), (3, 5)]), vec![(1, 5)]);
        assert_eq!(interval_union(&[(0, 2), (2, 4), (4, 6)]), vec![(0, 6)]);
    }

    #[test]
    fn nested_intervals() {
        // a big interval swallowing several smaller ones
        assert_eq!(
            interval_union(&[(0, 100), (10, 20), (30, 40), (50, 99)]),
            vec![(0, 100)]
        );
    }

    #[test]
    fn fully_duplicated() {
        assert_eq!(interval_union(&[(2, 5), (2, 5), (2, 5)]), vec![(2, 5)]);
    }

    #[test]
    fn mixed_overlap_and_disjoint() {
        assert_eq!(
            interval_union(&[(1, 3), (2, 6), (8, 10), (15, 18), (17, 20)]),
            vec![(1, 6), (8, 10), (15, 20)]
        );
    }

    #[test]
    fn negative_coordinates() {
        assert_eq!(
            interval_union(&[(-10, -5), (-7, -2), (0, 3)]),
            vec![(-10, -2), (0, 3)]
        );
    }

    #[test]
    #[should_panic(expected = "interval start must be <= end")]
    fn panics_on_inverted_interval() {
        let _ = interval_union(&[(5, 1)]);
    }

    /// For closed intervals over the integers, the covered point set is
    /// `{ x : start <= x <= end }`. The union must cover the same set of
    /// integer points as the input.
    #[quickcheck]
    fn prop_matches_brute_force(raw: Vec<(i8, i8)>) -> bool {
        let intervals: Vec<(i64, i64)> = raw
            .into_iter()
            .map(|(a, b)| {
                let (a, b) = (i64::from(a), i64::from(b));
                if a <= b {
                    (a, b)
                } else {
                    (b, a)
                }
            })
            .collect();

        let merged = interval_union(&intervals);

        // sorted, pairwise disjoint with a gap > 0 between runs
        for w in merged.windows(2) {
            if w[0].1 >= w[1].0 || w[0].0 > w[0].1 {
                return false;
            }
        }

        let expected: HashSet<i64> = intervals.iter().flat_map(|&(s, e)| s..=e).collect();
        let got: HashSet<i64> = merged.iter().flat_map(|&(s, e)| s..=e).collect();

        expected == got
    }

    #[quickcheck]
    fn prop_idempotent(raw: Vec<(i8, i8)>) -> bool {
        let intervals: Vec<(i64, i64)> = raw
            .into_iter()
            .map(|(a, b)| {
                let (a, b) = (i64::from(a), i64::from(b));
                if a <= b {
                    (a, b)
                } else {
                    (b, a)
                }
            })
            .collect();

        let once = interval_union(&intervals);
        let twice = interval_union(&once);
        once == twice
    }
}
