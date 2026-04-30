//! Activity selection (interval scheduling) by earliest-deadline-first greedy.
//!
//! Given a slice of `(start, end)` activities, picks a maximum-cardinality
//! subset of mutually non-overlapping activities. Runs in `O(n log n)` time
//! and `O(n)` auxiliary space (for the sorted index permutation).
//!
//! Non-overlap convention: activity `A` precedes activity `B` iff
//! `A.end <= B.start` — i.e. ends are closed and starts are open, so an
//! activity ending exactly at time `t` and another starting at time `t` are
//! considered compatible and can both be selected. Zero-duration activities
//! (`start == end`) are permitted and may chain back-to-back.
//!
//! Tie-breaking: indices are sorted by `end` ascending using a stable sort,
//! so activities sharing the same end time retain their original input order.

/// Returns indices into `activities` of a maximum-cardinality subset of
/// mutually non-overlapping intervals, sorted ascending by activity end time
/// (i.e. the order in which they would be performed).
///
/// Compatibility rule: two activities `A` and `B` are non-overlapping iff
/// `A.end <= B.start` or `B.end <= A.start`. Zero-duration activities are
/// allowed.
///
/// Empty input yields an empty `Vec`.
///
/// Time complexity: `O(n log n)` (dominated by the sort).
/// Space complexity: `O(n)` for the sorted index permutation.
pub fn select_activities(activities: &[(i64, i64)]) -> Vec<usize> {
    if activities.is_empty() {
        return Vec::new();
    }

    let mut order: Vec<usize> = (0..activities.len()).collect();
    order.sort_by_key(|&i| activities[i].1);

    let mut selected: Vec<usize> = Vec::new();
    let mut last_end: Option<i64> = None;
    for i in order {
        let (start, end) = activities[i];
        if last_end.is_none_or(|e| start >= e) {
            selected.push(i);
            last_end = Some(end);
        }
    }
    selected
}

#[cfg(test)]
mod tests {
    use super::select_activities;
    use quickcheck_macros::quickcheck;

    /// Brute-force the maximum cardinality of any non-overlapping subset.
    fn brute_force_max(activities: &[(i64, i64)]) -> usize {
        let n = activities.len();
        let mut best = 0_usize;
        for mask in 0_u32..(1_u32 << n) {
            let mut chosen: Vec<(i64, i64)> = Vec::new();
            for i in 0..n {
                if mask & (1 << i) != 0 {
                    chosen.push(activities[i]);
                }
            }
            chosen.sort_by_key(|a| a.1);
            let ok = chosen.windows(2).all(|w| w[0].1 <= w[1].0);
            if ok && chosen.len() > best {
                best = chosen.len();
            }
        }
        best
    }

    fn is_non_overlapping(activities: &[(i64, i64)], indices: &[usize]) -> bool {
        let mut chosen: Vec<(i64, i64)> = indices.iter().map(|&i| activities[i]).collect();
        chosen.sort_by_key(|a| a.1);
        chosen.windows(2).all(|w| w[0].1 <= w[1].0)
    }

    #[test]
    fn empty_input() {
        let result = select_activities(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn single_activity() {
        let activities = [(0_i64, 5_i64)];
        assert_eq!(select_activities(&activities), vec![0]);
    }

    #[test]
    fn all_disjoint_picks_all() {
        let activities = [(0_i64, 1_i64), (2, 3), (4, 5), (6, 7)];
        let result = select_activities(&activities);
        assert_eq!(result, vec![0, 1, 2, 3]);
    }

    #[test]
    fn all_overlap_picks_one() {
        // every activity contains time 5; exactly one can be selected
        let activities = [(0_i64, 10_i64), (1, 9), (2, 8), (3, 7), (4, 6)];
        let result = select_activities(&activities);
        assert_eq!(result.len(), 1);
        // earliest end time is index 4 ((4,6))
        assert_eq!(result, vec![4]);
    }

    #[test]
    fn classic_clrs_example() {
        // CLRS 16.1: 11 activities, optimal cardinality is 4.
        let activities = [
            (1_i64, 4_i64),
            (3, 5),
            (0, 6),
            (5, 7),
            (3, 9),
            (5, 9),
            (6, 10),
            (8, 11),
            (8, 12),
            (2, 14),
            (12, 16),
        ];
        let result = select_activities(&activities);
        assert_eq!(result.len(), 4);
        // With stable sort by end, the deterministic pick is:
        //   end=4 -> idx 0; then start>=4 with smallest end -> idx 3 (end=7);
        //   then start>=7 with smallest end -> idx 7 (end=11);
        //   then start>=11 with smallest end -> idx 10 (end=16).
        assert_eq!(result, vec![0, 3, 7, 10]);
        assert!(is_non_overlapping(&activities, &result));
    }

    #[test]
    fn touching_endpoints_are_compatible() {
        // (1,3) and (3,5) should both be selectable — closed-end / open-start.
        let activities = [(1_i64, 3_i64), (3, 5), (5, 7)];
        let result = select_activities(&activities);
        assert_eq!(result, vec![0, 1, 2]);
    }

    #[test]
    fn zero_duration_activities_chain() {
        // Three zero-duration activities at the same instant plus a longer one.
        let activities = [(2_i64, 2_i64), (2, 2), (0, 1), (1, 4)];
        let result = select_activities(&activities);
        // Cardinality should match the brute-force optimum.
        assert_eq!(result.len(), brute_force_max(&activities));
        assert!(is_non_overlapping(&activities, &result));
    }

    #[test]
    fn output_sorted_by_end_time() {
        let activities = [(0_i64, 6_i64), (1, 4), (5, 7), (8, 11)];
        let result = select_activities(&activities);
        let ends: Vec<i64> = result.iter().map(|&i| activities[i].1).collect();
        let mut sorted = ends.clone();
        sorted.sort_unstable();
        assert_eq!(ends, sorted);
    }

    #[allow(clippy::needless_pass_by_value)]
    #[quickcheck]
    fn greedy_matches_brute_force(raw: Vec<(i8, i8)>) -> bool {
        // Cap n at 10 to keep the 2^n brute force fast, and normalize so that
        // start <= end (treat malformed pairs as zero-duration at min(a,b)).
        let activities: Vec<(i64, i64)> = raw
            .into_iter()
            .take(10)
            .map(|(a, b)| {
                let s = i64::from(a.min(b));
                let e = i64::from(a.max(b));
                (s, e)
            })
            .collect();

        let greedy = select_activities(&activities);
        let optimal = brute_force_max(&activities);

        greedy.len() == optimal && is_non_overlapping(&activities, &greedy)
    }
}
