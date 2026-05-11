//! Minimum point cover for closed intervals (a.k.a. interval stabbing).
//!
//! Given a slice of closed intervals `[l, r]` on the integer line, returns a
//! minimum-cardinality set of points such that every interval contains at
//! least one chosen point. This is the classic greedy dual of activity
//! selection: sort by right endpoint and, whenever the current interval is
//! not yet covered, stab it at its rightmost point — that point is the
//! greediest possible choice because it maximizes future coverage.
//!
//! Convention: intervals are closed, so an interval `(l, r)` is considered
//! covered by point `p` iff `l <= p <= r`. Malformed intervals with `l > r`
//! are normalized to `(min, max)` rather than rejected.
//!
//! Time complexity: `O(n log n)` (dominated by the sort).
//! Space complexity: `O(n)` for the sorted permutation; the output set has
//! size at most `n`.
//!
//! Optimality: a standard exchange argument shows this greedy is optimal —
//! for the leftmost-ending uncovered interval `I`, any cover must include
//! some point in `I`, and choosing `I.r` covers a superset of the intervals
//! that any other point in `I` would cover.

/// Returns a minimum-cardinality set of points stabbing every interval in
/// `intervals`. The returned points are in ascending order. Empty input
/// yields an empty `Vec`.
///
/// Intervals are closed: a point `p` covers `(l, r)` iff `l <= p <= r`.
/// Pairs with `l > r` are normalized to `(min(l, r), max(l, r))`.
///
/// Time complexity: `O(n log n)`.
/// Space complexity: `O(n)`.
pub fn min_point_cover(intervals: &[(i64, i64)]) -> Vec<i64> {
    if intervals.is_empty() {
        return Vec::new();
    }

    let mut normalized: Vec<(i64, i64)> = intervals
        .iter()
        .map(|&(a, b)| (a.min(b), a.max(b)))
        .collect();
    // Sort by right endpoint; ties broken by left endpoint for determinism.
    normalized.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));

    let mut points: Vec<i64> = Vec::new();
    let mut last: Option<i64> = None;
    for (l, r) in normalized {
        if last.is_none_or(|p| p < l) {
            points.push(r);
            last = Some(r);
        }
    }
    points
}

#[cfg(test)]
mod tests {
    use super::min_point_cover;

    fn covers_all(intervals: &[(i64, i64)], points: &[i64]) -> bool {
        intervals.iter().all(|&(a, b)| {
            let (l, r) = (a.min(b), a.max(b));
            points.iter().any(|&p| l <= p && p <= r)
        })
    }

    fn is_sorted_asc(points: &[i64]) -> bool {
        points.windows(2).all(|w| w[0] <= w[1])
    }

    #[test]
    fn empty_input() {
        assert!(min_point_cover(&[]).is_empty());
    }

    #[test]
    fn single_interval_uses_right_endpoint() {
        assert_eq!(min_point_cover(&[(2, 7)]), vec![7]);
    }

    #[test]
    fn single_degenerate_point() {
        assert_eq!(min_point_cover(&[(4, 4)]), vec![4]);
    }

    #[test]
    fn disjoint_requires_one_per_interval() {
        let intervals = [(0_i64, 1_i64), (3, 4), (6, 7)];
        let points = min_point_cover(&intervals);
        assert_eq!(points, vec![1, 4, 7]);
        assert!(covers_all(&intervals, &points));
    }

    #[test]
    fn classic_three_intervals() {
        // [(1,3),(2,5),(4,6)] — sorted by right endpoint:
        //   (1,3) -> stab at 3 (covers (2,5) too); (4,6) uncovered -> stab at 6.
        let intervals = [(1_i64, 3_i64), (2, 5), (4, 6)];
        let points = min_point_cover(&intervals);
        assert_eq!(points, vec![3, 6]);
        assert!(covers_all(&intervals, &points));
    }

    #[test]
    fn all_overlapping_uses_one_point() {
        let intervals = [(0_i64, 10_i64), (1, 9), (2, 8), (3, 7), (4, 6)];
        let points = min_point_cover(&intervals);
        assert_eq!(points.len(), 1);
        assert!(covers_all(&intervals, &points));
        // The greedy stabs the leftmost-ending interval at its right endpoint.
        assert_eq!(points, vec![6]);
    }

    #[test]
    fn touching_endpoints_share_a_point() {
        // (1,3) and (3,5) both contain 3; one point suffices.
        let intervals = [(1_i64, 3_i64), (3, 5)];
        let points = min_point_cover(&intervals);
        assert_eq!(points, vec![3]);
        assert!(covers_all(&intervals, &points));
    }

    #[test]
    fn output_is_sorted_ascending() {
        let intervals = [(5_i64, 9_i64), (0, 2), (10, 12), (3, 4)];
        let points = min_point_cover(&intervals);
        assert!(is_sorted_asc(&points));
        assert!(covers_all(&intervals, &points));
    }

    #[test]
    fn unsorted_input_is_handled() {
        // Same multiset as the classic example, supplied in scrambled order.
        let intervals = [(4_i64, 6_i64), (1, 3), (2, 5)];
        let points = min_point_cover(&intervals);
        assert_eq!(points.len(), 2);
        assert!(covers_all(&intervals, &points));
    }

    #[test]
    fn malformed_pair_is_normalized() {
        // (8, 2) is treated as (2, 8); a single point in [2, 8] covers it.
        let intervals = [(8_i64, 2_i64)];
        let points = min_point_cover(&intervals);
        assert_eq!(points.len(), 1);
        let p = points[0];
        assert!((2..=8).contains(&p));
    }

    #[test]
    fn duplicates_dont_inflate_cover() {
        let intervals = [(1_i64, 4_i64), (1, 4), (1, 4)];
        let points = min_point_cover(&intervals);
        assert_eq!(points, vec![4]);
    }

    #[test]
    fn nested_intervals_use_inner_right() {
        // (0, 100) contains (10, 12); stabbing at 12 covers both.
        let intervals = [(0_i64, 100_i64), (10, 12)];
        let points = min_point_cover(&intervals);
        assert_eq!(points, vec![12]);
        assert!(covers_all(&intervals, &points));
    }
}
