//! Weighted interval scheduling.
//!
//! Given `n` intervals `(start, end, weight)`, pick a subset of mutually
//! non-overlapping intervals that maximises the total weight. Two intervals
//! `a` and `b` are considered compatible iff `a.end <= b.start` (touching at
//! a point is allowed, matching the standard CLRS formulation).
//!
//! Complexity: O(n log n) time (one sort by end + per-index binary search),
//! O(n) auxiliary space for the `dp` and `p` arrays.
//!
//! Preconditions: every input interval must satisfy `start <= end`.
//! Coordinates and weights are `i64` so callers can use shifted timelines or
//! signed offsets; non-positive-weight intervals are simply never chosen
//! because the empty selection has weight `0`. Zero-length intervals
//! (`start == end`) are permitted and behave like a point: they are
//! compatible with anything that does not strictly straddle that point.
//!
//! Tie-breaking: when two solutions have the same total weight, this
//! implementation prefers the "skip current" branch, mirroring the canonical
//! recurrence `dp[j] = max(dp[j-1], w[j] + dp[p(j)+1])` evaluated with `>`.
//! The returned indices refer to positions in the *original* input slice and
//! are sorted in ascending order so the output is deterministic.

/// Solve the weighted interval scheduling problem.
///
/// Returns `(max_total_weight, selected_indices_into_input)`. The selected
/// indices are positions in the original `intervals` slice, sorted ascending.
///
/// See the module docs for the compatibility rule (`a.end <= b.start`),
/// tie-breaking, and complexity.
pub fn weighted_interval_scheduling(intervals: &[(i64, i64, i64)]) -> (i64, Vec<usize>) {
    let n = intervals.len();
    if n == 0 {
        return (0, Vec::new());
    }

    // Sort a list of original indices by `(end, start)` ascending. The
    // start tie-break is essential when several intervals share an end:
    // the canonical recurrence requires the predecessor of `j` to be the
    // last interval with `end <= start[j]`, and an unstable end-only sort
    // can hide a feasible predecessor whose start is smaller.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by_key(|&i| (intervals[i].1, intervals[i].0));

    let ends: Vec<i64> = order.iter().map(|&i| intervals[i].1).collect();
    let starts: Vec<i64> = order.iter().map(|&i| intervals[i].0).collect();
    let weights: Vec<i64> = order.iter().map(|&i| intervals[i].2).collect();

    // p[j] = largest index i < j with ends[i] <= starts[j], or None.
    // We binary-search the contiguous prefix `ends[0..j]` for the rightmost
    // value that is `<= starts[j]`.
    let p: Vec<Option<usize>> = (0..n)
        .map(|j| {
            // partition_point returns the first index where the predicate is
            // false, so it gives the count of `end <= starts[j]` entries in
            // the prefix.
            let count = ends[..j].partition_point(|&e| e <= starts[j]);
            if count == 0 {
                None
            } else {
                Some(count - 1)
            }
        })
        .collect();

    // dp[j] = max weight using a subset of the first j sorted intervals.
    // Length n+1 so dp[0] = 0 is the empty-prefix base case.
    let mut dp = vec![0_i64; n + 1];
    for j in 0..n {
        let take = weights[j] + p[j].map_or(0, |pj| dp[pj + 1]);
        let skip = dp[j];
        dp[j + 1] = skip.max(take);
    }

    // Reconstruct: walk j from n down, taking interval j-1 iff `take > skip`.
    let mut chosen_sorted: Vec<usize> = Vec::new();
    let mut j = n;
    while j > 0 {
        let idx = j - 1;
        let take = weights[idx] + p[idx].map_or(0, |pj| dp[pj + 1]);
        let skip = dp[idx];
        if take > skip {
            chosen_sorted.push(order[idx]);
            j = p[idx].map_or(0, |pj| pj + 1);
        } else {
            j -= 1;
        }
    }

    chosen_sorted.sort_unstable();
    (dp[n], chosen_sorted)
}

#[cfg(test)]
mod tests {
    use super::weighted_interval_scheduling;
    use quickcheck_macros::quickcheck;

    /// Brute force: try every subset, keep the heaviest non-overlapping one.
    fn brute_force(intervals: &[(i64, i64, i64)]) -> i64 {
        let n = intervals.len();
        let mut best = 0_i64;
        for mask in 0_u32..(1_u32 << n) {
            let mut picked: Vec<(i64, i64, i64)> = (0..n)
                .filter(|i| mask & (1 << i) != 0)
                .map(|i| intervals[i])
                .collect();
            picked.sort_by_key(|iv| (iv.1, iv.0));
            let ok = picked.windows(2).all(|w| w[0].1 <= w[1].0);
            if ok {
                let total: i64 = picked.iter().map(|iv| iv.2).sum();
                if total > best {
                    best = total;
                }
            }
        }
        best
    }

    fn assert_valid_selection(intervals: &[(i64, i64, i64)], weight: i64, picked: &[usize]) {
        // Indices unique and in range, sorted ascending, weight matches sum.
        for w in picked.windows(2) {
            assert!(w[0] < w[1], "indices must be strictly ascending and unique");
        }
        for &i in picked {
            assert!(i < intervals.len());
        }
        let sum: i64 = picked.iter().map(|&i| intervals[i].2).sum();
        assert_eq!(sum, weight, "selected weight sum must match reported total");
        // Non-overlap when sorted by start.
        let mut by_start: Vec<(i64, i64, i64)> = picked.iter().map(|&i| intervals[i]).collect();
        by_start.sort_by_key(|iv| iv.0);
        assert!(by_start.windows(2).all(|w| w[0].1 <= w[1].0));
    }

    #[test]
    fn empty() {
        let (w, picked) = weighted_interval_scheduling(&[]);
        assert_eq!(w, 0);
        assert!(picked.is_empty());
    }

    #[test]
    fn single_positive_weight() {
        let ivs = [(0, 5, 7)];
        let (w, picked) = weighted_interval_scheduling(&ivs);
        assert_eq!(w, 7);
        assert_eq!(picked, vec![0]);
    }

    #[test]
    fn single_zero_weight_excluded() {
        // Adding an interval of weight 0 cannot improve over the empty set,
        // so the implementation prefers the empty selection.
        let ivs = [(0, 5, 0)];
        let (w, picked) = weighted_interval_scheduling(&ivs);
        assert_eq!(w, 0);
        assert!(picked.is_empty());
    }

    #[test]
    fn single_negative_weight_excluded() {
        let ivs = [(0, 5, -3)];
        let (w, picked) = weighted_interval_scheduling(&ivs);
        assert_eq!(w, 0);
        assert!(picked.is_empty());
    }

    #[test]
    fn all_disjoint_takes_all_positive() {
        let ivs = [(0, 1, 1), (1, 2, 2), (2, 3, 3), (3, 4, 4)];
        let (w, picked) = weighted_interval_scheduling(&ivs);
        assert_eq!(w, 10);
        assert_eq!(picked, vec![0, 1, 2, 3]);
    }

    #[test]
    fn all_overlapping_picks_heaviest() {
        // Every interval contains the point 5, so at most one is picked.
        let ivs = [(0, 10, 4), (1, 9, 7), (2, 8, 3), (3, 7, 5)];
        let (w, picked) = weighted_interval_scheduling(&ivs);
        assert_eq!(w, 7);
        assert_eq!(picked, vec![1]);
    }

    #[test]
    fn classic_clrs_style() {
        // Standard textbook example. With the `end <= start` compatibility
        // rule, the optimum is intervals {2, 6} (i.e. (0,6,8) and (6,10,7))
        // with total weight 15.
        let ivs = [
            (1, 4, 5),  // 0
            (3, 5, 1),  // 1
            (0, 6, 8),  // 2
            (5, 7, 4),  // 3
            (3, 8, 6),  // 4
            (5, 9, 3),  // 5
            (6, 10, 7), // 6
            (8, 11, 4), // 7
        ];
        let (w, picked) = weighted_interval_scheduling(&ivs);
        let bf = brute_force(&ivs);
        assert_eq!(w, bf);
        assert_valid_selection(&ivs, w, &picked);
        assert_eq!(w, 15);
        assert_eq!(picked, vec![2, 6]);
    }

    #[test]
    fn ties_on_weight_resolved_deterministically() {
        // Two equally good single-pick solutions: weight 5 either way.
        // Picking neither has weight 0, so a single 5 must win, but the
        // total weight is what matters. We assert determinism + correctness.
        let ivs = [(0, 5, 5), (0, 5, 5)];
        let (w, picked) = weighted_interval_scheduling(&ivs);
        assert_eq!(w, 5);
        assert_eq!(picked.len(), 1);
        assert_valid_selection(&ivs, w, &picked);
    }

    #[test]
    fn zero_length_interval_compatible() {
        // (3, 3, 5) is a point and is compatible with intervals ending at 3
        // or starting at 3. With the `end <= start` rule we can chain
        // (0, 3, 2) -> (3, 3, 5) -> (3, 6, 4) for total 11.
        let ivs = [(0, 3, 2), (3, 3, 5), (3, 6, 4)];
        let (w, picked) = weighted_interval_scheduling(&ivs);
        assert_eq!(w, 11);
        assert_valid_selection(&ivs, w, &picked);
        assert_eq!(picked, vec![0, 1, 2]);
    }

    #[test]
    fn descending_input_order_same_result() {
        let ascending = [(0, 1, 1), (1, 2, 2), (2, 3, 3), (3, 4, 4)];
        let mut descending = ascending;
        descending.reverse();
        let (w_a, _) = weighted_interval_scheduling(&ascending);
        let (w_d, picked_d) = weighted_interval_scheduling(&descending);
        assert_eq!(w_a, w_d);
        assert_eq!(w_d, 10);
        assert_valid_selection(&descending, w_d, &picked_d);
    }

    #[allow(clippy::needless_pass_by_value)]
    #[quickcheck]
    fn matches_brute_force(raw: Vec<(i8, i8, i8)>) -> bool {
        // Cap to 8 intervals so brute force stays cheap (2^8 = 256 subsets).
        let ivs: Vec<(i64, i64, i64)> = raw
            .into_iter()
            .take(8)
            .map(|(a, b, w)| {
                let (s, e) = (i64::from(a), i64::from(b));
                let (s, e) = if s <= e { (s, e) } else { (e, s) };
                (s, e, i64::from(w))
            })
            .collect();
        let (got, picked) = weighted_interval_scheduling(&ivs);
        if got != brute_force(&ivs) {
            return false;
        }
        // Validate the reconstructed selection itself, not just the total.
        if picked.windows(2).any(|w| w[0] >= w[1]) {
            return false;
        }
        if picked.iter().any(|&i| i >= ivs.len()) {
            return false;
        }
        let sum: i64 = picked.iter().map(|&i| ivs[i].2).sum();
        if sum != got {
            return false;
        }
        let mut by_end: Vec<(i64, i64, i64)> = picked.iter().map(|&i| ivs[i]).collect();
        by_end.sort_by_key(|iv| (iv.1, iv.0));
        by_end.windows(2).all(|w| w[0].1 <= w[1].0)
    }
}
