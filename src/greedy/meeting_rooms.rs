//! Minimum number of meeting rooms via sweep-line greedy.
//!
//! Given a slice of meetings as `(start, end)` pairs, returns the minimum
//! number of rooms required so that no two simultaneous meetings share a
//! room. A meeting occupies its room over the half-open interval
//! `[start, end)`: a meeting ending at time `t` and another starting at
//! time `t` may share a single room (back-to-back, touching endpoints).
//!
//! Strategy: extract the multiset of start times and the multiset of end
//! times, sort each independently, then sweep two pointers in lockstep.
//! At every start event we open a room; at every end event we free one.
//! The peak number of concurrently open rooms is the answer. With the
//! half-open convention, an end at time `t` is processed before a start
//! at the same time `t`, which is exactly what `start < end` produces
//! when comparing the two pointers.
//!
//! Convention: intervals are half-open `[start, end)`. Pairs with
//! `start > end` are normalized to `(min, max)`. Degenerate meetings
//! with `start == end` occupy the room for zero time and contribute
//! nothing to the room count.
//!
//! Time complexity: `O(n log n)` (dominated by the two sorts).
//! Space complexity: `O(n)` for the start/end vectors.
//!
//! Optimality: the answer equals the maximum clique of the interval
//! graph, which for intervals on the line equals the maximum point
//! depth — exactly what the sweep computes.

/// Returns the minimum number of rooms needed to host every meeting in
/// `intervals` without conflicts, treating each meeting as the half-open
/// interval `[start, end)`. Empty input yields `0`. Pairs with
/// `start > end` are normalized to `(min, max)`; degenerate meetings
/// with `start == end` are ignored (they occupy no time).
///
/// Time complexity: `O(n log n)`.
/// Space complexity: `O(n)`.
pub fn min_meeting_rooms(intervals: &[(i64, i64)]) -> i64 {
    if intervals.is_empty() {
        return 0;
    }

    let mut starts: Vec<i64> = Vec::with_capacity(intervals.len());
    let mut ends: Vec<i64> = Vec::with_capacity(intervals.len());
    for &(a, b) in intervals {
        let (s, e) = (a.min(b), a.max(b));
        if s == e {
            // Zero-length meeting: never holds the room.
            continue;
        }
        starts.push(s);
        ends.push(e);
    }

    if starts.is_empty() {
        return 0;
    }

    starts.sort_unstable();
    ends.sort_unstable();

    let n = starts.len();
    let (mut i, mut j) = (0usize, 0usize);
    let (mut active, mut peak) = (0i64, 0i64);
    while i < n {
        // Half-open intervals: an end at time t frees the room for a
        // start at the same time t, so use strict `<` here.
        if starts[i] < ends[j] {
            active += 1;
            if active > peak {
                peak = active;
            }
            i += 1;
        } else {
            active -= 1;
            j += 1;
        }
    }
    peak
}

#[cfg(test)]
mod tests {
    use super::min_meeting_rooms;

    #[test]
    fn empty_input() {
        assert_eq!(min_meeting_rooms(&[]), 0);
    }

    #[test]
    fn classic_three_meetings() {
        // (0,30) overlaps both (5,10) and (15,20); the latter two are disjoint.
        // Peak depth = 2.
        assert_eq!(min_meeting_rooms(&[(0, 30), (5, 10), (15, 20)]), 2);
    }

    #[test]
    fn two_disjoint_meetings_share_one_room() {
        assert_eq!(min_meeting_rooms(&[(7, 10), (2, 4)]), 1);
    }

    #[test]
    fn all_overlapping_needs_n_rooms() {
        let intervals = [(0_i64, 10_i64), (1, 9), (2, 8), (3, 7), (4, 6)];
        assert_eq!(min_meeting_rooms(&intervals), intervals.len() as i64);
    }

    #[test]
    fn back_to_back_touching_shares_one_room() {
        // Half-open: (1,5) ends at 5; (5,9) starts at 5. They may share.
        assert_eq!(min_meeting_rooms(&[(1, 5), (5, 9)]), 1);
        assert_eq!(min_meeting_rooms(&[(0, 1), (1, 2), (2, 3), (3, 4)]), 1);
    }

    #[test]
    fn single_meeting_needs_one_room() {
        assert_eq!(min_meeting_rooms(&[(0, 1)]), 1);
    }

    #[test]
    fn zero_length_meeting_ignored() {
        // Degenerate (5,5) holds no room; only (1,6) matters.
        assert_eq!(min_meeting_rooms(&[(5, 5), (1, 6)]), 1);
        assert_eq!(min_meeting_rooms(&[(5, 5), (5, 5)]), 0);
    }

    #[test]
    fn malformed_pair_is_normalized() {
        // (10, 2) is treated as (2, 10); overlaps (5, 7) -> 2 rooms.
        assert_eq!(min_meeting_rooms(&[(10, 2), (5, 7)]), 2);
    }

    #[test]
    fn duplicates_stack() {
        // Three identical meetings all overlap.
        assert_eq!(min_meeting_rooms(&[(1, 5), (1, 5), (1, 5)]), 3);
    }

    #[test]
    fn negative_times_supported() {
        assert_eq!(min_meeting_rooms(&[(-5, 0), (-3, -1), (-10, -8)]), 2);
    }

    #[test]
    fn matches_naive_on_random_inputs() {
        // Deterministic LCG so the test is reproducible without rand crate.
        fn naive(intervals: &[(i64, i64)]) -> i64 {
            // Half-open: at time t, count meetings with s <= t < e.
            // We only need to check at start times (where depth peaks).
            let mut peak = 0i64;
            for &(t, _) in intervals {
                let mut depth = 0i64;
                for &(s, e) in intervals {
                    let (s, e) = (s.min(e), s.max(e));
                    if s == e {
                        continue;
                    }
                    if s <= t && t < e {
                        depth += 1;
                    }
                }
                if depth > peak {
                    peak = depth;
                }
            }
            peak
        }

        let mut state: u64 = 0x00C0_FFEE_1234_5678;
        let mut next = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            state
        };

        for _trial in 0..20 {
            let n = 1 + (next() % 64) as usize;
            let mut intervals: Vec<(i64, i64)> = Vec::with_capacity(n);
            for _ in 0..n {
                let s = (next() % 100) as i64;
                let len = 1 + (next() % 20) as i64;
                intervals.push((s, s + len));
            }
            let got = min_meeting_rooms(&intervals);
            let want = naive(&intervals);
            assert_eq!(got, want, "mismatch on intervals = {intervals:?}");
        }
    }

    #[test]
    fn large_random_runs() {
        // Smoke test: ensure we handle a larger input without panicking
        // and produce a sensible bound (depth <= n).
        let mut state: u64 = 0xDEAD_BEEF_CAFE_BABE;
        let mut next = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            state
        };
        let n: usize = 5_000;
        let mut intervals: Vec<(i64, i64)> = Vec::with_capacity(n);
        for _ in 0..n {
            let s = (next() % 10_000) as i64;
            let len = 1 + (next() % 200) as i64;
            intervals.push((s, s + len));
        }
        let rooms = min_meeting_rooms(&intervals);
        assert!(rooms >= 1);
        assert!(rooms as usize <= n);
    }
}
