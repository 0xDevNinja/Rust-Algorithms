//! Area of the union of axis-aligned rectangles via sweep-line and
//! coordinate-compressed segment tree (Klee's algorithm). O(n log n) time, O(n) space.

#[derive(Debug, Copy, Clone)]
enum SegStatus {
    Open,
    Close,
}

#[derive(Debug, Copy, Clone)]
struct Event {
    x: i64,
    y1: i64,
    y2: i64,
    status: SegStatus,
}

#[derive(Debug, Clone)]
struct CoverTree {
    cover_count: Vec<i32>,
    covered_len: Vec<i64>,
    ys: Vec<i64>,
}

impl CoverTree {
    /// Builds a zero-initialised tree over the coordinate-compressed y-axis.
    /// `ys` must be sorted and deduplicated; the tree covers `ys.len() - 1` intervals.
    pub fn from(ys: Vec<i64>) -> Self {
        let interval_count = ys.len() - 1;
        // the segment tree is stored as two flat arrays indexed from 1 (root)
        let cover_count = vec![0i32; 4 * interval_count];
        let covered_len = vec![0i64; 4 * interval_count];

        Self {
            cover_count,
            covered_len,
            ys,
        }
    }

    /// Returns the total y-length currently covered at `node`.
    /// Called on the root (node 1) to get the total covered y-length across all intervals.
    fn total_covered_len(&self, node: usize) -> i64 {
        self.covered_len[node]
    }

    /// Returns the total y-length currently covered at the tree's root node.
    pub fn root_covered_len(&self) -> i64 {
        self.total_covered_len(1)
    }

    /// Increments or decrements the coverage count at `node` based on `status`.
    pub fn update_count(&mut self, node: usize, status: SegStatus) {
        self.cover_count[node] += match status {
            SegStatus::Open => 1,
            SegStatus::Close => -1,
        };
    }

    /// Recomputes `covered_len[node]` from the current state of `cover_count` and its children.
    pub fn pull_up(&mut self, node: usize, lo: usize, hi: usize) {
        self.covered_len[node] = if self.cover_count[node] > 0 {
            // at least one rectangle fully covers this range
            self.ys[hi + 1] - self.ys[lo]
        } else if lo == hi {
            // leaf with no direct coverage
            0
        } else {
            // internal node with no direct coverage — sum the children
            self.covered_len[2 * node] + self.covered_len[2 * node + 1]
        };
    }

    /// Translates an event's y-values into y-interval indices into the tree.
    pub fn event_interval_range(&self, event: &Event) -> (usize, usize) {
        let lo = self.ys.partition_point(|&v| v < event.y1);
        let hi = self.ys.partition_point(|&v| v < event.y2) - 1;
        (lo, hi)
    }

    /// Recursively applies an open or close event to the y-interval range `[event_lo, event_hi]`,
    /// then pulls up `covered_len` on the way back out.
    fn seg_update(&mut self, node: usize, lo: usize, hi: usize, event: &Event) {
        let event_lo = self.ys.partition_point(|&v| v < event.y1);
        let event_hi = self.ys.partition_point(|&v| v < event.y2) - 1;

        if event_hi < lo || hi < event_lo {
            // this node's range is entirely outside the event's range
            return;
        }
        if event_lo <= lo && hi <= event_hi {
            // this node's range is entirely inside the event's range — stop here,
            // no need to recurse; cover_count acts as a lazy accumulator
            self.update_count(node, event.status);
        } else {
            // partial overlap — push the update down to the children
            let mid = lo + (hi - lo) / 2;
            self.seg_update(2 * node, lo, mid, event);
            self.seg_update(2 * node + 1, mid + 1, hi, event);
        }
        self.pull_up(node, lo, hi);
    }
}

/// Produces one `Open` and one `Close` event per rectangle, sorted by x.
fn build_events(rects: &[(i64, i64, i64, i64)]) -> Vec<Event> {
    let mut events: Vec<Event> = rects
        .iter()
        .flat_map(|&(x1, y1, x2, y2)| {
            [
                Event {
                    x: x1,
                    y1,
                    y2,
                    status: SegStatus::Open,
                },
                Event {
                    x: x2,
                    y1,
                    y2,
                    status: SegStatus::Close,
                },
            ]
        })
        .collect();
    events.sort_unstable_by_key(|e| e.x);
    events
}

/// Collects all y-coordinates from the rectangles, sorted and deduplicated.
/// These become the boundaries of the coordinate-compressed y-intervals.
fn build_unique_ys(rects: &[(i64, i64, i64, i64)]) -> Vec<i64> {
    let mut ys: Vec<i64> = rects.iter().flat_map(|&(_, y1, _, y2)| [y1, y2]).collect();
    ys.sort_unstable();
    ys.dedup();
    ys
}

/// Returns the area covered by the union of `rects`.
///
/// Each rectangle is `(x1, y1, x2, y2)` with `x1 < x2` and `y1 < y2`.
/// Returns `0` for empty input.
pub fn klee(rects: &[(i64, i64, i64, i64)]) -> i64 {
    if rects.is_empty() {
        return 0;
    }

    let ys = build_unique_ys(rects);

    let interval_count = ys.len() - 1;
    if interval_count == 0 {
        // return 0 if the rectangles are all flat (height 0)
        return 0;
    }

    let mut cover_tree = CoverTree::from(ys);

    let events = build_events(rects);

    let mut area = 0i64;
    let mut prev_x = events[0].x;

    // 2 events for each rectangle (open and close)
    for event in events {
        // accumulate the area of the slab between the previous and current x;
        // `root_covered_len` returns the total covered y-length
        let slab_width = event.x - prev_x;
        area += slab_width * cover_tree.root_covered_len();
        let (lo, hi) = cover_tree.event_interval_range(&event);
        assert!(lo <= hi, "`y1` must be < `y2` for every rectangle");
        // update the cover tree after each event (O(log n))
        cover_tree.seg_update(1, 0, interval_count - 1, &event);
        prev_x = event.x;
    }

    area
}

#[cfg(test)]
mod tests {
    use super::klee;
    use quickcheck_macros::quickcheck;

    #[test]
    fn empty_input() {
        assert_eq!(klee(&[]), 0);
    }

    #[test]
    fn single_rects() {
        assert_eq!(klee(&[(0, 0, 3, 4)]), 12); // 3 * 4 = 12
        assert_eq!(klee(&[(1, 1, 5, 6)]), 20); // (5-1) * (6-1) = 20
    }

    #[test]
    fn two_disjoint() {
        assert_eq!(klee(&[(0, 0, 1, 1), (2, 2, 3, 3)]), 2); // 1*1 + (3-2)*(3-2) = 2
        assert_eq!(klee(&[(1, 1, 5, 5), (5, 5, 8, 8)]), 25); // (5-1)*(5-1) + (8-5)*(8-5) = 16 + 9 = 25
    }

    #[test]
    fn two_overlapping() {
        // union = 4 + 4 - 1 = 7
        assert_eq!(klee(&[(0, 0, 2, 2), (1, 1, 3, 3)]), 4 + 4 - 1);
        // union = 25 + 25 - 4 * 4 = 34
        assert_eq!(klee(&[(0, 0, 5, 5), (1, 1, 6, 6)]), 25 + 25 - 16);
    }

    #[test]
    fn full_containment() {
        assert_eq!(klee(&[(0, 0, 4, 4), (1, 1, 3, 3)]), 16); // 4 * 4 = 16
        assert_eq!(klee(&[(4, 4, 9, 9), (5, 5, 9, 9)]), 25); // 5 * 5 = 25
    }

    #[quickcheck]
    fn prop_matches_brute_force(rects: Vec<(i8, i8, i8, i8)>) -> bool {
        let rects: Vec<(i64, i64, i64, i64)> = rects
            .into_iter()
            .filter_map(|(x1, y1, x2, y2)| {
                let (x1, x2) = (i64::from(x1), i64::from(x2));
                let (y1, y2) = (i64::from(y1), i64::from(y2));
                (x1 < x2 && y1 < y2).then_some((x1, y1, x2, y2))
            })
            .collect();
        klee(&rects) == brute_force(&rects)
    }

    fn brute_force(rects: &[(i64, i64, i64, i64)]) -> i64 {
        if rects.is_empty() {
            return 0;
        }
        let mut xs: Vec<i64> = rects.iter().flat_map(|&(x1, _, x2, _)| [x1, x2]).collect();
        let mut ys: Vec<i64> = rects.iter().flat_map(|&(_, y1, _, y2)| [y1, y2]).collect();
        xs.sort_unstable();
        xs.dedup();
        ys.sort_unstable();
        ys.dedup();
        let mut area = 0i64;
        for xi in 0..xs.len() - 1 {
            for yi in 0..ys.len() - 1 {
                let (cx, cy) = (xs[xi], ys[yi]);
                if rects
                    .iter()
                    .any(|&(x1, y1, x2, y2)| x1 <= cx && cx < x2 && y1 <= cy && cy < y2)
                {
                    area += (xs[xi + 1] - xs[xi]) * (ys[yi + 1] - ys[yi]);
                }
            }
        }
        area
    }
}
