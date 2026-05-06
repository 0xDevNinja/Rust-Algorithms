//! Bentley-Ottmann sweep-line segment intersection.
//!
//! Given a set of `n` line segments in the plane, [`find_intersections`]
//! reports every intersection point among them.
//!
//! # Algorithm
//!
//! The classic Bentley-Ottmann sweep-line algorithm sweeps a vertical line
//! from left to right, maintaining the set of segments that currently cross
//! it (the *active set*) ordered by their y-coordinate at the sweep x. Three
//! types of events drive the sweep:
//!
//! * **Start** — the left endpoint of a segment; the segment is inserted into
//!   the active set and its new neighbours are tested for intersection.
//! * **End** — the right endpoint of a segment; the segment is removed from
//!   the active set and its former neighbours (now newly adjacent) are tested.
//! * **Intersect** — two segments swap their relative order; the intersection
//!   is recorded, the swap is performed, and the new outer neighbours are
//!   tested.
//!
//! # Complexity
//!
//! The canonical Bentley-Ottmann implementation achieves `O((n + k) log n)`
//! time where `k` is the number of intersections, using a balanced BST for
//! the active set. **This implementation uses a `Vec<usize>` re-sorted on
//! each event**, giving `O((n + k) · n)` time. This is asymptotically slower
//! but correct, shorter, and clearer — appropriate for an educational library.
//! Space is `O(n + k)`.
//!
//! # Preconditions
//!
//! The algorithm assumes the *general-position* simplifying conditions
//! standard in the literature:
//!
//! * **No vertical segments** — every segment must have `p1.0 != p2.0`.
//! * **No shared x-coordinates between endpoints** — all `2n` endpoints have
//!   distinct x-values.
//! * **No three segments concurrent** — no point lies on three or more
//!   segments simultaneously.
//!
//! If any precondition is violated the function may return duplicate points
//! or miss some intersections. Endpoint-only touches are not reported
//! (only interior crossings).

use std::collections::HashSet;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// A line segment defined by two endpoints.
///
/// Must not be vertical (`p1.0 == p2.0`); see module preconditions.
#[derive(Debug, Clone, Copy)]
pub struct Segment {
    /// First endpoint.
    pub p1: (f64, f64),
    /// Second endpoint.
    pub p2: (f64, f64),
}

impl Segment {
    /// Returns the y-coordinate of this segment at sweep-line x-position `x`.
    fn y_at(self, x: f64) -> f64 {
        let (x1, y1) = self.left();
        let (x2, y2) = self.right();
        let dx = x2 - x1;
        if dx.abs() < f64::EPSILON {
            return f64::midpoint(y1, y2);
        }
        y1 + (y2 - y1) * (x - x1) / dx
    }

    /// Left endpoint (smaller x).
    fn left(self) -> (f64, f64) {
        if self.p1.0 <= self.p2.0 {
            self.p1
        } else {
            self.p2
        }
    }

    /// Right endpoint (larger x).
    fn right(self) -> (f64, f64) {
        if self.p1.0 >= self.p2.0 {
            self.p1
        } else {
            self.p2
        }
    }
}

// ---------------------------------------------------------------------------
// Segment intersection helper
// ---------------------------------------------------------------------------

/// Twice the signed area of triangle `(p, q, r)`.
#[inline]
fn orient(p: (f64, f64), q: (f64, f64), r: (f64, f64)) -> f64 {
    (q.0 - p.0).mul_add(r.1 - p.1, -((q.1 - p.1) * (r.0 - p.0)))
}

/// Computes the proper (interior) intersection of segments `a` and `b`.
///
/// Returns `None` when the segments do not cross in their interiors
/// (parallel, endpoint-only touch, or collinear overlap).
fn intersect_interior(a: &Segment, b: &Segment) -> Option<(f64, f64)> {
    let (p1, p2) = (a.left(), a.right());
    let (p3, p4) = (b.left(), b.right());

    let d1 = orient(p3, p4, p1);
    let d2 = orient(p3, p4, p2);
    let d3 = orient(p1, p2, p3);
    let d4 = orient(p1, p2, p4);

    if !((d1 > 0.0 && d2 < 0.0) || (d1 < 0.0 && d2 > 0.0)) {
        return None;
    }
    if !((d3 > 0.0 && d4 < 0.0) || (d3 < 0.0 && d4 > 0.0)) {
        return None;
    }

    let r = (p2.0 - p1.0, p2.1 - p1.1);
    let s = (p4.0 - p3.0, p4.1 - p3.1);
    let denom = r.0.mul_add(s.1, -(r.1 * s.0));
    if denom == 0.0 {
        return None;
    }

    let qp = (p3.0 - p1.0, p3.1 - p1.1);
    let t = qp.0.mul_add(s.1, -(qp.1 * s.0)) / denom;
    Some((t.mul_add(r.0, p1.0), t.mul_add(r.1, p1.1)))
}

// ---------------------------------------------------------------------------
// Event queue
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum EventKind {
    Start,
    Intersect,
    End,
}

#[derive(Debug, Clone, Copy)]
struct Event {
    x: f64,
    y: f64,
    kind: EventKind,
    seg_a: usize,
    seg_b: usize,
}

/// Convert an `f64` to a `u64` that preserves total order, including
/// negative values. For positive floats, flip the sign bit; for negative
/// floats, flip all bits.
#[inline]
const fn f64_to_ord(v: f64) -> u64 {
    let bits = v.to_bits();
    // If the sign bit is set (negative), flip all bits; otherwise flip only
    // the sign bit. This maps the IEEE 754 representation to a sortable u64.
    if bits >> 63 != 0 {
        !bits
    } else {
        bits ^ (1u64 << 63)
    }
}

impl Event {
    const fn sort_key(&self) -> (u64, u64, EventKind) {
        (f64_to_ord(self.x), f64_to_ord(self.y), self.kind)
    }
}

impl PartialEq for Event {
    fn eq(&self, other: &Self) -> bool {
        self.sort_key() == other.sort_key()
    }
}

impl Eq for Event {}

impl PartialOrd for Event {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Event {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.sort_key().cmp(&other.sort_key())
    }
}

// ---------------------------------------------------------------------------
// Active-set helpers
// ---------------------------------------------------------------------------

/// Sort the active-set `Vec<usize>` by y-coordinate at sweep position `x`.
///
/// Using a plain `Vec<usize>` re-sorted on every event trades the O(log n)
/// BST operations of the strict algorithm for O(n log n) per event, which
/// is acceptable for clarity in an educational library.
fn sort_active(active: &mut [usize], segs: &[Segment], x: f64) {
    active.sort_by(|&a, &b| {
        segs[a]
            .y_at(x)
            .partial_cmp(&segs[b].y_at(x))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
}

fn position_of(active: &[usize], id: usize) -> Option<usize> {
    active.iter().position(|&s| s == id)
}

/// Schedule an intersection event between `a` and `b` if their crossing lies
/// strictly to the right of `sweep_x` and has not yet been scheduled at that
/// exact x-coordinate.
///
/// The deduplication key includes the intersection x-coordinate so that the
/// same pair can be rescheduled if they cross a second time (which cannot
/// happen with straight segments, but the key is conservative).
fn maybe_add_event(
    segs: &[Segment],
    events: &mut Vec<Event>,
    scheduled: &mut HashSet<(usize, usize, u64)>,
    a: usize,
    b: usize,
    sweep_x: f64,
) {
    if let Some((ix, iy)) = intersect_interior(&segs[a], &segs[b]) {
        if ix > sweep_x {
            let key = if a < b {
                (a, b, f64_to_ord(ix))
            } else {
                (b, a, f64_to_ord(ix))
            };
            if scheduled.insert(key) {
                events.push(Event {
                    x: ix,
                    y: iy,
                    kind: EventKind::Intersect,
                    seg_a: a,
                    seg_b: b,
                });
                events.sort_unstable();
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Find all pairwise interior intersection points among `segments`.
///
/// Returns a deduplicated list sorted by `(x, y)`. Endpoint-only touches
/// are excluded. See module documentation for complexity and preconditions.
#[must_use]
#[allow(clippy::too_many_lines)]
pub fn find_intersections(segments: &[Segment]) -> Vec<(f64, f64)> {
    if segments.len() < 2 {
        return Vec::new();
    }

    // Build initial event queue with start and end events.
    let mut events: Vec<Event> = Vec::with_capacity(segments.len() * 2);
    for (i, seg) in segments.iter().enumerate() {
        let (lx, ly) = seg.left();
        let (rx, ry) = seg.right();
        events.push(Event {
            x: lx,
            y: ly,
            kind: EventKind::Start,
            seg_a: i,
            seg_b: 0,
        });
        events.push(Event {
            x: rx,
            y: ry,
            kind: EventKind::End,
            seg_a: i,
            seg_b: 0,
        });
    }
    events.sort_unstable();

    let mut scheduled: HashSet<(usize, usize, u64)> = HashSet::new();
    let mut active: Vec<usize> = Vec::new();
    let mut results: Vec<(f64, f64)> = Vec::new();

    let mut ei = 0usize;
    while ei < events.len() {
        let ev = events[ei];
        ei += 1;

        match ev.kind {
            EventKind::Start => {
                let id = ev.seg_a;
                active.push(id);
                sort_active(&mut active, segments, ev.x);

                if let Some(pos) = position_of(&active, id) {
                    if pos > 0 {
                        maybe_add_event(
                            segments,
                            &mut events,
                            &mut scheduled,
                            id,
                            active[pos - 1],
                            ev.x,
                        );
                    }
                    if pos + 1 < active.len() {
                        maybe_add_event(
                            segments,
                            &mut events,
                            &mut scheduled,
                            id,
                            active[pos + 1],
                            ev.x,
                        );
                    }
                }
            }

            EventKind::End => {
                let id = ev.seg_a;
                if let Some(pos) = position_of(&active, id) {
                    if pos > 0 && pos + 1 < active.len() {
                        let (below, above) = (active[pos - 1], active[pos + 1]);
                        maybe_add_event(segments, &mut events, &mut scheduled, below, above, ev.x);
                    }
                }
                active.retain(|&s| s != id);
            }

            EventKind::Intersect => {
                let (a, b) = (ev.seg_a, ev.seg_b);
                results.push((ev.x, ev.y));

                // Find the two segments in the active set and swap them.
                // After the crossing, the segment that was lower is now upper
                // and vice versa. We identify which one should be lower after
                // the swap by evaluating y just past the crossing x.
                let pos_a = position_of(&active, a);
                let pos_b = position_of(&active, b);

                if let (Some(pa), Some(pb)) = (pos_a, pos_b) {
                    // Swap so that, after the crossing, the correct segment
                    // occupies each position. Evaluate y just past ev.x using
                    // a meaningful delta so floating-point does not collapse it.
                    // A relative delta of 1e-7 * (1 + |ev.x|) is safe for
                    // segment coordinates in the range [-128, 128].
                    let delta = 1e-7_f64.mul_add(ev.x.abs(), 1e-7);
                    let x_after = ev.x + delta;
                    let ya_after = segments[a].y_at(x_after);
                    let yb_after = segments[b].y_at(x_after);

                    // Determine which segment should be at the lower position
                    // after the crossing: the one with smaller y_after.
                    let (lo_seg, hi_seg) = if ya_after <= yb_after { (a, b) } else { (b, a) };
                    let lo = pa.min(pb);
                    let hi = pa.max(pb);

                    // Place lo_seg at position `lo` and hi_seg at position `hi`.
                    active[lo] = lo_seg;
                    active[hi] = hi_seg;

                    // Test new outer neighbours.
                    if lo > 0 {
                        maybe_add_event(
                            segments,
                            &mut events,
                            &mut scheduled,
                            active[lo],
                            active[lo - 1],
                            ev.x,
                        );
                    }
                    if hi + 1 < active.len() {
                        maybe_add_event(
                            segments,
                            &mut events,
                            &mut scheduled,
                            active[hi],
                            active[hi + 1],
                            ev.x,
                        );
                    }
                }
            }
        }
    }

    // Deduplicate and sort results by (x, y).
    results.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    });
    results.dedup_by(|a, b| (a.0 - b.0).abs() < 1e-9 && (a.1 - b.1).abs() < 1e-9);
    results
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::{find_intersections, intersect_interior, Segment};
    use quickcheck_macros::quickcheck;

    const EPS: f64 = 1e-7;

    fn pt_approx_eq(a: (f64, f64), b: (f64, f64)) -> bool {
        (a.0 - b.0).abs() < EPS && (a.1 - b.1).abs() < EPS
    }

    fn seg(x1: f64, y1: f64, x2: f64, y2: f64) -> Segment {
        Segment {
            p1: (x1, y1),
            p2: (x2, y2),
        }
    }

    // Brute-force oracle: test every pair.
    fn brute_force(segs: &[Segment]) -> Vec<(f64, f64)> {
        let mut out = Vec::new();
        for i in 0..segs.len() {
            for j in (i + 1)..segs.len() {
                if let Some(p) = intersect_interior(&segs[i], &segs[j]) {
                    out.push(p);
                }
            }
        }
        out.sort_by(|a, b| {
            a.0.partial_cmp(&b.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        });
        out.dedup_by(|a, b| (a.0 - b.0).abs() < EPS && (a.1 - b.1).abs() < EPS);
        out
    }

    #[test]
    fn empty_input_returns_empty() {
        assert!(find_intersections(&[]).is_empty());
    }

    #[test]
    fn single_segment_returns_empty() {
        assert!(find_intersections(&[seg(0.0, 0.0, 1.0, 1.0)]).is_empty());
    }

    #[test]
    fn two_non_intersecting_segments() {
        let segs = [seg(0.0, 0.0, 1.0, 0.0), seg(0.0, 1.0, 1.0, 1.0)];
        assert!(find_intersections(&segs).is_empty());
    }

    #[test]
    fn two_crossing_segments_known_point() {
        // Segments cross at (2, 2).
        let segs = [seg(0.0, 0.0, 4.0, 4.0), seg(0.0, 4.0, 4.0, 0.0)];
        let pts = find_intersections(&segs);
        assert_eq!(pts.len(), 1);
        assert!(pt_approx_eq(pts[0], (2.0, 2.0)));
    }

    #[test]
    fn triangle_three_intersections() {
        // Three segments chosen so each pair crosses and no two endpoints
        // share an x-coordinate.
        let segs = [
            seg(0.0, 0.0, 5.0, 5.0),
            seg(1.0, 4.0, 6.0, -1.0),
            seg(0.5, 3.0, 5.5, 1.0),
        ];
        let pts = find_intersections(&segs);
        let bf = brute_force(&segs);
        assert_eq!(
            pts.len(),
            bf.len(),
            "count mismatch: sweep={pts:?} brute={bf:?}"
        );
        for (p, q) in pts.iter().zip(bf.iter()) {
            assert!(pt_approx_eq(*p, *q), "point mismatch: {p:?} vs {q:?}");
        }
    }

    #[test]
    fn small_grid_matches_brute_force() {
        // 2 horizontal + 3 slanted segments.
        let segs = [
            seg(0.0, 1.0, 10.0, 1.0),
            seg(0.0, 3.0, 10.0, 3.0),
            seg(1.0, 0.0, 5.0, 5.0),
            seg(2.0, 5.0, 8.0, 0.0),
            seg(3.0, 0.5, 9.0, 4.5),
        ];
        let pts = find_intersections(&segs);
        let bf = brute_force(&segs);
        assert_eq!(pts.len(), bf.len(), "sweep={pts:?} brute={bf:?}");
        for (p, q) in pts.iter().zip(bf.iter()) {
            assert!(pt_approx_eq(*p, *q), "{p:?} vs {q:?}");
        }
    }

    /// Property: sweep result equals brute-force on inputs satisfying the
    /// general-position preconditions. Inputs that violate them are skipped.
    #[allow(clippy::needless_pass_by_value)]
    #[quickcheck]
    fn quickcheck_matches_brute_force(raw: Vec<(i8, i8, i8, i8)>) -> bool {
        let segs: Vec<Segment> = raw
            .iter()
            .take(8)
            .filter_map(|&(x1, y1, x2, y2)| {
                let (fx1, fy1) = (f64::from(x1), f64::from(y1));
                let (fx2, fy2) = (f64::from(x2), f64::from(y2));
                if (fx1 - fx2).abs() < 0.5 {
                    return None; // skip near-vertical
                }
                Some(Segment {
                    p1: (fx1, fy1),
                    p2: (fx2, fy2),
                })
            })
            .collect();

        if segs.len() < 2 {
            return true;
        }

        // Skip inputs where any two endpoints share an x-value.
        let mut xs: Vec<u64> = segs
            .iter()
            .flat_map(|s| [s.left().0.to_bits(), s.right().0.to_bits()])
            .collect();
        xs.sort_unstable();
        if xs.windows(2).any(|w| w[0] == w[1]) {
            return true;
        }

        // Skip inputs where an endpoint of one segment lies (nearly) on
        // another segment — these are degenerate cases that violate the
        // general-position assumption and may report differently between
        // sweep and brute force.
        let on_segment = |p: (f64, f64), s: &Segment| -> bool {
            let (ax, ay) = s.left();
            let (bx, by) = s.right();
            let cross = (bx - ax).mul_add(p.1 - ay, -((by - ay) * (p.0 - ax)));
            let len = (bx - ax).hypot(by - ay);
            if len == 0.0 {
                return false;
            }
            (cross / len).abs() < 1e-6 && p.0 >= ax.min(bx) - 1e-9 && p.0 <= ax.max(bx) + 1e-9
        };
        for i in 0..segs.len() {
            for j in 0..segs.len() {
                if i == j {
                    continue;
                }
                if on_segment(segs[i].p1, &segs[j]) || on_segment(segs[i].p2, &segs[j]) {
                    return true;
                }
            }
        }

        let sweep = find_intersections(&segs);
        let bf = brute_force(&segs);

        if sweep.len() != bf.len() {
            return false;
        }
        sweep
            .iter()
            .zip(bf.iter())
            .all(|(p, q)| pt_approx_eq(*p, *q))
    }
}
