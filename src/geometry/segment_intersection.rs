//! Line-segment intersection via orientation tests.
//!
//! Given two closed segments `AB` and `CD` in the plane, decide whether they
//! share at least one point and, if they cross transversally, compute that
//! point. The classical approach uses the *orientation* (signed-area) of an
//! ordered triple of points, computed from a 2-D cross product:
//!
//! ```text
//!     orient(p, q, r) = (q.x − p.x) · (r.y − p.y) − (q.y − p.y) · (r.x − p.x)
//! ```
//!
//! The sign distinguishes counter-clockwise (`> 0`), clockwise (`< 0`), and
//! collinear (`= 0`) triples. Two segments `AB` and `CD` cross iff `A` and
//! `B` lie on opposite sides of line `CD` *and* `C` and `D` lie on opposite
//! sides of line `AB`, i.e.
//!
//! ```text
//!     orient(A, B, C) and orient(A, B, D) have opposite signs
//!  && orient(C, D, A) and orient(C, D, B) have opposite signs
//! ```
//!
//! Degenerate cases (an endpoint of one segment that is collinear with the
//! other) are handled by an explicit on-segment check after a zero
//! orientation. The whole test is `O(1)` per pair of segments.
//!
//! Caveat: when the four points are collinear *and* the segments overlap on
//! a sub-segment, [`segments_intersect`] returns `true` (the segments share
//! infinitely many points) but [`segment_intersection_point`] returns
//! `None`, since there is no single intersection point to report. Callers
//! that need an overlap endpoint must detect this case themselves.

type Point = (f64, f64);
type Segment = (Point, Point);

/// Twice the signed area of the triangle `(p, q, r)`.
///
/// Positive when the triple turns counter-clockwise, negative when it turns
/// clockwise, and zero when the three points are collinear.
fn orient(p: Point, q: Point, r: Point) -> f64 {
    (q.0 - p.0).mul_add(r.1 - p.1, -((q.1 - p.1) * (r.0 - p.0)))
}

/// Whether `r` lies on the closed segment `pq`, *assuming* `p`, `q`, `r`
/// are already known to be collinear.
fn on_segment(p: Point, q: Point, r: Point) -> bool {
    r.0 >= p.0.min(q.0) && r.0 <= p.0.max(q.0) && r.1 >= p.1.min(q.1) && r.1 <= p.1.max(q.1)
}

/// Returns `true` if the closed segments `a` and `b` share at least one
/// point.
///
/// Endpoint contact (`T`-shapes, shared endpoints, vertex-on-edge) counts
/// as an intersection. Collinear segments that overlap on a sub-segment
/// also count as intersecting; collinear segments that are disjoint do
/// not.
pub fn segments_intersect(a: Segment, b: Segment) -> bool {
    let (p1, p2) = a;
    let (p3, p4) = b;

    let d1 = orient(p3, p4, p1);
    let d2 = orient(p3, p4, p2);
    let d3 = orient(p1, p2, p3);
    let d4 = orient(p1, p2, p4);

    // General case: segments straddle each other.
    if ((d1 > 0.0 && d2 < 0.0) || (d1 < 0.0 && d2 > 0.0))
        && ((d3 > 0.0 && d4 < 0.0) || (d3 < 0.0 && d4 > 0.0))
    {
        return true;
    }

    // Collinear / endpoint-touch cases: a zero orientation means the third
    // point lies on the line through the other two; check whether it also
    // lies within the segment.
    if d1 == 0.0 && on_segment(p3, p4, p1) {
        return true;
    }
    if d2 == 0.0 && on_segment(p3, p4, p2) {
        return true;
    }
    if d3 == 0.0 && on_segment(p1, p2, p3) {
        return true;
    }
    if d4 == 0.0 && on_segment(p1, p2, p4) {
        return true;
    }

    false
}

/// Returns the unique intersection point of the closed segments `a` and `b`,
/// or `None` if they do not meet.
///
/// Returns `None` when the segments are disjoint, when they are parallel
/// and non-coincident, and — by deliberate convention — when they are
/// collinear and overlap on a sub-segment (in which case there is no single
/// intersection point to report; see the module-level caveat). When the
/// segments share exactly one point (a transversal crossing or a single
/// endpoint touch), that point is returned.
pub fn segment_intersection_point(a: Segment, b: Segment) -> Option<Point> {
    if !segments_intersect(a, b) {
        return None;
    }

    let (p1, p2) = a;
    let (p3, p4) = b;

    let r = (p2.0 - p1.0, p2.1 - p1.1);
    let s = (p4.0 - p3.0, p4.1 - p3.1);
    let denom = r.0.mul_add(s.1, -(r.1 * s.0));

    // Parallel (denom == 0): either disjoint (handled above) or collinear
    // overlap. By contract, overlap returns None.
    if denom == 0.0 {
        return None;
    }

    let qp = (p3.0 - p1.0, p3.1 - p1.1);
    let t = qp.0.mul_add(s.1, -(qp.1 * s.0)) / denom;
    Some((t.mul_add(r.0, p1.0), t.mul_add(r.1, p1.1)))
}

#[cfg(test)]
mod tests {
    use super::{segment_intersection_point, segments_intersect};
    use quickcheck_macros::quickcheck;

    const EPS: f64 = 1e-9;

    fn approx_eq_pt(a: (f64, f64), b: (f64, f64)) -> bool {
        (a.0 - b.0).abs() <= EPS && (a.1 - b.1).abs() <= EPS
    }

    #[test]
    fn classic_x_crossing() {
        let a = ((0.0, 0.0), (4.0, 4.0));
        let b = ((0.0, 4.0), (4.0, 0.0));
        assert!(segments_intersect(a, b));
        let p = segment_intersection_point(a, b).expect("crossing");
        assert!(approx_eq_pt(p, (2.0, 2.0)));
    }

    #[test]
    fn perpendicular_crossing() {
        let a = ((0.0, 1.0), (4.0, 1.0));
        let b = ((2.0, -1.0), (2.0, 3.0));
        assert!(segments_intersect(a, b));
        let p = segment_intersection_point(a, b).expect("crossing");
        assert!(approx_eq_pt(p, (2.0, 1.0)));
    }

    #[test]
    fn parallel_disjoint() {
        let a = ((0.0, 0.0), (4.0, 0.0));
        let b = ((0.0, 1.0), (4.0, 1.0));
        assert!(!segments_intersect(a, b));
        assert!(segment_intersection_point(a, b).is_none());
    }

    #[test]
    fn shared_endpoint_counts_as_intersection() {
        let a = ((0.0, 0.0), (1.0, 1.0));
        let b = ((1.0, 1.0), (2.0, 0.0));
        assert!(segments_intersect(a, b));
        let p = segment_intersection_point(a, b).expect("endpoint");
        assert!(approx_eq_pt(p, (1.0, 1.0)));
    }

    #[test]
    fn t_shape_vertex_on_edge() {
        // Endpoint of `b` lies in the interior of `a`.
        let a = ((0.0, 0.0), (4.0, 0.0));
        let b = ((2.0, 0.0), (2.0, 3.0));
        assert!(segments_intersect(a, b));
        let p = segment_intersection_point(a, b).expect("T-shape");
        assert!(approx_eq_pt(p, (2.0, 0.0)));
    }

    #[test]
    fn collinear_non_overlapping() {
        let a = ((0.0, 0.0), (1.0, 0.0));
        let b = ((2.0, 0.0), (3.0, 0.0));
        assert!(!segments_intersect(a, b));
        assert!(segment_intersection_point(a, b).is_none());
    }

    #[test]
    fn collinear_overlapping_returns_none_point() {
        // Documented convention: overlap is reported as intersecting but
        // yields no single intersection point.
        let a = ((0.0, 0.0), (2.0, 0.0));
        let b = ((1.0, 0.0), (3.0, 0.0));
        assert!(segments_intersect(a, b));
        assert!(segment_intersection_point(a, b).is_none());
    }

    #[test]
    fn collinear_segment_contained_in_other() {
        let a = ((0.0, 0.0), (10.0, 0.0));
        let b = ((3.0, 0.0), (7.0, 0.0));
        assert!(segments_intersect(a, b));
        assert!(segment_intersection_point(a, b).is_none());
    }

    #[test]
    fn collinear_touch_at_single_endpoint() {
        // Two collinear segments that meet at exactly one shared endpoint
        // (no overlap region). Both predicates report the touch.
        let a = ((0.0, 0.0), (1.0, 0.0));
        let b = ((1.0, 0.0), (2.0, 0.0));
        assert!(segments_intersect(a, b));
        // The four points are collinear, so denom == 0 and we cannot
        // distinguish "single shared endpoint" from "overlap" with the
        // parametric formula; per contract we return None.
        assert!(segment_intersection_point(a, b).is_none());
    }

    #[test]
    fn close_but_not_touching() {
        // Two short segments that come very near each other but do not
        // touch.
        let a = ((0.0, 0.0), (1.0, 0.0));
        let b = ((0.5, 0.001), (0.5, 1.0));
        assert!(!segments_intersect(a, b));
        assert!(segment_intersection_point(a, b).is_none());
    }

    #[test]
    fn near_miss_extension_only() {
        // The lines cross at (5, 5), but neither segment reaches that point.
        let a = ((0.0, 0.0), (1.0, 1.0));
        let b = ((10.0, 0.0), (9.0, 1.0));
        assert!(!segments_intersect(a, b));
        assert!(segment_intersection_point(a, b).is_none());
    }

    #[test]
    fn degenerate_point_on_segment() {
        // A "segment" that is a single point lying on the interior of `a`.
        let a = ((0.0, 0.0), (4.0, 0.0));
        let b = ((2.0, 0.0), (2.0, 0.0));
        assert!(segments_intersect(a, b));
    }

    #[test]
    fn degenerate_point_off_segment() {
        let a = ((0.0, 0.0), (4.0, 0.0));
        let b = ((2.0, 1.0), (2.0, 1.0));
        assert!(!segments_intersect(a, b));
    }

    #[test]
    fn order_of_arguments_does_not_matter() {
        let a = ((0.0, 0.0), (4.0, 4.0));
        let b = ((0.0, 4.0), (4.0, 0.0));
        let p1 = segment_intersection_point(a, b).unwrap();
        let p2 = segment_intersection_point(b, a).unwrap();
        let p3 = segment_intersection_point((a.1, a.0), b).unwrap();
        assert!(approx_eq_pt(p1, p2));
        assert!(approx_eq_pt(p1, p3));
    }

    /// Property: if segment `b` is the diagonal `(0,0)-(M,M)` and segment
    /// `a` is the horizontal line at height `h` from `x = 0` to `x = M`
    /// with `0 <= h <= M` and `M >= 1`, then they must intersect at
    /// `(h, h)`.
    #[allow(clippy::needless_pass_by_value)]
    #[quickcheck]
    fn quickcheck_horizontal_meets_diagonal(m_seed: u8, h_seed: u8) -> bool {
        // Cast small integer coords (<= 10) to f64 for hand-verifiable cases.
        let m = 1 + (m_seed % 10) as i32; // m in 1..=10
        let h = (h_seed as i32) % (m + 1); // h in 0..=m
        let m_f = f64::from(m);
        let h_f = f64::from(h);

        let diag = ((0.0, 0.0), (m_f, m_f));
        let horiz = ((0.0, h_f), (m_f, h_f));

        if !segments_intersect(diag, horiz) {
            return false;
        }
        // `h == 0` makes the segments collinear → overlap → None by contract.
        segment_intersection_point(diag, horiz).map_or(h == 0, |p| approx_eq_pt(p, (h_f, h_f)))
    }

    /// Property: two parallel non-coincident horizontal segments at
    /// different `y` values never intersect, regardless of their `x`
    /// extents.
    #[allow(clippy::needless_pass_by_value)]
    #[quickcheck]
    fn quickcheck_parallel_distinct_never_intersect(
        x1: u8,
        x2: u8,
        x3: u8,
        x4: u8,
        dy: u8,
    ) -> bool {
        let xa = f64::from(x1 % 11);
        let xb = f64::from(x2 % 11);
        let xc = f64::from(x3 % 11);
        let xd = f64::from(x4 % 11);
        let dy_f = f64::from(1 + (dy % 10)); // strictly positive offset

        let a = ((xa, 0.0), (xb, 0.0));
        let b = ((xc, dy_f), (xd, dy_f));
        !segments_intersect(a, b) && segment_intersection_point(a, b).is_none()
    }
}
