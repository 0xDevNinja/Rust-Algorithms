//! Sutherland-Hodgman polygon clipping.
//!
//! Clips a *subject* polygon against a *convex clipping* polygon by processing
//! one half-plane at a time. For each directed edge of the clipper the entire
//! subject polygon is filtered: vertices inside the half-plane are kept,
//! vertices outside are dropped, and intersections with the clip edge are
//! inserted. The output is the polygon formed by the intersection of the
//! subject with the (convex) region bounded by the clipper.
//!
//! # Preconditions
//! * `clipper` **must** be convex and vertices **must** be listed in
//!   **counter-clockwise** order. If either precondition is violated the
//!   function does not panic but the result is unspecified.
//!
//! # Complexity
//! * Time: `O(n · m)` where `n = subject.len()` and `m = clipper.len()`.
//! * Space: `O(n + m)` for the working vertex buffer.

type Point = (f64, f64);

/// Returns a positive value when `p` is on the left (inside) side of the
/// directed edge from `a` to `b`, zero when it is exactly on the edge, and
/// a negative value when it is on the right (outside) side.
///
/// Equivalent to the signed area of the triangle `(a, b, p)` scaled by 2.
#[inline]
fn signed_area2(a: Point, b: Point, p: Point) -> f64 {
    (b.0 - a.0).mul_add(p.1 - a.1, -((b.1 - a.1) * (p.0 - a.0)))
}

/// Returns `true` if `p` is on or to the left of the directed edge `a → b`
/// (i.e. inside the CCW half-plane defined by that edge).
#[inline]
fn is_inside(a: Point, b: Point, p: Point) -> bool {
    signed_area2(a, b, p) >= 0.0
}

/// Returns the intersection of the infinite lines through `(a, b)` and
/// `(c, d)`.
///
/// Caller must ensure the lines are not parallel (the algorithm only calls
/// this when one endpoint is inside and the other is outside, so they
/// always cross).
#[inline]
fn line_intersect(a: Point, b: Point, c: Point, d: Point) -> Point {
    // Parametric form: P = a + t*(b-a), solved against line CD.
    let ab = (b.0 - a.0, b.1 - a.1);
    let cd = (d.0 - c.0, d.1 - c.1);
    let denom = ab.0.mul_add(cd.1, -(ab.1 * cd.0));
    let ac = (c.0 - a.0, c.1 - a.1);
    let t = ac.0.mul_add(cd.1, -(ac.1 * cd.0)) / denom;
    (t.mul_add(ab.0, a.0), t.mul_add(ab.1, a.1))
}

/// Clips `subject` against the convex polygon `clipper` using the
/// Sutherland-Hodgman algorithm.
///
/// # Arguments
/// * `subject` — The polygon to clip. Any simple polygon is accepted; the
///   result may be empty if the subject lies entirely outside the clipper.
/// * `clipper` — A **convex** polygon whose vertices are in
///   **counter-clockwise** order. Violating either condition produces an
///   unspecified result (no panic).
///
/// # Returns
/// The vertices of the clipped polygon in counter-clockwise order, or an
/// empty `Vec` if the intersection is empty.
pub fn clip(subject: &[Point], clipper: &[Point]) -> Vec<Point> {
    if subject.is_empty() || clipper.len() < 3 {
        return Vec::new();
    }

    let mut output: Vec<Point> = subject.to_vec();

    let m = clipper.len();
    for i in 0..m {
        if output.is_empty() {
            return Vec::new();
        }
        let a = clipper[i];
        let b = clipper[(i + 1) % m];

        let input = output.clone();
        output.clear();

        let n = input.len();
        for j in 0..n {
            // Walk each edge start → end; start is the previous vertex.
            let start = input[(n + j - 1) % n];
            let end = input[j];

            if is_inside(a, b, end) {
                if !is_inside(a, b, start) {
                    // Entering the half-plane: record the crossing first.
                    output.push(line_intersect(start, end, a, b));
                }
                output.push(end);
            } else if is_inside(a, b, start) {
                // Exiting the half-plane: record only the crossing.
                output.push(line_intersect(start, end, a, b));
            }
        }
    }

    output
}

#[cfg(test)]
mod tests {
    use super::clip;
    use crate::geometry::polygon_area::polygon_area;
    use quickcheck_macros::quickcheck;

    const EPS: f64 = 1e-9;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() <= eps
    }

    /// CCW unit square clipping polygon.
    fn unit_square() -> Vec<(f64, f64)> {
        vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    }

    #[test]
    fn empty_subject_returns_empty() {
        let result = clip(&[], &unit_square());
        assert!(result.is_empty());
    }

    #[test]
    fn subject_inside_clipper_unchanged_area() {
        // Small triangle fully inside the unit square.
        let subject = vec![(0.2, 0.2), (0.8, 0.2), (0.5, 0.8)];
        let result = clip(&subject, &unit_square());
        // The clipped polygon must have the same area as the original subject.
        assert!(approx_eq(
            polygon_area(&result),
            polygon_area(&subject),
            EPS,
        ));
    }

    #[test]
    fn subject_completely_outside_returns_empty() {
        // Triangle far to the right of the unit square.
        let subject = vec![(2.0, 0.0), (3.0, 0.0), (2.5, 1.0)];
        let result = clip(&subject, &unit_square());
        assert!(result.is_empty());
    }

    #[test]
    fn triangle_clipped_by_unit_square() {
        // Triangle A=(-0.5,-0.5), B=(1.5,-0.5), C=(0.5,1.5) clipped against
        // the unit square.  No vertex of the triangle is inside the square;
        // the triangle straddles all four sides.
        //
        // The clipped polygon has vertices (in CCW order):
        //   (1.0, 0.5), (0.75, 1.0), (0.25, 1.0), (0.0, 0.5),
        //   (0.0, 0.0), (1.0, 0.0)
        // Shoelace area = 0.875.
        let subject = vec![(-0.5, -0.5), (1.5, -0.5), (0.5, 1.5)];
        let result = clip(&subject, &unit_square());
        assert!(!result.is_empty(), "expected non-empty clipped polygon");
        let area = polygon_area(&result);
        assert!(
            approx_eq(area, 0.875, 1e-9),
            "expected area 0.875, got {area}"
        );
    }

    #[test]
    fn concave_subject_clipped_by_square() {
        // A concave (non-convex) "L"-shaped subject clipped by the unit square.
        // The L occupies [0,1]x[0,0.5] union [0,0.5]x[0.5,1]; area = 0.75.
        // Fully inside the unit square so the clipped area equals the L area.
        let subject = vec![
            (0.0, 0.0),
            (1.0, 0.0),
            (1.0, 0.5),
            (0.5, 0.5),
            (0.5, 1.0),
            (0.0, 1.0),
        ];
        let result = clip(&subject, &unit_square());
        assert!(approx_eq(polygon_area(&result), 0.75, EPS));
    }

    /// Property: clipping any polygon against a large bounding box that
    /// contains all vertices gives back approximately the original area.
    #[allow(clippy::needless_pass_by_value)]
    #[quickcheck]
    fn quickcheck_clip_by_huge_box_preserves_area(coords: Vec<(u8, u8)>) -> bool {
        if coords.len() < 3 {
            return true;
        }
        let subject: Vec<(f64, f64)> = coords
            .iter()
            .map(|&(x, y)| (f64::from(x), f64::from(y)))
            .collect();

        // A huge CCW bounding box that contains [0,255]x[0,255].
        let big_box = vec![(-1.0, -1.0), (256.0, -1.0), (256.0, 256.0), (-1.0, 256.0)];

        let result = clip(&subject, &big_box);
        let orig_area = polygon_area(&subject);
        let clipped_area = polygon_area(&result);
        (orig_area - clipped_area).abs() <= 1e-6_f64.max(1e-9 * orig_area)
    }
}
