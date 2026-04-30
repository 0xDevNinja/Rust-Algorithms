//! Convex hull via Andrew's monotone chain algorithm.
//!
//! Given a set of 2-D points, the convex hull is the smallest convex
//! polygon that contains them all. Andrew's monotone chain builds it in
//! `O(n log n)` time by:
//!
//! 1. Sorting the points lexicographically by `(x, y)`.
//! 2. Sweeping left-to-right to build the *lower* hull, popping any
//!    vertex that would create a non-left turn (cross product `<= 0`).
//! 3. Sweeping right-to-left to build the *upper* hull with the same
//!    rule.
//! 4. Concatenating the two chains, dropping the duplicated endpoints.
//!
//! Using a strict `>` comparison on the cross product means collinear
//! points on a hull edge are *dropped* — only the two extreme endpoints
//! of any maximal collinear run survive. The returned vertices are in
//! counter-clockwise order and the start point is not duplicated.
//!
//! Edge cases: an empty input returns an empty `Vec`; a single point or
//! two distinct points are returned unchanged; an input where all points
//! are collinear collapses to its two extreme endpoints.
//!
//! Complexity: `O(n log n)` time (sort dominated), `O(n)` extra space.

/// Cross product of vectors `o → a` and `o → b`.
///
/// Positive when `o → a → b` is a counter-clockwise turn, negative when
/// clockwise, and zero when the three points are collinear.
fn cross(o: (f64, f64), a: (f64, f64), b: (f64, f64)) -> f64 {
    (a.0 - o.0).mul_add(b.1 - o.1, -((a.1 - o.1) * (b.0 - o.0)))
}

/// Returns the convex hull of `points` in counter-clockwise order.
///
/// Uses Andrew's monotone chain in `O(n log n)` time. The start vertex is
/// not duplicated at the end. Collinear points lying on a hull edge are
/// excluded — only the extreme endpoints of any collinear run are kept.
///
/// Special cases:
/// - empty input → empty output;
/// - one or two distinct points → the input deduplicated, in lexicographic order;
/// - all points collinear → the two extreme endpoints.
pub fn convex_hull(points: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let n = points.len();
    if n == 0 {
        return Vec::new();
    }

    // Sort lexicographically by (x, y) and dedupe exact duplicates.
    let mut pts: Vec<(f64, f64)> = points.to_vec();
    pts.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    });
    pts.dedup();

    if pts.len() <= 2 {
        return pts;
    }

    let m = pts.len();
    let mut hull: Vec<(f64, f64)> = Vec::with_capacity(2 * m);

    // Lower hull.
    for &p in &pts {
        while hull.len() >= 2 && cross(hull[hull.len() - 2], hull[hull.len() - 1], p) <= 0.0 {
            hull.pop();
        }
        hull.push(p);
    }

    // Upper hull. `lower_count` marks where the upper hull begins so the
    // popping loop can't eat into the lower hull.
    let lower_count = hull.len() + 1;
    for &p in pts.iter().rev().skip(1) {
        while hull.len() >= lower_count
            && cross(hull[hull.len() - 2], hull[hull.len() - 1], p) <= 0.0
        {
            hull.pop();
        }
        hull.push(p);
    }

    // Drop the duplicated start point that closes the loop.
    hull.pop();
    hull
}

#[cfg(test)]
mod tests {
    use super::{convex_hull, cross};
    use quickcheck_macros::quickcheck;

    /// Brute-force check: every `q` lies on or inside the convex polygon
    /// `hull` (assumed CCW). Uses the sign of the cross product on each
    /// directed edge.
    fn point_on_or_inside(hull: &[(f64, f64)], q: (f64, f64)) -> bool {
        let n = hull.len();
        if n == 0 {
            return false;
        }
        if n == 1 {
            return hull[0] == q;
        }
        if n == 2 {
            // On the closed segment hull[0]–hull[1].
            let c = cross(hull[0], hull[1], q);
            if c.abs() > 1e-9 {
                return false;
            }
            let (ax, ay) = hull[0];
            let (bx, by) = hull[1];
            let dot = (q.0 - ax).mul_add(bx - ax, (q.1 - ay) * (by - ay));
            let len2 = (by - ay).mul_add(by - ay, (bx - ax).powi(2));
            dot >= -1e-9 && dot <= len2 + 1e-9
        } else {
            for i in 0..n {
                let a = hull[i];
                let b = hull[(i + 1) % n];
                if cross(a, b, q) < -1e-9 {
                    return false;
                }
            }
            true
        }
    }

    #[test]
    fn empty_input_returns_empty() {
        let pts: Vec<(f64, f64)> = Vec::new();
        assert!(convex_hull(&pts).is_empty());
    }

    #[test]
    fn single_point_returns_itself() {
        let pts = vec![(2.5, -1.0)];
        assert_eq!(convex_hull(&pts), vec![(2.5, -1.0)]);
    }

    #[test]
    fn two_distinct_points_returned_sorted() {
        let pts = vec![(3.0, 4.0), (0.0, 0.0)];
        assert_eq!(convex_hull(&pts), vec![(0.0, 0.0), (3.0, 4.0)]);
    }

    #[test]
    fn duplicate_points_deduplicated() {
        let pts = vec![(1.0, 1.0), (1.0, 1.0), (1.0, 1.0)];
        assert_eq!(convex_hull(&pts), vec![(1.0, 1.0)]);
    }

    #[test]
    fn three_collinear_returns_two_endpoints() {
        let pts = vec![(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)];
        let hull = convex_hull(&pts);
        assert_eq!(hull, vec![(0.0, 0.0), (2.0, 2.0)]);
    }

    #[test]
    fn all_collinear_horizontal_returns_endpoints() {
        let pts = vec![(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (5.0, 0.0), (-3.0, 0.0)];
        let hull = convex_hull(&pts);
        assert_eq!(hull, vec![(-3.0, 0.0), (5.0, 0.0)]);
    }

    #[test]
    fn triangle_returns_three_corners_ccw() {
        let pts = vec![(0.0, 0.0), (4.0, 0.0), (0.0, 3.0)];
        let hull = convex_hull(&pts);
        assert_eq!(hull.len(), 3);
        // Should be CCW starting from leftmost-bottom.
        assert_eq!(hull[0], (0.0, 0.0));
        // Verify CCW orientation by checking signed area > 0.
        let signed_area = (hull[1].0 - hull[0].0).mul_add(
            hull[2].1 - hull[0].1,
            -((hull[2].0 - hull[0].0) * (hull[1].1 - hull[0].1)),
        );
        assert!(signed_area > 0.0);
    }

    #[test]
    fn unit_square_returns_four_corners() {
        let pts = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        let hull = convex_hull(&pts);
        assert_eq!(hull.len(), 4);
        // Starts at lowest-leftmost and proceeds CCW.
        assert_eq!(hull[0], (0.0, 0.0));
        assert_eq!(hull[1], (1.0, 0.0));
        assert_eq!(hull[2], (1.0, 1.0));
        assert_eq!(hull[3], (0.0, 1.0));
    }

    #[test]
    fn interior_points_excluded() {
        // Square with a bunch of interior points scattered inside.
        let pts = vec![
            (0.0, 0.0),
            (4.0, 0.0),
            (4.0, 4.0),
            (0.0, 4.0),
            (1.0, 1.0),
            (2.0, 2.0),
            (3.0, 1.0),
            (1.5, 2.5),
            (2.0, 0.5),
        ];
        let hull = convex_hull(&pts);
        assert_eq!(hull.len(), 4);
        let expected: Vec<(f64, f64)> = vec![(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)];
        for v in &expected {
            assert!(hull.contains(v));
        }
    }

    #[test]
    fn collinear_edge_points_excluded() {
        // Square with extra points along its edges; those should be dropped
        // because we use strict `>` on the cross product.
        let pts = vec![
            (0.0, 0.0),
            (1.0, 0.0),
            (2.0, 0.0), // on bottom edge between corners
            (4.0, 0.0),
            (4.0, 2.0), // on right edge
            (4.0, 4.0),
            (2.0, 4.0), // on top edge
            (0.0, 4.0),
            (0.0, 2.0), // on left edge
        ];
        let hull = convex_hull(&pts);
        assert_eq!(hull.len(), 4);
        for v in &[(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)] {
            assert!(hull.contains(v));
        }
    }

    #[test]
    fn classic_example() {
        // A classic textbook example.
        let pts = vec![
            (0.0, 3.0),
            (1.0, 1.0),
            (2.0, 2.0),
            (4.0, 4.0),
            (0.0, 0.0),
            (1.0, 2.0),
            (3.0, 1.0),
            (3.0, 3.0),
        ];
        let hull = convex_hull(&pts);
        // Expected hull: (0,0) → (3,1) → (4,4) → (0,3) in CCW order.
        let expected: Vec<(f64, f64)> = vec![(0.0, 0.0), (3.0, 1.0), (4.0, 4.0), (0.0, 3.0)];
        assert_eq!(hull, expected);
    }

    #[test]
    fn ccw_orientation_signed_area_positive() {
        let pts = vec![(0.0, 0.0), (5.0, 0.0), (5.0, 5.0), (0.0, 5.0), (3.0, 2.0)];
        let hull = convex_hull(&pts);
        // Shoelace signed area should be positive (CCW).
        let n = hull.len();
        let mut s = 0.0;
        for i in 0..n {
            let (x0, y0) = hull[i];
            let (x1, y1) = hull[(i + 1) % n];
            s += x0.mul_add(y1, -(x1 * y0));
        }
        assert!(s > 0.0);
    }

    /// Property test: for any small input (≤ 20 points), every input
    /// point lies on or inside the hull, and every hull vertex is one
    /// of the input points.
    #[allow(clippy::needless_pass_by_value)]
    #[quickcheck]
    fn quickcheck_hull_contains_all_inputs(seeds: Vec<(i16, i16)>) -> bool {
        // Cap input size to keep runtime sane.
        let trimmed: Vec<(f64, f64)> = seeds
            .into_iter()
            .take(20)
            .map(|(a, b)| (f64::from(a) / 100.0, f64::from(b) / 100.0))
            .collect();
        let hull = convex_hull(&trimmed);

        // Every hull vertex must be one of the input points.
        for v in &hull {
            if !trimmed.contains(v) {
                return false;
            }
        }

        // Every input point must lie on or inside the hull.
        for &q in &trimmed {
            if !point_on_or_inside(&hull, q) {
                return false;
            }
        }
        true
    }
}
