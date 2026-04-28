//! Point-in-polygon test via ray casting.
//!
//! Casts a horizontal ray from the query point toward `+∞` along the `x`
//! axis and counts how many polygon edges it crosses. An odd count means
//! the point lies inside the polygon, an even count means it lies outside
//! (Jordan curve theorem). Each edge is tested with the standard
//! half-open convention
//!
//! ```text
//!     (yi > py) != (yj > py)
//! ```
//!
//! which treats edges as half-open in `y`: the upper endpoint is included
//! and the lower endpoint is excluded. This avoids double-counting when
//! the ray meets a shared vertex between two edges that lie on the same
//! side of the ray, while still counting it once when the two edges
//! straddle the ray.
//!
//! Complexity: `O(n)` time, `O(1)` extra space, where `n` is the number
//! of polygon vertices.
//!
//! Caveats:
//! * **Boundary instability.** Behaviour for points lying *exactly* on a
//!   polygon edge is implementation-defined and unstable under floating
//!   point: the same edge may classify the point as inside or outside
//!   depending on rounding. Callers that need a robust on-boundary
//!   predicate should compose this routine with a separate point-on-edge
//!   test.
//! * **Ray through a vertex.** When the horizontal ray passes through a
//!   vertex, the half-open `(yi > py) != (yj > py)` rule still gives a
//!   well-defined count, but the count depends on which edge endpoints
//!   are strictly above the ray. The tests below pin the current
//!   behaviour for the canonical degenerate case so any future change is
//!   intentional.

/// Returns `true` if `point` lies strictly inside the simple polygon
/// described by `polygon`, using the ray-casting algorithm.
///
/// `polygon` is an ordered list of vertices `(x, y)`; the closing edge
/// from the last vertex back to the first is implicit. Orientation
/// (clockwise or counter-clockwise) does not matter.
///
/// Returns `false` for inputs with fewer than three vertices. Behaviour
/// for points on the polygon boundary is implementation-defined; see the
/// module-level documentation.
///
/// Runs in `O(n)` time and `O(1)` extra space.
#[must_use]
pub fn point_in_polygon(polygon: &[(f64, f64)], point: (f64, f64)) -> bool {
    let n = polygon.len();
    if n < 3 {
        return false;
    }
    let (px, py) = point;
    let mut inside = false;
    let mut j = n - 1;
    for i in 0..n {
        let (xi, yi) = polygon[i];
        let (xj, yj) = polygon[j];
        // Does the edge (i, j) straddle the horizontal ray at y = py?
        if (yi > py) != (yj > py) {
            // x-coordinate where the edge crosses y = py.
            let x_cross = (xj - xi) * (py - yi) / (yj - yi) + xi;
            if px < x_cross {
                inside = !inside;
            }
        }
        j = i;
    }
    inside
}

#[cfg(test)]
mod tests {
    use super::point_in_polygon;
    use quickcheck_macros::quickcheck;
    use std::f64::consts::PI;

    fn unit_square() -> Vec<(f64, f64)> {
        vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    }

    #[test]
    fn empty_polygon_returns_false() {
        let v: Vec<(f64, f64)> = Vec::new();
        assert!(!point_in_polygon(&v, (0.0, 0.0)));
    }

    #[test]
    fn single_vertex_polygon_returns_false() {
        let v = vec![(0.0, 0.0)];
        assert!(!point_in_polygon(&v, (0.0, 0.0)));
    }

    #[test]
    fn two_vertex_polygon_returns_false() {
        let v = vec![(0.0, 0.0), (1.0, 1.0)];
        assert!(!point_in_polygon(&v, (0.5, 0.5)));
    }

    #[test]
    fn unit_square_interior() {
        let sq = unit_square();
        assert!(point_in_polygon(&sq, (0.5, 0.5)));
        assert!(point_in_polygon(&sq, (0.25, 0.75)));
        assert!(point_in_polygon(&sq, (0.999, 0.001)));
    }

    #[test]
    fn unit_square_exterior() {
        let sq = unit_square();
        assert!(!point_in_polygon(&sq, (2.0, 2.0)));
        assert!(!point_in_polygon(&sq, (-0.1, 0.5)));
        assert!(!point_in_polygon(&sq, (0.5, 1.1)));
        assert!(!point_in_polygon(&sq, (0.5, -0.1)));
    }

    // Behaviour on the boundary (corners and edge midpoints) is
    // implementation-defined under floating-point ray casting. We pin the
    // current outcomes so any change is intentional. Per the module doc,
    // callers needing robust on-boundary classification must layer their
    // own predicate on top.
    #[test]
    fn unit_square_corner_pinned_behaviour() {
        let sq = unit_square();
        // (0,0) is the lower-left corner. The bottom edge (y constant
        // at 0) does not straddle the ray under the strict
        // `(yi > py) != (yj > py)` rule. The left edge crosses at x=0,
        // and `px < x_cross` is `0 < 0` which is false, so it does not
        // flip. The right edge crosses at x=1 with `0 < 1` true, which
        // flips the count once. Result: the lower-left corner is
        // reported as inside. Pin this so any future change is
        // intentional.
        assert!(point_in_polygon(&sq, (0.0, 0.0)));
    }

    #[test]
    fn unit_square_edge_midpoint_pinned_behaviour() {
        let sq = unit_square();
        // Bottom edge midpoint (0.5, 0): the bottom edge does not
        // straddle the ray under the strict `>` rule. The right edge
        // at x=1 does straddle and `0.5 < 1` flips the count once, so
        // the midpoint is reported as inside. Per the module doc,
        // on-boundary classification is implementation-defined; this
        // test pins the current outcome.
        assert!(point_in_polygon(&sq, (0.5, 0.0)));
    }

    #[test]
    fn triangle_interior_and_exterior() {
        let tri = vec![(0.0, 0.0), (4.0, 0.0), (2.0, 3.0)];
        assert!(point_in_polygon(&tri, (2.0, 1.0)));
        assert!(point_in_polygon(&tri, (1.0, 0.5)));
        assert!(!point_in_polygon(&tri, (5.0, 1.0)));
        assert!(!point_in_polygon(&tri, (2.0, 4.0)));
        assert!(!point_in_polygon(&tri, (-1.0, -1.0)));
    }

    #[test]
    fn concave_l_shape_concavity_is_outside() {
        // L-shaped hexagon occupying the unit square minus the
        // upper-right quadrant.
        //
        //   (0,2) +------+ (2,2)
        //         |      |
        //         |      +------+ (3,1)... no, this one is simpler:
        //
        // Vertices (CCW):
        //   (0,0) -> (2,0) -> (2,1) -> (1,1) -> (1,2) -> (0,2) -> back.
        let l = vec![
            (0.0, 0.0),
            (2.0, 0.0),
            (2.0, 1.0),
            (1.0, 1.0),
            (1.0, 2.0),
            (0.0, 2.0),
        ];
        // Interior of the L: lower-right arm and upper-left arm.
        assert!(point_in_polygon(&l, (1.5, 0.5)));
        assert!(point_in_polygon(&l, (0.5, 1.5)));
        // Concavity (upper-right square): outside the L.
        assert!(!point_in_polygon(&l, (1.5, 1.5)));
        // Far outside.
        assert!(!point_in_polygon(&l, (3.0, 3.0)));
    }

    #[test]
    fn regular_hexagon_interior_and_exterior() {
        // Regular hexagon centred at the origin with circumradius 1.
        let mut hex = Vec::with_capacity(6);
        for i in 0..6 {
            let theta = 2.0 * PI * (i as f64) / 6.0;
            hex.push((theta.cos(), theta.sin()));
        }
        // Origin is well inside.
        assert!(point_in_polygon(&hex, (0.0, 0.0)));
        // Inside the inscribed circle (radius cos(π/6) ≈ 0.866).
        assert!(point_in_polygon(&hex, (0.5, 0.5)));
        // Outside the circumscribed circle.
        assert!(!point_in_polygon(&hex, (1.5, 0.0)));
        assert!(!point_in_polygon(&hex, (-2.0, -2.0)));
    }

    #[test]
    fn ray_through_vertex_pinned_behaviour() {
        // Diamond centred at the origin; vertices on the axes.
        //   (0,1), (1,0), (0,-1), (-1,0)
        // A horizontal ray from any point with y == 0 passes exactly
        // through the side vertices (1,0) and (-1,0). With the half-open
        // (yi > py) != (yj > py) rule, both edges meeting at (1,0) have
        // one endpoint strictly above y=0 and one not, so behaviour is
        // well-defined; pin it.
        let diamond = vec![(0.0, 1.0), (1.0, 0.0), (0.0, -1.0), (-1.0, 0.0)];

        // Point on the ray, to the left of the right vertex: the ray
        // grazes the apex at (0,1) and apex at (0,-1) too. Current
        // half-open rule reports this as inside the diamond.
        assert!(point_in_polygon(&diamond, (-0.5, 0.0)));
        // Point on the ray, but outside the diamond (x > 1): outside.
        assert!(!point_in_polygon(&diamond, (2.0, 0.0)));
    }

    #[test]
    fn orientation_independence() {
        let ccw = unit_square();
        let cw: Vec<(f64, f64)> = ccw.iter().rev().copied().collect();
        for &p in &[(0.5, 0.5), (0.25, 0.75), (2.0, 2.0), (-0.5, 0.5)] {
            assert_eq!(point_in_polygon(&ccw, p), point_in_polygon(&cw, p));
        }
    }

    // Property test: for a regular n-gon (n ∈ 3..=10) inscribed in a
    // circle of radius r centred at the origin, every point strictly
    // inside the inscribed circle of radius r·cos(π/n) is inside, and
    // every point strictly outside the circumscribed circle of radius r
    // is outside.
    #[allow(clippy::needless_pass_by_value)]
    #[quickcheck]
    fn quickcheck_regular_ngon_inside_outside(
        n_seed: u8,
        r_seed: u16,
        ix: i16,
        iy: i16,
        ox: i16,
        oy: i16,
    ) -> bool {
        let n = 3 + (n_seed as usize) % 8; // n in 3..=10
        let r = ((r_seed as f64) + 1.0) / 100.0; // r in (0.01, ~656]
        let mut poly = Vec::with_capacity(n);
        for i in 0..n {
            let theta = 2.0 * PI * (i as f64) / (n as f64);
            poly.push((r * theta.cos(), r * theta.sin()));
        }

        // Point well inside the inscribed circle (radius r·cos(π/n)).
        // Use 90% of the inscribed radius to leave a safety margin.
        let r_in = 0.9 * r * (PI / (n as f64)).cos();
        let theta_in = 2.0 * PI * (((ix as i32) + (iy as i32) * 7) as f64) / 360.0;
        let p_in = (r_in * theta_in.cos(), r_in * theta_in.sin());

        // Point well outside the circumscribed circle. Use 1.1·r plus
        // an offset so the magnitude exceeds r comfortably.
        let r_out = 1.1_f64.mul_add(r, 1.0);
        let theta_out = 2.0 * PI * (((ox as i32) + (oy as i32) * 11) as f64) / 360.0;
        let p_out = (r_out * theta_out.cos(), r_out * theta_out.sin());

        point_in_polygon(&poly, p_in) && !point_in_polygon(&poly, p_out)
    }
}
