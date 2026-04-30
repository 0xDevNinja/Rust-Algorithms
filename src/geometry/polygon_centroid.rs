//! Polygon centroid via the Shoelace-based weighted formula.
//!
//! For a simple polygon with vertices `(x_0, y_0), …, (x_{n-1}, y_{n-1})`
//! listed in order around the boundary, the geometric centroid
//! `(C_x, C_y)` is
//!
//! ```text
//!     C_x = (1 / (6A)) · Σ_{i=0}^{n-1} (x_i + x_{i+1}) · (x_i · y_{i+1} − x_{i+1} · y_i)
//!     C_y = (1 / (6A)) · Σ_{i=0}^{n-1} (y_i + y_{i+1}) · (x_i · y_{i+1} − x_{i+1} · y_i)
//! ```
//!
//! where `A` is the signed Shoelace area and indices wrap modulo `n`
//! (the last edge connects vertex `n-1` back to vertex `0`).
//!
//! Complexity: `O(n)` time, `O(1)` extra space.
//!
//! Caveat: the formula assumes a *simple* polygon (no self-intersections).
//! It is also undefined when the signed area is `0` — i.e. when all
//! vertices are collinear or otherwise degenerate — because the
//! `1 / (6A)` factor would divide by zero. In that case
//! [`polygon_centroid`] returns `None`.
//!
//! The result is independent of orientation: reversing the vertex order
//! flips the sign of both the area and each summand, so the ratios
//! defining `C_x` and `C_y` are unchanged.
//!
//! Vertices stored as `(f64, f64)` pairs.

/// Returns the geometric centroid of the simple polygon described by
/// `vertices`, or `None` if the input is degenerate.
///
/// `None` is returned when:
/// - `vertices.len() < 3` (a polygon needs at least three vertices), or
/// - the signed Shoelace area is exactly `0.0` (e.g. all vertices are
///   collinear), in which case the centroid is mathematically undefined.
///
/// For a well-formed simple polygon this returns `Some((C_x, C_y))` where
/// `(C_x, C_y)` is the geometric centroid (centre of mass of a uniform
/// lamina with the polygon's shape). The result is independent of the
/// vertex orientation (clockwise vs. counter-clockwise).
///
/// Runs in `O(n)` time and `O(1)` extra space.
pub fn polygon_centroid(vertices: &[(f64, f64)]) -> Option<(f64, f64)> {
    let n = vertices.len();
    if n < 3 {
        return None;
    }
    let mut signed_area_2 = 0.0_f64; // 2 * A
    let mut cx_acc = 0.0_f64;
    let mut cy_acc = 0.0_f64;
    for i in 0..n {
        let (x_i, y_i) = vertices[i];
        let (x_j, y_j) = vertices[(i + 1) % n];
        let cross = x_i.mul_add(y_j, -(x_j * y_i));
        signed_area_2 += cross;
        cx_acc += (x_i + x_j) * cross;
        cy_acc += (y_i + y_j) * cross;
    }
    if signed_area_2 == 0.0 {
        return None;
    }
    // signed_area_2 = 2A, so 1/(6A) = 1/(3 * signed_area_2).
    let inv = 1.0 / (3.0 * signed_area_2);
    Some((cx_acc * inv, cy_acc * inv))
}

#[cfg(test)]
mod tests {
    use super::polygon_centroid;
    use quickcheck_macros::quickcheck;
    use std::f64::consts::PI;

    const EPS: f64 = 1e-9;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() <= eps
    }

    fn approx_eq_pt(a: (f64, f64), b: (f64, f64), eps: f64) -> bool {
        approx_eq(a.0, b.0, eps) && approx_eq(a.1, b.1, eps)
    }

    #[test]
    fn empty_is_none() {
        let v: Vec<(f64, f64)> = Vec::new();
        assert_eq!(polygon_centroid(&v), None);
    }

    #[test]
    fn single_vertex_is_none() {
        let v = vec![(1.0, 2.0)];
        assert_eq!(polygon_centroid(&v), None);
    }

    #[test]
    fn two_vertices_is_none() {
        let v = vec![(0.0, 0.0), (1.0, 1.0)];
        assert_eq!(polygon_centroid(&v), None);
    }

    #[test]
    fn collinear_three_points_is_none() {
        // All three vertices on the line y = x → signed area is 0.
        let v = vec![(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)];
        assert_eq!(polygon_centroid(&v), None);
    }

    #[test]
    fn collinear_many_points_is_none() {
        let v = vec![(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)];
        assert_eq!(polygon_centroid(&v), None);
    }

    #[test]
    fn unit_square_centroid_is_center() {
        let v = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        let c = polygon_centroid(&v).unwrap();
        assert!(approx_eq_pt(c, (0.5, 0.5), EPS));
    }

    #[test]
    fn equilateral_triangle_centroid_equals_vertex_average() {
        // Equilateral triangle inscribed in the unit circle centred at the
        // origin; for any triangle, the centroid is the mean of the three
        // vertices.
        let mut v = Vec::with_capacity(3);
        for i in 0..3 {
            let theta = 2.0 * PI * (i as f64) / 3.0;
            v.push((theta.cos(), theta.sin()));
        }
        let mean = (
            v.iter().map(|p| p.0).sum::<f64>() / 3.0,
            v.iter().map(|p| p.1).sum::<f64>() / 3.0,
        );
        let c = polygon_centroid(&v).unwrap();
        assert!(approx_eq_pt(c, mean, 1e-12));
        // And by construction this should be the origin.
        assert!(approx_eq_pt(c, (0.0, 0.0), 1e-12));
    }

    #[test]
    fn regular_hexagon_centroid_is_center() {
        // Regular hexagon centred at (3, -2).
        let cx = 3.0;
        let cy = -2.0;
        let r = 2.5;
        let mut v = Vec::with_capacity(6);
        for i in 0..6 {
            let theta = 2.0 * PI * (i as f64) / 6.0;
            v.push((cx + r * theta.cos(), cy + r * theta.sin()));
        }
        let c = polygon_centroid(&v).unwrap();
        assert!(approx_eq_pt(c, (cx, cy), 1e-12));
    }

    #[test]
    fn translation_invariance() {
        let v = vec![(0.0, 0.0), (4.0, 0.0), (3.0, 3.0), (1.0, 3.0)];
        let dx = 5.0;
        let dy = -7.5;
        let shifted: Vec<(f64, f64)> = v.iter().map(|&(x, y)| (x + dx, y + dy)).collect();
        let c0 = polygon_centroid(&v).unwrap();
        let c1 = polygon_centroid(&shifted).unwrap();
        assert!(approx_eq_pt(c1, (c0.0 + dx, c0.1 + dy), EPS));
    }

    #[test]
    fn non_convex_l_shape() {
        // L-shape: a 2x2 square with the top-right 1x1 corner removed.
        // CCW boundary:
        let v = vec![
            (0.0, 0.0),
            (2.0, 0.0),
            (2.0, 1.0),
            (1.0, 1.0),
            (1.0, 2.0),
            (0.0, 2.0),
        ];
        // Decomposition into two unit-area pieces:
        //   bottom 2x1 rectangle  → area 2, centroid (1.0, 0.5)
        //   left   1x1 square      → area 1, centroid (0.5, 1.5)
        // Combined centroid =
        //   ((2 * 1.0 + 1 * 0.5) / 3, (2 * 0.5 + 1 * 1.5) / 3)
        //   = (2.5 / 3, 2.5 / 3).
        let expected = (2.5 / 3.0, 2.5 / 3.0);
        let c = polygon_centroid(&v).unwrap();
        assert!(approx_eq_pt(c, expected, EPS));
    }

    #[test]
    fn cw_and_ccw_give_same_centroid() {
        let ccw = vec![(0.0, 0.0), (4.0, 0.0), (3.0, 3.0), (1.0, 3.0)];
        let cw: Vec<(f64, f64)> = ccw.iter().rev().copied().collect();
        let c_ccw = polygon_centroid(&ccw).unwrap();
        let c_cw = polygon_centroid(&cw).unwrap();
        assert!(approx_eq_pt(c_ccw, c_cw, EPS));
    }

    // Property test: for any regular n-gon (3 ≤ n ≤ 10) with arbitrary
    // centre and positive radius, the computed centroid equals the centre
    // within a tight tolerance.
    #[allow(clippy::needless_pass_by_value)]
    #[quickcheck]
    fn quickcheck_regular_ngon_centroid_is_center(
        n_seed: u8,
        r_seed: u16,
        cx_seed: i16,
        cy_seed: i16,
    ) -> bool {
        let n = 3 + (n_seed as usize) % 8; // n in 3..=10
        let r = ((r_seed as f64) + 1.0) / 1000.0; // r in (0, ~65.5]
        let cx = (cx_seed as f64) / 100.0;
        let cy = (cy_seed as f64) / 100.0;
        let mut v = Vec::with_capacity(n);
        for i in 0..n {
            let theta = 2.0 * PI * (i as f64) / (n as f64);
            v.push((cx + r * theta.cos(), cy + r * theta.sin()));
        }
        let c = polygon_centroid(&v).unwrap();
        // Tolerance scales with coordinate magnitude. The centroid
        // computation involves products like (x_i + x_j) * cross, and the
        // cross terms are O(r * (|c| + r)). Dividing by 6A where A scales
        // with r^2, the absolute floating-point error is roughly
        // ε * (|c| + r) / r in each coordinate.
        let scale = (cx.abs() + cy.abs() + r) / r;
        let tol = 1e-9_f64.mul_add(scale, 1e-12);
        (c.0 - cx).abs() <= tol && (c.1 - cy).abs() <= tol
    }
}
