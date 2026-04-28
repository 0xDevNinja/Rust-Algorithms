//! Polygon area via the Shoelace formula.
//!
//! For a polygon with vertices `(x_0, y_0), (x_1, y_1), …, (x_{n-1}, y_{n-1})`
//! listed in order around the boundary, the signed area is
//!
//! ```text
//!     A = 0.5 * Σ_{i=0}^{n-1} (x_i · y_{i+1} − x_{i+1} · y_i)
//! ```
//!
//! with index arithmetic taken modulo `n` (so the last edge wraps back to
//! vertex 0). The unsigned (absolute) area is `|A|`.
//!
//! The signed area is positive when the vertices are listed
//! counter-clockwise and negative when listed clockwise, so the sign of
//! `signed_polygon_area` doubles as an orientation test.
//!
//! Complexity: `O(n)` time, `O(1)` extra space.
//!
//! Caveat: the Shoelace formula assumes a *simple* polygon (no
//! self-intersections). For self-intersecting input, this routine still
//! returns the algebraic Shoelace value, which counts oppositely-wound
//! sub-regions with opposite signs and therefore does *not* equal the
//! geometric area enclosed by the curve.

/// Returns the signed area of the polygon described by `vertices`.
///
/// The result is positive when the vertices are oriented counter-clockwise
/// (in a standard mathematical coordinate system with `y` pointing up) and
/// negative when oriented clockwise. With fewer than three vertices the
/// polygon has no area and `0.0` is returned.
pub fn signed_polygon_area(vertices: &[(f64, f64)]) -> f64 {
    let n = vertices.len();
    if n < 3 {
        return 0.0;
    }
    let mut sum = 0.0_f64;
    for i in 0..n {
        let (x_i, y_i) = vertices[i];
        let (x_j, y_j) = vertices[(i + 1) % n];
        sum += x_i.mul_add(y_j, -(x_j * y_i));
    }
    0.5 * sum
}

/// Returns the absolute (unsigned) area of the polygon described by
/// `vertices`, computed with the Shoelace formula.
///
/// Returns `0.0` for inputs with fewer than three vertices. For
/// self-intersecting polygons this returns `|A|` where `A` is the
/// algebraic Shoelace value, which is *not* the geometric area enclosed
/// by the curve.
pub fn polygon_area(vertices: &[(f64, f64)]) -> f64 {
    signed_polygon_area(vertices).abs()
}

#[cfg(test)]
mod tests {
    use super::{polygon_area, signed_polygon_area};
    use quickcheck_macros::quickcheck;
    use std::f64::consts::PI;

    const EPS: f64 = 1e-9;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() <= eps
    }

    #[test]
    fn empty_polygon_is_zero() {
        let v: Vec<(f64, f64)> = Vec::new();
        assert_eq!(polygon_area(&v), 0.0);
        assert_eq!(signed_polygon_area(&v), 0.0);
    }

    #[test]
    fn single_point_is_zero() {
        let v = vec![(1.0, 2.0)];
        assert_eq!(polygon_area(&v), 0.0);
        assert_eq!(signed_polygon_area(&v), 0.0);
    }

    #[test]
    fn two_points_is_zero() {
        let v = vec![(0.0, 0.0), (3.0, 4.0)];
        assert_eq!(polygon_area(&v), 0.0);
        assert_eq!(signed_polygon_area(&v), 0.0);
    }

    #[test]
    fn unit_square() {
        let v = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        assert!(approx_eq(polygon_area(&v), 1.0, EPS));
        assert!(approx_eq(signed_polygon_area(&v), 1.0, EPS));
    }

    #[test]
    fn unit_triangle() {
        // Right triangle with legs of length 1 → area 1/2.
        let v = vec![(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)];
        assert!(approx_eq(polygon_area(&v), 0.5, EPS));
        assert!(approx_eq(signed_polygon_area(&v), 0.5, EPS));
    }

    #[test]
    fn rational_shape_known_area() {
        // Trapezoid with parallel sides of length 4 (bottom) and 2 (top)
        // and height 3 → area = (4 + 2) * 3 / 2 = 9.
        let v = vec![(0.0, 0.0), (4.0, 0.0), (3.0, 3.0), (1.0, 3.0)];
        assert!(approx_eq(polygon_area(&v), 9.0, EPS));
    }

    #[test]
    fn regular_hexagon() {
        // Regular hexagon centred at origin with side length 1.
        // Area = 1.5 * sqrt(3) * side^2.
        let side = 1.0_f64;
        let mut v = Vec::with_capacity(6);
        for i in 0..6 {
            let theta = 2.0 * PI * (i as f64) / 6.0;
            v.push((side * theta.cos(), side * theta.sin()));
        }
        let expected = 1.5 * (3.0_f64).sqrt() * side * side;
        assert!(approx_eq(polygon_area(&v), expected, 1e-12));
    }

    #[test]
    fn ccw_vs_cw_sign_flips_absolute_equal() {
        let ccw = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        let cw: Vec<(f64, f64)> = ccw.iter().rev().copied().collect();
        let s_ccw = signed_polygon_area(&ccw);
        let s_cw = signed_polygon_area(&cw);
        assert!(s_ccw > 0.0);
        assert!(s_cw < 0.0);
        assert!(approx_eq(s_ccw, -s_cw, EPS));
        assert!(approx_eq(polygon_area(&ccw), polygon_area(&cw), EPS));
    }

    #[test]
    fn translation_invariance() {
        let v = vec![(0.0, 0.0), (4.0, 0.0), (3.0, 3.0), (1.0, 3.0)];
        let shifted: Vec<(f64, f64)> = v.iter().map(|&(x, y)| (x + 5.0, y - 3.0)).collect();
        assert!(approx_eq(polygon_area(&v), polygon_area(&shifted), EPS));
        assert!(approx_eq(
            signed_polygon_area(&v),
            signed_polygon_area(&shifted),
            EPS,
        ));
    }

    #[test]
    fn self_intersecting_returns_algebraic_value() {
        // A "bowtie": two triangles of equal area wound oppositely; the
        // algebraic Shoelace value is therefore 0, even though the
        // geometric figure-eight encloses non-zero area.
        let v = vec![(0.0, 0.0), (1.0, 1.0), (1.0, 0.0), (0.0, 1.0)];
        assert!(approx_eq(signed_polygon_area(&v), 0.0, EPS));
        assert!(approx_eq(polygon_area(&v), 0.0, EPS));
    }

    // Property test: for every regular n-gon (3 ≤ n ≤ 10) inscribed in a
    // circle of radius r > 0 centred at the origin, the Shoelace area
    // matches the closed form `0.5 * n * r^2 * sin(2π / n)`.
    #[allow(clippy::needless_pass_by_value)]
    #[quickcheck]
    fn quickcheck_regular_ngon_area(n_seed: u8, r_seed: u16) -> bool {
        let n = 3 + (n_seed as usize) % 8; // n in 3..=10
                                           // r in (0, ~65.5]; avoid zero radius.
        let r = ((r_seed as f64) + 1.0) / 1000.0;
        let mut v = Vec::with_capacity(n);
        for i in 0..n {
            let theta = 2.0 * PI * (i as f64) / (n as f64);
            v.push((r * theta.cos(), r * theta.sin()));
        }
        let expected = 0.5 * (n as f64) * r * r * (2.0 * PI / (n as f64)).sin();
        let got = polygon_area(&v);
        // Scale tolerance by expected magnitude to handle small radii.
        let tol = 1e-12_f64.mul_add(expected, 1e-9);
        (got - expected).abs() <= tol
    }
}
