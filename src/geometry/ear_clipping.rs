//! Polygon triangulation via the ear-clipping algorithm.
//!
//! Decomposes a *simple* polygon (no self-intersections, no holes) with `n`
//! vertices into `n - 2` non-overlapping triangles whose union exactly covers
//! the polygon interior.
//!
//! ## Algorithm
//! A vertex `v` is an *ear tip* when the triangle formed by its predecessor
//! `prev`, itself, and its successor `next` is:
//!
//! 1. **Convex** — the turn `prev → v → next` is a left turn (positive
//!    cross product) in the working CCW vertex list.
//! 2. **Empty** — no other active vertex lies strictly inside that triangle.
//!
//! The algorithm repeatedly finds an ear, emits its triangle, removes the ear
//! tip from the active vertex list, and repeats until only three vertices
//! remain (one final triangle).
//!
//! If the input polygon is oriented clockwise the vertices are reversed before
//! processing so that the convexity check always uses the same sign. The
//! emitted triangle vertices preserve the reversed order, so areas are still
//! positive but winding of individual triangles will match the normalised CCW
//! list rather than the original CW input.
//!
//! ## Complexity
//! * Time: `O(n²)` — up to `O(n)` ears are clipped, and each ear search
//!   scans `O(n)` vertices for the emptiness test.
//! * Space: `O(n)` — one `Vec` of active indices of length `n`.
//!
//! ## Preconditions
//! * The polygon must be **simple**: no self-intersections and no holes.
//! * Consecutive duplicate vertices and zero-length edges are tolerated but
//!   may produce degenerate (zero-area) triangles.
//!
//! **Correctness is NOT guaranteed for self-intersecting input.** The
//! function may return an incorrect or empty triangulation without signalling
//! an error; the caller is responsible for ensuring the polygon is simple.
//!
//! ## Epsilon
//! The convexity check uses a strict `> 0.0` comparison on the cross product;
//! no epsilon guard is applied there because the ear-clipping loop is
//! self-correcting (collinear ears are simply skipped and retried after
//! neighbouring ears are clipped). The point-in-triangle test uses a tolerance
//! of `1e-9` so that boundary-grazing vertices (which share an edge with the
//! candidate ear) are not incorrectly classified as interior points and do not
//! block valid ears.

use crate::geometry::polygon_area::signed_polygon_area;

// ── internal helpers ──────────────────────────────────────────────────────────

/// Signed z-component of the cross product of vectors `prev→curr` and
/// `prev→next`.
///
/// Positive ⟹ left turn (CCW); negative ⟹ right turn (CW); zero ⟹
/// collinear.
fn cross_z(prev: (f64, f64), curr: (f64, f64), next: (f64, f64)) -> f64 {
    (curr.0 - prev.0).mul_add(next.1 - prev.1, -((curr.1 - prev.1) * (next.0 - prev.0)))
}

/// Returns `true` when the turn `prev → curr → next` is a strict left turn,
/// i.e. the vertex `curr` is convex in a CCW-oriented polygon.
fn is_convex(prev: (f64, f64), curr: (f64, f64), next: (f64, f64)) -> bool {
    cross_z(prev, curr, next) > 0.0
}

/// Returns `true` when `p` lies strictly inside the triangle `(a, b, c)`.
///
/// Uses the sign-of-cross-product (barycentric) method. A tolerance of `1e-9`
/// is used so that vertices that share an edge with the triangle (and thus lie
/// numerically on or very near its boundary) are not counted as interior
/// points.
fn point_in_triangle(a: (f64, f64), b: (f64, f64), c: (f64, f64), p: (f64, f64)) -> bool {
    const EPS: f64 = 1e-9;
    let d1 = cross_z(a, b, p);
    let d2 = cross_z(b, c, p);
    let d3 = cross_z(c, a, p);
    let has_neg = (d1 < -EPS) || (d2 < -EPS) || (d3 < -EPS);
    let has_pos = (d1 > EPS) || (d2 > EPS) || (d3 > EPS);
    !(has_neg && has_pos)
}

/// Returns `true` when the vertex at position `i` in `indices` is an ear of
/// the polygon defined by `vertices`.
///
/// An ear requires:
/// 1. The turn at `indices[i]` is convex (left turn in the CCW list).
/// 2. No other active vertex lies strictly inside the ear triangle.
fn is_ear(indices: &[usize], i: usize, vertices: &[(f64, f64)]) -> bool {
    let n = indices.len();
    let prev = vertices[indices[(i + n - 1) % n]];
    let curr = vertices[indices[i]];
    let next = vertices[indices[(i + 1) % n]];

    if !is_convex(prev, curr, next) {
        return false;
    }

    // Verify that no other active vertex is strictly inside the ear triangle.
    for (j, &idx) in indices.iter().enumerate() {
        // Skip the three vertices that form the ear itself.
        if j == (i + n - 1) % n || j == i || j == (i + 1) % n {
            continue;
        }
        if point_in_triangle(prev, curr, next, vertices[idx]) {
            return false;
        }
    }
    true
}

// ── public API ────────────────────────────────────────────────────────────────

/// Triangulates a simple polygon by ear clipping.
///
/// `polygon` is an ordered list of vertices `(x, y)`. The closing edge from
/// the last vertex back to the first is implicit. Orientation (clockwise or
/// counter-clockwise) is detected automatically; clockwise input is reversed
/// before processing.
///
/// Returns a `Vec` of `n - 2` triangles, each represented as three `(f64,
/// f64)` vertices in CCW order. Returns an empty `Vec` for input with fewer
/// than three vertices.
///
/// **Precondition:** `polygon` must be a *simple* polygon. Self-intersecting
/// or degenerate input may produce incorrect results without any error.
///
/// Runs in `O(n²)` time and uses `O(n)` extra space.
pub fn triangulate(polygon: &[(f64, f64)]) -> Vec<[(f64, f64); 3]> {
    let n = polygon.len();
    if n < 3 {
        return Vec::new();
    }

    // Normalise to CCW by reversing CW polygons.
    let working: Vec<(f64, f64)> = if signed_polygon_area(polygon) < 0.0 {
        polygon.iter().copied().rev().collect()
    } else {
        polygon.to_vec()
    };

    // Active vertex indices into `working`.
    let mut indices: Vec<usize> = (0..working.len()).collect();
    let mut triangles: Vec<[(f64, f64); 3]> = Vec::with_capacity(n - 2);

    while indices.len() > 3 {
        let m = indices.len();
        let mut ear_found = false;

        for i in 0..m {
            if is_ear(&indices, i, &working) {
                let prev = working[indices[(i + m - 1) % m]];
                let curr = working[indices[i]];
                let next = working[indices[(i + 1) % m]];
                triangles.push([prev, curr, next]);
                indices.remove(i);
                ear_found = true;
                break;
            }
        }

        // Guard: if no ear is found the polygon is likely self-intersecting or
        // degenerate. Stop to avoid an infinite loop.
        if !ear_found {
            break;
        }
    }

    // Emit the final triangle.
    if indices.len() == 3 {
        triangles.push([
            working[indices[0]],
            working[indices[1]],
            working[indices[2]],
        ]);
    }

    triangles
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::triangulate;
    use crate::geometry::polygon_area::polygon_area;
    use quickcheck_macros::quickcheck;
    use std::f64::consts::PI;

    const EPS: f64 = 1e-9;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() <= EPS
    }

    fn triangle_area(t: &[(f64, f64); 3]) -> f64 {
        polygon_area(t)
    }

    fn total_area(tris: &[[(f64, f64); 3]]) -> f64 {
        tris.iter().map(triangle_area).sum()
    }

    // ── edge cases ────────────────────────────────────────────────────────────

    #[test]
    fn empty_returns_empty() {
        let v: Vec<(f64, f64)> = Vec::new();
        assert!(triangulate(&v).is_empty());
    }

    #[test]
    fn one_vertex_returns_empty() {
        assert!(triangulate(&[(0.0, 0.0)]).is_empty());
    }

    #[test]
    fn two_vertices_returns_empty() {
        assert!(triangulate(&[(0.0, 0.0), (1.0, 0.0)]).is_empty());
    }

    // ── triangle ──────────────────────────────────────────────────────────────

    #[test]
    fn triangle_returns_itself() {
        let tri = vec![(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)];
        let result = triangulate(&tri);
        assert_eq!(result.len(), 1);
        let expected_area = polygon_area(&tri);
        assert!(approx_eq(triangle_area(&result[0]), expected_area));
    }

    // ── convex quadrilateral ──────────────────────────────────────────────────

    #[test]
    fn unit_square_two_triangles() {
        // CCW unit square.
        let quad = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        let result = triangulate(&quad);
        assert_eq!(result.len(), 2);
        // Total area must equal 1.0.
        assert!(approx_eq(total_area(&result), 1.0));
    }

    // ── concave polygon ───────────────────────────────────────────────────────

    #[test]
    fn l_shape_triangulates_correctly() {
        // L-shaped hexagon (CCW):
        //  (0,0) → (2,0) → (2,1) → (1,1) → (1,2) → (0,2)
        // Area = 2×1 + 1×1 = 3.
        let l = vec![
            (0.0, 0.0),
            (2.0, 0.0),
            (2.0, 1.0),
            (1.0, 1.0),
            (1.0, 2.0),
            (0.0, 2.0),
        ];
        let result = triangulate(&l);
        assert_eq!(result.len(), 4); // n - 2 = 6 - 2 = 4
        let expected_area = polygon_area(&l);
        assert!(approx_eq(total_area(&result), expected_area));
    }

    #[test]
    fn arrow_shape_triangulates_correctly() {
        // Arrow / chevron pointing right (CCW):
        //   (0,1) → (2,0) → (4,1) → (2,0.5) — no, use a clean arrow:
        //   (0,0) → (3,1) → (2,1) → (2,3) → (1,3) → (1,1) → (0,2)
        // Simpler concave arrow (CCW, 5 vertices):
        //   (0,0) → (4,0) → (4,2) → (2,1) → (0,2)
        let arrow = vec![(0.0, 0.0), (4.0, 0.0), (4.0, 2.0), (2.0, 1.0), (0.0, 2.0)];
        let result = triangulate(&arrow);
        assert_eq!(result.len(), 3); // n - 2 = 5 - 2 = 3
        let expected_area = polygon_area(&arrow);
        assert!(approx_eq(total_area(&result), expected_area));
    }

    // ── CW input ──────────────────────────────────────────────────────────────

    #[test]
    fn cw_input_same_triangle_count_and_area_as_ccw() {
        let ccw = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        let cw: Vec<(f64, f64)> = ccw.iter().copied().rev().collect();

        let result_ccw = triangulate(&ccw);
        let result_cw = triangulate(&cw);

        assert_eq!(result_ccw.len(), result_cw.len());
        assert!(approx_eq(total_area(&result_ccw), total_area(&result_cw)));
    }

    // ── property test ─────────────────────────────────────────────────────────

    /// Property: for regular `n`-gons (convex by construction) the
    /// triangulation produces exactly `n - 2` triangles and the total
    /// triangle area equals `polygon_area(polygon)`.
    ///
    /// A regular `n`-gon inscribed in a circle of radius `r` is a provably
    /// convex simple polygon, so it exercises the algorithm on valid input
    /// without any ambiguity about orientation or self-intersection.
    #[allow(clippy::needless_pass_by_value)]
    #[quickcheck]
    fn quickcheck_regular_ngon(n_seed: u8, r_seed: u16) -> bool {
        // n in 3..=18; r in (0.001, ~65.535].
        let n = 3 + usize::from(n_seed) % 16;
        let r = (f64::from(r_seed) + 1.0) / 1000.0;

        // Build CCW regular n-gon.
        let poly: Vec<(f64, f64)> = (0..n)
            .map(|i| {
                let theta = 2.0 * PI * (i as f64) / (n as f64);
                (r * theta.cos(), r * theta.sin())
            })
            .collect();

        let expected_area = polygon_area(&poly);
        let result = triangulate(&poly);

        result.len() == n - 2 && (total_area(&result) - expected_area).abs() <= 1e-9
    }
}
