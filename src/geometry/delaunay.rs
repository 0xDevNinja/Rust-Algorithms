//! Delaunay triangulation via the Bowyer-Watson algorithm.
//!
//! Given a set of `n` points in the plane, this module computes a
//! **Delaunay triangulation**: a triangulation in which the circumcircle of
//! every triangle contains no other input point in its interior. The
//! Delaunay triangulation maximises the minimum angle over all triangulations,
//! which makes it useful for finite-element meshes, terrain modelling, and
//! nearest-neighbour graphs.
//!
//! # Algorithm
//!
//! The classic **Bowyer-Watson** incremental insertion scheme:
//!
//! 1. Construct a super-triangle that strictly contains all input points.
//! 2. Insert points one by one. For each new point `p`: find all triangles
//!    whose circumcircle contains it (the "bad" triangles), compute the
//!    polygonal cavity boundary (edges belonging to exactly one bad triangle),
//!    then connect the new point to every boundary edge.
//! 3. After all points are inserted, remove every triangle that shares a
//!    vertex with the super-triangle.
//!
//! # Complexity
//!
//! * Time: `O(n²)` in the worst case (and on average for uniformly random
//!   input with this baseline implementation). An `O(n log n)` variant is
//!   possible with spatial indexing but is not required here.
//! * Space: `O(n)` for the triangle list.
//!
//! # Preconditions
//!
//! * All coordinates must be finite (`NaN` or infinite inputs produce
//!   unspecified results).
//! * Collinear or duplicate points: no triangle can be formed from
//!   degenerate triples, so those inputs produce an empty or partial
//!   triangulation. The function does not panic on such inputs.
//!
//! # Output
//!
//! Each triangle is represented as three indices into the original `points`
//! slice and is oriented counter-clockwise (CCW).

/// A triangle stored as indices into the caller's point array *plus* the
/// three super-triangle sentinel indices (values `>= n_real`).
#[derive(Clone, Copy, Debug)]
struct Triangle {
    /// Vertex indices (may refer to super-triangle sentinels).
    v: [usize; 3],
    /// Circumcircle centre.
    cx: f64,
    /// Circumcircle centre.
    cy: f64,
    /// Circumcircle radius squared.
    r2: f64,
}

impl Triangle {
    /// Build a triangle from three point indices and precompute its
    /// circumcircle. Returns `None` when the three points are collinear
    /// (degenerate circumcircle).
    fn new(a: usize, b: usize, c: usize, pts: &[(f64, f64)]) -> Option<Self> {
        let (ax, ay) = pts[a];
        let (bx, by) = pts[b];
        let (cx, cy) = pts[c];

        // Translate so that `a` is at the origin; this improves numerical
        // stability when coordinates are large and close together.
        let bx = bx - ax;
        let by = by - ay;
        let cx2 = cx - ax;
        let cy2 = cy - ay;

        // Denominator of the circumcircle formula: 2 · |b × c|.
        let d = 2.0 * bx.mul_add(cy2, -(by * cx2));
        if d.abs() < 1e-20 {
            return None; // collinear
        }

        let b2 = bx.mul_add(bx, by * by);
        let c2 = cx2.mul_add(cx2, cy2 * cy2);

        let ux = cy2.mul_add(b2, -(by * c2)) / d;
        let uy = bx.mul_add(c2, -(cx2 * b2)) / d;

        let ccx = ax + ux;
        let ccy = ay + uy;
        let r2 = ux.mul_add(ux, uy * uy); // squared radius

        Some(Self {
            v: [a, b, c],
            cx: ccx,
            cy: ccy,
            r2,
        })
    }

    /// Returns `true` if `p` lies strictly inside this circumcircle.
    ///
    /// A small epsilon is subtracted from the radius so that a point sitting
    /// almost exactly on the circle boundary is not counted as "inside" by
    /// floating-point noise, which would cause Bowyer-Watson to flip
    /// otherwise-valid triangles on convex-position inputs.
    fn circumcircle_contains(&self, p: (f64, f64)) -> bool {
        let dx = p.0 - self.cx;
        let dy = p.1 - self.cy;
        let dist2 = dx.mul_add(dx, dy * dy);
        // Use a relative tolerance: if the point is within eps of the circle
        // boundary treat it as outside to preserve the existing triangulation.
        let eps = 1e-10 * self.r2.sqrt().max(1.0);
        dist2 < self.r2 - eps
    }
}

/// Returns the Delaunay triangulation of `points` as a list of triangles,
/// each given as 3 indices into `points` (CCW orientation). Empty input
/// or fewer than 3 points returns an empty vec.
///
/// The algorithm is Bowyer-Watson incremental insertion. It runs in
/// `O(n²)` time in the worst case. See the module-level documentation for
/// full details and preconditions.
///
/// # Examples
///
/// ```
/// use rust_algorithms::geometry::delaunay::delaunay;
///
/// // Four corners of a unit square → 2 triangles.
/// let pts = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
/// let tris = delaunay(&pts);
/// assert_eq!(tris.len(), 2);
/// ```
#[must_use]
pub fn delaunay(points: &[(f64, f64)]) -> Vec<[usize; 3]> {
    let n = points.len();
    if n < 3 {
        return Vec::new();
    }

    // -----------------------------------------------------------------------
    // Build a working point array: real points first, then the three
    // super-triangle vertices appended at indices n, n+1, n+2.
    // -----------------------------------------------------------------------
    let (min_x, max_x, min_y, max_y) = points.iter().fold(
        (
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
        ),
        |(lx, rx, ly, ry), &(x, y)| (lx.min(x), rx.max(x), ly.min(y), ry.max(y)),
    );

    let dx = max_x - min_x;
    let dy = max_y - min_y;
    // Scale the super-triangle so it comfortably surrounds all points even
    // when the cloud is very thin or nearly one-dimensional.
    let delta = dx.max(dy).max(1.0);
    let mid_x = (min_x + max_x) * 0.5;
    let mid_y = (min_y + max_y) * 0.5;

    // Super-triangle vertices (indices n, n+1, n+2 in the extended array).
    let st0 = (20.0_f64.mul_add(-delta, mid_x), mid_y - delta);
    let st1 = (mid_x, 20.0_f64.mul_add(delta, mid_y));
    let st2 = (20.0_f64.mul_add(delta, mid_x), mid_y - delta);

    let mut pts: Vec<(f64, f64)> = Vec::with_capacity(n + 3);
    pts.extend_from_slice(points);
    pts.push(st0);
    pts.push(st1);
    pts.push(st2);

    // Initial triangulation: just the super-triangle.
    let super_tri =
        Triangle::new(n, n + 1, n + 2, &pts).expect("super-triangle vertices are never collinear");
    let mut triangles: Vec<Triangle> = vec![super_tri];

    // -----------------------------------------------------------------------
    // Insert each real point.
    // -----------------------------------------------------------------------
    for i in 0..n {
        let p = pts[i];

        // Find all "bad" triangles whose circumcircle contains `p`.
        // We partition in-place: bad triangles go to the tail, good ones stay.
        let mut bad_start = triangles.len();
        for j in (0..bad_start).rev() {
            if triangles[j].circumcircle_contains(p) {
                bad_start -= 1;
                triangles.swap(j, bad_start);
            }
        }

        // Collect the boundary polygon of the cavity: edges that belong to
        // exactly one bad triangle.  An edge shared by two bad triangles is
        // internal and must be discarded.
        let bad = &triangles[bad_start..];

        // Build a list of all edges from bad triangles, then keep only those
        // that are not shared by two bad triangles (i.e. boundary edges).
        let mut edges: Vec<[usize; 2]> = Vec::with_capacity(bad.len() * 3);
        for t in bad {
            let [a, b, c] = t.v;
            edges.push([a, b]);
            edges.push([b, c]);
            edges.push([c, a]);
        }

        // An edge and its reverse [v1, v0] form an interior edge pair.
        // Mark interior edges by setting them to a sentinel [usize::MAX, …].
        let m = edges.len();
        for j in 0..m {
            for k in (j + 1)..m {
                if edges[j][0] == edges[k][1] && edges[j][1] == edges[k][0] {
                    edges[j] = [usize::MAX, usize::MAX];
                    edges[k] = [usize::MAX, usize::MAX];
                }
            }
        }

        // Remove bad triangles from the list.
        triangles.truncate(bad_start);

        // Re-triangulate the cavity by connecting `p` to each boundary edge.
        for &[ea, eb] in &edges {
            if ea == usize::MAX {
                continue; // interior edge, skip
            }
            // Triangle `ea -> eb -> i`. If circumcircle construction fails
            // (degenerate), skip silently — this can happen with nearly
            // collinear points.
            if let Some(t) = Triangle::new(ea, eb, i, &pts) {
                triangles.push(t);
            }
        }
    }

    // -----------------------------------------------------------------------
    // Filter out triangles touching the super-triangle and convert to
    // index triples in the original `points` slice.
    // -----------------------------------------------------------------------
    triangles
        .into_iter()
        .filter(|t| t.v.iter().all(|&v| v < n))
        .map(|t| {
            let [a, b, c] = t.v;
            // Ensure the output triangle is CCW with respect to the *original*
            // coordinates. The cross product of (b-a) × (c-a) should be > 0.
            let (ax, ay) = points[a];
            let (bx, by) = points[b];
            let (cx, cy) = points[c];
            let cross = (bx - ax).mul_add(cy - ay, -((by - ay) * (cx - ax)));
            if cross >= 0.0 {
                [a, b, c]
            } else {
                [a, c, b]
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::delaunay;
    use quickcheck_macros::quickcheck;

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    /// Signed area (positive = CCW).
    fn signed_area(pts: &[(f64, f64)], tri: [usize; 3]) -> f64 {
        let (ax, ay) = pts[tri[0]];
        let (bx, by) = pts[tri[1]];
        let (cx, cy) = pts[tri[2]];
        0.5 * ((bx - ax).mul_add(cy - ay, -((by - ay) * (cx - ax))))
    }

    /// Returns true if point `p` lies strictly inside the circumcircle of
    /// the triangle with vertices `a`, `b`, `c`.
    fn strictly_inside_circumcircle(
        p: (f64, f64),
        a: (f64, f64),
        b: (f64, f64),
        c: (f64, f64),
    ) -> bool {
        // Translate so `a` is at the origin.
        let bx = b.0 - a.0;
        let by = b.1 - a.1;
        let cx = c.0 - a.0;
        let cy = c.1 - a.1;
        let d = 2.0 * bx.mul_add(cy, -(by * cx));
        if d.abs() < 1e-20 {
            return false;
        }
        let b2 = bx.mul_add(bx, by * by);
        let c2 = cx.mul_add(cx, cy * cy);
        let ux = cy.mul_add(b2, -(by * c2)) / d;
        let uy = bx.mul_add(c2, -(cx * b2)) / d;
        let r2 = ux.mul_add(ux, uy * uy);
        let ccx = a.0 + ux;
        let ccy = a.1 + uy;
        let dx = p.0 - ccx;
        let dy = p.1 - ccy;
        let dist2 = dx.mul_add(dx, dy * dy);
        // Strict interior: subtract a generous relative eps so points nearly
        // on the boundary are not falsely flagged.
        let eps = 1e-9 * r2.sqrt().max(1.0);
        dist2 < r2 - eps
    }

    /// Verify the Delaunay property: no input point lies strictly inside
    /// the circumcircle of any triangle.
    fn check_delaunay_property(pts: &[(f64, f64)], tris: &[[usize; 3]]) -> bool {
        for &[a, b, c] in tris {
            for (i, &p) in pts.iter().enumerate() {
                if i == a || i == b || i == c {
                    continue;
                }
                if strictly_inside_circumcircle(p, pts[a], pts[b], pts[c]) {
                    return false;
                }
            }
        }
        true
    }

    // -----------------------------------------------------------------------
    // Unit tests
    // -----------------------------------------------------------------------

    #[test]
    fn empty_input_returns_empty() {
        let v: Vec<(f64, f64)> = Vec::new();
        assert!(delaunay(&v).is_empty());
    }

    #[test]
    fn one_point_returns_empty() {
        assert!(delaunay(&[(0.0, 0.0)]).is_empty());
    }

    #[test]
    fn two_points_returns_empty() {
        assert!(delaunay(&[(0.0, 0.0), (1.0, 0.0)]).is_empty());
    }

    #[test]
    fn three_points_single_triangle() {
        let pts = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)];
        let tris = delaunay(&pts);
        assert_eq!(tris.len(), 1, "expected exactly one triangle");
        let tri = tris[0];
        // All original indices present.
        let mut idx = tri;
        idx.sort_unstable();
        assert_eq!(idx, [0, 1, 2]);
        // CCW orientation.
        assert!(signed_area(&pts, tri) > 0.0);
    }

    #[test]
    fn unit_square_two_triangles() {
        // The unit square has exactly two Delaunay triangulations, both valid.
        // We only check count and the Delaunay property.
        let pts = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        let tris = delaunay(&pts);
        assert_eq!(tris.len(), 2, "expected 2 triangles for a convex quad");
        // Every triangle CCW.
        for &t in &tris {
            assert!(signed_area(&pts, t) > 0.0, "triangle must be CCW");
        }
        // Delaunay property.
        assert!(check_delaunay_property(&pts, &tris));
    }

    #[test]
    fn collinear_points_returns_empty_or_sane() {
        // Three or more collinear points: no non-degenerate triangle can be
        // formed.  The function should return an empty vec without panicking.
        let pts = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)];
        let tris = delaunay(&pts);
        // All collinear → no valid triangle.  Each triangle that does appear
        // (shouldn't) must still satisfy CCW + Delaunay.
        for &t in &tris {
            assert!(
                signed_area(&pts, t).abs() > 1e-12,
                "degenerate triangle should not appear"
            );
        }
    }

    #[test]
    fn regular_pentagon_delaunay_property() {
        use std::f64::consts::TAU;
        let pts: Vec<(f64, f64)> = (0..5)
            .map(|i| {
                let theta = TAU * (i as f64) / 5.0;
                (theta.cos(), theta.sin())
            })
            .collect();
        let tris = delaunay(&pts);
        assert!(!tris.is_empty());
        assert!(check_delaunay_property(&pts, &tris));
        for &t in &tris {
            assert!(signed_area(&pts, t) > 0.0);
        }
    }

    #[test]
    fn small_random_set_delaunay_property() {
        // Deterministic LCG so the test is reproducible without an RNG dep.
        let pts: Vec<(f64, f64)> = (0u64..12)
            .map(|i| {
                let x = i.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                let y = x.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                let xf = ((x >> 11) as f64 / ((1u64 << 53) as f64)).mul_add(10.0, -5.0);
                let yf = ((y >> 11) as f64 / ((1u64 << 53) as f64)).mul_add(10.0, -5.0);
                (xf, yf)
            })
            .collect();
        let tris = delaunay(&pts);
        assert!(!tris.is_empty());
        assert!(check_delaunay_property(&pts, &tris));
        for &t in &tris {
            assert!(signed_area(&pts, t) > 0.0);
        }
    }

    // -----------------------------------------------------------------------
    // QuickCheck property
    // -----------------------------------------------------------------------

    /// De-duplicate and jitter a set of `(i8, i8)` points so that no two
    /// points are closer than `1e-4` and the input is in general position
    /// (avoids true collinearity and cocircularity that would make the
    /// Delaunay property ill-defined at the epsilon we test with).
    fn jitter(raw: &[(i8, i8)]) -> Vec<(f64, f64)> {
        // Deterministic jitter: hash the index with a cheap LCG.
        let mut out: Vec<(f64, f64)> = Vec::new();
        'outer: for (k, &(xi, yi)) in raw.iter().enumerate() {
            let seed = k as u64;
            let h = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let jx = ((h >> 11) as f64 / ((1u64 << 53) as f64)).mul_add(1e-4, -5e-5);
            let h2 = h
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let jy = ((h2 >> 11) as f64 / ((1u64 << 53) as f64)).mul_add(1e-4, -5e-5);
            let x = f64::from(xi) + jx;
            let y = f64::from(yi) + jy;
            // Skip if too close to an existing point (crude dedup).
            for &(ox, oy) in &out {
                let dx = x - ox;
                let dy = y - oy;
                if dx.mul_add(dx, dy * dy) < 1e-8 {
                    continue 'outer;
                }
            }
            out.push((x, y));
        }
        out
    }

    #[allow(clippy::needless_pass_by_value)]
    #[quickcheck]
    fn quickcheck_delaunay_property(raw: Vec<(i8, i8)>) -> bool {
        if raw.len() > 20 {
            // Keep brute-force check fast.
            return true;
        }
        let pts = jitter(&raw);
        if pts.len() < 3 {
            return true;
        }
        let tris = delaunay(&pts);
        // Every returned triangle must be CCW and satisfy the empty-circle
        // property (no other point strictly inside its circumcircle).
        for &t in &tris {
            if signed_area(&pts, t) <= 0.0 {
                return false;
            }
        }
        check_delaunay_property(&pts, &tris)
    }
}
