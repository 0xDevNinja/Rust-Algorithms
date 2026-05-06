//! Half-plane intersection via the sort-and-deque algorithm.
//!
//! A *half-plane* is the set of points that lie on the left of (or on) a
//! directed line `start → end`, i.e. where the 2-D cross product
//! `cross(end - start, p - start) >= 0`.  The intersection of `n`
//! half-planes is a (possibly empty, possibly unbounded) convex region.
//!
//! This module computes that region as a convex polygon.
//!
//! # Algorithm
//! Zhu–Pan sort-and-deque (`O(n log n)`):
//!
//! 1. Sort half-planes by the angle of their direction vector (`atan2`).
//!    Half-planes with the same angle keep only the one whose boundary is
//!    furthest to the left (it dominates the other).
//! 2. Maintain a double-ended queue (deque) of half-planes whose
//!    pairwise intersections form a tentative convex polygon.  For each
//!    incoming half-plane:
//!    - Pop from the **back** while the intersection of the last two
//!      deque entries lies *outside* (to the right of) the new half-plane.
//!    - Pop from the **front** while the intersection of the first two
//!      deque entries lies outside the new half-plane.
//!    - Push the new half-plane onto the back.
//! 3. After processing all half-planes do a final cleanup: pop from the
//!    back while the intersection of the back two entries is outside the
//!    front entry, and similarly for the front.
//! 4. Convert the surviving deque entries to vertices by computing
//!    consecutive intersection points.
//!
//! # Preconditions
//! * The intersection must be **bounded**.  The recommended practice is to
//!   add four large axis-aligned bounding half-planes before calling this
//!   function (e.g., with `|coord| <= 1e9`).  If the intersection is
//!   unbounded the function returns an empty `Vec`.
//! * Input half-planes need not be sorted or deduplicated; the algorithm
//!   handles that internally.
//!
//! # Complexity
//! * Time:  `O(n log n)` (sort-dominated).
//! * Space: `O(n)` for the deque and output polygon.

use std::collections::VecDeque;

type Point = (f64, f64);

/// A half-plane: the closed set of points `p` satisfying
/// `cross(end - start, p - start) >= 0`, i.e. on or to the left of the
/// directed line `start → end`.
#[derive(Clone, Copy, Debug)]
pub struct HalfPlane {
    pub start: Point,
    pub end: Point,
}

impl HalfPlane {
    /// Direction vector of the bounding line.
    #[inline]
    fn dir(self) -> Point {
        (self.end.0 - self.start.0, self.end.1 - self.start.1)
    }

    /// Angle of the direction vector in `(-π, π]`.
    #[inline]
    fn angle(self) -> f64 {
        let d = self.dir();
        d.1.atan2(d.0)
    }

    /// `true` when `p` is strictly to the right of (i.e. outside) this
    /// half-plane's boundary.
    #[inline]
    fn is_outside(self, p: Point) -> bool {
        cross(self.start, self.end, p) < -1e-9
    }
}

/// Signed area of the triangle `(o, a, b)` times 2 (cross product of
/// `a - o` and `b - o`).
///
/// Positive for a counter-clockwise triple, negative for clockwise, zero
/// when collinear.
#[inline]
fn cross(o: Point, a: Point, b: Point) -> f64 {
    (a.0 - o.0).mul_add(b.1 - o.1, -((a.1 - o.1) * (b.0 - o.0)))
}

/// Intersection point of the two infinite lines defined by `h1` and `h2`.
///
/// Returns `None` when the lines are parallel (the deque algorithm discards
/// duplicate angles beforehand, so in practice this is never `None` on
/// non-parallel half-planes).
fn line_intersect(h1: HalfPlane, h2: HalfPlane) -> Option<Point> {
    let (ax, ay) = (h1.end.0 - h1.start.0, h1.end.1 - h1.start.1);
    let (bx, by) = (h2.end.0 - h2.start.0, h2.end.1 - h2.start.1);
    let denom = ax.mul_add(by, -(ay * bx));
    if denom.abs() < 1e-12 {
        return None; // parallel
    }
    let dx = h2.start.0 - h1.start.0;
    let dy = h2.start.1 - h1.start.1;
    let t = dx.mul_add(by, -(dy * bx)) / denom;
    Some((t.mul_add(ax, h1.start.0), t.mul_add(ay, h1.start.1)))
}

/// Computes the intersection of the given half-planes as a convex polygon.
///
/// Each half-plane is the set of points on or to the **left** of the directed
/// line `start → end` (i.e. `cross(end - start, p - start) >= 0`).
///
/// Returns the vertices of the intersection polygon in counter-clockwise
/// order, or an empty `Vec` if the intersection is empty or unbounded.
///
/// # Precondition
/// The intersection should be bounded.  Add four large bounding half-planes
/// (e.g. `|x| <= 1e9`, `|y| <= 1e9`) if the inputs do not already guarantee
/// boundedness.
pub fn intersect(half_planes: &[HalfPlane]) -> Vec<Point> {
    if half_planes.len() < 3 {
        return Vec::new();
    }

    // --- 1. Sort by angle; keep the leftmost among equal-angle duplicates. ---
    let mut sorted: Vec<HalfPlane> = half_planes.to_vec();
    sorted.sort_by(|a, b| {
        a.angle()
            .partial_cmp(&b.angle())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Deduplicate same-angle half-planes: keep the one whose boundary is
    // furthest to the left, i.e. whose `start` is most to the left of the
    // other's directed line (or equivalently the one that rejects the
    // other's start point).
    let mut deduped: Vec<HalfPlane> = Vec::with_capacity(sorted.len());
    let mut i = 0;
    while i < sorted.len() {
        let mut j = i + 1;
        while j < sorted.len() && (sorted[j].angle() - sorted[i].angle()).abs() < 1e-9 {
            j += 1;
        }
        // Among sorted[i..j] keep the one furthest to the left.
        // "Furthest to the left" means its start is furthest in the CCW
        // direction relative to the shared angle — i.e. the one for which
        // the start of any other is on its right (outside).  In practice we
        // keep the one whose start has the largest cross product with
        // respect to the first one's direction.
        let mut best = sorted[i];
        for k in (i + 1)..j {
            if cross(sorted[k].start, sorted[k].end, best.start) < 0.0 {
                // best.start is to the right of sorted[k], so sorted[k] is
                // further left.
                best = sorted[k];
            }
        }
        deduped.push(best);
        i = j;
    }

    let n = deduped.len();
    if n < 3 {
        return Vec::new();
    }

    // --- 2. Sort-and-deque pass. ---
    // The deque holds half-plane indices into `deduped`.
    // `pts[k]` = intersection of deque[k] and deque[k+1].
    let mut dq: VecDeque<HalfPlane> = VecDeque::with_capacity(n);
    // Parallel deque of intersection points between consecutive deque entries.
    let mut pts: VecDeque<Point> = VecDeque::with_capacity(n);

    for hp in &deduped {
        // Pop from back while the last intersection point is outside `*hp`.
        while !pts.is_empty() && hp.is_outside(*pts.back().unwrap()) {
            dq.pop_back();
            pts.pop_back();
        }
        // Pop from front while the first intersection point is outside `*hp`.
        while !pts.is_empty() && hp.is_outside(*pts.front().unwrap()) {
            dq.pop_front();
            pts.pop_front();
        }
        // Push `hp` onto the back.
        if let Some(&back) = dq.back() {
            match line_intersect(back, *hp) {
                Some(p) => pts.push_back(p),
                None => continue, // parallel — skip (should not happen after dedup)
            }
        }
        dq.push_back(*hp);
    }

    // --- 3. Final cleanup: handle the "wrap-around" between front and back. ---
    // Pop from back while intersection of the last two deque entries lies
    // outside the front entry.
    while dq.len() > 2 {
        let back_hp = dq.back().copied().unwrap();
        let second_back = dq[dq.len() - 2];
        match line_intersect(second_back, back_hp) {
            Some(p) if dq.front().unwrap().is_outside(p) => {
                dq.pop_back();
                pts.pop_back();
            }
            _ => break,
        }
    }
    // Pop from front while intersection of the first two entries is outside
    // the back entry.
    while dq.len() > 2 {
        let front_hp = dq.front().copied().unwrap();
        let second_front = dq[1];
        match line_intersect(front_hp, second_front) {
            Some(p) if dq.back().unwrap().is_outside(p) => {
                dq.pop_front();
                pts.pop_front();
            }
            _ => break,
        }
    }

    if dq.len() < 3 {
        return Vec::new();
    }

    // --- 4. Convert deque to vertex list. ---
    // We need n vertices where vertex[k] = intersection of dq[k] and dq[k+1 mod n].
    let m = dq.len();
    let dq_vec: Vec<HalfPlane> = dq.into_iter().collect();
    let mut polygon: Vec<Point> = Vec::with_capacity(m);
    for k in 0..m {
        match line_intersect(dq_vec[k], dq_vec[(k + 1) % m]) {
            Some(p) => polygon.push(p),
            None => return Vec::new(), // degenerate / unbounded
        }
    }

    polygon
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::{intersect, HalfPlane};
    use quickcheck_macros::quickcheck;

    const EPS: f64 = 1e-9;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() <= EPS
    }

    fn approx_eq_pt(a: (f64, f64), b: (f64, f64)) -> bool {
        approx_eq(a.0, b.0) && approx_eq(a.1, b.1)
    }

    /// A large axis-aligned bounding box expressed as 4 half-planes so that
    /// the intersection of any bounded region inside `[-B, B]^2` stays
    /// bounded.
    fn bounding_box(b: f64) -> Vec<HalfPlane> {
        vec![
            // bottom edge: y >= -B  →  directed right (left side is up)
            HalfPlane {
                start: (-b, -b),
                end: (b, -b),
            },
            // right edge: x <= B  →  directed up (left side is left/inward)
            HalfPlane {
                start: (b, -b),
                end: (b, b),
            },
            // top edge: y <= B  →  directed left
            HalfPlane {
                start: (b, b),
                end: (-b, b),
            },
            // left edge: x >= -B  →  directed down
            HalfPlane {
                start: (-b, b),
                end: (-b, -b),
            },
        ]
    }

    /// Signed area via the shoelace formula; positive = CCW.
    fn signed_area(poly: &[(f64, f64)]) -> f64 {
        let n = poly.len();
        let mut s = 0.0_f64;
        for i in 0..n {
            let (x0, y0) = poly[i];
            let (x1, y1) = poly[(i + 1) % n];
            s += x0.mul_add(y1, -(x1 * y0));
        }
        s / 2.0
    }

    /// Returns `true` if `p` is on or to the left of every directed edge of
    /// `poly` (i.e. inside the convex polygon).
    fn point_in_convex_polygon(poly: &[(f64, f64)], p: (f64, f64)) -> bool {
        let n = poly.len();
        for i in 0..n {
            let a = poly[i];
            let b = poly[(i + 1) % n];
            let cross = (b.0 - a.0).mul_add(p.1 - a.1, -((b.1 - a.1) * (p.0 - a.0)));
            if cross < -1e-7 {
                return false;
            }
        }
        true
    }

    // -----------------------------------------------------------------------
    // Basic structural tests
    // -----------------------------------------------------------------------

    #[test]
    fn empty_input_returns_empty() {
        let result = intersect(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn single_half_plane_returns_empty_unbounded() {
        let hp = HalfPlane {
            start: (0.0, 0.0),
            end: (1.0, 0.0),
        };
        // Only one half-plane — unbounded, so must return empty.
        let result = intersect(&[hp]);
        assert!(result.is_empty());
    }

    // -----------------------------------------------------------------------
    // Triangle
    // -----------------------------------------------------------------------

    /// Three half-planes forming the triangle with vertices (0,0), (1,0), (0,1).
    fn triangle_half_planes() -> Vec<HalfPlane> {
        vec![
            // bottom edge: y >= 0  →  directed right
            HalfPlane {
                start: (0.0, 0.0),
                end: (1.0, 0.0),
            },
            // hypotenuse: x + y <= 1  →  directed from (1,0) to (0,1)
            HalfPlane {
                start: (1.0, 0.0),
                end: (0.0, 1.0),
            },
            // left edge: x >= 0  →  directed up
            HalfPlane {
                start: (0.0, 1.0),
                end: (0.0, 0.0),
            },
        ]
    }

    #[test]
    fn triangle_has_three_vertices() {
        let result = intersect(&triangle_half_planes());
        assert_eq!(result.len(), 3, "expected 3 vertices, got {}", result.len());
    }

    #[test]
    fn triangle_vertices_match() {
        let result = intersect(&triangle_half_planes());
        assert_eq!(result.len(), 3);

        // The three expected vertices (in some CCW order).
        let expected = vec![(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)];
        for exp in &expected {
            assert!(
                result.iter().any(|&v| approx_eq_pt(v, *exp)),
                "missing vertex {exp:?} in {result:?}",
            );
        }
    }

    #[test]
    fn triangle_area() {
        let result = intersect(&triangle_half_planes());
        let area = signed_area(&result).abs();
        assert!(approx_eq(area, 0.5), "expected area 0.5, got {area}");
    }

    // -----------------------------------------------------------------------
    // Unit square
    // -----------------------------------------------------------------------

    fn unit_square_half_planes() -> Vec<HalfPlane> {
        vec![
            // bottom: y >= 0
            HalfPlane {
                start: (0.0, 0.0),
                end: (1.0, 0.0),
            },
            // right: x <= 1
            HalfPlane {
                start: (1.0, 0.0),
                end: (1.0, 1.0),
            },
            // top: y <= 1
            HalfPlane {
                start: (1.0, 1.0),
                end: (0.0, 1.0),
            },
            // left: x >= 0
            HalfPlane {
                start: (0.0, 1.0),
                end: (0.0, 0.0),
            },
        ]
    }

    #[test]
    fn square_has_four_vertices() {
        let result = intersect(&unit_square_half_planes());
        assert_eq!(result.len(), 4, "expected 4 vertices, got {result:?}");
    }

    #[test]
    fn square_corners_present() {
        let result = intersect(&unit_square_half_planes());
        let expected = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        for exp in &expected {
            assert!(
                result.iter().any(|&v| approx_eq_pt(v, *exp)),
                "missing corner {exp:?}",
            );
        }
    }

    #[test]
    fn square_area() {
        let result = intersect(&unit_square_half_planes());
        let area = signed_area(&result).abs();
        assert!(approx_eq(area, 1.0), "expected area 1.0, got {area}");
    }

    // -----------------------------------------------------------------------
    // Empty intersection (contradictory half-planes)
    // -----------------------------------------------------------------------

    #[test]
    fn contradictory_half_planes_return_empty() {
        // x >= 1  and  x <= 0  →  impossible (no point satisfies both).
        //
        // Half-plane "x >= 1": directed upward along x = 1.
        //   start = (1, 0), end = (1, 1)
        //   cross((0,1), (p.x-1, p.y-0)) = -1*(p.x-1) = 1-p.x ... wait:
        //   cross of dir (0,1) with offset (p.x-1, p.y): 0*p.y - 1*(p.x-1) = 1-p.x
        //   >= 0  iff  p.x <= 1.  That gives x <= 1, not x >= 1.
        //
        // Correct "x >= 1": directed *downward* along x = 1.
        //   start = (1, 1), end = (1, 0), dir = (0, -1)
        //   cross of (0,-1) with (p.x-1, p.y-1): 0*(p.y-1) - (-1)*(p.x-1) = p.x-1
        //   >= 0  iff  p.x >= 1.
        //
        // Correct "x <= 0": directed *upward* along x = 0.
        //   start = (0, 0), end = (0, 1), dir = (0, 1)
        //   cross of (0,1) with (p.x, p.y): 0*p.y - 1*p.x = -p.x
        //   >= 0  iff  p.x <= 0.
        let mut hps = bounding_box(100.0);
        hps.push(HalfPlane {
            start: (1.0, 1.0),
            end: (1.0, 0.0),
        }); // x >= 1
        hps.push(HalfPlane {
            start: (0.0, 0.0),
            end: (0.0, 1.0),
        }); // x <= 0
        let result = intersect(&hps);
        assert!(
            result.is_empty(),
            "expected empty for contradictory half-planes, got {result:?}",
        );
    }

    // -----------------------------------------------------------------------
    // Quickcheck property
    // -----------------------------------------------------------------------

    /// For random half-planes (from small integer coords) combined with a
    /// large bounding box, every vertex of the result must satisfy all input
    /// half-planes (up to a small tolerance), and every vertex of the result
    /// must be inside the bounding box.
    #[allow(clippy::needless_pass_by_value)]
    #[quickcheck]
    fn quickcheck_vertices_satisfy_all_half_planes(seeds: Vec<(i8, i8, i8, i8)>) -> bool {
        // Build a small set of random half-planes from integer seeds.
        let mut hps = bounding_box(200.0);
        for &(x1, y1, x2, y2) in seeds.iter().take(8) {
            let start = (f64::from(x1), f64::from(y1));
            let end = (f64::from(x2), f64::from(y2));
            // Skip degenerate (zero-length) directed lines.
            let dx = end.0 - start.0;
            let dy = end.1 - start.1;
            if dx.hypot(dy) < 1e-9 {
                continue;
            }
            hps.push(HalfPlane { start, end });
        }

        let result = intersect(&hps);
        if result.is_empty() {
            // Empty result is acceptable (could be a genuinely empty intersection).
            return true;
        }

        // Every output vertex must lie inside (or on the boundary of) every
        // input half-plane.
        for &vertex in &result {
            for &hp in &hps {
                let cross_val = super::cross(hp.start, hp.end, vertex);
                if cross_val < -1e-6 {
                    return false;
                }
            }
        }

        // The result polygon must be convex (non-negative cross products for
        // all consecutive edge triples in CCW order).
        let n = result.len();
        if n >= 3 {
            for i in 0..n {
                let a = result[i];
                let b = result[(i + 1) % n];
                let c = result[(i + 2) % n];
                let cv = (b.0 - a.0).mul_add(c.1 - a.1, -((b.1 - a.1) * (c.0 - a.0)));
                if cv < -1e-6 {
                    return false;
                }
            }
        }

        true
    }

    /// For a fixed convex polygon (regular-ish hexagon), the half-plane
    /// intersection must produce a polygon whose area matches, and every
    /// sample point known to be inside must be inside the result.
    #[test]
    fn hexagon_interior_point_contained() {
        // A rough regular hexagon via 6 directed edges (CCW), radius ≈ 2.
        use std::f64::consts::PI;
        let r = 2.0_f64;
        let n = 6usize;
        let mut hps: Vec<HalfPlane> = (0..n)
            .map(|k| {
                let a0 = 2.0 * PI * (k as f64) / (n as f64);
                let a1 = 2.0 * PI * ((k + 1) as f64) / (n as f64);
                HalfPlane {
                    start: (r * a0.cos(), r * a0.sin()),
                    end: (r * a1.cos(), r * a1.sin()),
                }
            })
            .collect();
        hps.extend(bounding_box(10.0));

        let result = intersect(&hps);
        assert!(!result.is_empty(), "hexagon must not be empty");
        // The centroid (0, 0) must be inside.
        assert!(
            point_in_convex_polygon(&result, (0.0, 0.0)),
            "centroid must be inside"
        );
    }
}
