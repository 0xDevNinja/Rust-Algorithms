//! Polygon diameter via the rotating calipers method.
//!
//! Given a *simple convex* polygon listed in counter-clockwise (CCW) order,
//! the **diameter** is the maximum Euclidean distance between any two
//! vertices. A classical brute-force search compares every pair of vertices
//! in `O(n²)` time. The rotating calipers technique exploits the convex
//! structure to do the same job in `O(n)`.
//!
//! Sketch of the algorithm: walk a pair of indices `(i, j)` around the
//! hull, advancing whichever one's edge makes a smaller angle with the
//! current "support line" — equivalently, advancing `j` while the cross
//! product `(p_{i+1} − p_i) × (p_{j+1} − p_j)` is positive. At every step
//! `(p_i, p_j)` is an *antipodal pair*, i.e. a pair of vertices that admit
//! parallel supporting lines on opposite sides of the polygon. The polygon
//! diameter is realised by some antipodal pair, so taking the max over the
//! `O(n)` antipodal pairs visited yields the answer.
//!
//! Complexity: `O(n)` time, `O(1)` extra space.
//!
//! Precondition: `hull` must describe a *simple convex* polygon with
//! vertices in CCW order. Convex-hull construction is the caller's
//! responsibility. Behaviour on non-convex, self-intersecting, or
//! clockwise-oriented input is undefined — the routine still returns a
//! number, but it is not guaranteed to be the polygon diameter. Duplicate
//! consecutive vertices and collinear edges are tolerated.
//!
//! Vertices are stored as `(f64, f64)` pairs.

/// Returns the diameter (maximum vertex-to-vertex Euclidean distance) of
/// the convex polygon described by `hull`.
///
/// Special cases:
/// - empty input → `0.0`,
/// - single vertex → `0.0`,
/// - two vertices → the distance between them,
/// - three or more vertices → rotating calipers in `O(n)`.
///
/// The input must be a simple convex polygon in counter-clockwise order;
/// see the module docs for the precondition.
pub fn polygon_diameter(hull: &[(f64, f64)]) -> f64 {
    let n = hull.len();
    match n {
        0 | 1 => return 0.0,
        2 => return dist(hull[0], hull[1]),
        _ => {}
    }

    let mut best_sq = 0.0_f64;
    let mut j = 1_usize;
    for i in 0..n {
        let next_i = (i + 1) % n;
        // Advance `j` while the triangle (p_i, p_{i+1}, p_{j+1}) has
        // greater area than (p_i, p_{i+1}, p_j) — i.e. while p_{j+1} is
        // farther from edge (p_i, p_{i+1}) than p_j.
        loop {
            let cur = triangle_cross(hull[i], hull[next_i], hull[j]);
            let nxt = triangle_cross(hull[i], hull[next_i], hull[(j + 1) % n]);
            if nxt <= cur {
                break;
            }
            j = (j + 1) % n;
        }
        let d_ij = dist_sq(hull[i], hull[j]);
        if d_ij > best_sq {
            best_sq = d_ij;
        }
        let d_ij1 = dist_sq(hull[next_i], hull[j]);
        if d_ij1 > best_sq {
            best_sq = d_ij1;
        }
    }
    best_sq.sqrt()
}

/// Returns the pair of vertices that realise the polygon diameter, or
/// `None` for inputs with fewer than two vertices.
///
/// For two-vertex input the pair is simply `(hull[0], hull[1])`. For three
/// or more vertices the farthest antipodal pair found by rotating calipers
/// is returned. Ties are broken by the order in which pairs are visited.
///
/// The same convex-CCW precondition as [`polygon_diameter`] applies.
pub fn diameter_pair(hull: &[(f64, f64)]) -> Option<((f64, f64), (f64, f64))> {
    let n = hull.len();
    match n {
        0 | 1 => return None,
        2 => return Some((hull[0], hull[1])),
        _ => {}
    }

    let mut best_sq = 0.0_f64;
    let mut best_pair = (hull[0], hull[1]);
    let mut j = 1_usize;
    for i in 0..n {
        let next_i = (i + 1) % n;
        loop {
            let cur = triangle_cross(hull[i], hull[next_i], hull[j]);
            let nxt = triangle_cross(hull[i], hull[next_i], hull[(j + 1) % n]);
            if nxt <= cur {
                break;
            }
            j = (j + 1) % n;
        }
        let d_ij = dist_sq(hull[i], hull[j]);
        if d_ij > best_sq {
            best_sq = d_ij;
            best_pair = (hull[i], hull[j]);
        }
        let d_ij1 = dist_sq(hull[next_i], hull[j]);
        if d_ij1 > best_sq {
            best_sq = d_ij1;
            best_pair = (hull[next_i], hull[j]);
        }
    }
    Some(best_pair)
}

#[inline]
fn dist_sq(a: (f64, f64), b: (f64, f64)) -> f64 {
    let dx = a.0 - b.0;
    let dy = a.1 - b.1;
    dx.mul_add(dx, dy * dy)
}

#[inline]
fn dist(a: (f64, f64), b: (f64, f64)) -> f64 {
    dist_sq(a, b).sqrt()
}

/// Twice the signed area of triangle `(a, b, c)` — i.e. the 2D cross
/// product `(b − a) × (c − a)`. Positive when `c` lies to the left of the
/// directed edge `a → b`.
#[inline]
fn triangle_cross(a: (f64, f64), b: (f64, f64), c: (f64, f64)) -> f64 {
    let abx = b.0 - a.0;
    let aby = b.1 - a.1;
    let acx = c.0 - a.0;
    let acy = c.1 - a.1;
    abx.mul_add(acy, -(aby * acx))
}

#[cfg(test)]
mod tests {
    use super::{diameter_pair, dist_sq, polygon_diameter};
    use quickcheck_macros::quickcheck;
    use std::f64::consts::PI;

    const EPS: f64 = 1e-9;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() <= eps
    }

    fn brute_force_diameter(pts: &[(f64, f64)]) -> f64 {
        let mut best = 0.0_f64;
        for i in 0..pts.len() {
            for j in (i + 1)..pts.len() {
                let d = dist_sq(pts[i], pts[j]);
                if d > best {
                    best = d;
                }
            }
        }
        best.sqrt()
    }

    #[test]
    fn empty_is_zero() {
        let v: Vec<(f64, f64)> = Vec::new();
        assert_eq!(polygon_diameter(&v), 0.0);
        assert_eq!(diameter_pair(&v), None);
    }

    #[test]
    fn single_point_is_zero() {
        let v = vec![(3.0, 4.0)];
        assert_eq!(polygon_diameter(&v), 0.0);
        assert_eq!(diameter_pair(&v), None);
    }

    #[test]
    fn two_points_is_distance() {
        let v = vec![(0.0, 0.0), (3.0, 4.0)];
        assert!(approx_eq(polygon_diameter(&v), 5.0, EPS));
        let pair = diameter_pair(&v).unwrap();
        assert_eq!(pair, ((0.0, 0.0), (3.0, 4.0)));
    }

    #[test]
    fn equilateral_triangle_diameter_is_side_length() {
        // Equilateral triangle with side 1, oriented CCW.
        let s = 1.0_f64;
        let h = s * (3.0_f64).sqrt() / 2.0;
        let v = vec![(0.0, 0.0), (s, 0.0), (s / 2.0, h)];
        assert!(approx_eq(polygon_diameter(&v), s, EPS));
    }

    #[test]
    fn unit_square_diameter_is_diagonal() {
        let v = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        let expected = (2.0_f64).sqrt();
        assert!(approx_eq(polygon_diameter(&v), expected, EPS));
        let (p, q) = diameter_pair(&v).unwrap();
        // The diametral pair must be one of the two diagonals.
        let d = dist_sq(p, q).sqrt();
        assert!(approx_eq(d, expected, EPS));
    }

    #[test]
    fn regular_hexagon_diameter_is_twice_radius() {
        let r = 2.5_f64;
        let mut v = Vec::with_capacity(6);
        for i in 0..6 {
            let theta = 2.0 * PI * (i as f64) / 6.0;
            v.push((r * theta.cos(), r * theta.sin()));
        }
        // Already CCW (theta increasing).
        assert!(approx_eq(polygon_diameter(&v), 2.0 * r, 1e-12));
    }

    #[test]
    fn classic_small_example() {
        // Convex pentagon, CCW. Longest pair is (0, 0) ↔ (4, 3) with
        // distance 5.
        let v = vec![(0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (2.0, 4.0), (0.0, 3.0)];
        assert!(approx_eq(polygon_diameter(&v), 5.0, EPS));
        assert!(approx_eq(
            polygon_diameter(&v),
            brute_force_diameter(&v),
            EPS
        ));
    }

    #[test]
    fn regular_polygon_many_sides() {
        // Regular 17-gon inscribed in a circle of radius r centred at the
        // origin. Diameter = 2r (within numerical tolerance).
        let r = 1.7_f64;
        let n = 17;
        let mut v = Vec::with_capacity(n);
        for i in 0..n {
            let theta = 2.0 * PI * (i as f64) / (n as f64);
            v.push((r * theta.cos(), r * theta.sin()));
        }
        let d = polygon_diameter(&v);
        // For odd-vertex regular polygons, the maximum vertex-to-vertex
        // distance is 2r * cos(π / (2n)).
        let expected = 2.0 * r * (PI / (2.0 * n as f64)).cos();
        assert!(approx_eq(d, expected, 1e-12));
        assert!(approx_eq(d, brute_force_diameter(&v), 1e-12));
    }

    /// Build the CCW convex hull of `pts` via the monotone-chain
    /// (Andrew's) algorithm. Used only to feed the property test below.
    fn convex_hull_ccw(pts: &[(f64, f64)]) -> Vec<(f64, f64)> {
        let mut p: Vec<(f64, f64)> = pts.to_vec();
        p.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        p.dedup();
        let n = p.len();
        if n <= 1 {
            return p;
        }
        let cross = |o: (f64, f64), a: (f64, f64), b: (f64, f64)| -> f64 {
            (a.0 - o.0).mul_add(b.1 - o.1, -((a.1 - o.1) * (b.0 - o.0)))
        };
        let mut h: Vec<(f64, f64)> = Vec::with_capacity(2 * n);
        // Lower hull.
        for &pt in &p {
            while h.len() >= 2 && cross(h[h.len() - 2], h[h.len() - 1], pt) <= 0.0 {
                h.pop();
            }
            h.push(pt);
        }
        // Upper hull.
        let lower_len = h.len() + 1;
        for &pt in p.iter().rev().skip(1) {
            while h.len() >= lower_len && cross(h[h.len() - 2], h[h.len() - 1], pt) <= 0.0 {
                h.pop();
            }
            h.push(pt);
        }
        h.pop();
        h
    }

    #[quickcheck]
    fn matches_brute_force_on_random_hulls(raw: Vec<(i16, i16)>) -> bool {
        // Cap to ≤ 12 points so the brute force stays cheap and the hulls
        // stay small.
        let pts: Vec<(f64, f64)> = raw
            .into_iter()
            .take(12)
            .map(|(x, y)| (f64::from(x), f64::from(y)))
            .collect();
        let hull = convex_hull_ccw(&pts);
        let calipers = polygon_diameter(&hull);
        let brute = brute_force_diameter(&hull);
        approx_eq(calipers, brute, 1e-9)
    }
}
