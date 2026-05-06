//! Manhattan ↔ Chebyshev distance transform.
//!
//! For any two planar points `p, q`,
//!
//! ```text
//!     |p.x - q.x| + |p.y - q.y|         (Manhattan / L1)
//! ```
//!
//! is preserved under the rotation-and-scaling map `(x, y) -> (x + y, x - y)`,
//! which carries it to the Chebyshev distance
//!
//! ```text
//!     max(|u - u'|, |v - v'|)            (Chebyshev / L∞)
//! ```
//!
//! The transform converts a Manhattan-norm problem into a Chebyshev-norm
//! problem (where each coordinate can be optimised independently) and vice
//! versa, so problems like "farthest pair under L1 in 2-D" reduce to a sweep
//! over the maximum spread on each transformed axis.
//!
//! Each conversion is `O(1)` per point, `O(n)` over a slice. The transform is
//! its own inverse up to a factor of two: `inverse(forward(p)) == (2x, 2y)`,
//! so when working over integers prefer the `*_i64` overload to avoid
//! introducing halves.

/// Forward transform `(x, y) -> (x + y, x - y)` over `f64`. Maps Manhattan
/// (L1) distance to Chebyshev (L∞) distance.
#[must_use]
pub fn forward_f64((x, y): (f64, f64)) -> (f64, f64) {
    (x + y, x - y)
}

/// Inverse transform `(u, v) -> ((u + v) / 2, (u - v) / 2)` over `f64`. Maps
/// Chebyshev distance back to Manhattan distance.
#[must_use]
pub fn inverse_f64((u, v): (f64, f64)) -> (f64, f64) {
    (f64::midpoint(u, v), f64::midpoint(u, -v))
}

/// Forward transform `(x, y) -> (x + y, x - y)` over `i64`. The inverse on
/// integers carries a factor of two, so use [`inverse_i64_doubled`] for the
/// integer round-trip.
#[must_use]
pub const fn forward_i64((x, y): (i64, i64)) -> (i64, i64) {
    (x + y, x - y)
}

/// Inverse transform on `i64` returning `(2x, 2y)` so the result stays
/// integer. Divide by two if you know the input came from a forward transform
/// of integers with the same parity in both coordinates.
#[must_use]
pub const fn inverse_i64_doubled((u, v): (i64, i64)) -> (i64, i64) {
    (u + v, u - v)
}

/// Manhattan (L1) distance between two `i64` points.
#[must_use]
pub const fn manhattan_distance(p: (i64, i64), q: (i64, i64)) -> i64 {
    (p.0 - q.0).abs() + (p.1 - q.1).abs()
}

/// Chebyshev (L∞) distance between two `i64` points.
#[must_use]
pub const fn chebyshev_distance(p: (i64, i64), q: (i64, i64)) -> i64 {
    let dx = (p.0 - q.0).abs();
    let dy = (p.1 - q.1).abs();
    if dx > dy {
        dx
    } else {
        dy
    }
}

/// Maximum Manhattan distance between any pair of points in `points`, in
/// `O(n)`. Returns `0` for fewer than two points.
///
/// Algorithm: forward-transform every point to Chebyshev coordinates, then
/// the maximum L∞ pairwise distance equals the larger of `max u - min u` and
/// `max v - min v`.
#[must_use]
pub fn max_pairwise_manhattan(points: &[(i64, i64)]) -> i64 {
    if points.len() < 2 {
        return 0;
    }
    let mut min_u = i64::MAX;
    let mut max_u = i64::MIN;
    let mut min_v = i64::MAX;
    let mut max_v = i64::MIN;
    for &p in points {
        let (u, v) = forward_i64(p);
        if u < min_u {
            min_u = u;
        }
        if u > max_u {
            max_u = u;
        }
        if v < min_v {
            min_v = v;
        }
        if v > max_v {
            max_v = v;
        }
    }
    (max_u - min_u).max(max_v - min_v)
}

#[cfg(test)]
mod tests {
    use super::{
        chebyshev_distance, forward_f64, forward_i64, inverse_f64, inverse_i64_doubled,
        manhattan_distance, max_pairwise_manhattan,
    };
    use quickcheck_macros::quickcheck;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-9
    }

    #[test]
    fn forward_inverse_round_trip_f64() {
        let p = (3.0_f64, -7.5);
        let r = inverse_f64(forward_f64(p));
        assert!(approx_eq(r.0, p.0));
        assert!(approx_eq(r.1, p.1));
    }

    #[test]
    fn forward_then_doubled_inverse_i64() {
        let p = (3_i64, -7);
        let (u, v) = forward_i64(p);
        let (dx, dy) = inverse_i64_doubled((u, v));
        assert_eq!((dx, dy), (2 * p.0, 2 * p.1));
    }

    #[test]
    fn manhattan_to_chebyshev_under_forward() {
        let p = (3_i64, 2);
        let q = (-1, 7);
        let m = manhattan_distance(p, q);
        let c = chebyshev_distance(forward_i64(p), forward_i64(q));
        assert_eq!(m, c);
    }

    #[test]
    fn max_pairwise_handles_trivial_inputs() {
        assert_eq!(max_pairwise_manhattan(&[]), 0);
        assert_eq!(max_pairwise_manhattan(&[(5, 5)]), 0);
    }

    #[test]
    fn max_pairwise_known_set() {
        // brute-force verification on a small set
        let pts = [(0_i64, 0), (3, 4), (-2, 1), (5, -1)];
        let mut want = 0;
        for i in 0..pts.len() {
            for j in (i + 1)..pts.len() {
                want = want.max(manhattan_distance(pts[i], pts[j]));
            }
        }
        assert_eq!(max_pairwise_manhattan(&pts), want);
    }

    fn brute_max_manhattan(pts: &[(i64, i64)]) -> i64 {
        let mut best = 0;
        for i in 0..pts.len() {
            for j in (i + 1)..pts.len() {
                best = best.max(manhattan_distance(pts[i], pts[j]));
            }
        }
        best
    }

    #[allow(clippy::needless_pass_by_value)]
    #[quickcheck]
    fn forward_preserves_distance(a: (i16, i16), b: (i16, i16)) -> bool {
        let p = (i64::from(a.0), i64::from(a.1));
        let q = (i64::from(b.0), i64::from(b.1));
        manhattan_distance(p, q) == chebyshev_distance(forward_i64(p), forward_i64(q))
    }

    #[allow(clippy::needless_pass_by_value)]
    #[quickcheck]
    fn max_pairwise_matches_brute(pts: Vec<(i16, i16)>) -> bool {
        let pts: Vec<(i64, i64)> = pts
            .into_iter()
            .take(20)
            .map(|(x, y)| (i64::from(x), i64::from(y)))
            .collect();
        max_pairwise_manhattan(&pts) == brute_max_manhattan(&pts)
    }
}
