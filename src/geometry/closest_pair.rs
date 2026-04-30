//! Closest pair of points via divide and conquer.
//!
//! Given a set of `n` points in the plane, return a pair `(p, q)` whose
//! Euclidean distance is minimal among all `n·(n−1)/2` pairs, together
//! with that distance.
//!
//! # Algorithm
//!
//! The classic Shamos–Hoey divide-and-conquer scheme:
//!
//! 1. Sort the input by `x` coordinate (ties broken by `y`).
//! 2. Recursively solve the left and right halves, obtaining their
//!    minimum distances `d_l` and `d_r`. Let `δ = min(d_l, d_r)` and let
//!    `(p_best, q_best)` be the corresponding pair.
//! 3. **Strip step.** Any cross-half pair closer than `δ` must lie in
//!    the vertical strip of width `2·δ` straddling the dividing line.
//!    Walk the strip points in order of increasing `y`; for each point,
//!    only the next handful of points within `δ` in `y` need to be
//!    checked. A standard packing argument bounds that handful by a
//!    constant (at most 7 successors), so the strip step is `O(n)`.
//! 4. Return whichever of the recursive best and the strip best is
//!    smaller.
//!
//! Sorting by `y` once up front and then taking strip-respecting
//! sub-slices on each recursive call lets the merge step run in linear
//! time, giving the overall `O(n log n)` time bound. Auxiliary storage
//! is `O(n)` for the presorted arrays and the strip buffer.
//!
//! # Complexity
//!
//! * Time: `O(n log n)`.
//! * Space: `O(n)` auxiliary.
//!
//! # Preconditions
//!
//! Coordinates must be finite (`NaN` or infinite inputs would corrupt
//! the sort). Duplicate points are permitted; their distance is `0` and
//! the routine will return them as the closest pair.
//!
//! # Determinism
//!
//! On ties (multiple pairs sharing the minimum distance) the specific
//! pair returned depends on the recursion's tie-breaking and is *not*
//! part of the public contract. Only the returned distance is
//! contractual.

/// Squared Euclidean distance between two points.
#[inline]
fn dist_sq(a: (f64, f64), b: (f64, f64)) -> f64 {
    let dx = a.0 - b.0;
    let dy = a.1 - b.1;
    dx.mul_add(dx, dy * dy)
}

/// Euclidean distance between two points.
#[inline]
fn dist(a: (f64, f64), b: (f64, f64)) -> f64 {
    dist_sq(a, b).sqrt()
}

/// `O(n²)` brute-force closest pair, used as the recursion base case
/// and as a reference in tests.
fn brute_force(points: &[(f64, f64)]) -> ((f64, f64), (f64, f64), f64) {
    debug_assert!(points.len() >= 2);
    let mut best_sq = f64::INFINITY;
    let mut best_pair = (points[0], points[1]);
    for i in 0..points.len() {
        for j in (i + 1)..points.len() {
            let d = dist_sq(points[i], points[j]);
            if d < best_sq {
                best_sq = d;
                best_pair = (points[i], points[j]);
            }
        }
    }
    (best_pair.0, best_pair.1, best_sq.sqrt())
}

/// Recursive closest-pair driver.
///
/// `px` is the slice of points sorted by `x` (ties broken by `y`); `py`
/// is the same set of points sorted by `y`. Both views must contain
/// exactly the same multiset of points.
fn closest_pair_rec(px: &[(f64, f64)], py: &[(f64, f64)]) -> ((f64, f64), (f64, f64), f64) {
    let n = px.len();
    // Small sub-problems: brute force is faster and avoids the overhead
    // of the strip step. The classic threshold of 3 also dodges the
    // mid = n/2 = 0 corner when n < 2.
    if n <= 3 {
        return brute_force(px);
    }

    let mid = n / 2;
    let midpoint = px[mid];
    // Split px in two halves at the median index. Both px and py are
    // sorted lexicographically — by (x, y) and (y, x) respectively —
    // and points sharing x == midpoint.x are therefore in the *same
    // y-order* in both views. That lets us partition py without any
    // per-point lookup: of the points with x == midpoint.x, the first
    // `split_left_count` (in y-order) belong to the left half, the
    // rest to the right.
    let (left_x, right_x) = px.split_at(mid);
    let split_left_count = left_x.iter().filter(|p| p.0 == midpoint.0).count();

    let mut py_left = Vec::with_capacity(mid);
    let mut py_right = Vec::with_capacity(n - mid);
    let mut split_seen = 0usize;
    for &p in py {
        if p.0 < midpoint.0 {
            py_left.push(p);
        } else if p.0 > midpoint.0 {
            py_right.push(p);
        } else {
            // x == midpoint.x: assign by encounter order in y.
            if split_seen < split_left_count {
                py_left.push(p);
            } else {
                py_right.push(p);
            }
            split_seen += 1;
        }
    }
    debug_assert_eq!(py_left.len(), mid);
    debug_assert_eq!(py_right.len(), n - mid);

    let left_best = closest_pair_rec(left_x, &py_left);
    let right_best = closest_pair_rec(right_x, &py_right);

    let mut best = if left_best.2 <= right_best.2 {
        left_best
    } else {
        right_best
    };

    // Strip step: collect points within delta in x of the dividing
    // line, in y-sorted order, then check each against the next few
    // successors.
    let delta = best.2;
    let strip: Vec<(f64, f64)> = py
        .iter()
        .copied()
        .filter(|p| (p.0 - midpoint.0).abs() < delta)
        .collect();

    for i in 0..strip.len() {
        // The packing argument bounds the inner loop by a constant
        // (≤ 7), but stop early as soon as the y-gap exceeds delta.
        let mut j = i + 1;
        while j < strip.len() && (strip[j].1 - strip[i].1) < best.2 {
            let d = dist(strip[i], strip[j]);
            if d < best.2 {
                best = (strip[i], strip[j], d);
            }
            j += 1;
        }
    }

    best
}

/// Returns `Some((p, q, distance))` for the two closest points in
/// `points`, or `None` if fewer than two points are supplied.
///
/// Runs in `O(n log n)` time and `O(n)` auxiliary space using the
/// classic divide-and-conquer scheme with the strip optimisation. See
/// the module-level documentation for the algorithm and preconditions.
///
/// On ties (multiple pairs sharing the minimum distance) the particular
/// pair returned is unspecified; only the distance is contractual.
#[must_use]
#[allow(clippy::type_complexity)]
pub fn closest_pair(points: &[(f64, f64)]) -> Option<((f64, f64), (f64, f64), f64)> {
    if points.len() < 2 {
        return None;
    }

    // Presort by x (ties by y) and by y (ties by x). The recursive
    // driver takes both views and never resorts.
    let mut px: Vec<(f64, f64)> = points.to_vec();
    px.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    });
    let mut py: Vec<(f64, f64)> = px.clone();
    py.sort_by(|a, b| {
        a.1.partial_cmp(&b.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
    });

    Some(closest_pair_rec(&px, &py))
}

#[cfg(test)]
mod tests {
    use super::{brute_force, closest_pair, dist};
    use quickcheck_macros::quickcheck;

    const EPS: f64 = 1e-9;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() <= eps
    }

    #[test]
    fn empty_returns_none() {
        let v: Vec<(f64, f64)> = Vec::new();
        assert!(closest_pair(&v).is_none());
    }

    #[test]
    fn single_returns_none() {
        let v = vec![(0.0, 0.0)];
        assert!(closest_pair(&v).is_none());
    }

    #[test]
    fn two_points() {
        let v = vec![(0.0, 0.0), (3.0, 4.0)];
        let (_, _, d) = closest_pair(&v).expect("two points");
        assert!(approx_eq(d, 5.0, EPS));
    }

    #[test]
    fn three_points_picks_minimum() {
        // (0,0)-(1,0) distance 1; (1,0)-(10,0) distance 9; (0,0)-(10,0) distance 10.
        let v = vec![(0.0, 0.0), (1.0, 0.0), (10.0, 0.0)];
        let (_, _, d) = closest_pair(&v).expect("three points");
        assert!(approx_eq(d, 1.0, EPS));
    }

    #[test]
    fn unit_square_corners() {
        // Four corners of the unit square: every adjacent pair is at
        // distance 1, the diagonals at sqrt(2). Closest pair distance is 1.
        let v = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        let (_, _, d) = closest_pair(&v).expect("four points");
        assert!(approx_eq(d, 1.0, EPS));
    }

    #[test]
    fn classic_small_example() {
        // Standard CLRS-style example. The closest pair is
        // (5,5)-(7,7) with distance 2*sqrt(2) ≈ 2.8284271…, while
        // (1,3)-(3,4) is sqrt(5) ≈ 2.236… which is closer still. So
        // the closest pair should be (1,3)-(3,4) at sqrt(5).
        let v = vec![
            (2.0, 3.0),
            (12.0, 30.0),
            (40.0, 50.0),
            (5.0, 1.0),
            (12.0, 10.0),
            (3.0, 4.0),
        ];
        let (_, _, d) = closest_pair(&v).expect("six points");
        // Brute-force the same input as oracle.
        let (_, _, expected) = brute_force(&v);
        assert!(approx_eq(d, expected, EPS));
    }

    #[test]
    fn identical_points_distance_zero() {
        let v = vec![(1.5, -2.5), (1.5, -2.5), (10.0, 10.0)];
        let (p, q, d) = closest_pair(&v).expect("three points");
        assert_eq!(d, 0.0);
        assert_eq!(p, (1.5, -2.5));
        assert_eq!(q, (1.5, -2.5));
    }

    #[test]
    fn duplicates_among_many() {
        // A duplicate hidden among many distinct points must be found.
        let mut v: Vec<(f64, f64)> = (0..50)
            .map(|i| (i as f64 * 3.0, (i as f64 * 0.7).sin() * 100.0))
            .collect();
        v.push((42.5, -7.25));
        v.push((42.5, -7.25));
        let (_, _, d) = closest_pair(&v).expect("plenty of points");
        assert_eq!(d, 0.0);
    }

    #[test]
    fn collinear_points() {
        // Increasing x, all y = 0; closest pair is the two with the
        // smallest x-gap.
        let v = vec![(0.0, 0.0), (5.0, 0.0), (5.5, 0.0), (10.0, 0.0), (12.5, 0.0)];
        let (_, _, d) = closest_pair(&v).expect("five points");
        assert!(approx_eq(d, 0.5, EPS));
    }

    #[test]
    fn vertical_collinear_points() {
        // All x = 0; closest pair is the two with smallest y-gap.
        let v = vec![(0.0, 0.0), (0.0, 0.25), (0.0, 1.0), (0.0, 3.0)];
        let (_, _, d) = closest_pair(&v).expect("four points");
        assert!(approx_eq(d, 0.25, EPS));
    }

    #[test]
    fn large_random_matches_brute_force() {
        // Deterministic LCG so the test stays reproducible without an
        // RNG dependency. Hash the index into a roughly-uniform f64.
        fn pseudo(i: u64) -> f64 {
            let x = i.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let r = (x >> 11) as f64 / ((1u64 << 53) as f64);
            r * 200.0 - 100.0
        }
        let n = 500usize;
        let v: Vec<(f64, f64)> = (0..n as u64)
            .map(|i| (pseudo(i * 2), pseudo(i * 2 + 1)))
            .collect();
        let (_, _, d) = closest_pair(&v).expect("many points");
        let (_, _, expected) = brute_force(&v);
        assert!(approx_eq(d, expected, 1e-12));
    }

    // Property test: against brute force on small random inputs.
    // QuickCheck generates a Vec<(i16, i16)>; we project the integers
    // into [-100, 100] and cap the size at 30 to keep brute force
    // cheap. The pair returned may differ from the brute-force pair on
    // ties, but the distance must match.
    #[allow(clippy::needless_pass_by_value)]
    #[quickcheck]
    fn quickcheck_matches_brute_force(seed: Vec<(i16, i16)>) -> bool {
        if seed.len() < 2 {
            return true;
        }
        let pts: Vec<(f64, f64)> = seed
            .iter()
            .take(30)
            .map(|&(x, y)| {
                let xf = (i32::from(x) % 201) as f64;
                let yf = (i32::from(y) % 201) as f64;
                (xf, yf)
            })
            .collect();
        if pts.len() < 2 {
            return true;
        }
        let (p, q, d) = closest_pair(&pts).expect("len >= 2");
        let (_, _, expected) = brute_force(&pts);
        // Distance must match brute force; the returned pair must
        // actually realise that distance.
        approx_eq(d, expected, 1e-12) && approx_eq(dist(p, q), d, 1e-12)
    }
}
