//! Rabin's randomised closest pair of points.
//!
//! Given a set of `n` points in the plane, return a pair `(p, q)` whose
//! Euclidean distance is minimal among all `n·(n−1)/2` pairs, together
//! with that distance. This module realises the classical randomised
//! algorithm attributed to M. O. Rabin (and later analysed and
//! popularised by Khuller and Matias).
//!
//! # Algorithm
//!
//! 1. **Sample.** Draw a random subset `S` of size `≈ n^{2/3}` from the
//!    input. Compute the closest-pair distance `δ` of `S` by brute
//!    force. (For small `n` we just brute-force the whole input.)
//! 2. **Bucket.** Lay down a uniform grid with cell side `δ/2`. Hash
//!    every point into the cell that contains it.
//! 3. **Scan.** For each input point, examine the points of the 5×5
//!    block of cells centred on its own cell. Any pair closer than `δ`
//!    must share such a 5×5 neighbourhood, so the scan finds the global
//!    closest pair. Each cell holds `O(1)` points in expectation, so
//!    the scan runs in expected `O(n)` time.
//!
//! # Complexity
//!
//! * Time: expected `O(n)`. The sample step costs `O(|S|^2) = O(n^{4/3})`,
//!   which is dominated by the `O(n)` scan in expectation; the constant
//!   in front of `n` is the expected number of points in a 5×5
//!   neighbourhood, which is bounded for random inputs.
//! * Space: `O(n)` for the grid hash map.
//!
//! # Determinism
//!
//! Randomness is supplied through the `seed` parameter and consumed by
//! a deterministic `xorshift64*` PRNG, so a given `(points, seed)` pair
//! always produces the same answer. The exact pair returned on ties is
//! unspecified; only the returned distance is contractual.
//!
//! # Preconditions
//!
//! Coordinates must be finite. Duplicate points are permitted; their
//! distance is `0` and the routine returns one of them as the closest
//! pair. If the sampled subset contains a duplicate, `δ = 0` makes the
//! grid step degenerate, so we fall through to a brute-force pass on
//! the (necessarily small) input — see the implementation for details.

use std::collections::HashMap;

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

/// `O(n²)` brute-force closest pair, used on the random subset and as
/// a fallback for small inputs.
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

/// Deterministic `xorshift64*` PRNG. State must be non-zero; we
/// substitute a fixed non-zero constant when the caller hands us `0`.
#[derive(Clone)]
struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    const fn new(seed: u64) -> Self {
        // xorshift requires non-zero state; pick an arbitrary odd
        // constant for the degenerate seed.
        let state = if seed == 0 {
            0x9E37_79B9_7F4A_7C15
        } else {
            seed
        };
        Self { state }
    }

    const fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform integer in `[0, bound)`. Uses Lemire's nearly-divisionless
    /// trick simplified for our purposes.
    fn gen_range(&mut self, bound: usize) -> usize {
        debug_assert!(bound > 0);
        let bound = bound as u64;
        // Modulo introduces a tiny bias; for our combinatorial use
        // (sampling indices) the bias is harmless.
        (self.next_u64() % bound) as usize
    }
}

/// Cell-coordinates of a point under a grid of side `cell`.
#[inline]
fn cell_of(p: (f64, f64), cell: f64) -> (i64, i64) {
    ((p.0 / cell).floor() as i64, (p.1 / cell).floor() as i64)
}

/// Scan every point against the 5×5 block of grid cells around it,
/// updating the running best pair. Returns the best pair found.
///
/// Buckets store input *indices* rather than coordinates so that a
/// point is never compared against itself, while genuine duplicates
/// (which sit at different indices) are still discovered with
/// distance `0`.
fn grid_scan(
    points: &[(f64, f64)],
    delta: f64,
    seed_best: ((f64, f64), (f64, f64), f64),
) -> ((f64, f64), (f64, f64), f64) {
    let cell = delta / 2.0;
    let mut grid: HashMap<(i64, i64), Vec<usize>> = HashMap::new();
    for (i, &p) in points.iter().enumerate() {
        grid.entry(cell_of(p, cell)).or_default().push(i);
    }

    let mut best = seed_best;
    for (i, &p) in points.iter().enumerate() {
        let (cx, cy) = cell_of(p, cell);
        for dx in -2..=2 {
            for dy in -2..=2 {
                if let Some(bucket) = grid.get(&(cx + dx, cy + dy)) {
                    for &j in bucket {
                        // Order pairs by index to count each unordered
                        // pair exactly once and to skip the (i,i)
                        // self-comparison without dropping duplicates.
                        if j <= i {
                            continue;
                        }
                        let q = points[j];
                        let d = dist(p, q);
                        if d < best.2 {
                            best = (p, q, d);
                        }
                    }
                }
            }
        }
    }
    best
}

/// Returns `Some((p, q, distance))` for the two closest points in
/// `points`, or `None` if fewer than two points are supplied.
///
/// Runs in expected `O(n)` time using Rabin's randomised sample-and-
/// bucket scheme. The `seed` parameter feeds a deterministic xorshift
/// PRNG, so calls with the same `(points, seed)` are reproducible.
///
/// On ties (multiple pairs sharing the minimum distance) the particular
/// pair returned is unspecified; only the distance is contractual.
#[must_use]
#[allow(clippy::type_complexity)]
pub fn closest_pair(points: &[(f64, f64)], seed: u64) -> Option<((f64, f64), (f64, f64), f64)> {
    let n = points.len();
    if n < 2 {
        return None;
    }
    // For small inputs the brute-force cost dominates the bookkeeping.
    if n <= 32 {
        return Some(brute_force(points));
    }

    let mut rng = XorShift64::new(seed);

    // Sample size: ceil(n^{2/3}), clamped to at least 2.
    let sample_size = {
        let s = (n as f64).powf(2.0 / 3.0).ceil() as usize;
        s.max(2).min(n)
    };

    // Sample *without* replacement via a partial Fisher–Yates shuffle
    // over the index space. Sampling with replacement would let the
    // same index be drawn twice, producing a spurious δ = 0 even when
    // the input has no real duplicates.
    let mut indices: Vec<usize> = (0..n).collect();
    for i in 0..sample_size {
        let j = i + rng.gen_range(n - i);
        indices.swap(i, j);
    }
    let sample: Vec<(f64, f64)> = indices[..sample_size].iter().map(|&i| points[i]).collect();

    let seed_best = brute_force(&sample);

    // delta == 0 here means the input genuinely contains a duplicate
    // (the sample indices are distinct), and that duplicate is a
    // global closest pair. The grid step would be undefined, so
    // short-circuit.
    if seed_best.2 == 0.0 {
        return Some(seed_best);
    }

    // Bucket every input point on a delta/2 grid and scan the 5×5
    // neighbourhood of each. Any pair closer than delta must collide
    // in some 5×5 block, so this finds the true minimum.
    Some(grid_scan(points, seed_best.2, seed_best))
}

#[cfg(test)]
mod tests {
    use super::{brute_force, closest_pair, dist};

    const EPS: f64 = 1e-9;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() <= eps
    }

    #[test]
    fn empty_returns_none() {
        let v: Vec<(f64, f64)> = Vec::new();
        assert!(closest_pair(&v, 0).is_none());
    }

    #[test]
    fn single_returns_none() {
        let v = vec![(0.0, 0.0)];
        assert!(closest_pair(&v, 0).is_none());
    }

    #[test]
    fn two_points_trivial() {
        let v = vec![(0.0, 0.0), (3.0, 4.0)];
        let (_, _, d) = closest_pair(&v, 1).expect("two points");
        assert!(approx_eq(d, 5.0, EPS));
    }

    #[test]
    fn three_collinear_points() {
        // (0,0)-(1,0) distance 1; (1,0)-(10,0) distance 9; (0,0)-(10,0) distance 10.
        let v = vec![(0.0, 0.0), (1.0, 0.0), (10.0, 0.0)];
        let (_, _, d) = closest_pair(&v, 42).expect("three points");
        assert!(approx_eq(d, 1.0, EPS));
    }

    #[test]
    fn unit_square_corners() {
        let v = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        let (_, _, d) = closest_pair(&v, 7).expect("four points");
        assert!(approx_eq(d, 1.0, EPS));
    }

    #[test]
    fn duplicate_points_distance_zero() {
        let v = vec![(1.5, -2.5), (1.5, -2.5), (10.0, 10.0)];
        let (_, _, d) = closest_pair(&v, 99).expect("three points");
        assert_eq!(d, 0.0);
    }

    #[test]
    fn known_minimum_exact_distance() {
        // Deliberately spread points apart; the (3,4)-(3.001,4.0) pair
        // is the unique nearest pair at distance exactly 0.001.
        let v = vec![
            (0.0, 0.0),
            (10.0, 0.0),
            (10.0, 10.0),
            (0.0, 10.0),
            (3.0, 4.0),
            (3.001, 4.0),
            (-5.0, 7.0),
        ];
        let (_, _, d) = closest_pair(&v, 12345).expect("seven points");
        assert!(approx_eq(d, 0.001, 1e-12));
    }

    /// Deterministic pseudorandom point generator for tests — avoids
    /// pulling in an RNG dependency and keeps the suite reproducible.
    fn pseudo(i: u64) -> f64 {
        let x = i.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        let r = (x >> 11) as f64 / ((1u64 << 53) as f64);
        r * 200.0 - 100.0
    }

    #[test]
    fn large_random_matches_brute_force() {
        let n = 400usize;
        let v: Vec<(f64, f64)> = (0..n as u64)
            .map(|i| (pseudo(i * 2), pseudo(i * 2 + 1)))
            .collect();
        let (p, q, d) = closest_pair(&v, 0x00C0_FFEE).expect("many points");
        let (_, _, expected) = brute_force(&v);
        assert!(approx_eq(d, expected, 1e-12));
        // Returned pair must actually realise the reported distance.
        assert!(approx_eq(dist(p, q), d, 1e-12));
    }

    #[test]
    fn matches_brute_force_across_seeds() {
        // Vary the seed to exercise the randomised sampling branch.
        let n = 150usize;
        let v: Vec<(f64, f64)> = (0..n as u64)
            .map(|i| (pseudo(i * 5 + 11), pseudo(i * 5 + 13)))
            .collect();
        let (_, _, expected) = brute_force(&v);
        for seed in [0u64, 1, 2, 3, 7, 31, 1024, u64::MAX] {
            let (_, _, d) = closest_pair(&v, seed).expect("many points");
            assert!(
                approx_eq(d, expected, 1e-12),
                "seed {seed} produced d={d}, expected {expected}"
            );
        }
    }

    #[test]
    fn deterministic_for_fixed_seed() {
        let n = 80usize;
        let v: Vec<(f64, f64)> = (0..n as u64)
            .map(|i| (pseudo(i * 3), pseudo(i * 3 + 1)))
            .collect();
        let a = closest_pair(&v, 2026).expect("many points");
        let b = closest_pair(&v, 2026).expect("many points");
        assert_eq!(a, b);
    }
}
