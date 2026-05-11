//! Planar bipartite matching without crossings.
//!
//! Given `n` red points and `n` blue points in the plane, in general
//! position, compute a perfect matching `red[i] ↔ blue[π(i)]` such that
//! no two matching segments cross.
//!
//! # Algorithm
//!
//! We rely on a classical observation: **any minimum-total-length
//! perfect matching between the two color classes is automatically
//! crossing-free.** The proof is the standard swap argument — if two
//! segments `r_a–b_x` and `r_c–b_y` were to cross at a point `p`, then by
//! the triangle inequality
//!
//! ```text
//! |r_a b_y| + |r_c b_x| < |r_a b_x| + |r_c b_y|,
//! ```
//!
//! so swapping the partners (`r_a ↔ b_y`, `r_c ↔ b_x`) strictly
//! decreases the total length, contradicting minimality.
//!
//! Therefore, finding *some* non-crossing matching reduces to finding
//! *the* minimum-length matching. For very small `n` (the issue caps at
//! `n ≤ 8`) we can simply enumerate all `n!` permutations of the blue
//! indices, score each by the sum of Euclidean segment lengths, and
//! keep the best one. No fancy assignment-problem machinery is needed.
//!
//! Permutations are generated iteratively with Heap's algorithm so we
//! avoid recursion overhead and any allocation inside the inner loop.
//!
//! # Complexity
//!
//! Let `n = |red| = |blue|`.
//!
//! * Time: `O(n · n!)` — `n!` permutations, each scored in `O(n)`.
//! * Space: `O(n)` auxiliary for the working permutation and the best
//!   permutation seen so far.
//!
//! For the documented input range `n ≤ 8` the worst case is
//! `8 · 8! = 322 560` segment evaluations, which is trivially fast.
//!
//! # Panics
//!
//! Panics if `red.len() != blue.len()`.

/// Squared Euclidean distance between two planar points.
#[inline]
fn sq_dist(a: (f64, f64), b: (f64, f64)) -> f64 {
    let dx = a.0 - b.0;
    let dy = a.1 - b.1;
    dx.mul_add(dx, dy * dy)
}

/// Total Euclidean length of the matching defined by `perm`,
/// where red index `i` is paired with blue index `perm[i]`.
fn total_length(red: &[(f64, f64)], blue: &[(f64, f64)], perm: &[usize]) -> f64 {
    let mut sum = 0.0;
    for (i, &j) in perm.iter().enumerate() {
        sum += sq_dist(red[i], blue[j]).sqrt();
    }
    sum
}

/// Find a perfect matching between `red` and `blue` whose line segments
/// do not cross, by computing a minimum-total-length matching via brute
/// force.
///
/// Returns a vector of `(red_index, blue_index)` pairs, sorted by
/// `red_index` (so the `i`-th entry is `(i, π(i))`).
///
/// # Panics
///
/// Panics if `red.len() != blue.len()`.
///
/// # Notes
///
/// Intended for small inputs (`n ≤ 8` per the issue). The cost grows as
/// `n!`, so callers with larger `n` should reach for the Hungarian
/// algorithm or a min-cost flow instead.
pub fn min_length_matching(red: &[(f64, f64)], blue: &[(f64, f64)]) -> Vec<(usize, usize)> {
    assert_eq!(
        red.len(),
        blue.len(),
        "planar bipartite matching requires equal-sized color classes"
    );

    let n = red.len();
    if n == 0 {
        return Vec::new();
    }

    let mut perm: Vec<usize> = (0..n).collect();
    let mut best_perm: Vec<usize> = perm.clone();
    let mut best_len = total_length(red, blue, &perm);

    // Heap's algorithm — iterative, in-place permutation enumeration.
    let mut c = vec![0usize; n];
    let mut i = 0;
    while i < n {
        if c[i] < i {
            if i % 2 == 0 {
                perm.swap(0, i);
            } else {
                perm.swap(c[i], i);
            }
            let len = total_length(red, blue, &perm);
            if len < best_len {
                best_len = len;
                best_perm.copy_from_slice(&perm);
            }
            c[i] += 1;
            i = 0;
        } else {
            c[i] = 0;
            i += 1;
        }
    }

    best_perm.into_iter().enumerate().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Brute-force check: do segments `(a1, a2)` and `(b1, b2)` strictly
    /// cross at an interior point? Used to verify the no-crossing
    /// guarantee in tests.
    fn orient(p: (f64, f64), q: (f64, f64), r: (f64, f64)) -> f64 {
        (q.0 - p.0).mul_add(r.1 - p.1, -((q.1 - p.1) * (r.0 - p.0)))
    }

    fn segments_cross(a1: (f64, f64), a2: (f64, f64), b1: (f64, f64), b2: (f64, f64)) -> bool {
        let o1 = orient(a1, a2, b1);
        let o2 = orient(a1, a2, b2);
        let o3 = orient(b1, b2, a1);
        let o4 = orient(b1, b2, a2);
        // Strict crossing — endpoints touching does not count.
        (o1 * o2 < 0.0) && (o3 * o4 < 0.0)
    }

    fn matching_length(red: &[(f64, f64)], blue: &[(f64, f64)], m: &[(usize, usize)]) -> f64 {
        m.iter()
            .map(|&(i, j)| sq_dist(red[i], blue[j]).sqrt())
            .sum()
    }

    fn assert_no_crossings(red: &[(f64, f64)], blue: &[(f64, f64)], m: &[(usize, usize)]) {
        for a in 0..m.len() {
            for b in (a + 1)..m.len() {
                let (ra, ba) = m[a];
                let (rb, bb) = m[b];
                assert!(
                    !segments_cross(red[ra], blue[ba], red[rb], blue[bb]),
                    "segments {:?}-{:?} and {:?}-{:?} cross",
                    red[ra],
                    blue[ba],
                    red[rb],
                    blue[bb]
                );
            }
        }
    }

    #[test]
    fn empty_inputs() {
        let m = min_length_matching(&[], &[]);
        assert!(m.is_empty());
    }

    #[test]
    fn single_pair() {
        let red = vec![(0.0, 0.0)];
        let blue = vec![(3.0, 4.0)];
        let m = min_length_matching(&red, &blue);
        assert_eq!(m, vec![(0, 0)]);
    }

    #[test]
    #[should_panic(expected = "equal-sized color classes")]
    fn mismatched_sizes_panic() {
        let _ = min_length_matching(&[(0.0, 0.0)], &[(0.0, 0.0), (1.0, 1.0)]);
    }

    #[test]
    fn two_vs_two_picks_uncrossed() {
        // Reds along y = 0, blues along y = 1, but blue indices are
        // swapped in input order so the "identity" matching would cross.
        // The minimum-length matching must straighten them out.
        let red = vec![(0.0, 0.0), (1.0, 0.0)];
        let blue = vec![(1.0, 1.0), (0.0, 1.0)];
        let m = min_length_matching(&red, &blue);
        // Sorted by red index already.
        assert_eq!(m, vec![(0, 1), (1, 0)]);
        assert_no_crossings(&red, &blue, &m);
    }

    #[test]
    fn three_vs_three_min_length() {
        let red = vec![(0.0, 0.0), (2.0, 0.0), (4.0, 0.0)];
        let blue = vec![(4.0, 1.0), (0.0, 1.0), (2.0, 1.0)];
        let m = min_length_matching(&red, &blue);
        // The vertically-stacked pairing has total length 3, the unique
        // minimum among the 6 permutations.
        let len = matching_length(&red, &blue, &m);
        assert!((len - 3.0).abs() < 1e-9, "expected length 3, got {len}");
        assert_no_crossings(&red, &blue, &m);
    }

    #[test]
    fn five_points_no_crossings() {
        let red = vec![(0.0, 0.0), (5.0, 1.0), (2.0, 4.0), (-1.0, 3.0), (3.0, -2.0)];
        let blue = vec![(1.0, 5.0), (4.0, -1.0), (-2.0, 1.0), (6.0, 3.0), (0.0, 2.0)];
        let m = min_length_matching(&red, &blue);
        assert_eq!(m.len(), 5);
        // Every red index appears exactly once in order.
        for (k, &(i, _)) in m.iter().enumerate() {
            assert_eq!(i, k);
        }
        // Every blue index appears exactly once.
        let mut seen = [false; 5];
        for &(_, j) in &m {
            assert!(!seen[j]);
            seen[j] = true;
        }
        // And — the whole point — no segments cross.
        assert_no_crossings(&red, &blue, &m);
    }

    /// Independent recursive brute-force minimum-length search.
    fn brute_force_min(
        v: &mut Vec<usize>,
        k: usize,
        red: &[(f64, f64)],
        blue: &[(f64, f64)],
        best: &mut f64,
    ) {
        if k == v.len() {
            let mut s = 0.0;
            for (i, &j) in v.iter().enumerate() {
                s += sq_dist(red[i], blue[j]).sqrt();
            }
            if s < *best {
                *best = s;
            }
            return;
        }
        for i in k..v.len() {
            v.swap(k, i);
            brute_force_min(v, k + 1, red, blue, best);
            v.swap(k, i);
        }
    }

    #[test]
    fn six_points_matches_brute_force() {
        let red = vec![
            (0.0, 0.0),
            (1.0, 3.0),
            (4.0, 2.0),
            (3.0, -1.0),
            (-2.0, 1.0),
            (5.0, 4.0),
        ];
        let blue = vec![
            (2.0, 5.0),
            (-1.0, -2.0),
            (6.0, 0.0),
            (3.0, 3.0),
            (0.0, 4.0),
            (4.0, -3.0),
        ];
        let m = min_length_matching(&red, &blue);
        let got = matching_length(&red, &blue, &m);

        let mut v: Vec<usize> = (0..red.len()).collect();
        let mut best = f64::INFINITY;
        brute_force_min(&mut v, 0, &red, &blue, &mut best);

        assert!((got - best).abs() < 1e-9, "got {got}, brute-force {best}");
        assert_no_crossings(&red, &blue, &m);
    }
}
