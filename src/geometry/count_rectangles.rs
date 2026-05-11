//! Count rectangles formed by points using the diagonal-signature trick.
//!
//! Given `n` points in the plane, count the number of (unordered)
//! quadruples `{a, b, c, d}` whose four points are the corners of a
//! (possibly rotated) rectangle.
//!
//! # Algorithm
//!
//! A quadrilateral is a rectangle iff its two diagonals
//!
//! * share the same midpoint, and
//! * have equal length.
//!
//! Conversely, any two distinct point pairs `(p, q)` and `(r, s)` with
//! the same midpoint and the same length form the diagonals of a
//! rectangle whose corners are `{p, q, r, s}`. This gives an `O(n^2)`
//! algorithm:
//!
//! 1. For each unordered pair of distinct points `(p, q)` compute its
//!    *diagonal signature* — the midpoint together with the squared
//!    length of the segment.
//! 2. Group pairs by signature in a hash map.
//! 3. For a group of `k` pairs, every unordered choice of two pairs
//!    yields one rectangle, contributing `C(k, 2) = k·(k − 1)/2`.
//!
//! To stay in exact integer arithmetic the midpoint `((px + qx)/2,
//! (py + qy)/2)` is stored as `(px + qx, py + qy)` — i.e. doubled.
//! The squared length `(px − qx)^2 + (py − qy)^2` is already an integer.
//! Together `(2·mx, 2·my, len²)` is a faithful, hashable signature
//! requiring no floating point.
//!
//! Axis-aligned rectangles are a special case of the general rectangle
//! count returned here; the routine counts them all in one pass.
//!
//! # Complexity
//!
//! * Time: `O(n^2)` — dominated by enumerating all `n·(n − 1)/2` pairs.
//! * Space: `O(n^2)` worst case for the hash map of diagonal
//!   signatures.
//!
//! # Preconditions
//!
//! Input points are assumed to be **distinct**. Duplicates would create
//! degenerate "diagonals" of length zero and inflate the count; callers
//! that may have duplicates should deduplicate first.
//!
//! Coordinates use `i64`. The intermediate arithmetic computes
//! `(px − qx)^2 + (py − qy)^2`, so each coordinate must satisfy
//! `|c| ≤ 2^31` to keep the squared length within `i64`.

use std::collections::HashMap;

/// Count the number of (axis-aligned or rotated) rectangles whose four
/// corners are points in `points`.
///
/// Two pairs of points form the diagonals of a rectangle iff they share
/// the same midpoint and the same length, so this routine groups
/// point-pairs by the integer signature `(2·mx, 2·my, len²)` and sums
/// `C(k, 2)` over each group. See the module documentation for the
/// derivation and complexity (`O(n^2)` time and space).
///
/// Input points are assumed to be distinct; duplicates are not
/// deduplicated and will inflate the count.
#[must_use]
pub fn count_general_rectangles(points: &[(i64, i64)]) -> u64 {
    let n = points.len();
    if n < 4 {
        return 0;
    }

    // Signature -> number of point-pairs with that diagonal.
    let mut groups: HashMap<(i64, i64, i64), u64> = HashMap::new();

    for i in 0..n {
        let (xi, yi) = points[i];
        for j in (i + 1)..n {
            let (xj, yj) = points[j];
            let mx2 = xi + xj;
            let my2 = yi + yj;
            let dx = xi - xj;
            let dy = yi - yj;
            let len_sq = dx * dx + dy * dy;
            *groups.entry((mx2, my2, len_sq)).or_insert(0) += 1;
        }
    }

    // For each group of k diagonals, choose 2 to pair into a rectangle.
    groups.values().map(|&k| k * k.saturating_sub(1) / 2).sum()
}

#[cfg(test)]
mod tests {
    use super::count_general_rectangles;

    #[test]
    fn empty_returns_zero() {
        let v: Vec<(i64, i64)> = Vec::new();
        assert_eq!(count_general_rectangles(&v), 0);
    }

    #[test]
    fn three_points_returns_zero() {
        let v = vec![(0, 0), (1, 0), (0, 1)];
        assert_eq!(count_general_rectangles(&v), 0);
    }

    #[test]
    fn unit_square_is_one_rectangle() {
        // The four corners of the unit square form exactly one rectangle.
        let v = vec![(0, 0), (1, 0), (1, 1), (0, 1)];
        assert_eq!(count_general_rectangles(&v), 1);
    }

    #[test]
    fn rotated_square_is_one_rectangle() {
        // Square rotated 45° with corners (1,0), (0,1), (-1,0), (0,-1).
        // Diagonals are (1,0)-(-1,0) and (0,1)-(0,-1): midpoint (0,0),
        // length² = 4 for both. → 1 rectangle.
        let v = vec![(1, 0), (0, 1), (-1, 0), (0, -1)];
        assert_eq!(count_general_rectangles(&v), 1);
    }

    #[test]
    fn non_rectangle_quadruple_returns_zero() {
        // A trapezoid: (0,0), (4,0), (3,2), (1,2). The diagonals
        // (0,0)-(3,2) and (4,0)-(1,2) share midpoint (5/2, 1) but their
        // squared lengths (9+4=13 vs 9+4=13) … here they're equal, so
        // pick a definitely-non-rectangle: an isoceles triangle plus a
        // stray point inside.
        let v = vec![(0, 0), (4, 0), (2, 3), (2, 1)];
        assert_eq!(count_general_rectangles(&v), 0);
    }

    #[test]
    fn two_axis_aligned_rectangles_sharing_an_edge() {
        // Points form two unit squares glued along the segment x=1:
        //   (0,0),(1,0),(1,1),(0,1)  — left square
        //   (1,0),(2,0),(2,1),(1,1)  — right square shares (1,0),(1,1)
        // Total distinct points: 6. As well as the two unit squares
        // there's a 2×1 rectangle (0,0),(2,0),(2,1),(0,1). So 3 total.
        let v = vec![(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)];
        assert_eq!(count_general_rectangles(&v), 3);
    }

    #[test]
    fn collinear_points_yield_zero() {
        let v: Vec<(i64, i64)> = (0..10).map(|i| (i, 0)).collect();
        assert_eq!(count_general_rectangles(&v), 0);
    }

    #[test]
    fn three_by_three_grid_count() {
        // 3×3 lattice: every pair of distinct rows and distinct columns
        // selects one axis-aligned rectangle: C(3,2)·C(3,2) = 9. Plus
        // the rotated square with corners (1,0),(2,1),(1,2),(0,1). Total
        // 10 rectangles.
        let mut v = Vec::new();
        for x in 0..3 {
            for y in 0..3 {
                v.push((x, y));
            }
        }
        assert_eq!(count_general_rectangles(&v), 10);
    }

    #[test]
    fn single_rotated_rectangle() {
        // Rectangle with corners (0,0), (2,1), (1,3), (-1,2). Midpoint
        // of (0,0)-(1,3) is (1/2, 3/2); midpoint of (2,1)-(-1,2) is
        // (1/2, 3/2). Squared lengths: 1+9=10 and 9+1=10. → 1 rectangle.
        let v = vec![(0, 0), (2, 1), (1, 3), (-1, 2)];
        assert_eq!(count_general_rectangles(&v), 1);
    }

    #[test]
    fn point_pair_with_shared_midpoint_but_different_length() {
        // Four points whose two pairings share a midpoint but with
        // unequal diagonal lengths — not a rectangle.
        // (-1,0),(1,0) and (0,-2),(0,2): midpoint (0,0); lengths² 4 vs 16.
        let v = vec![(-1, 0), (1, 0), (0, -2), (0, 2)];
        assert_eq!(count_general_rectangles(&v), 0);
    }
}
