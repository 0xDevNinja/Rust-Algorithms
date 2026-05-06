//! Pick's theorem for simple polygons with integer-coordinate vertices.
//!
//! For a simple polygon whose vertices lie on the integer lattice, Pick's
//! theorem states
//!
//! ```text
//!     A = I + B / 2 - 1
//! ```
//!
//! where `A` is the polygon's area, `I` the number of strictly interior
//! lattice points and `B` the number of lattice points on the boundary. The
//! boundary count is obtained by summing `gcd(|dx|, |dy|)` over each polygon
//! edge (this counts each lattice point on the segment, including its
//! endpoints, exactly once when the edges are traversed in order). The area
//! is computed via the integer Shoelace formula and is always a multiple of
//! `1/2`, so the resulting `I = A - B/2 + 1` is exact integer arithmetic.
//!
//! Time: `O(n)` over the vertex count.
//! Space: `O(1)`.
//!
//! Precondition: the polygon must be simple (no self-intersections); Pick's
//! theorem does not hold for self-intersecting curves.

/// Returns `(twice_signed_area, boundary_lattice_points)` for a simple
/// integer-coordinate polygon. The signed area is doubled so the value is an
/// exact integer; positive when vertices are listed counter-clockwise and
/// negative when clockwise.
///
/// - Time: `O(n)`.
/// - Space: `O(1)`.
pub fn polygon_stats(vertices: &[(i64, i64)]) -> (i64, u64) {
    let n = vertices.len();
    if n < 3 {
        return (0, 0);
    }
    let mut twice_area: i64 = 0;
    let mut boundary: u64 = 0;
    for i in 0..n {
        let (xi, yi) = vertices[i];
        let (xj, yj) = vertices[(i + 1) % n];
        twice_area += xi * yj - xj * yi;
        boundary += gcd((xj - xi).unsigned_abs(), (yj - yi).unsigned_abs());
    }
    (twice_area, boundary)
}

/// Number of lattice points strictly inside a simple integer-coordinate
/// polygon, computed via Pick's theorem `I = A - B/2 + 1`.
///
/// Returns `0` for degenerate polygons (fewer than 3 vertices).
///
/// - Time: `O(n)`.
/// - Space: `O(1)`.
pub fn interior_lattice_points(vertices: &[(i64, i64)]) -> u64 {
    if vertices.len() < 3 {
        return 0;
    }
    let (twice_area, boundary) = polygon_stats(vertices);
    // 2A = 2I + B - 2  =>  I = (|2A| + 2 - B) / 2
    let two_a = twice_area.unsigned_abs();
    // Boundary is always even when the polygon is closed and simple? No —
    // 2A - B + 2 is always even because 2A and B have the same parity for a
    // closed lattice polygon, so the division below is exact.
    (two_a + 2 - boundary) / 2
}

/// Number of lattice points on the boundary of a simple integer-coordinate
/// polygon (counting each vertex and each segment-interior lattice point
/// exactly once). Returns `0` for degenerate polygons.
///
/// - Time: `O(n)`.
/// - Space: `O(1)`.
pub fn boundary_lattice_points(vertices: &[(i64, i64)]) -> u64 {
    if vertices.len() < 3 {
        return 0;
    }
    polygon_stats(vertices).1
}

const fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

#[cfg(test)]
mod tests {
    use super::{boundary_lattice_points, interior_lattice_points, polygon_stats};

    #[test]
    fn empty_or_degenerate_returns_zeros() {
        assert_eq!(polygon_stats(&[]), (0, 0));
        assert_eq!(interior_lattice_points(&[]), 0);
        assert_eq!(boundary_lattice_points(&[(0, 0), (1, 1)]), 0);
    }

    #[test]
    fn unit_square() {
        // 1x1 square: A = 1, B = 4, I = 0
        let v = [(0_i64, 0), (1, 0), (1, 1), (0, 1)];
        assert_eq!(boundary_lattice_points(&v), 4);
        assert_eq!(interior_lattice_points(&v), 0);
    }

    #[test]
    fn three_by_three_square() {
        // Square with side 3: A=9, B=12, I=4
        let v = [(0_i64, 0), (3, 0), (3, 3), (0, 3)];
        assert_eq!(boundary_lattice_points(&v), 12);
        assert_eq!(interior_lattice_points(&v), 4);
    }

    #[test]
    fn right_triangle() {
        // Triangle (0,0)-(4,0)-(0,3): A=6, B=gcd(4,0)+gcd(4,3)+gcd(0,3) = 4+1+3 = 8, I = 6 - 4 + 1 = 3
        let v = [(0_i64, 0), (4, 0), (0, 3)];
        assert_eq!(boundary_lattice_points(&v), 8);
        assert_eq!(interior_lattice_points(&v), 3);
    }

    #[test]
    fn cw_orientation_gives_same_counts() {
        let ccw = [(0_i64, 0), (3, 0), (3, 3), (0, 3)];
        let cw: Vec<(i64, i64)> = ccw.iter().rev().copied().collect();
        assert_eq!(interior_lattice_points(&ccw), interior_lattice_points(&cw));
        assert_eq!(boundary_lattice_points(&ccw), boundary_lattice_points(&cw));
    }

    #[test]
    fn lattice_point_on_edge_counted_once() {
        // Triangle (0,0)-(2,0)-(0,2): edge (0,0)-(2,0) has 3 points, (2,0)-(0,2) has gcd(2,2)=2 segments => 3 lattice points
        // Total B counting endpoints once: 2 + 2 + 2 = 6
        let v = [(0_i64, 0), (2, 0), (0, 2)];
        assert_eq!(boundary_lattice_points(&v), 6);
        // A = 2, I = 2 - 3 + 1 = 0
        assert_eq!(interior_lattice_points(&v), 0);
    }

    #[test]
    fn translation_invariance() {
        let v = [(0_i64, 0), (5, 0), (5, 4), (0, 4)];
        let shifted: Vec<(i64, i64)> = v.iter().map(|&(x, y)| (x + 100, y - 50)).collect();
        assert_eq!(
            interior_lattice_points(&v),
            interior_lattice_points(&shifted)
        );
    }

    #[test]
    fn brute_force_small_rectangles() {
        for w in 1..=5_i64 {
            for h in 1..=5_i64 {
                let v = [(0, 0), (w, 0), (w, h), (0, h)];
                let i_pick = interior_lattice_points(&v);
                let mut i_brute = 0_u64;
                for x in 1..w {
                    for y in 1..h {
                        let _ = (x, y);
                        i_brute += 1;
                    }
                }
                assert_eq!(i_pick, i_brute, "rect {w}x{h}");
            }
        }
    }
}
