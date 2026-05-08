//! Burnside's lemma (also known as the Cauchy-Frobenius lemma) for
//! counting orbits of a finite group action.
//!
//! For a finite group `G` acting on a set `X`, the number of distinct
//! orbits equals the average size of the fixed-point sets:
//!
//! ```text
//!     |X / G| = (1 / |G|) * Σ_{g ∈ G} |Fix(g)|
//! ```
//!
//! This module provides a generic [`burnside_count`] driver plus two
//! ready-made applications: counting cyclic [`count_necklaces`] and
//! dihedral [`count_bracelets`] colourings of `n` beads in `k` colours.
//!
//! ## Complexity
//!
//! - [`burnside_count`]: `O(|G| * F)` where `F` is the cost of the
//!   fixed-point oracle.
//! - [`count_necklaces`]: `O(σ_0(n) * log n)` divisor enumeration plus
//!   modular powers; effectively near-linear in `n`.
//! - [`count_bracelets`]: same as necklaces plus `O(1)` reflection
//!   terms.

/// Euler's totient `φ(n)` computed by trial division of `n`.
const fn totient(mut n: u64) -> u64 {
    if n == 0 {
        return 0;
    }
    let mut result = n;
    let mut p: u64 = 2;
    while p * p <= n {
        if n.is_multiple_of(p) {
            while n.is_multiple_of(p) {
                n /= p;
            }
            result -= result / p;
        }
        p += 1;
    }
    if n > 1 {
        result -= result / n;
    }
    result
}

/// Integer power `base^exp`.
const fn pow_u64(mut base: u64, mut exp: u64) -> u64 {
    let mut acc: u64 = 1;
    while exp > 0 {
        if exp & 1 == 1 {
            acc = acc.wrapping_mul(base);
        }
        exp >>= 1;
        if exp > 0 {
            base = base.wrapping_mul(base);
        }
    }
    acc
}

/// Counts the orbits of a finite group action via Burnside's lemma.
///
/// `group` enumerates every element of `G`, and `fix(g)` returns the
/// number of points of the underlying set fixed by `g`. The result is
/// `(1 / |G|) * Σ_{g} fix(g)`.
///
/// # Preconditions
///
/// - `group` must be non-empty.
/// - The total `Σ fix(g)` must be divisible by `|group|` — Burnside's
///   lemma guarantees this whenever `fix` correctly reports fixed-point
///   counts of a genuine group action.
pub fn burnside_count<G, F>(group: &[G], fix: F) -> u64
where
    F: Fn(&G) -> u64,
{
    assert!(!group.is_empty(), "burnside_count: group must be non-empty");
    let total: u64 = group.iter().map(&fix).sum();
    let order = group.len() as u64;
    debug_assert_eq!(
        total % order,
        0,
        "burnside_count: Σ |Fix(g)| must be divisible by |G|"
    );
    total / order
}

/// Counts distinct `n`-bead `k`-colour necklaces under the cyclic
/// rotation group `Z_n`.
///
/// Uses the closed form derived from Burnside's lemma:
///
/// ```text
///     N(n, k) = (1 / n) * Σ_{d | n} φ(d) * k^(n / d)
/// ```
///
/// Returns `0` when `n == 0` (no beads, no necklaces).
pub const fn count_necklaces(n: u32, k: u32) -> u64 {
    if n == 0 {
        return 0;
    }
    let n64 = n as u64;
    let k64 = k as u64;
    let mut total: u64 = 0;
    let mut d: u64 = 1;
    while d * d <= n64 {
        if n64.is_multiple_of(d) {
            total += totient(d) * pow_u64(k64, n64 / d);
            let other = n64 / d;
            if other != d {
                total += totient(other) * pow_u64(k64, n64 / other);
            }
        }
        d += 1;
    }
    total / n64
}

/// Counts distinct `n`-bead `k`-colour bracelets under the dihedral
/// group `D_n` (rotations together with reflections).
///
/// Combines the rotation contribution from [`count_necklaces`] with
/// the `n` reflection terms. With `N = count_necklaces(n, k)`:
///
/// ```text
///     B(n, k) = (1 / 2) * (N + reflection_contribution(n, k) / n)
/// ```
///
/// where the reflection contribution depends on the parity of `n`:
/// odd `n` contributes `n * k^((n + 1) / 2)`; even `n` contributes
/// `(n / 2) * (k^(n / 2) + k^(n / 2 + 1))`.
///
/// Returns `0` when `n == 0`.
pub const fn count_bracelets(n: u32, k: u32) -> u64 {
    if n == 0 {
        return 0;
    }
    let n64 = n as u64;
    let k64 = k as u64;
    let necklaces = count_necklaces(n, k);
    let reflection_avg = if n64.is_multiple_of(2) {
        // n even: half the reflections pass through two beads
        // (k^(n/2 + 1) fixed), the other half through two edge
        // midpoints (k^(n/2) fixed). Their average is
        // (k^(n/2) + k^(n/2 + 1)) / 2.
        u64::midpoint(pow_u64(k64, n64 / 2), pow_u64(k64, n64 / 2 + 1))
    } else {
        // n odd: every reflection has axis through one bead and the
        // opposite edge midpoint, fixing k^((n + 1) / 2) colourings.
        pow_u64(k64, n64.div_ceil(2))
    };
    u64::midpoint(necklaces, reflection_avg)
}

#[cfg(test)]
mod tests {
    use super::{burnside_count, count_bracelets, count_necklaces, pow_u64, totient};

    #[test]
    fn totient_small_values() {
        // Reference values from OEIS A000010.
        let expected = [0, 1, 1, 2, 2, 4, 2, 6, 4, 6, 4];
        for (n, &phi) in expected.iter().enumerate() {
            assert_eq!(totient(n as u64), phi, "phi({n})");
        }
    }

    #[test]
    fn pow_u64_basics() {
        assert_eq!(pow_u64(2, 0), 1);
        assert_eq!(pow_u64(2, 10), 1024);
        assert_eq!(pow_u64(3, 5), 243);
    }

    #[test]
    fn necklaces_three_beads_two_colours() {
        // Canonical Burnside example: 4 distinct necklaces.
        assert_eq!(count_necklaces(3, 2), 4);
    }

    #[test]
    fn necklaces_four_beads_two_colours() {
        assert_eq!(count_necklaces(4, 2), 6);
    }

    #[test]
    fn necklaces_six_beads_three_colours() {
        // Standard textbook value for Z_6 acting on 3-colourings.
        assert_eq!(count_necklaces(6, 3), 130);
    }

    #[test]
    fn necklaces_degenerate_inputs() {
        assert_eq!(count_necklaces(0, 5), 0);
        // One bead, k colours → k necklaces.
        assert_eq!(count_necklaces(1, 5), 5);
        // Single colour, any number of beads → 1 necklace.
        assert_eq!(count_necklaces(7, 1), 1);
    }

    #[test]
    fn bracelets_four_beads_two_colours() {
        assert_eq!(count_bracelets(4, 2), 6);
    }

    #[test]
    fn bracelets_six_beads_two_colours() {
        assert_eq!(count_bracelets(6, 2), 13);
    }

    #[test]
    fn bracelets_degenerate_inputs() {
        assert_eq!(count_bracelets(0, 5), 0);
        assert_eq!(count_bracelets(1, 5), 5);
        assert_eq!(count_bracelets(7, 1), 1);
    }

    #[test]
    fn burnside_count_z4_on_square_corners() {
        // Z_4 = {e, r, r^2, r^3} acts on 2-colourings of the corners
        // of a square.
        //   - e fixes all 2^4 = 16 colourings.
        //   - r and r^3 fix only the monochromatic colourings (2 each).
        //   - r^2 fixes colourings invariant under the half-turn:
        //     opposite corners must agree, so 2^2 = 4 colourings.
        // (16 + 2 + 4 + 2) / 4 = 24 / 4 = 6 distinct colourings.
        let group = [0u8, 1, 2, 3];
        let count = burnside_count(&group, |&g| match g {
            0 => 16,
            1 | 3 => 2,
            2 => 4,
            _ => unreachable!(),
        });
        assert_eq!(count, 6);
    }

    #[test]
    fn burnside_count_trivial_group() {
        // |G| = 1, fixed-point count equals |X|.
        let group = [()];
        assert_eq!(burnside_count(&group, |()| 42), 42);
    }
}
