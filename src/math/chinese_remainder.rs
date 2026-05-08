//! Chinese Remainder Theorem solver for arbitrary (non-coprime) moduli.
//!
//! Given congruences `x ≡ residues[i] (mod moduli[i])`, returns the unique
//! solution `(x, M)` with `x` in `[0, M)` and `M = lcm(moduli)`, or `None`
//! if the system is inconsistent.
//!
//! Implementation merges congruences pairwise. To merge
//! `x ≡ r1 (mod m1)` and `x ≡ r2 (mod m2)`:
//! let `(g, p, _) = ext_gcd(m1, m2)`. The merged system is solvable iff
//! `g | (r2 - r1)`. When solvable, `lcm = m1 / g * m2` and a solution is
//! `x = r1 + m1 * ((r2 - r1) / g) * p (mod lcm)`.
//!
//! Each pairwise merge runs in `O(log min(m1, m2))` for `ext_gcd`, so the
//! total complexity over `k` congruences is `O(k · log M)`.
//!
//! # Examples
//!
//! ```
//! use rust_algorithms::math::chinese_remainder::crt;
//!
//! // x ≡ 2 (mod 3), x ≡ 3 (mod 5), x ≡ 2 (mod 7)  →  x = 23, M = 105.
//! assert_eq!(crt(&[2, 3, 2], &[3, 5, 7]), Some((23, 105)));
//!
//! // Inconsistent: x ≡ 1 (mod 4) and x ≡ 2 (mod 6) cannot both hold
//! // because 2 ≢ 1 (mod gcd(4, 6) = 2).
//! assert_eq!(crt(&[1, 2], &[4, 6]), None);
//! ```

use super::extended_euclidean::ext_gcd;

/// Solves the simultaneous congruences `x ≡ residues[i] (mod moduli[i])`.
///
/// Returns `Some((x, M))` where `M = lcm(moduli)` and `x` is the unique
/// representative in `[0, M)`, or `None` if the system is inconsistent.
///
/// `residues.len()` must equal `moduli.len()` and every modulus must be
/// positive; otherwise `None` is returned. An empty input returns
/// `Some((0, 1))` (the trivial system, satisfied by every integer, with
/// canonical representative `0` modulo `1`).
pub fn crt(residues: &[i64], moduli: &[i64]) -> Option<(i64, i64)> {
    if residues.len() != moduli.len() {
        return None;
    }
    if moduli.iter().any(|&m| m <= 0) {
        return None;
    }

    let mut x: i64 = 0;
    let mut m: i64 = 1;

    for (&r, &mi) in residues.iter().zip(moduli.iter()) {
        let ri = r.rem_euclid(mi);
        let (g, p, _) = ext_gcd(m, mi);
        let diff = ri - x;
        if diff.rem_euclid(g) != 0 {
            return None;
        }
        let lcm = m / g * mi;
        // Solve m * t ≡ diff (mod mi) with t = (diff / g) * p (mod mi/g).
        let step = mi / g;
        let t = ((diff / g).rem_euclid(step) * p.rem_euclid(step)).rem_euclid(step);
        x = (x + m * t).rem_euclid(lcm);
        m = lcm;
    }

    Some((x, m))
}

#[cfg(test)]
mod tests {
    use super::crt;
    use quickcheck_macros::quickcheck;

    #[test]
    fn empty_input_is_trivial() {
        assert_eq!(crt(&[], &[]), Some((0, 1)));
    }

    #[test]
    fn single_congruence_normalizes_residue() {
        assert_eq!(crt(&[7], &[5]), Some((2, 5)));
        assert_eq!(crt(&[-1], &[7]), Some((6, 7)));
    }

    #[test]
    fn classic_brahmagupta_example() {
        // x ≡ 2 (mod 3), x ≡ 3 (mod 5), x ≡ 2 (mod 7) → 23 (mod 105).
        assert_eq!(crt(&[2, 3, 2], &[3, 5, 7]), Some((23, 105)));
    }

    #[test]
    fn non_coprime_consistent() {
        // x ≡ 2 (mod 6), x ≡ 8 (mod 9):
        // gcd(6, 9) = 3 divides 8 - 2 = 6, so consistent.
        // lcm(6, 9) = 18; x = 8 satisfies both (8 mod 6 = 2, 8 mod 9 = 8).
        assert_eq!(crt(&[2, 8], &[6, 9]), Some((8, 18)));
    }

    #[test]
    fn non_coprime_inconsistent() {
        // gcd(4, 6) = 2 does not divide 2 - 1 = 1.
        assert_eq!(crt(&[1, 2], &[4, 6]), None);
        // x ≡ 0 (mod 4) but x ≡ 1 (mod 8) is impossible.
        assert_eq!(crt(&[0, 1], &[4, 8]), None);
    }

    #[test]
    fn matching_redundant_moduli() {
        // Same congruence twice collapses to itself.
        assert_eq!(crt(&[3, 3], &[7, 7]), Some((3, 7)));
        // Stronger constraint subsumes the weaker.
        assert_eq!(crt(&[1, 9], &[4, 16]), Some((9, 16)));
    }

    #[test]
    fn mismatched_lengths_return_none() {
        assert_eq!(crt(&[1, 2], &[3]), None);
    }

    #[test]
    fn non_positive_modulus_returns_none() {
        assert_eq!(crt(&[1], &[0]), None);
        assert_eq!(crt(&[1], &[-3]), None);
    }

    #[test]
    fn solution_is_in_canonical_range() {
        let (x, m) = crt(&[5, 7, 11], &[6, 8, 13]).unwrap();
        assert!((0..m).contains(&x));
        assert_eq!(x.rem_euclid(6), 5);
        assert_eq!(x.rem_euclid(8), 7);
        assert_eq!(x.rem_euclid(13), 11);
    }

    /// Property: for random small coprime moduli, the result satisfies every
    /// input congruence and lies in `[0, M)`.
    #[quickcheck]
    fn quickcheck_random_coprime_systems(seeds: Vec<(i64, u8)>) -> bool {
        // Pool of small pairwise-coprime moduli.
        let pool: [i64; 6] = [3, 5, 7, 11, 13, 17];
        let mut moduli: Vec<i64> = Vec::new();
        let mut residues: Vec<i64> = Vec::new();
        let mut seen = [false; 6];

        for (r, idx) in seeds.into_iter().take(pool.len()) {
            let i = (idx as usize) % pool.len();
            if seen[i] {
                continue;
            }
            seen[i] = true;
            moduli.push(pool[i]);
            residues.push(r);
        }

        let Some((x, m)) = crt(&residues, &moduli) else {
            // With pairwise-coprime positive moduli and any residues, CRT must succeed.
            return moduli.is_empty();
        };

        if !(0..m).contains(&x) {
            return false;
        }
        let expected_m: i64 = moduli.iter().product();
        if m != expected_m.max(1) {
            return false;
        }
        residues
            .iter()
            .zip(moduli.iter())
            .all(|(&r, &mi)| x.rem_euclid(mi) == r.rem_euclid(mi))
    }
}
