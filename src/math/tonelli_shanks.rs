//! Tonelli-Shanks algorithm: solves `x^2 ≡ n (mod p)` for an odd prime `p`.
//!
//! Returns one square root; the other is `p - x`. If `n` is a quadratic
//! non-residue mod `p`, returns `None`.
//!
//! # Algorithm
//! Factor `p - 1 = q * 2^s` with `q` odd.
//! - If `s == 1` (i.e. `p ≡ 3 mod 4`): return `n^((p+1)/4) mod p`.
//! - Otherwise: use the Tonelli–Shanks loop to iteratively refine a root via
//!   a Cipolla-style square-root lifting over the 2-Sylow subgroup.
//!
//! Reference: Tonelli (1891), Shanks (1973).
//!
//! # Complexity
//! Time: O(log² p) expected. Space: O(1).
//!
//! # Preconditions
//! `p` must be an odd prime. Behaviour is unspecified for composite moduli.

use crate::math::modular_exponentiation::mod_pow;

/// Returns a square root `x` of `n` modulo prime `p`, or `None` if `n` is
/// a non-residue. Requires `p` odd prime.
pub fn tonelli_shanks(n: u64, p: u64) -> Option<u64> {
    // Normalise n into [0, p).
    let n = n % p;

    // 0 is always a residue.
    if n == 0 {
        return Some(0);
    }

    // Euler's criterion: n^((p-1)/2) mod p must be 1, not p-1.
    if mod_pow(n, (p - 1) / 2, p) != 1 {
        return None;
    }

    // Factor p-1 = q * 2^s with q odd.
    let mut q = p - 1;
    let mut s = 0_u32;
    while q.is_multiple_of(2) {
        q /= 2;
        s += 1;
    }

    // Special case: p ≡ 3 (mod 4) — direct formula.
    if s == 1 {
        return Some(mod_pow(n, (p + 1) / 4, p));
    }

    // Find a quadratic non-residue z mod p via trial.
    let z = (2_u64..p)
        .find(|&z| mod_pow(z, (p - 1) / 2, p) == p - 1)
        .expect("a non-residue always exists for prime p");

    // Initialise the Tonelli-Shanks loop.
    let mut m = s;
    let mut c = mod_pow(z, q, p);
    let mut t = mod_pow(n, q, p);
    let mut r = mod_pow(n, q.div_ceil(2), p);

    loop {
        if t == 1 {
            return Some(r);
        }

        // Find the least i, 1 ≤ i < m, such that t^(2^i) ≡ 1 (mod p).
        let mut i = 1_u32;
        let mut tmp = mulmod(t, t, p);
        while tmp != 1 {
            tmp = mulmod(tmp, tmp, p);
            i += 1;
        }

        // b = c^(2^(m-i-1)) mod p
        let exp = 1_u64 << (m - i - 1);
        let b = mod_pow(c, exp, p);
        let b2 = mulmod(b, b, p);

        m = i;
        c = b2;
        t = mulmod(t, b2, p);
        r = mulmod(r, b, p);
    }
}

/// Computes `(a * b) mod m` using `u128` to prevent overflow.
#[inline]
fn mulmod(a: u64, b: u64, m: u64) -> u64 {
    ((u128::from(a) * u128::from(b)) % u128::from(m)) as u64
}

#[cfg(test)]
mod tests {
    use super::tonelli_shanks;
    use crate::math::modular_exponentiation::mod_pow;

    // ── unit tests ──────────────────────────────────────────────────────────

    #[test]
    fn n_zero_is_always_residue() {
        assert_eq!(tonelli_shanks(0, 7), Some(0));
        assert_eq!(tonelli_shanks(0, 13), Some(0));
    }

    #[test]
    fn n_one_p_seven() {
        // 1^2 ≡ 1 (mod 7)
        assert_eq!(tonelli_shanks(1, 7), Some(1));
    }

    #[test]
    fn known_residue_p13() {
        // 6^2 = 36 ≡ 10 (mod 13);  also 7^2 = 49 ≡ 10 (mod 13)
        let r = tonelli_shanks(10, 13).expect("10 is a QR mod 13");
        assert!(r == 6 || r == 7, "expected 6 or 7, got {r}");
    }

    #[test]
    fn non_residue_p7() {
        // 5 is a non-residue mod 7 (Euler: 5^3 = 125 ≡ 6 ≡ -1)
        assert_eq!(tonelli_shanks(5, 7), None);
    }

    // p ≡ 3 (mod 4): direct formula path (s == 1)
    #[test]
    fn prime_3_mod_4() {
        // p = 7 ≡ 3 (mod 4); 2^2 = 4 ≡ 4 (mod 7) — 4 is a QR
        let r = tonelli_shanks(4, 7).expect("4 is a QR mod 7");
        assert_eq!((r * r) % 7, 4);
    }

    // p ≡ 1 (mod 4): general Tonelli-Shanks loop path
    #[test]
    fn prime_1_mod_4() {
        // p = 17 ≡ 1 (mod 4)
        // 2^2 = 4, 4 is a QR mod 17
        let r = tonelli_shanks(4, 17).expect("4 is a QR mod 17");
        assert_eq!((r * r) % 17, 4);
    }

    #[test]
    fn prime_1_mod_4_larger() {
        // p = 41 ≡ 1 (mod 4)
        let p = 41_u64;
        for n in 1..p {
            let result = tonelli_shanks(n, p);
            if let Some(x) = result {
                assert_eq!((x * x) % p, n % p, "root check failed for n={n}");
            } else {
                // Verify it really is a non-residue via Euler's criterion.
                assert_ne!(
                    mod_pow(n, (p - 1) / 2, p),
                    1,
                    "Euler criterion mismatch for n={n}"
                );
            }
        }
    }

    // ── property-based tests ─────────────────────────────────────────────────

    #[cfg(test)]
    mod property {
        use super::*;
        use quickcheck_macros::quickcheck;

        /// A hand-picked list of small odd primes spanning both p ≡ 1 (mod 4)
        /// and p ≡ 3 (mod 4) cases.
        const PRIMES: &[u64] = &[
            3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73,
        ];

        /// For a random `n` in `[0, p)` chosen by `QuickCheck` via the index,
        /// verify the invariant: if `Some(x)` then `x*x ≡ n`, else Euler says
        /// it is a non-residue.
        #[quickcheck]
        fn root_is_correct_or_none(prime_idx: u8, n_raw: u64) -> bool {
            let p = PRIMES[prime_idx as usize % PRIMES.len()];
            let n = n_raw % p;
            tonelli_shanks(n, p)
                .map_or_else(|| mod_pow(n, (p - 1) / 2, p) != 1, |x| (x * x) % p == n)
        }
    }
}
