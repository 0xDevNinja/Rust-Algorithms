//! Binomial coefficient `C(n, r) mod p` with O(N) preprocessing and O(1) queries.
//!
//! # Method
//! Precomputes `fact[i] = i! mod p` for `i in 0..=n_max` and the corresponding
//! modular inverses `inv_fact[i] = (i!)^{-1} mod p`. With these tables,
//!
//! ```text
//! C(n, r) mod p = fact[n] * inv_fact[r] * inv_fact[n - r] mod p.
//! ```
//!
//! Modular inverses are obtained via **Fermat's little theorem**, which states
//! that for a prime `p` and any `a` not divisible by `p`,
//! `a^{p-1} ≡ 1 (mod p)`, hence `a^{-1} ≡ a^{p-2} (mod p)`.
//!
//! # Prime modulus required
//! The modulus `p` **must** be a prime that does not divide any of
//! `1, 2, …, n_max`; equivalently, `p > n_max`. Otherwise some factorial
//! `i!` is `0 mod p` and has no modular inverse, and Fermat's identity does
//! not apply. The typical competitive-programming choice is `1_000_000_007`.
//!
//! # Complexity
//! - Preprocessing: `O(n_max)` multiplications + a single `O(log p)` exponentiation.
//! - Query: `O(1)`.
//! - Space: `O(n_max)`.
//!
//! # Out-of-range queries
//! Queries with `n > n_max` (or `r > n_max`) return `0`. A genuine
//! `C(n, r)` is never zero for `r ≤ n`, so the `0` sentinel is unambiguous as
//! long as the caller respects the precomputed range. Queries with `r > n`
//! also return `0`, matching the combinatorial definition.

use crate::math::modular_exponentiation::mod_pow;

/// Precomputed factorial / inverse-factorial tables for fast `C(n, r) mod p`.
///
/// Construct once with [`ModularBinomial::new`] for the desired `n_max` and
/// prime `p`, then call [`ModularBinomial::ncr`] for O(1) queries.
pub struct ModularBinomial {
    p: u64,
    fact: Vec<u64>,
    inv_fact: Vec<u64>,
}

impl ModularBinomial {
    /// Builds tables of `i! mod p` and `(i!)^{-1} mod p` for `i in 0..=n_max`.
    ///
    /// # Requirements
    /// `p` must be a prime strictly greater than `n_max`. The constructor does
    /// **not** verify primality; passing a composite or too-small prime
    /// produces undefined (but non-panicking) results because Fermat's little
    /// theorem no longer yields valid inverses.
    ///
    /// # Complexity
    /// `O(n_max)` multiplications plus one `O(log p)` modular exponentiation
    /// for the trailing inverse factorial; the rest are derived in reverse via
    /// `inv_fact[i] = inv_fact[i + 1] * (i + 1) mod p`.
    #[must_use]
    pub fn new(n_max: usize, p: u64) -> Self {
        assert!(p > 1, "modulus must be a prime greater than 1");
        let len = n_max + 1;
        let mut fact = vec![1_u64; len];
        for i in 1..len {
            fact[i] = ((u128::from(fact[i - 1]) * i as u128) % u128::from(p)) as u64;
        }
        let mut inv_fact = vec![1_u64; len];
        // Fermat: a^{-1} = a^{p-2} mod p when p is prime and gcd(a, p) = 1.
        inv_fact[len - 1] = mod_pow(fact[len - 1], p - 2, p);
        for i in (0..len - 1).rev() {
            inv_fact[i] = ((u128::from(inv_fact[i + 1]) * (i + 1) as u128) % u128::from(p)) as u64;
        }
        Self { p, fact, inv_fact }
    }

    /// Returns `C(n, r) mod p`.
    ///
    /// # Returns
    /// - `1` for `C(n, 0)` and `C(n, n)`.
    /// - `0` if `r > n` (combinatorial convention).
    /// - `0` if either `n` or `r` exceeds the `n_max` supplied to
    ///   [`ModularBinomial::new`]. The caller must size `n_max` to cover the
    ///   range of expected queries — out-of-range queries are silently zeroed
    ///   rather than panicking.
    #[must_use]
    pub fn ncr(&self, n: usize, r: usize) -> u64 {
        if r > n || n >= self.fact.len() || r >= self.fact.len() {
            return 0;
        }
        let m = u128::from(self.p);
        let a = u128::from(self.fact[n]);
        let b = u128::from(self.inv_fact[r]);
        let c = u128::from(self.inv_fact[n - r]);
        ((a * b % m) * c % m) as u64
    }
}

#[cfg(test)]
mod tests {
    use super::ModularBinomial;
    use quickcheck_macros::quickcheck;

    const P: u64 = 1_000_000_007;

    #[test]
    fn c_0_0_is_one() {
        let b = ModularBinomial::new(0, P);
        assert_eq!(b.ncr(0, 0), 1);
    }

    #[test]
    fn c_5_2_is_ten() {
        let b = ModularBinomial::new(10, P);
        assert_eq!(b.ncr(5, 2), 10);
    }

    #[test]
    fn c_10_5_is_252() {
        let b = ModularBinomial::new(10, P);
        assert_eq!(b.ncr(10, 5), 252);
    }

    #[test]
    fn c_n_0_and_c_n_n_are_one() {
        let b = ModularBinomial::new(50, P);
        for n in 0..=50 {
            assert_eq!(b.ncr(n, 0), 1, "C({n}, 0)");
            assert_eq!(b.ncr(n, n), 1, "C({n}, {n})");
        }
    }

    #[test]
    fn r_greater_than_n_is_zero() {
        let b = ModularBinomial::new(10, P);
        assert_eq!(b.ncr(5, 6), 0);
        assert_eq!(b.ncr(0, 1), 0);
    }

    #[test]
    fn out_of_precomputed_range_is_zero() {
        let b = ModularBinomial::new(5, P);
        // n exceeds n_max
        assert_eq!(b.ncr(6, 2), 0);
        // r exceeds n_max
        assert_eq!(b.ncr(5, 100), 0);
    }

    #[test]
    fn pascals_triangle_row_5() {
        // 1, 5, 10, 10, 5, 1
        let b = ModularBinomial::new(5, P);
        let row: Vec<u64> = (0..=5).map(|r| b.ncr(5, r)).collect();
        assert_eq!(row, vec![1, 5, 10, 10, 5, 1]);
    }

    #[test]
    fn larger_value_under_prime() {
        // C(20, 10) = 184_756, well below 1e9+7 so the modulus doesn't bite.
        let b = ModularBinomial::new(20, P);
        assert_eq!(b.ncr(20, 10), 184_756);
    }

    /// Brute-force binomial using `u128` factorials. Returns `0` when `r > n`
    /// or when intermediate factorials would overflow.
    fn brute_force_binomial(n: u64, r: u64) -> u128 {
        if r > n {
            return 0;
        }
        let mut num: u128 = 1;
        let mut den: u128 = 1;
        for i in 0..r {
            num *= u128::from(n - i);
            den *= u128::from(i + 1);
        }
        num / den
    }

    /// Symmetry: `C(n, r) == C(n, n - r) (mod p)` for all `n, r in [0, n_max]`.
    #[allow(clippy::needless_pass_by_value)]
    #[quickcheck]
    fn prop_symmetry(n: u8, r: u8) -> bool {
        let n_max = 50_usize;
        let n = (n as usize) % (n_max + 1);
        let r = (r as usize) % (n_max + 1);
        let b = ModularBinomial::new(n_max, P);
        if r > n {
            // Both sides should be 0 by the r > n rule.
            return b.ncr(n, r) == 0 && b.ncr(n, n.saturating_sub(r)) == b.ncr(n, n - n.min(r));
        }
        b.ncr(n, r) == b.ncr(n, n - r)
    }

    /// For small `n, r ≤ 20`, the modular value matches the exact binomial
    /// coefficient (which fits comfortably below `1e9+7`).
    #[allow(clippy::needless_pass_by_value)]
    #[quickcheck]
    fn prop_matches_brute_force_small(n: u8, r: u8) -> bool {
        let n = (n as u64) % 21;
        let r = (r as u64) % 21;
        let b = ModularBinomial::new(20, P);
        let expected = brute_force_binomial(n, r) % u128::from(P);
        u128::from(b.ncr(n as usize, r as usize)) == expected
    }
}
