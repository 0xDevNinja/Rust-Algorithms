//! Toy RSA: textbook key generation, encryption, and decryption over `u64`.
//!
//! Given two distinct primes `p` and `q` and a public exponent `e` coprime
//! with `phi = (p-1)(q-1)`, [`keygen`] returns the modulus `n = p*q` together
//! with `(e, d)` where `d ≡ e^{-1} (mod phi)`. [`encrypt`] computes
//! `m^e mod n` and [`decrypt`] computes `c^d mod n`.
//!
//! - Key generation: `O(k log^2 n)` where `k` is the Miller-Rabin witness
//!   count (constant here — the witness set is fixed and deterministic for
//!   `u64`).
//! - Encrypt / decrypt: `O(log e)` and `O(log d)` modular multiplications.
//!
//! # Security warning
//!
//! **THIS IS EDUCATIONAL CODE. DO NOT USE FOR REAL CRYPTOGRAPHY.**
//!
//! - The modulus is at most 64 bits, which is trivially factorable on a
//!   laptop in milliseconds.
//! - There is no padding scheme (no PKCS#1 v1.5, no OAEP), so the raw
//!   construction is malleable and leaks structure of the plaintext.
//! - Modular arithmetic here is **not** constant-time: `mod_pow` branches
//!   on the bits of the exponent, leaking timing side channels for `d`.
//! - No blinding, no protection against fault attacks, no secure RNG.
//!
//! Real RSA needs ≥2048-bit moduli, OAEP padding, constant-time bigint
//! arithmetic, blinding, and a vetted library such as `ring` or
//! `RustCrypto/RSA`. This module exists only to demonstrate how the
//! Euler-totient and modular-inverse pieces fit together.

use crate::math::extended_euclidean::mod_inverse;
use crate::math::modular_exponentiation::mod_pow;

/// An RSA key pair sharing one modulus. Both halves are public-readable
/// because this is a teaching toy — in a real implementation, `d` would
/// live behind a private type and never leave the key holder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RsaKey {
    /// Modulus `n = p * q`.
    pub n: u64,
    /// Public exponent.
    pub e: u64,
    /// Private exponent satisfying `e * d ≡ 1 (mod phi(n))`.
    pub d: u64,
}

/// Returns `gcd(a, b)` for `u64` inputs (Euclidean algorithm).
const fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

/// Deterministic Miller-Rabin primality test for any `u64`.
///
/// Uses the witness set `{2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37}`,
/// which is known to be correct for all `n < 2^64` (Sinclair, 2011).
fn is_prime_u64(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    for &p in &[2_u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37] {
        if n == p {
            return true;
        }
        if n.is_multiple_of(p) {
            return false;
        }
    }
    // Write n - 1 = d * 2^s with d odd.
    let mut d = n - 1;
    let mut s = 0_u32;
    while d.is_multiple_of(2) {
        d /= 2;
        s += 1;
    }
    'witness: for &a in &[2_u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37] {
        let mut x = mod_pow(a, d, n);
        if x == 1 || x == n - 1 {
            continue;
        }
        for _ in 0..s - 1 {
            x = mod_pow(x, 2, n);
            if x == n - 1 {
                continue 'witness;
            }
        }
        return false;
    }
    true
}

/// Generates an RSA key from primes `p`, `q` and public exponent `e`.
///
/// Returns `None` when:
/// - `p == q`,
/// - `p` or `q` is not prime (validated by deterministic Miller-Rabin),
/// - `e < 2` or `e >= phi(n)`,
/// - `gcd(e, phi(n)) != 1` (so `e` has no inverse mod phi).
///
/// On success the modulus is `n = p * q` and the private exponent is the
/// unique `d` in `[1, phi(n))` with `e * d ≡ 1 (mod phi(n))`.
pub fn keygen(p: u64, q: u64, e: u64) -> Option<RsaKey> {
    if p == q {
        return None;
    }
    if !is_prime_u64(p) || !is_prime_u64(q) {
        return None;
    }
    // n = p*q must fit in u64. With p, q both ≤ 2^32 this always holds; the
    // checked_mul guards the general case where one is much larger.
    let n = p.checked_mul(q)?;
    let phi = (p - 1).checked_mul(q - 1)?;
    if e < 2 || e >= phi {
        return None;
    }
    if gcd(e, phi) != 1 {
        return None;
    }
    // mod_inverse works in i64; phi fits because p,q are u64 primes used in
    // a teaching context (callers stick to small primes). Reject if phi
    // would overflow i64 to keep the cast safe.
    if phi > i64::MAX as u64 || e > i64::MAX as u64 {
        return None;
    }
    let d = mod_inverse(e as i64, phi as i64)? as u64;
    Some(RsaKey { n, e, d })
}

/// Encrypts message `m` under public key `(n, e)` as `m^e mod n`.
///
/// The caller must ensure `m < n`; otherwise the ciphertext is congruent
/// to `m mod n` and round-tripping will lose information.
pub fn encrypt(m: u64, n: u64, e: u64) -> u64 {
    mod_pow(m, e, n)
}

/// Decrypts ciphertext `c` under private key `(n, d)` as `c^d mod n`.
pub fn decrypt(c: u64, n: u64, d: u64) -> u64 {
    mod_pow(c, d, n)
}

#[cfg(test)]
mod tests {
    use super::{decrypt, encrypt, is_prime_u64, keygen, RsaKey};

    #[test]
    fn miller_rabin_small_primes() {
        for n in [
            2_u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 61,
        ] {
            assert!(is_prime_u64(n), "{n} should be prime");
        }
    }

    #[test]
    fn miller_rabin_small_composites() {
        for n in [0_u64, 1, 4, 6, 9, 15, 21, 25, 27, 33, 35, 49, 91, 121] {
            assert!(!is_prime_u64(n), "{n} should be composite");
        }
    }

    #[test]
    fn miller_rabin_large_known_primes() {
        // Mersenne prime 2^31 - 1 and a few neighbours.
        assert!(is_prime_u64(2_147_483_647));
        assert!(!is_prime_u64(2_147_483_645));
        assert!(is_prime_u64(1_000_000_007));
        assert!(is_prime_u64(1_000_000_009));
        // Carmichael number 561 = 3·11·17 — fools Fermat, not Miller-Rabin.
        assert!(!is_prime_u64(561));
    }

    #[test]
    fn classic_textbook_example() {
        // The canonical Wikipedia example: p=61, q=53, e=17 → n=3233, d=2753.
        let key = keygen(61, 53, 17).expect("valid RSA parameters");
        assert_eq!(
            key,
            RsaKey {
                n: 3233,
                e: 17,
                d: 2753,
            }
        );
        // Sanity check the inverse relation.
        assert_eq!((key.e * key.d) % ((61 - 1) * (53 - 1)), 1);
    }

    #[test]
    fn encrypt_then_decrypt_round_trip() {
        let key = keygen(61, 53, 17).unwrap();
        for m in [0_u64, 1, 2, 3, 42, 65, 1234, 3232] {
            let c = encrypt(m, key.n, key.e);
            let m2 = decrypt(c, key.n, key.d);
            assert_eq!(m2, m, "round-trip failed for m={m}");
        }
    }

    #[test]
    fn keygen_rejects_non_prime_p() {
        // 60 is composite.
        assert!(keygen(60, 53, 17).is_none());
    }

    #[test]
    fn keygen_rejects_non_prime_q() {
        // 55 = 5*11 is composite.
        assert!(keygen(61, 55, 17).is_none());
    }

    #[test]
    fn keygen_rejects_equal_primes() {
        assert!(keygen(61, 61, 17).is_none());
    }

    #[test]
    fn keygen_rejects_non_coprime_e() {
        // phi(11*13) = 10*12 = 120. e=4 shares factor 2 with 120.
        assert!(keygen(11, 13, 4).is_none());
        // e=6 shares factor 6 with 120.
        assert!(keygen(11, 13, 6).is_none());
    }

    #[test]
    fn keygen_rejects_out_of_range_e() {
        // e must satisfy 2 ≤ e < phi.
        assert!(keygen(11, 13, 0).is_none());
        assert!(keygen(11, 13, 1).is_none());
        assert!(keygen(11, 13, 120).is_none()); // == phi
        assert!(keygen(11, 13, 999).is_none()); // > phi
    }

    /// Tiny linear-congruential PRNG so the property test is deterministic
    /// without pulling in `rand`.
    fn lcg_next(state: &mut u64) -> u64 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        *state
    }

    #[test]
    fn property_round_trip_over_small_prime_set() {
        let primes: [u64; 8] = [61, 67, 71, 73, 79, 83, 89, 97];
        let exponents: [u64; 4] = [3, 5, 17, 257];
        let mut rng_state: u64 = 0x00DE_ADBE_EFC0_FFEE;

        for i in 0..primes.len() {
            for j in 0..primes.len() {
                if i == j {
                    continue;
                }
                let p = primes[i];
                let q = primes[j];
                for &e in &exponents {
                    let Some(key) = keygen(p, q, e) else {
                        // Skip pairs where e isn't coprime with phi.
                        continue;
                    };
                    for _ in 0..8 {
                        let m = lcg_next(&mut rng_state) % key.n;
                        let c = encrypt(m, key.n, key.e);
                        let m2 = decrypt(c, key.n, key.d);
                        assert_eq!(m2, m, "p={p}, q={q}, e={e}, m={m}");
                    }
                }
            }
        }
    }
}
