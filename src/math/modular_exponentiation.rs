//! Modular exponentiation: `(base^exp) mod m` in O(log exp). Uses `u128`
//! intermediate products to avoid overflow when `m` fits in `u64`.

/// Returns `(base^exp) mod modulus`. `modulus = 0` panics.
pub fn mod_pow(base: u64, mut exp: u64, modulus: u64) -> u64 {
    assert!(modulus > 0, "modulus must be positive");
    if modulus == 1 {
        return 0;
    }
    let mut result = 1_u128;
    let m = u128::from(modulus);
    let mut b = u128::from(base) % m;
    while exp > 0 {
        if exp & 1 == 1 {
            result = (result * b) % m;
        }
        exp >>= 1;
        b = (b * b) % m;
    }
    result as u64
}

#[cfg(test)]
mod tests {
    use super::mod_pow;

    #[test]
    fn base_zero() {
        assert_eq!(mod_pow(0, 5, 13), 0);
        assert_eq!(mod_pow(0, 0, 13), 1); // 0^0 = 1 by convention
    }

    #[test]
    fn exp_zero() {
        assert_eq!(mod_pow(7, 0, 13), 1);
    }

    #[test]
    fn modulus_one_is_always_zero() {
        assert_eq!(mod_pow(123, 456, 1), 0);
    }

    #[test]
    fn small_examples() {
        assert_eq!(mod_pow(2, 10, 1000), 24); // 1024 % 1000
        assert_eq!(mod_pow(3, 5, 7), 5); // 243 % 7
        assert_eq!(mod_pow(5, 117, 19), 1); // by Fermat: 5^18 = 1, 117 = 6·18 + 9 → 5^9 mod 19 = 1
    }

    #[test]
    fn large_exponent_no_overflow() {
        // 2^62 mod 1_000_000_007
        assert_eq!(mod_pow(2, 62, 1_000_000_007), 145_586_002);
    }

    #[test]
    fn against_naive_for_small_inputs() {
        let modulus = 97_u64;
        for base in 0..15 {
            for exp in 0..15 {
                let mut naive = 1_u64;
                for _ in 0..exp {
                    naive = (naive * base) % modulus;
                }
                assert_eq!(mod_pow(base, exp, modulus), naive, "{base}^{exp}");
            }
        }
    }

    #[test]
    #[should_panic(expected = "modulus must be positive")]
    fn modulus_zero_panics() {
        mod_pow(2, 3, 0);
    }
}
