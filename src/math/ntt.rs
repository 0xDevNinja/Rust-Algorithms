//! Number-theoretic transform (NTT) over the prime field
//! `p = 998244353 = 119 * 2^23 + 1` with primitive root `g = 3`.
//!
//! The NTT is the analog of the FFT in `Z/pZ`: it evaluates a polynomial of
//! degree `< n` at the `n`-th roots of unity modulo `p`, where `n` is a power
//! of two with `n | (p - 1)`. Because all arithmetic is exact integer
//! arithmetic, NTT-based convolution has no floating-point error, making it
//! the standard tool for integer polynomial multiplication.
//!
//! Time complexity: `O(n log n)` for `ntt_in_place`, `O((n + m) log(n + m))`
//! for `convolve`. Space complexity: `O(n)`.
//!
//! Length must be a power of two and at most `2^23`, since `p - 1 = 119 * 2^23`
//! has exactly `2^23` as its largest power-of-two factor.
//!
//! Reference: <https://cp-algorithms.com/algebra/fft.html#number-theoretic-transform>.
//!
//! Implementation: iterative Cooley-Tukey radix-2 with bit-reversal
//! permutation. Inverse uses `g^{-1}` and a final multiplication by `n^{-1}`.

use crate::math::extended_euclidean::mod_inverse;
use crate::math::modular_exponentiation::mod_pow;

/// NTT-friendly prime: `998244353 = 119 * 2^23 + 1`.
pub const NTT_MOD: u64 = 998_244_353;
/// Primitive root of `NTT_MOD`.
pub const NTT_G: u64 = 3;
/// Largest power-of-two length supported (`2^23`).
pub const NTT_MAX_LEN: usize = 1 << 23;

#[inline]
const fn mul_mod(a: u64, b: u64) -> u64 {
    ((a as u128 * b as u128) % NTT_MOD as u128) as u64
}

#[inline]
const fn add_mod(a: u64, b: u64) -> u64 {
    let s = a + b;
    if s >= NTT_MOD {
        s - NTT_MOD
    } else {
        s
    }
}

#[inline]
const fn sub_mod(a: u64, b: u64) -> u64 {
    if a >= b {
        a - b
    } else {
        a + NTT_MOD - b
    }
}

const fn inv_mod_p(a: u64) -> u64 {
    // p is prime, so the extended Euclidean inverse always exists for nonzero a.
    match mod_inverse(a as i64, NTT_MOD as i64) {
        Some(inv) => inv as u64,
        None => panic!("a must be coprime to NTT_MOD"),
    }
}

/// Iterative Cooley-Tukey radix-2 NTT modulo `NTT_MOD`, in place.
///
/// `a.len()` must be a power of two and at most `NTT_MAX_LEN`. When
/// `invert == true`, performs the inverse transform (including the
/// `n^{-1}` scaling). All input values are reduced modulo `NTT_MOD`
/// on entry; output values lie in `[0, NTT_MOD)`.
#[allow(clippy::ptr_arg)]
pub fn ntt_in_place(a: &mut Vec<u64>, invert: bool) {
    let n = a.len();
    if n <= 1 {
        if let Some(x) = a.first_mut() {
            *x %= NTT_MOD;
        }
        return;
    }
    assert!(
        n.is_power_of_two(),
        "ntt length must be a power of two, got {n}"
    );
    assert!(
        n <= NTT_MAX_LEN,
        "ntt length {n} exceeds NTT_MAX_LEN ({NTT_MAX_LEN})"
    );

    // Reduce inputs to canonical form.
    for x in a.iter_mut() {
        *x %= NTT_MOD;
    }

    // Bit-reversal permutation.
    let mut j = 0_usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            a.swap(i, j);
        }
    }

    // Butterfly stages.
    let mut len = 2_usize;
    while len <= n {
        // w_len is a primitive len-th root of unity mod p (or its inverse).
        // For length len, exponent = (p - 1) / len.
        let exp = (NTT_MOD - 1) / (len as u64);
        let mut w_len = mod_pow(NTT_G, exp, NTT_MOD);
        if invert {
            w_len = inv_mod_p(w_len);
        }
        let half = len >> 1;
        let mut i = 0_usize;
        while i < n {
            let mut w = 1_u64;
            for k in 0..half {
                let u = a[i + k];
                let v = mul_mod(a[i + k + half], w);
                a[i + k] = add_mod(u, v);
                a[i + k + half] = sub_mod(u, v);
                w = mul_mod(w, w_len);
            }
            i += len;
        }
        len <<= 1;
    }

    if invert {
        let n_inv = inv_mod_p(n as u64);
        for x in a.iter_mut() {
            *x = mul_mod(*x, n_inv);
        }
    }
}

/// Convolves two integer polynomials (low coefficient first) modulo
/// `NTT_MOD`. Pads to the next power of two `>= |a| + |b| - 1`, performs
/// forward NTT on both, multiplies pointwise, then inverse NTT. Returns
/// the `|a| + |b| - 1` coefficients of the product, each in `[0, NTT_MOD)`.
///
/// If either input is empty, returns an empty vector.
pub fn convolve(a: &[u64], b: &[u64]) -> Vec<u64> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let result_len = a.len() + b.len() - 1;
    let mut n = 1_usize;
    while n < result_len {
        n <<= 1;
    }
    assert!(
        n <= NTT_MAX_LEN,
        "convolution length {n} exceeds NTT_MAX_LEN ({NTT_MAX_LEN})"
    );

    let mut fa: Vec<u64> = a.iter().map(|&x| x % NTT_MOD).collect();
    fa.resize(n, 0);
    let mut fb: Vec<u64> = b.iter().map(|&x| x % NTT_MOD).collect();
    fb.resize(n, 0);

    ntt_in_place(&mut fa, false);
    ntt_in_place(&mut fb, false);
    for i in 0..n {
        fa[i] = mul_mod(fa[i], fb[i]);
    }
    ntt_in_place(&mut fa, true);

    fa.truncate(result_len);
    fa
}

#[cfg(test)]
mod tests {
    use super::*;

    fn naive_convolve(a: &[u64], b: &[u64]) -> Vec<u64> {
        if a.is_empty() || b.is_empty() {
            return Vec::new();
        }
        let mut out = vec![0_u64; a.len() + b.len() - 1];
        for (i, &x) in a.iter().enumerate() {
            for (j, &y) in b.iter().enumerate() {
                out[i + j] = add_mod(out[i + j], mul_mod(x % NTT_MOD, y % NTT_MOD));
            }
        }
        out
    }

    #[test]
    fn closed_form_small_convolution() {
        // (1 + 2x + 3x^2) * (4 + 5x + 6x^2)
        // = 4 + (5+8)x + (6+10+12)x^2 + (12+15)x^3 + 18 x^4
        // = 4 + 13x + 28x^2 + 27x^3 + 18x^4
        let a = vec![1_u64, 2, 3];
        let b = vec![4_u64, 5, 6];
        assert_eq!(convolve(&a, &b), vec![4, 13, 28, 27, 18]);
    }

    #[test]
    fn identity_convolution() {
        let a = vec![1_u64];
        let b = vec![7_u64, 11, 13, 17, 19];
        assert_eq!(convolve(&a, &b), b);
        assert_eq!(convolve(&b, &a), b);
    }

    #[test]
    fn empty_input_returns_empty() {
        let empty: Vec<u64> = Vec::new();
        assert_eq!(convolve(&empty, &[1, 2, 3]), Vec::<u64>::new());
        assert_eq!(convolve(&[1, 2, 3], &empty), Vec::<u64>::new());
        assert_eq!(convolve(&empty, &empty), Vec::<u64>::new());
    }

    #[test]
    fn round_trip_recovers_original() {
        for &n in &[1_usize, 2, 4, 8, 16, 32, 64] {
            // Deterministic LCG to fill a power-of-two-length vector.
            let mut state = 0x1234_5678_u64;
            let mut data: Vec<u64> = (0..n)
                .map(|_| {
                    state = state
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    (state >> 33) % NTT_MOD
                })
                .collect();
            let original = data.clone();
            ntt_in_place(&mut data, false);
            ntt_in_place(&mut data, true);
            assert_eq!(data, original, "round trip failed at n = {n}");
        }
    }

    #[test]
    fn matches_naive_on_random_inputs() {
        // Deterministic pseudo-random tests across several length pairs.
        let mut state = 0xdead_beef_cafe_u64;
        let mut next = || -> u64 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 33) & 0xffff_ffff
        };

        for &(la, lb) in &[
            (1, 1),
            (1, 5),
            (5, 1),
            (3, 4),
            (7, 9),
            (16, 16),
            (33, 17),
            (50, 50),
        ] {
            let a: Vec<u64> = (0..la).map(|_| next() % NTT_MOD).collect();
            let b: Vec<u64> = (0..lb).map(|_| next() % NTT_MOD).collect();
            let got = convolve(&a, &b);
            let want = naive_convolve(&a, &b);
            assert_eq!(got, want, "mismatch for lengths ({la}, {lb})");
        }
    }

    #[test]
    fn ntt_zero_input_stays_zero() {
        let mut data = vec![0_u64; 8];
        ntt_in_place(&mut data, false);
        assert!(data.iter().all(|&x| x == 0));
        ntt_in_place(&mut data, true);
        assert!(data.iter().all(|&x| x == 0));
    }

    #[test]
    fn ntt_single_element_no_op() {
        let mut data = vec![42_u64];
        ntt_in_place(&mut data, false);
        assert_eq!(data, vec![42]);
        ntt_in_place(&mut data, true);
        assert_eq!(data, vec![42]);
    }

    #[test]
    fn output_is_canonical_mod_p() {
        // Inputs intentionally above NTT_MOD; output must still be in [0, p).
        let a: Vec<u64> = vec![NTT_MOD + 5, 2 * NTT_MOD + 7, 3];
        let b: Vec<u64> = vec![NTT_MOD - 1, 4, NTT_MOD + 9];
        let got = convolve(&a, &b);
        let want = naive_convolve(&a, &b);
        assert_eq!(got, want);
        assert!(got.iter().all(|&x| x < NTT_MOD));
    }

    #[test]
    #[should_panic(expected = "ntt length must be a power of two")]
    fn non_power_of_two_panics() {
        let mut data = vec![1_u64, 2, 3];
        ntt_in_place(&mut data, false);
    }
}
