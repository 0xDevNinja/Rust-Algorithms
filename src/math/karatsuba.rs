//! Karatsuba multiplication for arbitrary-precision non-negative integers.
//!
//! Operands are represented as little-endian base-`2^32` limb slices (`&[u32]`),
//! i.e. `a[0]` is the least significant limb. The result is returned in the same
//! representation with leading zero limbs trimmed.
//!
//! # Complexity
//! Naive schoolbook multiplication is `O(n*m)`. Karatsuba splits each operand at
//! a half-limb boundary and recurses on three (instead of four) sub-products,
//! achieving `O(n^log2 3) ≈ O(n^1.585)` time when both operands have `n` limbs.
//! Below a small limb threshold the recursion bottoms out into the schoolbook
//! routine, which is faster for tiny inputs.
//!
//! # Notes
//! No `unsafe`, no external dependencies. The schoolbook inner loop accumulates
//! `limb*limb + carry + acc` in a `u64`, which always fits since
//! `(2^32 - 1)^2 + 2*(2^32 - 1) = 2^64 - 1`.

/// Threshold below which the recursion falls back to schoolbook multiplication.
///
/// Operands with both lengths at most this many limbs are multiplied directly.
const KARATSUBA_THRESHOLD: usize = 32;

/// Removes trailing zero limbs (which represent leading zeros in the integer).
fn trim(v: &mut Vec<u32>) {
    while v.last().copied() == Some(0) {
        v.pop();
    }
}

/// Adds `b` into `a` in place, growing `a` as needed.
///
/// `a` is interpreted as a little-endian base-`2^32` integer. Carries propagate
/// through any extra high limbs of `a` and append a final limb if necessary.
pub fn add_assign(a: &mut Vec<u32>, b: &[u32]) {
    if a.len() < b.len() {
        a.resize(b.len(), 0);
    }
    let mut carry: u64 = 0;
    for i in 0..a.len() {
        let bi = if i < b.len() { u64::from(b[i]) } else { 0 };
        let sum = u64::from(a[i]) + bi + carry;
        a[i] = sum as u32;
        carry = sum >> 32;
        if carry == 0 && i >= b.len() {
            return;
        }
    }
    if carry != 0 {
        a.push(carry as u32);
    }
}

/// Subtracts `b` from `a` in place. Panics if `a < b`.
///
/// Both operands are little-endian base-`2^32` limb sequences.
pub fn sub_assign(a: &mut Vec<u32>, b: &[u32]) {
    let mut borrow: i64 = 0;
    let max_len = a.len().max(b.len());
    if a.len() < max_len {
        // Cannot possibly subtract a longer value; will be detected as underflow below.
        a.resize(max_len, 0);
    }
    for i in 0..max_len {
        let ai = i64::from(a[i]);
        let bi = if i < b.len() { i64::from(b[i]) } else { 0 };
        let diff = ai - bi - borrow;
        if diff < 0 {
            a[i] = (diff + (1_i64 << 32)) as u32;
            borrow = 1;
        } else {
            a[i] = diff as u32;
            borrow = 0;
        }
    }
    assert!(borrow == 0, "sub_assign: subtrahend larger than minuend");
    trim(a);
}

/// Schoolbook (long-form) multiplication of two limb slices.
///
/// Runs in `O(a.len() * b.len())`. Accumulates each column with a `u64` carry,
/// which is sufficient since `(2^32 - 1)^2 + 2*(2^32 - 1)` still fits in 64 bits.
pub fn schoolbook(a: &[u32], b: &[u32]) -> Vec<u32> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let mut out = vec![0_u32; a.len() + b.len()];
    for (i, &ai) in a.iter().enumerate() {
        if ai == 0 {
            continue;
        }
        let ai = u64::from(ai);
        let mut carry: u64 = 0;
        for (j, &bj) in b.iter().enumerate() {
            let cur = u64::from(out[i + j]) + ai * u64::from(bj) + carry;
            out[i + j] = cur as u32;
            carry = cur >> 32;
        }
        out[i + b.len()] = carry as u32;
    }
    trim(&mut out);
    out
}

/// Multiplies two non-negative big integers given as little-endian base-`2^32`
/// limb slices, returning the product as a trimmed `Vec<u32>`.
///
/// Uses Karatsuba's three-multiplication recursion above
/// [`KARATSUBA_THRESHOLD`] limbs and schoolbook multiplication below it. The
/// result has no trailing zero limbs (so the integer `0` is returned as `vec![]`).
pub fn karatsuba_mul(a: &[u32], b: &[u32]) -> Vec<u32> {
    // Strip leading zero limbs so the split point reflects the real magnitude.
    let mut a_end = a.len();
    while a_end > 0 && a[a_end - 1] == 0 {
        a_end -= 1;
    }
    let mut b_end = b.len();
    while b_end > 0 && b[b_end - 1] == 0 {
        b_end -= 1;
    }
    let a = &a[..a_end];
    let b = &b[..b_end];
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }

    if a.len() <= KARATSUBA_THRESHOLD || b.len() <= KARATSUBA_THRESHOLD {
        return schoolbook(a, b);
    }

    // Split at half the longer operand. Both halves are non-empty because
    // both lengths exceed the threshold here.
    let m = a.len().max(b.len()) / 2;

    let (a_lo, a_hi) = if a.len() > m {
        (&a[..m], &a[m..])
    } else {
        (a, &a[a.len()..])
    };
    let (b_lo, b_hi) = if b.len() > m {
        (&b[..m], &b[m..])
    } else {
        (b, &b[b.len()..])
    };

    // z0 = a_lo * b_lo
    let z0 = karatsuba_mul(a_lo, b_lo);
    // z2 = a_hi * b_hi
    let z2 = karatsuba_mul(a_hi, b_hi);

    // a_sum = a_lo + a_hi, b_sum = b_lo + b_hi
    let mut a_sum: Vec<u32> = a_lo.to_vec();
    add_assign(&mut a_sum, a_hi);
    let mut b_sum: Vec<u32> = b_lo.to_vec();
    add_assign(&mut b_sum, b_hi);

    // z1 = a_sum * b_sum - z0 - z2
    let mut z1 = karatsuba_mul(&a_sum, &b_sum);
    sub_assign(&mut z1, &z0);
    sub_assign(&mut z1, &z2);

    // result = z0 + (z1 << 32m) + (z2 << 64m)
    let mut result: Vec<u32> = Vec::with_capacity(a.len() + b.len() + 1);
    result.extend_from_slice(&z0);

    if !z1.is_empty() {
        let mut shifted_z1 = vec![0_u32; m];
        shifted_z1.extend_from_slice(&z1);
        add_assign(&mut result, &shifted_z1);
    }
    if !z2.is_empty() {
        let mut shifted_z2 = vec![0_u32; 2 * m];
        shifted_z2.extend_from_slice(&z2);
        add_assign(&mut result, &shifted_z2);
    }

    trim(&mut result);
    result
}

#[cfg(test)]
mod tests {
    use super::{add_assign, karatsuba_mul, schoolbook, sub_assign};
    use quickcheck_macros::quickcheck;

    /// Convert a `u128` to little-endian base-`2^32` limbs (trimmed).
    fn limbs_from_u128(x: u128) -> Vec<u32> {
        let mut v = Vec::new();
        let mut x = x;
        while x != 0 {
            v.push(x as u32);
            x >>= 32;
        }
        v
    }

    /// Convert little-endian base-`2^32` limbs back to a `u128` (panics on overflow).
    fn u128_from_limbs(limbs: &[u32]) -> u128 {
        let mut result: u128 = 0;
        for (i, &l) in limbs.iter().enumerate() {
            assert!(i < 4, "limbs do not fit in u128");
            result |= u128::from(l) << (32 * i);
        }
        result
    }

    /// Parse a non-negative decimal string into base-`2^32` limbs.
    fn limbs_from_decimal(s: &str) -> Vec<u32> {
        let mut limbs: Vec<u32> = Vec::new();
        for ch in s.chars() {
            let d = ch.to_digit(10).expect("decimal digit") as u64;
            // limbs *= 10
            let mut carry: u64 = d;
            for limb in &mut limbs {
                let cur = u64::from(*limb) * 10 + carry;
                *limb = cur as u32;
                carry = cur >> 32;
            }
            while carry != 0 {
                limbs.push(carry as u32);
                carry >>= 32;
            }
        }
        while limbs.last().copied() == Some(0) {
            limbs.pop();
        }
        limbs
    }

    /// Render base-`2^32` limbs as a decimal string.
    fn decimal_from_limbs(limbs: &[u32]) -> String {
        if limbs.is_empty() {
            return "0".to_string();
        }
        let mut limbs = limbs.to_vec();
        let mut digits = Vec::new();
        while !limbs.is_empty() {
            // divide limbs by 10, collect remainder.
            let mut rem: u64 = 0;
            for limb in limbs.iter_mut().rev() {
                let cur = (rem << 32) | u64::from(*limb);
                *limb = (cur / 10) as u32;
                rem = cur % 10;
            }
            digits.push(char::from(b'0' + rem as u8));
            while limbs.last().copied() == Some(0) {
                limbs.pop();
            }
        }
        digits.iter().rev().collect()
    }

    #[test]
    fn zero_times_anything() {
        let x = limbs_from_u128(123_456_789);
        assert!(karatsuba_mul(&[], &x).is_empty());
        assert!(karatsuba_mul(&x, &[]).is_empty());
        assert!(karatsuba_mul(&[0, 0, 0], &x).is_empty());
    }

    #[test]
    fn one_times_x_equals_x() {
        let x = limbs_from_u128(0x_dead_beef_cafe_babe_u128);
        let one = vec![1_u32];
        assert_eq!(karatsuba_mul(&one, &x), x);
        assert_eq!(karatsuba_mul(&x, &one), x);
    }

    #[test]
    fn small_products_match_u128() {
        for a in 0_u128..50 {
            for b in 0_u128..50 {
                let la = limbs_from_u128(a);
                let lb = limbs_from_u128(b);
                let prod = karatsuba_mul(&la, &lb);
                assert_eq!(u128_from_limbs(&prod), a * b, "{a} * {b}");
            }
        }
    }

    #[test]
    fn schoolbook_matches_u128() {
        for a in [
            0_u128,
            1,
            2,
            0xffff_ffff,
            0x1_0000_0001,
            0xffff_ffff_ffff_ffff,
        ] {
            for b in [0_u128, 1, 2, 0xffff_ffff, 0xffff_ffff_ffff_ffff] {
                let la = limbs_from_u128(a);
                let lb = limbs_from_u128(b);
                assert_eq!(u128_from_limbs(&schoolbook(&la, &lb)), a * b);
            }
        }
    }

    #[test]
    fn add_assign_basic() {
        let mut a = vec![0xffff_ffff_u32, 0];
        add_assign(&mut a, &[1]);
        assert_eq!(a, vec![0, 1]);

        let mut a = vec![0xffff_ffff_u32];
        add_assign(&mut a, &[0xffff_ffff]);
        assert_eq!(a, vec![0xffff_fffe, 1]);

        let mut a = vec![1_u32, 2, 3];
        add_assign(&mut a, &[10, 20]);
        assert_eq!(a, vec![11, 22, 3]);
    }

    #[test]
    fn sub_assign_basic() {
        let mut a = vec![0_u32, 1];
        sub_assign(&mut a, &[1]);
        assert_eq!(a, vec![0xffff_ffff]);

        let mut a = vec![5_u32, 5];
        sub_assign(&mut a, &[5, 5]);
        assert!(a.is_empty());
    }

    #[test]
    #[should_panic(expected = "subtrahend larger than minuend")]
    fn sub_assign_underflow_panics() {
        let mut a = vec![1_u32];
        sub_assign(&mut a, &[2]);
    }

    #[test]
    fn asymmetric_lengths() {
        // big * small should hit schoolbook for short side, recurse for long side.
        let big: Vec<u32> = (1_u32..=200).collect();
        let small: Vec<u32> = vec![3, 0, 5];
        let p1 = karatsuba_mul(&big, &small);
        let p2 = schoolbook(&big, &small);
        assert_eq!(p1, p2);

        // Lengths that stretch beyond the threshold on one side only.
        let a: Vec<u32> = (0..100).map(|i| i * 7 + 1).collect();
        let b: Vec<u32> = (0..5).map(|i| i + 1).collect();
        assert_eq!(karatsuba_mul(&a, &b), schoolbook(&a, &b));
        assert_eq!(karatsuba_mul(&b, &a), schoolbook(&a, &b));
    }

    #[test]
    fn large_recursive_matches_schoolbook() {
        // Both sides above the threshold so the karatsuba branch executes.
        let a: Vec<u32> = (1_u32..=80)
            .map(|i| i.wrapping_mul(2_654_435_761))
            .collect();
        let b: Vec<u32> = (1_u32..=70).map(|i| i.wrapping_mul(40_503)).collect();
        assert_eq!(karatsuba_mul(&a, &b), schoolbook(&a, &b));
    }

    #[test]
    fn hundred_digit_decimal_round_trip() {
        let a_str =
            "1234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890";
        let b_str =
            "9876543210987654321098765432109876543210987654321098765432109876543210987654321098765432109876543210";
        // Reference product computed with the same big-int helpers, but via schoolbook.
        let a = limbs_from_decimal(a_str);
        let b = limbs_from_decimal(b_str);

        let karat = karatsuba_mul(&a, &b);
        let book = schoolbook(&a, &b);
        assert_eq!(karat, book);

        // Round-trip the karatsuba product through the decimal helper and
        // verify against a digit-by-digit hand-rolled multiplication of the
        // input strings (Pen-and-paper schoolbook over chars).
        let prod_decimal = decimal_from_limbs(&karat);
        let expected = decimal_string_mul(a_str, b_str);
        assert_eq!(prod_decimal, expected);

        // And the limb helpers are themselves consistent.
        assert_eq!(decimal_from_limbs(&limbs_from_decimal(a_str)), a_str);
    }

    /// Schoolbook string-times-string multiplication used as an independent
    /// oracle for [`hundred_digit_decimal_round_trip`].
    fn decimal_string_mul(a: &str, b: &str) -> String {
        let a: Vec<u8> = a.bytes().rev().map(|c| c - b'0').collect();
        let b: Vec<u8> = b.bytes().rev().map(|c| c - b'0').collect();
        let mut out = vec![0_u32; a.len() + b.len()];
        for (i, &ai) in a.iter().enumerate() {
            for (j, &bj) in b.iter().enumerate() {
                out[i + j] += u32::from(ai) * u32::from(bj);
            }
        }
        // Carry propagation.
        let mut carry: u32 = 0;
        for digit in &mut out {
            let cur = *digit + carry;
            *digit = cur % 10;
            carry = cur / 10;
        }
        while carry != 0 {
            out.push(carry % 10);
            carry /= 10;
        }
        while out.last().copied() == Some(0) && out.len() > 1 {
            out.pop();
        }
        out.iter()
            .rev()
            .map(|d| char::from(b'0' + *d as u8))
            .collect()
    }

    #[quickcheck]
    fn prop_random_u64_pairs(a: u64, b: u64) -> bool {
        let la = limbs_from_u128(u128::from(a));
        let lb = limbs_from_u128(u128::from(b));
        let prod = karatsuba_mul(&la, &lb);
        u128_from_limbs(&prod) == u128::from(a) * u128::from(b)
    }

    #[quickcheck]
    #[allow(clippy::needless_pass_by_value)]
    fn prop_karatsuba_matches_schoolbook(a: Vec<u32>, b: Vec<u32>) -> bool {
        karatsuba_mul(&a, &b) == schoolbook(&a, &b)
    }
}
