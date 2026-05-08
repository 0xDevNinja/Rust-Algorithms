//! Unsigned arbitrary-precision integer arithmetic. Limbs are stored in
//! little-endian order in base `2^32` (`Vec<u32>`), and the representation
//! is always normalised so that the most significant limb is non-zero
//! (the integer zero is the empty limb vector).
//!
//! Subtraction saturates: if the minuend is smaller than the subtrahend,
//! the result is zero rather than panicking. The same convention applies
//! to the `Sub` operator overload.
//!
//! Complexity, with `n` and `m` denoting the limb counts of the operands:
//! - `add`, `sub`, `cmp`: `O(max(n, m))`
//! - `mul` (schoolbook): `O(n * m)`
//! - `div_rem` by `u32`: `O(n)`
//! - `div_rem` by `BigUint`: `O((n - m + 1) * m)` (long division by trial)
//! - `from_str_radix` / `to_string_radix`: `O(n * len(s))`

use core::cmp::Ordering;
use core::hash::{Hash, Hasher};
use core::ops::{Add, Div, Mul, Rem, Sub};

const BASE_BITS: u32 = 32;
const BASE: u64 = 1u64 << BASE_BITS;
const BASE_MASK: u64 = BASE - 1;

/// Unsigned arbitrary-precision integer.
///
/// Internally a normalised little-endian `Vec<u32>`: the limb at index
/// `i` carries weight `2^(32 * i)` and the most significant limb (if any)
/// is non-zero.
#[derive(Clone, Debug, Default)]
pub struct BigUint {
    limbs: Vec<u32>,
}

impl BigUint {
    /// The integer zero.
    #[must_use]
    pub const fn zero() -> Self {
        Self { limbs: Vec::new() }
    }

    /// Returns `true` iff this value equals zero.
    #[must_use]
    pub const fn is_zero(&self) -> bool {
        self.limbs.is_empty()
    }

    /// Builds a `BigUint` from a `u64`. Two limbs at most.
    #[must_use]
    pub fn from_u64(value: u64) -> Self {
        let mut limbs = Vec::with_capacity(2);
        let lo = (value & BASE_MASK) as u32;
        let hi = (value >> BASE_BITS) as u32;
        if lo != 0 || hi != 0 {
            limbs.push(lo);
            if hi != 0 {
                limbs.push(hi);
            }
        }
        Self { limbs }
    }

    fn normalise(&mut self) {
        while self.limbs.last().copied() == Some(0) {
            self.limbs.pop();
        }
    }

    /// Returns `self + other`.
    #[must_use]
    pub fn add(&self, other: &Self) -> Self {
        let n = self.limbs.len().max(other.limbs.len());
        let mut out = Vec::with_capacity(n + 1);
        let mut carry: u64 = 0;
        for i in 0..n {
            let a = u64::from(self.limbs.get(i).copied().unwrap_or(0));
            let b = u64::from(other.limbs.get(i).copied().unwrap_or(0));
            let sum = a + b + carry;
            out.push((sum & BASE_MASK) as u32);
            carry = sum >> BASE_BITS;
        }
        if carry != 0 {
            out.push(carry as u32);
        }
        let mut result = Self { limbs: out };
        result.normalise();
        result
    }

    /// Saturating subtraction: returns `self - other`, or zero if
    /// `self < other`.
    #[must_use]
    pub fn sub(&self, other: &Self) -> Self {
        if self.cmp(other) == Ordering::Less {
            return Self::zero();
        }
        let mut out = Vec::with_capacity(self.limbs.len());
        let mut borrow: i64 = 0;
        for i in 0..self.limbs.len() {
            let a = i64::from(self.limbs[i]);
            let b = i64::from(other.limbs.get(i).copied().unwrap_or(0));
            let diff = a - b - borrow;
            if diff < 0 {
                out.push((diff + BASE as i64) as u32);
                borrow = 1;
            } else {
                out.push(diff as u32);
                borrow = 0;
            }
        }
        let mut result = Self { limbs: out };
        result.normalise();
        result
    }

    /// Schoolbook multiplication: `self * other`.
    #[must_use]
    pub fn mul(&self, other: &Self) -> Self {
        if self.is_zero() || other.is_zero() {
            return Self::zero();
        }
        let mut out = vec![0u32; self.limbs.len() + other.limbs.len()];
        for i in 0..self.limbs.len() {
            let a = u64::from(self.limbs[i]);
            let mut carry: u64 = 0;
            for j in 0..other.limbs.len() {
                let b = u64::from(other.limbs[j]);
                let cur = u64::from(out[i + j]);
                let prod = a * b + cur + carry;
                out[i + j] = (prod & BASE_MASK) as u32;
                carry = prod >> BASE_BITS;
            }
            if carry != 0 {
                out[i + other.limbs.len()] = (u64::from(out[i + other.limbs.len()]) + carry) as u32;
            }
        }
        let mut result = Self { limbs: out };
        result.normalise();
        result
    }

    /// Divides by a single `u32`. Returns `(quotient, remainder)`.
    /// Panics if `divisor` is zero.
    #[must_use]
    pub fn div_rem_u32(&self, divisor: u32) -> (Self, u32) {
        assert!(divisor != 0, "division by zero");
        let d = u64::from(divisor);
        let mut q = vec![0u32; self.limbs.len()];
        let mut rem: u64 = 0;
        for i in (0..self.limbs.len()).rev() {
            let cur = (rem << BASE_BITS) | u64::from(self.limbs[i]);
            q[i] = (cur / d) as u32;
            rem = cur % d;
        }
        let mut quotient = Self { limbs: q };
        quotient.normalise();
        (quotient, rem as u32)
    }

    /// Long division: returns `(self / other, self % other)`.
    /// Panics if `other` is zero.
    #[must_use]
    pub fn div_rem(&self, other: &Self) -> (Self, Self) {
        assert!(!other.is_zero(), "division by zero");
        if let [d] = other.limbs[..] {
            let (q, r) = self.div_rem_u32(d);
            return (q, Self::from_u64(u64::from(r)));
        }
        if self.cmp(other) == Ordering::Less {
            return (Self::zero(), self.clone());
        }
        // Bit-by-bit long division. Simple, dependency-free, runs in
        // O(bits(self) * limbs(other)) which is fine for the test sizes.
        let bits = self.bit_length();
        let mut quotient_bits = vec![0u8; bits];
        let mut rem = Self::zero();
        for i in (0..bits).rev() {
            // rem := rem * 2 + bit i of self
            rem = rem.shl1();
            if self.bit(i) {
                rem = rem.add(&Self::from_u64(1));
            }
            if rem.cmp(other) != Ordering::Less {
                rem = rem.sub(other);
                quotient_bits[i] = 1;
            }
        }
        let q = Self::from_bits_le(&quotient_bits);
        (q, rem)
    }

    /// Number of bits needed to represent `self`. Returns 0 for zero.
    #[must_use]
    pub fn bit_length(&self) -> usize {
        if self.limbs.is_empty() {
            return 0;
        }
        let top = *self.limbs.last().unwrap();
        let top_bits = (BASE_BITS - top.leading_zeros()) as usize;
        (self.limbs.len() - 1) * BASE_BITS as usize + top_bits
    }

    fn bit(&self, index: usize) -> bool {
        let limb = index / BASE_BITS as usize;
        let off = index % BASE_BITS as usize;
        if limb >= self.limbs.len() {
            return false;
        }
        ((self.limbs[limb] >> off) & 1) == 1
    }

    fn shl1(&self) -> Self {
        if self.is_zero() {
            return Self::zero();
        }
        let mut out = Vec::with_capacity(self.limbs.len() + 1);
        let mut carry: u32 = 0;
        for &l in &self.limbs {
            let shifted = (u64::from(l) << 1) | u64::from(carry);
            out.push((shifted & BASE_MASK) as u32);
            carry = (shifted >> BASE_BITS) as u32;
        }
        if carry != 0 {
            out.push(carry);
        }
        let mut result = Self { limbs: out };
        result.normalise();
        result
    }

    fn from_bits_le(bits: &[u8]) -> Self {
        let limb_count = bits.len().div_ceil(BASE_BITS as usize).max(1);
        let mut limbs = vec![0u32; limb_count];
        for (i, &b) in bits.iter().enumerate() {
            if b != 0 {
                limbs[i / BASE_BITS as usize] |= 1u32 << (i % BASE_BITS as usize);
            }
        }
        let mut result = Self { limbs };
        result.normalise();
        result
    }

    /// Parses a string in the given radix, where `radix` is in `2..=16`.
    /// Returns `None` for empty input, an unsupported radix, or any
    /// invalid digit.
    #[must_use]
    pub fn from_str_radix(s: &str, radix: u32) -> Option<Self> {
        if !(2..=16).contains(&radix) || s.is_empty() {
            return None;
        }
        let radix_big = Self::from_u64(u64::from(radix));
        let mut acc = Self::zero();
        for c in s.bytes() {
            let digit = match c {
                b'0'..=b'9' => u32::from(c - b'0'),
                b'a'..=b'f' => u32::from(c - b'a') + 10,
                b'A'..=b'F' => u32::from(c - b'A') + 10,
                _ => return None,
            };
            if digit >= radix {
                return None;
            }
            acc = acc.mul(&radix_big).add(&Self::from_u64(u64::from(digit)));
        }
        Some(acc)
    }

    /// Renders `self` in the given radix, where `radix` is in `2..=16`.
    /// Panics if `radix` is outside that range.
    #[must_use]
    pub fn to_string_radix(&self, radix: u32) -> String {
        const ALPHABET: &[u8; 16] = b"0123456789abcdef";
        assert!((2..=16).contains(&radix), "radix must be in 2..=16");
        if self.is_zero() {
            return "0".to_string();
        }
        let mut buf = Vec::new();
        let mut cur = self.clone();
        while !cur.is_zero() {
            let (q, r) = cur.div_rem_u32(radix);
            buf.push(ALPHABET[r as usize]);
            cur = q;
        }
        buf.reverse();
        String::from_utf8(buf).expect("alphabet is ASCII")
    }
}

impl PartialEq for BigUint {
    fn eq(&self, other: &Self) -> bool {
        self.limbs == other.limbs
    }
}

impl Eq for BigUint {}

impl Hash for BigUint {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.limbs.hash(state);
    }
}

impl Ord for BigUint {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.limbs.len().cmp(&other.limbs.len()) {
            Ordering::Equal => {
                for i in (0..self.limbs.len()).rev() {
                    match self.limbs[i].cmp(&other.limbs[i]) {
                        Ordering::Equal => {}
                        non_eq => return non_eq,
                    }
                }
                Ordering::Equal
            }
            non_eq => non_eq,
        }
    }
}

impl PartialOrd for BigUint {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl core::fmt::Display for BigUint {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(&self.to_string_radix(10))
    }
}

impl From<u64> for BigUint {
    fn from(value: u64) -> Self {
        Self::from_u64(value)
    }
}

impl From<u32> for BigUint {
    fn from(value: u32) -> Self {
        Self::from_u64(u64::from(value))
    }
}

impl Add for BigUint {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self::add(&self, &rhs)
    }
}

impl Add<&Self> for BigUint {
    type Output = Self;
    fn add(self, rhs: &Self) -> Self::Output {
        Self::add(&self, rhs)
    }
}

impl Add for &BigUint {
    type Output = BigUint;
    fn add(self, rhs: Self) -> Self::Output {
        BigUint::add(self, rhs)
    }
}

impl Sub for BigUint {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Self::sub(&self, &rhs)
    }
}

impl Sub<&Self> for BigUint {
    type Output = Self;
    fn sub(self, rhs: &Self) -> Self::Output {
        Self::sub(&self, rhs)
    }
}

impl Sub for &BigUint {
    type Output = BigUint;
    fn sub(self, rhs: Self) -> Self::Output {
        BigUint::sub(self, rhs)
    }
}

impl Mul for BigUint {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        Self::mul(&self, &rhs)
    }
}

impl Mul<&Self> for BigUint {
    type Output = Self;
    fn mul(self, rhs: &Self) -> Self::Output {
        Self::mul(&self, rhs)
    }
}

impl Mul for &BigUint {
    type Output = BigUint;
    fn mul(self, rhs: Self) -> Self::Output {
        BigUint::mul(self, rhs)
    }
}

impl Div for BigUint {
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        self.div_rem(&rhs).0
    }
}

impl Div<&Self> for BigUint {
    type Output = Self;
    fn div(self, rhs: &Self) -> Self::Output {
        self.div_rem(rhs).0
    }
}

impl Div for &BigUint {
    type Output = BigUint;
    fn div(self, rhs: Self) -> Self::Output {
        self.div_rem(rhs).0
    }
}

impl Rem for BigUint {
    type Output = Self;
    fn rem(self, rhs: Self) -> Self::Output {
        self.div_rem(&rhs).1
    }
}

impl Rem<&Self> for BigUint {
    type Output = Self;
    fn rem(self, rhs: &Self) -> Self::Output {
        self.div_rem(rhs).1
    }
}

impl Rem for &BigUint {
    type Output = BigUint;
    fn rem(self, rhs: Self) -> Self::Output {
        self.div_rem(rhs).1
    }
}

#[cfg(test)]
mod tests {
    use super::BigUint;

    #[test]
    fn round_trip_small_decimals() {
        for s in ["0", "1", "12345", "9999999999999999999999"] {
            let n = BigUint::from_str_radix(s, 10).unwrap();
            assert_eq!(n.to_string(), s);
        }
    }

    #[test]
    fn u64_max_plus_one() {
        let n = BigUint::from(u64::MAX) + BigUint::from(1u64);
        assert_eq!(n.to_string(), "18446744073709551616");
    }

    #[test]
    fn factorial_ten() {
        let mut acc = BigUint::from(1u64);
        for i in 1u64..=10 {
            acc = acc * BigUint::from(i);
        }
        assert_eq!(acc.to_string(), "3628800");
    }

    #[test]
    fn factorial_fifty() {
        let mut acc = BigUint::from(1u64);
        for i in 1u64..=50 {
            acc = acc * BigUint::from(i);
        }
        assert_eq!(
            acc.to_string(),
            "30414093201713378043612608166064768844377641568960512000000000000"
        );
    }

    #[test]
    fn two_to_the_two_fifty_six() {
        let mut acc = BigUint::from(1u64);
        let two = BigUint::from(2u64);
        for _ in 0..256 {
            acc = acc * &two;
        }
        assert_eq!(
            acc.to_string(),
            "115792089237316195423570985008687907853269984665640564039457584007913129639936"
        );
    }

    #[test]
    fn subtraction_underflow_saturates() {
        let a = BigUint::from(5u64);
        let b = BigUint::from(10u64);
        let c = a - b;
        assert!(c.is_zero());
        assert_eq!(c.to_string(), "0");
    }

    #[test]
    fn division_large_by_small() {
        // 10! = 3628800, 3628800 / 7 = 518400
        let mut acc = BigUint::from(1u64);
        for i in 1u64..=10 {
            acc = acc * BigUint::from(i);
        }
        let (q, r) = acc.div_rem_u32(7);
        assert_eq!(q.to_string(), "518400");
        assert_eq!(r, 0);
        // 50! / 13 has known remainder check via cross product.
        let mut fifty = BigUint::from(1u64);
        for i in 1u64..=50 {
            fifty = fifty * BigUint::from(i);
        }
        let divisor = BigUint::from(13u64);
        let (q, r) = fifty.div_rem(&divisor);
        let reconstructed = q * &divisor + r;
        assert_eq!(reconstructed.to_string(), fifty.to_string());
    }

    #[test]
    fn division_large_by_large() {
        // (2^256) / (2^128) = 2^128
        let mut p256 = BigUint::from(1u64);
        let two = BigUint::from(2u64);
        for _ in 0..256 {
            p256 = p256 * &two;
        }
        let mut p128 = BigUint::from(1u64);
        for _ in 0..128 {
            p128 = p128 * &two;
        }
        let (q, r) = p256.div_rem(&p128);
        assert_eq!(q, p128);
        assert!(r.is_zero());

        // (2^256 + 12345) / (2^128) gives quotient 2^128, remainder 12345.
        let target = &p256 + &BigUint::from(12345u64);
        let (q, r) = target.div_rem(&p128);
        assert_eq!(q, p128);
        assert_eq!(r.to_string(), "12345");
    }

    #[test]
    fn hex_round_trip() {
        let s = "deadbeefcafebabe";
        let n = BigUint::from_str_radix(s, 16).unwrap();
        assert_eq!(n.to_string_radix(16), s);
        assert_eq!(n.to_string(), "16045690984503098046");
    }

    #[test]
    fn binary_round_trip() {
        let s = "1011010101110010110";
        let n = BigUint::from_str_radix(s, 2).unwrap();
        assert_eq!(n.to_string_radix(2), s);
    }

    #[test]
    fn parse_rejects_bad_input() {
        assert!(BigUint::from_str_radix("", 10).is_none());
        assert!(BigUint::from_str_radix("12x", 10).is_none());
        // '2' is not valid in base 2.
        assert!(BigUint::from_str_radix("102", 2).is_none());
        // Radix out of range.
        assert!(BigUint::from_str_radix("10", 1).is_none());
        assert!(BigUint::from_str_radix("10", 17).is_none());
    }

    #[test]
    fn ordering_and_equality() {
        let a = BigUint::from(100u64);
        let b = BigUint::from(200u64);
        let c = BigUint::from(100u64);
        assert!(a < b);
        assert!(b > a);
        assert_eq!(a, c);
        assert_ne!(a, b);
        let big = BigUint::from_str_radix("100000000000000000000", 10).unwrap();
        assert!(big > b);
    }

    #[test]
    fn hash_is_consistent() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(BigUint::from(42u64));
        set.insert(BigUint::from_u64(42));
        assert_eq!(set.len(), 1);
        set.insert(BigUint::from(43u64));
        assert_eq!(set.len(), 2);
    }

    #[quickcheck_macros::quickcheck]
    fn qc_add_matches_u128(a: u64, b: u64) -> bool {
        let sum = u128::from(a) + u128::from(b);
        let big = BigUint::from(a) + BigUint::from(b);
        big.to_string() == sum.to_string()
    }

    #[quickcheck_macros::quickcheck]
    fn qc_mul_matches_u128(a: u64, b: u64) -> bool {
        let prod = u128::from(a) * u128::from(b);
        let big = BigUint::from(a) * BigUint::from(b);
        big.to_string() == prod.to_string()
    }

    #[quickcheck_macros::quickcheck]
    fn qc_sub_when_a_ge_b(a: u64, b: u64) -> bool {
        let (lo, hi) = if a >= b { (b, a) } else { (a, b) };
        let diff = hi - lo;
        let big = BigUint::from(hi) - BigUint::from(lo);
        big.to_string() == diff.to_string()
    }

    #[quickcheck_macros::quickcheck]
    fn qc_sub_underflow_saturates(a: u64, b: u64) -> bool {
        if a >= b {
            return true;
        }
        (BigUint::from(a) - BigUint::from(b)).is_zero()
    }

    #[quickcheck_macros::quickcheck]
    fn qc_div_rem_round_trip(a: u64, b: u64) -> bool {
        if b == 0 {
            return true;
        }
        let big_a = BigUint::from(a);
        let big_b = BigUint::from(b);
        let (q, r) = big_a.div_rem(&big_b);
        let recon = q * &big_b + r;
        recon == BigUint::from(a)
    }

    #[quickcheck_macros::quickcheck]
    fn qc_decimal_round_trip(a: u64) -> bool {
        let s = BigUint::from(a).to_string();
        s == a.to_string()
    }
}
