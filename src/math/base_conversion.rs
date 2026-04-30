//! Arbitrary-base conversion between `u64` integers and their string
//! representation. Supports radices `2..=36`. The digit alphabet is
//! `0-9` followed by `a-z` (lowercase on output); parsing accepts both
//! upper and lower case ASCII alphanumerics. Each call runs in
//! `O(log_base n)` time.

const ALPHABET: &[u8; 36] = b"0123456789abcdefghijklmnopqrstuvwxyz";

/// Converts `n` into its string representation in `base`.
///
/// Returns `None` if `base` is outside `2..=36`. The case `n == 0` always
/// yields `"0"`.
pub fn to_base(n: u64, base: u32) -> Option<String> {
    if !(2..=36).contains(&base) {
        return None;
    }
    if n == 0 {
        return Some("0".to_string());
    }
    let base = base as u64;
    let mut buf = Vec::with_capacity(64);
    let mut n = n;
    while n > 0 {
        let digit = (n % base) as usize;
        buf.push(ALPHABET[digit]);
        n /= base;
    }
    buf.reverse();
    // SAFETY: every byte pushed comes from `ALPHABET`, which is ASCII.
    Some(String::from_utf8(buf).expect("alphabet is ASCII"))
}

/// Parses `s` as a non-negative integer in `base`.
///
/// Accepts both upper and lower case ASCII alphanumerics as digits.
/// Returns `None` if `base` is outside `2..=36`, if `s` is empty, if any
/// character is not a valid digit for the given base, or if the result
/// overflows `u64`.
pub fn from_base(s: &str, base: u32) -> Option<u64> {
    if !(2..=36).contains(&base) {
        return None;
    }
    if s.is_empty() {
        return None;
    }
    let base_u64 = base as u64;
    let mut acc: u64 = 0;
    for c in s.bytes() {
        let digit = match c {
            b'0'..=b'9' => (c - b'0') as u32,
            b'a'..=b'z' => (c - b'a') as u32 + 10,
            b'A'..=b'Z' => (c - b'A') as u32 + 10,
            _ => return None,
        };
        if digit >= base {
            return None;
        }
        acc = acc.checked_mul(base_u64)?.checked_add(digit as u64)?;
    }
    Some(acc)
}

#[cfg(test)]
mod tests {
    use super::{from_base, to_base};

    #[test]
    fn to_base_zero() {
        assert_eq!(to_base(0, 2).as_deref(), Some("0"));
        assert_eq!(to_base(0, 10).as_deref(), Some("0"));
        assert_eq!(to_base(0, 36).as_deref(), Some("0"));
    }

    #[test]
    fn to_base_known_values() {
        assert_eq!(to_base(10, 2).as_deref(), Some("1010"));
        assert_eq!(to_base(255, 16).as_deref(), Some("ff"));
        assert_eq!(to_base(7, 8).as_deref(), Some("7"));
        assert_eq!(to_base(8, 8).as_deref(), Some("10"));
        assert_eq!(to_base(35, 36).as_deref(), Some("z"));
        assert_eq!(to_base(36, 36).as_deref(), Some("10"));
    }

    #[test]
    fn to_base_max_u64_hex() {
        assert_eq!(to_base(u64::MAX, 16).as_deref(), Some("ffffffffffffffff"));
    }

    #[test]
    fn to_base_invalid_base() {
        assert!(to_base(10, 0).is_none());
        assert!(to_base(10, 1).is_none());
        assert!(to_base(10, 37).is_none());
    }

    #[test]
    fn from_base_known_values() {
        assert_eq!(from_base("1010", 2), Some(10));
        assert_eq!(from_base("ff", 16), Some(255));
        assert_eq!(from_base("FF", 16), Some(255));
        assert_eq!(from_base("z", 36), Some(35));
        assert_eq!(from_base("Z", 36), Some(35));
        assert_eq!(from_base("0", 2), Some(0));
    }

    #[test]
    fn from_base_max_u64_hex() {
        assert_eq!(from_base("ffffffffffffffff", 16), Some(u64::MAX));
        assert_eq!(from_base("FFFFFFFFFFFFFFFF", 16), Some(u64::MAX));
    }

    #[test]
    fn from_base_invalid_char() {
        assert!(from_base("12!3", 10).is_none());
        // '2' is not a valid binary digit.
        assert!(from_base("102", 2).is_none());
        // 'g' is not valid in base 16.
        assert!(from_base("g", 16).is_none());
    }

    #[test]
    fn from_base_empty() {
        assert!(from_base("", 10).is_none());
    }

    #[test]
    fn from_base_invalid_base() {
        assert!(from_base("1", 0).is_none());
        assert!(from_base("1", 1).is_none());
        assert!(from_base("1", 37).is_none());
    }

    #[test]
    fn from_base_overflow() {
        // u64::MAX in base 16 has 16 'f's; one more digit overflows.
        assert!(from_base("fffffffffffffffff", 16).is_none());
        // 2^64 = 18446744073709551616 overflows.
        assert!(from_base("18446744073709551616", 10).is_none());
    }

    #[test]
    fn roundtrip_property() {
        // Deterministic LCG; covers a wide spread of u64 values across
        // every supported base. Tests `from_base(to_base(n, b)) == Some(n)`.
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        for _ in 0..2_000 {
            // xorshift64* step
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            let n = state.wrapping_mul(0x2545_F491_4F6C_DD1D);
            // pick a base in 2..=36 from the high bits
            let base = 2 + ((n >> 58) as u32 % 35);
            let s = to_base(n, base).expect("valid base");
            assert_eq!(from_base(&s, base), Some(n), "n={n}, base={base}");
        }
    }

    #[test]
    fn roundtrip_edge_cases() {
        for &n in &[0_u64, 1, 2, u64::MAX - 1, u64::MAX] {
            for base in 2..=36 {
                let s = to_base(n, base).unwrap();
                assert_eq!(from_base(&s, base), Some(n));
            }
        }
    }
}
