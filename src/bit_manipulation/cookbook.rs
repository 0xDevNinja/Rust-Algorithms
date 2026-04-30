//! Bit-manipulation cookbook: a small library of well-known bit tricks.
//!
//! Each helper is a `pub const fn` so it can be evaluated in `const`
//! contexts (array sizes, other `const fn` bodies, etc.).
//!
//! Reference: each entry is a textbook idiom. The list below names the
//! helper and a one-line use case.
//!
//! - [`count_set_bits`]   — Hamming weight / population count.
//! - [`is_power_of_two`]  — fast `x == 2^k` check (also useful for alignment).
//! - [`next_power_of_two`] — round `x` up to the next power of two (capacity grow).
//! - [`lowest_set_bit`]   — isolate the least-significant 1 bit (BIT scan / Fenwick).
//! - [`clear_lowest_set_bit`] — drop the least-significant 1 bit (Brian Kernighan loop).
//! - [`parity`]           — odd/even popcount (parity bit / XOR-based hashing).
//! - [`swap_bits`]        — swap two bits at given positions (permutation networks).

/// Returns the number of set bits (Hamming weight) in `x`.
///
/// Delegates to [`u64::count_ones`] because `count_ones` lowers to the
/// dedicated `popcnt` instruction on targets that support it, which is
/// strictly faster than any portable manual implementation.
#[inline]
#[must_use]
pub const fn count_set_bits(x: u64) -> u32 {
    x.count_ones()
}

/// Returns `true` iff `x` is a (non-zero) power of two.
///
/// `0` is *not* a power of two. The trick relies on the fact that powers
/// of two have exactly one bit set, so subtracting 1 flips all lower bits
/// and the AND becomes 0.
///
/// We hand-roll the check (rather than calling [`u64::is_power_of_two`])
/// so this stays a `const fn` on stable Rust.
#[inline]
#[must_use]
#[allow(clippy::manual_is_power_of_two)]
pub const fn is_power_of_two(x: u64) -> bool {
    x != 0 && x & (x - 1) == 0
}

/// Smallest power of two greater than or equal to `x`.
///
/// `next_power_of_two(0) == 1` by convention. If the result would
/// overflow (i.e. `x > 2^63`), this returns `0`.
#[inline]
#[must_use]
pub const fn next_power_of_two(x: u64) -> u64 {
    if x <= 1 {
        return 1;
    }
    // For x in [2, 2^63], the answer is 1 << (64 - (x - 1).leading_zeros()).
    // For x > 2^63 the next power of two is 2^64 which does not fit in u64,
    // so we follow `checked_next_power_of_two` and return 0.
    let lz = (x - 1).leading_zeros();
    if lz == 0 {
        0
    } else {
        1u64 << (64 - lz)
    }
}

/// Returns the value of the lowest set bit of `x`, or `0` when `x == 0`.
///
/// Uses the classic `x & -x` idiom (with `wrapping_neg` for unsigned
/// arithmetic) to isolate the least-significant 1 bit.
#[inline]
#[must_use]
pub const fn lowest_set_bit(x: u64) -> u64 {
    x & x.wrapping_neg()
}

/// Returns `x` with its lowest set bit cleared. Returns `0` when `x == 0`.
///
/// This is the inner step of Brian Kernighan's popcount loop:
/// repeatedly clearing the lowest set bit lets you iterate over set bits
/// in O(popcount(x)) time.
#[inline]
#[must_use]
pub const fn clear_lowest_set_bit(x: u64) -> u64 {
    // For x == 0 this wraps to u64::MAX; AND with 0 yields 0, which is
    // the desired identity behaviour.
    x & x.wrapping_sub(1)
}

/// Returns `true` iff `x` has an odd number of set bits.
///
/// Equivalent to XOR-folding all bits of `x` into a single bit.
#[inline]
#[must_use]
pub const fn parity(x: u64) -> bool {
    x.count_ones() % 2 == 1
}

/// Swap the bits at positions `i` and `j` in `x`.
///
/// Both `i` and `j` must be `< 64`. If the two bits are equal, `x` is
/// returned unchanged; otherwise both positions are toggled with a
/// single XOR mask.
#[inline]
#[must_use]
pub const fn swap_bits(x: u64, i: u32, j: u32) -> u64 {
    debug_assert!(i < 64 && j < 64);
    let bi = (x >> i) & 1;
    let bj = (x >> j) & 1;
    if bi == bj {
        x
    } else {
        x ^ ((1u64 << i) | (1u64 << j))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quickcheck_macros::quickcheck;

    #[test]
    fn count_set_bits_known_values() {
        assert_eq!(count_set_bits(0), 0);
        assert_eq!(count_set_bits(1), 1);
        assert_eq!(count_set_bits(255), 8);
        assert_eq!(count_set_bits(0xFFFF_FFFF_FFFF_FFFF), 64);
        assert_eq!(count_set_bits(0xAAAA_AAAA_AAAA_AAAA), 32);
    }

    #[test]
    fn is_power_of_two_known_values() {
        assert!(!is_power_of_two(0));
        assert!(is_power_of_two(1));
        assert!(is_power_of_two(2));
        assert!(!is_power_of_two(3));
        assert!(is_power_of_two(1024));
        assert!(!is_power_of_two(1023));
        assert!(is_power_of_two(1u64 << 63));
        assert!(!is_power_of_two(u64::MAX));
    }

    #[test]
    fn next_power_of_two_known_values() {
        assert_eq!(next_power_of_two(0), 1);
        assert_eq!(next_power_of_two(1), 1);
        assert_eq!(next_power_of_two(2), 2);
        assert_eq!(next_power_of_two(3), 4);
        assert_eq!(next_power_of_two(5), 8);
        assert_eq!(next_power_of_two(1023), 1024);
        assert_eq!(next_power_of_two(1024), 1024);
        assert_eq!(next_power_of_two(1u64 << 63), 1u64 << 63);
        // Overflow: > 2^63 has no representable next power of two.
        assert_eq!(next_power_of_two((1u64 << 63) + 1), 0);
    }

    #[test]
    fn lowest_set_bit_known_values() {
        assert_eq!(lowest_set_bit(0), 0);
        assert_eq!(lowest_set_bit(1), 1);
        assert_eq!(lowest_set_bit(2), 2);
        assert_eq!(lowest_set_bit(12), 4); // 0b1100 -> 0b0100
        assert_eq!(lowest_set_bit(0b1010_0000), 0b0010_0000);
        assert_eq!(lowest_set_bit(1u64 << 63), 1u64 << 63);
    }

    #[test]
    fn clear_lowest_set_bit_known_values() {
        assert_eq!(clear_lowest_set_bit(0), 0);
        assert_eq!(clear_lowest_set_bit(1), 0);
        assert_eq!(clear_lowest_set_bit(12), 8); // 0b1100 -> 0b1000
        assert_eq!(clear_lowest_set_bit(0b1011), 0b1010);
        assert_eq!(clear_lowest_set_bit(0xFF), 0xFE);
    }

    #[test]
    fn parity_known_values() {
        assert!(!parity(0));
        assert!(parity(1));
        assert!(parity(7)); // three set bits
        assert!(!parity(0b1010));
        assert!(!parity(0xFFFF_FFFF_FFFF_FFFF)); // 64 set bits
        assert!(parity(0x7FFF_FFFF_FFFF_FFFF)); // 63 set bits
    }

    #[test]
    fn swap_bits_known_values() {
        assert_eq!(swap_bits(0b1010, 0, 1), 0b1001);
        assert_eq!(swap_bits(0b1010, 1, 3), 0b1010); // both 1 -> unchanged
        assert_eq!(swap_bits(0b1010, 0, 2), 0b1010); // both 0 -> unchanged
        assert_eq!(swap_bits(0b1010, 1, 2), 0b1100);
        assert_eq!(swap_bits(0, 5, 7), 0);
        assert_eq!(swap_bits(1u64 << 63, 0, 63), 1);
    }

    #[test]
    fn swap_bits_is_involution() {
        let x = 0xDEAD_BEEF_1234_5678_u64;
        assert_eq!(swap_bits(swap_bits(x, 7, 42), 7, 42), x);
    }

    // ----- property tests -----

    #[quickcheck]
    fn qc_parity_matches_count_ones(x: u64) -> bool {
        parity(x) == (x.count_ones() % 2 == 1)
    }

    #[quickcheck]
    fn qc_pow2_iff_next_pow2_fixed_point(x: u64) -> bool {
        if x == 0 {
            return true; // skip the degenerate case
        }
        is_power_of_two(x) == (next_power_of_two(x) == x)
    }

    #[quickcheck]
    fn qc_clear_or_lowest_recovers_x(x: u64) -> bool {
        if x == 0 {
            return true;
        }
        clear_lowest_set_bit(x) | lowest_set_bit(x) == x
    }

    #[quickcheck]
    fn qc_count_set_bits_matches_count_ones(x: u64) -> bool {
        count_set_bits(x) == x.count_ones()
    }

    #[quickcheck]
    fn qc_lowest_set_bit_is_power_of_two(x: u64) -> bool {
        let l = lowest_set_bit(x);
        l == 0 || is_power_of_two(l)
    }

    #[quickcheck]
    fn qc_next_power_of_two_is_ge(x: u64) -> bool {
        // Skip values where the answer would overflow.
        if x > (1u64 << 63) {
            return true;
        }
        let n = next_power_of_two(x);
        n >= x && (n == 1 || is_power_of_two(n))
    }
}
