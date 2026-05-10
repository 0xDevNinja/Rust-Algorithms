//! Bit reversal of fixed-width unsigned integers.
//!
//! This module computes the bit-reversed value of a `u32` or `u64` using
//! the classic *parallel-prefix* (a.k.a. *butterfly*) technique. The
//! input is treated as a packed vector of 1-bit lanes, and adjacent
//! lanes are swapped at successively coarser granularities:
//!
//! - swap odd / even bits      (1-bit lanes), mask `0x5555…`
//! - swap adjacent 2-bit lanes,                mask `0x3333…`
//! - swap adjacent nibbles    (4-bit lanes),   mask `0x0F0F…`
//! - swap adjacent bytes      (8-bit lanes),   mask `0x00FF…`
//! - swap adjacent 16-bit lanes,               mask `0x0000FFFF…`
//! - (`u64` only) swap the two 32-bit halves.
//!
//! After all `log2(W)` passes every bit has migrated to position
//! `W - 1 - i`, which is exactly the reverse permutation.
//!
//! Rust's standard library exposes [`u32::reverse_bits`] and
//! [`u64::reverse_bits`] which already do this (often via a single
//! hardware instruction on modern CPUs). We deliberately re-implement
//! the algorithm here so the masks and shift counts are explicit; the
//! tests cross-check against the standard-library version.
//!
//! # Complexity
//!
//! Both routines run in O(1) time and O(1) space — `log2(W)` mask /
//! shift / OR triples, where `W` is the width in bits.

/// Reverse the bit order of `x`, returning a `u32` whose bit `i` is the
/// original bit `31 - i` of `x`.
///
/// Implemented as five parallel-prefix swaps on lane widths
/// 1, 2, 4, 8, 16 — see the module-level docs.
///
/// # Complexity
///
/// O(1) — five mask-and-shift rounds, no loops, no branches.
///
/// # Examples
///
/// ```
/// use rust_algorithms::bit_manipulation::bit_reversal::reverse_u32;
///
/// assert_eq!(reverse_u32(0), 0);
/// assert_eq!(reverse_u32(1), 1u32 << 31);
/// assert_eq!(reverse_u32(0xFFFF_FFFF), 0xFFFF_FFFF);
/// // Double reversal is the identity.
/// assert_eq!(reverse_u32(reverse_u32(0xDEAD_BEEF)), 0xDEAD_BEEF);
/// ```
#[inline]
#[must_use]
pub const fn reverse_u32(mut x: u32) -> u32 {
    // Swap odd / even single bits.
    x = ((x & 0x5555_5555) << 1) | ((x & 0xAAAA_AAAA) >> 1);
    // Swap adjacent 2-bit lanes.
    x = ((x & 0x3333_3333) << 2) | ((x & 0xCCCC_CCCC) >> 2);
    // Swap adjacent 4-bit lanes (nibbles).
    x = ((x & 0x0F0F_0F0F) << 4) | ((x & 0xF0F0_F0F0) >> 4);
    // Swap adjacent 8-bit lanes (bytes).
    x = ((x & 0x00FF_00FF) << 8) | ((x & 0xFF00_FF00) >> 8);
    // Swap the two 16-bit halves. This is mathematically a rotate by
    // half the width; we spell it that way both to satisfy
    // `clippy::manual_rotate` and because on most ISAs it lowers to a
    // single instruction.
    x.rotate_left(16)
}

/// Reverse the bit order of `x`, returning a `u64` whose bit `i` is the
/// original bit `63 - i` of `x`.
///
/// Implemented as six parallel-prefix swaps on lane widths
/// 1, 2, 4, 8, 16, 32 — see the module-level docs.
///
/// # Complexity
///
/// O(1) — six mask-and-shift rounds, no loops, no branches.
///
/// # Examples
///
/// ```
/// use rust_algorithms::bit_manipulation::bit_reversal::reverse_u64;
///
/// assert_eq!(reverse_u64(0), 0);
/// assert_eq!(reverse_u64(1), 1u64 << 63);
/// assert_eq!(reverse_u64(u64::MAX), u64::MAX);
/// // Double reversal is the identity.
/// assert_eq!(reverse_u64(reverse_u64(0x0123_4567_89AB_CDEF)), 0x0123_4567_89AB_CDEF);
/// ```
#[inline]
#[must_use]
pub const fn reverse_u64(mut x: u64) -> u64 {
    // Swap odd / even single bits.
    x = ((x & 0x5555_5555_5555_5555) << 1) | ((x & 0xAAAA_AAAA_AAAA_AAAA) >> 1);
    // Swap adjacent 2-bit lanes.
    x = ((x & 0x3333_3333_3333_3333) << 2) | ((x & 0xCCCC_CCCC_CCCC_CCCC) >> 2);
    // Swap adjacent 4-bit lanes (nibbles).
    x = ((x & 0x0F0F_0F0F_0F0F_0F0F) << 4) | ((x & 0xF0F0_F0F0_F0F0_F0F0) >> 4);
    // Swap adjacent 8-bit lanes (bytes).
    x = ((x & 0x00FF_00FF_00FF_00FF) << 8) | ((x & 0xFF00_FF00_FF00_FF00) >> 8);
    // Swap adjacent 16-bit lanes.
    x = ((x & 0x0000_FFFF_0000_FFFF) << 16) | ((x & 0xFFFF_0000_FFFF_0000) >> 16);
    // Swap the two 32-bit halves. Spelt as a rotate so clippy's
    // `manual_rotate` lint is happy and the compiler emits a single
    // rotate instruction on most ISAs.
    x.rotate_left(32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use quickcheck_macros::quickcheck;

    // ----- reverse_u32 -----

    #[test]
    fn reverse_u32_zero_is_zero() {
        assert_eq!(reverse_u32(0), 0);
    }

    #[test]
    fn reverse_u32_one_is_high_bit() {
        assert_eq!(reverse_u32(1), 1u32 << 31);
    }

    #[test]
    fn reverse_u32_high_bit_is_one() {
        assert_eq!(reverse_u32(1u32 << 31), 1);
    }

    #[test]
    fn reverse_u32_all_ones_fixed() {
        assert_eq!(reverse_u32(0xFFFF_FFFF), 0xFFFF_FFFF);
    }

    #[test]
    fn reverse_u32_double_reversal_is_identity_fixed_table() {
        for &x in &[
            0u32,
            1,
            2,
            0xDEAD_BEEF,
            0x1234_5678,
            0xAAAA_AAAA,
            0x5555_5555,
            0xFFFF_FFFF,
            0x8000_0001,
            0x0F0F_0F0F,
        ] {
            assert_eq!(reverse_u32(reverse_u32(x)), x, "x = {x:#010x}");
        }
    }

    #[test]
    fn reverse_u32_matches_std_fixed_table() {
        for &x in &[
            0u32,
            1,
            2,
            3,
            0xDEAD_BEEF,
            0x1234_5678,
            0xAAAA_AAAA,
            0x5555_5555,
            0xFFFF_FFFF,
            0x8000_0001,
            0x0F0F_0F0F,
            0xCAFE_BABE,
            0xFEED_FACE,
        ] {
            assert_eq!(reverse_u32(x), x.reverse_bits(), "x = {x:#010x}");
        }
    }

    // ----- reverse_u64 -----

    #[test]
    fn reverse_u64_zero_is_zero() {
        assert_eq!(reverse_u64(0), 0);
    }

    #[test]
    fn reverse_u64_one_is_high_bit() {
        assert_eq!(reverse_u64(1), 1u64 << 63);
    }

    #[test]
    fn reverse_u64_high_bit_is_one() {
        assert_eq!(reverse_u64(1u64 << 63), 1);
    }

    #[test]
    fn reverse_u64_all_ones_fixed() {
        assert_eq!(reverse_u64(u64::MAX), u64::MAX);
        assert_eq!(reverse_u64(0xFFFF_FFFF_FFFF_FFFF), 0xFFFF_FFFF_FFFF_FFFF);
    }

    #[test]
    fn reverse_u64_double_reversal_is_identity_fixed_table() {
        for &x in &[
            0u64,
            1,
            2,
            0x0123_4567_89AB_CDEF,
            0xDEAD_BEEF_CAFE_BABE,
            0xAAAA_AAAA_AAAA_AAAA,
            0x5555_5555_5555_5555,
            u64::MAX,
            0x8000_0000_0000_0001,
            0x0F0F_0F0F_0F0F_0F0F,
        ] {
            assert_eq!(reverse_u64(reverse_u64(x)), x, "x = {x:#018x}");
        }
    }

    #[test]
    fn reverse_u64_matches_std_fixed_table() {
        for &x in &[
            0u64,
            1,
            2,
            3,
            0x0123_4567_89AB_CDEF,
            0xDEAD_BEEF_CAFE_BABE,
            0xAAAA_AAAA_AAAA_AAAA,
            0x5555_5555_5555_5555,
            u64::MAX,
            0x8000_0000_0000_0001,
            0x0F0F_0F0F_0F0F_0F0F,
            0xFEED_FACE_DEAD_BEEF,
        ] {
            assert_eq!(reverse_u64(x), x.reverse_bits(), "x = {x:#018x}");
        }
    }

    #[test]
    fn reverse_u64_low_u32_relation() {
        // Reversing a value whose top 32 bits are zero places those
        // 32 bits of payload into the *high* half of the result, in the
        // same order as `reverse_u32` would produce.
        for &x in &[0u32, 1, 0xDEAD_BEEF, 0x1234_5678, 0xFFFF_FFFF] {
            let lifted = u64::from(x);
            let expected = u64::from(reverse_u32(x)) << 32;
            assert_eq!(reverse_u64(lifted), expected, "x = {x:#010x}");
        }
    }

    // ----- property tests -----

    #[quickcheck]
    fn qc_reverse_u32_matches_std(x: u32) -> bool {
        reverse_u32(x) == x.reverse_bits()
    }

    #[quickcheck]
    fn qc_reverse_u32_double_is_identity(x: u32) -> bool {
        reverse_u32(reverse_u32(x)) == x
    }

    #[quickcheck]
    fn qc_reverse_u32_preserves_popcount(x: u32) -> bool {
        reverse_u32(x).count_ones() == x.count_ones()
    }

    #[quickcheck]
    fn qc_reverse_u64_matches_std(x: u64) -> bool {
        reverse_u64(x) == x.reverse_bits()
    }

    #[quickcheck]
    fn qc_reverse_u64_double_is_identity(x: u64) -> bool {
        reverse_u64(reverse_u64(x)) == x
    }

    #[quickcheck]
    fn qc_reverse_u64_preserves_popcount(x: u64) -> bool {
        reverse_u64(x).count_ones() == x.count_ones()
    }
}
