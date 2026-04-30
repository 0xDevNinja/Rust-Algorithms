//! Subset enumeration utilities.
//!
//! This module provides two iterator-producing helpers commonly used in
//! competitive programming and combinatorial search:
//!
//! 1. [`subsets`] — enumerate every submask of a given mask using the
//!    classic identity `s = (s - 1) & mask`. Starting from `s = mask`,
//!    repeatedly applying this update walks every submask exactly once
//!    in strictly *descending* numeric order, terminating when `s`
//!    underflows after `0`.
//!
//! 2. [`k_subsets_of_n`] — enumerate every bitmask with exactly `k`
//!    bits set, drawn from the `n` lowest bits, using Gosper's hack:
//!    given a value with `k` bits set, the next-larger value with the
//!    same popcount is computed in O(1) via
//!    `t = x | (x - 1); next = (t + 1) | (((!t & -!t) - 1) >> (ctz(x) + 1))`.
//!    Iteration is in strictly *ascending* numeric order.
//!
//! ## Complexity
//!
//! Both iterators are *output-sensitive*:
//!
//! - [`subsets(mask)`](subsets) yields `2^popcount(mask)` items, each in
//!   amortised O(1) time and O(1) auxiliary space.
//! - [`k_subsets_of_n(n, k)`](k_subsets_of_n) yields `C(n, k)` items,
//!   each in O(1) time and O(1) auxiliary space.
//!
//! ## Preconditions
//!
//! - [`k_subsets_of_n`] requires `k <= n`. When `k > n`, the iterator is
//!   empty (no `n`-bit mask has more set bits than `n`). When
//!   `k == 0`, the iterator yields `0` exactly once. Both `n` and `k`
//!   must be `<= 32` to fit in a `u32`; for `n == 32` and `k == 32` the
//!   single yielded value is `u32::MAX`.

/// Iterator yielding every submask of `mask` in strictly descending order.
///
/// Implements the textbook identity `s = (s - 1) & mask`. Starting at
/// `s = mask`, this walks every value `s'` such that `(s' & mask) == s'`,
/// finishing with `0` and then terminating.
///
/// The iterator yields exactly `2^mask.count_ones()` items, including
/// both `mask` itself and `0`.
///
/// # Complexity
///
/// O(1) per item, O(1) auxiliary space. Total work is
/// O(`2^popcount(mask)`), i.e. linear in the output size.
///
/// # Examples
///
/// ```
/// use rust_algorithms::bit_manipulation::subset_enumeration::subsets;
///
/// let got: Vec<u32> = subsets(0b101).collect();
/// assert_eq!(got, vec![0b101, 0b100, 0b001, 0]);
/// ```
#[inline]
#[must_use]
pub const fn subsets(mask: u32) -> Subsets {
    Subsets {
        mask,
        next: Some(mask),
    }
}

/// Iterator returned by [`subsets`].
#[derive(Debug, Clone)]
pub struct Subsets {
    mask: u32,
    /// The next value to yield. `None` once iteration is finished.
    next: Option<u32>,
}

impl Iterator for Subsets {
    type Item = u32;

    fn next(&mut self) -> Option<u32> {
        let cur = self.next?;
        // Compute the successor *before* yielding `cur`. After yielding
        // `0` the iteration must terminate; otherwise compute the next
        // submask via the standard trick.
        self.next = if cur == 0 {
            None
        } else {
            Some((cur - 1) & self.mask)
        };
        Some(cur)
    }
}

/// Iterator yielding every `u32` bitmask with exactly `k` bits set,
/// restricted to the `n` lowest bits, in strictly ascending order.
///
/// Uses Gosper's hack to advance from one valid mask to the next in
/// O(1). The first yielded mask is `(1 << k) - 1` (the `k` lowest bits
/// set); iteration stops once advancing would set a bit at or above
/// position `n`.
///
/// # Edge cases
///
/// - `k == 0` yields `0` exactly once (the empty subset).
/// - `n == 0` and `k == 0` yields `0` exactly once.
/// - `n == 0` and `k > 0` yields nothing.
/// - `k > n` yields nothing.
///
/// # Panics
///
/// Panics if `n > 32` or `k > 32`, since the result must fit in a `u32`.
///
/// # Complexity
///
/// O(1) per item, O(1) auxiliary space. Total work is O(`C(n, k)`),
/// i.e. linear in the output size.
///
/// # Examples
///
/// ```
/// use rust_algorithms::bit_manipulation::subset_enumeration::k_subsets_of_n;
///
/// let got: Vec<u32> = k_subsets_of_n(4, 2).collect();
/// assert_eq!(got, vec![0b0011, 0b0101, 0b0110, 0b1001, 0b1010, 0b1100]);
/// ```
#[inline]
#[must_use]
pub fn k_subsets_of_n(n: u32, k: u32) -> KSubsets {
    assert!(n <= 32, "n must be <= 32 to fit in u32");
    assert!(k <= 32, "k must be <= 32 to fit in u32");

    // Upper bound on the iteration: any valid mask must be strictly
    // less than `1 << n` (or equal to `u32::MAX` when n == 32).
    // We track this as an inclusive bound on `n` for clarity.
    let next = if k > n {
        None
    } else if k == 0 {
        // The empty subset; emitted once regardless of n.
        Some(0)
    } else {
        // (1 << k) - 1 is safe because k <= 32; for k == 32 we use
        // u32::MAX to avoid the shift-overflow.
        Some(if k == 32 { u32::MAX } else { (1u32 << k) - 1 })
    };

    KSubsets { n, k, next }
}

/// Iterator returned by [`k_subsets_of_n`].
#[derive(Debug, Clone)]
pub struct KSubsets {
    n: u32,
    k: u32,
    /// The next value to yield. `None` once iteration is finished.
    next: Option<u32>,
}

impl Iterator for KSubsets {
    type Item = u32;

    fn next(&mut self) -> Option<u32> {
        let cur = self.next?;

        // Compute the successor of `cur` via Gosper's hack and check
        // that it still fits within the n lowest bits.
        self.next = if self.k == 0 {
            // Only the empty subset; no successor.
            None
        } else {
            // Gosper's hack on u64 to dodge intermediate u32 overflow
            // when cur has bit 31 set.
            let x = u64::from(cur);
            let c = x & x.wrapping_neg();
            let r = x + c;
            // r ^ x has the low run of 1s plus the bit that just
            // toggled; dividing by c (a power of two) right-shifts it
            // appropriately, then >> 2 lines it up at the bottom.
            let next64 = r | (((x ^ r) / c) >> 2);

            // Bound check: must fit in the n lowest bits.
            // For n == 32 the bound is 1 << 32, computed in u64.
            let bound: u64 = 1u64 << self.n;
            if next64 < bound {
                Some(next64 as u32)
            } else {
                None
            }
        };

        Some(cur)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quickcheck_macros::quickcheck;

    // ----- subsets -----

    #[test]
    fn subsets_of_zero_yields_only_zero() {
        let got: Vec<u32> = subsets(0).collect();
        assert_eq!(got, vec![0]);
    }

    #[test]
    fn subsets_of_single_bit() {
        let got: Vec<u32> = subsets(0b100).collect();
        assert_eq!(got, vec![0b100, 0]);
    }

    #[test]
    fn subsets_of_two_bits_descending() {
        let got: Vec<u32> = subsets(0b101).collect();
        assert_eq!(got, vec![0b101, 0b100, 0b001, 0]);
    }

    #[test]
    fn subsets_of_three_bits_full_set() {
        // mask = 0b111 should enumerate 0..=7 in descending order.
        let got: Vec<u32> = subsets(0b111).collect();
        assert_eq!(got, vec![7, 6, 5, 4, 3, 2, 1, 0]);
        assert_eq!(got.len(), 8);
    }

    #[test]
    fn subsets_count_matches_two_to_popcount() {
        for mask in 0u32..256 {
            let count = subsets(mask).count();
            assert_eq!(count, 1usize << mask.count_ones(), "mask = {mask:#b}");
        }
    }

    #[test]
    fn subsets_descending_and_unique() {
        let mask: u32 = 0b1011_0101;
        let got: Vec<u32> = subsets(mask).collect();
        // Strictly descending.
        for w in got.windows(2) {
            assert!(w[0] > w[1], "not descending: {} -> {}", w[0], w[1]);
        }
        // All distinct (implied by strictly descending) and submasks.
        for &s in &got {
            assert_eq!(s & mask, s, "{s:#b} is not a submask of {mask:#b}");
        }
        assert_eq!(got.len(), 1usize << mask.count_ones());
    }

    // ----- k_subsets_of_n -----

    #[test]
    fn k_subsets_4_choose_2() {
        let got: Vec<u32> = k_subsets_of_n(4, 2).collect();
        assert_eq!(got, vec![0b0011, 0b0101, 0b0110, 0b1001, 0b1010, 0b1100]);
    }

    #[test]
    fn k_subsets_n_5_k_0_yields_zero_once() {
        let got: Vec<u32> = k_subsets_of_n(5, 0).collect();
        assert_eq!(got, vec![0]);
    }

    #[test]
    fn k_subsets_n_0_k_0_yields_zero_once() {
        let got: Vec<u32> = k_subsets_of_n(0, 0).collect();
        assert_eq!(got, vec![0]);
    }

    #[test]
    fn k_subsets_n_0_k_positive_yields_nothing() {
        assert!(k_subsets_of_n(0, 1).next().is_none());
    }

    #[test]
    fn k_subsets_k_greater_than_n_yields_nothing() {
        assert!(k_subsets_of_n(3, 5).next().is_none());
    }

    #[test]
    fn k_subsets_k_equals_n_yields_full_mask() {
        let got: Vec<u32> = k_subsets_of_n(5, 5).collect();
        assert_eq!(got, vec![0b1_1111]);
    }

    #[test]
    fn k_subsets_count_matches_binomial() {
        // C(n, k) for n in 0..=8.
        let binom = |n: u32, k: u32| -> u64 {
            if k > n {
                return 0;
            }
            let k = k.min(n - k);
            let mut acc: u64 = 1;
            for i in 0..k {
                acc = acc * u64::from(n - i) / u64::from(i + 1);
            }
            acc
        };
        for n in 0u32..=8 {
            for k in 0u32..=8 {
                let got: Vec<u32> = k_subsets_of_n(n, k).collect();
                assert_eq!(
                    got.len() as u64,
                    binom(n, k),
                    "C({n}, {k}) mismatch: got {got:?}"
                );
            }
        }
    }

    #[test]
    fn k_subsets_ascending_and_within_bounds() {
        let n = 6;
        let k = 3;
        let got: Vec<u32> = k_subsets_of_n(n, k).collect();
        for w in got.windows(2) {
            assert!(w[0] < w[1], "not ascending: {} -> {}", w[0], w[1]);
        }
        for &m in &got {
            assert_eq!(m.count_ones(), k, "popcount mismatch on {m:#b}");
            assert!(m < (1u32 << n), "mask {m:#b} exceeds n={n} bits");
        }
    }

    // ----- property tests -----

    #[quickcheck]
    fn qc_subsets_count_matches_popcount(m: u8) -> bool {
        let mask = u32::from(m);
        subsets(mask).count() == 1usize << mask.count_ones()
    }

    #[quickcheck]
    fn qc_subsets_are_actual_submasks(m: u8) -> bool {
        let mask = u32::from(m);
        subsets(mask).all(|s| (s & mask) == s)
    }

    #[quickcheck]
    fn qc_subsets_strictly_descending(m: u8) -> bool {
        let mask = u32::from(m);
        let v: Vec<u32> = subsets(mask).collect();
        v.windows(2).all(|w| w[0] > w[1])
    }

    #[quickcheck]
    fn qc_k_subsets_have_correct_popcount(nk: (u8, u8)) -> bool {
        // Restrict to small n to keep iteration cheap.
        let n = u32::from(nk.0 % 9); // 0..=8
        let k = u32::from(nk.1 % 9);
        k_subsets_of_n(n, k).all(|m| m.count_ones() == k)
    }

    #[quickcheck]
    fn qc_k_subsets_within_n_bits(nk: (u8, u8)) -> bool {
        let n = u32::from(nk.0 % 9);
        let k = u32::from(nk.1 % 9);
        // For n == 0 the only legal mask is 0, which trivially has all
        // bits within "the 0 lowest bits" (none).
        let bound: u64 = 1u64 << n;
        k_subsets_of_n(n, k).all(|m| u64::from(m) < bound)
    }

    #[quickcheck]
    fn qc_k_subsets_strictly_ascending(nk: (u8, u8)) -> bool {
        let n = u32::from(nk.0 % 9);
        let k = u32::from(nk.1 % 9);
        let v: Vec<u32> = k_subsets_of_n(n, k).collect();
        v.windows(2).all(|w| w[0] < w[1])
    }
}
