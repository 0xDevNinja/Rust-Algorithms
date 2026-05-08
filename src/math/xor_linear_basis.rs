//! XOR linear basis over GF(2).
//!
//! Treats each `u64` as a 64-dimensional vector over GF(2). The basis stores
//! at most one vector per leading bit position, kept in row-reduced form.
//! Standard tool for problems asking about XOR-extremal values, span
//! membership, or counting distinct XOR sums of a multiset.
//!
//! # Complexity
//!
//! For `B` = bit width (= 64) and `n` insertions:
//! - `insert`: `O(B)` per call, `O(n * B)` total.
//! - `contains`: `O(B)`.
//! - `max_xor`, `min_xor`, `rank`: `O(B)`.
//! - Space: `O(B)`.

/// Linear basis of `u64` values under XOR.
///
/// `basis[i]` either holds a vector whose highest set bit is `i`, or `0` if
/// that slot is empty. The stored vectors are linearly independent and span
/// the same subspace as every value ever inserted.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct XorBasis {
    basis: [u64; 64],
}

impl Default for XorBasis {
    fn default() -> Self {
        Self::new()
    }
}

impl XorBasis {
    /// Constructs an empty basis spanning only `{0}`.
    #[must_use]
    pub const fn new() -> Self {
        Self { basis: [0; 64] }
    }

    /// Inserts `x` into the basis using Gaussian elimination on bits.
    ///
    /// Returns `true` if `x` was linearly independent of the current span
    /// (the basis grew), `false` if `x` was already representable.
    pub const fn insert(&mut self, mut x: u64) -> bool {
        while x != 0 {
            let bit = 63 - x.leading_zeros() as usize;
            if self.basis[bit] == 0 {
                self.basis[bit] = x;
                return true;
            }
            x ^= self.basis[bit];
        }
        false
    }

    /// Returns `true` if `x` is in the span of the basis. The zero vector
    /// is always contained.
    #[must_use]
    pub const fn contains(&self, mut x: u64) -> bool {
        while x != 0 {
            let bit = 63 - x.leading_zeros() as usize;
            if self.basis[bit] == 0 {
                return false;
            }
            x ^= self.basis[bit];
        }
        true
    }

    /// Returns the maximum value of `init ^ s` over every `s` in the span
    /// of the basis. Greedy from the highest bit: XOR in any basis vector
    /// that flips the current high bit on.
    #[must_use]
    pub fn max_xor(&self, init: u64) -> u64 {
        let mut best = init;
        for i in (0..64).rev() {
            if self.basis[i] != 0 && (best >> i) & 1 == 0 {
                best ^= self.basis[i];
            }
        }
        best
    }

    /// Returns the minimum non-zero value reachable as a XOR of basis
    /// elements, or `0` if the basis is empty (only the zero vector is
    /// reachable).
    ///
    /// Because the stored basis is in row-echelon form (one leading bit per
    /// slot), the smallest non-zero combination is simply the lowest stored
    /// vector.
    #[must_use]
    pub fn min_xor(&self) -> u64 {
        for i in 0..64 {
            if self.basis[i] != 0 {
                return self.basis[i];
            }
        }
        0
    }

    /// Returns the dimension of the spanned subspace, i.e. the number of
    /// non-zero slots. The span has size `2^rank`.
    #[must_use]
    pub fn rank(&self) -> u32 {
        self.basis.iter().filter(|&&v| v != 0).count() as u32
    }
}

#[cfg(test)]
mod tests {
    use super::XorBasis;
    use quickcheck::TestResult;
    use quickcheck_macros::quickcheck;

    #[test]
    fn empty_basis() {
        let b = XorBasis::new();
        assert_eq!(b.rank(), 0);
        assert_eq!(b.max_xor(0), 0);
        assert_eq!(b.max_xor(42), 42);
        assert!(b.contains(0));
        assert!(!b.contains(1));
        assert!(!b.contains(u64::MAX));
        assert_eq!(b.min_xor(), 0);
    }

    #[test]
    fn one_two_three_has_rank_two() {
        let mut b = XorBasis::new();
        assert!(b.insert(1));
        assert!(b.insert(2));
        // 3 = 1 ^ 2 — already in the span.
        assert!(!b.insert(3));
        assert_eq!(b.rank(), 2);
        assert!(b.contains(0));
        assert!(b.contains(1));
        assert!(b.contains(2));
        assert!(b.contains(3));
        assert!(!b.contains(4));
        assert_eq!(b.max_xor(0), 3);
    }

    #[test]
    fn duplicates_return_false() {
        let mut b = XorBasis::new();
        assert!(b.insert(5));
        assert!(!b.insert(5));
        assert!(!b.insert(0));
        assert_eq!(b.rank(), 1);
    }

    #[test]
    fn max_xor_canonical_three_five_six_seven() {
        // {3,5,6} are linearly dependent (3 ^ 5 = 6) so their span is
        // {0,3,5,6} with maximum 6. Adding 7 lifts the rank to 3 and the
        // span becomes all of [0, 8), giving max = 7 — the canonical CP
        // "maximum subset XOR" answer.
        let mut b = XorBasis::new();
        b.insert(3);
        b.insert(5);
        b.insert(6);
        assert_eq!(b.rank(), 2);
        assert_eq!(b.max_xor(0), 6);

        b.insert(7);
        assert_eq!(b.rank(), 3);
        assert_eq!(b.max_xor(0), 7);
    }

    #[test]
    fn max_xor_classic_nine_eight_five() {
        // Textbook example: maximum subset XOR over {9, 8, 5} is 13 (= 8 ^ 5).
        let mut b = XorBasis::new();
        b.insert(9);
        b.insert(8);
        b.insert(5);
        assert_eq!(b.max_xor(0), 13);
    }

    #[test]
    fn min_xor_after_reduction() {
        // {1,1,2,3}: rank-2 basis; smallest non-zero combination is 1.
        let mut b = XorBasis::new();
        b.insert(1);
        b.insert(1);
        b.insert(2);
        b.insert(3);
        assert_eq!(b.rank(), 2);
        assert_eq!(b.min_xor(), 1);
    }

    #[test]
    fn high_bits_round_trip() {
        let mut b = XorBasis::new();
        b.insert(1u64 << 63);
        b.insert(1u64 << 62);
        assert_eq!(b.rank(), 2);
        assert!(b.contains(0));
        assert!(b.contains((1u64 << 63) ^ (1u64 << 62)));
        assert_eq!(b.max_xor(0), (1u64 << 63) | (1u64 << 62));
    }

    #[test]
    fn max_xor_with_init() {
        // Basis {1, 2}; init = 4 → reachable XORs are {4,5,6,7}; max = 7.
        let mut b = XorBasis::new();
        b.insert(1);
        b.insert(2);
        assert_eq!(b.max_xor(4), 7);
    }

    /// Brute-force XOR-span of a slice of values.
    fn brute_span(vals: &[u64]) -> Vec<u64> {
        let n = vals.len();
        let mut out = Vec::with_capacity(1 << n);
        for mask in 0u32..(1u32 << n) {
            let mut x = 0u64;
            for (i, &v) in vals.iter().enumerate() {
                if (mask >> i) & 1 == 1 {
                    x ^= v;
                }
            }
            out.push(x);
        }
        out.sort_unstable();
        out.dedup();
        out
    }

    #[quickcheck]
    #[allow(clippy::needless_pass_by_value)]
    fn prop_span_matches_brute_force(values: Vec<u64>) -> TestResult {
        if values.len() > 8 {
            return TestResult::discard();
        }
        let mut basis = XorBasis::new();
        for &v in &values {
            basis.insert(v);
        }
        let span = brute_span(&values);

        // Every brute-force span element must be in the basis span, and the
        // basis must reject anything outside it.
        for &x in &span {
            if !basis.contains(x) {
                return TestResult::failed();
            }
        }
        // Rank must match log2(|span|).
        if 1u64 << basis.rank() != span.len() as u64 {
            return TestResult::failed();
        }
        // max_xor(0) must equal the maximum element of the span.
        let span_max = *span.iter().max().unwrap_or(&0);
        if basis.max_xor(0) != span_max {
            return TestResult::failed();
        }
        // min_xor must equal the smallest non-zero span element (or 0 if
        // none exists).
        let span_min_nonzero = span.iter().copied().find(|&v| v != 0).unwrap_or(0);
        if basis.min_xor() != span_min_nonzero {
            return TestResult::failed();
        }
        TestResult::passed()
    }

    #[quickcheck]
    #[allow(clippy::needless_pass_by_value)]
    fn prop_outside_span_rejected(values: Vec<u64>, probe: u64) -> TestResult {
        if values.len() > 8 {
            return TestResult::discard();
        }
        let mut basis = XorBasis::new();
        for &v in &values {
            basis.insert(v);
        }
        let span = brute_span(&values);
        let in_span = span.binary_search(&probe).is_ok();
        if basis.contains(probe) != in_span {
            return TestResult::failed();
        }
        TestResult::passed()
    }
}
