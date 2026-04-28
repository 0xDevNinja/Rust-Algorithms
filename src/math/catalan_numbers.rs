//! Catalan numbers — compute `C_n` via the convolution recurrence.
//!
//! # Definition
//! The Catalan numbers count, among many other things, the number of distinct
//! binary trees on `n` nodes, the number of valid sequences of `n` pairs of
//! parentheses, and the number of monotonic lattice paths along the edges of
//! an `n × n` grid that do not cross the diagonal.
//!
//! # Recurrence
//! With `C_0 = 1`,
//! ```text
//! C_{n+1} = sum_{i = 0..=n} C_i * C_{n - i}
//! ```
//! Equivalently, the closed form is `C_n = binomial(2n, n) / (n + 1)`. This
//! module uses the convolution recurrence: it builds the full prefix
//! `C_0..=C_n` and is the most direct expression of the combinatorial
//! identity, with no intermediate values larger than the result.
//!
//! # Complexity
//! - Time:  O(n²) word-sized multiplications and additions.
//! - Space: O(n) — a `Vec<u128>` of length `n + 1` while building the prefix.
//!
//! # Safe range
//! Results are returned as [`u128`]. `C_35 = 3_116_285_494_907_301_262` fits
//! comfortably, but `C_36` already exceeds `u128::MAX`, so this routine
//! panics on overflow for `n >= 36`. The caller is responsible for staying
//! within `n <= 35`.

/// Returns the `n`-th Catalan number `C_n` as a [`u128`].
///
/// Computed by iterating the convolution recurrence
/// `C_{k+1} = sum_{i=0..=k} C_i * C_{k-i}` from `C_0 = 1`. Runs in O(n²)
/// time and O(n) auxiliary space.
///
/// # Panics
/// Panics on arithmetic overflow. `C_36` already exceeds [`u128::MAX`], so
/// this routine is only safe for `n <= 35`.
///
/// # Examples
/// ```
/// use rust_algorithms::math::catalan_numbers::catalan;
/// assert_eq!(catalan(0), 1);
/// assert_eq!(catalan(1), 1);
/// assert_eq!(catalan(5), 42);
/// assert_eq!(catalan(10), 16_796);
/// ```
#[must_use]
pub fn catalan(n: u32) -> u128 {
    catalan_sequence(n)[n as usize]
}

/// Returns the prefix `[C_0, C_1, ..., C_{n_max}]` of Catalan numbers.
///
/// Useful when several consecutive Catalan numbers are needed: the prefix is
/// built bottom-up using the convolution recurrence, so amortizing across
/// `n_max + 1` values still costs only `O(n_max²)` total work.
///
/// # Panics
/// Panics on arithmetic overflow. `C_36` already exceeds [`u128::MAX`], so
/// this routine is only safe for `n_max <= 35`.
///
/// # Examples
/// ```
/// use rust_algorithms::math::catalan_numbers::catalan_sequence;
/// assert_eq!(
///     catalan_sequence(6),
///     vec![1, 1, 2, 5, 14, 42, 132],
/// );
/// ```
#[must_use]
pub fn catalan_sequence(n_max: u32) -> Vec<u128> {
    let len = n_max as usize + 1;
    let mut c: Vec<u128> = Vec::with_capacity(len);
    c.push(1); // C_0 = 1.

    // Build C_{k+1} from C_0..=C_k via the convolution sum. Indices `i` and
    // `k - i` walk the sequence from opposite ends, mirroring the symmetry
    // C_i * C_{k-i} = C_{k-i} * C_i.
    for k in 0..n_max as usize {
        let mut next: u128 = 0;
        for i in 0..=k {
            next += c[i] * c[k - i];
        }
        c.push(next);
    }

    c
}

#[cfg(test)]
mod tests {
    use super::{catalan, catalan_sequence};
    use quickcheck_macros::quickcheck;

    /// First eleven Catalan numbers, `C_0..=C_10`, as listed in OEIS A000108.
    const KNOWN: [u128; 11] = [1, 1, 2, 5, 14, 42, 132, 429, 1430, 4862, 16796];

    #[test]
    fn known_small_values() {
        for (n, &expected) in KNOWN.iter().enumerate() {
            assert_eq!(catalan(n as u32), expected, "C_{n} mismatch");
        }
    }

    #[test]
    fn sequence_matches_known() {
        assert_eq!(catalan_sequence(10), KNOWN.to_vec());
    }

    #[test]
    fn c_0_is_one() {
        assert_eq!(catalan(0), 1);
    }

    #[test]
    fn sequence_length_is_n_plus_one() {
        assert_eq!(catalan_sequence(0).len(), 1);
        assert_eq!(catalan_sequence(7).len(), 8);
    }

    /// `C_35` is the largest Catalan number that fits in a `u128`.
    #[test]
    fn c_35_largest_safe() {
        assert_eq!(catalan(35), 3_116_285_494_907_301_262);
    }

    /// For every `n <= 15`, `catalan(n)` must agree with the `n`-th element
    /// of the prefix returned by `catalan_sequence(n)`.
    #[quickcheck]
    fn prop_catalan_matches_sequence(n: u8) -> bool {
        let n = u32::from(n % 16);
        catalan(n) == catalan_sequence(n)[n as usize]
    }

    /// The convolution recurrence must be self-consistent: re-deriving
    /// `C_{n+1}` inline from `C_0..=C_n` reproduces `catalan(n + 1)`.
    #[quickcheck]
    fn prop_recurrence_self_consistent(n: u8) -> bool {
        let n = u32::from(n % 15) as usize;
        let prefix = catalan_sequence(n as u32);
        let mut sum: u128 = 0;
        for i in 0..=n {
            sum += prefix[i] * prefix[n - i];
        }
        catalan(n as u32 + 1) == sum
    }
}
