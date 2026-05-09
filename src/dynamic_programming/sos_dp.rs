//! Sum over Subsets (SOS) DP: given an array indexed by bitmasks of `n` bits,
//! compute, for every mask `m`, the sum of `values[s]` over all `s ⊆ m`. The
//! dual transform aggregates over supersets instead. Both run in
//! O(n · 2^n) time and O(2^n) extra space, performing an in-place
//! bit-by-bit relaxation across the `n` bit-layers.

/// Returns `out` such that `out[mask] = Σ_{sub ⊆ mask} values[sub]`.
/// `values.len()` must be a power of two; it equals `2^n` where `n` is the
/// number of bits the masks range over.
///
/// # Panics
/// Panics if `values.len()` is zero or not a power of two.
pub fn sos_dp(values: &[i64]) -> Vec<i64> {
    let len = values.len();
    assert!(
        len.is_power_of_two(),
        "values length must be a power of two"
    );
    let bits = len.trailing_zeros() as usize;
    let mut dp = values.to_vec();
    for i in 0..bits {
        let bit = 1usize << i;
        for mask in 0..len {
            if mask & bit != 0 {
                dp[mask] += dp[mask ^ bit];
            }
        }
    }
    dp
}

/// Returns `out` such that `out[mask] = Σ_{mask ⊆ super} values[super]`.
/// `values.len()` must be a power of two.
///
/// # Panics
/// Panics if `values.len()` is zero or not a power of two.
pub fn superset_sum(values: &[i64]) -> Vec<i64> {
    let len = values.len();
    assert!(
        len.is_power_of_two(),
        "values length must be a power of two"
    );
    let bits = len.trailing_zeros() as usize;
    let mut dp = values.to_vec();
    for i in 0..bits {
        let bit = 1usize << i;
        for mask in 0..len {
            if mask & bit == 0 {
                dp[mask] += dp[mask | bit];
            }
        }
    }
    dp
}

#[cfg(test)]
mod tests {
    use super::{sos_dp, superset_sum};

    fn brute_subset_sum(values: &[i64]) -> Vec<i64> {
        let len = values.len();
        let mut out = vec![0_i64; len];
        for mask in 0..len {
            let mut sub = mask;
            loop {
                out[mask] += values[sub];
                if sub == 0 {
                    break;
                }
                sub = (sub - 1) & mask;
            }
        }
        out
    }

    fn brute_superset_sum(values: &[i64]) -> Vec<i64> {
        let len = values.len();
        let mut out = vec![0_i64; len];
        for mask in 0..len {
            for sup in 0..len {
                if mask & sup == mask {
                    out[mask] += values[sup];
                }
            }
        }
        out
    }

    #[test]
    fn n_zero_is_identity() {
        // 2^0 = 1, single value — both transforms are the identity.
        let v = vec![42];
        assert_eq!(sos_dp(&v), vec![42]);
        assert_eq!(superset_sum(&v), vec![42]);
    }

    #[test]
    fn n_two_hand_example() {
        // values = [1, 2, 3, 4]:
        //   out[00] = v[00]                   = 1
        //   out[01] = v[00] + v[01]           = 1 + 2 = 3
        //   out[10] = v[00] + v[10]           = 1 + 3 = 4
        //   out[11] = v[00] + v[01] + v[10] + v[11] = 10
        let v = vec![1, 2, 3, 4];
        assert_eq!(sos_dp(&v), vec![1, 3, 4, 10]);
    }

    #[test]
    fn n_three_against_brute_force() {
        let v: Vec<i64> = (1..=8).collect();
        assert_eq!(sos_dp(&v), brute_subset_sum(&v));
        assert_eq!(superset_sum(&v), brute_superset_sum(&v));
    }

    #[test]
    fn superset_self_consistency() {
        // For superset_sum, Σ_mask out[mask] = Σ_super (popcount-cofactor)
        // counts each `values[super]` once per subset of `super`, i.e.
        // 2^popcount(super) times. We sanity-check the all-ones input where
        // every value is 1: then out[mask] = 2^(n - popcount(mask)) and the
        // grand total is Σ_mask 2^(n - popcount(mask)) = 3^n.
        for n in 0..=4 {
            let len = 1 << n;
            let v = vec![1_i64; len];
            let out = superset_sum(&v);
            let total: i64 = out.iter().sum();
            assert_eq!(total, 3_i64.pow(n as u32));
            for (mask, &val) in out.iter().enumerate() {
                let pop = (mask as u32).count_ones();
                assert_eq!(val, 1_i64 << (n as u32 - pop));
            }
        }
    }

    #[test]
    fn property_against_brute_force_small_n() {
        // Deterministic LCG-ish sequence so the test is reproducible.
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut next = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            (state >> 33) as i64 - (1 << 30)
        };
        for n in 2..=4 {
            let len = 1 << n;
            let v: Vec<i64> = (0..len).map(|_| next()).collect();
            assert_eq!(sos_dp(&v), brute_subset_sum(&v));
            assert_eq!(superset_sum(&v), brute_superset_sum(&v));
        }
    }

    #[test]
    #[should_panic(expected = "values length must be a power of two")]
    fn panics_on_non_power_of_two_subset() {
        let _ = sos_dp(&[1, 2, 3]);
    }

    #[test]
    #[should_panic(expected = "values length must be a power of two")]
    fn panics_on_non_power_of_two_superset() {
        let _ = superset_sum(&[1, 2, 3]);
    }
}
