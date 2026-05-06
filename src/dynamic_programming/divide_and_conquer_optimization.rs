//! Divide-and-conquer DP optimisation.
//!
//! Speeds up recurrences of the form
//!
//! ```text
//! dp[i][j] = min_{k <= j} ( dp[i-1][k-1] + cost(k, j) )
//! ```
//!
//! from **O(k · n²)** to **O(k · n · log n)** when the `cost` function
//! satisfies the **Monge array / quadrangle inequality**:
//!
//! ```text
//! cost(a, c) + cost(b, d) <= cost(a, d) + cost(b, c)   for a <= b <= c <= d
//! ```
//!
//! Under this condition the optimal split point `opt(j)` is monotone in `j`,
//! which lets us recurse on left/right halves while keeping `k` within a
//! shrinking window — giving the O(n log n) layer cost.
//!
//! ## Canonical use case
//!
//! Partitioning a sequence into **k** contiguous segments minimising total
//! cost, e.g. minimising the sum of squared segment sums (used in
//! load-balancing and quantisation problems).
//!
//! ## Complexity
//!
//! | | Time | Space |
//! |---|---|---|
//! | Single layer (`dnc_layer`) | O(n log n) | O(n) |
//! | Full k-layer (`min_partition_cost`) | O(k · n · log n) | O(n) |
//!
//! ## Preconditions
//!
//! * The `cost` function must satisfy the Monge / quadrangle inequality so
//!   that the optimal split point is non-decreasing.  The algorithms produce
//!   **incorrect results** if this precondition is violated.
//! * `prev` must have length `n + 1` (indices 0 … n); `prev[0]` is the base
//!   case (cost of choosing nothing from the previous layer).

const INF: i64 = i64::MAX / 2;

/// Computes one DP layer using divide-and-conquer optimisation.
///
/// Given `prev[0..=n]` (previous layer) and `cost(l, r)` returning the
/// cost of the segment `a[l..=r]` (1-indexed, inclusive), returns
/// `cur[0..=n]` where
///
/// ```text
/// cur[j] = min over k in [1, j] of ( prev[k - 1] + cost(k, j) )
/// ```
///
/// `cur[0]` is always `INF / 2` (no valid 1-segment partition of an empty
/// prefix).
///
/// # Panics
///
/// Does not panic for well-formed input (`prev.len() >= 1`).
pub fn dnc_layer<F>(prev: &[i64], cost: F) -> Vec<i64>
where
    F: Fn(usize, usize) -> i64,
{
    let n = prev.len() - 1; // prev has indices 0..=n
    let mut cur = vec![INF; n + 1];

    // solve fills cur[lo..=hi] with opt-k restricted to [k_lo, k_hi].
    solve(prev, &cost, &mut cur, 1, n, 1, n);

    cur
}

/// Recursive divide-and-conquer kernel.
///
/// Fills `cur[lo..=hi]`; the optimal split point for any `j` in `[lo, hi]`
/// is guaranteed to lie in `[k_lo, k_hi]`.
fn solve<F>(prev: &[i64], cost: &F, cur: &mut [i64], lo: usize, hi: usize, k_lo: usize, k_hi: usize)
where
    F: Fn(usize, usize) -> i64,
{
    if lo > hi {
        return;
    }
    let mid = lo + (hi - lo) / 2;

    // Find the optimal k for cur[mid] by scanning [k_lo, min(k_hi, mid)].
    let k_upper = k_hi.min(mid);
    let mut best_val = INF;
    let mut best_k = k_lo;

    for k in k_lo..=k_upper {
        let prev_val = prev[k - 1];
        if prev_val >= INF {
            continue;
        }
        let candidate = prev_val.saturating_add(cost(k, mid));
        if candidate < best_val {
            best_val = candidate;
            best_k = k;
        }
    }
    cur[mid] = best_val;

    // Left half: j in [lo, mid-1], opt in [k_lo, best_k].
    if lo < mid {
        solve(prev, cost, cur, lo, mid - 1, k_lo, best_k);
    }
    // Right half: j in [mid+1, hi], opt in [best_k, k_hi].
    solve(prev, cost, cur, mid + 1, hi, best_k, k_hi);
}

/// Minimum cost of partitioning `a` into exactly `k` contiguous segments,
/// where the cost of a segment is `(sum of its elements)²` — a classic Monge
/// cost that is used in load-balancing and quantisation problems.
///
/// **Precondition:** all elements of `a` must be **non-negative**.  The
/// squared-sum cost satisfies the Monge / quadrangle inequality only when the
/// prefix sums are non-decreasing, which requires non-negative values.
/// Negative elements violate the Monge condition and produce **incorrect
/// results**.
///
/// Returns `i64::MAX / 2` if `k == 0` or `k > a.len()`.
///
/// # Complexity
///
/// Time O(k · n · log n), space O(n).
pub fn min_partition_cost(a: &[i64], k: usize) -> i64 {
    let n = a.len();
    if k == 0 || k > n {
        return INF;
    }

    // Build prefix sums for O(1) range-sum queries.
    let mut prefix = vec![0_i64; n + 1];
    for i in 0..n {
        prefix[i + 1] = prefix[i] + a[i];
    }

    // cost(l, r): squared sum of a[l-1..=r-1]  (1-indexed l, r).
    let seg_cost = |l: usize, r: usize| -> i64 {
        let s = prefix[r] - prefix[l - 1];
        s * s
    };

    // Base (layer 0): dp[0][0] = 0 (empty prefix, zero segments), rest INF.
    // dnc_layer is called once per layer; after k calls we have layer k.
    let mut dp = vec![INF; n + 1];
    dp[0] = 0;

    // Iterate layers 1..=k.
    for _ in 1..=k {
        dp = dnc_layer(&dp, seg_cost);
    }

    dp[n]
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::{dnc_layer, min_partition_cost, INF};

    // ── Brute-force reference ────────────────────────────────────────────────

    /// O(k · n²) brute-force implementation used as an oracle.
    fn brute_min_partition(a: &[i64], k: usize) -> i64 {
        let n = a.len();
        if k == 0 || k > n {
            return INF;
        }
        let mut prefix = vec![0_i64; n + 1];
        for i in 0..n {
            prefix[i + 1] = prefix[i] + a[i];
        }
        let seg_cost = |l: usize, r: usize| -> i64 {
            let s = prefix[r] - prefix[l - 1];
            s * s
        };

        // dp[i][j]: min cost using i segments over first j elements.
        let mut dp = vec![vec![INF; n + 1]; k + 1];
        dp[0][0] = 0;
        for seg in 1..=k {
            for j in seg..=n {
                for split in seg..=j {
                    if dp[seg - 1][split - 1] < INF {
                        let candidate = dp[seg - 1][split - 1] + seg_cost(split, j);
                        if candidate < dp[seg][j] {
                            dp[seg][j] = candidate;
                        }
                    }
                }
            }
        }
        dp[k][n]
    }

    // ── Unit tests ───────────────────────────────────────────────────────────

    #[test]
    fn k_equals_one_returns_total_cost() {
        let a = [1_i64, 2, 3, 4];
        // single segment: (1+2+3+4)^2 = 100
        assert_eq!(min_partition_cost(&a, 1), 100);
    }

    #[test]
    fn k_equals_n_each_element_own_segment() {
        let a = [1_i64, 2, 3, 4];
        // 1^2 + 2^2 + 3^2 + 4^2 = 1 + 4 + 9 + 16 = 30
        assert_eq!(min_partition_cost(&a, 4), 30);
    }

    #[test]
    fn k_greater_than_n_returns_inf() {
        let a = [1_i64, 2, 3];
        assert_eq!(min_partition_cost(&a, 4), INF);
    }

    #[test]
    fn k_zero_returns_inf() {
        assert_eq!(min_partition_cost(&[1_i64, 2], 0), INF);
    }

    #[test]
    fn empty_slice_k_zero() {
        assert_eq!(min_partition_cost(&[], 0), INF);
    }

    #[test]
    fn single_element_k1() {
        assert_eq!(min_partition_cost(&[5_i64], 1), 25);
    }

    #[test]
    fn two_elements_k2() {
        // [3, 4] k=2: 3^2 + 4^2 = 25
        assert_eq!(min_partition_cost(&[3_i64, 4], 2), 25);
    }

    #[test]
    fn compare_to_brute_small() {
        let a = [1_i64, 2, 3, 4, 5];
        for k in 1..=5 {
            assert_eq!(
                min_partition_cost(&a, k),
                brute_min_partition(&a, k),
                "mismatch at k={k}"
            );
        }
    }

    #[test]
    fn dnc_layer_basic() {
        // With prev representing "0 segments" base: prev[0]=0, rest=INF,
        // and a 3-element array [1,2,3], dnc_layer should reproduce
        // the first layer (k=1) of the brute DP.
        let a = [1_i64, 2, 3];
        let n = a.len();
        let mut prefix = vec![0_i64; n + 1];
        for i in 0..n {
            prefix[i + 1] = prefix[i] + a[i];
        }
        let seg_cost = |l: usize, r: usize| -> i64 {
            let s = prefix[r] - prefix[l - 1];
            s * s
        };
        let mut prev = vec![INF; n + 1];
        prev[0] = 0;
        // layer 1: cur[j] = seg_cost(1, j)
        let expected: Vec<i64> = (0..=n)
            .map(|j| if j == 0 { INF } else { seg_cost(1, j) })
            .collect();
        // layer 1 is built by dnc_layer applied to the base layer
        let cur = dnc_layer(&prev, seg_cost);
        assert_eq!(cur, expected);
    }

    // ── Property-based test ──────────────────────────────────────────────────

    #[cfg(test)]
    mod qc {
        use super::*;
        use quickcheck::TestResult;
        use quickcheck_macros::quickcheck;

        /// Property: divide-and-conquer result matches brute-force for small
        /// non-negative inputs (non-negative values are required for the
        /// squared-sum cost to satisfy the Monge condition).
        #[quickcheck]
        #[allow(clippy::needless_pass_by_value)]
        fn prop_matches_brute(xs: Vec<u8>, k_raw: u8) -> TestResult {
            // Keep inputs small to stay fast.
            if xs.is_empty() || xs.len() > 8 {
                return TestResult::discard();
            }
            // Use non-negative values; squared-sum is Monge only for non-neg.
            let a: Vec<i64> = xs.iter().map(|&x| i64::from(x)).collect();
            let n = a.len();
            let k = (k_raw as usize % n) + 1; // k in [1, n]
            let got = min_partition_cost(&a, k);
            let want = brute_min_partition(&a, k);
            if got == want {
                TestResult::passed()
            } else {
                TestResult::failed()
            }
        }
    }
}
