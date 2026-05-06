//! Knuth's DP optimisation (Knuth–Yao speedup).
//!
//! Speeds up the recurrence
//!
//! ```text
//! dp[l][r] = w(l, r) + min_{l <= k < r} ( dp[l][k] + dp[k+1][r] )
//! dp[l][l] = 0
//! ```
//!
//! from **O(n³)** to **O(n²)** whenever the cost function `w` satisfies:
//!
//! 1. **Quadrangle (Monge) inequality** — `w(a,c) + w(b,d) <= w(a,d) + w(b,c)`
//!    for every `a <= b <= c <= d`.
//! 2. **Monotonicity** — `w(b,c) <= w(a,d)` for every `a <= b <= c <= d`.
//!
//! Under these conditions the optimal split `opt[l][r]` is monotone:
//!
//! ```text
//! opt[l][r-1] <= opt[l][r] <= opt[l+1][r]
//! ```
//!
//! so the inner loop over `k` costs amortised O(1) per cell, yielding O(n²)
//! total.
//!
//! # Canonical use cases
//! * Optimal binary search tree (OBST)
//! * Optimal file / tape merging (Hu–Shing)
//! * Optimal polygon triangulation
//!
//! # Complexities
//! | Phase | Time  | Space  |
//! |-------|-------|--------|
//! | Setup | O(n²) | O(n²)  |
//! | Query | O(1)  | —      |

/// Solves `dp[l][r] = w(l, r) + min_{l <= k < r} (dp[l][k] + dp[k+1][r])`
/// with Knuth's optimisation, given the cost function `w`.
///
/// Returns `dp[0][n-1]` (0 if `n <= 1`).
///
/// # Preconditions
/// `w` must satisfy the quadrangle inequality and monotonicity (see module
/// doc). The function does **not** verify these properties at runtime.
///
/// # Panics
/// Never panics for valid `n` (allocates two `n × n` tables).
///
/// # Complexity
/// Time O(n²), space O(n²).
pub fn knuth_dp<F: Fn(usize, usize) -> i64>(n: usize, w: F) -> i64 {
    if n <= 1 {
        return 0;
    }

    // dp[l][r]  – minimum cost for the sub-problem on interval [l, r].
    // opt[l][r] – argmin k that achieved dp[l][r].
    let mut dp = vec![vec![0_i64; n]; n];
    let mut opt = vec![vec![0_usize; n]; n];

    // Base: single elements cost 0; opt[i][i] = i.
    for i in 0..n {
        opt[i][i] = i;
    }

    // Iterate by interval length len = r - l.
    for len in 1..n {
        for l in 0..n - len {
            let r = l + len;
            let lo = opt[l][r - 1];
            let hi = if r + 1 < n { opt[l + 1][r] } else { r - 1 };
            // Clamp hi to r-1 (the valid range for k is [l, r-1]).
            let hi = hi.min(r - 1);

            let mut best = i64::MAX;
            let mut best_k = lo;

            for k in lo..=hi {
                // dp[l][k] + dp[k+1][r]; dp[k+1][r] is valid because k < r.
                let cost = dp[l][k]
                    .saturating_add(dp[k + 1][r])
                    .saturating_add(w(l, r));
                if cost < best {
                    best = cost;
                    best_k = k;
                }
            }

            dp[l][r] = best;
            opt[l][r] = best_k;
        }
    }

    dp[0][n - 1]
}

/// Optimal-merge cost: given file sizes, returns the minimum total cost of
/// merging all files into one, where merging two groups costs the sum of their
/// sizes.
///
/// This is the classic *optimal file merging* (Huffman-style, but for the
/// general two-way merge problem solved via interval DP).
///
/// # Examples
/// ```
/// use rust_algorithms::dynamic_programming::knuth_optimization::optimal_file_merge;
/// assert_eq!(optimal_file_merge(&[1, 2, 3, 4, 5]), 33);
/// ```
///
/// # Complexity
/// Time O(n²), space O(n²).
pub fn optimal_file_merge(sizes: &[i64]) -> i64 {
    let n = sizes.len();
    if n <= 1 {
        return 0;
    }

    // Prefix sums so that sum(l..=r) = prefix[r+1] - prefix[l].
    let mut prefix = vec![0_i64; n + 1];
    for (i, &s) in sizes.iter().enumerate() {
        prefix[i + 1] = prefix[i] + s;
    }

    // Cost of merging the contiguous block [l, r] is the sum of all sizes in
    // that range (you always pay for the combined file you create).
    let w = |l: usize, r: usize| prefix[r + 1] - prefix[l];

    knuth_dp(n, w)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::{knuth_dp, optimal_file_merge};

    // ------------------------------------------------------------------
    // Unit tests – edge cases
    // ------------------------------------------------------------------

    #[test]
    fn n_zero_returns_zero() {
        assert_eq!(knuth_dp(0, |_, _| 0), 0);
    }

    #[test]
    fn n_one_returns_zero() {
        assert_eq!(knuth_dp(1, |_, _| 0), 0);
    }

    #[test]
    fn n_two_cost_equals_w() {
        // Single merge: dp[0][1] = w(0,1) + dp[0][0] + dp[1][1] = w(0,1).
        assert_eq!(knuth_dp(2, |_, _| 7), 7);
    }

    #[test]
    fn file_merge_empty() {
        assert_eq!(optimal_file_merge(&[]), 0);
    }

    #[test]
    fn file_merge_single() {
        assert_eq!(optimal_file_merge(&[42]), 0);
    }

    #[test]
    fn file_merge_two() {
        // Merging [3, 5] costs 3 + 5 = 8.
        assert_eq!(optimal_file_merge(&[3, 5]), 8);
    }

    #[test]
    fn file_merge_known_five() {
        // [1, 2, 3, 4, 5] → verified minimum = 33.
        // Optimal: merge 1+2=3 (cost 3), then 3+3=6 (cost 6),
        // then 4+5=9 (cost 9), then 6+9=15 (cost 15). Total = 33.
        assert_eq!(optimal_file_merge(&[1, 2, 3, 4, 5]), 33);
    }

    // ------------------------------------------------------------------
    // Correctness: compare against brute-force O(n^3) interval DP
    // ------------------------------------------------------------------

    /// Brute-force O(n³) solver – identical recurrence, no Knuth bound.
    fn brute_force<F: Fn(usize, usize) -> i64>(n: usize, w: F) -> i64 {
        if n <= 1 {
            return 0;
        }
        let mut dp = vec![vec![0_i64; n]; n];
        for len in 1..n {
            for l in 0..n - len {
                let r = l + len;
                let mut best = i64::MAX;
                for k in l..r {
                    let cost = dp[l][k]
                        .saturating_add(dp[k + 1][r])
                        .saturating_add(w(l, r));
                    if cost < best {
                        best = cost;
                    }
                }
                dp[l][r] = best;
            }
        }
        dp[0][n - 1]
    }

    /// Build a cost matrix satisfying the quadrangle inequality from
    /// prefix sums (the standard construction used by `optimal_file_merge`).
    fn prefix_w(sizes: &[i64]) -> impl Fn(usize, usize) -> i64 + '_ {
        let n = sizes.len();
        let mut prefix = vec![0_i64; n + 1];
        for (i, &s) in sizes.iter().enumerate() {
            prefix[i + 1] = prefix[i] + s;
        }
        // Return a closure; prefix is owned by the closure.
        move |l: usize, r: usize| prefix[r + 1] - prefix[l]
    }

    #[test]
    fn matches_brute_force_small() {
        let cases: &[&[i64]] = &[
            &[1],
            &[1, 2],
            &[1, 2, 3],
            &[3, 1, 4, 1, 5],
            &[10, 1, 1, 1, 1, 1, 1, 1, 1, 10],
        ];
        for &sizes in cases {
            let n = sizes.len();
            let w = prefix_w(sizes);
            let expected = brute_force(n, &w);
            let got = knuth_dp(n, &w);
            assert_eq!(got, expected, "knuth_dp != brute_force for sizes={sizes:?}");
        }
    }

    // ------------------------------------------------------------------
    // Property-based test: knuth vs brute-force over random size vectors
    // ------------------------------------------------------------------

    #[cfg(test)]
    mod prop {
        use super::{brute_force, knuth_dp, prefix_w};
        use quickcheck::TestResult;
        use quickcheck_macros::quickcheck;

        #[quickcheck]
        #[allow(clippy::needless_pass_by_value)]
        fn prop_knuth_matches_brute_force(sizes: Vec<u8>) -> TestResult {
            // Limit to length <= 8 for speed; require non-empty.
            if sizes.len() > 8 {
                return TestResult::discard();
            }
            let sizes: Vec<i64> = sizes.iter().map(|&x| i64::from(x) + 1).collect();
            let n = sizes.len();
            let w = prefix_w(&sizes);
            let expected = brute_force(n, &w);
            let got = knuth_dp(n, &w);
            TestResult::from_bool(got == expected)
        }

        #[quickcheck]
        #[allow(clippy::needless_pass_by_value)]
        fn prop_optimal_file_merge_matches_brute_force(sizes: Vec<u8>) -> TestResult {
            if sizes.len() > 8 {
                return TestResult::discard();
            }
            let sizes: Vec<i64> = sizes.iter().map(|&x| i64::from(x) + 1).collect();
            let n = sizes.len();
            let w = prefix_w(&sizes);
            let expected = brute_force(n, &w);
            let got = super::optimal_file_merge(&sizes);
            TestResult::from_bool(got == expected)
        }
    }
}
