//! Optimal binary search tree (OBST) construction.
//!
//! Given sorted keys `k[0..n)` with access frequencies / weights `p[i]`,
//! computes the binary search tree minimising the **expected search cost**
//!
//! ```text
//!     C(T) = sum_{i=0}^{n-1} depth_T(k_i) * p[i]      (root at depth 1)
//! ```
//!
//! The standard interval DP is
//!
//! ```text
//!     e[i][j] = 0                                                  if i == j
//!     e[i][j] = W(i, j) + min_{i <= r < j} ( e[i][r] + e[r+1][j] ) otherwise
//! ```
//!
//! where `W(i, j) = sum_{t=i}^{j-1} p[t]`.  The `+W(i, j)` term is what bumps
//! every key in the chosen subtree down by one level — applied recursively this
//! produces exactly `sum depth · p`.
//!
//! Knuth's monotone-root optimisation observes that the optimal root index
//! `root[i][j]` is monotone in both arguments:
//!
//! ```text
//!     root[i][j-1] <= root[i][j] <= root[i+1][j]
//! ```
//!
//! so the inner search runs in amortised O(1) per cell, dropping the total
//! cost from O(n³) to **O(n²)**.
//!
//! ## Complexities
//!
//! | Function                  | Time   | Space  |
//! |---------------------------|--------|--------|
//! | [`optimal_bst_cost`]      | O(n²)  | O(n²)  |
//! | [`optimal_bst_structure`] | O(n²)  | O(n²)  |
//!
//! A private O(n³) reference solver is kept for testing and clarity; it is
//! used by the property tests to validate the Knuth result.

/// Returns the minimum expected search cost of any BST holding `n = p.len()`
/// keys with access frequencies `p`.
///
/// Uses Knuth's O(n²) monotone-root optimisation.
///
/// Returns `0` when `p` is empty.
///
/// # Examples
///
/// ```
/// use rust_algorithms::dynamic_programming::optimal_bst::optimal_bst_cost;
/// assert_eq!(optimal_bst_cost(&[]), 0);
/// assert_eq!(optimal_bst_cost(&[7]), 7);
/// ```
///
/// # Complexity
/// Time O(n²), space O(n²).
#[must_use]
pub fn optimal_bst_cost(p: &[u64]) -> u64 {
    optimal_bst_structure(p).0
}

/// Returns `(cost, root)` where `cost` is the minimum expected search cost
/// and `root[i][j]` is the index (into `p`) of the optimal root for the
/// sub-tree built from keys `p[i..j]` (with `i < j`).  Cells outside that
/// range hold `0` and should be ignored.
///
/// Uses Knuth's O(n²) optimisation.
///
/// Returns `(0, vec![])` when `p` is empty.
///
/// # Complexity
/// Time O(n²), space O(n²).
#[must_use]
pub fn optimal_bst_structure(p: &[u64]) -> (u64, Vec<Vec<usize>>) {
    let n = p.len();
    if n == 0 {
        return (0, Vec::new());
    }

    // Prefix sums of p so that W(i, j) = prefix[j] - prefix[i] = sum p[i..j].
    let mut prefix = vec![0_u64; n + 1];
    for (i, &pi) in p.iter().enumerate() {
        prefix[i + 1] = prefix[i] + pi;
    }
    let w = |i: usize, j: usize| prefix[j] - prefix[i];

    // e[i][j] is the optimal cost over keys p[i..j]; valid for 0 <= i <= j <= n.
    // root[i][j] is the optimal root index (in p) for the sub-tree on p[i..j];
    // only meaningful when i < j.
    let mut e = vec![vec![0_u64; n + 1]; n + 1];
    let mut root = vec![vec![0_usize; n + 1]; n + 1];

    // Base: single-key sub-trees.  e[i][i+1] = p[i] (depth 1 root).
    for i in 0..n {
        e[i][i + 1] = p[i];
        root[i][i + 1] = i;
    }

    // Iterate by sub-tree size `len = j - i`.
    for len in 2..=n {
        for i in 0..=n - len {
            let j = i + len;
            // Knuth window: root[i][j-1] <= r* <= root[i+1][j].
            let lo = root[i][j - 1];
            let hi = root[i + 1][j];
            // Both bounds are in [i, j-1] by construction; assert with debug
            // checks rather than runtime branches.
            debug_assert!(lo >= i && hi < j && lo <= hi);

            let wij = w(i, j);
            let mut best = u64::MAX;
            let mut best_r = lo;

            for r in lo..=hi {
                // Left sub-tree on p[i..r], right sub-tree on p[r+1..j].
                let left = e[i][r];
                let right = e[r + 1][j];
                let cost = left + right + wij;
                if cost < best {
                    best = cost;
                    best_r = r;
                }
            }

            e[i][j] = best;
            root[i][j] = best_r;
        }
    }

    (e[0][n], root)
}

/// Reference O(n³) solver used by tests.  Same recurrence, no Knuth bound on
/// the root scan.
#[cfg(test)]
fn optimal_bst_cost_cubic(p: &[u64]) -> u64 {
    let n = p.len();
    if n == 0 {
        return 0;
    }
    let mut prefix = vec![0_u64; n + 1];
    for (i, &pi) in p.iter().enumerate() {
        prefix[i + 1] = prefix[i] + pi;
    }
    let w = |i: usize, j: usize| prefix[j] - prefix[i];

    let mut e = vec![vec![0_u64; n + 1]; n + 1];
    for i in 0..n {
        e[i][i + 1] = p[i];
    }
    for len in 2..=n {
        for i in 0..=n - len {
            let j = i + len;
            let wij = w(i, j);
            let mut best = u64::MAX;
            for r in i..j {
                let cost = e[i][r] + e[r + 1][j] + wij;
                if cost < best {
                    best = cost;
                }
            }
            e[i][j] = best;
        }
    }
    e[0][n]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::{optimal_bst_cost, optimal_bst_cost_cubic, optimal_bst_structure};

    #[test]
    fn empty_is_zero() {
        assert_eq!(optimal_bst_cost(&[]), 0);
        let (cost, root) = optimal_bst_structure(&[]);
        assert_eq!(cost, 0);
        assert!(root.is_empty());
    }

    #[test]
    fn single_key_is_its_weight() {
        assert_eq!(optimal_bst_cost(&[42]), 42);
        let (cost, root) = optimal_bst_structure(&[42]);
        assert_eq!(cost, 42);
        assert_eq!(root[0][1], 0);
    }

    #[test]
    fn two_keys_root_is_heavier() {
        // Either tree has cost p[0] + p[1] + min(p[0], p[1]) when the lighter
        // key is the leaf.  For [3, 5]: rooting at 5 ⇒ 5 + 2*3 = 11.
        // Rooting at 3 ⇒ 3 + 2*5 = 13.  Minimum is 11.
        assert_eq!(optimal_bst_cost(&[3, 5]), 11);
    }

    #[test]
    fn clrs_canonical_example() {
        // The 11 access frequencies from the canonical CLRS OBST example
        // (Section 15.5), scaled ×100 to integers.  Under the *standard*
        // formulation used by `optimal_bst_cost` — `sum depth(k_i) · p[i]`
        // with the root at depth 1 and no dummy/failure keys — the optimum
        // for this flat 11-key sequence is 265, which the O(n³) baseline
        // confirms (see `knuth_matches_cubic_small_fixed`).  Note that the
        // 2.75 figure in CLRS is for the full `p + q` (success + failure)
        // formulation; the bare-keys variant tested here drops the failure
        // contribution.
        let p = [15, 10, 5, 10, 20, 5, 10, 5, 5, 5, 10];
        assert_eq!(optimal_bst_cost(&p), 265);
    }

    #[test]
    fn knuth_matches_cubic_small_fixed() {
        let cases: &[&[u64]] = &[
            &[1],
            &[1, 1],
            &[1, 2, 3],
            &[5, 1, 4, 1, 5],
            &[3, 1, 4, 1, 5, 9, 2, 6],
            &[15, 10, 5, 10, 20, 5, 10, 5, 5, 5, 10],
        ];
        for &p in cases {
            assert_eq!(
                optimal_bst_cost(p),
                optimal_bst_cost_cubic(p),
                "knuth != cubic for p = {p:?}"
            );
        }
    }

    #[test]
    fn root_indices_are_in_range() {
        let p = [15, 10, 5, 10, 20, 5, 10, 5, 5, 5, 10];
        let n = p.len();
        let (_, root) = optimal_bst_structure(&p);
        for i in 0..n {
            for j in i + 1..=n {
                let r = root[i][j];
                assert!(r >= i && r < j, "root[{i}][{j}] = {r} out of [{i}, {j})");
            }
        }
    }

    // ------------------------------------------------------------------
    // Property test: Knuth O(n²) result agrees with the O(n³) baseline.
    // ------------------------------------------------------------------
    mod prop {
        use super::{optimal_bst_cost, optimal_bst_cost_cubic};
        use quickcheck::TestResult;
        use quickcheck_macros::quickcheck;

        #[quickcheck]
        #[allow(clippy::needless_pass_by_value)]
        fn knuth_matches_cubic(p: Vec<u8>) -> TestResult {
            // Restrict to n <= 6 as the issue requests.
            if p.len() > 6 {
                return TestResult::discard();
            }
            // Cast to u64; allow zeros (still satisfies non-negativity).
            let p: Vec<u64> = p.iter().map(|&x| u64::from(x)).collect();
            TestResult::from_bool(optimal_bst_cost(&p) == optimal_bst_cost_cubic(&p))
        }
    }
}
