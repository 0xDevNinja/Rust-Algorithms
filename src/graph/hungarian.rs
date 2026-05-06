//! Hungarian algorithm (Kuhn–Munkres) for the minimum-cost assignment problem
//! on a rectangular n×m bipartite cost matrix with n ≤ m.
//!
//! The algorithm maintains dual variables (potentials) `u[i]` for each row and
//! `v[j]` for each column, and an alternating-tree shortest-path search
//! (Dijkstra-style over reduced costs) to extend the current matching.
//!
//! - **Time:**  O(n² · m) — n augmentation phases each doing O(n · m) work.
//! - **Space:** O(n + m) auxiliary storage beyond the input.
//! - **Preconditions:**
//!   - `cost` must be non-empty with at least one row and one column, OR
//!     empty (returns `(0, [])`).
//!   - All rows must have the same length.
//!   - `cost.len() ≤ cost[0].len()` (rows ≤ columns).
//!   - For maximisation, negate every entry before calling and negate the
//!     returned total.
//!   - Entries are `i64`; intermediate reduced costs stay within `i64` range
//!     provided the input values are reasonably bounded.

const INF: i64 = i64::MAX / 2;

/// Returns `(total_cost, assignment)` where `assignment[i]` is the column
/// assigned to row `i`. Operates on a cost matrix `cost[i][j]` with `i64`
/// entries; `cost.len() ≤ cost[0].len()` (rows ≤ columns).
///
/// Panics if any row has a different length from `cost[0]`, or if
/// `cost.len() > cost[0].len()`.
pub fn hungarian(cost: &[Vec<i64>]) -> (i64, Vec<usize>) {
    let n = cost.len();
    if n == 0 {
        return (0, vec![]);
    }
    let m = cost[0].len();
    assert!(
        n <= m,
        "hungarian: number of rows ({n}) must be ≤ number of columns ({m})"
    );
    for row in cost {
        assert_eq!(
            row.len(),
            m,
            "hungarian: all rows must have the same length"
        );
    }

    // Use 1-based indexing for rows (1..=n) and columns (1..=m) to keep
    // sentinel index 0 as "unmatched".
    //
    // p[j]  = row currently assigned to column j  (0 = free)
    // way[j] = predecessor column in the augmenting path ending at j
    // u[i]  = row potential   (1..=n; u[0] unused)
    // v[j]  = column potential (0..=m; v[0] unused)
    // minv[j] = current minimum reduced cost to reach column j
    // used[j] = column j is in the current augmenting tree

    let mut p = vec![0usize; m + 1]; // p[j]: row matched to column j
    let mut u = vec![0i64; n + 1];
    let mut v = vec![0i64; m + 1];

    for i in 1..=n {
        // Add row i to the matching; column 0 is a virtual "free" column.
        p[0] = i;
        let mut j0 = 0usize;
        let mut minv = vec![INF; m + 1];
        let mut used = vec![false; m + 1];
        let mut way = vec![0usize; m + 1];

        // Dijkstra-style augmentation.
        loop {
            used[j0] = true;
            let i0 = p[j0]; // current row being extended
            let mut delta = INF;
            let mut j1 = 0usize;

            for j in 1..=m {
                if !used[j] {
                    // Reduced cost of edge (i0, j).
                    let cur = cost[i0 - 1][j - 1] - u[i0] - v[j];
                    if cur < minv[j] {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if minv[j] < delta {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }

            // Update potentials.
            for j in 0..=m {
                if used[j] {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }

            j0 = j1;
            if p[j0] == 0 {
                break; // j0 is free — augmenting path complete
            }
        }

        // Trace back and flip the matching along the augmenting path.
        while j0 != 0 {
            p[j0] = p[way[j0]];
            j0 = way[j0];
        }
    }

    // Build the assignment vector: assignment[i] = column (0-based) for row i.
    let mut assignment = vec![0usize; n];
    for j in 1..=m {
        if p[j] != 0 {
            assignment[p[j] - 1] = j - 1;
        }
    }

    let total: i64 = (0..n).map(|i| cost[i][assignment[i]]).sum();
    (total, assignment)
}

#[cfg(test)]
mod tests {
    use super::hungarian;
    use quickcheck::TestResult;
    use quickcheck_macros::quickcheck;

    #[test]
    fn empty() {
        assert_eq!(hungarian(&[]), (0, vec![]));
    }

    #[test]
    fn one_by_one() {
        assert_eq!(hungarian(&[vec![42]]), (42, vec![0]));
        assert_eq!(hungarian(&[vec![-7]]), (-7, vec![0]));
    }

    #[test]
    fn two_by_two_known() {
        // Optimal: row 0 → col 0 (cost 1), row 1 → col 1 (cost 1) = total 2.
        let cost = vec![vec![1, 2], vec![2, 1]];
        let (total, asgn) = hungarian(&cost);
        assert_eq!(total, 2);
        assert_eq!(asgn, vec![0, 1]);
    }

    #[test]
    fn three_by_three_with_negatives() {
        // Rows represent workers, cols represent tasks.
        // cost = [[-1, 2, 3],
        //         [ 4,-5, 6],
        //         [ 7, 8,-9]]
        // Optimal: (0,0)=-1, (1,1)=-5, (2,2)=-9  =>  total = -15
        let cost = vec![vec![-1, 2, 3], vec![4, -5, 6], vec![7, 8, -9]];
        let (total, asgn) = hungarian(&cost);
        assert_eq!(total, -15);
        assert_eq!(asgn, vec![0, 1, 2]);
    }

    #[test]
    fn rectangular_two_by_four() {
        // cost = [[3, 1, 2, 4],
        //         [2, 3, 1, 5]]
        // Optimal: row 0→col 1 (1), row 1→col 2 (1) = total 2.
        let cost = vec![vec![3, 1, 2, 4], vec![2, 3, 1, 5]];
        let (total, asgn) = hungarian(&cost);
        assert_eq!(total, 2);
        // Columns must be distinct.
        assert_ne!(asgn[0], asgn[1]);
        // Each column is within bounds.
        assert!(asgn[0] < 4 && asgn[1] < 4);
        // Verify optimality by value.
        let brute = brute_min_assignment(&cost);
        assert_eq!(total, brute);
    }

    #[test]
    fn three_by_three_classic() {
        // Classic 3×3 example from textbooks.
        // cost = [[9, 2, 7],
        //         [3, 6, 4],
        //         [1, 8, 5]]
        // Optimal: (0,1)=2, (1,2)=4, (2,0)=1  =>  total = 7
        let cost = vec![vec![9, 2, 7], vec![3, 6, 4], vec![1, 8, 5]];
        let (total, asgn) = hungarian(&cost);
        assert_eq!(total, 7);
        assert_eq!(asgn, vec![1, 2, 0]);
    }

    // -----------------------------------------------------------------------
    // Brute-force helper used in the quickcheck property.
    // Enumerates all ways to choose n distinct columns from m for n rows and
    // returns the minimum cost.
    // -----------------------------------------------------------------------

    fn brute_min_assignment(cost: &[Vec<i64>]) -> i64 {
        let n = cost.len();
        let m = cost[0].len();
        // Generate all combinations of n columns from 0..m, then permute them.
        let cols: Vec<usize> = (0..m).collect();
        let mut min_cost = i64::MAX;
        for_each_permutation_of_size(&cols, n, &mut |perm: &[usize]| {
            let c: i64 = (0..n).map(|i| cost[i][perm[i]]).sum();
            if c < min_cost {
                min_cost = c;
            }
        });
        min_cost
    }

    /// Calls `callback` with every ordered selection (without replacement) of
    /// `k` items from `items`.  O(m! / (m-k)!) calls.
    fn for_each_permutation_of_size<F>(items: &[usize], k: usize, callback: &mut F)
    where
        F: FnMut(&[usize]),
    {
        let mut buf: Vec<usize> = Vec::with_capacity(k);
        let mut used = vec![false; items.len()];
        recurse(items, k, &mut used, &mut buf, callback);
    }

    fn recurse<F>(
        items: &[usize],
        k: usize,
        used: &mut Vec<bool>,
        buf: &mut Vec<usize>,
        callback: &mut F,
    ) where
        F: FnMut(&[usize]),
    {
        if buf.len() == k {
            callback(buf);
            return;
        }
        for i in 0..items.len() {
            if !used[i] {
                used[i] = true;
                buf.push(items[i]);
                recurse(items, k, used, buf, callback);
                buf.pop();
                used[i] = false;
            }
        }
    }

    // -----------------------------------------------------------------------
    // QuickCheck property: Hungarian ≤ brute force on small non-negative
    // matrices (n ≤ 4, m ≤ 6).
    // -----------------------------------------------------------------------

    #[quickcheck]
    #[allow(clippy::needless_pass_by_value)]
    fn prop_hungarian_matches_brute_force(rows: Vec<Vec<u8>>) -> TestResult {
        // Constrain dimensions: 1..=4 rows, 1..=6 cols.
        if rows.is_empty() || rows.len() > 4 {
            return TestResult::discard();
        }
        let m = rows[0].len();
        if m == 0 || m > 6 || m < rows.len() {
            return TestResult::discard();
        }
        // All rows must have the same length.
        if rows.iter().any(|r| r.len() != m) {
            return TestResult::discard();
        }
        // Convert to i64.
        let cost: Vec<Vec<i64>> = rows
            .iter()
            .map(|r| r.iter().map(|&x| i64::from(x)).collect())
            .collect();

        let (total, asgn) = hungarian(&cost);

        // Assignment must have correct length and distinct columns in bounds.
        let n = cost.len();
        if asgn.len() != n {
            return TestResult::failed();
        }
        let mut seen = vec![false; m];
        for &col in &asgn {
            if col >= m || seen[col] {
                return TestResult::failed();
            }
            seen[col] = true;
        }
        // Verify total equals sum of assigned costs.
        let computed: i64 = (0..n).map(|i| cost[i][asgn[i]]).sum();
        if computed != total {
            return TestResult::failed();
        }
        // Compare against brute-force optimum.
        let expected = brute_min_assignment(&cost);
        TestResult::from_bool(total == expected)
    }
}
