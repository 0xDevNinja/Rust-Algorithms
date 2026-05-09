//! Convex hull trick (monotonic / offline variant).
//!
//! Maintains a set of lines `y = m·x + b` and answers "minimum value at `x`"
//! queries in **amortised O(1)** per insertion and per query, provided two
//! monotonicity preconditions hold.
//!
//! ## What it solves
//!
//! Many DP recurrences have the form
//!
//! ```text
//! dp[i] = min over j < i of ( m_j · x_i + b_j )
//! ```
//!
//! where `m_j` and `b_j` depend only on `j` and `x_i` depends only on `i`.
//! Each prior state `j` defines a line `L_j(x) = m_j · x + b_j`, and the DP
//! step is "evaluate the lower envelope of `{L_j}` at `x = x_i`".  Naively
//! this is `O(n²)`; the convex hull trick collapses it to `O(n)` (or
//! `O(n log n)` with a more general variant — not implemented here).
//!
//! ## Preconditions (monotonic / offline variant)
//!
//! * **Lines are inserted in non-increasing slope order** (`m` non-increasing).
//!   This is the natural order for *minimum* queries: a flatter line can only
//!   ever beat a steeper one for sufficiently large `x`, so steep lines
//!   become useless first.
//! * **Queries arrive in non-decreasing `x` order.**  This lets us advance a
//!   pointer through the deque instead of binary searching.
//!
//! Violating either precondition produces **incorrect results**.  For the
//! fully online / dynamic case use Li Chao tree or a sorted-set CHT — neither
//! is part of this module.
//!
//! ## Convention
//!
//! This module implements the **minimum** CHT.  To compute a maximum, negate
//! both `m` and `b` on insertion and negate the query result.
//!
//! ## Complexity
//!
//! | Operation | Amortised | Worst case |
//! |---|---|---|
//! | `add_line` | O(1) | O(n) (one pop chain, charged to past inserts) |
//! | `query` | O(1) | O(n) (pointer advance, charged to past queries) |
//!
//! Memory: `O(n)` for the deque of useful lines.

/// Container for the lower envelope of a set of lines.
///
/// See module docs for preconditions on insertion / query order.
#[derive(Debug, Default, Clone)]
pub struct LineContainer {
    // Each entry is (slope, intercept).  The vector stores only lines that
    // currently contribute to the lower envelope, ordered left-to-right
    // along the x-axis (i.e. by decreasing slope, since slopes are inserted
    // non-increasing).
    lines: Vec<(i64, i64)>,
    // Pointer into `lines` for the next query.  Monotonically non-decreasing
    // queries let us scan forward without resetting.
    ptr: usize,
}

impl LineContainer {
    /// Creates an empty container.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            lines: Vec::new(),
            ptr: 0,
        }
    }

    /// Returns the number of lines currently on the lower envelope.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.lines.len()
    }

    /// Returns `true` if no lines have been added (or all have been popped).
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.lines.is_empty()
    }

    /// Inserts a new line `y = m · x + b`.
    ///
    /// **Precondition:** `m` must be **non-increasing** across calls
    /// (suited to the minimum-CHT convention).  Equal slopes are accepted —
    /// the line with smaller `b` dominates and the other is dropped.
    ///
    /// Amortised `O(1)`: the inner loop pops earlier lines that have become
    /// dominated, and each popped line was pushed exactly once before.
    pub fn add_line(&mut self, m: i64, b: i64) {
        // First, fold in equal-slope lines: if the last line has the same
        // slope, keep whichever has the smaller intercept (the other is
        // strictly dominated everywhere).
        if let Some(&(m_last, c_last)) = self.lines.last() {
            if m_last == m {
                if c_last <= b {
                    return;
                }
                self.lines.pop();
            }
        }
        // Pop the back line `b2` while it is no longer on the lower envelope
        // after the new line `n` is added.  With slopes non-increasing
        // (m1 > m2 > m_n strictly, after dedup above), `b2` stays on the
        // envelope iff the intersection of (b1, b2) is strictly to the left
        // of the intersection of (b2, n):
        //
        //     (c2 - c1) / (m1 - m2)  <  (b - c2) / (m2 - m_n)
        //
        // Both denominators are positive, so cross-multiply.  Use i128 to
        // avoid overflow for i64 inputs.
        while self.lines.len() >= 2 {
            let n_len = self.lines.len();
            let (m1, c1) = self.lines[n_len - 2];
            let (m2, c2) = self.lines[n_len - 1];
            let lhs = i128::from(c2 - c1) * i128::from(m2 - m);
            let rhs = i128::from(b - c2) * i128::from(m1 - m2);
            if lhs >= rhs {
                self.lines.pop();
            } else {
                break;
            }
        }
        self.lines.push((m, b));
    }

    /// Returns the minimum of `m · x + b` over all stored lines at the given
    /// `x`.
    ///
    /// **Precondition:** queries must be made with **non-decreasing `x`**.
    /// The internal pointer only moves forward, so a smaller `x` than the
    /// previous query produces an incorrect answer.
    ///
    /// # Panics
    ///
    /// Panics if no lines have been added.
    pub fn query(&mut self, x: i64) -> i64 {
        assert!(
            !self.lines.is_empty(),
            "query on empty LineContainer is undefined"
        );
        if self.ptr >= self.lines.len() {
            self.ptr = self.lines.len() - 1;
        }
        // Advance the pointer while the next line gives a smaller value at x.
        // Because slopes are non-increasing, once the next line stops winning
        // it never wins again for larger x.
        while self.ptr + 1 < self.lines.len() {
            let (m_a, b_a) = self.lines[self.ptr];
            let (m_b, b_b) = self.lines[self.ptr + 1];
            if m_b * x + b_b <= m_a * x + b_a {
                self.ptr += 1;
            } else {
                break;
            }
        }
        let (m, b) = self.lines[self.ptr];
        m * x + b
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::LineContainer;

    /// Brute-force minimum over all lines at `x`.
    fn brute_min(lines: &[(i64, i64)], x: i64) -> i64 {
        lines
            .iter()
            .map(|&(m, b)| m * x + b)
            .min()
            .expect("at least one line")
    }

    #[test]
    fn single_line_returns_its_value() {
        let mut cht = LineContainer::new();
        cht.add_line(3, 5);
        assert_eq!(cht.query(0), 5);
        assert_eq!(cht.query(2), 11);
        assert_eq!(cht.query(10), 35);
    }

    #[test]
    fn empty_initially() {
        let cht = LineContainer::new();
        assert!(cht.is_empty());
        assert_eq!(cht.len(), 0);
    }

    #[test]
    fn three_lines_lower_envelope() {
        // Three lines, slopes inserted non-increasing.
        //   L0: y = 3x + 0   (steepest, wins for small x)
        //   L1: y = 1x + 2   (medium slope, wins in the middle)
        //   L2: y = 0x + 5   (flat, wins for large x — but L1 stays better here)
        // Lower envelope:
        //   x = 0: min(0, 2, 5)   = 0  (L0)
        //   x = 1: min(3, 3, 5)   = 3  (L0 or L1)
        //   x = 2: min(6, 4, 5)   = 4  (L1)
        //   x = 5: min(15, 7, 5)  = 5  (L2)
        //   x = 10:min(30, 12, 5) = 5  (L2)
        let mut cht = LineContainer::new();
        cht.add_line(3, 0);
        cht.add_line(1, 2);
        cht.add_line(0, 5);

        let lines = [(3_i64, 0_i64), (1, 2), (0, 5)];
        for &x in &[0_i64, 1, 2, 5, 10] {
            assert_eq!(cht.query(x), brute_min(&lines, x), "x = {x}");
        }
    }

    #[test]
    fn duplicate_slope_keeps_lower_intercept() {
        let mut cht = LineContainer::new();
        cht.add_line(2, 10);
        cht.add_line(2, 3); // same slope, lower intercept — replaces previous
        assert_eq!(cht.query(0), 3);
        assert_eq!(cht.query(5), 13);
    }

    #[test]
    fn duplicate_slope_higher_intercept_ignored() {
        let mut cht = LineContainer::new();
        cht.add_line(2, 3);
        cht.add_line(2, 10); // same slope, worse — ignored
        assert_eq!(cht.query(0), 3);
        assert_eq!(cht.query(5), 13);
    }

    #[test]
    fn dominated_line_is_popped() {
        // Insert a line in the middle that becomes fully dominated by a later
        // one.  Slopes still non-increasing.
        let mut cht = LineContainer::new();
        cht.add_line(5, 0); // steep
        cht.add_line(3, 1); // medium — will be dominated by next
        cht.add_line(1, 0); // shallow with smaller intercept, dominates middle
                            // The middle line is dominated whenever there exists x where
                            // the third beats it AND the first beats it.  Middle line at x=0
                            // is 1, but L2 at x=0 is 0; L0 at x=0 is 0 too.  At x=1 middle
                            // is 4, L0 is 5, L2 is 1 → L2 wins.  Middle should be popped.
        assert_eq!(cht.len(), 2);
        let lines = [(5_i64, 0_i64), (3, 1), (1, 0)];
        for &x in &[0_i64, 1, 2, 5, 10] {
            assert_eq!(cht.query(x), brute_min(&lines, x), "x = {x}");
        }
    }

    /// Brute O(n²) DP: dp[i] = min over j < i of ( a[j] · x[i] + b[j] ).
    fn brute_dp(a: &[i64], b_coef: &[i64], x: &[i64]) -> Vec<i64> {
        let n = a.len();
        assert_eq!(b_coef.len(), n);
        assert_eq!(x.len(), n);
        let mut dp = vec![i64::MAX; n];
        // The recurrence requires j < i; treat j = 0 as the seed line.
        dp[0] = a[0] * x[0] + b_coef[0]; // value of line 0 at x[0]
        for i in 1..n {
            let mut best = i64::MAX;
            for j in 0..i {
                let cand = a[j] * x[i] + b_coef[j];
                if cand < best {
                    best = cand;
                }
            }
            dp[i] = best;
        }
        dp
    }

    /// CHT version of the same recurrence, exploiting non-increasing `a`
    /// (slopes) and non-decreasing `x` (queries).
    fn cht_dp(a: &[i64], b_coef: &[i64], x: &[i64]) -> Vec<i64> {
        let n = a.len();
        let mut dp = vec![0_i64; n];
        let mut cht = LineContainer::new();
        cht.add_line(a[0], b_coef[0]);
        dp[0] = cht.query(x[0]);
        for i in 1..n {
            // For this DP shape, lines come from j < i; we add line for j = i-1
            // before answering query for i.  Already added line 0 above; for i=1
            // we still need line 0 only (j < 1).  So add line for i-1 only when
            // i-1 >= 1.
            if i >= 1 {
                // Line j = i - 1 was already added in iteration i - 1 via this
                // branch when i - 1 >= 1, except for i = 1 we already added
                // line 0 outside the loop.  To keep this simple: add at start
                // of each iteration except i = 1.
            }
            if i >= 2 {
                cht.add_line(a[i - 1], b_coef[i - 1]);
            }
            dp[i] = cht.query(x[i]);
        }
        dp
    }

    #[test]
    fn dp_application_matches_brute() {
        // Slopes non-increasing, queries non-decreasing.
        let a = [5_i64, 3, 2, 1, 0]; // slopes
        let b = [0_i64, 4, 6, 8, 12]; // intercepts
        let x = [0_i64, 1, 3, 5, 8]; // query points
        let cht_res = cht_dp(&a, &b, &x);
        let brute_res = brute_dp(&a, &b, &x);
        assert_eq!(cht_res, brute_res);
    }

    // ── Property-based test ──────────────────────────────────────────────────

    #[cfg(test)]
    mod qc {
        use super::*;
        use quickcheck::TestResult;
        use quickcheck_macros::quickcheck;

        /// Property: querying the CHT after adding lines in non-increasing
        /// slope order, with non-decreasing query x's, matches brute-force
        /// minimum over the full line set.
        #[quickcheck]
        #[allow(clippy::needless_pass_by_value)]
        fn prop_matches_brute_force(
            slopes_raw: Vec<i16>,
            ints_raw: Vec<i16>,
            xs_raw: Vec<i16>,
        ) -> TestResult {
            // Bound sizes for speed.
            if slopes_raw.is_empty()
                || slopes_raw.len() > 12
                || ints_raw.is_empty()
                || xs_raw.is_empty()
            {
                return TestResult::discard();
            }
            // Truncate to common length and produce monotonic inputs.
            let n = slopes_raw.len().min(ints_raw.len()).min(12);
            let mut slopes: Vec<i64> = slopes_raw[..n].iter().map(|&v| i64::from(v)).collect();
            slopes.sort_unstable();
            slopes.reverse(); // non-increasing
            let intercepts: Vec<i64> = ints_raw[..n].iter().map(|&v| i64::from(v)).collect();

            let mut xs: Vec<i64> = xs_raw.iter().take(8).map(|&v| i64::from(v)).collect();
            if xs.is_empty() {
                return TestResult::discard();
            }
            xs.sort_unstable(); // non-decreasing

            let lines: Vec<(i64, i64)> = slopes
                .iter()
                .copied()
                .zip(intercepts.iter().copied())
                .collect();

            let mut cht = LineContainer::new();
            for &(m, b) in &lines {
                cht.add_line(m, b);
            }
            for &x in &xs {
                let got = cht.query(x);
                let want = lines
                    .iter()
                    .map(|&(m, b)| m * x + b)
                    .min()
                    .expect("non-empty");
                if got != want {
                    return TestResult::failed();
                }
            }
            TestResult::passed()
        }
    }
}
