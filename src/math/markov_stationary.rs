//! Stationary distribution of a finite-state Markov chain.
//!
//! A row-stochastic matrix `P` describes a chain on `n` states where
//! `P[i][j]` is the probability of moving from state `i` to state `j`.
//! A distribution `π` is *stationary* iff `π P = π` and `Σπ = 1`.
//!
//! Two solvers are provided:
//!
//! * [`stationary_power_iteration`] — repeatedly apply `P` to the uniform
//!   vector until the L1 change drops below `tol` or `max_iter` is hit.
//!   For an irreducible aperiodic chain this converges to the unique
//!   stationary vector.
//! * [`stationary_solve`] — solve the homogeneous system `(Pᵀ − I) π = 0`
//!   together with the constraint `Σπ = 1` by Gaussian elimination on a
//!   small `(n+1) × n` augmented system. An inline eliminator is used so
//!   this module does not depend on any sibling linear-algebra module.
//!
//! Complexity:
//!   * power iteration — `O(k · n²)` for `k` iterations,
//!   * direct solve   — `O(n³)`.

/// Tolerance used when validating that the input matrix is row-stochastic.
const ROW_SUM_TOL: f64 = 1.0e-9;

/// Returns `true` iff `p` is square and every row sums to 1 within
/// `ROW_SUM_TOL`, with all entries finite and non-negative.
fn is_row_stochastic(p: &[Vec<f64>]) -> bool {
    let n = p.len();
    if n == 0 {
        return false;
    }
    for row in p {
        if row.len() != n {
            return false;
        }
        let mut sum = 0.0;
        for &v in row {
            if !v.is_finite() || v < 0.0 {
                return false;
            }
            sum += v;
        }
        if (sum - 1.0).abs() > ROW_SUM_TOL {
            return false;
        }
    }
    true
}

/// Power-iteration stationary distribution.
///
/// Starts from the uniform distribution and repeatedly applies the
/// row-stochastic transition matrix `p` (i.e. `π ← π · P`). Stops when
/// the L1 change between successive iterates is below `tol` or after
/// `max_iter` iterations.
///
/// Returns `None` if `p` is not square / not row-stochastic, if `tol`
/// is non-positive or non-finite, or if convergence was not reached
/// within `max_iter` iterations.
///
/// On the identity matrix every distribution is stationary; the
/// returned vector is the uniform one because that is the chosen
/// initial guess.
pub fn stationary_power_iteration(p: &[Vec<f64>], tol: f64, max_iter: u32) -> Option<Vec<f64>> {
    if !is_row_stochastic(p) || !tol.is_finite() || tol <= 0.0 {
        return None;
    }
    let n = p.len();
    let mut pi = vec![1.0 / n as f64; n];
    let mut next = vec![0.0_f64; n];

    for _ in 0..max_iter {
        // next[j] = Σ_i pi[i] * p[i][j]
        next.fill(0.0);
        for i in 0..n {
            let pi_i = pi[i];
            if pi_i == 0.0 {
                continue;
            }
            for j in 0..n {
                next[j] += pi_i * p[i][j];
            }
        }
        // Renormalise to guard against floating drift.
        let s: f64 = next.iter().sum();
        if !s.is_finite() || s <= 0.0 {
            return None;
        }
        for v in &mut next {
            *v /= s;
        }
        let diff: f64 = pi.iter().zip(next.iter()).map(|(a, b)| (a - b).abs()).sum();
        std::mem::swap(&mut pi, &mut next);
        if diff < tol {
            return Some(pi);
        }
    }
    None
}

/// Direct stationary distribution by Gaussian elimination.
///
/// Builds the augmented `(n+1) × n` system consisting of `Pᵀ − I`
/// (which gives `n` rows of `(Pᵀ − I) π = 0`) plus the normalisation
/// row `[1, 1, …, 1]` with right-hand side `1`. A small inline
/// eliminator with partial pivoting is then used to solve it in the
/// least-squares sense by reducing to an `n × n` system after dropping
/// one redundant balance row.
///
/// Returns `None` if `p` is not square / not row-stochastic, or if the
/// reduced system is singular.
///
/// On the identity matrix the system is rank-deficient with infinitely
/// many solutions; the returned vector is whichever is selected by the
/// elimination's tie-break (and is therefore not specified beyond
/// `Σπ = 1`).
pub fn stationary_solve(p: &[Vec<f64>]) -> Option<Vec<f64>> {
    if !is_row_stochastic(p) {
        return None;
    }
    let n = p.len();

    // Build A = Pᵀ − I of shape n × n, b = 0.
    // Replace the last balance equation with Σπ = 1 to pin the solution.
    let mut aug = vec![vec![0.0_f64; n + 1]; n];
    for i in 0..n {
        for j in 0..n {
            aug[i][j] = p[j][i]; // Pᵀ[i][j] = P[j][i]
        }
        aug[i][i] -= 1.0;
        aug[i][n] = 0.0;
    }
    // Replace last row with the normalisation constraint.
    for j in 0..n {
        aug[n - 1][j] = 1.0;
    }
    aug[n - 1][n] = 1.0;

    // Gaussian elimination with partial pivoting.
    for col in 0..n {
        // Find pivot.
        let mut pivot = col;
        let mut best = aug[col][col].abs();
        for row in (col + 1)..n {
            let v = aug[row][col].abs();
            if v > best {
                best = v;
                pivot = row;
            }
        }
        if best < 1.0e-12 {
            return None;
        }
        if pivot != col {
            aug.swap(col, pivot);
        }
        // Eliminate below.
        for row in (col + 1)..n {
            let factor = aug[row][col] / aug[col][col];
            if factor == 0.0 {
                continue;
            }
            for k in col..=n {
                aug[row][k] -= factor * aug[col][k];
            }
        }
    }

    // Back substitution.
    let mut pi = vec![0.0_f64; n];
    for i in (0..n).rev() {
        let mut s = aug[i][n];
        for j in (i + 1)..n {
            s -= aug[i][j] * pi[j];
        }
        let diag = aug[i][i];
        if diag.abs() < 1.0e-12 {
            return None;
        }
        pi[i] = s / diag;
    }

    // Sanity: clamp tiny negatives that came from floating noise and renormalise.
    let mut sum = 0.0;
    for v in &mut pi {
        if *v < 0.0 && *v > -1.0e-9 {
            *v = 0.0;
        }
        if !v.is_finite() || *v < 0.0 {
            return None;
        }
        sum += *v;
    }
    if !sum.is_finite() || sum <= 0.0 {
        return None;
    }
    for v in &mut pi {
        *v /= sum;
    }
    Some(pi)
}

#[cfg(test)]
mod tests {
    use super::{stationary_power_iteration, stationary_solve};

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    fn vec_approx(a: &[f64], b: &[f64], tol: f64) -> bool {
        a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| approx(*x, *y, tol))
    }

    #[test]
    fn two_state_closed_form_power() {
        let p = vec![vec![0.7, 0.3], vec![0.4, 0.6]];
        let pi = stationary_power_iteration(&p, 1.0e-12, 10_000).unwrap();
        let expected = [4.0 / 7.0, 3.0 / 7.0];
        assert!(vec_approx(&pi, &expected, 1.0e-9));
        assert!(approx(pi.iter().sum::<f64>(), 1.0, 1.0e-12));
    }

    #[test]
    fn two_state_closed_form_solve() {
        let p = vec![vec![0.7, 0.3], vec![0.4, 0.6]];
        let pi = stationary_solve(&p).unwrap();
        let expected = [4.0 / 7.0, 3.0 / 7.0];
        assert!(vec_approx(&pi, &expected, 1.0e-12));
    }

    #[test]
    fn three_state_known_stationary() {
        // Doubly-stochastic 3×3: stationary distribution is uniform [1/3, 1/3, 1/3].
        let p = vec![
            vec![0.5, 0.25, 0.25],
            vec![0.25, 0.5, 0.25],
            vec![0.25, 0.25, 0.5],
        ];
        let pi_iter = stationary_power_iteration(&p, 1.0e-12, 10_000).unwrap();
        let pi_solve = stationary_solve(&p).unwrap();
        let expected = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0];
        assert!(vec_approx(&pi_iter, &expected, 1.0e-9));
        assert!(vec_approx(&pi_solve, &expected, 1.0e-12));
    }

    #[test]
    fn three_state_asymmetric() {
        // Hand-verified: stationary [0.2, 0.4, 0.4].
        // π P = π:
        //   0.2 = 0.2*0   + 0.4*0.5 + 0.4*0.25 = 0.0 + 0.2 + 0.1 -> 0.3? Recompute.
        // We instead use a chain whose stationary we derive symbolically.
        // P below is the random walk on the path 0-1-2 with self-loops.
        // Detailed-balance check:
        //   π[0]*P[0][1] = π[1]*P[1][0]: 0.25 * 0.5  = 0.5 * 0.25 ✓
        //   π[1]*P[1][2] = π[2]*P[2][1]: 0.5  * 0.25 = 0.25 * 0.5 ✓
        let p = vec![
            vec![0.5, 0.5, 0.0],
            vec![0.25, 0.5, 0.25],
            vec![0.0, 0.5, 0.5],
        ];
        let expected = [0.25, 0.5, 0.25];
        let pi_iter = stationary_power_iteration(&p, 1.0e-12, 100_000).unwrap();
        let pi_solve = stationary_solve(&p).unwrap();
        assert!(vec_approx(&pi_iter, &expected, 1.0e-8));
        assert!(vec_approx(&pi_solve, &expected, 1.0e-12));
    }

    #[test]
    fn returned_vector_is_a_distribution() {
        let p = vec![vec![0.7, 0.3], vec![0.4, 0.6]];
        let pi_iter = stationary_power_iteration(&p, 1.0e-12, 10_000).unwrap();
        let pi_solve = stationary_solve(&p).unwrap();
        assert_eq!(pi_iter.len(), 2);
        assert_eq!(pi_solve.len(), 2);
        assert!(approx(pi_iter.iter().sum::<f64>(), 1.0, 1.0e-12));
        assert!(approx(pi_solve.iter().sum::<f64>(), 1.0, 1.0e-12));
        for v in pi_iter.iter().chain(pi_solve.iter()) {
            assert!(*v >= 0.0);
        }
    }

    #[test]
    fn identity_power_iteration_returns_uniform() {
        // Every distribution is stationary; power iteration's tie-break
        // is the uniform initial guess.
        let n = 4;
        let mut p = vec![vec![0.0; n]; n];
        for i in 0..n {
            p[i][i] = 1.0;
        }
        let pi = stationary_power_iteration(&p, 1.0e-12, 10).unwrap();
        let uniform = vec![1.0 / n as f64; n];
        assert!(vec_approx(&pi, &uniform, 1.0e-15));
    }

    #[test]
    fn identity_solve_returns_some_distribution() {
        // Reducible: tie-break is whatever Gaussian elimination picks.
        // We only require the result is a valid distribution.
        let n = 3;
        let mut p = vec![vec![0.0; n]; n];
        for i in 0..n {
            p[i][i] = 1.0;
        }
        if let Some(pi) = stationary_solve(&p) {
            assert_eq!(pi.len(), n);
            assert!(approx(pi.iter().sum::<f64>(), 1.0, 1.0e-12));
            for v in &pi {
                assert!(*v >= 0.0);
            }
        }
        // None is also acceptable (singular reduced system).
    }

    #[test]
    fn reducible_chain_either_outcome_is_acceptable() {
        // Two absorbing states: chain is reducible, so the stationary
        // distribution is not unique. Power iteration may or may not
        // converge depending on the starting vector; we only assert
        // that *if* it returns Some, the result is a valid distribution.
        let p = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.5, 0.5, 0.0, 0.0],
            vec![0.0, 0.0, 0.5, 0.5],
            vec![0.0, 0.0, 0.0, 1.0],
        ];
        if let Some(pi) = stationary_power_iteration(&p, 1.0e-12, 10_000) {
            assert_eq!(pi.len(), 4);
            assert!(approx(pi.iter().sum::<f64>(), 1.0, 1.0e-9));
            for v in &pi {
                assert!(*v >= -1.0e-12);
            }
        }
    }

    #[test]
    fn solvers_agree_on_irreducible_aperiodic() {
        let chains: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.7, 0.3], vec![0.4, 0.6]],
            vec![vec![0.1, 0.9], vec![0.5, 0.5]],
            vec![
                vec![0.2, 0.5, 0.3],
                vec![0.4, 0.4, 0.2],
                vec![0.1, 0.3, 0.6],
            ],
            vec![
                vec![0.5, 0.25, 0.25],
                vec![0.25, 0.5, 0.25],
                vec![0.25, 0.25, 0.5],
            ],
        ];
        for p in &chains {
            let pi_iter = stationary_power_iteration(p, 1.0e-14, 1_000_000).unwrap();
            let pi_solve = stationary_solve(p).unwrap();
            assert!(vec_approx(&pi_iter, &pi_solve, 1.0e-7));
        }
    }

    #[test]
    fn rejects_non_square_matrix() {
        let p = vec![vec![0.5, 0.5], vec![0.5, 0.5], vec![0.5, 0.5]];
        assert!(stationary_power_iteration(&p, 1.0e-9, 10).is_none());
        assert!(stationary_solve(&p).is_none());
    }

    #[test]
    fn rejects_non_row_stochastic() {
        let p = vec![vec![0.6, 0.6], vec![0.4, 0.6]];
        assert!(stationary_power_iteration(&p, 1.0e-9, 10).is_none());
        assert!(stationary_solve(&p).is_none());
    }

    #[test]
    fn rejects_negative_entries() {
        let p = vec![vec![1.2, -0.2], vec![0.4, 0.6]];
        assert!(stationary_power_iteration(&p, 1.0e-9, 10).is_none());
        assert!(stationary_solve(&p).is_none());
    }

    #[test]
    fn rejects_empty_matrix() {
        let p: Vec<Vec<f64>> = Vec::new();
        assert!(stationary_power_iteration(&p, 1.0e-9, 10).is_none());
        assert!(stationary_solve(&p).is_none());
    }

    #[test]
    fn rejects_bad_tolerance() {
        let p = vec![vec![0.7, 0.3], vec![0.4, 0.6]];
        assert!(stationary_power_iteration(&p, 0.0, 10).is_none());
        assert!(stationary_power_iteration(&p, -1.0e-9, 10).is_none());
        assert!(stationary_power_iteration(&p, f64::NAN, 10).is_none());
    }

    #[test]
    fn does_not_converge_with_tiny_iter_budget() {
        // A non-doubly-stochastic chain: starting from the uniform
        // distribution the iterate moves towards the true stationary
        // [4/7, 3/7], which is far from uniform, so a single iteration
        // at machine-precision tolerance cannot satisfy convergence.
        let p = vec![vec![0.7, 0.3], vec![0.4, 0.6]];
        assert!(stationary_power_iteration(&p, 1.0e-15, 1).is_none());
    }
}
