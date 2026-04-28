//! 2-SAT decision procedure via the implication graph + strongly-connected
//! components. A 2-CNF formula `∧_k (l_k1 ∨ l_k2)` is satisfiable iff no
//! variable shares an SCC with its negation in the graph that has an edge
//! `¬a → b` and `¬b → a` for every clause `(a ∨ b)`.
//!
//! When satisfiable, an assignment is read off from the condensation: variable
//! `i` is set to `true` iff the SCC containing the literal "`i = true`"
//! appears *later* in topological order than the SCC containing "`i = false`".
//! Tarjan's SCC routine returns components in reverse-topological order
//! (sinks first → smaller index), so the rule becomes
//! `comp_id[true-literal] < comp_id[false-literal]`.
//!
//! Time and space: O(n + m) where `n` is the number of variables and `m` the
//! number of clauses.
//!
//! Encoding: variable `i` true is node `2 * i`, variable `i` false is node
//! `2 * i + 1`. The internal graph therefore has `2 * n` nodes.

use crate::graph::tarjan_scc::tarjan_scc;

/// Builder + solver for a 2-SAT instance over `n` boolean variables.
///
/// Add clauses with [`TwoSat::add_clause`], then call [`TwoSat::solve`] to
/// obtain a satisfying assignment (if one exists).
pub struct TwoSat {
    n: usize,
    implications: Vec<Vec<usize>>,
}

impl TwoSat {
    /// Creates a new 2-SAT instance with `n` boolean variables and no
    /// clauses. The internal implication graph is allocated with `2 * n`
    /// nodes (two per variable: one for `true`, one for `false`).
    #[must_use]
    pub fn new(n: usize) -> Self {
        Self {
            n,
            implications: vec![Vec::new(); 2 * n],
        }
    }

    /// Adds the disjunctive clause `(x ∨ y)` where the literal for variable
    /// `x_var` is `x_var = x_val` and likewise for `y`.
    ///
    /// Internally records the two contrapositives `¬x → y` and `¬y → x`,
    /// which together are equivalent to `x ∨ y`.
    ///
    /// # Panics
    /// Panics if either `x_var` or `y_var` is out of range (`>= n`).
    pub fn add_clause(&mut self, x_var: usize, x_val: bool, y_var: usize, y_val: bool) {
        assert!(
            x_var < self.n && y_var < self.n,
            "variable index out of range"
        );
        // Node for "v = true" is 2*v, "v = false" is 2*v + 1.
        // lit(v, val) is the node asserting v == val; neg(v, val) is its negation.
        let lit_x = 2 * x_var + usize::from(!x_val);
        let neg_x = 2 * x_var + usize::from(x_val);
        let lit_y = 2 * y_var + usize::from(!y_val);
        let neg_y = 2 * y_var + usize::from(y_val);
        // Clause (x ∨ y) is equivalent to (¬x → y) and (¬y → x).
        self.implications[neg_x].push(lit_y);
        self.implications[neg_y].push(lit_x);
    }

    /// Returns `Some(assignment)` of length `n` satisfying every added
    /// clause, or `None` when the formula is unsatisfiable.
    ///
    /// Runs Tarjan's SCC over the implication graph and inspects, for each
    /// variable, whether its two literal nodes ended up in the same SCC
    /// (UNSAT) or different SCCs. Time and space `O(n + m)`.
    #[must_use]
    pub fn solve(&self) -> Option<Vec<bool>> {
        // tarjan_scc returns components in reverse-topological order
        // (sinks first). Map each node to its component index in that order.
        let components = tarjan_scc(&self.implications);
        let mut comp_id = vec![0usize; 2 * self.n];
        for (idx, comp) in components.iter().enumerate() {
            for &node in comp {
                comp_id[node] = idx;
            }
        }

        let mut assignment = vec![false; self.n];
        for i in 0..self.n {
            let t_id = comp_id[2 * i];
            let f_id = comp_id[2 * i + 1];
            if t_id == f_id {
                return None;
            }
            // Tarjan numbers components in reverse-topological order (sinks
            // first → smaller index). The standard 2-SAT rule sets variable
            // `i = true` when its true-literal sits *later* in topological
            // order than its false-literal — equivalently, *earlier* in the
            // reverse-topological order Tarjan returns: smaller index.
            assignment[i] = t_id < f_id;
        }
        Some(assignment)
    }
}

#[cfg(test)]
mod tests {
    use super::TwoSat;
    use quickcheck_macros::quickcheck;

    /// Verify an assignment satisfies every clause in `clauses`.
    /// A clause is `(x_var, x_val, y_var, y_val)`.
    fn satisfies(assignment: &[bool], clauses: &[(usize, bool, usize, bool)]) -> bool {
        clauses
            .iter()
            .all(|&(x, xv, y, yv)| (assignment[x] == xv) || (assignment[y] == yv))
    }

    #[test]
    fn empty_instance_returns_empty_assignment() {
        let solver = TwoSat::new(0);
        assert_eq!(solver.solve(), Some(vec![]));
    }

    #[test]
    fn no_clauses_any_assignment_works() {
        let solver = TwoSat::new(3);
        let a = solver.solve().expect("trivially satisfiable");
        assert_eq!(a.len(), 3);
    }

    #[test]
    fn single_clause_is_satisfiable() {
        let mut solver = TwoSat::new(2);
        solver.add_clause(0, true, 1, false);
        let a = solver.solve().expect("sat");
        assert!(a[0] || !a[1]);
    }

    #[test]
    fn single_variable_conflict_is_unsat() {
        // (x ∨ x) AND (¬x ∨ ¬x): forces x=true and x=false.
        let mut solver = TwoSat::new(1);
        solver.add_clause(0, true, 0, true);
        solver.add_clause(0, false, 0, false);
        assert_eq!(solver.solve(), None);
    }

    #[test]
    fn classic_three_variable_example() {
        // (x0 ∨ x1) ∧ (¬x0 ∨ x2) ∧ (¬x1 ∨ ¬x2) ∧ (x0 ∨ ¬x2)
        let mut solver = TwoSat::new(3);
        solver.add_clause(0, true, 1, true);
        solver.add_clause(0, false, 2, true);
        solver.add_clause(1, false, 2, false);
        solver.add_clause(0, true, 2, false);
        let clauses = vec![
            (0, true, 1, true),
            (0, false, 2, true),
            (1, false, 2, false),
            (0, true, 2, false),
        ];
        let a = solver.solve().expect("sat");
        assert_eq!(a.len(), 3);
        assert!(satisfies(&a, &clauses));
    }

    #[test]
    fn implication_cycle_is_satisfiable() {
        // x0 → x1, x1 → x2, x2 → x0 — encoded as clauses
        // (¬x0 ∨ x1), (¬x1 ∨ x2), (¬x2 ∨ x0). All-true and all-false both
        // satisfy. Verify the solver returns one of them.
        let mut solver = TwoSat::new(3);
        solver.add_clause(0, false, 1, true);
        solver.add_clause(1, false, 2, true);
        solver.add_clause(2, false, 0, true);
        let a = solver.solve().expect("sat");
        assert!(a.iter().all(|&b| b) || a.iter().all(|&b| !b));
    }

    #[test]
    fn forced_assignment_via_unit_like_clauses() {
        // (x0 ∨ x0) forces x0=true, (¬x1 ∨ ¬x1) forces x1=false.
        let mut solver = TwoSat::new(2);
        solver.add_clause(0, true, 0, true);
        solver.add_clause(1, false, 1, false);
        let a = solver.solve().expect("sat");
        assert!(a[0]);
        assert!(!a[1]);
    }

    /// Brute-force: a formula over `n ≤ 5` variables is satisfiable iff some
    /// assignment in the `2^n` cube satisfies every clause.
    fn brute_force_sat(n: usize, clauses: &[(usize, bool, usize, bool)]) -> bool {
        for mask in 0u32..(1u32 << n) {
            let assignment: Vec<bool> = (0..n).map(|i| (mask >> i) & 1 == 1).collect();
            if satisfies(&assignment, clauses) {
                return true;
            }
        }
        false
    }

    #[allow(clippy::needless_pass_by_value)]
    #[quickcheck]
    fn matches_brute_force(raw: Vec<(u8, bool, u8, bool)>) -> bool {
        // Bound n ≤ 5 and ≤ 10 clauses by deriving from the input.
        let n: usize = ((raw.len() % 5) + 1).max(1);
        let clauses: Vec<(usize, bool, usize, bool)> = raw
            .into_iter()
            .take(10)
            .map(|(x, xv, y, yv)| ((x as usize) % n, xv, (y as usize) % n, yv))
            .collect();

        let mut solver = TwoSat::new(n);
        for &(x, xv, y, yv) in &clauses {
            solver.add_clause(x, xv, y, yv);
        }

        solver.solve().map_or_else(
            || !brute_force_sat(n, &clauses),
            |a| a.len() == n && satisfies(&a, &clauses),
        )
    }
}
