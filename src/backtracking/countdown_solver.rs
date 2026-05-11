//! Le Compte est Bon (Countdown numbers round) solver.
//!
//! Given a multiset of small positive integers (classically six numbers,
//! each at most `999`) and a target value (also at most `999`), find any
//! arithmetic expression that:
//!
//! - uses each input number **at most once**, and
//! - combines the chosen numbers with the four binary operators
//!   `+`, `-`, `*`, `/`,
//!
//! whose value equals the target. Division is integer division and is
//! only allowed when the dividend is exactly divisible by the divisor;
//! all intermediate values must be positive integers (the original
//! televised game forbids zero and negative intermediates).
//!
//! # Algorithm
//! Backtracking by combination. At each step the search holds a list of
//! currently available `(value, expression)` pairs. It picks an unordered
//! pair `(a, b)` from the list, applies every operator that yields a
//! positive integer result, and recurses on the reduced list with the new
//! value substituted in. A solution is reported as soon as any reachable
//! value equals the target — including any singleton input that already
//! matches.
//!
//! Operator pruning:
//! - `a + b` is only tried with `a >= b` (commutative, dedup).
//! - `a * b` is only tried with `a >= b` (commutative, dedup) and
//!   skipped when `a == 1` or `b == 1` (multiplying by `1` is a no-op).
//! - `a - b` is only tried with `a > b` (must stay positive; `a == b`
//!   would produce `0`, which is forbidden).
//! - `a / b` is only tried when `b != 0`, `b != 1`, and `a % b == 0`
//!   (must be exact; dividing by `1` is a no-op).
//!
//! These prunings preserve completeness: every reachable target value is
//! still found, but each unordered combination is explored once.
//!
//! # Output
//! [`countdown`] returns `Some(expr)` for any exact solution, where `expr`
//! is a fully-parenthesised infix string such as `"((25*7)+(50/(10-3+1)))"`
//! that evaluates to the target under standard integer arithmetic. The
//! exact expression returned depends on the search order and is not
//! guaranteed to be canonical or shortest. Returns `None` when no exact
//! solution exists.
//!
//! # Complexity
//! Let `n` be the number of input numbers. The search explores at most
//! `O(n! * 4^(n-1))` operator/order combinations in the worst case
//! (loosely: pick a pair, pick an op, recurse on `n - 1` items). Pruning
//! and early exit on first solution make typical `n = 6` Countdown
//! puzzles solve in milliseconds.
//! - **Time**: exponential in `n`; bounded by `O(n! * 4^(n-1))`.
//! - **Space**: `O(n)` recursion depth plus `O(n)` per-frame scratch for
//!   the residual list and expression strings.

/// Solves a Countdown numbers puzzle.
///
/// Returns `Some(expr)` where `expr` is a fully-parenthesised infix
/// expression that uses each entry of `nums` at most once, combines them
/// with `+ - * /` (integer division, exact only), and evaluates to
/// `target`. Returns `None` if no such expression exists, or if any
/// input number or the target is non-positive.
///
/// All intermediate values must be positive integers: subtractions that
/// would produce zero or a negative number, and divisions that are not
/// exact, are rejected during the search. This matches the rules of the
/// televised game *Le Compte est Bon* / Countdown.
///
/// # Examples
///
/// ```
/// use rust_algorithms::backtracking::countdown_solver::countdown;
///
/// // A singleton equal to the target trivially solves itself.
/// assert_eq!(countdown(&[5], 5).as_deref(), Some("5"));
///
/// // No solution on a tiny set: 2 and 3 cannot reach 100.
/// assert!(countdown(&[2, 3], 100).is_none());
/// ```
pub fn countdown(nums: &[i64], target: i64) -> Option<String> {
    if target <= 0 || nums.iter().any(|&x| x <= 0) {
        return None;
    }
    let mut state: Vec<(i64, String)> = nums.iter().map(|&n| (n, n.to_string())).collect();
    solve(&mut state, target)
}

/// Recursive worker. Tries every unordered pair from `state` and every
/// valid operator, returning the first expression whose value equals
/// `target`.
fn solve(state: &mut Vec<(i64, String)>, target: i64) -> Option<String> {
    // Any current term that already equals the target is an answer.
    for (v, e) in state.iter() {
        if *v == target {
            return Some(e.clone());
        }
    }
    if state.len() < 2 {
        return None;
    }

    let n = state.len();
    for i in 0..n {
        for j in (i + 1)..n {
            // Borrow-friendly: clone the two endpoints, then mutate `state`.
            let (a_val, a_expr) = state[i].clone();
            let (b_val, b_expr) = state[j].clone();

            for combined in combine(a_val, &a_expr, b_val, &b_expr) {
                // Build the residual list: remove indices i and j, push combined.
                // Remove the larger index first so the smaller index stays valid.
                let removed_j = state.remove(j);
                let removed_i = state.remove(i);
                state.push(combined);

                if let Some(answer) = solve(state, target) {
                    return Some(answer);
                }

                // Restore state for the next iteration.
                state.pop();
                state.insert(i, removed_i);
                state.insert(j, removed_j);
            }
        }
    }
    None
}

/// Returns every `(value, expression)` reachable by applying one of the
/// four operators to the unordered pair `(a, b)`. Each result is a
/// positive integer; commutative duplicates and no-op operations
/// (`* 1`, `/ 1`) are pruned.
fn combine(a: i64, a_expr: &str, b: i64, b_expr: &str) -> Vec<(i64, String)> {
    // Order so `hi >= lo` for the commutative + and *, and so subtraction
    // / division try the larger-over-smaller form (the only one that can
    // stay a positive integer).
    let (hi, hi_expr, lo, lo_expr) = if a >= b {
        (a, a_expr, b, b_expr)
    } else {
        (b, b_expr, a, a_expr)
    };

    let mut out: Vec<(i64, String)> = Vec::with_capacity(4);

    // Addition (commutative): always valid for positive integers.
    out.push((hi + lo, format!("({hi_expr}+{lo_expr})")));

    // Subtraction: must stay strictly positive; `hi == lo` would give 0.
    if hi > lo {
        out.push((hi - lo, format!("({hi_expr}-{lo_expr})")));
    }

    // Multiplication (commutative): skip `* 1`, which is a no-op.
    if lo != 1 {
        out.push((hi * lo, format!("({hi_expr}*{lo_expr})")));
    }

    // Division: must be exact; skip `/ 1`, which is a no-op.
    if lo != 0 && lo != 1 && hi % lo == 0 {
        out.push((hi / lo, format!("({hi_expr}/{lo_expr})")));
    }

    out
}

#[cfg(test)]
mod tests {
    use super::countdown;

    /// Recursively evaluates a fully-parenthesised expression of
    /// non-negative integers and `+ - * /` (exact integer division).
    /// Used to independently verify candidate solutions returned by
    /// [`countdown`].
    fn eval(expr: &str) -> i64 {
        let bytes = expr.as_bytes();
        let (value, end) = parse(bytes, 0);
        assert_eq!(end, bytes.len(), "trailing input in expression: {expr}");
        value
    }

    fn parse(bytes: &[u8], i: usize) -> (i64, usize) {
        if bytes[i] == b'(' {
            let (lhs, after_lhs) = parse(bytes, i + 1);
            let op = bytes[after_lhs];
            let (rhs, after_rhs) = parse(bytes, after_lhs + 1);
            assert_eq!(bytes[after_rhs], b')');
            let value = match op {
                b'+' => lhs + rhs,
                b'-' => lhs - rhs,
                b'*' => lhs * rhs,
                b'/' => {
                    assert!(rhs != 0 && lhs % rhs == 0, "non-exact division");
                    lhs / rhs
                }
                _ => panic!("unexpected operator {}", op as char),
            };
            (value, after_rhs + 1)
        } else {
            // Parse a non-negative integer.
            let mut j = i;
            while j < bytes.len() && bytes[j].is_ascii_digit() {
                j += 1;
            }
            let n: i64 = std::str::from_utf8(&bytes[i..j])
                .unwrap()
                .parse()
                .expect("integer literal");
            (n, j)
        }
    }

    #[test]
    fn singleton_equal_to_target_returns_itself() {
        assert_eq!(countdown(&[5], 5).as_deref(), Some("5"));
    }

    #[test]
    fn singleton_not_equal_to_target_returns_none() {
        assert!(countdown(&[5], 7).is_none());
    }

    #[test]
    fn classic_countdown_example_finds_some_valid_expression() {
        // The classic six-number Countdown set with a target chosen to be
        // reachable: 765 = (50 + 25) * (10 + 7) / (3 - 1 + 1)? Use a
        // simpler reachable target instead so we have ground truth.
        // 765 = (25 + 10) * 7 * 3 + 50 + 1 - 6? Use 765 = 50*10 + 25*10 + 15.
        // Simplest: 765 = (50 - 25) * 30 + 15 — but we don't have 30.
        // Verified target: 765 = (25 + 50/10) * (3 * 7) * ... too messy.
        // Use the iconic puzzle target 952 from {25,50,75,100,3,6}? Out
        // of input set. Pick a target we can reach: 765 = 7 * (50 + 10) +
        // 25 * (3 * 7 - 18) — overcomplicated.
        // Verified by hand: 765 = (((50 - 1) + 25) * 10) + (3 * ?). Skip.
        // Use a target that is provably reachable: 7 * (10 * (25 - 3)) -
        // 50 - 1 = 7 * 220 - 51 = 1540 - 51 = 1489. Not 765.
        //
        // Reachable construction: (25 - 10) * 50 + (3 * 7) - 6 + 1 = 750
        // + 21 + ... hmm. Just construct one: (50 + 25) * 10 + (7 + 3 + 1
        // + ?)/? Too fragile.
        //
        // Use target 100 from {1, 3, 7, 10, 25, 50}: 100 = (1 + 3) * 25.
        let nums = [1i64, 3, 7, 10, 25, 50];
        let solution = countdown(&nums, 100).expect("100 is reachable from this set");
        assert_eq!(eval(&solution), 100, "solution: {solution}");
    }

    #[test]
    fn classic_countdown_target_765_reachable() {
        // 765 = 15 * 51 = (10 + 3 + ...) * (50 + 1) ; let's verify:
        // (50 + 1) = 51, (10 + 3 + ?) — we have {7, 25} left. 7 + 3 + ...
        // Try: 765 = 50 * 10 + 25 * 10 + 15. We can only use 10 once.
        // Try: 765 = (50 + 25) * 10 + (7 + 3 + ?). 750 + 15 = 765 needs
        // 15 from {7, 3, 1} — yes: 15 = 7 * (3 - 1) + 1? = 14 + 1 = 15.
        // So 765 = (50 + 25) * 10 + 7 * (3 - 1) + 1 — uses each at most
        // once. Solver should find some such expression.
        let nums = [1i64, 3, 7, 10, 25, 50];
        let solution = countdown(&nums, 765).expect("765 is reachable from {1,3,7,10,25,50}");
        assert_eq!(eval(&solution), 765, "solution: {solution}");
    }

    #[test]
    fn unreachable_target_returns_none() {
        // From {2, 3} the reachable positive results are 2, 3, 5, 6.
        // 100 is far out of range.
        assert!(countdown(&[2, 3], 100).is_none());
    }

    #[test]
    fn unreachable_target_on_small_set() {
        // {1, 2} can only reach 1, 2, 3 (1+2). No way to get 7.
        assert!(countdown(&[1, 2], 7).is_none());
    }

    #[test]
    fn target_reachable_by_addition_only() {
        let solution = countdown(&[10, 20, 30], 60).expect("10+20+30=60");
        assert_eq!(eval(&solution), 60);
    }

    #[test]
    fn target_reachable_by_subtraction() {
        let solution = countdown(&[100, 25], 75).expect("100-25=75");
        assert_eq!(eval(&solution), 75);
    }

    #[test]
    fn target_reachable_by_multiplication() {
        let solution = countdown(&[6, 7], 42).expect("6*7=42");
        assert_eq!(eval(&solution), 42);
    }

    #[test]
    fn target_reachable_by_exact_division() {
        let solution = countdown(&[100, 4], 25).expect("100/4=25");
        assert_eq!(eval(&solution), 25);
    }

    #[test]
    fn non_exact_division_alone_does_not_count() {
        // {7, 2}: reachable = {7, 2, 9, 5, 14}. 3 is unreachable because
        // 7/2 is not exact.
        assert!(countdown(&[7, 2], 3).is_none());
    }

    #[test]
    fn solution_uses_each_number_at_most_once() {
        // Sanity check: solver should not duplicate inputs. Asking for
        // 4 from {2} alone is unreachable (would need 2+2 or 2*2).
        assert!(countdown(&[2], 4).is_none());
    }

    #[test]
    fn negative_or_zero_target_returns_none() {
        assert!(countdown(&[1, 2, 3], 0).is_none());
        assert!(countdown(&[1, 2, 3], -5).is_none());
    }

    #[test]
    fn empty_inputs_return_none_unless_target_is_zero_which_is_invalid() {
        // No numbers → no expression → None.
        assert!(countdown(&[], 5).is_none());
    }

    #[test]
    fn classic_six_number_target_solves() {
        // {1, 3, 7, 10, 25, 50} → 813 is reachable: 813 = (10 + 7) * (50
        // - 1) - 25 * 3 - ?  = 17 * 49 - 75 = 833 - 75 = 758. Not 813.
        // Just trust the solver to find any answer for a well-known
        // reachable target. Use 952 from a permissive set.
        // From {1, 3, 7, 10, 25, 50}: 952 = ? Hard by hand.
        // Use 200: 200 = (25 - 10) * (3 + 7) + 50. = 150 + 50 = 200. OK.
        let nums = [1i64, 3, 7, 10, 25, 50];
        let solution = countdown(&nums, 200).expect("200 is reachable");
        assert_eq!(eval(&solution), 200);
    }
}
