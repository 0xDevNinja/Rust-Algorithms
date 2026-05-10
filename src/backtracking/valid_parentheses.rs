//! Generate all well-formed combinations of `n` pairs of parentheses.
//!
//! # Algorithm
//! Backtracking on a running string buffer while tracking two counters:
//! `open` — the number of `'('` already placed, and `close` — the number of
//! `')'` already placed. At each step:
//! - Append `'('` whenever `open < n` (we still have unused opens).
//! - Append `')'` whenever `close < open` (every close must match a prior
//!   open, preserving prefix balance).
//! - When `open == n && close == n`, the buffer is a complete well-formed
//!   string and is cloned into the output.
//!
//! Each push is paired with a matching `pop` after recursion to restore
//! the buffer for the next branch.
//!
//! # Complexity
//! Let `C_n` be the `n`-th Catalan number (the count of valid combinations).
//! - **Time** `O(n · C_n)` — there are `C_n` outputs, each of length `2n`
//!   to materialise.
//! - **Space** `O(n)` auxiliary (recursion depth + working buffer),
//!   excluding the `O(n · C_n)` output itself.
//!
//! For reference, `C_n` for `n = 0..=8` is
//! `1, 1, 2, 5, 14, 42, 132, 429, 1430`.

/// Returns every well-formed string of `n` pairs of parentheses.
///
/// `n == 0` yields `vec![String::new()]` — the single empty string,
/// matching the convention `C_0 = 1`. The output count is the `n`-th
/// Catalan number.
pub fn generate_parenthesis(n: u32) -> Vec<String> {
    let mut out = Vec::new();
    let mut buf = String::with_capacity(2 * n as usize);
    backtrack(n, 0, 0, &mut buf, &mut out);
    out
}

/// Recursive helper for `generate_parenthesis`.
///
/// Maintains the invariants `open <= n` and `close <= open` so that every
/// completed string of length `2n` is well-formed by construction.
fn backtrack(n: u32, open: u32, close: u32, buf: &mut String, out: &mut Vec<String>) {
    if open == n && close == n {
        out.push(buf.clone());
        return;
    }
    if open < n {
        buf.push('(');
        backtrack(n, open + 1, close, buf, out);
        buf.pop();
    }
    if close < open {
        buf.push(')');
        backtrack(n, open, close + 1, buf, out);
        buf.pop();
    }
}

#[cfg(test)]
mod tests {
    use super::generate_parenthesis;
    use std::collections::HashSet;

    /// `C_n` for `n = 0..=8` — used to validate the output count.
    const CATALAN: [usize; 9] = [1, 1, 2, 5, 14, 42, 132, 429, 1430];

    /// Verify a string is a well-formed parenthesis sequence with exactly
    /// `n` pairs: length is `2n`, every prefix has `open >= close`, and the
    /// totals match.
    fn is_well_formed(s: &str, n: u32) -> bool {
        if s.len() != 2 * n as usize {
            return false;
        }
        let mut balance: i64 = 0;
        for c in s.chars() {
            match c {
                '(' => balance += 1,
                ')' => balance -= 1,
                _ => return false,
            }
            if balance < 0 {
                return false;
            }
        }
        balance == 0
    }

    #[test]
    fn n_zero_returns_single_empty_string() {
        let out = generate_parenthesis(0);
        assert_eq!(out, vec![String::new()]);
    }

    #[test]
    fn n_one_returns_single_pair() {
        let out = generate_parenthesis(1);
        assert_eq!(out, vec!["()".to_string()]);
    }

    #[test]
    fn n_two_returns_two_combinations() {
        let mut out = generate_parenthesis(2);
        out.sort();
        assert_eq!(out, vec!["(())".to_string(), "()()".to_string()]);
    }

    #[test]
    fn n_three_returns_five_combinations() {
        let out = generate_parenthesis(3);
        assert_eq!(out.len(), 5);
        let expected: HashSet<String> = ["((()))", "(()())", "(())()", "()(())", "()()()"]
            .iter()
            .map(|s| (*s).to_string())
            .collect();
        let got: HashSet<String> = out.into_iter().collect();
        assert_eq!(got, expected);
    }

    #[test]
    fn count_matches_catalan_numbers_up_to_eight() {
        for (n, &expected) in CATALAN.iter().enumerate() {
            let out = generate_parenthesis(n as u32);
            assert_eq!(
                out.len(),
                expected,
                "expected C_{n} = {expected} combinations for n = {n}",
            );
        }
    }

    #[test]
    fn all_outputs_are_well_formed_and_unique() {
        for n in 0u32..=6 {
            let out = generate_parenthesis(n);
            for s in &out {
                assert!(is_well_formed(s, n), "not well-formed: {s:?} (n = {n})");
            }
            let unique: HashSet<&String> = out.iter().collect();
            assert_eq!(unique.len(), out.len(), "duplicate outputs at n = {n}");
        }
    }
}
