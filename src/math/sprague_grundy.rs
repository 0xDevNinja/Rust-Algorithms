//! Sprague-Grundy theorem helpers for impartial combinatorial games.
//!
//! Provides:
//! - `mex`: minimum excludant of a multiset of non-negative integers.
//! - `grundy`: memoised Sprague-Grundy number of a state given a move
//!   function. The state space reachable from the initial state must be
//!   finite and acyclic; otherwise the recursion will not terminate.
//! - `nim_winner`: first-player win check for classic Nim, which by the
//!   Sprague-Grundy theorem reduces to the XOR of pile sizes.
//!
//! Complexity:
//! - `mex(values)`: O(n log n) due to sorting a copy of the input.
//! - `grundy(state, moves)`: O(S * (M + log M)) where S is the number of
//!   reachable states and M is the maximum branching factor; each state's
//!   Grundy number is computed once and cached in a `HashMap`.
//! - `nim_winner(piles)`: O(n).

use std::collections::HashMap;

/// Returns the minimum excludant (smallest non-negative integer absent
/// from `values`).
///
/// Duplicates in the input are permitted and ignored.
pub fn mex(values: &[u32]) -> u32 {
    let mut sorted: Vec<u32> = values.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    let mut expected: u32 = 0;
    for v in sorted {
        if v == expected {
            expected += 1;
        } else if v > expected {
            break;
        }
    }
    expected
}

/// Computes the Sprague-Grundy number of `state`, where `moves(s)` lists
/// the states reachable from `s` in one move. A terminal state (no moves)
/// has Grundy value 0.
///
/// Precondition: the state graph reachable from `state` must be finite
/// and acyclic. Cyclic move graphs would cause infinite recursion.
pub fn grundy<F>(state: u64, moves: F) -> u32
where
    F: Fn(u64) -> Vec<u64>,
{
    let mut memo: HashMap<u64, u32> = HashMap::new();
    grundy_inner(state, &moves, &mut memo)
}

fn grundy_inner<F>(state: u64, moves: &F, memo: &mut HashMap<u64, u32>) -> u32
where
    F: Fn(u64) -> Vec<u64>,
{
    if let Some(&g) = memo.get(&state) {
        return g;
    }
    let next_states = moves(state);
    let child_grundies: Vec<u32> = next_states
        .into_iter()
        .map(|s| grundy_inner(s, moves, memo))
        .collect();
    let g = mex(&child_grundies);
    memo.insert(state, g);
    g
}

/// Returns true if the first player wins ordinary Nim on `piles`.
///
/// By the Sprague-Grundy theorem the Grundy value of a Nim position is
/// the XOR of the pile sizes; the first player wins iff this XOR is
/// non-zero.
pub fn nim_winner(piles: &[u64]) -> bool {
    piles.iter().fold(0u64, |acc, &p| acc ^ p) != 0
}

#[cfg(test)]
mod tests {
    use super::{grundy, mex, nim_winner};

    #[test]
    fn mex_empty() {
        assert_eq!(mex(&[]), 0);
    }

    #[test]
    fn mex_contiguous_from_zero() {
        assert_eq!(mex(&[0, 1, 2]), 3);
    }

    #[test]
    fn mex_missing_zero() {
        assert_eq!(mex(&[1, 3]), 0);
    }

    #[test]
    fn mex_skips_one() {
        assert_eq!(mex(&[0, 2, 4]), 1);
    }

    #[test]
    fn mex_with_duplicates() {
        assert_eq!(mex(&[0, 0, 1, 1, 2, 2]), 3);
    }

    #[test]
    fn nim_first_player_wins() {
        // 3 ^ 4 ^ 5 = 2 ≠ 0
        assert!(nim_winner(&[3, 4, 5]));
    }

    #[test]
    fn nim_first_player_loses() {
        // 1 ^ 1 = 0
        assert!(!nim_winner(&[1, 1]));
    }

    #[test]
    fn nim_empty_is_loss() {
        assert!(!nim_winner(&[]));
    }

    #[test]
    fn nim_single_nonempty_pile_is_win() {
        assert!(nim_winner(&[7]));
    }

    #[test]
    fn subtraction_game_grundy_is_n_mod_4() {
        // Moves: remove 1, 2, or 3 stones from a single pile.
        let moves = |s: u64| -> Vec<u64> {
            let mut out = Vec::new();
            for k in 1..=3u64 {
                if s >= k {
                    out.push(s - k);
                }
            }
            out
        };
        for n in 0..=15u64 {
            assert_eq!(grundy(n, moves), (n % 4) as u32, "n = {n}");
        }
    }

    #[test]
    fn staircase_nim_xor_of_odd_positions() {
        // Staircase Nim. Stairs are indexed 0..N; stones on stair 0 are
        // dead (the "ground"). A move picks any stair i >= 1 with stones
        // and shifts k (1..=stones) of them onto stair i-1. Game ends
        // when stones only sit on stair 0.
        //
        // Encoding: 4 stairs as base-256 digits in a u64, lowest stair
        // in the low byte.
        //
        // Known result: Grundy value equals XOR of pile sizes on
        // odd-indexed stairs (stair 1, stair 3, ...).
        let stair = |state: u64, i: u32| -> u64 { (state >> (8 * i)) & 0xff };
        let set_stair = |state: u64, i: u32, v: u64| -> u64 {
            (state & !(0xffu64 << (8 * i))) | ((v & 0xff) << (8 * i))
        };

        let moves = move |s: u64| -> Vec<u64> {
            let mut out = Vec::new();
            // Only stairs 1..=3 are active; stair 0 is the ground.
            for i in 1..4u32 {
                let here = stair(s, i);
                if here == 0 {
                    continue;
                }
                for k in 1..=here {
                    let after_here = set_stair(s, i, here - k);
                    let below = stair(after_here, i - 1);
                    let next = set_stair(after_here, i - 1, below + k);
                    out.push(next);
                }
            }
            out
        };

        // Pack four stairs (s0, s1, s2, s3) into a single u64.
        let pack = |s0: u64, s1: u64, s2: u64, s3: u64| -> u64 {
            s0 | (s1 << 8) | (s2 << 16) | (s3 << 24)
        };

        // Odd-indexed stairs are 1 and 3, so expected Grundy = s1 ^ s3.
        let cases: [(u64, u64, u64, u64); 7] = [
            (0, 0, 0, 0),
            (5, 0, 0, 0), // only ground stones -> no moves -> Grundy 0
            (0, 3, 0, 0),
            (0, 0, 2, 0),
            (0, 1, 0, 1), // 1 ^ 1 = 0
            (0, 2, 1, 3), // 2 ^ 3 = 1
            (4, 1, 2, 2), // 1 ^ 2 = 3
        ];
        let moves_ref = &moves;
        for (s0, s1, s2, s3) in cases {
            let state = pack(s0, s1, s2, s3);
            let expected = (s1 ^ s3) as u32;
            assert_eq!(
                grundy(state, moves_ref),
                expected,
                "state = ({s0}, {s1}, {s2}, {s3})"
            );
        }
    }
}
