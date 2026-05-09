//! Minimax search with alpha-beta pruning.
//!
//! Given a two-player zero-sum game described by a [`GameState`], computes
//! the value of the position under perfect play, expanded up to a fixed
//! search depth.  Alpha-beta pruning skips subtrees that cannot improve the
//! best move already found, returning the same value as plain minimax while
//! visiting fewer nodes in practice.
//!
//! The maximiser tries to maximise [`GameState::evaluate`]; the minimiser
//! tries to minimise it.  Whose turn it is at a given state is reported by
//! [`GameState::maximizer`].
//!
//! ## Complexity
//!
//! For branching factor `b` and search depth `d`:
//!
//! | | Time | Space |
//! |---|---|---|
//! | Plain minimax | O(bᵈ) | O(d) |
//! | Alpha-beta (worst case) | O(bᵈ) | O(d) |
//! | Alpha-beta (best-ordered) | O(b^(d/2)) | O(d) |
//!
//! Space is the recursion depth; no transposition table is kept.

/// A two-player zero-sum game state usable with [`alpha_beta`].
///
/// Implementations provide a static evaluation, the legal successor states,
/// a terminal predicate, and which side is to move.
pub trait GameState: Clone {
    /// Returns `true` when the position has no legal continuations or is
    /// otherwise terminal (e.g. checkmate, draw).  The search stops descending
    /// from a terminal state and returns [`evaluate`](GameState::evaluate).
    fn is_terminal(&self) -> bool;

    /// Static evaluation of the position from the maximiser's perspective.
    /// Larger values favour the maximiser, smaller values favour the
    /// minimiser.
    fn evaluate(&self) -> i64;

    /// All legal successor states reachable in one move.
    fn moves(&self) -> Vec<Self>;

    /// `true` if the side to move at this state is the maximiser.
    fn maximizer(&self) -> bool;
}

/// Returns the minimax value of `state` searched to `depth` plies, using
/// alpha-beta pruning.
///
/// `alpha` and `beta` are the current search window; callers usually pass
/// `i64::MIN` and `i64::MAX` respectively.  When `depth` reaches zero, or
/// when the state is terminal, the static [`GameState::evaluate`] is
/// returned.  If the side to move has no legal moves at a non-terminal
/// state, the static evaluation is also returned.
///
/// The result is identical to plain minimax over the same tree; pruning
/// only changes which subtrees are visited, not the value at the root.
pub fn alpha_beta<S: GameState>(state: &S, depth: u32, alpha: i64, beta: i64) -> i64 {
    if depth == 0 || state.is_terminal() {
        return state.evaluate();
    }
    let children = state.moves();
    if children.is_empty() {
        return state.evaluate();
    }

    let mut alpha = alpha;
    let mut beta = beta;

    if state.maximizer() {
        let mut best = i64::MIN;
        for child in &children {
            let v = alpha_beta(child, depth - 1, alpha, beta);
            if v > best {
                best = v;
            }
            if best > alpha {
                alpha = best;
            }
            if alpha >= beta {
                break; // beta cutoff
            }
        }
        best
    } else {
        let mut best = i64::MAX;
        for child in &children {
            let v = alpha_beta(child, depth - 1, alpha, beta);
            if v < best {
                best = v;
            }
            if best < beta {
                beta = best;
            }
            if alpha >= beta {
                break; // alpha cutoff
            }
        }
        best
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::{alpha_beta, GameState};

    // ── Plain minimax baseline (no pruning) ──────────────────────────────────

    /// Reference minimax used as an oracle to verify alpha-beta agreement.
    fn plain_minimax<S: GameState>(state: &S, depth: u32) -> i64 {
        if depth == 0 || state.is_terminal() {
            return state.evaluate();
        }
        let children = state.moves();
        if children.is_empty() {
            return state.evaluate();
        }
        if state.maximizer() {
            let mut best = i64::MIN;
            for c in &children {
                let v = plain_minimax(c, depth - 1);
                if v > best {
                    best = v;
                }
            }
            best
        } else {
            let mut best = i64::MAX;
            for c in &children {
                let v = plain_minimax(c, depth - 1);
                if v < best {
                    best = v;
                }
            }
            best
        }
    }

    // ── Trivial single-state game ────────────────────────────────────────────

    /// Always terminal, with a fixed evaluation.
    #[derive(Clone)]
    struct Trivial(i64);

    impl GameState for Trivial {
        fn is_terminal(&self) -> bool {
            true
        }
        fn evaluate(&self) -> i64 {
            self.0
        }
        fn moves(&self) -> Vec<Self> {
            Vec::new()
        }
        fn maximizer(&self) -> bool {
            true
        }
    }

    #[test]
    fn trivial_returns_evaluate() {
        let s = Trivial(42);
        assert_eq!(alpha_beta(&s, 5, i64::MIN, i64::MAX), 42);
        let s = Trivial(-7);
        assert_eq!(alpha_beta(&s, 0, i64::MIN, i64::MAX), -7);
    }

    // ── Tiny Nim-like game ───────────────────────────────────────────────────
    //
    // Two piles; on each turn the side to move removes one stone from one
    // pile.  The player who cannot move (both piles empty) loses, i.e.
    // the *other* side has just made the winning move.  Evaluation is from
    // the maximiser's perspective: +1 if the maximiser wins, -1 if it loses.

    #[derive(Clone)]
    struct Nim {
        piles: [u32; 2],
        maximizer_to_move: bool,
    }

    impl Nim {
        fn new(a: u32, b: u32) -> Self {
            Self {
                piles: [a, b],
                maximizer_to_move: true,
            }
        }
    }

    impl GameState for Nim {
        fn is_terminal(&self) -> bool {
            self.piles[0] == 0 && self.piles[1] == 0
        }

        fn evaluate(&self) -> i64 {
            // Terminal means side-to-move has no stones to take and loses.
            if self.is_terminal() {
                if self.maximizer_to_move {
                    -1
                } else {
                    1
                }
            } else {
                0
            }
        }

        fn moves(&self) -> Vec<Self> {
            let mut out = Vec::new();
            for (i, &p) in self.piles.iter().enumerate() {
                if p > 0 {
                    let mut next = self.clone();
                    next.piles[i] = p - 1;
                    next.maximizer_to_move = !self.maximizer_to_move;
                    out.push(next);
                }
            }
            out
        }

        fn maximizer(&self) -> bool {
            self.maximizer_to_move
        }
    }

    #[test]
    fn nim_one_one_is_losing_for_mover() {
        // [1, 1] with maximiser to move: any move leaves [1, 0] or [0, 1]
        // with 1 stone left for the minimiser, who takes it and leaves the
        // empty position back on the maximiser, who then loses.
        let s = Nim::new(1, 1);
        assert_eq!(alpha_beta(&s, 4, i64::MIN, i64::MAX), -1);
    }

    #[test]
    fn nim_one_zero_is_winning_for_mover() {
        // Maximiser takes the only stone, leaving [0,0] on the minimiser,
        // who loses. Maximiser wins => +1.
        let s = Nim::new(1, 0);
        assert_eq!(alpha_beta(&s, 4, i64::MIN, i64::MAX), 1);
    }

    #[test]
    fn nim_two_one_is_winning_for_mover() {
        // From [2,1] maximiser plays to [1,1] (a known losing position for
        // the side to move), so wins => +1.
        let s = Nim::new(2, 1);
        assert_eq!(alpha_beta(&s, 6, i64::MIN, i64::MAX), 1);
    }

    // ── Tic-tac-toe scratch position ─────────────────────────────────────────
    //
    // Cells: 0 empty, 1 X (maximiser), 2 O (minimiser). Evaluation is +10
    // for an X line, -10 for an O line, 0 otherwise. is_terminal fires on
    // any completed line or a full board.

    #[derive(Clone)]
    struct Ttt {
        b: [u8; 9],
        x_to_move: bool,
    }

    impl Ttt {
        fn winner(&self) -> u8 {
            const LINES: [[usize; 3]; 8] = [
                [0, 1, 2],
                [3, 4, 5],
                [6, 7, 8],
                [0, 3, 6],
                [1, 4, 7],
                [2, 5, 8],
                [0, 4, 8],
                [2, 4, 6],
            ];
            for ln in &LINES {
                let a = self.b[ln[0]];
                if a != 0 && a == self.b[ln[1]] && a == self.b[ln[2]] {
                    return a;
                }
            }
            0
        }

        fn full(&self) -> bool {
            self.b.iter().all(|&c| c != 0)
        }
    }

    impl GameState for Ttt {
        fn is_terminal(&self) -> bool {
            self.winner() != 0 || self.full()
        }

        fn evaluate(&self) -> i64 {
            match self.winner() {
                1 => 10,
                2 => -10,
                _ => 0,
            }
        }

        fn moves(&self) -> Vec<Self> {
            if self.is_terminal() {
                return Vec::new();
            }
            let me = if self.x_to_move { 1 } else { 2 };
            let mut out = Vec::new();
            for i in 0..9 {
                if self.b[i] == 0 {
                    let mut nb = self.b;
                    nb[i] = me;
                    out.push(Self {
                        b: nb,
                        x_to_move: !self.x_to_move,
                    });
                }
            }
            out
        }

        fn maximizer(&self) -> bool {
            self.x_to_move
        }
    }

    #[test]
    fn ttt_forced_win_for_x() {
        // X to move with two-in-a-row across the top (cells 0 and 1).
        // Playing cell 2 wins immediately for X (+10).
        //
        //   X X .
        //   . O .
        //   . O .
        //
        // Even at depth 1 the win is forced.
        let s = Ttt {
            b: [1, 1, 0, 0, 2, 0, 0, 2, 0],
            x_to_move: true,
        };
        let v = alpha_beta(&s, 1, i64::MIN, i64::MAX);
        assert!(v > 0, "expected positive value for forced X win, got {v}");

        // Also true at greater search depth.
        let v_deep = alpha_beta(&s, 6, i64::MIN, i64::MAX);
        assert!(v_deep > 0);
    }

    #[test]
    fn ttt_empty_board_is_a_draw() {
        // Tic-tac-toe with perfect play from the empty board is a draw.
        let s = Ttt {
            b: [0; 9],
            x_to_move: true,
        };
        assert_eq!(alpha_beta(&s, 9, i64::MIN, i64::MAX), 0);
    }

    // ── Agreement with plain minimax ─────────────────────────────────────────

    #[test]
    fn alpha_beta_matches_plain_minimax_on_nim() {
        // Run both algorithms across a range of small Nim positions and
        // search depths, asserting identical values everywhere.  This
        // verifies that pruning is correct.
        for a in 0..=3 {
            for b in 0..=3 {
                for d in 0..=8 {
                    let s = Nim::new(a, b);
                    let ab = alpha_beta(&s, d, i64::MIN, i64::MAX);
                    let mm = plain_minimax(&s, d);
                    assert_eq!(ab, mm, "mismatch at piles=[{a},{b}] depth={d}");
                }
            }
        }
    }

    #[test]
    fn alpha_beta_matches_plain_minimax_on_ttt() {
        // A handful of mid-game tic-tac-toe positions.
        let positions: [[u8; 9]; 4] = [
            [1, 0, 0, 0, 2, 0, 0, 0, 0],
            [1, 2, 0, 0, 1, 0, 0, 0, 2],
            [1, 1, 0, 0, 2, 0, 0, 2, 0],
            [0, 1, 0, 2, 1, 0, 0, 2, 0],
        ];
        for b in &positions {
            // Whose turn from the cell counts.
            let mut xs = 0_usize;
            let mut os = 0_usize;
            for &c in b {
                match c {
                    1 => xs += 1,
                    2 => os += 1,
                    _ => {}
                }
            }
            let x_to_move = xs == os;
            let s = Ttt { b: *b, x_to_move };
            for d in 0..=5 {
                let ab = alpha_beta(&s, d, i64::MIN, i64::MAX);
                let mm = plain_minimax(&s, d);
                assert_eq!(ab, mm, "mismatch on board {b:?} depth={d}");
            }
        }
    }
}
