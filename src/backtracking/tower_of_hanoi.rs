//! Tower of Hanoi: classic recursive puzzle. Move `n` disks from peg `A`
//! to peg `C` using peg `B` as auxiliary, never placing a larger disk on a
//! smaller one.
//!
//! Complexity: the optimal solution performs exactly `2^n - 1` moves.
//! `hanoi_moves` runs in `O(2^n)` time and space (one entry per move),
//! `hanoi_count` runs in `O(1)`.
//!
//! `hanoi_moves` caps `n` at 60 to avoid allocating an exponentially large
//! vector (`2^60` entries would exhaust memory).

/// Maximum `n` accepted by [`hanoi_moves`]. `2^60 - 1` moves is already far
/// beyond what fits in memory; this guard prevents accidental misuse.
pub const HANOI_MAX_N: u32 = 60;

/// Returns the sequence of moves that solves Tower of Hanoi for `n` disks,
/// moving them from peg `'A'` to peg `'C'` using `'B'` as auxiliary.
///
/// Each move is `(from, to)`. Returns an empty vector for `n == 0`.
///
/// # Panics
/// Panics if `n > HANOI_MAX_N` (60), since the move list would be too large.
pub fn hanoi_moves(n: u32) -> Vec<(char, char)> {
    assert!(
        n <= HANOI_MAX_N,
        "hanoi_moves: n must be <= {HANOI_MAX_N} (got {n})"
    );
    let mut moves = Vec::with_capacity(if n == 0 {
        0
    } else {
        // 2^n - 1 fits comfortably in usize for n <= 60 on 64-bit targets.
        ((1u64 << n) - 1) as usize
    });
    solve(n, 'A', 'C', 'B', &mut moves);
    moves
}

/// Returns the optimal number of moves needed to solve Tower of Hanoi for
/// `n` disks, equal to `2^n - 1`. Returns 0 for `n == 0`.
///
/// # Panics
/// Panics if `n >= 64`, since `2^n - 1` would overflow `u64`.
pub fn hanoi_count(n: u32) -> u64 {
    assert!(
        n < 64,
        "hanoi_count: n must be < 64 to fit in u64 (got {n})"
    );
    if n == 0 {
        0
    } else {
        (1u64 << n) - 1
    }
}

fn solve(n: u32, from: char, to: char, via: char, moves: &mut Vec<(char, char)>) {
    if n == 0 {
        return;
    }
    solve(n - 1, from, via, to, moves);
    moves.push((from, to));
    solve(n - 1, via, to, from, moves);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn n_zero_yields_no_moves_and_zero_count() {
        assert_eq!(hanoi_moves(0), Vec::<(char, char)>::new());
        assert_eq!(hanoi_count(0), 0);
    }

    #[test]
    fn n_one_moves_a_to_c() {
        assert_eq!(hanoi_moves(1), vec![('A', 'C')]);
        assert_eq!(hanoi_count(1), 1);
    }

    #[test]
    fn n_two_canonical_sequence() {
        let expected = vec![('A', 'B'), ('A', 'C'), ('B', 'C')];
        assert_eq!(hanoi_moves(2), expected);
        assert_eq!(hanoi_count(2), 3);
    }

    #[test]
    fn n_three_canonical_sequence() {
        let expected = vec![
            ('A', 'C'),
            ('A', 'B'),
            ('C', 'B'),
            ('A', 'C'),
            ('B', 'A'),
            ('B', 'C'),
            ('A', 'C'),
        ];
        assert_eq!(hanoi_moves(3), expected);
        assert_eq!(hanoi_count(3), 7);
    }

    #[test]
    fn move_count_matches_formula_up_to_ten() {
        for n in 0..=10 {
            assert_eq!(hanoi_moves(n).len() as u64, hanoi_count(n));
            let expected = if n == 0 { 0 } else { (1u64 << n) - 1 };
            assert_eq!(hanoi_count(n), expected);
        }
    }

    #[test]
    fn replay_lands_all_disks_on_c() {
        for n in 0..=10 {
            let mut pegs: [Vec<u32>; 3] = [Vec::new(), Vec::new(), Vec::new()];
            // Initial state: disks n..1 stacked on A (largest at bottom).
            for d in (1..=n).rev() {
                pegs[0].push(d);
            }
            for (from, to) in hanoi_moves(n) {
                let fi = peg_index(from);
                let ti = peg_index(to);
                let disk = pegs[fi].pop().expect("source peg must be non-empty");
                if let Some(&top) = pegs[ti].last() {
                    assert!(disk < top, "larger disk placed on smaller one");
                }
                pegs[ti].push(disk);
            }
            assert!(pegs[0].is_empty());
            assert!(pegs[1].is_empty());
            assert_eq!(pegs[2].len() as u32, n);
            // Disks on C must be sorted largest-to-smallest from bottom to top.
            for (i, &disk) in pegs[2].iter().enumerate() {
                assert_eq!(disk, n - i as u32);
            }
        }
    }

    fn peg_index(c: char) -> usize {
        match c {
            'A' => 0,
            'B' => 1,
            'C' => 2,
            _ => panic!("invalid peg label: {c}"),
        }
    }

    #[test]
    #[should_panic(expected = "n must be <=")]
    fn hanoi_moves_panics_above_cap() {
        let _ = hanoi_moves(HANOI_MAX_N + 1);
    }
}
