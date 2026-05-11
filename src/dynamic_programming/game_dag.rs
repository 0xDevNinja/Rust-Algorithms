//! Winning-strategy DP for two-player games on a DAG.
//!
//! Models a finite, acyclic, perfect-information, no-draws game whose state
//! space is encoded as a directed acyclic graph (DAG).  Each vertex is a
//! position; each outgoing edge is a legal move for the player whose turn it
//! is.  Players alternate; the player who cannot move (a position with no
//! out-edges) **loses**.
//!
//! ## Recurrence
//!
//! For a vertex `v`:
//!
//! * If `v` has no out-edges → the player to move loses → [`Outcome::Lost`].
//! * Else `v` is [`Outcome::Won`] iff some successor is [`Outcome::Lost`]
//!   (the mover picks that edge, handing a losing position to the opponent);
//!   otherwise `v` is [`Outcome::Lost`].
//!
//! Because the game graph is a DAG, classifications can be propagated in
//! reverse topological order — every vertex is decided once all its
//! successors are.  This implementation runs Kahn's algorithm on the
//! **reverse** graph: it pops "ready" vertices (those whose successors are
//! all decided) from a queue, classifies them, and decrements the unresolved
//! out-degree of each predecessor.
//!
//! ## Complexity
//!
//! `O(V + E)` time and `O(V + E)` space (the reverse-edge list dominates).
//!
//! ## Precondition
//!
//! The input adjacency list **must** describe a DAG.  If a directed cycle is
//! present, some vertices can never be classified; [`classify_positions`]
//! panics in that case rather than returning a half-filled answer.

use std::collections::VecDeque;

/// Game-theoretic value of a position from the perspective of the player
/// whose turn it is to move from that position.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    /// The player to move has a winning strategy.
    Won,
    /// The player to move loses under optimal play (includes terminal
    /// positions, which have no legal moves).
    Lost,
}

/// Classifies every position of a DAG game as [`Outcome::Won`] or
/// [`Outcome::Lost`] for the player to move.
///
/// `adj[v]` lists the vertices reachable from `v` in one move.  The returned
/// vector has the same length as `adj` and is indexed identically.
///
/// Runs in `O(V + E)`.
///
/// # Panics
///
/// Panics if `adj` contains a directed cycle (the input must be a DAG) or if
/// any successor index is out of bounds.
pub fn classify_positions(adj: &[Vec<usize>]) -> Vec<Outcome> {
    let n = adj.len();
    let mut result = vec![Outcome::Lost; n];
    let mut decided = vec![false; n];
    // Number of successors of `v` not yet classified as `Lost`.  When this
    // hits zero on an undecided vertex, every successor must be `Won`, so `v`
    // is `Lost`.
    let mut remaining_succ = vec![0usize; n];
    // Reverse adjacency: predecessors of each vertex.
    let mut preds: Vec<Vec<usize>> = vec![Vec::new(); n];

    for (v, succs) in adj.iter().enumerate() {
        remaining_succ[v] = succs.len();
        for &u in succs {
            assert!(u < n, "successor index out of bounds");
            preds[u].push(v);
        }
    }

    let mut queue: VecDeque<usize> = VecDeque::new();
    // Seed: terminal vertices (no out-edges) are immediately Lost.
    for v in 0..n {
        if remaining_succ[v] == 0 {
            decided[v] = true;
            result[v] = Outcome::Lost;
            queue.push_back(v);
        }
    }

    let mut processed = 0usize;
    while let Some(v) = queue.pop_front() {
        processed += 1;
        match result[v] {
            Outcome::Lost => {
                // Any predecessor can move to `v` (a losing position for the
                // opponent), so the predecessor is Won.
                for &p in &preds[v] {
                    if !decided[p] {
                        decided[p] = true;
                        result[p] = Outcome::Won;
                        queue.push_back(p);
                    }
                }
            }
            Outcome::Won => {
                // `v` being Won merely shrinks the count of unresolved
                // successors at each predecessor.  If a predecessor exhausts
                // its successors without ever seeing a Lost one, it is Lost.
                for &p in &preds[v] {
                    if decided[p] {
                        continue;
                    }
                    remaining_succ[p] -= 1;
                    if remaining_succ[p] == 0 {
                        decided[p] = true;
                        result[p] = Outcome::Lost;
                        queue.push_back(p);
                    }
                }
            }
        }
    }

    assert!(
        processed == n,
        "classify_positions: input graph contains a cycle"
    );
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_sink_is_lost() {
        let adj: Vec<Vec<usize>> = vec![vec![]];
        assert_eq!(classify_positions(&adj), vec![Outcome::Lost]);
    }

    #[test]
    fn two_node_chain() {
        // 0 -> 1, 1 is sink.
        let adj = vec![vec![1], vec![]];
        assert_eq!(classify_positions(&adj), vec![Outcome::Won, Outcome::Lost]);
    }

    #[test]
    fn three_node_chain() {
        // 0 -> 1 -> 2, 2 is sink.
        let adj = vec![vec![1], vec![2], vec![]];
        assert_eq!(
            classify_positions(&adj),
            vec![Outcome::Lost, Outcome::Won, Outcome::Lost]
        );
    }

    #[test]
    fn branching_dag() {
        //          0
        //         / \
        //        1   2
        //       / \   \
        //      3   4   4
        // Sinks: 3, 4.  3,4 = Lost.  1 has Lost successor -> Won.
        // 2 has only successor 4 (Lost) -> Won.  0 has successors 1,2 both
        // Won -> Lost.
        let adj = vec![vec![1, 2], vec![3, 4], vec![4], vec![], vec![]];
        let out = classify_positions(&adj);
        assert_eq!(
            out,
            vec![
                Outcome::Lost, // 0
                Outcome::Won,  // 1
                Outcome::Won,  // 2
                Outcome::Lost, // 3
                Outcome::Lost, // 4
            ]
        );
    }

    #[test]
    fn diamond_with_extra_won_branch() {
        // 0 -> 1 -> 3 (sink, Lost)
        // 0 -> 2 -> 3
        // 2 -> 4 -> 5 -> 6 (sink). 6 Lost, 5 Won, 4 Lost.
        // 1: succ {3=Lost} -> Won.
        // 2: succ {3=Lost, 4=Lost} -> Won.
        // 0: succ {1=Won, 2=Won} -> Lost.
        let adj = vec![
            vec![1, 2], // 0
            vec![3],    // 1
            vec![3, 4], // 2
            vec![],     // 3
            vec![5],    // 4
            vec![6],    // 5
            vec![],     // 6
        ];
        let out = classify_positions(&adj);
        assert_eq!(
            out,
            vec![
                Outcome::Lost, // 0
                Outcome::Won,  // 1
                Outcome::Won,  // 2
                Outcome::Lost, // 3
                Outcome::Lost, // 4
                Outcome::Won,  // 5
                Outcome::Lost, // 6
            ]
        );
    }

    #[test]
    fn empty_graph() {
        let adj: Vec<Vec<usize>> = vec![];
        assert!(classify_positions(&adj).is_empty());
    }

    #[test]
    #[should_panic(expected = "cycle")]
    fn cycle_panics() {
        // 0 -> 1 -> 2 -> 0  (no sinks reachable; not a DAG).
        let adj = vec![vec![1], vec![2], vec![0]];
        let _ = classify_positions(&adj);
    }

    #[test]
    #[should_panic(expected = "cycle")]
    fn self_loop_panics() {
        let adj = vec![vec![0]];
        let _ = classify_positions(&adj);
    }
}
