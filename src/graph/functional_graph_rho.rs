//! Cycle detection in a **functional graph** via **Floyd's tortoise & hare**.
//!
//! A functional graph assigns every node `i` exactly one outgoing edge
//! `next[i]`. Following `next` from any starting node therefore traces a
//! sequence that must eventually revisit a node — at which point it has
//! entered the unique cycle reachable from `start`. The trajectory has the
//! shape of the Greek letter ρ (rho): a tail of length `μ` leading into a
//! cycle of length `λ`.
//!
//! # Algorithm
//! Two pointers walk `next` from `start`. The slow pointer advances one step
//! per iteration; the fast pointer advances two. They meet inside the cycle.
//! From the meeting point, resetting the slow pointer to `start` and
//! advancing both at one step per iteration locates the cycle entrance — the
//! number of steps to that meeting equals the tail length `μ`. Continuing
//! one more lap counts the cycle length `λ`.
//!
//! # Complexity
//! - Time:  O(μ + λ) — linear in the rho structure visited from `start`.
//! - Space: O(1) — three `usize` pointers, no auxiliary arrays.
//!
//! # Preconditions
//! - `start < next.len()`. Out-of-range starts panic.
//! - Every entry of `next` must satisfy `next[i] < next.len()`. Violating
//!   this is **undefined behaviour**: the functions read `next[i]` directly
//!   and will panic on out-of-bounds indexing or loop indefinitely if the
//!   indices form a non-functional graph.

/// Returns `(tail_length, cycle_length)` for the rho-shaped trajectory of
/// `next` starting at `start`.
///
/// `tail_length` (`μ`) is the number of steps from `start` to the first node
/// that lies on the cycle; `cycle_length` (`λ`) is the period of the cycle.
/// If `start` itself sits on the cycle the tail length is `0`. A self-loop
/// (`next[start] == start`) yields `(0, 1)`.
///
/// # Panics
/// Panics if `start >= next.len()` or if `next` is empty.
#[must_use]
pub fn rho_structure(next: &[usize], start: usize) -> (usize, usize) {
    assert!(
        start < next.len(),
        "start index {start} out of bounds for n = {}",
        next.len()
    );

    // Phase 1: find a meeting point inside the cycle. Slow advances one step
    // per iteration, fast advances two. They are guaranteed to meet because
    // a functional graph from any start enters a cycle in finite time.
    let mut slow = next[start];
    let mut fast = next[next[start]];
    while slow != fast {
        slow = next[slow];
        fast = next[next[fast]];
    }

    // Phase 2: locate the cycle entrance. Reset slow to start and advance
    // both pointers one step at a time. The number of steps until they meet
    // equals the tail length μ; the meeting node is the cycle entrance.
    let mut tail = 0usize;
    slow = start;
    while slow != fast {
        slow = next[slow];
        fast = next[fast];
        tail += 1;
    }

    // Phase 3: count the cycle length. Walk fast around once until it comes
    // back to the entrance.
    let mut cycle = 1usize;
    let mut cur = next[fast];
    while cur != fast {
        cur = next[cur];
        cycle += 1;
    }

    (tail, cycle)
}

/// Returns the nodes that form the cycle reachable from `start`, listed in
/// traversal order beginning at the cycle entrance.
///
/// The returned vector has length equal to the second component of
/// [`rho_structure`].
///
/// # Panics
/// Panics if `start >= next.len()` or if `next` is empty.
#[must_use]
pub fn cycle_nodes(next: &[usize], start: usize) -> Vec<usize> {
    assert!(
        start < next.len(),
        "start index {start} out of bounds for n = {}",
        next.len()
    );

    // Re-run the meeting + entrance-locating phases to find one node on the
    // cycle, then walk the cycle once to collect every member.
    let mut slow = next[start];
    let mut fast = next[next[start]];
    while slow != fast {
        slow = next[slow];
        fast = next[next[fast]];
    }
    slow = start;
    while slow != fast {
        slow = next[slow];
        fast = next[fast];
    }

    let entrance = slow;
    let mut out = vec![entrance];
    let mut cur = next[entrance];
    while cur != entrance {
        out.push(cur);
        cur = next[cur];
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{cycle_nodes, rho_structure};
    use quickcheck_macros::quickcheck;
    use std::collections::HashMap;

    // Brute-force reference: walk `next` from `start`, recording the first
    // step at which each node is seen. The first repeat reveals the cycle
    // entrance; the difference between visit indices gives the cycle length.
    fn brute_force(next: &[usize], start: usize) -> (usize, usize) {
        let mut seen: HashMap<usize, usize> = HashMap::new();
        let mut cur = start;
        let mut step = 0usize;
        loop {
            if let Some(&first) = seen.get(&cur) {
                return (first, step - first);
            }
            seen.insert(cur, step);
            cur = next[cur];
            step += 1;
        }
    }

    #[test]
    fn single_fixed_point() {
        // 0 -> 0 — cycle of length 1, no tail.
        let next = [0];
        assert_eq!(rho_structure(&next, 0), (0, 1));
        assert_eq!(cycle_nodes(&next, 0), vec![0]);
    }

    #[test]
    fn two_node_cycle() {
        // 0 -> 1 -> 0 — cycle of length 2, no tail from either node.
        let next = [1, 0];
        assert_eq!(rho_structure(&next, 0), (0, 2));
        assert_eq!(rho_structure(&next, 1), (0, 2));
        let cyc = cycle_nodes(&next, 0);
        assert_eq!(cyc.len(), 2);
        assert!(cyc.contains(&0) && cyc.contains(&1));
    }

    #[test]
    fn pure_rho_shape() {
        // 0 -> 1 -> 2 -> 3 -> 2 — tail 0->1->2 (length 2), cycle 2->3->2
        // (length 2).
        let next = [1, 2, 3, 2];
        assert_eq!(rho_structure(&next, 0), (2, 2));
        let cyc = cycle_nodes(&next, 0);
        assert_eq!(cyc.len(), 2);
        assert!(cyc.contains(&2) && cyc.contains(&3));
    }

    #[test]
    fn start_already_on_cycle() {
        // 0 -> 1 -> 2 -> 0 — every node sits on the cycle.
        let next = [1, 2, 0];
        assert_eq!(rho_structure(&next, 1), (0, 3));
        assert_eq!(rho_structure(&next, 2), (0, 3));
        assert_eq!(rho_structure(&next, 0), (0, 3));
    }

    #[test]
    fn long_tail_short_cycle() {
        // Tail: 0 -> 1 -> 2 -> 3 -> 4 -> 5 (six tail nodes).
        // Cycle: 5 -> 6 -> 5 (length 2). Entrance is node 5, tail length 5.
        let next = [1, 2, 3, 4, 5, 6, 5];
        assert_eq!(rho_structure(&next, 0), (5, 2));
        let cyc = cycle_nodes(&next, 0);
        assert_eq!(cyc.len(), 2);
        assert!(cyc.contains(&5) && cyc.contains(&6));
    }

    #[test]
    fn short_tail_long_cycle() {
        // 0 -> 1 -> 2 -> 3 -> 4 -> 1: tail length 1, cycle length 4.
        let next = [1, 2, 3, 4, 1];
        assert_eq!(rho_structure(&next, 0), (1, 4));
        let cyc = cycle_nodes(&next, 0);
        assert_eq!(cyc.len(), 4);
    }

    #[test]
    fn cycle_nodes_lists_each_member_once() {
        // Walking the returned cycle by repeatedly applying `next` must
        // visit every entry exactly once and return to the start.
        let next = [1, 2, 3, 4, 5, 2];
        let cyc = cycle_nodes(&next, 0);
        let (_, lambda) = rho_structure(&next, 0);
        assert_eq!(cyc.len(), lambda);
        let mut cur = cyc[0];
        for &expected in cyc.iter().skip(1) {
            cur = next[cur];
            assert_eq!(cur, expected);
        }
        // One more step must close the loop.
        assert_eq!(next[cur], cyc[0]);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn out_of_range_start_panics() {
        let next = [0, 1];
        let _ = rho_structure(&next, 5);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn cycle_nodes_out_of_range_start_panics() {
        let next = [0, 1];
        let _ = cycle_nodes(&next, 5);
    }

    // Property test: random functional graphs on n in 1..=32, random start.
    // Compare the two-pointer answer to a brute-force HashMap reference.
    // Build a random functional graph of size `n` (>= 1) from a byte seed.
    // If the seed is empty, every entry is 0 (the all-points-fixed graph at
    // node 0); otherwise entry i is `seed[i % seed.len()] % n`.
    fn make_next(seed: &[u8], n: usize) -> Vec<usize> {
        (0..n)
            .map(|i| {
                if seed.is_empty() {
                    0
                } else {
                    (seed[i % seed.len()] as usize) % n
                }
            })
            .collect()
    }

    #[allow(clippy::needless_pass_by_value)]
    #[quickcheck]
    fn quickcheck_matches_brute_force(raw: Vec<u8>, size_seed: u8, start_seed: u8) -> bool {
        // Independent size in 1..=32 so an empty `raw` still produces a graph.
        let n = ((size_seed as usize) % 32) + 1;
        let next = make_next(&raw, n);
        let start = (start_seed as usize) % n;
        rho_structure(&next, start) == brute_force(&next, start)
    }

    // Property test: cycle_nodes always returns a cycle of the length
    // reported by rho_structure, and applying `next` cycles it.
    #[allow(clippy::needless_pass_by_value)]
    #[quickcheck]
    fn quickcheck_cycle_nodes_consistent(raw: Vec<u8>, size_seed: u8, start_seed: u8) -> bool {
        let n = ((size_seed as usize) % 32) + 1;
        let next = make_next(&raw, n);
        let start = (start_seed as usize) % n;
        let (_, lambda) = rho_structure(&next, start);
        let cyc = cycle_nodes(&next, start);
        if cyc.len() != lambda {
            return false;
        }
        // Walking from cyc[i] via `next` lands on cyc[(i + 1) % lambda].
        for i in 0..lambda {
            if next[cyc[i]] != cyc[(i + 1) % lambda] {
                return false;
            }
        }
        true
    }
}
