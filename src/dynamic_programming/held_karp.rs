//! Held-Karp exact solver for the Travelling Salesman Problem on a complete
//! directed graph. Bitmask DP: `dp[mask][j]` is the cost of the cheapest path
//! that starts at city 0, visits exactly the cities in `mask` (which must
//! contain `j`), and ends at `j`. O(n² · 2ⁿ) time, O(n · 2ⁿ) space.
//!
//! `n` is capped at 20 so masks fit comfortably in a `u32`. Distances are
//! `u64`; `u64::MAX` is treated as +∞ and additions saturate.

/// Sentinel meaning "no edge" / "unreachable" for the integer variant.
const INF: u64 = u64::MAX;

/// Solves TSP exactly on the complete graph described by `dist` and returns
/// `(cost, tour)` where `tour` is a Hamiltonian cycle of length `n + 1` that
/// starts and ends at city `0`. Edges with weight `u64::MAX` are treated as
/// missing, and addition saturates so they never overflow into a finite cost.
///
/// # Panics
/// Panics if `dist` is not a square matrix, if `n == 0`, or if `n > 20`.
pub fn held_karp(dist: &[Vec<u64>]) -> (u64, Vec<usize>) {
    let n = dist.len();
    assert!(n > 0, "dist must be non-empty");
    assert!(n <= 20, "Held-Karp is capped at n <= 20");
    for row in dist {
        assert_eq!(row.len(), n, "dist must be a square matrix");
    }

    if n == 1 {
        return (0, vec![0, 0]);
    }

    let size = 1usize << n;
    // dp[mask * n + j] = cheapest cost from 0 visiting exactly `mask` ending at j.
    let mut dp = vec![INF; size * n];
    let mut parent = vec![usize::MAX; size * n];

    // Base: just the starting city.
    let start_mask = 1usize; // bit 0 set
    dp[start_mask * n] = 0; // dp[{0}][0] = 0

    for mask in 1..size {
        if mask & 1 == 0 {
            // Every reachable subset must include city 0.
            continue;
        }
        for j in 0..n {
            if mask & (1 << j) == 0 {
                continue;
            }
            let cur = dp[mask * n + j];
            if cur == INF {
                continue;
            }
            // Try to extend the path to a new city k.
            for k in 1..n {
                if mask & (1 << k) != 0 {
                    continue;
                }
                let edge = dist[j][k];
                if edge == INF {
                    continue;
                }
                let next_mask = mask | (1 << k);
                let candidate = cur.saturating_add(edge);
                let slot = next_mask * n + k;
                if candidate < dp[slot] {
                    dp[slot] = candidate;
                    parent[slot] = j;
                }
            }
        }
    }

    let full = size - 1;
    let mut best_cost = INF;
    let mut best_end = usize::MAX;
    for j in 1..n {
        let path = dp[full * n + j];
        if path == INF {
            continue;
        }
        let back = dist[j][0];
        if back == INF {
            continue;
        }
        let total = path.saturating_add(back);
        if total < best_cost {
            best_cost = total;
            best_end = j;
        }
    }

    assert!(
        best_cost != INF,
        "no Hamiltonian cycle exists in the given graph"
    );

    // Reconstruct tour by walking the parent pointers.
    let mut tour = Vec::with_capacity(n + 1);
    let mut mask = full;
    let mut cur = best_end;
    while cur != 0 {
        tour.push(cur);
        let prev = parent[mask * n + cur];
        mask ^= 1 << cur;
        cur = prev;
    }
    tour.push(0);
    tour.reverse();
    tour.push(0);

    (best_cost, tour)
}

/// `f64` variant of [`held_karp`]. Uses `f64::INFINITY` as the missing-edge
/// sentinel. Note that floating-point addition is not associative, so for
/// pathological inputs the reported optimum may differ from the integer
/// version by a few ULPs; for typical inputs the result is exact.
///
/// # Panics
/// Panics if `dist` is not a square matrix, if `n == 0`, or if `n > 20`.
pub fn held_karp_f64(dist: &[Vec<f64>]) -> (f64, Vec<usize>) {
    let n = dist.len();
    assert!(n > 0, "dist must be non-empty");
    assert!(n <= 20, "Held-Karp is capped at n <= 20");
    for row in dist {
        assert_eq!(row.len(), n, "dist must be a square matrix");
    }

    if n == 1 {
        return (0.0, vec![0, 0]);
    }

    let size = 1usize << n;
    let mut dp = vec![f64::INFINITY; size * n];
    let mut parent = vec![usize::MAX; size * n];

    dp[n] = 0.0; // mask = {0}, end = 0

    for mask in 1..size {
        if mask & 1 == 0 {
            continue;
        }
        for j in 0..n {
            if mask & (1 << j) == 0 {
                continue;
            }
            let cur = dp[mask * n + j];
            if !cur.is_finite() {
                continue;
            }
            for k in 1..n {
                if mask & (1 << k) != 0 {
                    continue;
                }
                let edge = dist[j][k];
                if !edge.is_finite() {
                    continue;
                }
                let next_mask = mask | (1 << k);
                let candidate = cur + edge;
                let slot = next_mask * n + k;
                if candidate < dp[slot] {
                    dp[slot] = candidate;
                    parent[slot] = j;
                }
            }
        }
    }

    let full = size - 1;
    let mut best_cost = f64::INFINITY;
    let mut best_end = usize::MAX;
    for j in 1..n {
        let path = dp[full * n + j];
        let back = dist[j][0];
        if !path.is_finite() || !back.is_finite() {
            continue;
        }
        let total = path + back;
        if total < best_cost {
            best_cost = total;
            best_end = j;
        }
    }

    assert!(
        best_cost.is_finite(),
        "no Hamiltonian cycle exists in the given graph"
    );

    let mut tour = Vec::with_capacity(n + 1);
    let mut mask = full;
    let mut cur = best_end;
    while cur != 0 {
        tour.push(cur);
        let prev = parent[mask * n + cur];
        mask ^= 1 << cur;
        cur = prev;
    }
    tour.push(0);
    tour.reverse();
    tour.push(0);

    (best_cost, tour)
}

#[cfg(test)]
mod tests {
    use super::{held_karp, held_karp_f64};

    fn tour_cost(dist: &[Vec<u64>], tour: &[usize]) -> u64 {
        tour.windows(2)
            .map(|w| dist[w[0]][w[1]])
            .fold(0u64, u64::saturating_add)
    }

    fn brute_force(dist: &[Vec<u64>]) -> u64 {
        let n = dist.len();
        if n == 1 {
            return 0;
        }
        let rest: Vec<usize> = (1..n).collect();
        let mut best = u64::MAX;
        permute(&rest, 0, &mut rest.clone(), &mut |perm| {
            let mut cost = 0u64;
            let mut prev = 0usize;
            let mut ok = true;
            for &c in perm {
                let e = dist[prev][c];
                if e == u64::MAX {
                    ok = false;
                    break;
                }
                cost = cost.saturating_add(e);
                prev = c;
            }
            if ok {
                let back = dist[prev][0];
                if back != u64::MAX {
                    cost = cost.saturating_add(back);
                    if cost < best {
                        best = cost;
                    }
                }
            }
        });
        best
    }

    fn permute<F: FnMut(&[usize])>(
        original: &[usize],
        depth: usize,
        scratch: &mut Vec<usize>,
        visit: &mut F,
    ) {
        let n = original.len();
        if depth == n {
            visit(scratch);
            return;
        }
        for i in depth..n {
            scratch.swap(depth, i);
            permute(original, depth + 1, scratch, visit);
            scratch.swap(depth, i);
        }
    }

    #[test]
    fn single_city() {
        let dist = vec![vec![0]];
        assert_eq!(held_karp(&dist), (0, vec![0, 0]));
    }

    #[test]
    fn two_cities() {
        let dist = vec![vec![0, 7], vec![3, 0]];
        let (cost, tour) = held_karp(&dist);
        assert_eq!(cost, 10);
        assert_eq!(tour, vec![0, 1, 0]);
    }

    #[test]
    fn unit_square() {
        // 4 corners of a unit square (city 0..3 going around). Any Hamiltonian
        // cycle on the square has cost 4 because we must traverse 4 unit edges
        // (the diagonals cost 2 but you'd still have to come back via either
        // a unit or a diagonal). Actually min cost = 4 visiting in order.
        let d = vec![
            vec![0, 1, 2, 1],
            vec![1, 0, 1, 2],
            vec![2, 1, 0, 1],
            vec![1, 2, 1, 0],
        ];
        let (cost, tour) = held_karp(&d);
        assert_eq!(cost, 4);
        assert_eq!(tour.len(), 5);
        assert_eq!(tour[0], 0);
        assert_eq!(*tour.last().unwrap(), 0);
        assert_eq!(tour_cost(&d, &tour), 4);
    }

    #[test]
    fn five_city_asymmetric() {
        // Hand-picked asymmetric instance. Optimal tour 0 -> 1 -> 2 -> 3 -> 4 -> 0
        // with cost 1 + 2 + 3 + 4 + 5 = 15. Other orders are strictly worse
        // because of the inflated reverse edges.
        let d = vec![
            vec![0, 1, 50, 50, 50],
            vec![50, 0, 2, 50, 50],
            vec![50, 50, 0, 3, 50],
            vec![50, 50, 50, 0, 4],
            vec![5, 50, 50, 50, 0],
        ];
        let (cost, tour) = held_karp(&d);
        assert_eq!(cost, 15);
        assert_eq!(tour, vec![0, 1, 2, 3, 4, 0]);
    }

    #[test]
    fn matches_brute_force_random() {
        // Tiny LCG so we don't pull a dep just for tests.
        let mut state: u64 = 0x00C0_FFEE_1234_5678;
        let mut next = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            state
        };
        for n in 1..=6usize {
            for _ in 0..6 {
                let mut d = vec![vec![0u64; n]; n];
                for i in 0..n {
                    for j in 0..n {
                        if i != j {
                            d[i][j] = (next() % 50) + 1;
                        }
                    }
                }
                let (cost, tour) = held_karp(&d);
                assert_eq!(tour.len(), n + 1);
                assert_eq!(tour[0], 0);
                assert_eq!(*tour.last().unwrap(), 0);
                assert_eq!(tour_cost(&d, &tour), cost);
                let mut visited = vec![false; n];
                for &c in &tour[..n] {
                    assert!(!visited[c], "city {c} visited twice");
                    visited[c] = true;
                }
                assert!(visited.iter().all(|&v| v));
                assert_eq!(cost, brute_force(&d));
            }
        }
    }

    #[test]
    fn f64_variant_matches_integer() {
        let d_int = vec![
            vec![0, 10, 15, 20],
            vec![10, 0, 35, 25],
            vec![15, 35, 0, 30],
            vec![20, 25, 30, 0],
        ];
        let d_f64: Vec<Vec<f64>> = d_int
            .iter()
            .map(|row| row.iter().map(|&x| x as f64).collect())
            .collect();
        let (ci, _) = held_karp(&d_int);
        let (cf, _) = held_karp_f64(&d_f64);
        assert!((cf - ci as f64).abs() < 1e-9);
    }

    #[test]
    #[should_panic(expected = "Held-Karp is capped at n <= 20")]
    fn rejects_large_n() {
        let d = vec![vec![0u64; 21]; 21];
        let _ = held_karp(&d);
    }

    #[test]
    #[should_panic(expected = "dist must be a square matrix")]
    fn rejects_non_square() {
        let d = vec![vec![0, 1, 2], vec![1, 0, 3]];
        let _ = held_karp(&d);
    }
}
